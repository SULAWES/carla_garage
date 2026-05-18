#!/usr/bin/env python3
"""Minimal CARLA-native DiffusionDrive training entrypoint."""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GlobalConfig
from diffusiondrive.carla_native_dataset import Bench2DriveDiffusionDataset
from diffusiondrive.config_adapter import DiffusionDriveRuntimeOverrides, build_diffusiondrive_config
from diffusiondrive.model import V2TransfuserModel


CHECKPOINT_WRAPPER_PREFIXES = ("agent", "model", "module", "_transfuser_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--id", default="diffusiondrive_carla_native")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--anchor-path",
        default=os.environ.get(
            "DIFFUSIONDRIVE_ANCHOR_PATH",
            "/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy",
        ),
    )
    parser.add_argument(
        "--backbone-path",
        default=os.environ.get(
            "DIFFUSIONDRIVE_BACKBONE_PATH",
            "/home/HeavenlySU/sitp_workspace/pytorch_model.bin",
        ),
    )
    parser.add_argument("--load-file", default=os.environ.get("DIFFUSIONDRIVE_CHECKPOINT", ""))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_features_to_device(batch: dict, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features = {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in batch["features"].items()
    }
    targets = {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in batch["targets"].items()
    }
    return features, targets


def extract_checkpoint_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema_state_dict", "network", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint, "root"
    raise RuntimeError("Unable to locate a tensor state_dict inside the provided checkpoint.")


def normalize_checkpoint_key(key: str) -> str:
    parts = key.split(".")
    while parts and parts[0] in CHECKPOINT_WRAPPER_PREFIXES:
        parts = parts[1:]
    return ".".join(parts)


def load_checkpoint_partial(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, source_name = extract_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    filtered = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state_dict.items():
        normalized_key = normalize_checkpoint_key(key)
        if normalized_key not in model_state:
            unexpected.append(normalized_key)
            continue
        if tuple(value.shape) != tuple(model_state[normalized_key].shape):
            shape_mismatch.append(
                f"{normalized_key}: ckpt{tuple(value.shape)} != model{tuple(model_state[normalized_key].shape)}"
            )
            continue
        filtered[normalized_key] = value
    model.load_state_dict(filtered, strict=False)
    print(
        "Loaded DiffusionDrive checkpoint "
        f"'{source_name}' from {checkpoint_path}: matched {len(filtered)}/{len(model_state)} tensors.",
        flush=True,
    )
    if shape_mismatch:
        print(f"Shape mismatches skipped ({len(shape_mismatch)}): {shape_mismatch[:10]}", flush=True)
    if unexpected:
        print(f"Unexpected tensors skipped ({len(unexpected)}): {sorted(unexpected)[:10]}", flush=True)


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch{epoch:03d}_step{step:07d}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
        },
        path,
    )
    print(f"Saved checkpoint: {path}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    seed_everything(args.seed)

    device = torch.device(args.device)
    output_dir = args.logdir / args.id
    output_dir.mkdir(parents=True, exist_ok=True)

    global_config = GlobalConfig()
    dd_config = build_diffusiondrive_config(
        global_config,
        DiffusionDriveRuntimeOverrides(anchor_path=args.anchor_path, backbone_path=args.backbone_path),
    )
    dataset = Bench2DriveDiffusionDataset(
        args.root_dir,
        config=global_config,
        num_poses=dd_config.trajectory_sampling.num_poses,
        future_stride=args.future_stride,
        frame_sampling=args.frame_sampling,
        max_samples=args.max_samples,
        route_glob=args.route_glob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Dataset samples: {len(dataset)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    model = V2TransfuserModel(dd_config).to(device)
    load_checkpoint_partial(model, args.load_file, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    running_loss = 0.0
    running_count = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            features, targets = move_features_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, targets=targets)
            loss = outputs["trajectory_loss"]
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1
            if global_step == 1 or global_step % args.log_every == 0:
                avg_loss = running_loss / max(running_count, 1)
                running_loss = 0.0
                running_count = 0
                loss_dict = outputs.get("trajectory_loss_dict", {})
                loss_parts = ", ".join(
                    f"{name}={float(value.detach().cpu()):.4f}" for name, value in sorted(loss_dict.items())
                )
                print(
                    f"epoch={epoch} step={global_step} loss={float(loss.detach().cpu()):.4f} "
                    f"avg={avg_loss:.4f} {loss_parts}",
                    flush=True,
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir, model, optimizer, epoch, global_step, args)

            if args.max_steps is not None and global_step >= args.max_steps:
                save_checkpoint(output_dir, model, optimizer, epoch, global_step, args)
                return

    save_checkpoint(output_dir, model, optimizer, args.epochs - 1, global_step, args)


if __name__ == "__main__":
    main()
