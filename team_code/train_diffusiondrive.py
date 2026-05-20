#!/usr/bin/env python3
"""Minimal CARLA-native DiffusionDrive training entrypoint."""

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GlobalConfig
from diffusiondrive.carla_native_dataset import Bench2DriveDiffusionDataset
from diffusiondrive.config_adapter import (
    DiffusionDriveRuntimeOverrides,
    build_diffusiondrive_config,
    validate_diffusiondrive_config,
)
from diffusiondrive.model import V2TransfuserModel
from diffusiondrive.status import STATUS_FEATURE_SCHEMA


CHECKPOINT_WRAPPER_PREFIXES = ("agent", "model", "module", "_transfuser_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-root-dir", type=Path, nargs="+", default=None)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--id", default="diffusiondrive_carla_native")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw",), default="adamw")
    parser.add_argument("--scheduler", choices=("none", "cosine", "multistep"), default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-steps", default="70")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--val-frame-sampling", type=int, default=None)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--dataset-mode", default="b2d_full_raw")
    parser.add_argument("--target-mode", choices=("spatial_path", "future_ego_time"), default="spatial_path")
    parser.add_argument("--spatial-target-first-distance", type=float, default=2.5)
    parser.add_argument("--spatial-target-interval", type=float, default=1.0)
    parser.add_argument("--spatial-target-max-future-frames", type=int, default=120)
    parser.add_argument("--assumed-frame-interval", type=float, default=0.1)
    parser.add_argument("--b2d-source-image-height", type=int, default=900)
    parser.add_argument("--b2d-source-image-width", type=int, default=1600)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--val-route-glob", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--no-jpeg-artifact", action="store_true")
    parser.add_argument("--model-image-height", type=int, default=256)
    parser.add_argument("--model-image-width", type=int, default=1024)
    parser.add_argument("--trajectory-weight", type=float, default=None)
    parser.add_argument("--trajectory-cls-weight", type=float, default=None)
    parser.add_argument("--trajectory-reg-weight", type=float, default=None)
    parser.add_argument("--trajectory-focal-alpha", type=float, default=None)
    parser.add_argument("--trajectory-focal-gamma", type=float, default=None)
    parser.add_argument("--diffusion-num-train-timesteps", type=int, default=None)
    parser.add_argument("--diffusion-train-timestep-min", type=int, default=None)
    parser.add_argument("--diffusion-train-timestep-max", type=int, default=None)
    parser.add_argument("--diffusion-infer-step-num", type=int, default=None)
    parser.add_argument("--diffusion-infer-timestep-span", type=int, default=None)
    parser.add_argument("--diffusion-infer-trunc-timesteps", type=int, default=None)
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
    parser.add_argument("--resume-file", default="")
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


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def apply_runtime_training_overrides(dd_config: Any, args: argparse.Namespace) -> None:
    overrides = {
        "trajectory_weight": args.trajectory_weight,
        "trajectory_cls_weight": args.trajectory_cls_weight,
        "trajectory_reg_weight": args.trajectory_reg_weight,
        "trajectory_focal_alpha": args.trajectory_focal_alpha,
        "trajectory_focal_gamma": args.trajectory_focal_gamma,
        "diffusion_num_train_timesteps": args.diffusion_num_train_timesteps,
        "diffusion_train_timestep_min": args.diffusion_train_timestep_min,
        "diffusion_train_timestep_max": args.diffusion_train_timestep_max,
        "diffusion_infer_step_num": args.diffusion_infer_step_num,
        "diffusion_infer_timestep_span": args.diffusion_infer_timestep_span,
        "diffusion_infer_trunc_timesteps": args.diffusion_infer_trunc_timesteps,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(dd_config, name, value)
    validate_diffusiondrive_config(dd_config)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    return value


def write_run_config(output_dir: Path, args: argparse.Namespace, global_config: GlobalConfig, dd_config: Any) -> None:
    anchor_shape = None
    if dd_config.plan_anchor_path and Path(dd_config.plan_anchor_path).is_file():
        anchor_shape = list(np.load(dd_config.plan_anchor_path, mmap_mode="r").shape)

    run_config = {
        "args": make_json_safe(vars(args)),
        "diffusiondrive": make_json_safe(asdict(dd_config)),
        "data": {
            "dataset_mode": args.dataset_mode,
            "train_root_dir": make_json_safe(args.root_dir),
            "val_root_dir": make_json_safe(args.val_root_dir),
            "route_glob": args.route_glob,
            "val_route_glob": args.val_route_glob or args.route_glob,
            "frame_sampling": args.frame_sampling,
            "val_frame_sampling": args.val_frame_sampling or args.frame_sampling,
            "future_stride": args.future_stride,
            "target_mode": args.target_mode,
            "max_samples": args.max_samples,
            "val_max_samples": args.val_max_samples,
        },
        "target": {
            "mode": args.target_mode,
            "spatial_path": {
                "first_distance_meters": args.spatial_target_first_distance,
                "interval_meters": args.spatial_target_interval,
                "num_poses": dd_config.trajectory_sampling.num_poses,
                "last_distance_meters": (
                    args.spatial_target_first_distance
                    + args.spatial_target_interval * (dd_config.trajectory_sampling.num_poses - 1)
                ),
                "max_future_frames_for_path": args.spatial_target_max_future_frames,
                "max_future_seconds_for_path": args.spatial_target_max_future_frames * args.assumed_frame_interval,
                "source": "future ego path resampled by distance; sample discovery requires at least one future annotation and extrapolates from the last path segment or command direction when needed",
            },
            "future_ego_time_legacy": {
                "future_stride_frames": args.future_stride,
                "target_interval_seconds": args.future_stride * args.assumed_frame_interval,
            },
        },
        "time_semantics": {
            "assumed_frame_interval_seconds": args.assumed_frame_interval,
            "future_stride_frames_if_time_target": args.future_stride,
            "target_interval_seconds_if_time_target": args.future_stride * args.assumed_frame_interval,
            "trajectory_sampling_interval_seconds": dd_config.trajectory_sampling.interval_length,
            "trajectory_sampling_num_poses": dd_config.trajectory_sampling.num_poses,
            "trajectory_sampling_time_horizon_seconds": dd_config.trajectory_sampling.time_horizon,
            "anchor_interval_seconds_assumption": None,
            "anchor_spatial_first_distance_meters": args.spatial_target_first_distance,
            "anchor_spatial_interval_meters": args.spatial_target_interval,
        },
        "anchor": {
            "path": dd_config.plan_anchor_path,
            "shape": anchor_shape,
            "num_modes": dd_config.num_anchor_modes,
            "semantics": "spatial route/checkpoint anchor, not fixed-time trajectory",
        },
        "sensor_contract": {
            "dataset_mode": args.dataset_mode,
            "source": "Bench2Drive Full raw route directories",
            "source_image_size": [args.b2d_source_image_height, args.b2d_source_image_width],
            "source_front_camera": {
                "size": [args.b2d_source_image_height, args.b2d_source_image_width],
                "fov": 70,
                "location": [0.8, 0.0, 1.6],
            },
            "source_lidar": {
                "location": [-0.39, 0.0, 1.84],
                "yaw_degrees": 0,
                "range_meters": 85,
            },
            "online_garage_front_camera": {
                "size": [global_config.camera_height, global_config.camera_width],
                "fov": global_config.camera_fov,
                "location": global_config.camera_pos,
            },
            "online_garage_lidar": {
                "location": global_config.lidar_pos,
                "rotation": global_config.lidar_rot,
                "lidar_resolution": [global_config.lidar_resolution_height, global_config.lidar_resolution_width],
                "pixels_per_meter": global_config.pixels_per_meter,
            },
            "known_gap": "B2D Full raw camera/LiDAR geometry is not identical to the online garage sensor suite.",
        },
        "preprocessing": {
            "source_image_size": [args.b2d_source_image_height, args.b2d_source_image_width],
            "online_sensor_size": [global_config.camera_height, global_config.camera_width],
            "crop_image": global_config.crop_image,
            "cropped_size": [global_config.cropped_height, global_config.cropped_width],
            "model_image_size": [args.model_image_height, args.model_image_width],
            "input_range": "[0,1]",
            "imagenet_normalization": False,
            "jpeg_artifact": not args.no_jpeg_artifact,
        },
        "status_feature": {
            "schema": STATUS_FEATURE_SCHEMA,
            "dim": dd_config.status_dim,
            "normalized": False,
        },
    }
    config_path = output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True)
    print(f"Wrote run config: {config_path}", flush=True)


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer != "adamw":
        raise RuntimeError(f"Unsupported optimizer: {args.optimizer}")
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if args.scheduler == "none" and args.warmup_steps <= 0:
        return None

    lr_steps = parse_int_list(args.lr_steps)
    min_lr_scale = args.min_lr / args.lr if args.lr > 0 else 0.0
    decay_steps = max(total_steps - args.warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        if args.scheduler == "cosine":
            progress = min(max((step - args.warmup_steps) / decay_steps, 0.0), 1.0)
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine_scale
        if args.scheduler == "multistep":
            milestone_count = sum(step >= milestone for milestone in lr_steps)
            return args.lr_gamma ** milestone_count
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def total_planned_steps(dataloader: DataLoader, args: argparse.Namespace) -> int:
    epoch_steps = len(dataloader) * args.epochs
    if args.max_steps is not None:
        return min(epoch_steps, args.max_steps)
    return epoch_steps


def compute_total_loss(outputs: Dict[str, torch.Tensor], dd_config: Any) -> torch.Tensor:
    return dd_config.trajectory_weight * outputs["trajectory_loss"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dd_config: Any,
    max_steps: Optional[int],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_trajectory_loss = 0.0
    count = 0
    for batch in dataloader:
        features, targets = move_features_to_device(batch, device)
        outputs = model(features, targets=targets)
        total = compute_total_loss(outputs, dd_config)
        total_loss += float(total.detach().cpu())
        total_trajectory_loss += float(outputs["trajectory_loss"].detach().cpu())
        count += 1
        if max_steps is not None and count >= max_steps:
            break
    if was_training:
        model.train()
    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "trajectory_loss": total_trajectory_loss / denom,
        "steps": float(count),
    }


def load_training_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
) -> Tuple[int, int]:
    if not checkpoint_path:
        return 0, 0
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("step", 0))
    print(
        f"Resumed training checkpoint from {checkpoint_path}: start_epoch={start_epoch} step={global_step}",
        flush=True,
    )
    return start_epoch, global_step


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    epoch: int,
    step: int,
    args: argparse.Namespace,
    dd_config: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_epoch{epoch:03d}_step{step:07d}.pth"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": make_json_safe(vars(args)),
        "diffusiondrive_config": make_json_safe(asdict(dd_config)),
    }
    torch.save(payload, path)
    torch.save(payload, output_dir / "latest.pth")
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
    dd_config.camera_height = args.model_image_height
    dd_config.camera_width = args.model_image_width
    dd_config.__post_init__()
    apply_runtime_training_overrides(dd_config, args)
    write_run_config(output_dir, args, global_config, dd_config)

    dataset = Bench2DriveDiffusionDataset(
        args.root_dir,
        config=global_config,
        num_poses=dd_config.trajectory_sampling.num_poses,
        future_stride=args.future_stride,
        frame_sampling=args.frame_sampling,
        max_samples=args.max_samples,
        route_glob=args.route_glob,
        model_image_size=(dd_config.camera_height, dd_config.camera_width),
        jpeg_artifact=not args.no_jpeg_artifact,
        target_mode=args.target_mode,
        spatial_target_first_distance=args.spatial_target_first_distance,
        spatial_target_interval=args.spatial_target_interval,
        spatial_target_max_future_frames=args.spatial_target_max_future_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=args.drop_last,
    )
    val_dataloader = None
    if args.val_root_dir:
        val_dataset = Bench2DriveDiffusionDataset(
            args.val_root_dir,
            config=global_config,
            num_poses=dd_config.trajectory_sampling.num_poses,
            future_stride=args.future_stride,
            frame_sampling=args.val_frame_sampling or args.frame_sampling,
            max_samples=args.val_max_samples,
            route_glob=args.val_route_glob or args.route_glob,
            model_image_size=(dd_config.camera_height, dd_config.camera_width),
            jpeg_artifact=not args.no_jpeg_artifact,
            target_mode=args.target_mode,
            spatial_target_first_distance=args.spatial_target_first_distance,
            spatial_target_interval=args.spatial_target_interval,
            spatial_target_max_future_frames=args.spatial_target_max_future_frames,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        print(f"Validation samples: {len(val_dataset)}", flush=True)
    print(f"Dataset samples: {len(dataset)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    model = V2TransfuserModel(dd_config).to(device)
    model.train()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, total_planned_steps(dataloader, args))

    if args.resume_file:
        start_epoch, global_step = load_training_checkpoint(args.resume_file, model, optimizer, scheduler, device)
    else:
        load_checkpoint_partial(model, args.load_file, device)
        start_epoch, global_step = 0, 0

    running_loss = 0.0
    running_count = 0
    for epoch in range(start_epoch, args.epochs):
        for batch in dataloader:
            features, targets = move_features_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, targets=targets)
            loss = compute_total_loss(outputs, dd_config)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={global_step} lr={lr:.6g} loss={float(loss.detach().cpu()):.4f} "
                    f"avg={avg_loss:.4f} trajectory_unweighted={float(outputs['trajectory_loss'].detach().cpu()):.4f} "
                    f"{loss_parts}",
                    flush=True,
                )

            if val_dataloader is not None and args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                metrics = evaluate(model, val_dataloader, device, dd_config, args.max_val_steps)
                print(
                    f"validation step={global_step} loss={metrics['loss']:.4f} "
                    f"trajectory_unweighted={metrics['trajectory_loss']:.4f} steps={int(metrics['steps'])}",
                    flush=True,
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir, model, optimizer, scheduler, epoch, global_step, args, dd_config)

            if args.max_steps is not None and global_step >= args.max_steps:
                save_checkpoint(output_dir, model, optimizer, scheduler, epoch, global_step, args, dd_config)
                return

    if val_dataloader is not None and args.val_every_steps == 0:
        metrics = evaluate(model, val_dataloader, device, dd_config, args.max_val_steps)
        print(
            f"validation final loss={metrics['loss']:.4f} "
            f"trajectory_unweighted={metrics['trajectory_loss']:.4f} steps={int(metrics['steps'])}",
            flush=True,
        )
    save_checkpoint(output_dir, model, optimizer, scheduler, args.epochs - 1, global_step, args, dd_config)


if __name__ == "__main__":
    main()
