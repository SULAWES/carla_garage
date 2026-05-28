#!/usr/bin/env python3
"""Inspect high-error DiffusionDrive trajectory samples.

This tool reports per-sample inference trajectory errors. The score is not the
batch-reduced training loss; it is a deterministic diagnostic metric computed
from the predicted trajectory and the dataset target.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from config import GlobalConfig  # noqa: E402
from diffusiondrive.carla_native_dataset import (  # noqa: E402
    TARGET_MODE_FUTURE_EGO_TIME,
    TARGET_MODE_SPATIAL_PATH,
    Bench2DriveDiffusionDataset,
    load_annotation,
)
from diffusiondrive.config_adapter import (  # noqa: E402
    DiffusionDriveRuntimeOverrides,
    build_diffusiondrive_config,
)
from diffusiondrive.model import V2TransfuserModel  # noqa: E402


class IndexedDataset(Dataset):
    def __init__(self, dataset: Bench2DriveDiffusionDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        item["index"] = index
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mode", choices=(TARGET_MODE_SPATIAL_PATH, TARGET_MODE_FUTURE_EGO_TIME), default=TARGET_MODE_SPATIAL_PATH)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--spatial-target-first-distance", type=float, default=2.5)
    parser.add_argument("--spatial-target-interval", type=float, default=1.0)
    parser.add_argument("--spatial-target-max-future-frames", type=int, default=120)
    parser.add_argument("--model-image-height", type=int, default=256)
    parser.add_argument("--model-image-width", type=int, default=1024)
    parser.add_argument("--no-jpeg-artifact", action="store_true")
    parser.add_argument("--balanced-scenarios", action="store_true")
    parser.add_argument("--max-samples-per-scenario", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
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
    return parser.parse_args()


def move_features_to_device(batch: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in batch["features"].items()
    }


def extract_model_state(checkpoint: Any, path: Path) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict):
            return checkpoint["model"]
        for key in ("state_dict", "model_state_dict", "ema_state_dict", "network", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError(f"Unable to locate model weights in checkpoint: {path}")


def load_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = extract_model_state(checkpoint, checkpoint_path)
    model.load_state_dict(state, strict=True)
    print(f"Loaded model checkpoint: {checkpoint_path}", flush=True)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * percent)), len(ordered) - 1)
    return ordered[index]


def target_path_length(target: torch.Tensor) -> float:
    origin = torch.zeros((1, target.shape[-1]), dtype=target.dtype)
    points = torch.cat([origin, target.cpu()], dim=0)
    return float(torch.linalg.norm(points[1:] - points[:-1], dim=-1).sum())


def command_from_annotation(annotation: dict) -> int:
    return int(annotation.get("command_far", annotation.get("command", annotation.get("next_command", 4))))


def sample_records(
    *,
    model: torch.nn.Module,
    dataset: Bench2DriveDiffusionDataset,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int | None,
) -> Iterator[dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            features = move_features_to_device(batch, device)
            targets = batch["targets"]["trajectory"].to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = model(features)
            prediction = outputs["trajectory"].detach().float()
            target = targets[..., :2].detach().float()
            l1 = torch.abs(prediction - target).mean(dim=(1, 2))
            ade = torch.linalg.norm(prediction - target, dim=-1).mean(dim=1)
            fde = torch.linalg.norm(prediction[:, -1] - target[:, -1], dim=-1)

            indices = batch["index"].tolist()
            for row, sample_index in enumerate(indices):
                route_dir, frame = dataset.samples[int(sample_index)]
                annotation = load_annotation(route_dir, int(frame))
                target_cpu = target[row].cpu()
                pred_cpu = prediction[row].cpu()
                yield {
                    "scenario": route_dir.parent.name,
                    "route": route_dir.name,
                    "frame": int(frame),
                    "l1": float(l1[row].cpu()),
                    "ade": float(ade[row].cpu()),
                    "fde": float(fde[row].cpu()),
                    "speed": float(annotation.get("speed", 0.0)),
                    "command": command_from_annotation(annotation),
                    "target_path_length": target_path_length(target_cpu),
                    "target_end_x": float(target_cpu[-1, 0]),
                    "target_end_y": float(target_cpu[-1, 1]),
                    "pred_end_x": float(pred_cpu[-1, 0]),
                    "pred_end_y": float(pred_cpu[-1, 1]),
                }

            if max_steps is not None and step >= max_steps:
                break


def write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {path}", flush=True)


def print_summary(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No samples evaluated.")
        return
    for key in ("l1", "ade", "fde"):
        values = [float(record[key]) for record in records]
        print(
            f"{key}: mean={mean(values):.4f} median={median(values):.4f} "
            f"p90={percentile(values, 0.90):.4f} p95={percentile(values, 0.95):.4f} max={max(values):.4f}",
            flush=True,
        )


def print_top(records: list[dict[str, Any]], top_k: int) -> None:
    print("Top samples by L1 trajectory error:", flush=True)
    header = (
        "rank scenario route frame l1 ade fde speed command "
        "target_path_length target_end pred_end"
    )
    print(header, flush=True)
    for rank, record in enumerate(sorted(records, key=lambda item: item["l1"], reverse=True)[:top_k], start=1):
        target_end = f"({record['target_end_x']:.2f},{record['target_end_y']:.2f})"
        pred_end = f"({record['pred_end_x']:.2f},{record['pred_end_y']:.2f})"
        print(
            f"{rank} {record['scenario']} {record['route']} {record['frame']} "
            f"{record['l1']:.4f} {record['ade']:.4f} {record['fde']:.4f} "
            f"{record['speed']:.3f} {record['command']} {record['target_path_length']:.3f} "
            f"{target_end} {pred_end}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    global_config = GlobalConfig()
    dd_config = build_diffusiondrive_config(
        global_config,
        DiffusionDriveRuntimeOverrides(anchor_path=args.anchor_path, backbone_path=args.backbone_path),
    )
    dd_config.camera_height = args.model_image_height
    dd_config.camera_width = args.model_image_width
    dd_config.__post_init__()

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
        balanced_scenarios=args.balanced_scenarios,
        max_samples_per_scenario=args.max_samples_per_scenario,
        sample_manifest_path=args.sample_manifest,
    )
    dataloader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"Samples: {len(dataset)}", flush=True)
    print(f"Scenario samples: {dataset.scenario_sample_counts}", flush=True)

    model = V2TransfuserModel(dd_config).to(device)
    load_model(model, args.checkpoint, device)

    records = list(
        sample_records(
            model=model,
            dataset=dataset,
            dataloader=dataloader,
            device=device,
            max_steps=args.max_steps,
        )
    )
    print_summary(records)
    print_top(records, args.top_k)
    if args.output_csv is not None:
        write_csv(args.output_csv, records)


if __name__ == "__main__":
    main()
