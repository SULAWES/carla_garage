#!/usr/bin/env python3
"""Minimal CARLA-native DiffusionDrive training entrypoint."""

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from config import GlobalConfig
from diffusiondrive.carla_native_dataset import (
    Bench2DriveDiffusionDataset,
    target_speed_label_metadata,
)
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
    parser.add_argument("--image-encoder-lr-mult", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("none", "cosine", "multistep"), default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-steps", default="70")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
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
    parser.add_argument("--balanced-scenarios", action="store_true")
    parser.add_argument("--max-samples-per-scenario", type=int, default=None)
    parser.add_argument("--hard-left-turn-stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-left-turn-command", type=int, default=1)
    parser.add_argument("--hard-left-turn-speed-threshold", type=float, default=0.1)
    parser.add_argument("--hard-left-turn-y-threshold", type=float, default=4.0)
    parser.add_argument("--dataset-stats-max-samples", type=int, default=4096)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--val-sample-manifest", type=Path, default=None)
    parser.add_argument("--rebuild-sample-manifest", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--val-route-glob", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distributed", choices=("auto", "none", "ddp"), default="auto")
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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
    torch.set_num_threads(1)
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        cv2.setNumThreads(0)
    except ImportError:
        pass


def init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    if args.distributed == "none":
        return False, 0, 1, 0

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.distributed == "auto" and world_size <= 1:
        return False, 0, 1, 0
    if args.distributed == "ddp" and world_size <= 1:
        raise RuntimeError("--distributed ddp requires torchrun or WORLD_SIZE > 1")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_barrier(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


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
        "optimizer": {
            "type": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_encoder_lr_mult": args.image_encoder_lr_mult,
            "image_encoder_lr": args.lr * args.image_encoder_lr_mult,
        },
        "distributed": {
            "mode": args.distributed,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        },
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
            "balanced_scenarios": args.balanced_scenarios,
            "max_samples_per_scenario": args.max_samples_per_scenario,
            "sample_manifest": make_json_safe(args.sample_manifest),
            "val_sample_manifest": make_json_safe(args.val_sample_manifest),
            "rebuild_sample_manifest": args.rebuild_sample_manifest,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.num_workers > 0 and args.persistent_workers,
        },
        "hard_case_weighting": {
            "enabled": args.hard_left_turn_stop_loss_weight != 1.0,
            "loss_weight": args.hard_left_turn_stop_loss_weight,
            "applies_to": "training only; validation and eval-only losses remain unweighted",
            "criteria": {
                "command": args.hard_left_turn_command,
                "speed_lt": args.hard_left_turn_speed_threshold,
                "abs_target_end_y_gt": args.hard_left_turn_y_threshold,
            },
            "failure_mode": "low-speed command=1 samples whose spatial target ends with large lateral offset",
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
        "target_speed_label": target_speed_label_metadata(),
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
    if args.image_encoder_lr_mult <= 0:
        raise RuntimeError(f"--image-encoder-lr-mult must be > 0, got {args.image_encoder_lr_mult}")

    image_encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder" in name:
            image_encoder_params.append(param)
        else:
            other_params.append(param)

    if not image_encoder_params:
        print("Warning: no parameters matched image_encoder lr multiplier.", flush=True)
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    param_groups = [
        {"params": other_params, "lr": args.lr, "name": "default"},
        {"params": image_encoder_params, "lr": args.lr * args.image_encoder_lr_mult, "name": "image_encoder"},
    ]
    print(
        "Optimizer parameter groups: "
        f"default={sum(param.numel() for param in other_params)} params lr={args.lr:.6g}; "
        f"image_encoder={sum(param.numel() for param in image_encoder_params)} params "
        f"lr={args.lr * args.image_encoder_lr_mult:.6g} "
        f"(mult={args.image_encoder_lr_mult:.6g})",
        flush=True,
    )
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


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


def zero_loss_for_ddp_outputs(outputs: Dict[str, Any]) -> torch.Tensor | float:
    """Touch auxiliary outputs so DDP sees zero gradients for currently unsupervised heads."""

    zero_loss: torch.Tensor | float = 0.0
    for value in outputs.values():
        if torch.is_tensor(value) and value.requires_grad:
            zero_loss = zero_loss + value.sum() * 0.0
    return zero_loss


def format_scenario_sample_counts(counts: dict[str, int], limit: int = 12) -> str:
    if not counts:
        return "{}"
    items = sorted(counts.items())
    shown = ", ".join(f"{name}={count}" for name, count in items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit} more)"
    return shown


def _speed_bin(speed: float) -> str:
    if speed < 0.1:
        return "<0.1"
    if speed < 2.0:
        return "0.1-2"
    if speed < 5.0:
        return "2-5"
    return ">=5"


def _abs_y_bin(abs_y: float) -> str:
    if abs_y < 2.0:
        return "<2"
    if abs_y < 4.0:
        return "2-4"
    return ">=4"


def summarize_sample_distribution(
    dataset: Bench2DriveDiffusionDataset,
    *,
    name: str,
    output_dir: Path,
    max_samples: int,
) -> dict[str, Any]:
    total = len(dataset.samples)
    if max_samples <= 0 or total == 0:
        summary = {"total_samples": total, "sampled_samples": 0, "skipped": True}
    else:
        sampled = min(total, max_samples)
        if sampled == total:
            indices = list(range(total))
        else:
            indices = sorted({int(index) for index in np.linspace(0, total - 1, sampled)})

        command_counts: Counter[str] = Counter()
        speed_bins: Counter[str] = Counter()
        target_speed_class_counts: Counter[str] = Counter()
        brake_counts: Counter[str] = Counter()
        abs_target_end_y_bins: Counter[str] = Counter()
        hard_by_scenario: Counter[str] = Counter()
        hard_count = 0
        target_speed_valid_count = 0
        target_end_y_values: list[float] = []

        for index in indices:
            sample = dataset.get_sample_summary(index)
            route_dir = sample["route_dir"]
            trajectory = sample["trajectory"]
            command = int(sample["command"])
            speed = float(sample["speed"])
            abs_target_end_y = abs(float(trajectory[-1, 1]))

            command_counts[str(command)] += 1
            speed_bins[_speed_bin(speed)] += 1
            target_speed_class_counts[str(int(sample["target_speed_class"]))] += 1
            brake_counts[str(int(bool(sample["brake"])))] += 1
            target_speed_valid_count += int(sample["target_speed_label_valid"])
            abs_target_end_y_bins[_abs_y_bin(abs_target_end_y)] += 1
            target_end_y_values.append(abs_target_end_y)

            if sample["hard_left_turn_stop"]:
                hard_count += 1
                hard_by_scenario[route_dir.parent.name] += 1

        summary = {
            "total_samples": total,
            "sampled_samples": len(indices),
            "hard_left_turn_stop_count": hard_count,
            "hard_left_turn_stop_fraction": hard_count / max(len(indices), 1),
            "hard_left_turn_stop_criteria": {
                "command": dataset.hard_left_turn_command,
                "speed_lt": dataset.hard_left_turn_speed_threshold,
                "abs_target_end_y_gt": dataset.hard_left_turn_y_threshold,
            },
            "command_counts": dict(sorted(command_counts.items())),
            "speed_bins_mps": dict(sorted(speed_bins.items())),
            "target_speed_label_schema": target_speed_label_metadata(),
            "target_speed_class_counts": dict(sorted(target_speed_class_counts.items())),
            "target_speed_label_valid_count": target_speed_valid_count,
            "target_speed_label_valid_fraction": target_speed_valid_count / max(len(indices), 1),
            "brake_counts": dict(sorted(brake_counts.items())),
            "abs_target_end_y_bins_m": dict(sorted(abs_target_end_y_bins.items())),
            "hard_left_turn_stop_by_scenario": dict(sorted(hard_by_scenario.items())),
            "abs_target_end_y_mean": float(np.mean(target_end_y_values)) if target_end_y_values else 0.0,
            "abs_target_end_y_p95": float(np.percentile(target_end_y_values, 95)) if target_end_y_values else 0.0,
        }

    path = output_dir / f"{name}_sample_distribution.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(summary), file, indent=2, sort_keys=True)
    print(
        f"{name} distribution: sampled={summary['sampled_samples']}/{summary['total_samples']} "
        f"hard_left_turn_stop={summary.get('hard_left_turn_stop_count', 0)} "
        f"fraction={summary.get('hard_left_turn_stop_fraction', 0.0):.4f} "
        f"file={path}",
        flush=True,
    )
    return summary


def build_dataset(
    root_dirs: list[Path],
    args: argparse.Namespace,
    global_config: GlobalConfig,
    dd_config: Any,
    frame_sampling: int,
    max_samples: Optional[int],
    route_glob: str,
    enable_hard_case_weight: bool,
    sample_manifest_path: Optional[Path],
) -> Bench2DriveDiffusionDataset:
    return Bench2DriveDiffusionDataset(
        root_dirs,
        config=global_config,
        num_poses=dd_config.trajectory_sampling.num_poses,
        future_stride=args.future_stride,
        frame_sampling=frame_sampling,
        max_samples=max_samples,
        route_glob=route_glob,
        model_image_size=(dd_config.camera_height, dd_config.camera_width),
        jpeg_artifact=not args.no_jpeg_artifact,
        target_mode=args.target_mode,
        spatial_target_first_distance=args.spatial_target_first_distance,
        spatial_target_interval=args.spatial_target_interval,
        spatial_target_max_future_frames=args.spatial_target_max_future_frames,
        balanced_scenarios=args.balanced_scenarios,
        max_samples_per_scenario=args.max_samples_per_scenario,
        hard_left_turn_stop_loss_weight=(
            args.hard_left_turn_stop_loss_weight if enable_hard_case_weight else 1.0
        ),
        hard_left_turn_command=args.hard_left_turn_command,
        hard_left_turn_speed_threshold=args.hard_left_turn_speed_threshold,
        hard_left_turn_y_threshold=args.hard_left_turn_y_threshold,
        sample_manifest_path=sample_manifest_path,
        rebuild_sample_manifest=args.rebuild_sample_manifest,
    )


def build_dataloader(
    dataset: Bench2DriveDiffusionDataset,
    args: argparse.Namespace,
    device: torch.device,
    shuffle: bool,
    drop_last: bool,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": drop_last,
        "worker_init_fn": seed_worker if args.num_workers > 0 else None,
        "persistent_workers": args.num_workers > 0 and args.persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


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
    total_loss_dict: Dict[str, float] = {}
    count = 0
    for batch in dataloader:
        features, targets = move_features_to_device(batch, device)
        outputs = model(features, targets=targets)
        total = compute_total_loss(outputs, dd_config)
        total_loss += float(total.detach().cpu())
        total_trajectory_loss += float(outputs["trajectory_loss"].detach().cpu())
        for name, value in outputs.get("trajectory_loss_dict", {}).items():
            total_loss_dict[name] = total_loss_dict.get(name, 0.0) + float(value.detach().cpu())
        count += 1
        if max_steps is not None and count >= max_steps:
            break
    if was_training:
        model.train()
    denom = max(count, 1)
    return {
        "loss": total_loss / denom,
        "trajectory_loss": total_trajectory_loss / denom,
        **{name: value / denom for name, value in sorted(total_loss_dict.items())},
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
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
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


def load_model_for_eval(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    if args.resume_file:
        checkpoint = torch.load(args.resume_file, map_location=device, weights_only=False)
        if "model" not in checkpoint:
            raise RuntimeError(f"Eval checkpoint does not contain a 'model' state: {args.resume_file}")
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded eval model from training checkpoint: {args.resume_file}", flush=True)
        return
    if args.load_file:
        load_checkpoint_partial(model, args.load_file, device)
        return
    raise RuntimeError("Eval-only mode requires --resume-file for a training checkpoint or --load-file for model weights.")


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    loss_parts = ", ".join(
        f"{name}={value:.4f}"
        for name, value in sorted(metrics.items())
        if name not in {"loss", "trajectory_loss", "steps"}
    )
    suffix = f" {loss_parts}" if loss_parts else ""
    print(
        f"{prefix} loss={metrics['loss']:.4f} "
        f"trajectory_unweighted={metrics['trajectory_loss']:.4f} steps={int(metrics['steps'])}{suffix}",
        flush=True,
    )


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
        "model": unwrap_model(model).state_dict(),
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
    is_distributed, rank, world_size, local_rank = init_distributed(args)
    seed_everything(args.seed)

    if is_distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    output_dir = args.logdir / args.id
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(is_distributed)

    global_config = GlobalConfig()
    dd_config = build_diffusiondrive_config(
        global_config,
        DiffusionDriveRuntimeOverrides(anchor_path=args.anchor_path, backbone_path=args.backbone_path),
    )
    dd_config.camera_height = args.model_image_height
    dd_config.camera_width = args.model_image_width
    dd_config.__post_init__()
    apply_runtime_training_overrides(dd_config, args)
    if is_main_process(rank):
        write_run_config(output_dir, args, global_config, dd_config)

    if args.eval_only:
        if not is_main_process(rank):
            distributed_barrier(is_distributed)
            cleanup_distributed(is_distributed)
            return
        eval_root_dirs = args.val_root_dir or args.root_dir
        eval_frame_sampling = args.val_frame_sampling or args.frame_sampling
        eval_max_samples = args.val_max_samples if args.val_root_dir else args.max_samples
        eval_route_glob = args.val_route_glob or args.route_glob
        eval_manifest = args.val_sample_manifest if args.val_root_dir else args.sample_manifest
        eval_dataset = build_dataset(
            eval_root_dirs,
            args,
            global_config,
            dd_config,
            frame_sampling=eval_frame_sampling,
            max_samples=eval_max_samples,
            route_glob=eval_route_glob,
            enable_hard_case_weight=False,
            sample_manifest_path=eval_manifest,
        )
        eval_dataloader = build_dataloader(eval_dataset, args, device, shuffle=False, drop_last=False)
        print(f"Eval samples: {len(eval_dataset)}", flush=True)
        print(f"Eval scenario samples: {format_scenario_sample_counts(eval_dataset.scenario_sample_counts)}", flush=True)
        summarize_sample_distribution(
            eval_dataset,
            name="eval",
            output_dir=output_dir,
            max_samples=args.dataset_stats_max_samples,
        )
        print(f"Output dir: {output_dir}", flush=True)

        model = V2TransfuserModel(dd_config).to(device)
        load_model_for_eval(model, args, device)
        metrics = evaluate(model, eval_dataloader, device, dd_config, args.max_val_steps)
        print_metrics("eval", metrics)
        distributed_barrier(is_distributed)
        cleanup_distributed(is_distributed)
        return

    dataset = build_dataset(
        args.root_dir,
        args,
        global_config,
        dd_config,
        frame_sampling=args.frame_sampling,
        max_samples=args.max_samples,
        route_glob=args.route_glob,
        enable_hard_case_weight=True,
        sample_manifest_path=args.sample_manifest,
    )
    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=args.drop_last,
        )
        if is_distributed
        else None
    )
    dataloader = build_dataloader(
        dataset,
        args,
        device,
        shuffle=True,
        drop_last=args.drop_last,
        sampler=train_sampler,
    )
    val_dataloader = None
    if args.val_root_dir and is_main_process(rank):
        val_dataset = build_dataset(
            args.val_root_dir,
            args,
            global_config,
            dd_config,
            frame_sampling=args.val_frame_sampling or args.frame_sampling,
            max_samples=args.val_max_samples,
            route_glob=args.val_route_glob or args.route_glob,
            enable_hard_case_weight=False,
            sample_manifest_path=args.val_sample_manifest,
        )
        val_dataloader = build_dataloader(val_dataset, args, device, shuffle=False, drop_last=False)
        print(f"Validation samples: {len(val_dataset)}", flush=True)
        print(f"Validation scenario samples: {format_scenario_sample_counts(val_dataset.scenario_sample_counts)}", flush=True)
        summarize_sample_distribution(
            val_dataset,
            name="validation",
            output_dir=output_dir,
            max_samples=args.dataset_stats_max_samples,
        )
    if is_main_process(rank):
        print(f"Dataset samples: {len(dataset)}", flush=True)
        print(f"Dataset scenario samples: {format_scenario_sample_counts(dataset.scenario_sample_counts)}", flush=True)
        if is_distributed:
            print(
                f"Distributed training: world_size={world_size} rank={rank} local_rank={local_rank} "
                f"per_rank_batch_size={args.batch_size} global_batch_size={args.batch_size * world_size} "
                f"steps_per_epoch_per_rank={len(dataloader)}",
                flush=True,
            )
        summarize_sample_distribution(
            dataset,
            name="train",
            output_dir=output_dir,
            max_samples=args.dataset_stats_max_samples,
        )
        print(f"Output dir: {output_dir}", flush=True)
    distributed_barrier(is_distributed)

    model = V2TransfuserModel(dd_config).to(device)
    model.train()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, total_planned_steps(dataloader, args))

    if args.resume_file:
        start_epoch, global_step = load_training_checkpoint(args.resume_file, model, optimizer, scheduler, device)
    else:
        load_checkpoint_partial(model, args.load_file, device)
        start_epoch, global_step = 0, 0

    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    running_loss = 0.0
    running_count = 0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in dataloader:
            features, targets = move_features_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, targets=targets)
            loss = compute_total_loss(outputs, dd_config)
            if is_distributed:
                loss = loss + zero_loss_for_ddp_outputs(outputs)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1
            if is_main_process(rank) and (global_step == 1 or global_step % args.log_every == 0):
                avg_loss = running_loss / max(running_count, 1)
                running_loss = 0.0
                running_count = 0
                loss_dict = outputs.get("trajectory_loss_dict", {})
                loss_parts = ", ".join(
                    f"{name}={float(value.detach().cpu()):.4f}" for name, value in sorted(loss_dict.items())
                )
                hard_count = int(targets["hard_left_turn_stop"].detach().sum().cpu())
                mean_sample_weight = float(targets["trajectory_sample_weight"].detach().mean().cpu())
                lr = optimizer.param_groups[0]["lr"]
                image_encoder_lr = next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("name") == "image_encoder"
                    ),
                    None,
                )
                lr_text = f"lr={lr:.6g}"
                if image_encoder_lr is not None:
                    lr_text += f" image_encoder_lr={image_encoder_lr:.6g}"
                print(
                    f"epoch={epoch} step={global_step} {lr_text} loss={float(loss.detach().cpu()):.4f} "
                    f"avg={avg_loss:.4f} trajectory_unweighted={float(outputs['trajectory_loss'].detach().cpu()):.4f} "
                    f"hard_left_turn_stop={hard_count}/{targets['hard_left_turn_stop'].numel()} "
                    f"mean_sample_weight={mean_sample_weight:.3f} {loss_parts}",
                    flush=True,
                )

            if val_dataloader is not None and args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                metrics = evaluate(unwrap_model(model), val_dataloader, device, dd_config, args.max_val_steps)
                print_metrics(f"validation step={global_step}", metrics)
            if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                distributed_barrier(is_distributed)

            if is_main_process(rank) and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(output_dir, model, optimizer, scheduler, epoch, global_step, args, dd_config)
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                distributed_barrier(is_distributed)

            if args.max_steps is not None and global_step >= args.max_steps:
                if is_main_process(rank):
                    save_checkpoint(output_dir, model, optimizer, scheduler, epoch, global_step, args, dd_config)
                distributed_barrier(is_distributed)
                cleanup_distributed(is_distributed)
                return

    if val_dataloader is not None and args.val_every_steps == 0 and is_main_process(rank):
        metrics = evaluate(unwrap_model(model), val_dataloader, device, dd_config, args.max_val_steps)
        print_metrics("validation final", metrics)
    distributed_barrier(is_distributed)
    if is_main_process(rank):
        save_checkpoint(output_dir, model, optimizer, scheduler, args.epochs - 1, global_step, args, dd_config)
    distributed_barrier(is_distributed)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
