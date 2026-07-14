"""Adapter between garage runtime config and DiffusionDrive model config."""

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import DiffusionDriveConfig
from .inference_diagnostics import (
    normalize_diffusion_noise_mode,
    validate_diffusion_noise_seed,
)


@dataclass(frozen=True)
class DiffusionDriveRuntimeOverrides:
    """Runtime-only paths and knobs that should not live in the model config."""

    anchor_path: str = ""
    backbone_path: str = ""
    diffusion_infer_noise_mode: Optional[str] = None
    diffusion_infer_noise_seed: Optional[int] = None

    @classmethod
    def from_environment(cls, cwd: Optional[str] = None) -> "DiffusionDriveRuntimeOverrides":
        anchor_path = os.environ.get("DIFFUSIONDRIVE_ANCHOR_PATH", "")
        backbone_path = os.environ.get("DIFFUSIONDRIVE_BACKBONE_PATH", "")
        noise_mode = os.environ.get("DIFFUSIONDRIVE_DIFFUSION_NOISE_MODE")
        noise_seed_raw = os.environ.get("DIFFUSIONDRIVE_DIFFUSION_NOISE_SEED")
        noise_seed = None
        if noise_seed_raw is not None and noise_seed_raw != "":
            try:
                noise_seed = int(noise_seed_raw)
            except ValueError as exc:
                raise RuntimeError(
                    "DIFFUSIONDRIVE_DIFFUSION_NOISE_SEED must be an integer; "
                    f"got {noise_seed_raw!r}."
                ) from exc

        if not backbone_path:
            default_backbone = os.path.join(cwd or os.getcwd(), "pytorch_model.bin")
            if os.path.exists(default_backbone):
                backbone_path = default_backbone

        return cls(
            anchor_path=anchor_path,
            backbone_path=backbone_path,
            diffusion_infer_noise_mode=noise_mode,
            diffusion_infer_noise_seed=noise_seed,
        )


def build_diffusiondrive_config(
    global_config: Any,
    overrides: Optional[DiffusionDriveRuntimeOverrides] = None,
) -> DiffusionDriveConfig:
    """Build the DiffusionDrive model config from the garage runtime config."""

    overrides = overrides or DiffusionDriveRuntimeOverrides()
    dd_config = DiffusionDriveConfig(
        plan_anchor_path=overrides.anchor_path,
        bkb_path=overrides.backbone_path,
    )
    if overrides.diffusion_infer_noise_mode is not None:
        dd_config.diffusion_infer_noise_mode = normalize_diffusion_noise_mode(
            overrides.diffusion_infer_noise_mode
        )
    if overrides.diffusion_infer_noise_seed is not None:
        dd_config.diffusion_infer_noise_seed = validate_diffusion_noise_seed(
            overrides.diffusion_infer_noise_seed
        )
    if dd_config.plan_anchor_path and os.path.isfile(dd_config.plan_anchor_path):
        plan_anchor_shape = np.load(dd_config.plan_anchor_path, mmap_mode="r").shape
        if len(plan_anchor_shape) == 3:
            dd_config.num_anchor_modes = int(plan_anchor_shape[0])
            dd_config.trajectory_sampling.time_horizon = (
                float(plan_anchor_shape[1]) * dd_config.trajectory_sampling.interval_length
            )

    dd_config.lidar_min_x = global_config.min_x
    dd_config.lidar_max_x = global_config.max_x
    dd_config.lidar_min_y = global_config.min_y
    dd_config.lidar_max_y = global_config.max_y
    dd_config.lidar_resolution_height = global_config.lidar_resolution_height
    dd_config.lidar_resolution_width = global_config.lidar_resolution_width
    dd_config.lidar_seq_len = global_config.lidar_seq_len
    dd_config.use_ground_plane = global_config.use_ground_plane
    dd_config.max_height_lidar = global_config.max_height_lidar
    dd_config.pixels_per_meter = global_config.pixels_per_meter
    dd_config.hist_max_per_pixel = global_config.hist_max_per_pixel
    dd_config.lidar_split_height = global_config.lidar_split_height
    dd_config.__post_init__()

    validate_diffusiondrive_config(dd_config)
    return dd_config


def validate_diffusiondrive_config(config: DiffusionDriveConfig) -> None:
    """Fail early for model/runtime mismatches that otherwise surface deep in inference."""

    if not config.plan_anchor_path:
        raise RuntimeError("DIFFUSIONDRIVE_ANCHOR_PATH is required for DiffusionDrive (plan anchor .npy).")
    if not os.path.isfile(config.plan_anchor_path):
        raise RuntimeError(f"DiffusionDrive plan anchor file does not exist: {config.plan_anchor_path}")

    plan_anchor = np.load(config.plan_anchor_path, mmap_mode="r")
    expected_num_poses = config.trajectory_sampling.num_poses
    expected_num_modes = config.num_anchor_modes
    if plan_anchor.ndim != 3:
        raise RuntimeError(
            "DiffusionDrive plan anchor must have shape (num_modes, num_poses, 2); "
            f"got {plan_anchor.shape}."
        )
    if plan_anchor.shape[0] != expected_num_modes:
        raise RuntimeError(
            "DiffusionDrive plan anchor mode count does not match num_anchor_modes: "
            f"anchor shape {plan_anchor.shape}, expected {expected_num_modes} modes."
        )
    if plan_anchor.shape[1] != expected_num_poses:
        raise RuntimeError(
            "DiffusionDrive plan anchor pose count does not match trajectory_sampling.num_poses: "
            f"anchor shape {plan_anchor.shape}, expected {expected_num_poses} poses."
        )
    if plan_anchor.shape[2] != 2:
        raise RuntimeError(
            "DiffusionDrive plan anchor must be 2D (x, y) for the current CARLA port; "
            f"got last dimension {plan_anchor.shape[2]}."
        )

    expected_status_dim = config.command_dim + config.speed_dim
    if config.status_dim != expected_status_dim:
        raise RuntimeError(
            f"DiffusionDrive status_dim mismatch: got {config.status_dim}, expected {expected_status_dim}."
        )
    if config.route_condition_enabled and config.route_condition_dim != 4:
        raise RuntimeError(
            "DiffusionDrive route_condition_dim must be 4 when route condition is enabled; "
            f"got {config.route_condition_dim}."
        )

    if config.camera_height % 32 != 0 or config.camera_width % 32 != 0:
        raise RuntimeError(
            "DiffusionDrive camera input dimensions must be divisible by 32; "
            f"got {config.camera_height}x{config.camera_width}."
        )
    if config.lidar_resolution_height % 32 != 0 or config.lidar_resolution_width % 32 != 0:
        raise RuntimeError(
            "DiffusionDrive LiDAR input dimensions must be divisible by 32; "
            f"got {config.lidar_resolution_height}x{config.lidar_resolution_width}."
        )

    expected_lidar_channels = config.lidar_seq_len * (2 if config.use_ground_plane else 1)
    if expected_lidar_channels <= 0:
        raise RuntimeError(
            "DiffusionDrive LiDAR channels must be positive; "
            f"lidar_seq_len={config.lidar_seq_len}, use_ground_plane={config.use_ground_plane}."
        )

    if config.diffusion_num_train_timesteps <= 0:
        raise RuntimeError(
            "DiffusionDrive diffusion_num_train_timesteps must be positive; "
            f"got {config.diffusion_num_train_timesteps}."
        )
    if not (0 <= config.diffusion_train_timestep_min < config.diffusion_train_timestep_max):
        raise RuntimeError(
            "DiffusionDrive training timestep range must satisfy "
            "0 <= min < max; "
            f"got min={config.diffusion_train_timestep_min}, max={config.diffusion_train_timestep_max}."
        )
    if config.diffusion_train_timestep_max > config.diffusion_num_train_timesteps:
        raise RuntimeError(
            "DiffusionDrive diffusion_train_timestep_max must be <= diffusion_num_train_timesteps; "
            f"got max={config.diffusion_train_timestep_max}, "
            f"num_train_timesteps={config.diffusion_num_train_timesteps}."
        )
    if config.diffusion_infer_step_num <= 0:
        raise RuntimeError(
            "DiffusionDrive diffusion_infer_step_num must be positive; "
            f"got {config.diffusion_infer_step_num}."
        )
    if config.diffusion_infer_timestep_span <= 0:
        raise RuntimeError(
            "DiffusionDrive diffusion_infer_timestep_span must be positive; "
            f"got {config.diffusion_infer_timestep_span}."
        )
    if not (0 <= config.diffusion_infer_trunc_timesteps < config.diffusion_num_train_timesteps):
        raise RuntimeError(
            "DiffusionDrive diffusion_infer_trunc_timesteps must satisfy "
            "0 <= trunc < diffusion_num_train_timesteps; "
            f"got trunc={config.diffusion_infer_trunc_timesteps}, "
            f"num_train_timesteps={config.diffusion_num_train_timesteps}."
        )
    config.diffusion_infer_noise_mode = normalize_diffusion_noise_mode(
        config.diffusion_infer_noise_mode
    )
    config.diffusion_infer_noise_seed = validate_diffusion_noise_seed(
        config.diffusion_infer_noise_seed
    )
