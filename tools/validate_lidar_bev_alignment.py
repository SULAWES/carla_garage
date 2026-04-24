#!/usr/bin/env python3
"""Synthetic validation for CARLA Garage LiDAR/BEV alignment.

This script mirrors the current formulas used by:
  - team_code/transfuser_utils.py
  - team_code/data.py
  - team_code/diffusiondrive_agent.py
  - team_code/sensor_agent.py

It is intentionally dependency-light so it can run even outside the full
CARLA/PyTorch environment. The goal is to answer three concrete questions:

1. Does raw LiDAR -> ego-frame -> BEV use a self-consistent axis convention?
2. Does online realignment match the offline training-time realignment math?
3. Does DiffusionDrive's temporal indexing pair LiDAR history with the right
   ego pose?

The script also reports whether the current realignment math matches a
canonical rigid SE(2) transform on synthetic static-world points. If this
check warns, it does not mean training/inference are inconsistent; it means
the shared online/offline math differs from the canonical transform under the
synthetic assumptions used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarConfig:
    lidar_pos: tuple[float, float, float] = (0.0, 0.0, 2.5)
    lidar_yaw_deg: float = -90.0
    min_x: float = -32.0
    max_x: float = 32.0
    min_y: float = -32.0
    max_y: float = 32.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5
    max_height_lidar: float = 100.0
    lidar_split_height: float = 0.2


CFG = LidarConfig()


def normalize_angle(x: float) -> float:
    """Mirror transfuser_utils.normalize_angle()."""
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def lidar_to_ego_coordinate(raw_points: np.ndarray, cfg: LidarConfig = CFG) -> np.ndarray:
    """Mirror transfuser_utils.lidar_to_ego_coordinate()."""
    yaw = np.deg2rad(cfg.lidar_yaw_deg)
    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array(cfg.lidar_pos)
    return (rotation_matrix @ raw_points[:, :3].T).T + translation


def algin_lidar(lidar: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    """Mirror transfuser_utils.algin_lidar()."""
    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (rotation_matrix.T @ (lidar - translation).T).T


def online_align_lidar(lidar: np.ndarray, pose_from: tuple[float, float, float],
                       pose_to: tuple[float, float, float]) -> np.ndarray:
    """Mirror DiffusionDriveAgent.align_lidar()."""
    x, y, orientation = pose_from
    x_target, y_target, orientation_target = pose_to
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = normalize_angle(orientation_target - orientation)
    rotation_matrix = np.array(
        [
            [np.cos(orientation_target), -np.sin(orientation_target), 0.0],
            [np.sin(orientation_target), np.cos(orientation_target), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pos_diff = rotation_matrix.T @ pos_diff
    return algin_lidar(lidar, pos_diff, rot_diff)


def offline_align_lidar(
    lidar: np.ndarray,
    measurement_from: dict[str, object],
    measurement_to: dict[str, object],
) -> np.ndarray:
    """Mirror CARLA_Data.align() with zero augmentation."""
    pos_1 = np.array([measurement_to["pos_global"][0], measurement_to["pos_global"][1], 0.0])
    pos_0 = np.array([measurement_from["pos_global"][0], measurement_from["pos_global"][1], 0.0])
    pos_diff = pos_1 - pos_0
    rot_diff = normalize_angle(float(measurement_to["theta"]) - float(measurement_from["theta"]))
    rotation_matrix = np.array(
        [
            [np.cos(float(measurement_to["theta"])), -np.sin(float(measurement_to["theta"])), 0.0],
            [np.sin(float(measurement_to["theta"])), np.cos(float(measurement_to["theta"])), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pos_diff = rotation_matrix.T @ pos_diff
    return algin_lidar(lidar, pos_diff, rot_diff)


def world_to_ego(world_points: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    """Canonical rigid transform from world to ego frame."""
    x, y, theta = pose
    translation = np.array([x, y, 0.0])
    rotation_matrix = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (rotation_matrix.T @ (world_points - translation).T).T


def lidar_to_histogram_features(lidar: np.ndarray, cfg: LidarConfig = CFG) -> np.ndarray:
    """Mirror CARLA_Data.lidar_to_histogram_features() for use_ground_plane=False."""

    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        xbins = np.linspace(cfg.min_x, cfg.max_x, int((cfg.max_x - cfg.min_x) * cfg.pixels_per_meter) + 1)
        ybins = np.linspace(cfg.min_y, cfg.max_y, int((cfg.max_y - cfg.min_y) * cfg.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > cfg.hist_max_per_pixel] = cfg.hist_max_per_pixel
        return (hist / cfg.hist_max_per_pixel).T

    lidar = lidar[lidar[:, 2] < cfg.max_height_lidar]
    above = lidar[lidar[:, 2] > cfg.lidar_split_height]
    features = np.stack([splat_points(above)], axis=-1)
    return np.transpose(features, (2, 0, 1)).astype(np.float32)


def print_result(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "WARN"
    print(f"[{status}] {name}: {detail}")


def validate_axis_mapping() -> bool:
    """Check raw-sensor axes land in expected ego/BEV directions."""
    raw_points = np.array(
        [
            [0.0, 10.0, -1.0, 1.0],   # expected ego-front
            [-10.0, 0.0, -1.0, 1.0],  # expected ego-right
            [10.0, 0.0, -1.0, 1.0],   # expected ego-left
            [0.0, -10.0, -1.0, 1.0],  # expected ego-back
        ],
        dtype=np.float32,
    )
    ego_points = lidar_to_ego_coordinate(raw_points)
    expected = np.array(
        [
            [10.0, 0.0, 1.5],
            [0.0, 10.0, 1.5],
            [0.0, -10.0, 1.5],
            [-10.0, 0.0, 1.5],
        ],
        dtype=np.float32,
    )
    max_err = float(np.abs(ego_points - expected).max())
    bev_nonzero = np.argwhere(lidar_to_histogram_features(ego_points)[0] > 0)
    passed = max_err < 1e-6 and len(bev_nonzero) == 4
    print_result(
        "Axis Mapping",
        passed,
        f"max ego-frame error={max_err:.6f}, nonzero BEV pixels={bev_nonzero.tolist()}",
    )
    return passed


def validate_online_offline_equivalence() -> bool:
    """Check current inference-time and training-time formulas are identical."""
    rng = np.random.default_rng(0)
    world = np.column_stack(
        [
            rng.uniform(5.0, 25.0, size=256),
            rng.uniform(-8.0, 8.0, size=256),
            rng.uniform(0.5, 2.5, size=256),
        ]
    ).astype(np.float32)
    pose0 = (0.0, 0.0, np.deg2rad(0.0))
    pose1 = (1.7, -0.4, np.deg2rad(15.0))
    cloud0 = world_to_ego(world, pose0)
    cloud1_online = online_align_lidar(cloud0, pose0, pose1)
    cloud1_offline = offline_align_lidar(
        cloud0,
        {"pos_global": pose0[:2], "theta": pose0[2]},
        {"pos_global": pose1[:2], "theta": pose1[2]},
    )
    max_err = float(np.abs(cloud1_online - cloud1_offline).max())
    passed = max_err < 1e-9
    print_result("Online/Offline Formula Match", passed, f"max point error={max_err:.6f}")
    return passed


def validate_temporal_index_pairing() -> bool:
    """Check DiffusionDrive's negative indexing is better paired than sensor_agent's indexing."""
    rng = np.random.default_rng(1)
    world = np.column_stack(
        [
            rng.uniform(5.0, 25.0, size=512),
            rng.uniform(-10.0, 10.0, size=512),
            rng.uniform(0.5, 2.5, size=512),
        ]
    ).astype(np.float32)
    poses = [
        (0.0, 0.0, np.deg2rad(0.0)),
        (1.0, 0.1, np.deg2rad(3.0)),
        (2.1, 0.25, np.deg2rad(6.0)),
        (3.2, 0.45, np.deg2rad(9.0)),
        (4.4, 0.7, np.deg2rad(12.0)),
        (5.6, 1.0, np.deg2rad(15.0)),
    ]
    clouds = [world_to_ego(world, pose) for pose in poses]
    state_log = [np.array([pose[0], pose[1], pose[2], 0.0], dtype=np.float32) for pose in poses]

    # Simulate lidar_seq_len=2 with data_save_freq=5, so lidar_indices=[0, 5].
    # The historical cloud is lidar_buffer[-6], which should pair with state_log[-6].
    historical_cloud = clouds[0]
    current_pose = poses[-1]
    canonical_current = world_to_ego(world, current_pose)
    diffusion_realign = online_align_lidar(historical_cloud, tuple(state_log[-6][:3]), current_pose)
    legacy_realign = online_align_lidar(historical_cloud, tuple(state_log[5][:3]), current_pose)

    diffusion_point_err = float(np.abs(diffusion_realign - canonical_current).max())
    legacy_point_err = float(np.abs(legacy_realign - canonical_current).max())
    diffusion_bev_err = float(
        np.abs(lidar_to_histogram_features(diffusion_realign) - lidar_to_histogram_features(canonical_current)).sum()
    )
    legacy_bev_err = float(
        np.abs(lidar_to_histogram_features(legacy_realign) - lidar_to_histogram_features(canonical_current)).sum()
    )

    passed = diffusion_point_err < legacy_point_err and diffusion_bev_err < legacy_bev_err
    print_result(
        "Temporal Index Pairing",
        passed,
        (
            f"DiffusionDrive point/bev err={diffusion_point_err:.6f}/{diffusion_bev_err:.6f}, "
            f"sensor_agent point/bev err={legacy_point_err:.6f}/{legacy_bev_err:.6f}"
        ),
    )
    return passed


def validate_canonical_se2_consistency() -> bool:
    """Report whether the shared formula matches a standard rigid transform."""
    rng = np.random.default_rng(2)
    world = np.column_stack(
        [
            rng.uniform(5.0, 25.0, size=256),
            rng.uniform(-8.0, 8.0, size=256),
            rng.uniform(0.5, 2.5, size=256),
        ]
    ).astype(np.float32)
    pose0 = (0.0, 0.0, np.deg2rad(0.0))
    pose1 = (1.7, -0.4, np.deg2rad(15.0))
    cloud0 = world_to_ego(world, pose0)
    canonical_current = world_to_ego(world, pose1)
    shared_formula = online_align_lidar(cloud0, pose0, pose1)
    point_err = float(np.abs(shared_formula - canonical_current).max())
    bev_err = float(np.abs(lidar_to_histogram_features(shared_formula) - lidar_to_histogram_features(canonical_current)).sum())
    passed = point_err < 1e-6 and bev_err < 1e-6
    print_result(
        "Canonical SE(2) Check",
        passed,
        f"point error={point_err:.6f}, bev error={bev_err:.6f}",
    )
    return passed


def main() -> int:
    print("Synthetic LiDAR/BEV alignment validation")
    print("Mirrors current formulas in team_code/*.py")
    print()

    validate_axis_mapping()
    validate_online_offline_equivalence()
    validate_temporal_index_pairing()
    validate_canonical_se2_consistency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
