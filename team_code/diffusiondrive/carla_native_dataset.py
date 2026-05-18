"""CARLA-native DiffusionDrive dataset helpers."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

from config import GlobalConfig
import transfuser_utils as t_u


class Bench2DriveDiffusionDataset(Dataset):
    """Raw Bench2Drive route dataset for the CARLA DiffusionDrive input contract."""

    def __init__(
        self,
        root_dirs: Iterable[str | Path],
        config: GlobalConfig,
        num_poses: int,
        future_stride: int = 10,
        frame_sampling: int = 1,
        max_samples: Optional[int] = None,
        route_glob: str = "*",
    ) -> None:
        self.config = config
        self.num_poses = num_poses
        self.future_stride = future_stride
        self.frame_sampling = frame_sampling
        self.samples = self._discover_samples(root_dirs, route_glob, max_samples)
        if not self.samples:
            roots = ", ".join(str(root) for root in root_dirs)
            raise RuntimeError(f"No trainable Bench2Drive samples found under: {roots}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        route_dir, frame = self.samples[index]
        annotation = load_annotation(route_dir, frame)
        return {
            "features": {
                "camera_feature": build_camera_feature(route_dir, frame, self.config),
                "lidar_feature": build_lidar_feature(route_dir, frame, self.config),
                "status_feature": build_status_feature(annotation),
            },
            "targets": {
                "trajectory": build_trajectory_target(
                    route_dir,
                    frame,
                    self.num_poses,
                    self.future_stride,
                ),
            },
            "route": route_dir.name,
            "frame": frame,
        }

    def _discover_samples(
        self,
        root_dirs: Iterable[str | Path],
        route_glob: str,
        max_samples: Optional[int],
    ) -> List[tuple[Path, int]]:
        samples: List[tuple[Path, int]] = []
        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists():
                raise RuntimeError(f"Bench2Drive root does not exist: {root_path}")
            for route_dir in sorted(root_path.glob(route_glob)):
                if not _is_b2d_route(route_dir):
                    continue
                frames = sorted(int(path.stem.split(".")[0]) for path in (route_dir / "anno").glob("*.json.gz"))
                for frame in frames[:: self.frame_sampling]:
                    if self._has_required_files(route_dir, frame):
                        samples.append((route_dir, frame))
                        if max_samples is not None and len(samples) >= max_samples:
                            return samples
        return samples

    def _has_required_files(self, route_dir: Path, frame: int) -> bool:
        if not (route_dir / "camera" / "rgb_front" / f"{frame:05d}.jpg").is_file():
            return False
        if not (route_dir / "lidar" / f"{frame:05d}.laz").is_file():
            return False
        for offset in range(self.future_stride, self.future_stride * (self.num_poses + 1), self.future_stride):
            if not (route_dir / "anno" / f"{frame + offset:05d}.json.gz").is_file():
                return False
        return True


def _is_b2d_route(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "camera" / "rgb_front").is_dir()
        and (path / "lidar").is_dir()
        and (path / "anno").is_dir()
    )


def load_annotation(route_dir: Path, frame: int) -> dict:
    path = route_dir / "anno" / f"{frame:05d}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def build_camera_feature(route_dir: Path, frame: int, config: GlobalConfig) -> torch.Tensor:
    image_path = route_dir / "camera" / "rgb_front" / f"{frame:05d}.jpg"
    camera_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if camera_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Raw B2D images are commonly 900x1600. For the CARLA-native baseline,
    # map them into the online sensor size before applying the online crop.
    camera_bgr = cv2.resize(camera_bgr, (config.camera_width, config.camera_height), interpolation=cv2.INTER_LINEAR)
    _, encoded = cv2.imencode(".jpg", camera_bgr)
    camera_bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    camera_rgb = cv2.cvtColor(camera_bgr, cv2.COLOR_BGR2RGB)
    camera_rgb = t_u.crop_array(config, camera_rgb)
    camera_chw = np.transpose(camera_rgb, (2, 0, 1))
    camera_tensor = torch.from_numpy(camera_chw).float().unsqueeze(0) / 255.0
    camera_tensor = torch.nn.functional.interpolate(
        camera_tensor,
        size=(256, 1024),
        mode="bilinear",
        align_corners=False,
    )
    return camera_tensor.squeeze(0)


def build_lidar_feature(
    route_dir: Path,
    frame: int,
    config: GlobalConfig,
) -> torch.Tensor:
    lidar_path = route_dir / "lidar" / f"{frame:05d}.laz"
    lidar = laspy.read(str(lidar_path)).xyz
    lidar_hist = lidar_to_histogram_features(lidar, config)
    return torch.from_numpy(lidar_hist).float()


def lidar_to_histogram_features(lidar: np.ndarray, config: GlobalConfig) -> np.ndarray:
    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        xbins = np.linspace(
            config.min_x,
            config.max_x,
            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1,
        )
        ybins = np.linspace(
            config.min_y,
            config.max_y,
            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1,
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        return (hist / config.hist_max_per_pixel).T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    above_features = splat_points(above)
    if config.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    return np.transpose(features, (2, 0, 1)).astype(np.float32)


def build_status_feature(annotation: dict) -> torch.Tensor:
    command = int(annotation.get("command_far", annotation.get("command_near", 4)))
    command_one_hot = torch.from_numpy(t_u.command_to_one_hot(command)).float()
    speed = float(annotation["speed"])
    acceleration = annotation.get("acceleration", [0.0, 0.0, 0.0])
    accel_x = float(acceleration[0]) if acceleration else 0.0
    velocity_tensor = torch.tensor([speed, 0.0], dtype=torch.float32)
    accel_tensor = torch.tensor([accel_x, 0.0], dtype=torch.float32)
    return torch.cat([command_one_hot, velocity_tensor, accel_tensor], dim=0)


def build_trajectory_target(route_dir: Path, frame: int, num_poses: int, stride: int) -> torch.Tensor:
    origin = load_annotation(route_dir, frame)
    trajectory = []
    for offset in range(stride, stride * (num_poses + 1), stride):
        future = load_annotation(route_dir, frame + offset)
        trajectory.append(ego_relative_xy(origin, future))
    return torch.tensor(trajectory, dtype=torch.float32)


def ego_relative_xy(origin: dict, future: dict) -> list[float]:
    dx = float(future["x"]) - float(origin["x"])
    dy = float(future["y"]) - float(origin["y"])
    theta = float(origin["theta"])
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return [
        cos_theta * dx + sin_theta * dy,
        -sin_theta * dx + cos_theta * dy,
    ]
