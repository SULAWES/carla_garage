#!/usr/bin/env python3
"""Render offline LiDAR/BEV alignment diagnostics for CARLA Garage routes.

This script is designed for remote-server use. It reads an existing route
directory containing:

  route_dir/
    lidar/0000.laz
    measurements/0000.json.gz
    boxes/0000.json.gz

and produces:
  - BEV overlays before/after alignment
  - occupancy IoU metrics with dynamic actors masked out via boxes/*.json.gz
  - optional local dx/dy/yaw refinement around the measurement-based alignment
  - a JSON summary

The alignment math mirrors the current implementation in:
  - team_code/data.py: CARLA_Data.align()
  - team_code/diffusiondrive_agent.py: DiffusionDriveAgent.align_lidar()

Examples:
  python tools/render_lidar_bev_alignment_v4.py \
    --route-dir data/.../SomeRoute \
    --frame 120 \
    --history 1 2 3 \
    --output-dir /tmp/lidar_bev_debug

  python tools/render_lidar_bev_alignment_v4.py \
    --root-dir data/50x36_Town13 \
    --history 1 2 3 \
    --frames-per-route 2 \
    --max-routes 10 \
    --output-dir /tmp/lidar_bev_batch
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on target environment
    Image = None


@dataclass(frozen=True)
class BevConfig:
    min_x: float = -32.0
    max_x: float = 32.0
    min_y: float = -32.0
    max_y: float = 32.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5
    max_height_lidar: float = 100.0
    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    @property
    def width(self) -> int:
        return int((self.max_y - self.min_y) * self.pixels_per_meter)

    @property
    def height(self) -> int:
        return int((self.max_x - self.min_x) * self.pixels_per_meter)


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    xy_range: float = 0.8
    xy_step: float = 0.2
    yaw_range_deg: float = 5.0
    yaw_step_deg: float = 1.0
    fallback_if_worse: bool = True


COLOR_CYCLE = np.array(
    [
        [255, 80, 80],
        [80, 220, 120],
        [80, 160, 255],
        [255, 200, 80],
        [220, 80, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", type=Path, default=None, help="Path to one route directory.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Root directory to recursively scan for route directories containing lidar/ and measurements/.",
    )
    parser.add_argument(
        "--route-name",
        type=str,
        default=None,
        help="Optional route basename to locate under --root-dir, e.g. Town12_Rep0_2488_0_route0_11_08_02_55_17.",
    )
    parser.add_argument("--frame", type=int, default=None, help="Current frame id. Default: middle valid frame.")
    parser.add_argument(
        "--history",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="History offsets in saved-frame units, e.g. 1 2 3 means t-1/t-2/t-3.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs into.")
    parser.add_argument(
        "--frames-per-route",
        type=int,
        default=1,
        help="How many frames to render per route when --frame is not specified.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional cap on the number of discovered routes in batch mode.",
    )
    parser.add_argument("--min-x", type=float, default=-32.0)
    parser.add_argument("--max-x", type=float, default=32.0)
    parser.add_argument("--min-y", type=float, default=-32.0)
    parser.add_argument("--max-y", type=float, default=32.0)
    parser.add_argument("--pixels-per-meter", type=float, default=4.0)
    parser.add_argument("--hist-max-per-pixel", type=int, default=5)
    parser.add_argument("--max-height-lidar", type=float, default=100.0)
    parser.add_argument("--lidar-split-height", type=float, default=0.2)
    parser.add_argument(
        "--include-ground-plane",
        action="store_true",
        help="Render both below/above split channels instead of the default above-only channel.",
    )
    parser.add_argument(
        "--dynamic-classes",
        nargs="+",
        default=["car", "walker", "ego_car"],
        help="Bounding-box classes to mask out before computing occupancy IoU.",
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable local dx/dy/yaw refinement after the measurement-based alignment.",
    )
    parser.add_argument(
        "--refine-xy-range",
        type=float,
        default=0.8,
        help="Residual search range in meters for dx/dy refinement.",
    )
    parser.add_argument(
        "--refine-xy-step",
        type=float,
        default=0.2,
        help="Residual search step in meters for dx/dy refinement.",
    )
    parser.add_argument(
        "--refine-yaw-range-deg",
        type=float,
        default=5.0,
        help="Residual search range in degrees for yaw refinement.",
    )
    parser.add_argument(
        "--refine-yaw-step-deg",
        type=float,
        default=1.0,
        help="Residual search step in degrees for yaw refinement.",
    )
    parser.add_argument(
        "--allow-worse-refinement",
        action="store_true",
        help="Keep the best searched refinement even if it is worse than the initial alignment.",
    )
    args = parser.parse_args()
    if args.route_dir is None and args.root_dir is None:
        parser.error("Specify at least one of --route-dir or --root-dir.")
    if args.route_dir is not None and args.root_dir is not None and args.route_name is None:
        parser.error("Do not specify both --route-dir and --root-dir unless --route-name is used for lookup.")
    if args.route_name is not None and args.root_dir is None:
        parser.error("--route-name requires --root-dir.")
    if args.frames_per_route < 1:
        parser.error("--frames-per-route must be >= 1.")
    if args.refine_xy_range < 0 or args.refine_xy_step <= 0:
        parser.error("--refine-xy-range must be >= 0 and --refine-xy-step must be > 0.")
    if args.refine_yaw_range_deg < 0 or args.refine_yaw_step_deg <= 0:
        parser.error("--refine-yaw-range-deg must be >= 0 and --refine-yaw-step-deg must be > 0.")
    return args


def normalize_angle(x: float) -> float:
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def load_lidar(lidar_path: Path) -> np.ndarray:
    try:
        las = laspy.read(lidar_path)
    except Exception as exc:
        message = str(exc)
        if "No LazBackend selected" in message:
            raise RuntimeError(
                "Failed to read compressed .laz LiDAR because laspy has no LAZ backend installed.\n"
                "Install one of the following in the same Python environment and rerun:\n"
                "  python -m pip install lazrs\n"
                "or:\n"
                "  python -m pip install \"laspy[lazrs]\"\n"
                f"Problematic file: {lidar_path}"
            ) from exc
        raise
    return np.asarray(las.xyz, dtype=np.float32)


def load_measurement(measurement_path: Path) -> dict:
    with gzip.open(measurement_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_boxes(boxes_path: Path) -> list[dict]:
    with gzip.open(boxes_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def alignment_delta(measurement_0: dict, measurement_1: dict) -> tuple[np.ndarray, float]:
    pos_1 = np.array([measurement_1["pos_global"][0], measurement_1["pos_global"][1], 0.0], dtype=np.float32)
    pos_0 = np.array([measurement_0["pos_global"][0], measurement_0["pos_global"][1], 0.0], dtype=np.float32)
    pos_diff = pos_1 - pos_0
    rot_diff = normalize_angle(float(measurement_1["theta"]) - float(measurement_0["theta"]))

    rotation_matrix = np.array(
        [
            [np.cos(float(measurement_1["theta"])), -np.sin(float(measurement_1["theta"])), 0.0],
            [np.sin(float(measurement_1["theta"])), np.cos(float(measurement_1["theta"])), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pos_diff = rotation_matrix.T @ pos_diff
    return pos_diff, rot_diff


def align_lidar(lidar_0: np.ndarray, measurement_0: dict, measurement_1: dict) -> np.ndarray:
    """Mirror CARLA_Data.align() without augmentation."""
    pos_diff, rot_diff = alignment_delta(measurement_0, measurement_1)
    return algin_lidar(lidar_0, pos_diff, rot_diff)


def algin_lidar(lidar: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rotation_matrix.T @ (lidar - translation).T).T


def lidar_to_histogram_features(lidar: np.ndarray, cfg: BevConfig) -> np.ndarray:
    """Mirror CARLA_Data.lidar_to_histogram_features()."""

    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        xbins = np.linspace(cfg.min_x, cfg.max_x, int((cfg.max_x - cfg.min_x) * cfg.pixels_per_meter) + 1)
        ybins = np.linspace(cfg.min_y, cfg.max_y, int((cfg.max_y - cfg.min_y) * cfg.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > cfg.hist_max_per_pixel] = cfg.hist_max_per_pixel
        overhead_splat = hist / cfg.hist_max_per_pixel
        return overhead_splat.T

    lidar = lidar[lidar[..., 2] < cfg.max_height_lidar]
    below = lidar[lidar[..., 2] <= cfg.lidar_split_height]
    above = lidar[lidar[..., 2] > cfg.lidar_split_height]
    above_features = splat_points(above)
    if cfg.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    return np.transpose(features, (2, 0, 1)).astype(np.float32)


def to_uint8_grayscale(channel: np.ndarray) -> np.ndarray:
    return np.clip(channel * 255.0, 0, 255).astype(np.uint8)


def occupancy(channel: np.ndarray) -> np.ndarray:
    return channel > 0


def occupancy_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def box_to_mask(box: dict, shape: tuple[int, int], cfg: BevConfig) -> np.ndarray:
    center_x = float(box["position"][0])
    center_y = float(box["position"][1])
    extent_x = float(box["extent"][0])
    extent_y = float(box["extent"][1])
    yaw = float(box["yaw"])

    rotation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    corners = np.array(
        [
            [-extent_x, -extent_y],
            [extent_x, -extent_y],
            [extent_x, extent_y],
            [-extent_x, extent_y],
        ],
        dtype=np.float32,
    )
    world_corners = (rotation_matrix @ corners.T).T + np.array([center_x, center_y], dtype=np.float32)

    ppm = cfg.pixels_per_meter
    col_min = max(int(np.floor((world_corners[:, 0].min() - cfg.min_x) * ppm)), 0)
    col_max = min(int(np.ceil((world_corners[:, 0].max() - cfg.min_x) * ppm)), shape[1])
    row_min = max(int(np.floor((world_corners[:, 1].min() - cfg.min_y) * ppm)), 0)
    row_max = min(int(np.ceil((world_corners[:, 1].max() - cfg.min_y) * ppm)), shape[0])
    if row_min >= row_max or col_min >= col_max:
        return np.zeros(shape, dtype=bool)

    x_centers = cfg.min_x + (np.arange(col_min, col_max, dtype=np.float32) + 0.5) / ppm
    y_centers = cfg.min_y + (np.arange(row_min, row_max, dtype=np.float32) + 0.5) / ppm
    xx, yy = np.meshgrid(x_centers, y_centers)

    local_x = np.cos(yaw) * (xx - center_x) + np.sin(yaw) * (yy - center_y)
    local_y = -np.sin(yaw) * (xx - center_x) + np.cos(yaw) * (yy - center_y)
    local_mask = (np.abs(local_x) <= extent_x) & (np.abs(local_y) <= extent_y)

    mask = np.zeros(shape, dtype=bool)
    mask[row_min:row_max, col_min:col_max] = local_mask
    return mask


def dynamic_actor_boxes(boxes: list[dict], dynamic_classes: set[str]) -> list[dict]:
    filtered = []
    for box in boxes:
        if box.get("class") not in dynamic_classes:
            continue
        num_points = box.get("num_points")
        if num_points is not None and box.get("class") != "ego_car" and int(num_points) <= 0:
            continue
        filtered.append(box)
    return filtered


def align_boxes(boxes: list[dict], measurement_0: dict, measurement_1: dict) -> list[dict]:
    pos_diff, rot_diff = alignment_delta(measurement_0, measurement_1)
    return apply_transform_to_boxes(boxes, pos_diff, rot_diff)


def apply_transform_to_boxes(boxes: list[dict], translation: np.ndarray, yaw: float) -> list[dict]:
    aligned = []
    for box in boxes:
        center = np.array(
            [[float(box["position"][0]), float(box["position"][1]), float(box["position"][2])]],
            dtype=np.float32,
        )
        aligned_center = algin_lidar(center, translation, yaw)[0]
        aligned.append(
            {
                **box,
                "position": [float(aligned_center[0]), float(aligned_center[1]), float(aligned_center[2])],
                "yaw": normalize_angle(float(box["yaw"]) - yaw),
            }
        )
    return aligned


def dynamic_mask(boxes: list[dict], shape: tuple[int, int], cfg: BevConfig, dynamic_classes: set[str]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for box in dynamic_actor_boxes(boxes, dynamic_classes):
        mask |= box_to_mask(box, shape, cfg)
    return mask


def masked_occupancy_iou(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.shape != a.shape or mask.shape != b.shape:
        raise ValueError("Mask shape must match occupancy grids.")
    a_occ = occupancy(a) & ~mask
    b_occ = occupancy(b) & ~mask
    return occupancy_iou(a_occ, b_occ)


def refinement_values(value_range: float, step: float) -> np.ndarray:
    steps = int(np.floor((2.0 * value_range) / step)) + 1
    values = np.linspace(-value_range, value_range, num=max(steps, 1), dtype=np.float32)
    if 0.0 not in values:
        values = np.unique(np.concatenate([values, np.array([0.0], dtype=np.float32)]))
    return values


def refine_alignment(
    aligned_lidar: np.ndarray,
    aligned_boxes: list[dict],
    current_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    cfg: BevConfig,
    dynamic_classes: set[str],
    refinement_cfg: RefinementConfig,
) -> tuple[np.ndarray, list[dict], dict[str, object]]:
    def refinement_cost(translation: np.ndarray, yaw: float) -> float:
        return float(translation[0] ** 2 + translation[1] ** 2 + np.rad2deg(yaw) ** 2)

    initial_bev = lidar_to_histogram_features(aligned_lidar, cfg)
    channel = 1 if cfg.use_ground_plane else 0
    initial_channel = initial_bev[channel]
    initial_dynamic_mask = dynamic_mask(aligned_boxes, initial_channel.shape, cfg, dynamic_classes)
    initial_eval_mask = current_dynamic_mask | initial_dynamic_mask
    initial_iou = masked_occupancy_iou(current_channel, initial_channel, initial_eval_mask)

    best = {
        "iou": initial_iou,
        "translation": np.zeros(3, dtype=np.float32),
        "yaw": 0.0,
        "lidar": aligned_lidar,
        "boxes": aligned_boxes,
        "channel": initial_channel,
        "dynamic_mask": initial_dynamic_mask,
        "eval_mask": initial_eval_mask,
        "searched_candidates": 1,
        "used_fallback": False,
        "cost": 0.0,
    }

    if not refinement_cfg.enabled:
        return best["lidar"], best["boxes"], best

    dx_values = refinement_values(refinement_cfg.xy_range, refinement_cfg.xy_step)
    dy_values = refinement_values(refinement_cfg.xy_range, refinement_cfg.xy_step)
    yaw_values_deg = refinement_values(refinement_cfg.yaw_range_deg, refinement_cfg.yaw_step_deg)

    searched_candidates = 0
    for dx in dx_values:
        for dy in dy_values:
            translation = np.array([dx, dy, 0.0], dtype=np.float32)
            for yaw_deg in yaw_values_deg:
                searched_candidates += 1
                yaw = float(np.deg2rad(float(yaw_deg)))
                if dx == 0.0 and dy == 0.0 and yaw == 0.0:
                    continue

                candidate_lidar = algin_lidar(aligned_lidar, translation, yaw)
                candidate_boxes = apply_transform_to_boxes(aligned_boxes, translation, yaw)
                candidate_bev = lidar_to_histogram_features(candidate_lidar, cfg)
                candidate_channel = candidate_bev[channel]
                candidate_dynamic_mask = dynamic_mask(candidate_boxes, candidate_channel.shape, cfg, dynamic_classes)
                candidate_eval_mask = current_dynamic_mask | candidate_dynamic_mask
                candidate_iou = masked_occupancy_iou(current_channel, candidate_channel, candidate_eval_mask)
                candidate_cost = refinement_cost(translation, yaw)

                if candidate_iou > best["iou"] + 1e-9 or (
                    abs(candidate_iou - best["iou"]) <= 1e-9 and candidate_cost < best["cost"]
                ):
                    best = {
                        "iou": candidate_iou,
                        "translation": translation.copy(),
                        "yaw": yaw,
                        "lidar": candidate_lidar,
                        "boxes": candidate_boxes,
                        "channel": candidate_channel,
                        "dynamic_mask": candidate_dynamic_mask,
                        "eval_mask": candidate_eval_mask,
                        "searched_candidates": searched_candidates,
                        "used_fallback": False,
                        "cost": candidate_cost,
                    }

    best["searched_candidates"] = searched_candidates
    if refinement_cfg.fallback_if_worse and best["iou"] < initial_iou:
        best = {
            "iou": initial_iou,
            "translation": np.zeros(3, dtype=np.float32),
            "yaw": 0.0,
            "lidar": aligned_lidar,
            "boxes": aligned_boxes,
            "channel": initial_channel,
            "dynamic_mask": initial_dynamic_mask,
            "eval_mask": initial_eval_mask,
            "searched_candidates": searched_candidates,
            "used_fallback": True,
            "cost": 0.0,
        }

    return best["lidar"], best["boxes"], best


def make_overlay(base_channel: np.ndarray, compare_channel: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return RGB overlay where base is white and compare is colored."""
    base_occ = occupancy(base_channel)
    compare_occ = occupancy(compare_channel)
    image = np.zeros((base_channel.shape[0], base_channel.shape[1], 3), dtype=np.uint8)
    image[base_occ] = np.maximum(image[base_occ], np.array([255, 255, 255], dtype=np.uint8))
    image[compare_occ] = np.maximum(image[compare_occ], color)
    overlap = np.logical_and(base_occ, compare_occ)
    image[overlap] = np.array([255, 255, 0], dtype=np.uint8)
    return image


def add_separator(images: list[np.ndarray], width: int = 8) -> np.ndarray:
    separator = np.full((images[0].shape[0], width, 3), 30, dtype=np.uint8)
    stitched = []
    for index, image in enumerate(images):
        stitched.append(image)
        if index != len(images) - 1:
            stitched.append(separator)
    return np.concatenate(stitched, axis=1)


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(image).save(path)
        return

    # Pillow missing: write a binary PPM so the output is still viewable.
    ppm_path = path.with_suffix(".ppm")
    with ppm_path.open("wb") as handle:
        handle.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def collect_frame_ids(route_dir: Path) -> list[int]:
    lidar_dir = route_dir / "lidar"
    measurement_dir = route_dir / "measurements"
    boxes_dir = route_dir / "boxes"
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"Missing lidar dir: {lidar_dir}")
    if not measurement_dir.is_dir():
        raise FileNotFoundError(f"Missing measurements dir: {measurement_dir}")
    if not boxes_dir.is_dir():
        raise FileNotFoundError(f"Missing boxes dir: {boxes_dir}")

    lidar_ids = {int(path.stem) for path in lidar_dir.glob("*.laz")}
    measurement_ids = {int(path.name.split(".")[0]) for path in measurement_dir.glob("*.json.gz")}
    box_ids = {int(path.name.split(".")[0]) for path in boxes_dir.glob("*.json.gz")}
    frame_ids = sorted(lidar_ids & measurement_ids & box_ids)
    if not frame_ids:
        raise RuntimeError(f"No matching lidar/measurement/boxes frames found under {route_dir}")
    return frame_ids


def choose_frame(frame_ids: list[int], requested: int | None, max_history: int) -> int:
    if requested is not None:
        if requested not in frame_ids:
            raise ValueError(f"Requested frame {requested:04d} not found in route.")
        return requested

    usable = [frame_id for frame_id in frame_ids if frame_id - max_history in frame_ids]
    if not usable:
        raise RuntimeError("No frame has enough history for the requested offsets.")
    return usable[len(usable) // 2]


def choose_frames(frame_ids: list[int], requested: int | None, max_history: int, frames_per_route: int) -> list[int]:
    if requested is not None:
        return [choose_frame(frame_ids, requested, max_history)]

    usable = [frame_id for frame_id in frame_ids if frame_id - max_history in frame_ids]
    if not usable:
        raise RuntimeError("No frame has enough history for the requested offsets.")

    if frames_per_route >= len(usable):
        return usable

    indices = np.linspace(0, len(usable) - 1, num=frames_per_route, dtype=int)
    selected = [usable[index] for index in indices]
    # Keep order stable and remove duplicates from rounding.
    return list(dict.fromkeys(selected))


def discover_route_dirs(root_dir: Path) -> list[Path]:
    route_dirs = []
    scanned_dirs = 0
    print(f"Scanning route dirs under {root_dir} ...", flush=True)
    for current_root, dirnames, _filenames in os.walk(root_dir):
        scanned_dirs += 1
        current_path = Path(current_root)
        if "lidar" in dirnames and "measurements" in dirnames:
            route_dirs.append(current_path)
            # No need to walk inside a discovered route dir.
            dirnames[:] = [name for name in dirnames if name not in {"lidar", "measurements"}]
        if scanned_dirs % 500 == 0:
            print(
                f"  scanned_dirs={scanned_dirs} discovered_routes={len(route_dirs)} current={current_path}",
                flush=True,
            )
    route_dirs = sorted(route_dirs)
    print(f"Finished scanning. discovered_routes={len(route_dirs)}", flush=True)
    if not route_dirs:
        raise RuntimeError(f"No route directories found under {root_dir}")
    return route_dirs


def route_output_dir(output_dir: Path, route_dir: Path) -> Path:
    safe_route = "__".join(route_dir.parts[-4:])
    return output_dir / safe_route


def find_route_dir_by_name(root_dir: Path, route_name: str) -> Path:
    route_dirs = discover_route_dirs(root_dir)
    matches = [route_dir for route_dir in route_dirs if route_dir.name == route_name]
    if not matches:
        raise RuntimeError(f"Could not find route named {route_name} under {root_dir}")
    if len(matches) > 1:
        joined = "\n".join(str(path) for path in matches[:20])
        raise RuntimeError(
            f"Found multiple routes named {route_name} under {root_dir}. "
            f"Use --route-dir directly.\nMatches:\n{joined}"
        )
    return matches[0]


def render_for_history(
    route_dir: Path,
    current_frame: int,
    history_offset: int,
    output_dir: Path,
    cfg: BevConfig,
    color: np.ndarray,
    dynamic_classes: set[str],
    refinement_cfg: RefinementConfig,
) -> dict[str, object]:
    current_lidar = load_lidar(route_dir / "lidar" / f"{current_frame:04d}.laz")
    current_measurement = load_measurement(route_dir / "measurements" / f"{current_frame:04d}.json.gz")
    current_boxes = load_boxes(route_dir / "boxes" / f"{current_frame:04d}.json.gz")

    history_frame = current_frame - history_offset
    history_lidar = load_lidar(route_dir / "lidar" / f"{history_frame:04d}.laz")
    history_measurement = load_measurement(route_dir / "measurements" / f"{history_frame:04d}.json.gz")
    history_boxes = load_boxes(route_dir / "boxes" / f"{history_frame:04d}.json.gz")

    aligned_lidar = align_lidar(history_lidar, history_measurement, current_measurement)
    aligned_history_boxes = align_boxes(history_boxes, history_measurement, current_measurement)

    current_bev = lidar_to_histogram_features(current_lidar, cfg)
    history_bev = lidar_to_histogram_features(history_lidar, cfg)
    aligned_bev = lidar_to_histogram_features(aligned_lidar, cfg)

    channel = 1 if cfg.use_ground_plane else 0
    current_channel = current_bev[channel]
    history_channel = history_bev[channel]
    aligned_channel = aligned_bev[channel]

    current_dynamic_mask = dynamic_mask(current_boxes, current_channel.shape, cfg, dynamic_classes)
    history_dynamic_mask = dynamic_mask(history_boxes, history_channel.shape, cfg, dynamic_classes)
    aligned_history_dynamic_mask = dynamic_mask(aligned_history_boxes, aligned_channel.shape, cfg, dynamic_classes)

    before_eval_mask = current_dynamic_mask | history_dynamic_mask
    initial_after_eval_mask = current_dynamic_mask | aligned_history_dynamic_mask

    before_iou_raw = occupancy_iou(current_channel, history_channel)
    initial_after_iou_raw = occupancy_iou(current_channel, aligned_channel)
    before_iou = masked_occupancy_iou(current_channel, history_channel, before_eval_mask)
    initial_after_iou = masked_occupancy_iou(current_channel, aligned_channel, initial_after_eval_mask)

    refined_lidar, refined_history_boxes, refinement = refine_alignment(
        aligned_lidar=aligned_lidar,
        aligned_boxes=aligned_history_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        refinement_cfg=refinement_cfg,
    )
    refined_bev = lidar_to_histogram_features(refined_lidar, cfg)
    refined_channel = refined_bev[channel]
    refined_history_dynamic_mask = refinement["dynamic_mask"]
    refined_eval_mask = refinement["eval_mask"]
    after_iou_raw = occupancy_iou(current_channel, refined_channel)
    after_iou = refinement["iou"]
    visualization_mask = before_eval_mask | refined_eval_mask

    current_channel_vis = np.where(visualization_mask, 0.0, current_channel)
    current_before_vis = np.where(before_eval_mask, 0.0, current_channel)
    current_after_vis = np.where(refined_eval_mask, 0.0, current_channel)
    history_channel_vis = np.where(before_eval_mask, 0.0, history_channel)
    aligned_channel_vis = np.where(refined_eval_mask, 0.0, refined_channel)

    current_gray = np.repeat(to_uint8_grayscale(current_channel_vis)[..., None], 3, axis=2)
    before_overlay = make_overlay(current_before_vis, history_channel_vis, color)
    after_overlay = make_overlay(current_after_vis, aligned_channel_vis, color)
    comparison = add_separator([current_gray, before_overlay, after_overlay])

    save_image(
        comparison,
        output_dir / f"frame_{current_frame:04d}" / f"history_{history_offset:02d}.png",
    )

    return {
        "current_frame": current_frame,
        "history_frame": history_frame,
        "history_offset": history_offset,
        "before_iou_raw": before_iou_raw,
        "initial_after_iou_raw": initial_after_iou_raw,
        "after_iou_raw": after_iou_raw,
        "initial_iou_gain_raw": initial_after_iou_raw - before_iou_raw,
        "iou_gain_raw": after_iou_raw - before_iou_raw,
        "before_iou": before_iou,
        "initial_after_iou": initial_after_iou,
        "after_iou": after_iou,
        "initial_iou_gain": initial_after_iou - before_iou,
        "iou_gain": after_iou - before_iou,
        "current_dynamic_pixels": int(current_dynamic_mask.sum()),
        "history_dynamic_pixels": int(history_dynamic_mask.sum()),
        "aligned_history_dynamic_pixels": int(aligned_history_dynamic_mask.sum()),
        "refined_history_dynamic_pixels": int(refined_history_dynamic_mask.sum()),
        "before_masked_pixels": int(before_eval_mask.sum()),
        "initial_after_masked_pixels": int(initial_after_eval_mask.sum()),
        "after_masked_pixels": int(refined_eval_mask.sum()),
        "refinement_enabled": refinement_cfg.enabled,
        "refinement_used_fallback": bool(refinement["used_fallback"]),
        "refinement_dx": float(refinement["translation"][0]),
        "refinement_dy": float(refinement["translation"][1]),
        "refinement_yaw_deg": float(np.rad2deg(refinement["yaw"])),
        "refinement_candidates": int(refinement["searched_candidates"]),
    }


def process_route(
    route_dir: Path,
    output_dir: Path,
    cfg: BevConfig,
    requested_frame: int | None,
    history_offsets: list[int],
    frames_per_route: int,
    dynamic_classes: set[str],
    refinement_cfg: RefinementConfig,
) -> dict[str, object]:
    print(f"Processing route: {route_dir}", flush=True)
    frame_ids = collect_frame_ids(route_dir)
    max_history = max(history_offsets)
    frames = choose_frames(frame_ids, requested_frame, max_history, frames_per_route)

    route_summary = {
        "route_dir": str(route_dir),
        "frames": [],
    }

    for current_frame in frames:
        frame_summary = {
            "current_frame": current_frame,
            "results": [],
        }
        for index, history_offset in enumerate(history_offsets):
            result = render_for_history(
                route_dir=route_dir,
                current_frame=current_frame,
                history_offset=history_offset,
                output_dir=output_dir,
                cfg=cfg,
                color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
                dynamic_classes=dynamic_classes,
                refinement_cfg=refinement_cfg,
            )
            frame_summary["results"].append(result)
            print(
                f"{route_dir.name} frame={current_frame:04d} history={history_offset:02d} "
                f"before_iou={result['before_iou']:.4f} "
                f"initial_after_iou={result['initial_after_iou']:.4f} "
                f"after_iou={result['after_iou']:.4f} "
                f"gain={result['iou_gain']:.4f} "
                f"initial_gain={result['initial_iou_gain']:.4f} "
                f"raw_gain={result['iou_gain_raw']:.4f} "
                f"refine=({result['refinement_dx']:.2f},{result['refinement_dy']:.2f},{result['refinement_yaw_deg']:.2f}deg) "
                f"masked_pixels={result['after_masked_pixels']}",
                flush=True,
            )
        route_summary["frames"].append(frame_summary)

    return route_summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}", flush=True)
    if args.route_dir is not None and args.route_name is None:
        print(f"Mode: single route ({args.route_dir})", flush=True)
    elif args.route_name is not None:
        print(f"Mode: route lookup root={args.root_dir} route_name={args.route_name}", flush=True)
    else:
        print(f"Mode: batch root scan ({args.root_dir})", flush=True)
    print(
        f"history={args.history} frames_per_route={args.frames_per_route} "
        f"frame={args.frame} max_routes={args.max_routes}",
        flush=True,
    )

    cfg = BevConfig(
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        pixels_per_meter=args.pixels_per_meter,
        hist_max_per_pixel=args.hist_max_per_pixel,
        max_height_lidar=args.max_height_lidar,
        lidar_split_height=args.lidar_split_height,
        use_ground_plane=args.include_ground_plane,
    )
    dynamic_classes = set(args.dynamic_classes)
    refinement_cfg = RefinementConfig(
        enabled=not args.disable_refinement,
        xy_range=args.refine_xy_range,
        xy_step=args.refine_xy_step,
        yaw_range_deg=args.refine_yaw_range_deg,
        yaw_step_deg=args.refine_yaw_step_deg,
        fallback_if_worse=not args.allow_worse_refinement,
    )

    if args.route_name is not None:
        resolved_route_dir = find_route_dir_by_name(args.root_dir, args.route_name)
        print(f"Resolved route_dir: {resolved_route_dir}", flush=True)
        route_dirs = [resolved_route_dir]
    elif args.route_dir is not None:
        route_dirs = [args.route_dir]
    else:
        route_dirs = discover_route_dirs(args.root_dir)
        if args.max_routes is not None:
            route_dirs = route_dirs[:args.max_routes]

    summary = {
        "mode": "single_route" if (args.route_dir is not None or args.route_name is not None) else "batch",
        "route_count": len(route_dirs),
        "history_offsets": args.history,
        "config": {
            "min_x": cfg.min_x,
            "max_x": cfg.max_x,
            "min_y": cfg.min_y,
            "max_y": cfg.max_y,
            "pixels_per_meter": cfg.pixels_per_meter,
            "hist_max_per_pixel": cfg.hist_max_per_pixel,
            "max_height_lidar": cfg.max_height_lidar,
            "lidar_split_height": cfg.lidar_split_height,
            "use_ground_plane": cfg.use_ground_plane,
            "dynamic_classes": sorted(dynamic_classes),
            "refinement_enabled": refinement_cfg.enabled,
            "refine_xy_range": refinement_cfg.xy_range,
            "refine_xy_step": refinement_cfg.xy_step,
            "refine_yaw_range_deg": refinement_cfg.yaw_range_deg,
            "refine_yaw_step_deg": refinement_cfg.yaw_step_deg,
            "refinement_fallback_if_worse": refinement_cfg.fallback_if_worse,
        },
        "routes": [],
    }

    for route_dir in route_dirs:
        route_summary = process_route(
            route_dir=route_dir,
            output_dir=route_output_dir(output_dir, route_dir),
            cfg=cfg,
            requested_frame=args.frame,
            history_offsets=args.history,
            frames_per_route=args.frames_per_route,
            dynamic_classes=dynamic_classes,
            refinement_cfg=refinement_cfg,
        )
        summary["routes"].append(route_summary)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
