#!/usr/bin/env python3
"""Render offline LiDAR/BEV alignment diagnostics for CARLA Garage routes.

This script is designed for remote-server use. It reads an existing route
directory containing:

  route_dir/
    lidar/0000.laz
    measurements/0000.json.gz

and produces:
  - BEV overlays before/after alignment
  - occupancy IoU metrics
  - a JSON summary

The alignment math mirrors the current implementation in:
  - team_code/data.py: CARLA_Data.align()
  - team_code/diffusiondrive_agent.py: DiffusionDriveAgent.align_lidar()

Examples:
  python tools/render_lidar_bev_alignment.py \
    --route-dir data/.../SomeRoute \
    --frame 120 \
    --history 1 2 3 \
    --output-dir /tmp/lidar_bev_debug

  python tools/render_lidar_bev_alignment.py \
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
    args = parser.parse_args()
    if (args.route_dir is None) == (args.root_dir is None):
        parser.error("Specify exactly one of --route-dir or --root-dir.")
    if args.frames_per_route < 1:
        parser.error("--frames-per-route must be >= 1.")
    return args


def normalize_angle(x: float) -> float:
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def load_lidar(lidar_path: Path) -> np.ndarray:
    las = laspy.read(lidar_path)
    return np.asarray(las.xyz, dtype=np.float32)


def load_measurement(measurement_path: Path) -> dict:
    with gzip.open(measurement_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def align_lidar(lidar_0: np.ndarray, measurement_0: dict, measurement_1: dict) -> np.ndarray:
    """Mirror CARLA_Data.align() without augmentation."""
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
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"Missing lidar dir: {lidar_dir}")
    if not measurement_dir.is_dir():
        raise FileNotFoundError(f"Missing measurements dir: {measurement_dir}")

    lidar_ids = {int(path.stem) for path in lidar_dir.glob("*.laz")}
    measurement_ids = {int(path.name.split(".")[0]) for path in measurement_dir.glob("*.json.gz")}
    frame_ids = sorted(lidar_ids & measurement_ids)
    if not frame_ids:
        raise RuntimeError(f"No matching lidar/measurement frames found under {route_dir}")
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
    for candidate in sorted(root_dir.rglob("*")):
        if not candidate.is_dir():
            continue
        if (candidate / "lidar").is_dir() and (candidate / "measurements").is_dir():
            route_dirs.append(candidate)
    if not route_dirs:
        raise RuntimeError(f"No route directories found under {root_dir}")
    return route_dirs


def route_output_dir(output_dir: Path, route_dir: Path) -> Path:
    safe_route = "__".join(route_dir.parts[-4:])
    return output_dir / safe_route


def render_for_history(
    route_dir: Path,
    current_frame: int,
    history_offset: int,
    output_dir: Path,
    cfg: BevConfig,
    color: np.ndarray,
) -> dict[str, object]:
    current_lidar = load_lidar(route_dir / "lidar" / f"{current_frame:04d}.laz")
    current_measurement = load_measurement(route_dir / "measurements" / f"{current_frame:04d}.json.gz")

    history_frame = current_frame - history_offset
    history_lidar = load_lidar(route_dir / "lidar" / f"{history_frame:04d}.laz")
    history_measurement = load_measurement(route_dir / "measurements" / f"{history_frame:04d}.json.gz")

    aligned_lidar = align_lidar(history_lidar, history_measurement, current_measurement)

    current_bev = lidar_to_histogram_features(current_lidar, cfg)
    history_bev = lidar_to_histogram_features(history_lidar, cfg)
    aligned_bev = lidar_to_histogram_features(aligned_lidar, cfg)

    channel = 1 if cfg.use_ground_plane else 0
    current_channel = current_bev[channel]
    history_channel = history_bev[channel]
    aligned_channel = aligned_bev[channel]

    before_iou = occupancy_iou(current_channel, history_channel)
    after_iou = occupancy_iou(current_channel, aligned_channel)

    current_gray = np.repeat(to_uint8_grayscale(current_channel)[..., None], 3, axis=2)
    before_overlay = make_overlay(current_channel, history_channel, color)
    after_overlay = make_overlay(current_channel, aligned_channel, color)
    comparison = add_separator([current_gray, before_overlay, after_overlay])

    save_image(
        comparison,
        output_dir / f"frame_{current_frame:04d}" / f"history_{history_offset:02d}.png",
    )

    return {
        "current_frame": current_frame,
        "history_frame": history_frame,
        "history_offset": history_offset,
        "before_iou": before_iou,
        "after_iou": after_iou,
        "iou_gain": after_iou - before_iou,
    }


def process_route(
    route_dir: Path,
    output_dir: Path,
    cfg: BevConfig,
    requested_frame: int | None,
    history_offsets: list[int],
    frames_per_route: int,
) -> dict[str, object]:
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
            )
            frame_summary["results"].append(result)
            print(
                f"{route_dir.name} frame={current_frame:04d} history={history_offset:02d} "
                f"before_iou={result['before_iou']:.4f} "
                f"after_iou={result['after_iou']:.4f} "
                f"gain={result['iou_gain']:.4f}"
            )
        route_summary["frames"].append(frame_summary)

    return route_summary


def main() -> int:
    args = parse_args()
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

    if args.route_dir is not None:
        route_dirs = [args.route_dir]
    else:
        route_dirs = discover_route_dirs(args.root_dir)
        if args.max_routes is not None:
            route_dirs = route_dirs[:args.max_routes]

    summary = {
        "mode": "single_route" if args.route_dir is not None else "batch",
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
        },
        "routes": [],
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for route_dir in route_dirs:
        route_summary = process_route(
            route_dir=route_dir,
            output_dir=route_output_dir(output_dir, route_dir),
            cfg=cfg,
            requested_frame=args.frame,
            history_offsets=args.history,
            frames_per_route=args.frames_per_route,
        )
        summary["routes"].append(route_summary)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
