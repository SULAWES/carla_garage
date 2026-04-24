#!/usr/bin/env python3
"""Inspect progress for render_lidar_bev_alignment_v2.py outputs.

This script is read-only. It scans an existing output directory and reports:
  - how many route/frame/history artifacts already exist
  - optional expected totals and completion ratio
  - incomplete route/frame items that are still missing history images

It is intended for remote-server use while a batch job is still running.

Examples:
  python tools/check_lidar_bev_progress_v2.py \
    --output-dir /share/home/.../lidar_bev_batch

  python tools/check_lidar_bev_progress_v2.py \
    --output-dir /share/home/.../lidar_bev_batch \
    --root-dir /share/home/.../carla_dataset \
    --history 1 3 5 \
    --frames-per-route 3 \
    --max-routes 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameProgress:
    route_dir: Path
    frame_dir: Path
    history_count: int
    expected_history_count: int | None

    @property
    def is_complete(self) -> bool:
        return self.expected_history_count is not None and self.history_count >= self.expected_history_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory of render_lidar_bev_alignment_v2.py")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Optional dataset root used to infer expected route count in batch mode.",
    )
    parser.add_argument("--route-dir", type=Path, default=None, help="Optional single route dir used for exact expectations.")
    parser.add_argument(
        "--history",
        type=int,
        nargs="+",
        default=None,
        help="History offsets used by the render job. If omitted, only observed counts are shown.",
    )
    parser.add_argument(
        "--frames-per-route",
        type=int,
        default=1,
        help="Expected frames per route when the render job runs without a fixed --frame.",
    )
    parser.add_argument("--frame", type=int, default=None, help="If the render job used a fixed --frame, set it here.")
    parser.add_argument("--max-routes", type=int, default=None, help="Same meaning as in render_lidar_bev_alignment_v2.py")
    parser.add_argument(
        "--show-incomplete",
        type=int,
        default=20,
        help="Maximum number of incomplete frame entries to print.",
    )
    return parser.parse_args()


def route_output_name(route_dir: Path) -> str:
    return "__".join(route_dir.parts[-4:])


def discover_expected_route_dirs(root_dir: Path) -> list[Path]:
    routes = []
    for candidate in sorted(root_dir.rglob("*")):
        if candidate.is_dir() and (candidate / "lidar").is_dir() and (candidate / "measurements").is_dir():
            routes.append(candidate)
    return routes


def count_history_images(frame_dir: Path) -> int:
    count = 0
    for pattern in ("history_*.png", "history_*.ppm"):
        count += len(list(frame_dir.glob(pattern)))
    return count


def scan_output_dir(output_dir: Path, expected_history_count: int | None) -> tuple[list[Path], list[FrameProgress]]:
    route_dirs = []
    frame_progress = []

    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        route_dirs.append(child)
        for frame_dir in sorted(child.glob("frame_*")):
            if not frame_dir.is_dir():
                continue
            frame_progress.append(
                FrameProgress(
                    route_dir=child,
                    frame_dir=frame_dir,
                    history_count=count_history_images(frame_dir),
                    expected_history_count=expected_history_count,
                )
            )

    return route_dirs, frame_progress


def summarize_existing_summary(output_dir: Path) -> dict | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def print_line(label: str, value: str) -> None:
    print(f"{label:<24} {value}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_dir():
        raise SystemExit(f"Missing output dir: {output_dir}")

    expected_history_count = len(args.history) if args.history else None
    observed_route_dirs, frame_progress = scan_output_dir(output_dir, expected_history_count)
    existing_summary = summarize_existing_summary(output_dir)

    total_history_images = sum(item.history_count for item in frame_progress)
    frames_started = len(frame_progress)
    frames_complete = sum(1 for item in frame_progress if item.is_complete)

    expected_route_dirs = None
    if args.route_dir is not None:
        expected_route_dirs = [args.route_dir]
    elif args.root_dir is not None:
        expected_route_dirs = discover_expected_route_dirs(args.root_dir)
        if args.max_routes is not None:
            expected_route_dirs = expected_route_dirs[:args.max_routes]

    expected_route_count = len(expected_route_dirs) if expected_route_dirs is not None else None
    expected_frames_per_route = 1 if args.frame is not None else args.frames_per_route
    expected_frame_count = None
    expected_history_images = None
    if expected_route_count is not None and expected_history_count is not None:
        expected_frame_count = expected_route_count * expected_frames_per_route
        expected_history_images = expected_frame_count * expected_history_count

    print("LiDAR BEV Alignment Progress")
    print()
    print_line("Output Dir", str(output_dir))
    print_line("Observed Routes", str(len(observed_route_dirs)))
    print_line("Observed Frames", str(frames_started))
    print_line("Observed Images", str(total_history_images))

    if expected_route_count is not None:
        print_line("Expected Routes", str(expected_route_count))
    if expected_frame_count is not None:
        print_line("Expected Frames", str(expected_frame_count))
    if expected_history_images is not None:
        print_line("Expected Images", str(expected_history_images))
        ratio = 100.0 * total_history_images / max(expected_history_images, 1)
        print_line("Image Progress", f"{ratio:.1f}%")
        frame_ratio = 100.0 * frames_complete / max(expected_frame_count, 1)
        print_line("Frame Progress", f"{frames_complete}/{expected_frame_count} ({frame_ratio:.1f}%)")
    elif expected_history_count is not None:
        print_line("Frames Complete", str(frames_complete))

    if existing_summary is not None:
        print_line("Top Summary", "present")
        if "route_count" in existing_summary:
            print_line("Summary Route Count", str(existing_summary["route_count"]))
        if "mode" in existing_summary:
            print_line("Summary Mode", str(existing_summary["mode"]))
    else:
        print_line("Top Summary", "not written yet")

    if expected_route_dirs is not None:
        missing_routes = []
        observed_names = {route_dir.name for route_dir in observed_route_dirs}
        for route_dir in expected_route_dirs:
            safe_name = route_output_name(route_dir)
            if safe_name not in observed_names:
                missing_routes.append(safe_name)
        print_line("Routes Started", f"{expected_route_count - len(missing_routes)}/{expected_route_count}")
        if missing_routes:
            print()
            print("Routes with no outputs yet:")
            for route_name in missing_routes[: args.show_incomplete]:
                print(f"  {route_name}")

    incomplete = [item for item in frame_progress if not item.is_complete] if expected_history_count is not None else []
    if incomplete:
        print()
        print("Incomplete Frames:")
        for item in incomplete[: args.show_incomplete]:
            print(
                f"  {item.route_dir.name}/{item.frame_dir.name}: "
                f"{item.history_count}/{item.expected_history_count} history images"
            )

    if frame_progress:
        print()
        print("Most Recent Frame Dirs:")
        recent = sorted(frame_progress, key=lambda item: item.frame_dir.stat().st_mtime, reverse=True)
        for item in recent[: min(10, len(recent))]:
            print(
                f"  {item.route_dir.name}/{item.frame_dir.name}: "
                f"{item.history_count} images"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
