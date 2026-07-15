#!/usr/bin/env python3
"""Build DiffusionDrive sample manifests without starting training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from config import GlobalConfig  # noqa: E402
from diffusiondrive.carla_native_dataset import (  # noqa: E402
    TARGET_MODE_FUTURE_EGO_TIME,
    TARGET_MODE_SPATIAL_PATH,
    Bench2DriveDiffusionDataset,
    build_route_condition_metadata,
    build_target_speed_metadata,
    build_trajectory_target,
    is_hard_left_turn_stop_sample_from_values,
    load_annotation,
    route_condition_feature_metadata,
    target_speed_label_metadata,
)
from diffusiondrive.spatial_target import spatial_target_metadata  # noqa: E402
from b2d_quality_filter import QUALITY_FILTERS, QUALITY_FILTER_NONE, quality_filter_decision  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--skip-first-frames", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--balanced-scenarios", action="store_true")
    parser.add_argument("--max-samples-per-scenario", type=int, default=None)
    parser.add_argument(
        "--quality-filter",
        choices=QUALITY_FILTERS,
        default=QUALITY_FILTER_NONE,
        help=(
            "Route-level quality filter before writing samples. "
            "soft_clean drops missing results / hard failed routes; "
            "syb_clean additionally keeps only score=100 routes or min-speed-only infractions."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_SPATIAL_PATH, TARGET_MODE_FUTURE_EGO_TIME),
        default=TARGET_MODE_SPATIAL_PATH,
    )
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--spatial-target-first-distance", type=float, default=2.5)
    parser.add_argument("--spatial-target-interval", type=float, default=1.0)
    parser.add_argument("--spatial-target-max-future-frames", type=int, default=120)
    parser.add_argument("--hard-left-turn-command", type=int, default=1)
    parser.add_argument("--hard-left-turn-speed-threshold", type=float, default=0.1)
    parser.add_argument("--hard-left-turn-y-threshold", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--verify-load", action="store_true")
    return parser.parse_args()


def make_header(args: argparse.Namespace, sample_count: int, scenario_counts: dict[str, int]) -> dict:
    return {
        "type": "metadata",
        "format": "diffusiondrive_sample_manifest_v1",
        "builder": "tools/build_diffusiondrive_manifest.py",
        "root_dir": [str(path) for path in args.root_dir],
        "route_glob": args.route_glob,
        "frame_sampling": args.frame_sampling,
        "skip_first_frames": args.skip_first_frames,
        "max_samples": args.max_samples,
        "balanced_scenarios": args.balanced_scenarios,
        "max_samples_per_scenario": args.max_samples_per_scenario,
        "quality_filter": args.quality_filter,
        "target_mode": args.target_mode,
        "num_poses": args.num_poses,
        "future_stride": args.future_stride,
        "spatial_target_first_distance": args.spatial_target_first_distance,
        "spatial_target_interval": args.spatial_target_interval,
        "spatial_target_max_future_frames": args.spatial_target_max_future_frames,
        "spatial_target": spatial_target_metadata(),
        "target_speed_label": target_speed_label_metadata(),
        "route_condition_feature": route_condition_feature_metadata(),
        "sample_count": sample_count,
        "scenario_counts": scenario_counts,
        "num_workers": args.num_workers,
        "chunksize": args.chunksize,
    }


def discover_samples(args: argparse.Namespace) -> Bench2DriveDiffusionDataset:
    dataset = Bench2DriveDiffusionDataset(
        args.root_dir,
        config=GlobalConfig(),
        num_poses=args.num_poses,
        future_stride=args.future_stride,
        frame_sampling=args.frame_sampling,
        skip_first_frames=args.skip_first_frames,
        max_samples=args.max_samples,
        route_glob=args.route_glob,
        target_mode=args.target_mode,
        spatial_target_first_distance=args.spatial_target_first_distance,
        spatial_target_interval=args.spatial_target_interval,
        spatial_target_max_future_frames=args.spatial_target_max_future_frames,
        balanced_scenarios=args.balanced_scenarios,
        max_samples_per_scenario=args.max_samples_per_scenario,
    )
    if args.quality_filter != QUALITY_FILTER_NONE:
        apply_quality_filter(dataset, args.quality_filter)
    return dataset


def apply_quality_filter(dataset: Bench2DriveDiffusionDataset, quality_filter: str) -> None:
    route_decisions = {}
    reason_counts: Counter[str] = Counter()
    kept_samples = []

    for route_dir, frame in dataset.samples:
        route_key = route_dir.resolve()
        decision = route_decisions.get(route_key)
        if decision is None:
            decision = quality_filter_decision(route_dir, quality_filter)
            route_decisions[route_key] = decision
            if not decision.keep:
                reason_counts[decision.reason] += 1
        if decision.keep:
            kept_samples.append((route_dir, frame))

    before_samples = len(dataset.samples)
    before_routes = len(route_decisions)
    kept_routes = sum(1 for decision in route_decisions.values() if decision.keep)
    dataset.samples = kept_samples
    dataset.scenario_sample_counts = recount_scenarios(kept_samples)

    print(
        f"Quality filter {quality_filter}: routes={kept_routes}/{before_routes} "
        f"samples={len(kept_samples)}/{before_samples} skipped={dict(reason_counts)}",
        flush=True,
    )


def recount_scenarios(samples: list[tuple[Path, int]]) -> dict[str, int]:
    scenario_counts: dict[str, int] = {}
    for route_dir, _frame in samples:
        scenario = route_dir.parent.name
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    return scenario_counts


def build_record(task: tuple[int, str, int, dict]) -> tuple[int, dict, bool]:
    index, route_dir_text, frame, params = task
    route_dir = Path(route_dir_text)
    annotation = load_annotation(route_dir, frame)
    trajectory = build_trajectory_target(
        route_dir,
        frame,
        params["num_poses"],
        params["future_stride"],
        target_mode=params["target_mode"],
        spatial_first_distance=params["spatial_target_first_distance"],
        spatial_interval=params["spatial_target_interval"],
        spatial_max_future_frames=params["spatial_target_max_future_frames"],
    )
    command = int(annotation.get("command_far", annotation.get("command_near", 4)))
    speed = float(annotation["speed"])
    hard_case = is_hard_left_turn_stop_sample_from_values(
        command,
        speed,
        trajectory,
        command=params["hard_left_turn_command"],
        speed_threshold=params["hard_left_turn_speed_threshold"],
        y_threshold=params["hard_left_turn_y_threshold"],
    )
    record = {
        "route_dir": str(route_dir.resolve()),
        "scenario": route_dir.parent.name,
        "route": route_dir.name,
        "frame": int(frame),
        "command": command,
        "speed": speed,
        "trajectory": trajectory.tolist(),
    }
    record.update(build_target_speed_metadata(annotation))
    record.update(build_route_condition_metadata(annotation))
    return index, record, hard_case


def iter_tasks(dataset: Bench2DriveDiffusionDataset, args: argparse.Namespace) -> Iterable[tuple[int, str, int, dict]]:
    params = {
        "target_mode": args.target_mode,
        "num_poses": args.num_poses,
        "future_stride": args.future_stride,
        "spatial_target_first_distance": args.spatial_target_first_distance,
        "spatial_target_interval": args.spatial_target_interval,
        "spatial_target_max_future_frames": args.spatial_target_max_future_frames,
        "hard_left_turn_command": args.hard_left_turn_command,
        "hard_left_turn_speed_threshold": args.hard_left_turn_speed_threshold,
        "hard_left_turn_y_threshold": args.hard_left_turn_y_threshold,
    }
    for index, (route_dir, frame) in enumerate(dataset.samples):
        yield index, str(route_dir), int(frame), params


def build_manifest(dataset: Bench2DriveDiffusionDataset, args: argparse.Namespace) -> None:
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    header = make_header(args, len(dataset), dataset.scenario_sample_counts)
    hard_case_count = 0

    with args.output_manifest.open("w", encoding="utf-8") as file:
        file.write(json.dumps(header, sort_keys=True) + "\n")
        tasks = iter_tasks(dataset, args)
        if args.num_workers > 0:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                chunksize = args.chunksize or max(1, len(dataset) // max(1, args.num_workers * 16))
                records = executor.map(build_record, tasks, chunksize=chunksize)
                hard_case_count = write_records(file, records, len(dataset), args.progress_every)
        else:
            records = (build_record(task) for task in tasks)
            hard_case_count = write_records(file, records, len(dataset), args.progress_every)

    print(
        f"Wrote sample manifest: {args.output_manifest} "
        f"({len(dataset)} samples, hard_left_turn_stop={hard_case_count})",
        flush=True,
    )


def write_records(file, records: Iterable[tuple[int, dict, bool]], total: int, progress_every: int) -> int:
    hard_case_count = 0
    for index, record, hard_case in records:
        file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        hard_case_count += int(hard_case)
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"Built manifest records: {index + 1}/{total}", flush=True)
    return hard_case_count


def count_manifest_samples(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "metadata":
                count += 1
    return count


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")

    if args.output_manifest.exists() and not args.rebuild:
        count = count_manifest_samples(args.output_manifest)
        print(f"Manifest already exists: {args.output_manifest} ({count} samples). Use --rebuild to overwrite.", flush=True)
        return

    dataset = discover_samples(args)
    print(f"Discovered samples: {len(dataset)}", flush=True)
    if dataset.scenario_sample_counts:
        counts = ", ".join(f"{name}={count}" for name, count in sorted(dataset.scenario_sample_counts.items()))
        print(f"Scenario samples: {counts}", flush=True)

    build_manifest(dataset, args)

    if args.verify_load:
        loaded = Bench2DriveDiffusionDataset(
            args.root_dir,
            config=GlobalConfig(),
            num_poses=args.num_poses,
            future_stride=args.future_stride,
            frame_sampling=args.frame_sampling,
            skip_first_frames=args.skip_first_frames,
            max_samples=args.max_samples,
            route_glob=args.route_glob,
            target_mode=args.target_mode,
            spatial_target_first_distance=args.spatial_target_first_distance,
            spatial_target_interval=args.spatial_target_interval,
            spatial_target_max_future_frames=args.spatial_target_max_future_frames,
            balanced_scenarios=args.balanced_scenarios,
            max_samples_per_scenario=args.max_samples_per_scenario,
            sample_manifest_path=args.output_manifest,
        )
        print(f"Verified manifest load: {len(loaded)} samples", flush=True)


if __name__ == "__main__":
    main()
