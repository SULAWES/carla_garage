#!/usr/bin/env python3
"""Measure the saved B2D LiDAR contract with the online diagnostic schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from config import GlobalConfig  # noqa: E402
from diffusiondrive.carla_native_dataset import (  # noqa: E402
    TARGET_MODE_FUTURE_EGO_TIME,
    TARGET_MODE_SPATIAL_PATH,
    Bench2DriveDiffusionDataset,
    lidar_to_histogram_features,
    load_lidar_points,
)
from diffusiondrive.lidar_diagnostics import (  # noqa: E402
    LidarDiagnosticWriter,
    compute_lidar_contract_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--route-glob", default="*")
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--skip-first-frames", type=int, default=0)
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_SPATIAL_PATH, TARGET_MODE_FUTURE_EGO_TIME),
        default=TARGET_MODE_SPATIAL_PATH,
    )
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--spatial-target-first-distance", type=float, default=2.5)
    parser.add_argument("--spatial-target-interval", type=float, default=1.0)
    parser.add_argument("--spatial-target-max-future-frames", type=int, default=120)
    parser.add_argument("--balanced-scenarios", action="store_true")
    parser.add_argument("--max-samples-per-scenario", type=int, default=None)
    parser.add_argument("--dump-every", type=int, default=0)
    parser.add_argument("--max-dumps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise RuntimeError("--max-records must be positive when provided.")
    if args.dump_every < 0 or args.max_dumps < 0:
        raise RuntimeError("--dump-every and --max-dumps must be non-negative.")

    config = GlobalConfig()
    dataset = Bench2DriveDiffusionDataset(
        args.root_dir,
        config=config,
        num_poses=args.num_poses,
        future_stride=args.future_stride,
        frame_sampling=args.frame_sampling,
        route_glob=args.route_glob,
        target_mode=args.target_mode,
        spatial_target_first_distance=args.spatial_target_first_distance,
        spatial_target_interval=args.spatial_target_interval,
        spatial_target_max_future_frames=args.spatial_target_max_future_frames,
        balanced_scenarios=args.balanced_scenarios,
        max_samples_per_scenario=args.max_samples_per_scenario,
        sample_manifest_path=args.sample_manifest,
        skip_first_frames=args.skip_first_frames,
    )
    total = len(dataset)
    if args.max_records is not None:
        total = min(total, args.max_records)
    print(f"LiDAR contract samples: {total}/{len(dataset)}", flush=True)
    print(f"Scenario samples: {dataset.scenario_sample_counts}", flush=True)

    writer = LidarDiagnosticWriter(args.output_dir, max_dumps=args.max_dumps)
    processed = 0
    try:
        for index, (route_dir, frame) in enumerate(dataset.samples[:total]):
            points = load_lidar_points(route_dir, int(frame))
            bev = lidar_to_histogram_features(points, config)
            record = compute_lidar_contract_record(
                points,
                bev,
                config,
                source="b2d_raw",
                stage="saved_full_scan",
                step=index,
                context={
                    "sample_index": index,
                    "scenario": route_dir.parent.name,
                    "route": route_dir.name,
                    "frame": int(frame),
                },
            )
            writer.record(record)
            processed += 1

            should_dump = args.dump_every > 0 and (index % args.dump_every) == 0
            if should_dump:
                writer.dump_bundle(
                    step=index,
                    event="periodic",
                    arrays={"lidar_points": points, "lidar_bev": bev},
                    metadata=record,
                )
            if args.log_every > 0 and (processed == 1 or processed % args.log_every == 0):
                print(
                    "[DiffusionDriveLidarContractOffline] "
                    f"processed={processed}/{total} "
                    f"points={record['point_count']} "
                    f"model_points={record['model_point_count']} "
                    f"bev_nonzero={record['bev_nonzero_ratio']:.6f} "
                    f"bev_saturation={record['bev_saturation_ratio']:.6f}",
                    flush=True,
                )
    finally:
        writer.close(context={
            "source": "b2d_raw",
            "processed_samples": processed,
            "requested_samples": total,
            "root_dir": [str(path) for path in args.root_dir],
            "route_glob": args.route_glob,
            "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
            "frame_sampling": args.frame_sampling,
            "skip_first_frames": args.skip_first_frames,
            "target_mode": args.target_mode,
        })

    print(f"Wrote LiDAR contract diagnostics: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
