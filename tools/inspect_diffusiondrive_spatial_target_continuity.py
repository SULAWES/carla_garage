#!/usr/bin/env python3
"""Inspect cached and recomputed spatial target continuity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from diffusiondrive.carla_native_dataset import (  # noqa: E402
    TARGET_MODE_SPATIAL_PATH,
    build_spatial_path_target,
)
from diffusiondrive.spatial_target import spatial_target_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--max-frame-gap", type=int, default=10)
    parser.add_argument("--jump-thresholds", type=float, nargs="+", default=(5.0, 12.0))
    parser.add_argument("--event-threshold", type=float, default=12.0)
    parser.add_argument("--low-speed-threshold", type=float, default=0.1)
    parser.add_argument("--stable-condition-threshold", type=float, default=0.25)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--recompute-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def load_manifest(
    path: Path,
    scenarios: set[str],
    routes: set[str],
) -> tuple[dict, list[dict], int]:
    header = None
    records = []
    total_records = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "metadata":
                header = record
                continue
            total_records += 1
            if scenarios and str(record.get("scenario")) not in scenarios:
                continue
            if routes and str(record.get("route")) not in routes:
                continue
            records.append(record)
    if header is None:
        raise RuntimeError(f"Manifest is missing metadata header: {path}")
    if header.get("target_mode") != TARGET_MODE_SPATIAL_PATH:
        raise RuntimeError(
            f"Expected target_mode={TARGET_MODE_SPATIAL_PATH!r}, got {header.get('target_mode')!r}"
        )
    return header, records, total_records


def build_transitions(records: Iterable[dict], max_frame_gap: int) -> list[dict]:
    route_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        route_records[str(record["route_dir"])].append(record)

    transitions = []
    for route_dir, samples in route_records.items():
        samples.sort(key=lambda record: int(record["frame"]))
        for before, after in zip(samples, samples[1:]):
            frame_gap = int(after["frame"]) - int(before["frame"])
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                continue
            before_target = trajectory_array(before)
            after_target = trajectory_array(after)
            before_condition = condition_array(before)
            after_condition = condition_array(after)
            transitions.append(
                {
                    "scenario": str(after.get("scenario", Path(route_dir).parent.name)),
                    "route": str(after.get("route", Path(route_dir).name)),
                    "route_dir": route_dir,
                    "before_frame": int(before["frame"]),
                    "after_frame": int(after["frame"]),
                    "frame_gap": frame_gap,
                    "before_speed": float(before["speed"]),
                    "after_speed": float(after["speed"]),
                    "before_command": int(before["command"]),
                    "after_command": int(after["command"]),
                    "before_endpoint_x": float(before_target[-1, 0]),
                    "before_endpoint_y": float(before_target[-1, 1]),
                    "after_endpoint_x": float(after_target[-1, 0]),
                    "after_endpoint_y": float(after_target[-1, 1]),
                    "manifest_endpoint_jump_meters": float(
                        np.linalg.norm(after_target[-1] - before_target[-1])
                    ),
                    "manifest_target_max_abs_delta_meters": float(
                        np.max(np.abs(after_target - before_target))
                    ),
                    "route_condition_delta_meters": float(
                        np.linalg.norm(after_condition - before_condition)
                    ),
                }
            )
    return transitions


def trajectory_array(record: dict) -> np.ndarray:
    trajectory = np.asarray(record["trajectory"], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
        raise RuntimeError(
            f"Invalid trajectory: route={record.get('route')} frame={record.get('frame')} "
            f"shape={trajectory.shape}"
        )
    return trajectory[:, :2]


def condition_array(record: dict) -> np.ndarray:
    values = record.get("route_condition_feature")
    if values is None:
        values = list(record.get("target_point", (0.0, 0.0))) + list(
            record.get("target_point_next", (0.0, 0.0))
        )
    condition = np.asarray(values, dtype=np.float64)
    if condition.shape != (4,):
        raise RuntimeError(
            f"Invalid route condition: route={record.get('route')} frame={record.get('frame')} "
            f"shape={condition.shape}"
        )
    return condition


def compare_manifest_records(records: list[dict], reference_records: list[dict]) -> dict:
    keys = [sample_key(record) for record in records]
    reference_keys = [sample_key(record) for record in reference_records]
    key_counts = Counter(keys)
    reference_key_counts = Counter(reference_keys)
    records_by_key = {sample_key(record): record for record in records}
    reference_by_key = {sample_key(record): record for record in reference_records}
    key_set = set(keys)
    reference_key_set = set(reference_keys)
    common_keys = sorted(key_set & reference_key_set)

    invariant_mismatch_counts = Counter()
    for key in common_keys:
        record = records_by_key[key]
        reference = reference_by_key[key]
        if int(record["command"]) != int(reference["command"]):
            invariant_mismatch_counts["command"] += 1
        if not np.isclose(float(record["speed"]), float(reference["speed"])):
            invariant_mismatch_counts["speed"] += 1
        if not np.allclose(condition_array(record), condition_array(reference)):
            invariant_mismatch_counts["route_condition_feature"] += 1
        if not np.isclose(
            float(record.get("target_speed", record["speed"])),
            float(reference.get("target_speed", reference["speed"])),
        ):
            invariant_mismatch_counts["target_speed"] += 1
        if bool(record.get("brake", False)) != bool(reference.get("brake", False)):
            invariant_mismatch_counts["brake"] += 1
        if int(record.get("target_speed_class", -1)) != int(
            reference.get("target_speed_class", -1)
        ):
            invariant_mismatch_counts["target_speed_class"] += 1
        if int(record.get("target_speed_label_valid", 0)) != int(
            reference.get("target_speed_label_valid", 0)
        ):
            invariant_mismatch_counts["target_speed_label_valid"] += 1
        target_speed_twohot = np.asarray(
            record.get("target_speed_twohot", ()), dtype=np.float64
        )
        reference_target_speed_twohot = np.asarray(
            reference.get("target_speed_twohot", ()), dtype=np.float64
        )
        if (
            target_speed_twohot.shape != reference_target_speed_twohot.shape
            or not np.allclose(target_speed_twohot, reference_target_speed_twohot)
        ):
            invariant_mismatch_counts["target_speed_twohot"] += 1

    missing_keys = sorted(reference_key_set - key_set)
    extra_keys = sorted(key_set - reference_key_set)
    return {
        "record_count": len(records),
        "reference_record_count": len(reference_records),
        "duplicate_key_count": sum(count - 1 for count in key_counts.values() if count > 1),
        "reference_duplicate_key_count": sum(
            count - 1 for count in reference_key_counts.values() if count > 1
        ),
        "same_key_set": key_set == reference_key_set,
        "same_key_order": keys == reference_keys,
        "common_key_count": len(common_keys),
        "missing_key_count": len(missing_keys),
        "extra_key_count": len(extra_keys),
        "missing_key_examples": [list(key) for key in missing_keys[:10]],
        "extra_key_examples": [list(key) for key in extra_keys[:10]],
        "invariant_mismatch_counts": dict(sorted(invariant_mismatch_counts.items())),
    }


def sample_key(record: dict) -> tuple[str, str, int]:
    return (
        str(record.get("scenario", Path(record["route_dir"]).parent.name)),
        str(record.get("route", Path(record["route_dir"]).name)),
        int(record["frame"]),
    )


def recompute_sample(task: tuple[str, int, int, float, float, int]) -> tuple[str, int, dict]:
    route_dir_text, frame, num_poses, first_distance, interval, max_future_frames = task
    diagnostics = {}
    build_spatial_path_target(
        Path(route_dir_text),
        frame,
        num_poses,
        first_distance=first_distance,
        interval=interval,
        max_future_frames=max_future_frames,
        diagnostics=diagnostics,
    )
    return route_dir_text, frame, diagnostics


def recompute_event_samples(
    events: list[dict],
    header: dict,
    num_workers: int,
    progress_every: int,
) -> dict[tuple[str, int], dict]:
    sample_keys = []
    seen = set()
    for event in events:
        for frame_key in ("before_frame", "after_frame"):
            key = (event["route_dir"], int(event[frame_key]))
            if key not in seen:
                seen.add(key)
                sample_keys.append(key)

    params = (
        int(header["num_poses"]),
        float(header["spatial_target_first_distance"]),
        float(header["spatial_target_interval"]),
        int(header["spatial_target_max_future_frames"]),
    )
    tasks = [(*key, *params) for key in sample_keys]
    if num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(recompute_sample, tasks, chunksize=1)
            return collect_recomputed(results, len(tasks), progress_every)
    return collect_recomputed(
        (recompute_sample(task) for task in tasks),
        len(tasks),
        progress_every,
    )


def collect_recomputed(
    results: Iterable[tuple[str, int, dict]],
    total: int,
    progress_every: int,
) -> dict[tuple[str, int], dict]:
    recomputed = {}
    for index, (route_dir, frame, diagnostics) in enumerate(results, start=1):
        recomputed[(route_dir, frame)] = diagnostics
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"Recomputed spatial targets: {index}/{total}", flush=True)
    return recomputed


def build_event_outputs(
    events: list[dict],
    record_lookup: dict[tuple[str, int], dict],
    recomputed: dict[tuple[str, int], dict],
    event_threshold: float,
    recompute_tolerance: float,
) -> tuple[list[dict], list[dict]]:
    rows = []
    traces = []
    for event in events:
        before_key = (event["route_dir"], int(event["before_frame"]))
        after_key = (event["route_dir"], int(event["after_frame"]))
        before_record = record_lookup[before_key]
        after_record = record_lookup[after_key]
        before_diagnostics = recomputed[before_key]
        after_diagnostics = recomputed[after_key]
        before_manifest = trajectory_array(before_record)
        after_manifest = trajectory_array(after_record)
        before_current = np.asarray(before_diagnostics["target"], dtype=np.float64)
        after_current = np.asarray(after_diagnostics["target"], dtype=np.float64)
        before_cache_delta = float(np.max(np.abs(before_current - before_manifest)))
        after_cache_delta = float(np.max(np.abs(after_current - after_manifest)))
        current_jump = float(np.linalg.norm(after_current[-1] - before_current[-1]))
        changed = max(before_cache_delta, after_cache_delta) > recompute_tolerance
        if changed and current_jump <= event_threshold:
            outcome = "resolved_by_current_contract"
        elif changed:
            outcome = "changed_but_jump_persists"
        else:
            outcome = "unchanged"

        before_resampling = before_diagnostics["resampling"]
        after_resampling = after_diagnostics["resampling"]
        row = dict(event)
        row.update(
            {
                "current_endpoint_jump_meters": current_jump,
                "before_manifest_current_max_abs_meters": before_cache_delta,
                "after_manifest_current_max_abs_meters": after_cache_delta,
                "recompute_outcome": outcome,
                "before_direction_source": before_resampling[
                    "extrapolation_direction_source"
                ],
                "after_direction_source": after_resampling[
                    "extrapolation_direction_source"
                ],
                "before_direction_baseline_meters": before_resampling[
                    "extrapolation_direction_baseline_meters"
                ],
                "after_direction_baseline_meters": after_resampling[
                    "extrapolation_direction_baseline_meters"
                ],
                "before_observed_path_length_meters": before_resampling[
                    "observed_path_length_meters"
                ],
                "after_observed_path_length_meters": after_resampling[
                    "observed_path_length_meters"
                ],
                "before_coordinate_sources": ",".join(
                    sorted(set(before_diagnostics["coordinate_sources"]))
                ),
                "after_coordinate_sources": ",".join(
                    sorted(set(after_diagnostics["coordinate_sources"]))
                ),
            }
        )
        rows.append(row)
        traces.append(
            {
                "transition": event,
                "comparison": row,
                "before": before_diagnostics,
                "after": after_diagnostics,
            }
        )
    return rows, traces


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transition_summary(
    transitions: list[dict],
    thresholds: list[float],
    low_speed_threshold: float,
    stable_condition_threshold: float,
) -> dict:
    summary = {}
    for threshold in thresholds:
        matched = [
            row for row in transitions if row["manifest_endpoint_jump_meters"] > threshold
        ]
        low_speed = [
            row
            for row in matched
            if row["before_speed"] <= low_speed_threshold
            and row["after_speed"] <= low_speed_threshold
        ]
        low_speed_stable_condition = [
            row
            for row in low_speed
            if row["route_condition_delta_meters"] <= stable_condition_threshold
        ]
        summary[f"gt_{threshold:g}m"] = {
            "count": len(matched),
            "fraction": len(matched) / max(len(transitions), 1),
            "low_speed_both_count": len(low_speed),
            "low_speed_stable_condition_count": len(low_speed_stable_condition),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.max_frame_gap <= 0:
        raise RuntimeError("--max-frame-gap must be > 0")
    if args.max_events < 0:
        raise RuntimeError("--max-events must be >= 0")
    if args.num_workers < 0:
        raise RuntimeError("--num-workers must be >= 0")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")

    header, records, total_records = load_manifest(
        args.sample_manifest,
        set(args.scenario),
        set(args.route),
    )
    reference_comparison = None
    if args.reference_manifest is not None:
        _reference_header, reference_records, _reference_total = load_manifest(
            args.reference_manifest,
            set(args.scenario),
            set(args.route),
        )
        reference_comparison = compare_manifest_records(records, reference_records)
    transitions = build_transitions(records, args.max_frame_gap)
    transitions.sort(
        key=lambda row: (
            row["scenario"],
            row["route"],
            row["before_frame"],
        )
    )
    event_candidates = [
        row
        for row in transitions
        if row["manifest_endpoint_jump_meters"] > args.event_threshold
    ]
    event_candidates.sort(key=lambda row: row["manifest_endpoint_jump_meters"], reverse=True)
    events = event_candidates[: args.max_events]

    record_lookup = {
        (str(record["route_dir"]), int(record["frame"])): record for record in records
    }
    recomputed = recompute_event_samples(
        events,
        header,
        args.num_workers,
        args.progress_every,
    ) if events else {}
    event_rows, traces = build_event_outputs(
        events,
        record_lookup,
        recomputed,
        args.event_threshold,
        args.recompute_tolerance,
    ) if events else ([], [])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "transitions.csv", transitions)
    write_csv(args.output_dir / "events.csv", event_rows)
    with (args.output_dir / "event_traces.jsonl").open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, separators=(",", ":"), sort_keys=True) + "\n")

    outcome_counts = Counter(row["recompute_outcome"] for row in event_rows)
    summary = {
        "sample_manifest": str(args.sample_manifest),
        "reference_manifest": (
            str(args.reference_manifest) if args.reference_manifest is not None else None
        ),
        "reference_comparison": reference_comparison,
        "manifest_total_record_count": total_records,
        "selected_record_count": len(records),
        "route_count": len({record["route_dir"] for record in records}),
        "transition_count": len(transitions),
        "max_frame_gap": args.max_frame_gap,
        "manifest_spatial_target_contract": header.get("spatial_target"),
        "current_spatial_target_contract": spatial_target_metadata(),
        "manifest_contract_matches_current": (
            header.get("spatial_target") == spatial_target_metadata()
        ),
        "jump_thresholds": transition_summary(
            transitions,
            sorted(set(args.jump_thresholds)),
            args.low_speed_threshold,
            args.stable_condition_threshold,
        ),
        "event_threshold_meters": args.event_threshold,
        "event_candidate_count": len(event_candidates),
        "event_recomputed_count": len(event_rows),
        "recompute_outcome_counts": dict(sorted(outcome_counts.items())),
        "scenario_filters": args.scenario,
        "route_filters": args.route,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Wrote spatial target continuity diagnostics: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
