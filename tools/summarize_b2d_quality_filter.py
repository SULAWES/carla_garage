#!/usr/bin/env python3
"""Summarize Bench2Drive route quality filters before building manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_MODE_SPATIAL_PATH = "spatial_path"
TARGET_MODE_FUTURE_EGO_TIME = "future_ego_time"
TARGET_SPEED_CLASSES_MPS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)
FAILED_STATUSES = {
    "Failed",
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


@dataclass
class RouteSummary:
    route_dir: Path
    scenario: str
    route: str
    sample_count: int
    has_results: bool
    status: str
    score_composed: float | None
    num_infractions: int | None
    min_speed_infractions: int
    full_keep: bool
    soft_clean_keep: bool
    soft_clean_reason: str
    syb_clean_keep: bool
    syb_clean_reason: str
    command_counts: Counter[str]
    speed_bin_counts: Counter[str]
    target_speed_class_counts: Counter[str]
    brake_counts: Counter[str]
    target_speed_valid_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--route-glob", default="*/*")
    parser.add_argument("--frame-sampling", type=int, default=5)
    parser.add_argument("--max-routes", type=int, default=None, help="Debug limit after route discovery.")
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_SPATIAL_PATH, TARGET_MODE_FUTURE_EGO_TIME),
        default=TARGET_MODE_SPATIAL_PATH,
    )
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = list(iter_route_dirs(args.root_dir, args.route_glob))
    if args.max_routes is not None:
        routes = routes[: args.max_routes]
    if not routes:
        roots = ", ".join(str(root) for root in args.root_dir)
        raise RuntimeError(f"No Bench2Drive route dirs found under {roots} with route_glob={args.route_glob!r}")

    route_summaries = [summarize_route(route_dir, args) for route_dir in routes]
    summary = build_summary(route_summaries, args)
    print_summary(summary)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "quality_filter_summary.json", summary)
        write_route_csv(args.output_dir / "route_quality_decisions.csv", route_summaries)
        write_scenario_csv(args.output_dir / "scenario_quality_summary.csv", summary["per_scenario"])
        print(f"Wrote stats to: {args.output_dir}", flush=True)


def iter_route_dirs(root_dirs: Iterable[Path], route_glob: str) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in root_dirs:
        root = root.expanduser()
        if _is_b2d_route(root):
            route = root.resolve()
            if route not in seen:
                seen.add(route)
                yield route
        if not root.exists():
            raise RuntimeError(f"Bench2Drive root does not exist: {root}")
        for route_dir in sorted(root.glob(route_glob)):
            if not _is_b2d_route(route_dir):
                continue
            route = route_dir.resolve()
            if route in seen:
                continue
            seen.add(route)
            yield route


def summarize_route(route_dir: Path, args: argparse.Namespace) -> RouteSummary:
    result = load_result(route_dir)
    soft_keep, soft_reason = soft_clean_decision(route_dir, result)
    syb_keep, syb_reason = syb_clean_decision(route_dir, result, soft_keep, soft_reason)
    sample_stats = summarize_samples(route_dir, args)
    return RouteSummary(
        route_dir=route_dir,
        scenario=scenario_name(route_dir),
        route=route_dir.name,
        sample_count=sample_stats["sample_count"],
        has_results=result is not None,
        status=str(result.get("status", "")) if result else "",
        score_composed=result_score(result),
        num_infractions=result_num_infractions(result),
        min_speed_infractions=result_min_speed_infractions(result),
        full_keep=True,
        soft_clean_keep=soft_keep,
        soft_clean_reason=soft_reason,
        syb_clean_keep=syb_keep,
        syb_clean_reason=syb_reason,
        command_counts=sample_stats["command_counts"],
        speed_bin_counts=sample_stats["speed_bin_counts"],
        target_speed_class_counts=sample_stats["target_speed_class_counts"],
        brake_counts=sample_stats["brake_counts"],
        target_speed_valid_count=sample_stats["target_speed_valid_count"],
    )


def summarize_samples(route_dir: Path, args: argparse.Namespace) -> dict:
    command_counts: Counter[str] = Counter()
    speed_bin_counts: Counter[str] = Counter()
    target_speed_class_counts: Counter[str] = Counter()
    brake_counts: Counter[str] = Counter()
    target_speed_valid_count = 0
    sample_count = 0
    frames = sorted(int(path.stem.split(".")[0]) for path in annotation_dir(route_dir).glob("*.json.gz"))
    for frame in frames[:: args.frame_sampling]:
        if not has_required_sample_files(route_dir, frame, args):
            continue
        annotation = load_annotation(route_dir, frame)
        command = int(annotation.get("command_far", annotation.get("command_near", annotation.get("command", 4))))
        speed = float(annotation.get("speed", 0.0))
        target_speed_valid = "target_speed" in annotation and ("brake" in annotation or "control_brake" in annotation)
        target_speed = float(annotation.get("target_speed", speed))
        brake = annotation_bool(annotation.get("brake", annotation.get("control_brake", False)))

        command_counts[str(command)] += 1
        speed_bin_counts[speed_bin(speed)] += 1
        target_speed_class_counts[str(target_speed_class(target_speed, brake))] += 1
        brake_counts[str(int(brake))] += 1
        target_speed_valid_count += int(target_speed_valid)
        sample_count += 1
    return {
        "sample_count": sample_count,
        "command_counts": command_counts,
        "speed_bin_counts": speed_bin_counts,
        "target_speed_class_counts": target_speed_class_counts,
        "brake_counts": brake_counts,
        "target_speed_valid_count": target_speed_valid_count,
    }


def has_required_sample_files(route_dir: Path, frame: int, args: argparse.Namespace) -> bool:
    if not has_frame_file(image_dir(route_dir), frame, ".jpg"):
        return False
    if not has_frame_file(route_dir / "lidar", frame, ".laz"):
        return False
    if args.target_mode == TARGET_MODE_SPATIAL_PATH:
        return has_frame_file(annotation_dir(route_dir), frame + 1, ".json.gz")
    if args.target_mode == TARGET_MODE_FUTURE_EGO_TIME:
        for offset in range(args.future_stride, args.future_stride * (args.num_poses + 1), args.future_stride):
            if not has_frame_file(annotation_dir(route_dir), frame + offset, ".json.gz"):
                return False
        return True
    raise RuntimeError(f"Unsupported target mode: {args.target_mode}")


def build_summary(route_summaries: list[RouteSummary], args: argparse.Namespace) -> dict:
    policies = {
        "full": lambda route: route.full_keep,
        "soft_clean": lambda route: route.soft_clean_keep,
        "syb_clean": lambda route: route.syb_clean_keep,
    }
    full_sample_count = sum(route.sample_count for route in route_summaries)
    full_route_count = len(route_summaries)

    policy_summary = {}
    for name, keep_fn in policies.items():
        kept = [route for route in route_summaries if keep_fn(route)]
        policy_summary[name] = aggregate_routes(kept, full_sample_count, full_route_count)

    per_scenario: dict[str, dict] = defaultdict(dict)
    scenarios = sorted({route.scenario for route in route_summaries})
    for scenario in scenarios:
        scenario_routes = [route for route in route_summaries if route.scenario == scenario]
        full_samples = sum(route.sample_count for route in scenario_routes)
        full_routes = len(scenario_routes)
        row = {
            "scenario": scenario,
            "full_routes": full_routes,
            "full_samples": full_samples,
        }
        for policy_name, keep_fn in policies.items():
            kept = [route for route in scenario_routes if keep_fn(route)]
            kept_samples = sum(route.sample_count for route in kept)
            row[f"{policy_name}_routes"] = len(kept)
            row[f"{policy_name}_samples"] = kept_samples
            row[f"{policy_name}_sample_retention"] = safe_div(kept_samples, full_samples)
            row[f"{policy_name}_route_retention"] = safe_div(len(kept), full_routes)
        per_scenario[scenario] = row

    skip_reasons = {
        "soft_clean": dict(Counter(route.soft_clean_reason for route in route_summaries if not route.soft_clean_keep)),
        "syb_clean": dict(Counter(route.syb_clean_reason for route in route_summaries if not route.syb_clean_keep)),
    }
    return {
        "args": {
            "root_dir": [str(path) for path in args.root_dir],
            "route_glob": args.route_glob,
            "frame_sampling": args.frame_sampling,
            "target_mode": args.target_mode,
            "num_poses": args.num_poses,
            "future_stride": args.future_stride,
            "max_routes": args.max_routes,
        },
        "target_speed_label": {
            "schema": "syb_twohot_target_speed_v1",
            "classes_mps": list(TARGET_SPEED_CLASSES_MPS),
            "brake_class_index": 0,
        },
        "policies": policy_summary,
        "skip_reasons": skip_reasons,
        "per_scenario": dict(per_scenario),
    }


def aggregate_routes(routes: list[RouteSummary], full_sample_count: int, full_route_count: int) -> dict:
    command_counts: Counter[str] = Counter()
    speed_bin_counts: Counter[str] = Counter()
    target_speed_class_counts: Counter[str] = Counter()
    brake_counts: Counter[str] = Counter()
    target_speed_valid_count = 0
    sample_count = 0
    for route in routes:
        sample_count += route.sample_count
        target_speed_valid_count += route.target_speed_valid_count
        command_counts.update(route.command_counts)
        speed_bin_counts.update(route.speed_bin_counts)
        target_speed_class_counts.update(route.target_speed_class_counts)
        brake_counts.update(route.brake_counts)
    return {
        "routes": len(routes),
        "samples": sample_count,
        "route_retention_vs_full": safe_div(len(routes), full_route_count),
        "sample_retention_vs_full": safe_div(sample_count, full_sample_count),
        "scenarios": len({route.scenario for route in routes}),
        "command_counts": dict(sorted(command_counts.items())),
        "speed_bins_mps": dict(sorted(speed_bin_counts.items())),
        "target_speed_class_counts": dict(sorted(target_speed_class_counts.items())),
        "target_speed_label_valid_count": target_speed_valid_count,
        "target_speed_label_valid_fraction": safe_div(target_speed_valid_count, sample_count),
        "brake_counts": dict(sorted(brake_counts.items())),
    }


def soft_clean_decision(route_dir: Path, result: dict | None) -> tuple[bool, str]:
    if route_dir.name.startswith("FAILED_"):
        return False, "failed_prefix"
    if result is None:
        return False, "missing_results"
    status = str(result.get("status", ""))
    if status in FAILED_STATUSES:
        return False, status
    return True, "keep"


def syb_clean_decision(
    route_dir: Path,
    result: dict | None,
    soft_keep: bool,
    soft_reason: str,
) -> tuple[bool, str]:
    if not soft_keep:
        return False, soft_reason
    score = result_score(result)
    if score is None:
        return False, "missing_score"
    num_infractions = result_num_infractions(result)
    min_speed_infractions = result_min_speed_infractions(result)
    min_speed_only = (
        num_infractions is not None
        and num_infractions == min_speed_infractions
    )
    if score < 100.0 - 1e-6 and not min_speed_only:
        return False, "score_lt_100_non_minspeed"
    return True, "keep"


def print_summary(summary: dict) -> None:
    for name, data in summary["policies"].items():
        print(
            f"{name}: routes={data['routes']} samples={data['samples']} "
            f"route_retention={data['route_retention_vs_full']:.3f} "
            f"sample_retention={data['sample_retention_vs_full']:.3f} "
            f"scenarios={data['scenarios']} "
            f"target_speed_valid={data['target_speed_label_valid_fraction']:.3f}",
            flush=True,
        )
    print(f"soft_clean skip reasons: {summary['skip_reasons']['soft_clean']}", flush=True)
    print(f"syb_clean skip reasons: {summary['skip_reasons']['syb_clean']}", flush=True)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def write_route_csv(path: Path, routes: list[RouteSummary]) -> None:
    fieldnames = [
        "scenario",
        "route",
        "route_dir",
        "sample_count",
        "has_results",
        "status",
        "score_composed",
        "num_infractions",
        "min_speed_infractions",
        "soft_clean_keep",
        "soft_clean_reason",
        "syb_clean_keep",
        "syb_clean_reason",
        "target_speed_valid_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for route in routes:
            writer.writerow({
                "scenario": route.scenario,
                "route": route.route,
                "route_dir": str(route.route_dir),
                "sample_count": route.sample_count,
                "has_results": int(route.has_results),
                "status": route.status,
                "score_composed": route.score_composed,
                "num_infractions": route.num_infractions,
                "min_speed_infractions": route.min_speed_infractions,
                "soft_clean_keep": int(route.soft_clean_keep),
                "soft_clean_reason": route.soft_clean_reason,
                "syb_clean_keep": int(route.syb_clean_keep),
                "syb_clean_reason": route.syb_clean_reason,
                "target_speed_valid_count": route.target_speed_valid_count,
            })


def write_scenario_csv(path: Path, per_scenario: dict[str, dict]) -> None:
    fieldnames = [
        "scenario",
        "full_routes",
        "full_samples",
        "soft_clean_routes",
        "soft_clean_samples",
        "soft_clean_route_retention",
        "soft_clean_sample_retention",
        "syb_clean_routes",
        "syb_clean_samples",
        "syb_clean_route_retention",
        "syb_clean_sample_retention",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in sorted(per_scenario):
            row = per_scenario[scenario]
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_result(route_dir: Path) -> dict | None:
    for path in (route_dir / "results.json.gz", route_dir / "results.json"):
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as file:
                return json.load(file)
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return None


def load_annotation(route_dir: Path, frame: int) -> dict:
    path = frame_path(annotation_dir(route_dir), frame, ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        annotation = json.load(file)
    if "pos_global" in annotation and "x" not in annotation:
        annotation = dict(annotation)
        annotation["x"] = float(annotation["pos_global"][0])
        annotation["y"] = float(annotation["pos_global"][1])
        annotation.setdefault("command_far", int(annotation.get("command", annotation.get("next_command", 4))))
        annotation.setdefault("command_near", int(annotation.get("next_command", annotation["command_far"])))
    return annotation


def result_score(result: dict | None) -> float | None:
    if not result:
        return None
    try:
        return float(result.get("scores", {}).get("score_composed"))
    except (TypeError, ValueError):
        return None


def result_num_infractions(result: dict | None) -> int | None:
    if not result:
        return None
    value = result.get("num_infractions")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    infractions = result.get("infractions", {})
    if isinstance(infractions, dict):
        return sum(len(value) for value in infractions.values() if isinstance(value, list))
    return None


def result_min_speed_infractions(result: dict | None) -> int:
    if not result:
        return 0
    infractions = result.get("infractions", {})
    if not isinstance(infractions, dict):
        return 0
    min_speed = infractions.get("min_speed_infractions", [])
    return len(min_speed) if isinstance(min_speed, list) else 0


def target_speed_class(target_speed: float, brake: bool) -> int:
    if brake or target_speed <= TARGET_SPEED_CLASSES_MPS[0]:
        return 0
    if target_speed >= TARGET_SPEED_CLASSES_MPS[-1]:
        return len(TARGET_SPEED_CLASSES_MPS) - 1
    upper_idx = next(index for index, value in enumerate(TARGET_SPEED_CLASSES_MPS) if value > target_speed)
    lower_idx = max(upper_idx - 1, 0)
    lower = TARGET_SPEED_CLASSES_MPS[lower_idx]
    upper = TARGET_SPEED_CLASSES_MPS[upper_idx]
    lower_weight = (upper - target_speed) / max(upper - lower, 1e-6)
    upper_weight = (target_speed - lower) / max(upper - lower, 1e-6)
    return lower_idx if lower_weight >= upper_weight else upper_idx


def speed_bin(speed: float) -> str:
    if speed < 0.1:
        return "<0.1"
    if speed < 2.0:
        return "0.1-2"
    if speed < 5.0:
        return "2-5"
    return ">=5"


def annotation_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _is_b2d_route(path: Path) -> bool:
    return path.is_dir() and image_dir(path).is_dir() and (path / "lidar").is_dir() and annotation_dir(path).is_dir()


def scenario_name(route_dir: Path) -> str:
    return route_dir.parent.name


def annotation_dir(route_dir: Path) -> Path:
    if (route_dir / "anno").is_dir():
        return route_dir / "anno"
    return route_dir / "measurements"


def image_dir(route_dir: Path) -> Path:
    if (route_dir / "camera" / "rgb_front").is_dir():
        return route_dir / "camera" / "rgb_front"
    return route_dir / "rgb"


def frame_path(directory: Path, frame: int, suffix: str) -> Path:
    for width in (5, 4, 0):
        stem = str(frame) if width == 0 else f"{frame:0{width}d}"
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise RuntimeError(f"Missing frame file in {directory}: frame={frame} suffix={suffix}")


def has_frame_file(directory: Path, frame: int, suffix: str) -> bool:
    for width in (5, 4, 0):
        stem = str(frame) if width == 0 else f"{frame:0{width}d}"
        if (directory / f"{stem}{suffix}").is_file():
            return True
    return False


if __name__ == "__main__":
    main()
