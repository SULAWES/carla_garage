#!/usr/bin/env python3
"""Analyze residual dx/dy distributions from LiDAR BEV batch summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Summary paths, optionally prefixed as name=path.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Number of top routes/scenarios to print.")
    parser.add_argument(
        "--large-threshold",
        type=float,
        default=0.5,
        help="Residual norm threshold used for large-residual grouping.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - index) + sorted_values[high] * (index - low)


def load_rows(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for route in data.get("routes", []):
        route_name = Path(route.get("route_dir", "")).name
        route_parts = route_name.split("_")
        town = route_parts[0] if route_parts else ""
        scenario = "_".join(route_parts[:3]) if len(route_parts) >= 3 else route_name
        for frame in route.get("frames", []):
            for result in frame.get("results", []):
                dx = float(result.get("refinement_dx", 0.0))
                dy = float(result.get("refinement_dy", 0.0))
                rows.append(
                    {
                        **result,
                        "route": route_name,
                        "town": town,
                        "scenario": scenario,
                        "dx": dx,
                        "dy": dy,
                        "norm": math.hypot(dx, dy),
                    }
                )
    return rows


def stats(rows: list[dict[str, object]]) -> dict[str, float]:
    dx = [float(row["dx"]) for row in rows]
    dy = [float(row["dy"]) for row in rows]
    norm = [float(row["norm"]) for row in rows]
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_dx": mean(dx),
        "median_dx": quantile(dx, 0.5),
        "p10_dx": quantile(dx, 0.1),
        "p90_dx": quantile(dx, 0.9),
        "mean_dy": mean(dy),
        "median_dy": quantile(dy, 0.5),
        "p10_dy": quantile(dy, 0.1),
        "p90_dy": quantile(dy, 0.9),
        "mean_norm": mean(norm),
        "median_norm": quantile(norm, 0.5),
        "p90_norm": quantile(norm, 0.9),
        "p95_norm": quantile(norm, 0.95),
        "max_norm": max(norm),
        "pos_dx_rate": sum(value > 0.05 for value in dx) / len(dx),
        "neg_dx_rate": sum(value < -0.05 for value in dx) / len(dx),
        "pos_dy_rate": sum(value > 0.05 for value in dy) / len(dy),
        "neg_dy_rate": sum(value < -0.05 for value in dy) / len(dy),
        "near_zero_rate": sum(value <= 0.1 for value in norm) / len(norm),
        "large_gt_1_rate": sum(value > 1.0 for value in norm) / len(norm),
        "large_gt_2_rate": sum(value > 2.0 for value in norm) / len(norm),
    }


def print_stats(label: str, rows: list[dict[str, object]]) -> None:
    print(label, json.dumps(stats(rows), ensure_ascii=False))


def summarize(name: str, rows: list[dict[str, object]], top_k: int, large_threshold: float) -> None:
    print(f"=== {name} ===")
    print_stats("all", rows)

    for history in sorted({int(row["history_offset"]) for row in rows}):
        print_stats(f"history={history}", [row for row in rows if int(row["history_offset"]) == history])

    print_stats("frame=5", [row for row in rows if int(row["current_frame"]) == 5])
    print_stats("frame>5", [row for row in rows if int(row["current_frame"]) > 5])
    print_stats(f"norm>{large_threshold}", [row for row in rows if float(row["norm"]) > large_threshold])
    print_stats("norm>1", [row for row in rows if float(row["norm"]) > 1.0])

    for town in sorted({str(row["town"]) for row in rows}):
        print_stats(f"town={town}", [row for row in rows if row["town"] == town])

    print("top_scenarios_by_mean_norm")
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario"])].append(row)
    scenario_stats = []
    for scenario, scenario_rows in by_scenario.items():
        scenario_stats.append(
            (
                mean([float(row["norm"]) for row in scenario_rows]),
                mean([float(row["dx"]) for row in scenario_rows]),
                mean([float(row["dy"]) for row in scenario_rows]),
                scenario,
                len(scenario_rows),
                sum(float(row["norm"]) > large_threshold for row in scenario_rows),
            )
        )
    for mean_norm, mean_dx, mean_dy, scenario, count, large_count in sorted(scenario_stats, reverse=True)[:top_k]:
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "n": count,
                    "mean_norm": mean_norm,
                    "mean_dx": mean_dx,
                    "mean_dy": mean_dy,
                    "large_count": large_count,
                },
                ensure_ascii=False,
            )
        )

    print("common_rounded_residuals")
    counter = Counter(
        (round(float(row["dx"]), 2), round(float(row["dy"]), 2), int(row["history_offset"])) for row in rows
    )
    for (dx, dy, history), count in counter.most_common(top_k):
        print(json.dumps({"history": history, "dx": dx, "dy": dy, "count": count}, ensure_ascii=False))
    print()


def parse_summary_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.parent.name or path.name, path


def main() -> int:
    args = parse_args()
    for value in args.summaries:
        name, path = parse_summary_arg(value)
        summarize(name, load_rows(path), top_k=args.top_k, large_threshold=args.large_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
