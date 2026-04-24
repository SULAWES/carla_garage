#!/usr/bin/env python3
"""Find original route paths from LiDAR BEV batch outputs.

Examples:
  python tools/find_route_from_bev_batch.py \
    --summary-json /path/to/lidar_bev_batch_v3/summary.json \
    --query Town12_Rep0_2488_0_route0_11_08_02_55_17 \
    --search-root /data

  python tools/find_route_from_bev_batch.py \
    --summary-json /path/to/lidar_bev_batch_v3/summary.json \
    --query djy__carla_dataset__Accident__Town12_Rep0_2488_0_route0_11_08_02_55_17 \
    --search-root /data1 --search-root /data2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to batch summary.json.")
    parser.add_argument(
        "--query",
        required=True,
        help="Route basename, batch output folder name, or a substring of route_dir.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Optional data root to search for accessible local route paths. Can be specified multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of matched routes to print from summary.json.",
    )
    return parser.parse_args()


def safe_route_name(route_dir: str) -> str:
    parts = Path(route_dir).parts
    return "__".join(parts[-4:])


def normalize_query(query: str) -> str:
    return query.strip()


def match_route(query: str, route_dir: str) -> bool:
    route_name = Path(route_dir).name
    safe_name = safe_route_name(route_dir)
    return query in route_dir or query == route_name or query == safe_name


def find_local_matches(route_name: str, search_roots: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(route_name):
            if path.is_dir():
                matches.append(path)
    return sorted(set(matches))


def main() -> int:
    args = parse_args()
    query = normalize_query(args.query)
    with args.summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    matched_routes = [route for route in summary["routes"] if match_route(query, route["route_dir"])]
    if not matched_routes:
        print(f"No routes in summary matched query: {query}")
        return 1

    print(f"Matched {len(matched_routes)} route(s) in summary.")
    for index, route in enumerate(matched_routes[: args.limit], start=1):
        route_dir = route["route_dir"]
        route_name = Path(route_dir).name
        safe_name = safe_route_name(route_dir)
        local_matches = find_local_matches(route_name, args.search_root)

        print()
        print(f"[{index}] route_name: {route_name}")
        print(f"summary_route_dir: {route_dir}")
        print(f"batch_output_dir_name: {safe_name}")
        if local_matches:
            print("local_matches:")
            for match in local_matches:
                print(f"  {match}")
        else:
            print("local_matches: none found under provided --search-root")

    if len(matched_routes) > args.limit:
        print()
        print(f"Truncated output to first {args.limit} matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
