"""Bench2Drive route-level quality filter helpers."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path


QUALITY_FILTER_NONE = "none"
QUALITY_FILTER_SOFT_CLEAN = "soft_clean"
QUALITY_FILTER_SYB_CLEAN = "syb_clean"
QUALITY_FILTERS = (QUALITY_FILTER_NONE, QUALITY_FILTER_SOFT_CLEAN, QUALITY_FILTER_SYB_CLEAN)

FAILED_STATUSES = {
    "Failed",
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    reason: str


def quality_filter_decision(route_dir: Path, quality_filter: str) -> QualityDecision:
    if quality_filter == QUALITY_FILTER_NONE:
        return QualityDecision(True, "keep")
    result = load_result(route_dir)
    soft_keep, soft_reason = soft_clean_decision(route_dir, result)
    if quality_filter == QUALITY_FILTER_SOFT_CLEAN:
        return QualityDecision(soft_keep, soft_reason)
    if quality_filter == QUALITY_FILTER_SYB_CLEAN:
        syb_keep, syb_reason = syb_clean_decision(route_dir, result, soft_keep, soft_reason)
        return QualityDecision(syb_keep, syb_reason)
    raise RuntimeError(f"Unsupported quality filter: {quality_filter}")


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
    min_speed_only = num_infractions is not None and num_infractions == min_speed_infractions
    if score < 100.0 - 1e-6 and not min_speed_only:
        return False, "score_lt_100_non_minspeed"
    return True, "keep"


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
