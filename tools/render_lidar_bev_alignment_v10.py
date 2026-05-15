#!/usr/bin/env python3
"""Render LiDAR/BEV alignment diagnostics with near-field protected scoring.

v10 keeps the v9 metric/evaluation output and the v8 residual-motion prior, but
changes the refinement candidate objective. A candidate is selected by a
composite score that includes:

  - v9 support/dynamic-fallback IoU
  - full dynamic-masked IoU
  - near-field raw IoU
  - near-field dynamic-masked IoU
  - residual translation/yaw prior penalty

It can also reject candidates that degrade the near-field raw/dynamic IoU too
much relative to the measurement-aligned initial candidate. This is intended to
prevent common-support scoring from improving far-field lane/curb structure
while damaging the 0-8m region that matters most for temporal LiDAR input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import render_lidar_bev_alignment_v8 as v8
import render_lidar_bev_alignment_v9 as v9


base = v9.base


@dataclass(frozen=True)
class NearFieldScoringConfig:
    enabled: bool = True
    near_field_max_m: float = 8.0
    support_iou_weight: float = 1.0
    dynamic_iou_weight: float = 0.5
    near_field_raw_weight: float = 1.0
    near_field_dynamic_weight: float = 2.0
    min_near_field_raw_gain: float = -0.02
    min_near_field_dynamic_gain: float = -0.02


V10_SCORING_CONFIG = NearFieldScoringConfig()


def parse_v10_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--partial-summary-every",
        type=int,
        default=1,
        help="Write partial_summary.json after every N completed routes. Use 0 to disable partial writes.",
    )
    parser.add_argument(
        "--resume-from-partial",
        action="store_true",
        help="Load output-dir/partial_summary.json and skip routes already present in it.",
    )
    parser.add_argument(
        "--start-route-index",
        type=int,
        default=0,
        help="Skip this many routes after route discovery and --max-routes slicing. Useful for tail-only reruns.",
    )
    parser.add_argument(
        "--disable-near-field-constraints",
        action="store_true",
        help="Disable hard near-field rejection while keeping the composite score weights.",
    )
    parser.add_argument(
        "--near-field-max-m",
        type=float,
        default=8.0,
        help="Near-field radius in meters used by v10 scoring.",
    )
    parser.add_argument(
        "--support-iou-weight",
        type=float,
        default=1.0,
        help="Weight for the v9 support/dynamic-fallback IoU in the refinement score.",
    )
    parser.add_argument(
        "--dynamic-iou-weight",
        type=float,
        default=0.5,
        help="Weight for full dynamic-masked IoU in the refinement score.",
    )
    parser.add_argument(
        "--near-field-raw-weight",
        type=float,
        default=1.0,
        help="Weight for raw near-field IoU in the refinement score.",
    )
    parser.add_argument(
        "--near-field-dynamic-weight",
        type=float,
        default=2.0,
        help="Weight for dynamic-masked near-field IoU in the refinement score.",
    )
    parser.add_argument(
        "--min-near-field-raw-gain",
        type=float,
        default=-0.02,
        help="Reject candidates whose near-field raw IoU drops below initial alignment by more than this.",
    )
    parser.add_argument(
        "--min-near-field-dynamic-gain",
        type=float,
        default=-0.02,
        help="Reject candidates whose near-field dynamic IoU drops below initial alignment by more than this.",
    )
    args, remaining = parser.parse_known_args(argv)
    if args.partial_summary_every < 0:
        parser.error("--partial-summary-every must be >= 0.")
    if args.start_route_index < 0:
        parser.error("--start-route-index must be >= 0.")
    if args.near_field_max_m <= 0:
        parser.error("--near-field-max-m must be > 0.")
    for name in (
        "support_iou_weight",
        "dynamic_iou_weight",
        "near_field_raw_weight",
        "near_field_dynamic_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0.")
    return args, remaining


def near_field_include_mask(shape: tuple[int, int], cfg: base.BevConfig) -> np.ndarray:
    dist = v9.distance_grid(shape, cfg)
    return dist < V10_SCORING_CONFIG.near_field_max_m


def near_field_iou_fields(
    current_channel: np.ndarray,
    compare_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    compare_dynamic_mask: np.ndarray,
    cfg: base.BevConfig,
) -> dict[str, float]:
    include = near_field_include_mask(current_channel.shape, cfg)
    outside_mask = ~include
    raw_iou = base.masked_occupancy_iou(current_channel, compare_channel, outside_mask)
    dynamic_iou = base.masked_occupancy_iou(
        current_channel,
        compare_channel,
        current_dynamic_mask | compare_dynamic_mask | outside_mask,
    )
    return {
        "near_field_raw_iou": float(raw_iou),
        "near_field_dynamic_iou": float(dynamic_iou),
        "near_field_pixels": int(include.sum()),
    }


def candidate_metrics(
    lidar: np.ndarray,
    boxes: list[dict],
    current_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    cfg: base.BevConfig,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
) -> dict[str, object]:
    evaluation = v9.evaluate_alignment(
        lidar=lidar,
        boxes=boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
    )
    evaluation.update(
        near_field_iou_fields(
            current_channel=current_channel,
            compare_channel=evaluation["channel"],
            current_dynamic_mask=current_dynamic_mask,
            compare_dynamic_mask=evaluation["dynamic_mask"],
            cfg=cfg,
        )
    )
    return evaluation


def composite_score(
    evaluation: dict[str, object],
    total_translation: np.ndarray,
    total_yaw: float,
    initial_near_raw_iou: float,
    initial_near_dynamic_iou: float,
) -> tuple[float, float, float, float, float]:
    near_raw_gain = float(evaluation["near_field_raw_iou"]) - initial_near_raw_iou
    near_dynamic_gain = float(evaluation["near_field_dynamic_iou"]) - initial_near_dynamic_iou
    _, penalty, norm = v8.refinement_score(
        float(evaluation["iou"]),
        total_translation,
        total_yaw,
        v8.SCORING_CONFIG,
    )
    score = (
        V10_SCORING_CONFIG.support_iou_weight * float(evaluation["iou"])
        + V10_SCORING_CONFIG.dynamic_iou_weight * float(evaluation["dynamic_iou"])
        + V10_SCORING_CONFIG.near_field_raw_weight * float(evaluation["near_field_raw_iou"])
        + V10_SCORING_CONFIG.near_field_dynamic_weight * float(evaluation["near_field_dynamic_iou"])
        - penalty
    )
    return float(score), float(penalty), float(norm), near_raw_gain, near_dynamic_gain


def is_candidate_allowed(
    total_translation: np.ndarray,
    near_raw_gain: float,
    near_dynamic_gain: float,
) -> bool:
    if not v8.is_candidate_allowed(total_translation, v8.SCORING_CONFIG):
        return False
    if not V10_SCORING_CONFIG.enabled:
        return True
    if near_raw_gain < V10_SCORING_CONFIG.min_near_field_raw_gain:
        return False
    if near_dynamic_gain < V10_SCORING_CONFIG.min_near_field_dynamic_gain:
        return False
    return True


def is_better_candidate(candidate: dict[str, object], best: dict[str, object]) -> bool:
    candidate_score = float(candidate["score"])
    best_score = float(best["score"])
    if candidate_score > best_score + 1e-9:
        return True
    if abs(candidate_score - best_score) > 1e-9:
        return False
    candidate_iou = float(candidate["iou"])
    best_iou = float(best["iou"])
    if candidate_iou > best_iou + 1e-9:
        return True
    if abs(candidate_iou - best_iou) > 1e-9:
        return False
    return float(candidate["cost"]) < float(best["cost"])


def make_candidate(
    evaluation: dict[str, object],
    lidar: np.ndarray,
    boxes: list[dict],
    translation: np.ndarray,
    total_translation: np.ndarray,
    yaw: float,
    total_yaw: float,
    initial_near_raw_iou: float,
    initial_near_dynamic_iou: float,
    searched_candidates: int,
    rejected_candidates: int,
) -> dict[str, object]:
    score, penalty, norm, near_raw_gain, near_dynamic_gain = composite_score(
        evaluation,
        total_translation,
        total_yaw,
        initial_near_raw_iou,
        initial_near_dynamic_iou,
    )
    return {
        "iou": evaluation["iou"],
        "score": score,
        "penalty": penalty,
        "translation_norm": norm,
        "translation": translation.copy(),
        "total_translation": total_translation.copy(),
        "yaw": yaw,
        "total_yaw": total_yaw,
        "lidar": lidar,
        "boxes": boxes,
        "channel": evaluation["channel"],
        "dynamic_mask": evaluation["dynamic_mask"],
        "eval_mask": evaluation["eval_mask"],
        "support_pixels": evaluation["support_pixels"],
        "searched_candidates": searched_candidates,
        "rejected_candidates": rejected_candidates,
        "cost": v8.refinement_cost(total_translation, total_yaw),
        "near_field_raw_iou": evaluation["near_field_raw_iou"],
        "near_field_dynamic_iou": evaluation["near_field_dynamic_iou"],
        "near_field_raw_gain": near_raw_gain,
        "near_field_dynamic_gain": near_dynamic_gain,
    }


def search_refinement_stage(
    base_lidar: np.ndarray,
    base_boxes: list[dict],
    current_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    cfg: base.BevConfig,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
    xy_range: float,
    xy_step: float,
    yaw_range_deg: float,
    yaw_step_deg: float,
    prior_translation: np.ndarray,
    prior_yaw: float,
    initial_near_raw_iou: float,
    initial_near_dynamic_iou: float,
) -> dict[str, object]:
    base_eval = candidate_metrics(
        lidar=base_lidar,
        boxes=base_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
    )
    best = make_candidate(
        evaluation=base_eval,
        lidar=base_lidar,
        boxes=base_boxes,
        translation=np.zeros(3, dtype=np.float32),
        total_translation=prior_translation,
        yaw=0.0,
        total_yaw=prior_yaw,
        initial_near_raw_iou=initial_near_raw_iou,
        initial_near_dynamic_iou=initial_near_dynamic_iou,
        searched_candidates=1,
        rejected_candidates=0,
    )

    dx_values = base.refinement_values(xy_range, xy_step)
    dy_values = base.refinement_values(xy_range, xy_step)
    yaw_values_deg = base.refinement_values(yaw_range_deg, yaw_step_deg)

    searched_candidates = 0
    rejected_candidates = 0
    for dx in dx_values:
        for dy in dy_values:
            translation = np.array([dx, dy, 0.0], dtype=np.float32)
            for yaw_deg in yaw_values_deg:
                searched_candidates += 1
                yaw = float(np.deg2rad(float(yaw_deg)))
                total_translation, total_yaw = base.compose_residual_transform(
                    prior_translation, prior_yaw, translation, yaw
                )
                if dx == 0.0 and dy == 0.0 and yaw == 0.0:
                    continue

                candidate_lidar = base.algin_lidar(base_lidar, translation, yaw)
                candidate_boxes = base.apply_transform_to_boxes(base_boxes, translation, yaw)
                candidate_eval = candidate_metrics(
                    lidar=candidate_lidar,
                    boxes=candidate_boxes,
                    current_channel=current_channel,
                    current_dynamic_mask=current_dynamic_mask,
                    cfg=cfg,
                    dynamic_classes=dynamic_classes,
                    support_cfg=support_cfg,
                )
                candidate = make_candidate(
                    evaluation=candidate_eval,
                    lidar=candidate_lidar,
                    boxes=candidate_boxes,
                    translation=translation,
                    total_translation=total_translation,
                    yaw=yaw,
                    total_yaw=total_yaw,
                    initial_near_raw_iou=initial_near_raw_iou,
                    initial_near_dynamic_iou=initial_near_dynamic_iou,
                    searched_candidates=searched_candidates,
                    rejected_candidates=rejected_candidates,
                )
                if not is_candidate_allowed(
                    total_translation,
                    float(candidate["near_field_raw_gain"]),
                    float(candidate["near_field_dynamic_gain"]),
                ):
                    rejected_candidates += 1
                    continue
                if is_better_candidate(candidate, best):
                    best = candidate

    best["searched_candidates"] = searched_candidates
    best["rejected_candidates"] = rejected_candidates
    return best


def refine_alignment(
    aligned_lidar: np.ndarray,
    aligned_boxes: list[dict],
    current_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    cfg: base.BevConfig,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
    refinement_cfg: base.RefinementConfig,
) -> tuple[np.ndarray, list[dict], dict[str, object]]:
    initial_eval = candidate_metrics(
        lidar=aligned_lidar,
        boxes=aligned_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
    )
    zero_translation = np.zeros(3, dtype=np.float32)
    initial = make_candidate(
        evaluation=initial_eval,
        lidar=aligned_lidar,
        boxes=aligned_boxes,
        translation=zero_translation,
        total_translation=zero_translation,
        yaw=0.0,
        total_yaw=0.0,
        initial_near_raw_iou=float(initial_eval["near_field_raw_iou"]),
        initial_near_dynamic_iou=float(initial_eval["near_field_dynamic_iou"]),
        searched_candidates=1,
        rejected_candidates=0,
    )
    best = {
        **initial,
        "used_fallback": False,
        "initial_score": initial["score"],
        "initial_penalty": initial["penalty"],
        "coarse_translation": zero_translation.copy(),
        "coarse_yaw": 0.0,
        "coarse_iou": initial["iou"],
        "coarse_score": initial["score"],
        "coarse_penalty": initial["penalty"],
        "coarse_translation_norm": initial["translation_norm"],
        "coarse_candidates": 1,
        "coarse_rejected_candidates": 0,
        "fine_translation": zero_translation.copy(),
        "fine_yaw": 0.0,
        "fine_iou": initial["iou"],
        "fine_score": initial["score"],
        "fine_penalty": initial["penalty"],
        "fine_translation_norm": initial["translation_norm"],
        "fine_candidates": 1,
        "fine_rejected_candidates": 0,
    }

    if not refinement_cfg.enabled:
        return best["lidar"], best["boxes"], best

    coarse = search_refinement_stage(
        base_lidar=aligned_lidar,
        base_boxes=aligned_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
        xy_range=refinement_cfg.coarse_xy_range,
        xy_step=refinement_cfg.coarse_xy_step,
        yaw_range_deg=refinement_cfg.coarse_yaw_range_deg,
        yaw_step_deg=refinement_cfg.coarse_yaw_step_deg,
        prior_translation=zero_translation,
        prior_yaw=0.0,
        initial_near_raw_iou=float(initial_eval["near_field_raw_iou"]),
        initial_near_dynamic_iou=float(initial_eval["near_field_dynamic_iou"]),
    )
    fine = search_refinement_stage(
        base_lidar=coarse["lidar"],
        base_boxes=coarse["boxes"],
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
        xy_range=refinement_cfg.fine_xy_range,
        xy_step=refinement_cfg.fine_xy_step,
        yaw_range_deg=refinement_cfg.fine_yaw_range_deg,
        yaw_step_deg=refinement_cfg.fine_yaw_step_deg,
        prior_translation=coarse["total_translation"],
        prior_yaw=float(coarse["total_yaw"]),
        initial_near_raw_iou=float(initial_eval["near_field_raw_iou"]),
        initial_near_dynamic_iou=float(initial_eval["near_field_dynamic_iou"]),
    )
    total_translation, total_yaw = base.compose_residual_transform(
        coarse["translation"], coarse["yaw"], fine["translation"], fine["yaw"]
    )
    best = {
        **fine,
        "translation": total_translation,
        "yaw": total_yaw,
        "searched_candidates": int(coarse["searched_candidates"] + fine["searched_candidates"]),
        "rejected_candidates": int(coarse["rejected_candidates"] + fine["rejected_candidates"]),
        "used_fallback": False,
        "initial_score": initial["score"],
        "initial_penalty": initial["penalty"],
        "coarse_translation": coarse["translation"].copy(),
        "coarse_yaw": coarse["yaw"],
        "coarse_iou": coarse["iou"],
        "coarse_score": coarse["score"],
        "coarse_penalty": coarse["penalty"],
        "coarse_translation_norm": coarse["translation_norm"],
        "coarse_candidates": int(coarse["searched_candidates"]),
        "coarse_rejected_candidates": int(coarse["rejected_candidates"]),
        "fine_translation": fine["translation"].copy(),
        "fine_yaw": fine["yaw"],
        "fine_iou": fine["iou"],
        "fine_score": fine["score"],
        "fine_penalty": fine["penalty"],
        "fine_translation_norm": fine["translation_norm"],
        "fine_candidates": int(fine["searched_candidates"]),
        "fine_rejected_candidates": int(fine["rejected_candidates"]),
    }
    if refinement_cfg.fallback_if_worse and float(best["score"]) < float(initial["score"]):
        best = {
            **best,
            **initial,
            "used_fallback": True,
            "searched_candidates": int(coarse["searched_candidates"] + fine["searched_candidates"]),
            "rejected_candidates": int(coarse["rejected_candidates"] + fine["rejected_candidates"]),
            "initial_score": initial["score"],
            "initial_penalty": initial["penalty"],
            "coarse_translation": coarse["translation"].copy(),
            "coarse_yaw": coarse["yaw"],
            "coarse_iou": coarse["iou"],
            "coarse_score": coarse["score"],
            "coarse_penalty": coarse["penalty"],
            "coarse_translation_norm": coarse["translation_norm"],
            "coarse_candidates": int(coarse["searched_candidates"]),
            "coarse_rejected_candidates": int(coarse["rejected_candidates"]),
            "fine_translation": fine["translation"].copy(),
            "fine_yaw": fine["yaw"],
            "fine_iou": fine["iou"],
            "fine_score": fine["score"],
            "fine_penalty": fine["penalty"],
            "fine_translation_norm": fine["translation_norm"],
            "fine_candidates": int(fine["searched_candidates"]),
            "fine_rejected_candidates": int(fine["rejected_candidates"]),
        }

    return best["lidar"], best["boxes"], best


def patch_summary_data(summary: dict[str, object]) -> dict[str, object]:
    summary.setdefault("config", {}).update(
        {
            "script_version": "v9",
            "refinement_prior_enabled": v8.SCORING_CONFIG.enabled,
            "translation_penalty_weight": v8.SCORING_CONFIG.translation_penalty_weight,
            "yaw_penalty_weight": v8.SCORING_CONFIG.yaw_penalty_weight,
            "max_total_translation_norm": v8.SCORING_CONFIG.max_total_translation_norm,
            "min_score_improvement": v8.SCORING_CONFIG.min_score_improvement,
            "min_common_support_pixels": v9.SUPPORT_EVAL_CONFIG.min_common_support_pixels,
            "support_fallback_to_dynamic": v9.SUPPORT_EVAL_CONFIG.fallback_to_dynamic,
            "distance_bins": v9.DISTANCE_BINS,
        }
    )
    summary.setdefault("config", {}).update(
        {
            "script_version": "v10",
            "near_field_constraints_enabled": V10_SCORING_CONFIG.enabled,
            "near_field_max_m": V10_SCORING_CONFIG.near_field_max_m,
            "support_iou_weight": V10_SCORING_CONFIG.support_iou_weight,
            "dynamic_iou_weight": V10_SCORING_CONFIG.dynamic_iou_weight,
            "near_field_raw_weight": V10_SCORING_CONFIG.near_field_raw_weight,
            "near_field_dynamic_weight": V10_SCORING_CONFIG.near_field_dynamic_weight,
            "min_near_field_raw_gain": V10_SCORING_CONFIG.min_near_field_raw_gain,
            "min_near_field_dynamic_gain": V10_SCORING_CONFIG.min_near_field_dynamic_gain,
        }
    )
    return summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(patch_summary_data(summary), handle, indent=2)
    tmp_path.replace(path)


def patch_summary(output_dir: Path | None) -> None:
    if output_dir is None:
        return
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    write_summary(summary_path, summary)


def parse_base_args(argv: list[str]) -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return base.parse_args()
    finally:
        sys.argv = old_argv


def build_summary(
    args: argparse.Namespace,
    route_dirs: list[Path],
    cfg: base.BevConfig,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
    refinement_cfg: base.RefinementConfig,
) -> dict[str, object]:
    return {
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
            "common_support_mask_enabled": support_cfg.enabled,
            "support_dilation_pixels": support_cfg.dilation_pixels,
            "refinement_enabled": refinement_cfg.enabled,
            "coarse_refine_xy_range": refinement_cfg.coarse_xy_range,
            "coarse_refine_xy_step": refinement_cfg.coarse_xy_step,
            "coarse_refine_yaw_range_deg": refinement_cfg.coarse_yaw_range_deg,
            "coarse_refine_yaw_step_deg": refinement_cfg.coarse_yaw_step_deg,
            "fine_refine_xy_range": refinement_cfg.fine_xy_range,
            "fine_refine_xy_step": refinement_cfg.fine_xy_step,
            "fine_refine_yaw_range_deg": refinement_cfg.fine_yaw_range_deg,
            "fine_refine_yaw_step_deg": refinement_cfg.fine_yaw_step_deg,
            "refinement_fallback_if_worse": refinement_cfg.fallback_if_worse,
        },
        "routes": [],
    }


def load_partial_routes(partial_summary_path: Path) -> dict[str, dict[str, object]]:
    if not partial_summary_path.exists():
        return {}
    with partial_summary_path.open("r", encoding="utf-8") as handle:
        partial = json.load(handle)
    routes = partial.get("routes", [])
    if not isinstance(routes, list):
        return {}
    return {
        str(route.get("route_dir")): route
        for route in routes
        if isinstance(route, dict) and route.get("route_dir") is not None
    }


def run_base_main_with_partial(v10_args: argparse.Namespace, base_argv: list[str]) -> int:
    args = parse_base_args(base_argv)
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

    cfg = base.BevConfig(
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
    support_cfg = base.SupportConfig(
        enabled=not args.disable_common_support_mask,
        dilation_pixels=args.support_dilation_pixels,
    )
    refinement_cfg = base.RefinementConfig(
        enabled=not args.disable_refinement,
        coarse_xy_range=args.coarse_refine_xy_range,
        coarse_xy_step=args.coarse_refine_xy_step,
        coarse_yaw_range_deg=args.coarse_refine_yaw_range_deg,
        coarse_yaw_step_deg=args.coarse_refine_yaw_step_deg,
        fine_xy_range=args.fine_refine_xy_range,
        fine_xy_step=args.fine_refine_xy_step,
        fine_yaw_range_deg=args.fine_refine_yaw_range_deg,
        fine_yaw_step_deg=args.fine_refine_yaw_step_deg,
        fallback_if_worse=not args.allow_worse_refinement,
    )

    if args.route_name is not None:
        resolved_route_dir = base.find_route_dir_by_name(args.root_dir, args.route_name)
        print(f"Resolved route_dir: {resolved_route_dir}", flush=True)
        route_dirs = [resolved_route_dir]
    elif args.route_dir is not None:
        route_dirs = [args.route_dir]
    else:
        route_dirs = base.discover_route_dirs(args.root_dir)
        if args.max_routes is not None:
            route_dirs = route_dirs[: args.max_routes]

    if v10_args.start_route_index:
        if v10_args.start_route_index >= len(route_dirs):
            raise ValueError(
                f"--start-route-index {v10_args.start_route_index} is outside route_count={len(route_dirs)}"
            )
        print(f"Starting from route index {v10_args.start_route_index}", flush=True)
        route_dirs = route_dirs[v10_args.start_route_index :]

    summary = build_summary(args, route_dirs, cfg, dynamic_classes, support_cfg, refinement_cfg)
    partial_summary_path = output_dir / "partial_summary.json"
    partial_routes = load_partial_routes(partial_summary_path) if v10_args.resume_from_partial else {}
    if partial_routes:
        print(f"Loaded partial summary with {len(partial_routes)} completed routes", flush=True)

    completed_since_write = 0
    for route_dir in route_dirs:
        partial_route = partial_routes.get(str(route_dir))
        if partial_route is not None:
            print(f"Skipping route already in partial summary: {route_dir}", flush=True)
            summary["routes"].append(partial_route)
            continue

        route_summary = base.process_route(
            route_dir=route_dir,
            output_dir=base.route_output_dir(output_dir, route_dir),
            cfg=cfg,
            requested_frame=args.frame,
            history_offsets=args.history,
            frames_per_route=args.frames_per_route,
            dynamic_classes=dynamic_classes,
            support_cfg=support_cfg,
            refinement_cfg=refinement_cfg,
        )
        summary["routes"].append(route_summary)
        completed_since_write += 1
        if v10_args.partial_summary_every > 0 and completed_since_write >= v10_args.partial_summary_every:
            write_summary(partial_summary_path, summary)
            print(f"Saved partial summary to {partial_summary_path}", flush=True)
            completed_since_write = 0

    if v10_args.partial_summary_every > 0:
        write_summary(partial_summary_path, summary)
    write_summary(output_dir / "summary.json", summary)

    print(f"Saved outputs to {output_dir}", flush=True)
    return 0


def main() -> int:
    global V10_SCORING_CONFIG

    v10_args, remaining_after_v10 = parse_v10_args(sys.argv[1:])
    v9_args, remaining_after_v9 = v9.parse_v9_args(remaining_after_v10)
    v8_args, remaining = v8.parse_v8_args(remaining_after_v9)

    V10_SCORING_CONFIG = NearFieldScoringConfig(
        enabled=not v10_args.disable_near_field_constraints,
        near_field_max_m=v10_args.near_field_max_m,
        support_iou_weight=v10_args.support_iou_weight,
        dynamic_iou_weight=v10_args.dynamic_iou_weight,
        near_field_raw_weight=v10_args.near_field_raw_weight,
        near_field_dynamic_weight=v10_args.near_field_dynamic_weight,
        min_near_field_raw_gain=v10_args.min_near_field_raw_gain,
        min_near_field_dynamic_gain=v10_args.min_near_field_dynamic_gain,
    )
    v9.SUPPORT_EVAL_CONFIG = v9.SupportEvaluationConfig(
        min_common_support_pixels=v9_args.min_common_support_pixels,
        fallback_to_dynamic=not v9_args.disable_support_fallback,
    )
    v9.DISTANCE_BINS = [float(value) for value in v9_args.distance_bins]
    v8.SCORING_CONFIG = v8.ScoringConfig(
        enabled=not v8_args.disable_refinement_prior,
        translation_penalty_weight=v8_args.translation_penalty_weight,
        yaw_penalty_weight=v8_args.yaw_penalty_weight,
        max_total_translation_norm=v8_args.max_total_translation_norm,
        min_score_improvement=v8_args.min_score_improvement,
    )

    output_dir = v9.output_dir_from_args(remaining)
    base.evaluate_alignment = v9.evaluate_alignment
    v9.v8.refine_alignment = refine_alignment
    base.render_for_history = v9.render_for_history

    exit_code = run_base_main_with_partial(v10_args, remaining)

    if exit_code == 0:
        patch_summary(output_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
