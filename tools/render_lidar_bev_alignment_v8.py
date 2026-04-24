#!/usr/bin/env python3
"""Render LiDAR/BEV alignment diagnostics with a residual-motion prior.

This v8 wrapper reuses v7 for data loading, BEV rendering, dynamic masking, and
common-support scoring. It replaces only the refinement objective:

  score = common_support_iou - translation_penalty_weight * (dx^2 + dy^2)

It also applies an optional hard cap on total residual translation. The goal is
to keep the useful common-support mask from v7 while avoiding large, boundary
seeking residual shifts that improve local IoU but are unlikely to be a real
alignment correction.

Example:
  python tools/render_lidar_bev_alignment_v8.py \
    --root-dir data/... \
    --route-name Town12_Rep0_2488_0_route0_11_08_02_55_17 \
    --frame 35 \
    --history 5 \
    --coarse-refine-xy-range 4.0 \
    --fine-refine-xy-range 0.5 \
    --output-dir /tmp/lidar_bev_v8_case0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import render_lidar_bev_alignment_v7 as base


ORIGINAL_RENDER_FOR_HISTORY = base.render_for_history
LAST_REFINEMENT: dict[str, object] | None = None


@dataclass(frozen=True)
class ScoringConfig:
    enabled: bool = True
    translation_penalty_weight: float = 0.01
    yaw_penalty_weight: float = 0.0
    max_total_translation_norm: float = 2.0
    min_score_improvement: float = 0.0


SCORING_CONFIG = ScoringConfig()


def parse_v8_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--disable-refinement-prior",
        action="store_true",
        help="Disable the v8 residual-motion prior and fall back to v7 scoring.",
    )
    parser.add_argument(
        "--translation-penalty-weight",
        type=float,
        default=0.01,
        help="Penalty subtracted from score per squared meter of total residual translation.",
    )
    parser.add_argument(
        "--yaw-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty subtracted from score per squared degree of total residual yaw.",
    )
    parser.add_argument(
        "--max-total-translation-norm",
        type=float,
        default=2.0,
        help="Reject candidates whose total residual translation norm exceeds this many meters. Use <=0 to disable.",
    )
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=0.0,
        help="Minimum score improvement required to replace the current best refinement candidate.",
    )
    args, remaining = parser.parse_known_args(argv)

    if args.translation_penalty_weight < 0:
        parser.error("--translation-penalty-weight must be >= 0.")
    if args.yaw_penalty_weight < 0:
        parser.error("--yaw-penalty-weight must be >= 0.")
    if args.min_score_improvement < 0:
        parser.error("--min-score-improvement must be >= 0.")

    return args, remaining


def refinement_cost(translation: np.ndarray, yaw: float) -> float:
    return float(translation[0] ** 2 + translation[1] ** 2 + np.rad2deg(yaw) ** 2)


def total_translation_norm(translation: np.ndarray) -> float:
    return float(np.sqrt(float(translation[0] ** 2 + translation[1] ** 2)))


def refinement_score(
    iou: float,
    total_translation: np.ndarray,
    total_yaw: float,
    scoring_cfg: ScoringConfig,
) -> tuple[float, float, float]:
    norm = total_translation_norm(total_translation)
    if not scoring_cfg.enabled:
        return float(iou), 0.0, norm

    yaw_deg = float(np.rad2deg(total_yaw))
    penalty = (
        scoring_cfg.translation_penalty_weight * float(total_translation[0] ** 2 + total_translation[1] ** 2)
        + scoring_cfg.yaw_penalty_weight * yaw_deg * yaw_deg
    )
    return float(iou - penalty), float(penalty), norm


def is_candidate_allowed(total_translation: np.ndarray, scoring_cfg: ScoringConfig) -> bool:
    if not scoring_cfg.enabled:
        return True
    if scoring_cfg.max_total_translation_norm <= 0:
        return True
    return total_translation_norm(total_translation) <= scoring_cfg.max_total_translation_norm + 1e-9


def is_better_candidate(
    candidate: dict[str, object],
    best: dict[str, object],
    scoring_cfg: ScoringConfig,
) -> bool:
    min_delta = scoring_cfg.min_score_improvement if scoring_cfg.enabled else 0.0
    candidate_score = float(candidate["score"])
    best_score = float(best["score"])
    if candidate_score > best_score + min_delta + 1e-9:
        return True
    if abs(candidate_score - best_score) > min_delta + 1e-9:
        return False
    candidate_iou = float(candidate["iou"])
    best_iou = float(best["iou"])
    if candidate_iou > best_iou + 1e-9:
        return True
    if abs(candidate_iou - best_iou) > 1e-9:
        return False
    return float(candidate["cost"]) < float(best["cost"])


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
    scoring_cfg: ScoringConfig,
) -> dict[str, object]:
    base_eval = base.evaluate_alignment(
        lidar=base_lidar,
        boxes=base_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
    )
    base_score, base_penalty, base_norm = refinement_score(
        float(base_eval["iou"]), prior_translation, prior_yaw, scoring_cfg
    )
    best = {
        "iou": base_eval["iou"],
        "score": base_score,
        "penalty": base_penalty,
        "translation_norm": base_norm,
        "translation": np.zeros(3, dtype=np.float32),
        "total_translation": prior_translation.copy(),
        "yaw": 0.0,
        "total_yaw": prior_yaw,
        "lidar": base_lidar,
        "boxes": base_boxes,
        "channel": base_eval["channel"],
        "dynamic_mask": base_eval["dynamic_mask"],
        "eval_mask": base_eval["eval_mask"],
        "support_pixels": base_eval["support_pixels"],
        "searched_candidates": 1,
        "rejected_candidates": 0,
        "cost": refinement_cost(prior_translation, prior_yaw),
    }

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
                if not is_candidate_allowed(total_translation, scoring_cfg):
                    rejected_candidates += 1
                    continue
                if dx == 0.0 and dy == 0.0 and yaw == 0.0:
                    continue

                candidate_lidar = base.algin_lidar(base_lidar, translation, yaw)
                candidate_boxes = base.apply_transform_to_boxes(base_boxes, translation, yaw)
                candidate_eval = base.evaluate_alignment(
                    lidar=candidate_lidar,
                    boxes=candidate_boxes,
                    current_channel=current_channel,
                    current_dynamic_mask=current_dynamic_mask,
                    cfg=cfg,
                    dynamic_classes=dynamic_classes,
                    support_cfg=support_cfg,
                )
                candidate_score, candidate_penalty, candidate_norm = refinement_score(
                    float(candidate_eval["iou"]), total_translation, total_yaw, scoring_cfg
                )
                candidate = {
                    "iou": candidate_eval["iou"],
                    "score": candidate_score,
                    "penalty": candidate_penalty,
                    "translation_norm": candidate_norm,
                    "translation": translation.copy(),
                    "total_translation": total_translation.copy(),
                    "yaw": yaw,
                    "total_yaw": total_yaw,
                    "lidar": candidate_lidar,
                    "boxes": candidate_boxes,
                    "channel": candidate_eval["channel"],
                    "dynamic_mask": candidate_eval["dynamic_mask"],
                    "eval_mask": candidate_eval["eval_mask"],
                    "support_pixels": candidate_eval["support_pixels"],
                    "searched_candidates": searched_candidates,
                    "rejected_candidates": rejected_candidates,
                    "cost": refinement_cost(total_translation, total_yaw),
                }

                if is_better_candidate(candidate, best, scoring_cfg):
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
    global LAST_REFINEMENT

    initial_eval = base.evaluate_alignment(
        lidar=aligned_lidar,
        boxes=aligned_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
    )
    initial_iou = float(initial_eval["iou"])
    zero_translation = np.zeros(3, dtype=np.float32)
    initial_score, initial_penalty, initial_norm = refinement_score(
        initial_iou, zero_translation, 0.0, SCORING_CONFIG
    )

    best = {
        "iou": initial_iou,
        "score": initial_score,
        "penalty": initial_penalty,
        "translation_norm": initial_norm,
        "translation": zero_translation.copy(),
        "yaw": 0.0,
        "lidar": aligned_lidar,
        "boxes": aligned_boxes,
        "channel": initial_eval["channel"],
        "dynamic_mask": initial_eval["dynamic_mask"],
        "eval_mask": initial_eval["eval_mask"],
        "support_pixels": initial_eval["support_pixels"],
        "searched_candidates": 1,
        "rejected_candidates": 0,
        "used_fallback": False,
        "cost": 0.0,
        "initial_score": initial_score,
        "initial_penalty": initial_penalty,
        "coarse_translation": zero_translation.copy(),
        "coarse_yaw": 0.0,
        "coarse_iou": initial_iou,
        "coarse_score": initial_score,
        "coarse_penalty": initial_penalty,
        "coarse_translation_norm": initial_norm,
        "coarse_candidates": 1,
        "coarse_rejected_candidates": 0,
        "fine_translation": zero_translation.copy(),
        "fine_yaw": 0.0,
        "fine_iou": initial_iou,
        "fine_score": initial_score,
        "fine_penalty": initial_penalty,
        "fine_translation_norm": initial_norm,
        "fine_candidates": 1,
        "fine_rejected_candidates": 0,
    }

    if not refinement_cfg.enabled:
        LAST_REFINEMENT = best
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
        scoring_cfg=SCORING_CONFIG,
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
        scoring_cfg=SCORING_CONFIG,
    )
    total_translation, total_yaw = base.compose_residual_transform(
        coarse["translation"], coarse["yaw"], fine["translation"], fine["yaw"]
    )
    best = {
        "iou": fine["iou"],
        "score": fine["score"],
        "penalty": fine["penalty"],
        "translation_norm": fine["translation_norm"],
        "translation": total_translation,
        "yaw": total_yaw,
        "lidar": fine["lidar"],
        "boxes": fine["boxes"],
        "channel": fine["channel"],
        "dynamic_mask": fine["dynamic_mask"],
        "eval_mask": fine["eval_mask"],
        "support_pixels": fine["support_pixels"],
        "searched_candidates": int(coarse["searched_candidates"] + fine["searched_candidates"]),
        "rejected_candidates": int(coarse["rejected_candidates"] + fine["rejected_candidates"]),
        "used_fallback": False,
        "cost": refinement_cost(total_translation, total_yaw),
        "initial_score": initial_score,
        "initial_penalty": initial_penalty,
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
    if refinement_cfg.fallback_if_worse and float(best["score"]) < initial_score:
        best = {
            **best,
            "iou": initial_iou,
            "score": initial_score,
            "penalty": initial_penalty,
            "translation_norm": initial_norm,
            "translation": zero_translation.copy(),
            "yaw": 0.0,
            "lidar": aligned_lidar,
            "boxes": aligned_boxes,
            "channel": initial_eval["channel"],
            "dynamic_mask": initial_eval["dynamic_mask"],
            "eval_mask": initial_eval["eval_mask"],
            "support_pixels": initial_eval["support_pixels"],
            "used_fallback": True,
            "cost": 0.0,
        }

    LAST_REFINEMENT = best
    return best["lidar"], best["boxes"], best


def render_for_history(
    route_dir: Path,
    current_frame: int,
    history_offset: int,
    output_dir: Path,
    cfg: base.BevConfig,
    color: np.ndarray,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
    refinement_cfg: base.RefinementConfig,
) -> dict[str, object]:
    result = ORIGINAL_RENDER_FOR_HISTORY(
        route_dir=route_dir,
        current_frame=current_frame,
        history_offset=history_offset,
        output_dir=output_dir,
        cfg=cfg,
        color=color,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
        refinement_cfg=refinement_cfg,
    )

    if LAST_REFINEMENT is None:
        raise RuntimeError("v8 refinement diagnostics were not recorded.")
    refinement = LAST_REFINEMENT

    result.update(
        {
            "refinement_score": float(refinement["score"]),
            "initial_refinement_score": float(refinement["initial_score"]),
            "refinement_penalty": float(refinement["penalty"]),
            "refinement_translation_norm": float(refinement["translation_norm"]),
            "refinement_rejected_candidates": int(refinement["rejected_candidates"]),
            "coarse_refinement_score": float(refinement["coarse_score"]),
            "coarse_refinement_penalty": float(refinement["coarse_penalty"]),
            "coarse_refinement_translation_norm": float(refinement["coarse_translation_norm"]),
            "coarse_refinement_rejected_candidates": int(refinement["coarse_rejected_candidates"]),
            "fine_refinement_score": float(refinement["fine_score"]),
            "fine_refinement_penalty": float(refinement["fine_penalty"]),
            "fine_refinement_translation_norm": float(refinement["fine_translation_norm"]),
            "fine_refinement_rejected_candidates": int(refinement["fine_rejected_candidates"]),
        }
    )
    return result


def output_dir_from_args(argv: list[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith("--output-dir="):
            return Path(arg.split("=", 1)[1])
    return None


def patch_summary(output_dir: Path | None) -> None:
    if output_dir is None:
        return
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.setdefault("config", {}).update(
        {
            "script_version": "v8",
            "refinement_prior_enabled": SCORING_CONFIG.enabled,
            "translation_penalty_weight": SCORING_CONFIG.translation_penalty_weight,
            "yaw_penalty_weight": SCORING_CONFIG.yaw_penalty_weight,
            "max_total_translation_norm": SCORING_CONFIG.max_total_translation_norm,
            "min_score_improvement": SCORING_CONFIG.min_score_improvement,
        }
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    global SCORING_CONFIG

    v8_args, remaining = parse_v8_args(sys.argv[1:])
    SCORING_CONFIG = ScoringConfig(
        enabled=not v8_args.disable_refinement_prior,
        translation_penalty_weight=v8_args.translation_penalty_weight,
        yaw_penalty_weight=v8_args.yaw_penalty_weight,
        max_total_translation_norm=v8_args.max_total_translation_norm,
        min_score_improvement=v8_args.min_score_improvement,
    )

    output_dir = output_dir_from_args(remaining)
    base.refine_alignment = refine_alignment
    base.render_for_history = render_for_history

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        exit_code = base.main()
    finally:
        sys.argv = old_argv

    if exit_code == 0:
        patch_summary(output_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
