#!/usr/bin/env python3
"""Render LiDAR/BEV alignment diagnostics with fair support scoring.

v9 keeps the v8 residual-motion prior, but fixes the metric issue found in
batch_100:

  - before / initial_after / after are evaluated with the same scoring rule
  - common-support scoring is used only when support is large enough
  - candidates with tiny common support fall back to dynamic-only scoring

This prevents a candidate with only a few support pixels from producing a high
masked IoU that is not reflected by the full BEV occupancy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import render_lidar_bev_alignment_v8 as v8


base = v8.base


@dataclass(frozen=True)
class SupportEvaluationConfig:
    min_common_support_pixels: int = 1000
    fallback_to_dynamic: bool = True


SUPPORT_EVAL_CONFIG = SupportEvaluationConfig()
DISTANCE_BINS = [0.0, 8.0, 16.0, 24.0, 32.0]


def parse_v9_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--min-common-support-pixels",
        type=int,
        default=1000,
        help="Use common-support scoring only when support has at least this many pixels.",
    )
    parser.add_argument(
        "--disable-support-fallback",
        action="store_true",
        help="Do not fall back to dynamic-only scoring when common support is too small.",
    )
    parser.add_argument(
        "--distance-bins",
        type=float,
        nargs="+",
        default=[0.0, 8.0, 16.0, 24.0, 32.0],
        help="Distance bin edges in meters for diagnostic IoU fields.",
    )
    args, remaining = parser.parse_known_args(argv)
    if args.min_common_support_pixels < 0:
        parser.error("--min-common-support-pixels must be >= 0.")
    if len(args.distance_bins) < 2:
        parser.error("--distance-bins requires at least two edges.")
    if any(right <= left for left, right in zip(args.distance_bins, args.distance_bins[1:])):
        parser.error("--distance-bins must be strictly increasing.")
    return args, remaining


def distance_grid(shape: tuple[int, int], cfg: base.BevConfig) -> np.ndarray:
    rows, cols = shape
    y_centers = cfg.min_y + (np.arange(rows, dtype=np.float32) + 0.5) / cfg.pixels_per_meter
    x_centers = cfg.min_x + (np.arange(cols, dtype=np.float32) + 0.5) / cfg.pixels_per_meter
    xx, yy = np.meshgrid(x_centers, y_centers)
    return np.sqrt(xx * xx + yy * yy)


def bin_label(left: float, right: float) -> str:
    def format_edge(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "p")

    return f"{format_edge(left)}_{format_edge(right)}m"


def binned_iou_fields(
    prefix: str,
    current_channel: np.ndarray,
    compare_channel: np.ndarray,
    eval_mask: np.ndarray,
    raw_mask: np.ndarray,
    cfg: base.BevConfig,
) -> dict[str, object]:
    dist = distance_grid(current_channel.shape, cfg)
    fields: dict[str, object] = {}
    for left, right in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        include = (dist >= left) & (dist < right)
        label = bin_label(left, right)
        fields[f"{prefix}_iou_{label}"] = base.masked_occupancy_iou(
            current_channel, compare_channel, eval_mask | ~include
        )
        fields[f"{prefix}_iou_raw_{label}"] = base.masked_occupancy_iou(
            current_channel, compare_channel, raw_mask | ~include
        )
        fields[f"{prefix}_pixels_{label}"] = int(include.sum())
    return fields


def evaluate_pair(
    current_channel: np.ndarray,
    compare_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    compare_dynamic_mask: np.ndarray,
    support_cfg: base.SupportConfig,
) -> dict[str, object]:
    dynamic_eval_mask = current_dynamic_mask | compare_dynamic_mask
    dynamic_iou = base.masked_occupancy_iou(current_channel, compare_channel, dynamic_eval_mask)

    support_include = base.common_support_include_mask(current_channel, compare_channel, support_cfg)
    support_pixels = int(support_include.sum())
    support_eval_mask = dynamic_eval_mask | ~support_include
    support_iou = 0.0 if support_pixels == 0 else base.masked_occupancy_iou(
        current_channel, compare_channel, support_eval_mask
    )
    support_valid = bool(support_cfg.enabled and support_pixels >= SUPPORT_EVAL_CONFIG.min_common_support_pixels)

    if support_valid:
        return {
            "iou": support_iou,
            "eval_mask": support_eval_mask,
            "support_pixels": support_pixels,
            "support_valid": True,
            "score_mode": "common_support",
            "dynamic_iou": dynamic_iou,
            "support_iou": support_iou,
            "dynamic_masked_pixels": int(dynamic_eval_mask.sum()),
            "masked_pixels": int(support_eval_mask.sum()),
        }

    if SUPPORT_EVAL_CONFIG.fallback_to_dynamic:
        score_mode = "dynamic_fallback" if support_cfg.enabled else "dynamic_only"
        return {
            "iou": dynamic_iou,
            "eval_mask": dynamic_eval_mask,
            "support_pixels": support_pixels,
            "support_valid": False,
            "score_mode": score_mode,
            "dynamic_iou": dynamic_iou,
            "support_iou": support_iou,
            "dynamic_masked_pixels": int(dynamic_eval_mask.sum()),
            "masked_pixels": int(dynamic_eval_mask.sum()),
        }

    return {
        "iou": support_iou,
        "eval_mask": support_eval_mask,
        "support_pixels": support_pixels,
        "support_valid": False,
        "score_mode": "common_support_below_min",
        "dynamic_iou": dynamic_iou,
        "support_iou": support_iou,
        "dynamic_masked_pixels": int(dynamic_eval_mask.sum()),
        "masked_pixels": int(support_eval_mask.sum()),
    }


def evaluate_alignment(
    lidar: np.ndarray,
    boxes: list[dict],
    current_channel: np.ndarray,
    current_dynamic_mask: np.ndarray,
    cfg: base.BevConfig,
    dynamic_classes: set[str],
    support_cfg: base.SupportConfig,
) -> dict[str, object]:
    bev = base.lidar_to_histogram_features(lidar, cfg)
    channel_index = 1 if cfg.use_ground_plane else 0
    channel = bev[channel_index]
    dynamic_actor_mask = base.dynamic_mask(boxes, channel.shape, cfg, dynamic_classes)
    pair_eval = evaluate_pair(current_channel, channel, current_dynamic_mask, dynamic_actor_mask, support_cfg)
    return {
        "iou": pair_eval["iou"],
        "channel": channel,
        "dynamic_mask": dynamic_actor_mask,
        "eval_mask": pair_eval["eval_mask"],
        "support_pixels": pair_eval["support_pixels"],
        "support_valid": pair_eval["support_valid"],
        "score_mode": pair_eval["score_mode"],
        "dynamic_iou": pair_eval["dynamic_iou"],
        "support_iou": pair_eval["support_iou"],
        "dynamic_masked_pixels": pair_eval["dynamic_masked_pixels"],
        "masked_pixels": pair_eval["masked_pixels"],
    }


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
    current_lidar = base.load_lidar(route_dir / "lidar" / f"{current_frame:04d}.laz")
    current_measurement = base.load_measurement(route_dir / "measurements" / f"{current_frame:04d}.json.gz")
    current_boxes = base.load_boxes(route_dir / "boxes" / f"{current_frame:04d}.json.gz")

    history_frame = current_frame - history_offset
    history_lidar = base.load_lidar(route_dir / "lidar" / f"{history_frame:04d}.laz")
    history_measurement = base.load_measurement(route_dir / "measurements" / f"{history_frame:04d}.json.gz")
    history_boxes = base.load_boxes(route_dir / "boxes" / f"{history_frame:04d}.json.gz")

    aligned_lidar = base.align_lidar(history_lidar, history_measurement, current_measurement)
    aligned_history_boxes = base.align_boxes(history_boxes, history_measurement, current_measurement)

    current_bev = base.lidar_to_histogram_features(current_lidar, cfg)
    history_bev = base.lidar_to_histogram_features(history_lidar, cfg)
    aligned_bev = base.lidar_to_histogram_features(aligned_lidar, cfg)

    channel_index = 1 if cfg.use_ground_plane else 0
    current_channel = current_bev[channel_index]
    history_channel = history_bev[channel_index]
    aligned_channel = aligned_bev[channel_index]

    current_dynamic_mask = base.dynamic_mask(current_boxes, current_channel.shape, cfg, dynamic_classes)
    history_dynamic_mask = base.dynamic_mask(history_boxes, history_channel.shape, cfg, dynamic_classes)
    aligned_history_dynamic_mask = base.dynamic_mask(aligned_history_boxes, aligned_channel.shape, cfg, dynamic_classes)

    before_eval = evaluate_pair(
        current_channel, history_channel, current_dynamic_mask, history_dynamic_mask, support_cfg
    )
    initial_after_eval = evaluate_pair(
        current_channel, aligned_channel, current_dynamic_mask, aligned_history_dynamic_mask, support_cfg
    )

    refined_lidar, _refined_history_boxes, refinement = v8.refine_alignment(
        aligned_lidar=aligned_lidar,
        aligned_boxes=aligned_history_boxes,
        current_channel=current_channel,
        current_dynamic_mask=current_dynamic_mask,
        cfg=cfg,
        dynamic_classes=dynamic_classes,
        support_cfg=support_cfg,
        refinement_cfg=refinement_cfg,
    )
    refined_bev = base.lidar_to_histogram_features(refined_lidar, cfg)
    refined_channel = refined_bev[channel_index]
    refined_history_dynamic_mask = refinement["dynamic_mask"]
    after_eval = evaluate_pair(
        current_channel, refined_channel, current_dynamic_mask, refined_history_dynamic_mask, support_cfg
    )
    refined_eval_mask = after_eval["eval_mask"]

    before_iou_raw = base.occupancy_iou(current_channel, history_channel)
    initial_after_iou_raw = base.occupancy_iou(current_channel, aligned_channel)
    after_iou_raw = base.occupancy_iou(current_channel, refined_channel)

    before_iou = float(before_eval["iou"])
    initial_after_iou = float(initial_after_eval["iou"])
    after_iou = float(after_eval["iou"])
    no_mask = np.zeros_like(current_channel, dtype=bool)
    before_bin_fields = binned_iou_fields(
        "before", current_channel, history_channel, before_eval["eval_mask"], no_mask, cfg
    )
    initial_after_bin_fields = binned_iou_fields(
        "initial_after", current_channel, aligned_channel, initial_after_eval["eval_mask"], no_mask, cfg
    )
    after_bin_fields = binned_iou_fields(
        "after", current_channel, refined_channel, after_eval["eval_mask"], no_mask, cfg
    )

    visualization_mask = before_eval["eval_mask"] | refined_eval_mask
    current_channel_vis = np.where(visualization_mask, 0.0, current_channel)
    current_before_vis = np.where(before_eval["eval_mask"], 0.0, current_channel)
    current_after_vis = np.where(refined_eval_mask, 0.0, current_channel)
    history_channel_vis = np.where(before_eval["eval_mask"], 0.0, history_channel)
    aligned_channel_vis = np.where(refined_eval_mask, 0.0, refined_channel)

    current_gray = np.repeat(base.to_uint8_grayscale(current_channel_vis)[..., None], 3, axis=2)
    before_overlay = base.make_overlay(current_before_vis, history_channel_vis, color)
    after_overlay = base.make_overlay(current_after_vis, aligned_channel_vis, color)
    comparison = base.add_separator([current_gray, before_overlay, after_overlay])
    base.save_image(
        comparison,
        output_dir / f"frame_{current_frame:04d}" / f"history_{history_offset:02d}.png",
    )

    result = {
        "current_frame": current_frame,
        "history_frame": history_frame,
        "history_offset": history_offset,
        "before_iou_raw": before_iou_raw,
        "initial_after_iou_raw": initial_after_iou_raw,
        "after_iou_raw": after_iou_raw,
        "initial_iou_gain_raw": initial_after_iou_raw - before_iou_raw,
        "iou_gain_raw": after_iou_raw - before_iou_raw,
        "before_iou": before_iou,
        "initial_after_iou": initial_after_iou,
        "after_iou": after_iou,
        "initial_iou_gain": initial_after_iou - before_iou,
        "iou_gain": after_iou - before_iou,
        "before_iou_dynamic": float(before_eval["dynamic_iou"]),
        "initial_after_iou_dynamic": float(initial_after_eval["dynamic_iou"]),
        "after_iou_dynamic": float(after_eval["dynamic_iou"]),
        "before_iou_support": float(before_eval["support_iou"]),
        "initial_after_iou_support": float(initial_after_eval["support_iou"]),
        "after_iou_support": float(after_eval["support_iou"]),
        "before_score_mode": str(before_eval["score_mode"]),
        "initial_after_score_mode": str(initial_after_eval["score_mode"]),
        "after_score_mode": str(after_eval["score_mode"]),
        "before_support_pixels": int(before_eval["support_pixels"]),
        "initial_after_support_pixels": int(initial_after_eval["support_pixels"]),
        "support_pixels": int(after_eval["support_pixels"]),
        "before_support_valid": bool(before_eval["support_valid"]),
        "initial_after_support_valid": bool(initial_after_eval["support_valid"]),
        "after_support_valid": bool(after_eval["support_valid"]),
        "current_dynamic_pixels": int(current_dynamic_mask.sum()),
        "history_dynamic_pixels": int(history_dynamic_mask.sum()),
        "aligned_history_dynamic_pixels": int(aligned_history_dynamic_mask.sum()),
        "refined_history_dynamic_pixels": int(refined_history_dynamic_mask.sum()),
        "before_masked_pixels": int(before_eval["masked_pixels"]),
        "initial_after_masked_pixels": int(initial_after_eval["masked_pixels"]),
        "after_masked_pixels": int(after_eval["masked_pixels"]),
        "refinement_enabled": refinement_cfg.enabled,
        "refinement_used_fallback": bool(refinement["used_fallback"]),
        "refinement_dx": float(refinement["translation"][0]),
        "refinement_dy": float(refinement["translation"][1]),
        "refinement_yaw_deg": float(np.rad2deg(refinement["yaw"])),
        "refinement_candidates": int(refinement["searched_candidates"]),
        "refinement_score": float(refinement["score"]),
        "initial_refinement_score": float(refinement["initial_score"]),
        "refinement_penalty": float(refinement["penalty"]),
        "refinement_translation_norm": float(refinement["translation_norm"]),
        "refinement_rejected_candidates": int(refinement["rejected_candidates"]),
        "coarse_refinement_dx": float(refinement["coarse_translation"][0]),
        "coarse_refinement_dy": float(refinement["coarse_translation"][1]),
        "coarse_refinement_yaw_deg": float(np.rad2deg(refinement["coarse_yaw"])),
        "coarse_refinement_iou": float(refinement["coarse_iou"]),
        "coarse_refinement_score": float(refinement["coarse_score"]),
        "coarse_refinement_penalty": float(refinement["coarse_penalty"]),
        "coarse_refinement_translation_norm": float(refinement["coarse_translation_norm"]),
        "coarse_refinement_candidates": int(refinement["coarse_candidates"]),
        "coarse_refinement_rejected_candidates": int(refinement["coarse_rejected_candidates"]),
        "fine_refinement_dx": float(refinement["fine_translation"][0]),
        "fine_refinement_dy": float(refinement["fine_translation"][1]),
        "fine_refinement_yaw_deg": float(np.rad2deg(refinement["fine_yaw"])),
        "fine_refinement_iou": float(refinement["fine_iou"]),
        "fine_refinement_score": float(refinement["fine_score"]),
        "fine_refinement_penalty": float(refinement["fine_penalty"]),
        "fine_refinement_translation_norm": float(refinement["fine_translation_norm"]),
        "fine_refinement_candidates": int(refinement["fine_candidates"]),
        "fine_refinement_rejected_candidates": int(refinement["fine_rejected_candidates"]),
    }
    result.update(before_bin_fields)
    result.update(initial_after_bin_fields)
    result.update(after_bin_fields)
    for left, right in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        label = bin_label(left, right)
        result[f"iou_gain_{label}"] = result[f"after_iou_{label}"] - result[f"before_iou_{label}"]
        result[f"iou_gain_raw_{label}"] = result[f"after_iou_raw_{label}"] - result[f"before_iou_raw_{label}"]
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
            "script_version": "v9",
            "refinement_prior_enabled": v8.SCORING_CONFIG.enabled,
            "translation_penalty_weight": v8.SCORING_CONFIG.translation_penalty_weight,
            "yaw_penalty_weight": v8.SCORING_CONFIG.yaw_penalty_weight,
            "max_total_translation_norm": v8.SCORING_CONFIG.max_total_translation_norm,
            "min_score_improvement": v8.SCORING_CONFIG.min_score_improvement,
            "min_common_support_pixels": SUPPORT_EVAL_CONFIG.min_common_support_pixels,
            "support_fallback_to_dynamic": SUPPORT_EVAL_CONFIG.fallback_to_dynamic,
            "distance_bins": DISTANCE_BINS,
        }
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    global DISTANCE_BINS, SUPPORT_EVAL_CONFIG

    v9_args, remaining_after_v9 = parse_v9_args(sys.argv[1:])
    v8_args, remaining = v8.parse_v8_args(remaining_after_v9)

    SUPPORT_EVAL_CONFIG = SupportEvaluationConfig(
        min_common_support_pixels=v9_args.min_common_support_pixels,
        fallback_to_dynamic=not v9_args.disable_support_fallback,
    )
    DISTANCE_BINS = [float(value) for value in v9_args.distance_bins]
    v8.SCORING_CONFIG = v8.ScoringConfig(
        enabled=not v8_args.disable_refinement_prior,
        translation_penalty_weight=v8_args.translation_penalty_weight,
        yaw_penalty_weight=v8_args.yaw_penalty_weight,
        max_total_translation_norm=v8_args.max_total_translation_norm,
        min_score_improvement=v8_args.min_score_improvement,
    )

    output_dir = output_dir_from_args(remaining)
    base.evaluate_alignment = evaluate_alignment
    base.refine_alignment = v8.refine_alignment
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
