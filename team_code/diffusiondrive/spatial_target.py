"""Spatial trajectory target resampling contract."""

from __future__ import annotations

from typing import Optional

import numpy as np


SPATIAL_TARGET_SCHEMA = "arc_length_stable_extrapolation_v2"
SPATIAL_TARGET_MIN_SEGMENT_LENGTH_METERS = 1e-3
SPATIAL_TARGET_MIN_EXTRAPOLATION_DISPLACEMENT_METERS = 0.5
SPATIAL_TARGET_EXTRAPOLATION_STRATEGY = "trailing_displacement_or_route_condition"


def spatial_target_metadata() -> dict:
    return {
        "schema": SPATIAL_TARGET_SCHEMA,
        "min_segment_length_meters": SPATIAL_TARGET_MIN_SEGMENT_LENGTH_METERS,
        "extrapolation_strategy": SPATIAL_TARGET_EXTRAPOLATION_STRATEGY,
        "min_extrapolation_displacement_meters": (
            SPATIAL_TARGET_MIN_EXTRAPOLATION_DISPLACEMENT_METERS
        ),
    }


def deduplicate_polyline(
    points: np.ndarray,
    min_segment_length: float = SPATIAL_TARGET_MIN_SEGMENT_LENGTH_METERS,
) -> np.ndarray:
    points = _validate_points(points)
    if min_segment_length < 0:
        raise ValueError(f"min_segment_length must be >= 0, got {min_segment_length}")
    if points.shape[0] <= 1:
        return points

    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_segment_length:
            kept.append(point)
    return np.asarray(kept, dtype=np.float64)


def resample_polyline_by_distance(
    points: np.ndarray,
    distances: np.ndarray,
    fallback_direction: np.ndarray,
    *,
    min_extrapolation_displacement: float = SPATIAL_TARGET_MIN_EXTRAPOLATION_DISPLACEMENT_METERS,
    diagnostics: Optional[dict] = None,
) -> np.ndarray:
    points = _validate_points(points)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError(f"distances must have shape [N], got {distances.shape}")
    if np.any(distances < 0):
        raise ValueError("distances must be non-negative")
    if min_extrapolation_displacement < 0:
        raise ValueError(
            "min_extrapolation_displacement must be >= 0, "
            f"got {min_extrapolation_displacement}"
        )

    if points.shape[0] == 0:
        points = np.zeros((1, 2), dtype=np.float64)

    fallback = normalize_direction(fallback_direction)
    if points.shape[0] == 1:
        samples = points[0][None, :] + distances[:, None] * fallback[None, :]
        _update_diagnostics(
            diagnostics,
            points=points,
            segment_lengths=np.empty(0, dtype=np.float64),
            path_length=0.0,
            extrapolated_count=len(distances),
            direction=fallback,
            direction_source="route_condition",
            direction_baseline=0.0,
            fallback=fallback,
            min_extrapolation_displacement=min_extrapolation_displacement,
        )
        return samples

    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    valid = segment_lengths > 1e-6
    if not np.any(valid):
        samples = points[0][None, :] + distances[:, None] * fallback[None, :]
        _update_diagnostics(
            diagnostics,
            points=points,
            segment_lengths=segment_lengths,
            path_length=0.0,
            extrapolated_count=len(distances),
            direction=fallback,
            direction_source="route_condition",
            direction_baseline=0.0,
            fallback=fallback,
            min_extrapolation_displacement=min_extrapolation_displacement,
        )
        return samples

    segments = segments[valid]
    segment_lengths = segment_lengths[valid]
    start_points = points[:-1][valid]
    end_points = points[1:][valid]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    path_length = float(cumulative[-1])
    direction, direction_source, direction_baseline = _stable_extrapolation_direction(
        points,
        segments,
        fallback,
        min_extrapolation_displacement,
    )

    samples = []
    extrapolated_count = 0
    for distance in distances:
        if distance <= path_length:
            segment_idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
            segment_idx = min(segment_idx, len(segment_lengths) - 1)
            local = (distance - cumulative[segment_idx]) / segment_lengths[segment_idx]
            samples.append(
                start_points[segment_idx]
                + local * (end_points[segment_idx] - start_points[segment_idx])
            )
        else:
            extrapolated_count += 1
            samples.append(end_points[-1] + (distance - path_length) * direction)

    _update_diagnostics(
        diagnostics,
        points=points,
        segment_lengths=segment_lengths,
        path_length=path_length,
        extrapolated_count=extrapolated_count,
        direction=direction,
        direction_source=direction_source,
        direction_baseline=direction_baseline,
        fallback=fallback,
        min_extrapolation_displacement=min_extrapolation_displacement,
    )
    return np.asarray(samples, dtype=np.float64)


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    if direction.ndim != 1 or direction.shape[0] < 2:
        raise ValueError(f"direction must contain at least two values, got {direction.shape}")
    direction = direction[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def _stable_extrapolation_direction(
    points: np.ndarray,
    segments: np.ndarray,
    fallback: np.ndarray,
    min_displacement: float,
) -> tuple[np.ndarray, str, float]:
    if min_displacement <= 0:
        segment = segments[-1]
        return normalize_direction(segment), "last_segment", float(np.linalg.norm(segment))

    endpoint = points[-1]
    max_displacement = 0.0
    for point in points[-2::-1]:
        displacement = endpoint - point
        displacement_norm = float(np.linalg.norm(displacement))
        max_displacement = max(max_displacement, displacement_norm)
        if displacement_norm >= min_displacement:
            return normalize_direction(displacement), "trailing_displacement", displacement_norm
    return fallback, "route_condition", max_displacement


def _update_diagnostics(
    diagnostics: Optional[dict],
    *,
    points: np.ndarray,
    segment_lengths: np.ndarray,
    path_length: float,
    extrapolated_count: int,
    direction: np.ndarray,
    direction_source: str,
    direction_baseline: float,
    fallback: np.ndarray,
    min_extrapolation_displacement: float,
) -> None:
    if diagnostics is None:
        return
    endpoint_displacement = float(np.linalg.norm(points[-1] - points[0]))
    diagnostics.update(
        {
            "polyline_point_count": int(points.shape[0]),
            "observed_path_length_meters": float(path_length),
            "observed_endpoint_displacement_meters": endpoint_displacement,
            "max_segment_length_meters": (
                float(np.max(segment_lengths)) if segment_lengths.size else 0.0
            ),
            "last_segment_length_meters": (
                float(segment_lengths[-1]) if segment_lengths.size else 0.0
            ),
            "extrapolated_pose_count": int(extrapolated_count),
            "extrapolation_direction": direction.tolist(),
            "extrapolation_direction_source": direction_source,
            "extrapolation_direction_baseline_meters": float(direction_baseline),
            "fallback_direction": fallback.tolist(),
            "min_extrapolation_displacement_meters": float(
                min_extrapolation_displacement
            ),
            "extrapolation_fallback_alignment": float(np.dot(direction, fallback)),
        }
    )


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape [N,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")
    return points
