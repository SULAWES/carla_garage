"""Shared LiDAR contract statistics and bounded diagnostic dumps."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


LIDAR_CONTRACT_SCHEMA = "diffusiondrive_lidar_contract_v1"
SUMMARY_METRICS = (
    "point_count",
    "finite_point_count",
    "in_bev_point_count",
    "in_bev_ratio",
    "height_eligible_point_count",
    "height_eligible_point_ratio",
    "below_split_count",
    "below_split_ratio",
    "above_split_count",
    "above_split_ratio",
    "below_channel_point_count",
    "below_channel_point_ratio",
    "above_channel_point_count",
    "above_channel_point_ratio",
    "model_point_count",
    "model_point_ratio",
    "front_ratio",
    "back_ratio",
    "left_ratio",
    "right_ratio",
    "z_p05",
    "z_p50",
    "z_p95",
    "bev_nonzero_ratio",
    "bev_mean",
    "bev_mean_nonzero",
    "bev_max",
    "bev_saturation_ratio",
)
CHANNEL_SUMMARY_METRICS = (
    "nonzero_ratio",
    "mean",
    "mean_nonzero",
    "max",
    "saturation_ratio",
)


def _config_value(config: Any, *names: str, default: float) -> float:
    for name in names:
        if hasattr(config, name):
            return float(getattr(config, name))
    return float(default)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return _optional_float(np.percentile(values, percentile))


def make_json_safe(value: Any) -> Any:
    """Convert NumPy/Torch-like values and non-finite floats to strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def compute_lidar_contract_record(
    points: Any | None,
    bev: Any | None,
    config: Any,
    *,
    source: str,
    stage: str,
    step: int | None = None,
    event: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute comparable point-cloud and BEV statistics.

    CARLA coordinates are interpreted as x forward and y right. Point-region
    ratios use points that pass the same XY, max-height, and above-split filters
    as the current one-channel model input.
    """

    min_x = _config_value(config, "min_x", "lidar_min_x", default=-32.0)
    max_x = _config_value(config, "max_x", "lidar_max_x", default=32.0)
    min_y = _config_value(config, "min_y", "lidar_min_y", default=-32.0)
    max_y = _config_value(config, "max_y", "lidar_max_y", default=32.0)
    split_height = _config_value(config, "lidar_split_height", default=0.2)
    max_height = _config_value(config, "max_height_lidar", default=100.0)
    use_ground_plane = bool(getattr(config, "use_ground_plane", False))

    record: dict[str, Any] = {
        "schema": LIDAR_CONTRACT_SCHEMA,
        "source": str(source),
        "stage": str(stage),
        "step": int(step) if step is not None else None,
        "event": str(event) if event is not None else None,
        "bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "lidar_split_height": split_height,
            "max_height_lidar": max_height,
            "use_ground_plane": use_ground_plane,
        },
    }

    if points is None:
        point_array = np.empty((0, 3), dtype=np.float32)
        raw_point_count = 0
    else:
        point_array = _as_numpy(points)
        if point_array.ndim != 2 or point_array.shape[1] < 3:
            raise ValueError(f"LiDAR points must have shape (N, >=3), got {point_array.shape}.")
        raw_point_count = int(point_array.shape[0])
        point_array = np.asarray(point_array[:, :3], dtype=np.float64)

    finite_mask = np.all(np.isfinite(point_array), axis=1)
    finite_points = point_array[finite_mask]
    finite_count = int(finite_points.shape[0])
    if finite_count:
        z_values = finite_points[:, 2]
        in_bev_mask = (
            (finite_points[:, 0] >= min_x)
            & (finite_points[:, 0] <= max_x)
            & (finite_points[:, 1] >= min_y)
            & (finite_points[:, 1] <= max_y)
        )
        height_eligible_mask = finite_points[:, 2] < max_height
        below_mask = height_eligible_mask & (finite_points[:, 2] <= split_height)
        above_mask = height_eligible_mask & (finite_points[:, 2] > split_height)
        model_mask = in_bev_mask & (height_eligible_mask if use_ground_plane else above_mask)
    else:
        z_values = np.empty((0,), dtype=np.float64)
        in_bev_mask = np.zeros((0,), dtype=bool)
        height_eligible_mask = np.zeros((0,), dtype=bool)
        below_mask = np.zeros((0,), dtype=bool)
        above_mask = np.zeros((0,), dtype=bool)
        model_mask = np.zeros((0,), dtype=bool)

    in_bev_count = int(np.count_nonzero(in_bev_mask))
    height_eligible_count = int(np.count_nonzero(height_eligible_mask))
    below_count = int(np.count_nonzero(below_mask))
    above_count = int(np.count_nonzero(above_mask))
    below_channel_count = int(np.count_nonzero(in_bev_mask & below_mask))
    above_channel_count = int(np.count_nonzero(in_bev_mask & above_mask))
    channel_point_count = below_channel_count + above_channel_count
    model_points = finite_points[model_mask]
    model_count = int(model_points.shape[0])

    record.update({
        "point_count": raw_point_count,
        "finite_point_count": finite_count,
        "invalid_point_count": raw_point_count - finite_count,
        "in_bev_point_count": in_bev_count,
        "in_bev_ratio": _ratio(in_bev_count, finite_count),
        "height_eligible_point_count": height_eligible_count,
        "height_eligible_point_ratio": _ratio(height_eligible_count, finite_count),
        "below_split_count": below_count,
        "below_split_ratio": _ratio(below_count, finite_count),
        "above_split_count": above_count,
        "above_split_ratio": _ratio(above_count, finite_count),
        "below_channel_point_count": below_channel_count,
        "below_channel_point_ratio": _ratio(below_channel_count, channel_point_count),
        "above_channel_point_count": above_channel_count,
        "above_channel_point_ratio": _ratio(above_channel_count, channel_point_count),
        "model_point_count": model_count,
        "model_point_ratio": _ratio(model_count, finite_count),
        "z_min": _optional_float(np.min(z_values)) if z_values.size else None,
        "z_p01": _percentile(z_values, 1),
        "z_p05": _percentile(z_values, 5),
        "z_p25": _percentile(z_values, 25),
        "z_p50": _percentile(z_values, 50),
        "z_p75": _percentile(z_values, 75),
        "z_p95": _percentile(z_values, 95),
        "z_p99": _percentile(z_values, 99),
        "z_max": _optional_float(np.max(z_values)) if z_values.size else None,
    })

    if model_count:
        front_count = int(np.count_nonzero(model_points[:, 0] >= 0.0))
        back_count = model_count - front_count
        right_count = int(np.count_nonzero(model_points[:, 1] >= 0.0))
        left_count = model_count - right_count
    else:
        front_count = back_count = left_count = right_count = 0
    record.update({
        "front_count": front_count,
        "front_ratio": _ratio(front_count, model_count),
        "back_count": back_count,
        "back_ratio": _ratio(back_count, model_count),
        "left_count": left_count,
        "left_ratio": _ratio(left_count, model_count),
        "right_count": right_count,
        "right_ratio": _ratio(right_count, model_count),
    })

    if bev is None:
        record.update({
            "bev_shape": None,
            "bev_nonzero_count": None,
            "bev_nonzero_ratio": None,
            "bev_mean": None,
            "bev_mean_nonzero": None,
            "bev_max": None,
            "bev_saturation_count": None,
            "bev_saturation_ratio": None,
            "bev_channel_stats": [],
        })
    else:
        bev_array = np.asarray(_as_numpy(bev), dtype=np.float64)
        if bev_array.ndim == 4 and bev_array.shape[0] == 1:
            bev_array = bev_array[0]
        if bev_array.ndim == 2:
            bev_array = bev_array[None, ...]
        if bev_array.ndim != 3:
            raise ValueError(f"LiDAR BEV must have shape (C, H, W) or (1, C, H, W), got {bev_array.shape}.")
        finite_bev = np.where(np.isfinite(bev_array), bev_array, 0.0)
        nonzero = finite_bev > 0.0
        saturated = finite_bev >= (1.0 - 1e-6)
        nonzero_count = int(np.count_nonzero(nonzero))
        saturation_count = int(np.count_nonzero(saturated))
        value_count = int(finite_bev.size)
        channel_stats = []
        for channel_index, channel in enumerate(finite_bev):
            channel_nonzero = channel > 0.0
            channel_nonzero_count = int(np.count_nonzero(channel_nonzero))
            channel_stats.append({
                "channel": channel_index,
                "nonzero_ratio": _ratio(channel_nonzero_count, int(channel.size)),
                "mean": _optional_float(np.mean(channel)),
                "mean_nonzero": (
                    _optional_float(np.mean(channel[channel_nonzero])) if channel_nonzero_count else 0.0
                ),
                "max": _optional_float(np.max(channel)) if channel.size else None,
                "saturation_ratio": _ratio(int(np.count_nonzero(channel >= (1.0 - 1e-6))), int(channel.size)),
            })
        record.update({
            "bev_shape": [int(value) for value in finite_bev.shape],
            "bev_nonzero_count": nonzero_count,
            "bev_nonzero_ratio": _ratio(nonzero_count, value_count),
            "bev_mean": _optional_float(np.mean(finite_bev)),
            "bev_mean_nonzero": (
                _optional_float(np.mean(finite_bev[nonzero])) if nonzero_count else 0.0
            ),
            "bev_max": _optional_float(np.max(finite_bev)) if value_count else None,
            "bev_saturation_count": saturation_count,
            "bev_saturation_ratio": _ratio(saturation_count, value_count),
            "bev_channel_stats": channel_stats,
        })

    if context:
        record["context"] = make_json_safe(context)
    return make_json_safe(record)


class LidarContractAccumulator:
    """Streaming metric accumulator with per-stage percentile summaries."""

    def __init__(self) -> None:
        self.record_count = 0
        self._stage_counts: dict[str, int] = defaultdict(int)
        self._values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._channel_values: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def add(self, record: Mapping[str, Any]) -> None:
        stage = str(record.get("stage", "unknown"))
        self.record_count += 1
        self._stage_counts[stage] += 1
        for metric in SUMMARY_METRICS:
            value = _optional_float(record.get(metric))
            if value is not None:
                self._values[stage][metric].append(value)
        for channel_record in record.get("bev_channel_stats", ()):
            if not isinstance(channel_record, Mapping):
                continue
            try:
                channel_index = int(channel_record["channel"])
            except (KeyError, TypeError, ValueError):
                continue
            for metric in CHANNEL_SUMMARY_METRICS:
                value = _optional_float(channel_record.get(metric))
                if value is not None:
                    self._channel_values[stage][channel_index][metric].append(value)

    @staticmethod
    def _distribution(values: Iterable[float]) -> dict[str, Any]:
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0:
            return {"count": 0}
        return {
            "count": int(array.size),
            "min": float(np.min(array)),
            "p05": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "mean": float(np.mean(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    def summary(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        stages = {}
        for stage in sorted(self._stage_counts):
            stages[stage] = {
                "record_count": self._stage_counts[stage],
                "metrics": {
                    metric: self._distribution(self._values[stage].get(metric, []))
                    for metric in SUMMARY_METRICS
                },
                "channels": {
                    str(channel_index): {
                        "metrics": {
                            metric: self._distribution(channel_metrics.get(metric, []))
                            for metric in CHANNEL_SUMMARY_METRICS
                        }
                    }
                    for channel_index, channel_metrics in sorted(self._channel_values[stage].items())
                },
            }
        summary = {
            "schema": LIDAR_CONTRACT_SCHEMA,
            "record_count": self.record_count,
            "stages": stages,
        }
        if context:
            summary["context"] = make_json_safe(context)
        return summary


class LidarDiagnosticWriter:
    """Write strict JSONL statistics and a bounded number of NumPy bundles."""

    def __init__(self, output_dir: str | Path, *, max_dumps: int = 0) -> None:
        self.output_dir = Path(output_dir)
        self.dump_dir = self.output_dir / "dumps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.max_dumps = max(0, int(max_dumps))
        self.dump_count = 0
        self.accumulator = LidarContractAccumulator()
        self._records_file = (self.output_dir / "records.jsonl").open("w", encoding="utf-8")
        self._closed = False

    def record(self, record: Mapping[str, Any]) -> None:
        safe_record = make_json_safe(record)
        self.accumulator.add(safe_record)
        self._records_file.write(json.dumps(safe_record, separators=(",", ":"), sort_keys=True) + "\n")
        self._records_file.flush()

    def dump_bundle(
        self,
        *,
        step: int,
        event: str,
        arrays: Mapping[str, Any],
        metadata: Mapping[str, Any],
        recent_history: Iterable[Mapping[str, Any]] = (),
    ) -> Path | None:
        if self.dump_count >= self.max_dumps:
            return None
        safe_event = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(event)).strip("_") or "event"
        stem = f"step_{int(step):06d}_{safe_event}_{self.dump_count:03d}"
        array_path = self.dump_dir / f"{stem}.npz"
        metadata_path = self.dump_dir / f"{stem}.json"
        prepared_arrays = {}
        for name, value in arrays.items():
            if value is None:
                continue
            array = _as_numpy(value)
            if array.dtype.kind == "f":
                array = array.astype(np.float32, copy=False)
            prepared_arrays[re.sub(r"[^A-Za-z0-9_]+", "_", str(name))] = array
        np.savez(array_path, **prepared_arrays)
        payload = {
            "schema": LIDAR_CONTRACT_SCHEMA,
            "step": int(step),
            "event": str(event),
            "array_file": array_path.name,
            "array_shapes": {name: list(array.shape) for name, array in prepared_arrays.items()},
            "metadata": make_json_safe(metadata),
            "recent_history": make_json_safe(list(recent_history)),
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.dump_count += 1
        return array_path

    def close(
        self,
        *,
        context: Mapping[str, Any] | None = None,
        recent_history: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        if self._closed:
            return
        self._records_file.close()
        summary = self.accumulator.summary(context=context)
        summary["dump_count"] = self.dump_count
        summary["recent_history"] = make_json_safe(list(recent_history))
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._closed = True
