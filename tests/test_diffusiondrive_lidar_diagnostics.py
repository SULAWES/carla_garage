from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "team_code" / "diffusiondrive" / "lidar_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("diffusiondrive_lidar_diagnostics", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load LiDAR diagnostics module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LidarDiagnosticWriter = MODULE.LidarDiagnosticWriter
compute_lidar_contract_record = MODULE.compute_lidar_contract_record


class LidarDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(
            min_x=-2.0,
            max_x=2.0,
            min_y=-2.0,
            max_y=2.0,
            lidar_split_height=0.2,
            max_height_lidar=5.0,
        )

    def test_contract_filters_and_regions_match_model_input(self) -> None:
        points = np.array([
            [-1.0, -1.0, 0.1],
            [1.0, -1.0, 0.3],
            [-1.0, 1.0, 0.5],
            [4.0, 0.0, 1.0],
            [np.nan, 0.0, 0.5],
        ], dtype=np.float32)
        bev = np.zeros((1, 4, 4), dtype=np.float32)
        bev[0, 0, 0] = 0.2
        bev[0, 1, 1] = 1.0

        record = compute_lidar_contract_record(
            points,
            bev,
            self.config,
            source="test",
            stage="full_scan",
        )

        self.assertEqual(record["point_count"], 5)
        self.assertEqual(record["finite_point_count"], 4)
        self.assertEqual(record["in_bev_point_count"], 3)
        self.assertEqual(record["below_split_count"], 1)
        self.assertEqual(record["above_split_count"], 3)
        self.assertEqual(record["below_channel_point_count"], 1)
        self.assertEqual(record["above_channel_point_count"], 2)
        self.assertEqual(record["model_point_count"], 2)
        self.assertEqual(record["front_count"], 1)
        self.assertEqual(record["back_count"], 1)
        self.assertEqual(record["left_count"], 1)
        self.assertEqual(record["right_count"], 1)
        self.assertAlmostEqual(record["bev_nonzero_ratio"], 2 / 16)
        self.assertAlmostEqual(record["bev_saturation_ratio"], 1 / 16)

        self.config.use_ground_plane = True
        two_channel_record = compute_lidar_contract_record(
            points,
            np.concatenate([bev, bev], axis=0),
            self.config,
            source="test",
            stage="full_scan",
        )
        self.assertEqual(two_channel_record["model_point_count"], 3)

    def test_writer_bounds_dumps_and_emits_strict_summary(self) -> None:
        points = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
        bev = np.ones((1, 2, 2), dtype=np.float32)
        record = compute_lidar_contract_record(
            points,
            bev,
            self.config,
            source="test",
            stage="model_input",
            context={"non_finite": float("nan")},
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            writer = LidarDiagnosticWriter(temporary_dir, max_dumps=1)
            writer.record(record)
            first = writer.dump_bundle(
                step=1,
                event="periodic",
                arrays={"points": points, "bev": bev},
                metadata=record,
            )
            second = writer.dump_bundle(
                step=2,
                event="periodic",
                arrays={"points": points},
                metadata=record,
            )
            writer.close(context={"finished": True})

            self.assertIsNotNone(first)
            self.assertIsNone(second)
            summary = json.loads((Path(temporary_dir) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["dump_count"], 1)
            self.assertEqual(summary["stages"]["model_input"]["record_count"], 1)
            channel_summary = summary["stages"]["model_input"]["channels"]["0"]["metrics"]
            self.assertEqual(channel_summary["mean"]["count"], 1)
            self.assertEqual(channel_summary["mean"]["mean"], 1.0)
            record_line = (Path(temporary_dir) / "records.jsonl").read_text(encoding="utf-8")
            self.assertIsNone(json.loads(record_line)["context"]["non_finite"])


if __name__ == "__main__":
    unittest.main()
