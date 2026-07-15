from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "team_code" / "diffusiondrive" / "spatial_target.py"
SPEC = importlib.util.spec_from_file_location("diffusiondrive_spatial_target", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load spatial target module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiffusionDriveSpatialTargetTest(unittest.TestCase):
    def test_low_displacement_uses_route_condition_instead_of_last_jitter(self) -> None:
        points = np.asarray([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.198, 0.0015],
        ])
        distances = np.asarray([2.5, 3.5])
        diagnostics = {}

        stable = MODULE.resample_polyline_by_distance(
            points,
            distances,
            np.asarray([1.0, 0.0]),
            diagnostics=diagnostics,
        )
        legacy = MODULE.resample_polyline_by_distance(
            points,
            distances,
            np.asarray([1.0, 0.0]),
            min_extrapolation_displacement=0.0,
        )

        self.assertEqual(diagnostics["extrapolation_direction_source"], "route_condition")
        self.assertEqual(diagnostics["extrapolated_pose_count"], 2)
        self.assertGreater(stable[0, 0], 2.4)
        self.assertLess(legacy[0, 0], 0.0)

    def test_moving_path_uses_trailing_displacement_not_last_jitter(self) -> None:
        points = np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.998, 0.0015],
        ])
        diagnostics = {}

        target = MODULE.resample_polyline_by_distance(
            points,
            np.asarray([2.5, 3.5]),
            np.asarray([0.0, 1.0]),
            diagnostics=diagnostics,
        )

        self.assertEqual(diagnostics["extrapolation_direction_source"], "trailing_displacement")
        self.assertGreaterEqual(diagnostics["extrapolation_direction_baseline_meters"], 0.5)
        self.assertGreater(target[0, 0], 2.4)
        self.assertLess(abs(target[0, 1]), 0.01)

    def test_observed_arc_length_interpolation_is_unchanged(self) -> None:
        points = np.asarray([[0.0, 0.0], [4.0, 0.0]])
        diagnostics = {}

        target = MODULE.resample_polyline_by_distance(
            points,
            np.asarray([1.0, 2.5, 4.0]),
            np.asarray([0.0, 1.0]),
            diagnostics=diagnostics,
        )

        np.testing.assert_allclose(target, [[1.0, 0.0], [2.5, 0.0], [4.0, 0.0]])
        self.assertEqual(diagnostics["extrapolated_pose_count"], 0)

    def test_contract_metadata_records_stability_threshold(self) -> None:
        metadata = MODULE.spatial_target_metadata()

        self.assertEqual(metadata["schema"], "arc_length_stable_extrapolation_v2")
        self.assertEqual(metadata["extrapolation_strategy"], "trailing_displacement_or_route_condition")
        self.assertAlmostEqual(metadata["min_extrapolation_displacement_meters"], 0.5)


if __name__ == "__main__":
    unittest.main()
