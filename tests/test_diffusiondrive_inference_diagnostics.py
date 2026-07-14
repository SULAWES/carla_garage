from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "team_code" / "diffusiondrive" / "inference_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("diffusiondrive_inference_diagnostics", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load inference diagnostics module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiffusionInferenceDiagnosticsTest(unittest.TestCase):
    def test_random_noise_preserves_global_rng_behavior(self) -> None:
        reference = torch.zeros((2, 3, 4, 2), dtype=torch.float32)
        torch.manual_seed(41)
        expected = torch.randn(reference.shape, dtype=reference.dtype)

        torch.manual_seed(41)
        actual = MODULE.resolve_diffusion_inference_noise(reference, mode="random", seed=999)

        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_noise_is_repeatable_and_batch_invariant(self) -> None:
        single = torch.zeros((1, 3, 4, 2), dtype=torch.float32)
        batch = torch.zeros((5, 3, 4, 2), dtype=torch.float32)

        first = MODULE.resolve_diffusion_inference_noise(single, mode="fixed", seed=17)
        second = MODULE.resolve_diffusion_inference_noise(single, mode="fixed", seed=17)
        batched = MODULE.resolve_diffusion_inference_noise(batch, mode="fixed", seed=17)
        changed_seed = MODULE.resolve_diffusion_inference_noise(single, mode="fixed", seed=18)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], batched[0]))
        self.assertTrue(torch.equal(batched[0], batched[-1]))
        self.assertFalse(torch.equal(first, changed_seed))

    def test_fixed_noise_does_not_consume_global_rng(self) -> None:
        reference = torch.zeros((2, 3, 4, 2), dtype=torch.float32)
        torch.manual_seed(123)
        expected = torch.randn(8)

        torch.manual_seed(123)
        MODULE.resolve_diffusion_inference_noise(reference, mode="fixed", seed=9)
        actual = torch.randn(8)

        self.assertTrue(torch.equal(actual, expected))

    def test_zero_and_explicit_seeded_noise(self) -> None:
        reference = torch.zeros((2, 3, 4, 2), dtype=torch.float32)
        zero = MODULE.resolve_diffusion_inference_noise(reference, mode="zero", seed=0)
        explicit = torch.full_like(reference, 0.25)
        seeded = MODULE.resolve_diffusion_inference_noise(
            reference,
            mode="seeded",
            seed=11,
            explicit_noise=explicit,
        )

        self.assertEqual(torch.count_nonzero(zero).item(), 0)
        self.assertTrue(torch.equal(seeded, explicit))
        with self.assertRaisesRegex(RuntimeError, "requires explicit per-sample noise"):
            MODULE.resolve_diffusion_inference_noise(reference, mode="seeded", seed=11)
        with self.assertRaisesRegex(RuntimeError, "shape must match"):
            MODULE.resolve_diffusion_inference_noise(
                reference,
                mode="seeded",
                seed=11,
                explicit_noise=explicit[:1],
            )
        with self.assertRaisesRegex(RuntimeError, "only accepted"):
            MODULE.resolve_diffusion_inference_noise(
                reference,
                mode="fixed",
                seed=11,
                explicit_noise=explicit,
            )

    def test_sample_seed_is_stable_and_keyed(self) -> None:
        first = MODULE.stable_diffusion_noise_seed(5, "scenario", "route", 10)
        repeated = MODULE.stable_diffusion_noise_seed(5, "scenario", "route", 10)
        changed = MODULE.stable_diffusion_noise_seed(5, "scenario", "route", 11)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        noise_a = MODULE.gaussian_noise_from_seed((3, 4, 2), first)
        noise_b = MODULE.gaussian_noise_from_seed((3, 4, 2), repeated)
        self.assertTrue(torch.equal(noise_a, noise_b))
        with self.assertRaisesRegex(RuntimeError, "must be one of"):
            MODULE.normalize_diffusion_noise_mode("unknown")
        with self.assertRaisesRegex(RuntimeError, "must be in"):
            MODULE.validate_diffusion_noise_seed(-1)

    def test_trajectory_output_reports_top2_mode_diagnostics(self) -> None:
        poses_reg = torch.tensor(
            [
                [
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[0.0, 0.0], [2.0, 2.0]],
                    [[0.0, 0.0], [3.0, 3.0]],
                ]
            ],
            dtype=torch.float32,
        )
        poses_cls = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)

        output = MODULE.build_trajectory_inference_output(poses_reg, poses_cls)

        self.assertEqual(output["trajectory_mode_index"].tolist(), [1])
        self.assertEqual(output["trajectory_mode_top2_index"].tolist(), [[1, 2]])
        self.assertTrue(torch.equal(output["trajectory"][0], poses_reg[0, 1]))
        self.assertEqual(
            output["trajectory_mode_top2_endpoint"].tolist(),
            [[[2.0, 2.0], [3.0, 3.0]]],
        )
        self.assertGreater(output["trajectory_mode_margin"].item(), 0.0)
        self.assertGreaterEqual(output["trajectory_mode_entropy_normalized"].item(), 0.0)
        self.assertLessEqual(output["trajectory_mode_entropy_normalized"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
