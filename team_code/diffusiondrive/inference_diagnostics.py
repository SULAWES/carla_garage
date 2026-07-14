"""Deterministic diffusion noise and trajectory-mode diagnostics."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Sequence

import torch


DIFFUSION_INFER_NOISE_MODES = ("random", "fixed", "zero", "seeded")
MAX_DIFFUSION_NOISE_SEED = (1 << 63) - 1


def normalize_diffusion_noise_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DIFFUSION_INFER_NOISE_MODES:
        choices = ", ".join(DIFFUSION_INFER_NOISE_MODES)
        raise RuntimeError(f"Diffusion inference noise mode must be one of {choices}; got {mode!r}.")
    return normalized


def validate_diffusion_noise_seed(seed: int) -> int:
    value = int(seed)
    if not 0 <= value <= MAX_DIFFUSION_NOISE_SEED:
        raise RuntimeError(
            "Diffusion inference noise seed must be in "
            f"[0, {MAX_DIFFUSION_NOISE_SEED}]; got {seed!r}."
        )
    return value


def stable_diffusion_noise_seed(base_seed: int, *key_parts: Any) -> int:
    """Derive a process-independent seed from a stable sample or online-step key."""

    base_seed = validate_diffusion_noise_seed(base_seed)
    hasher = hashlib.blake2b(digest_size=8, person=b"dd-noise-v1")
    for part in (base_seed, *key_parts):
        encoded = str(part).encode("utf-8")
        hasher.update(len(encoded).to_bytes(4, byteorder="little", signed=False))
        hasher.update(encoded)
    return (
        int.from_bytes(hasher.digest(), byteorder="little", signed=False)
        & MAX_DIFFUSION_NOISE_SEED
    )


def gaussian_noise_from_seed(
    shape: Sequence[int],
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate Gaussian noise without consuming the global torch RNG."""

    if not dtype.is_floating_point:
        raise RuntimeError(f"Diffusion inference noise dtype must be floating point; got {dtype}.")
    seed = validate_diffusion_noise_seed(seed)
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        tuple(int(value) for value in shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def resolve_diffusion_inference_noise(
    reference: torch.Tensor,
    *,
    mode: str,
    seed: int,
    explicit_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve inference noise while keeping fixed mode batch-size invariant."""

    if not reference.dtype.is_floating_point:
        raise RuntimeError(f"Diffusion reference tensor must be floating point; got {reference.dtype}.")
    mode = normalize_diffusion_noise_mode(mode)
    seed = validate_diffusion_noise_seed(seed)

    if explicit_noise is not None:
        if mode != "seeded":
            raise RuntimeError(
                "Explicit diffusion noise is only accepted when diffusion inference "
                "noise mode is 'seeded'."
            )
        if tuple(explicit_noise.shape) != tuple(reference.shape):
            raise RuntimeError(
                "Explicit diffusion noise shape must match the normalized plan anchors: "
                f"got {tuple(explicit_noise.shape)}, expected {tuple(reference.shape)}."
            )
        if not explicit_noise.dtype.is_floating_point:
            raise RuntimeError(
                f"Explicit diffusion noise must be floating point; got {explicit_noise.dtype}."
            )
        return explicit_noise.to(device=reference.device, dtype=reference.dtype, non_blocking=True)

    if mode == "random":
        return torch.randn_like(reference)
    if mode == "zero":
        return torch.zeros_like(reference)
    if mode == "seeded":
        raise RuntimeError(
            "Seeded diffusion inference requires explicit per-sample noise in "
            "features['diffusion_noise']."
        )

    template = gaussian_noise_from_seed(
        (1, *reference.shape[1:]),
        seed,
        device=reference.device,
        dtype=reference.dtype,
    )
    return template.expand_as(reference)


def build_trajectory_inference_output(
    poses_reg: torch.Tensor,
    poses_cls: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Select the best trajectory and expose compact final-mode diagnostics."""

    if poses_reg.ndim != 4:
        raise RuntimeError(
            "Trajectory regression output must have shape (batch, modes, poses, coordinates); "
            f"got {tuple(poses_reg.shape)}."
        )
    if poses_cls.ndim != 2:
        raise RuntimeError(
            "Trajectory classification output must have shape (batch, modes); "
            f"got {tuple(poses_cls.shape)}."
        )
    if poses_reg.shape[:2] != poses_cls.shape:
        raise RuntimeError(
            "Trajectory regression/classification batch and mode dimensions must match: "
            f"reg={tuple(poses_reg.shape)}, cls={tuple(poses_cls.shape)}."
        )
    if poses_cls.shape[1] < 2:
        raise RuntimeError("Trajectory mode diagnostics require at least two anchor modes.")

    probabilities = torch.softmax(poses_cls, dim=-1)
    # Preserve the previous argmax selection semantics exactly, including ties.
    mode_index = poses_cls.argmax(dim=-1)
    top1_prob = torch.gather(probabilities, dim=1, index=mode_index[:, None]).squeeze(1)
    remaining_probabilities = probabilities.clone()
    remaining_probabilities.scatter_(1, mode_index[:, None], -1.0)
    second_index = remaining_probabilities.argmax(dim=-1)
    second_prob = torch.gather(probabilities, dim=1, index=second_index[:, None]).squeeze(1)
    top2_index = torch.stack((mode_index, second_index), dim=1)
    top2_prob = torch.stack((top1_prob, second_prob), dim=1)
    gather_index = mode_index[:, None, None, None].expand(
        -1,
        1,
        poses_reg.shape[2],
        poses_reg.shape[3],
    )
    trajectory = torch.gather(poses_reg, dim=1, index=gather_index).squeeze(1)

    endpoint_index = top2_index[..., None].expand(-1, -1, poses_reg.shape[-1])
    top2_endpoint = torch.gather(poses_reg[:, :, -1, :], dim=1, index=endpoint_index)
    entropy = -(
        probabilities
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1)

    return {
        "trajectory": trajectory,
        "trajectory_mode_index": mode_index,
        "trajectory_mode_top2_index": top2_index,
        "trajectory_mode_top2_prob": top2_prob,
        "trajectory_mode_margin": top2_prob[:, 0] - top2_prob[:, 1],
        "trajectory_mode_entropy": entropy,
        "trajectory_mode_entropy_normalized": entropy / math.log(poses_cls.shape[1]),
        "trajectory_mode_top2_endpoint": top2_endpoint,
    }
