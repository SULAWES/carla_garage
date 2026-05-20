"""Shared DiffusionDrive status feature builders."""

from __future__ import annotations

import numpy as np
import torch

import transfuser_utils as t_u


STATUS_FEATURE_SCHEMA = "command_one_hot(6)+speed(1)"
STATUS_FEATURE_DIM = 7


def build_status_feature(
    command_one_hot: np.ndarray | torch.Tensor,
    speed: float | np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the shared 7D status feature used by training and inference."""

    command = torch.as_tensor(command_one_hot, device=device, dtype=dtype)
    if command.ndim == 1:
        squeeze_batch = True
        command = command.unsqueeze(0)
    elif command.ndim == 2:
        squeeze_batch = False
    else:
        raise RuntimeError(f"command_one_hot must be 1D or 2D, got shape {tuple(command.shape)}.")

    if command.shape[-1] != 6:
        raise RuntimeError(f"command_one_hot must have 6 channels, got shape {tuple(command.shape)}.")

    speed_tensor = torch.as_tensor(speed, device=command.device, dtype=command.dtype)
    if speed_tensor.ndim == 0:
        speed_tensor = speed_tensor.view(1, 1)
    elif speed_tensor.ndim == 1:
        speed_tensor = speed_tensor.view(-1, 1)
    elif speed_tensor.ndim == 2 and speed_tensor.shape[-1] == 1:
        pass
    else:
        raise RuntimeError(f"speed must be scalar, 1D, or Bx1, got shape {tuple(speed_tensor.shape)}.")

    if speed_tensor.shape[0] != command.shape[0]:
        raise RuntimeError(
            "status feature batch mismatch: "
            f"command batch {command.shape[0]} vs speed batch {speed_tensor.shape[0]}."
        )

    status = torch.cat([command, speed_tensor], dim=1)
    return status.squeeze(0) if squeeze_batch else status


def build_status_feature_from_command(
    command: int,
    speed: float | np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build status from a CARLA route command id and scalar speed."""

    command_one_hot = t_u.command_to_one_hot(int(command))
    return build_status_feature(command_one_hot, speed, device=device, dtype=dtype)
