from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import cast

import torch


def native_gae_available() -> bool:
    """Return whether the optional native CUDA GAE extension can be imported."""

    try:
        _load_extension()
    except RuntimeError:
        return False
    return True


def compute_gae_native(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    next_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with the optional native CUDA extension.

    This function never falls back to CPU, NumPy, or PyTorch implementations.
    """

    _validate_inputs(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        next_values=next_values,
    )
    extension = _load_extension()
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        extension.compute_gae_native(
            rewards,
            values,
            dones,
            float(gamma),
            float(gae_lambda),
            next_values,
        ),
    )


def _load_extension() -> ModuleType:
    try:
        return import_module("cuda_rl._native_gae")
    except Exception as exception:
        raise RuntimeError(
            "Native CUDA GAE extension is not available. Build it explicitly "
            "with CUDA_RL_BUILD_NATIVE_EXT=1 before requesting native_cuda."
        ) from exception


def _validate_inputs(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    next_values: torch.Tensor | None,
) -> None:
    if not rewards.is_cuda or not values.is_cuda or not dones.is_cuda:
        raise ValueError("native GAE requires CUDA tensors.")
    if rewards.dtype != torch.float32 or values.dtype != torch.float32:
        raise TypeError("native GAE supports float32 rewards and values only.")
    if dones.dtype != torch.float32:
        raise TypeError("native GAE supports float32 dones only.")
    if rewards.ndim != 2 or values.ndim != 2 or dones.ndim != 2:
        raise ValueError("native GAE inputs must be [timesteps, num_envs].")
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("native GAE inputs must have matching shapes.")
    if rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("native GAE requires timesteps >= 1 and num_envs >= 1.")
    if (
        not rewards.is_contiguous()
        or not values.is_contiguous()
        or not dones.is_contiguous()
    ):
        raise ValueError("native GAE requires contiguous tensors.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")
    if next_values is None:
        return
    if not next_values.is_cuda:
        raise ValueError("next_values must be a CUDA tensor.")
    if next_values.dtype != torch.float32:
        raise TypeError("next_values must be float32.")
    if tuple(next_values.shape) != (rewards.shape[1],):
        raise ValueError("next_values must have shape [num_envs].")
    if not next_values.is_contiguous():
        raise ValueError("next_values must be contiguous.")
