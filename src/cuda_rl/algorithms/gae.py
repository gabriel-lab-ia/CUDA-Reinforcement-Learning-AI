from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    next_value: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute generalized advantage estimates and bootstrapped returns."""

    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be one-dimensional tensors.")
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have matching shapes.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    bootstrap_value = (
        torch.zeros((), dtype=values.dtype, device=values.device)
        if next_value is None
        else next_value.to(device=values.device, dtype=values.dtype).reshape(())
    )
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    next_step_value = bootstrap_value

    for step in reversed(range(rewards.numel())):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_step_value * non_terminal - values[step]
        running_advantage = (
            delta + gamma * gae_lambda * non_terminal * running_advantage
        )
        advantages[step] = running_advantage
        next_step_value = values[step]

    returns = advantages + values
    return advantages, returns


def compute_gae_batched(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    next_values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute GAE for C-contiguous [timesteps, num_envs] tensors."""

    if rewards.ndim != 2 or values.ndim != 2 or dones.ndim != 2:
        raise ValueError("rewards, values, and dones must be two-dimensional tensors.")
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have matching shapes.")
    if rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("GAE tensors must have shape [timesteps >= 1, num_envs >= 1].")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")
    if next_values is not None and tuple(next_values.shape) != (rewards.shape[1],):
        raise ValueError("next_values must have shape [num_envs].")

    bootstrap_values = (
        torch.zeros(rewards.shape[1], dtype=values.dtype, device=values.device)
        if next_values is None
        else next_values.to(device=values.device, dtype=values.dtype)
    )
    advantages = torch.zeros_like(rewards)
    running_advantages = torch.zeros(
        rewards.shape[1], dtype=values.dtype, device=values.device
    )
    next_step_values = bootstrap_values

    for step in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_step_values * non_terminal - values[step]
        running_advantages = (
            delta + gamma * gae_lambda * non_terminal * running_advantages
        )
        advantages[step] = running_advantages
        next_step_values = values[step]

    returns = advantages + values
    return advantages, returns
