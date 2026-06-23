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
