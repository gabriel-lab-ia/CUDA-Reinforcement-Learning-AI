from __future__ import annotations

import numpy as np


def compute_gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    next_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy float32 reference for C-contiguous [timesteps, num_envs] GAE."""

    if rewards.ndim != 2 or values.ndim != 2 or dones.ndim != 2:
        raise ValueError("rewards, values, and dones must be two-dimensional arrays.")
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have matching shapes.")
    if rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("GAE arrays must have shape [timesteps >= 1, num_envs >= 1].")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")
    if next_values is not None and tuple(next_values.shape) != (rewards.shape[1],):
        raise ValueError("next_values must have shape [num_envs].")

    rewards = np.ascontiguousarray(rewards, dtype=np.float32)
    values = np.ascontiguousarray(values, dtype=np.float32)
    dones = np.ascontiguousarray(dones, dtype=np.float32)
    bootstrap_values = (
        np.zeros(rewards.shape[1], dtype=np.float32)
        if next_values is None
        else np.ascontiguousarray(next_values, dtype=np.float32)
    )

    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_advantages = np.zeros(rewards.shape[1], dtype=np.float32)
    next_step_values = bootstrap_values

    for step in range(rewards.shape[0] - 1, -1, -1):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_step_values * non_terminal - values[step]
        running_advantages = (
            delta + gamma * gae_lambda * non_terminal * running_advantages
        )
        advantages[step] = running_advantages
        next_step_values = values[step]

    return advantages, advantages + values
