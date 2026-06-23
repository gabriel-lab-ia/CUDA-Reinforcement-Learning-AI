from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PPOLossConfig:
    clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    normalize_advantages: bool = True

    def validate(self) -> None:
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive.")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("value_loss_coefficient must be non-negative.")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative.")


@dataclass(frozen=True, slots=True)
class PPOLossResult:
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_bonus: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def compute_ppo_loss(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    values: Tensor,
    entropies: Tensor,
    config: PPOLossConfig,
) -> PPOLossResult:
    config.validate()
    _validate_same_shape(
        new_log_probabilities,
        old_log_probabilities,
        advantages,
        returns,
        values,
        entropies,
    )

    effective_advantages = advantages
    if config.normalize_advantages and advantages.numel() > 1:
        effective_advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = torch.exp(log_ratio)
    unclipped = ratio * effective_advantages
    clipped = (
        torch.clamp(
            ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range,
        )
        * effective_advantages
    )
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()
    loss = (
        policy_loss
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * entropy_bonus
    )
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = (
        (torch.abs(ratio - 1.0) > config.clip_range).to(torch.float32).mean()
    )
    return PPOLossResult(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


def _validate_same_shape(*tensors: Tensor) -> None:
    expected = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != expected:
            raise ValueError("all PPO loss tensors must have matching shapes.")
