from __future__ import annotations

import pytest
import torch

from cuda_rl.algorithms import PPOLossConfig, compute_gae, compute_ppo_loss


def test_compute_gae_returns_expected_shapes() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.all(returns >= values)


def test_compute_gae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        compute_gae(
            torch.ones(2),
            torch.ones(3),
            torch.zeros(2),
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_compute_ppo_loss_returns_diagnostics() -> None:
    old_log_probabilities = torch.log(torch.tensor([0.4, 0.6, 0.5]))
    new_log_probabilities = torch.log(torch.tensor([0.5, 0.5, 0.55]))
    advantages = torch.tensor([1.0, -0.5, 0.25])
    returns = torch.tensor([1.2, 0.3, 0.8])
    values = torch.tensor([1.0, 0.4, 0.7])
    entropies = torch.tensor([0.2, 0.3, 0.25])

    result = compute_ppo_loss(
        new_log_probabilities=new_log_probabilities,
        old_log_probabilities=old_log_probabilities,
        advantages=advantages,
        returns=returns,
        values=values,
        entropies=entropies,
        config=PPOLossConfig(),
    )

    assert result.loss.ndim == 0
    assert result.approximate_kl.ndim == 0
    assert 0.0 <= float(result.clip_fraction.item()) <= 1.0
