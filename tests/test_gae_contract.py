from __future__ import annotations

import numpy as np
import pytest
import torch

from cuda_rl.algorithms import compute_gae_batched, compute_gae_numpy
from cuda_rl.native import compute_gae_native, native_gae_available

RTOL = 1e-5
ATOL = 1e-6


@pytest.mark.parametrize(
    ("timesteps", "num_envs", "gamma", "gae_lambda", "seed"),
    [
        (1, 1, 0.99, 0.95, 0),
        (128, 1, 0.99, 0.95, 1),
        (128, 8, 0.99, 0.95, 2),
        (512, 32, 1.0, 0.95, 3),
        (2048, 128, 0.99, 0.0, 4),
        (128, 8, 0.99, 1.0, 5),
    ],
)
def test_batched_torch_gae_matches_numpy_reference(
    timesteps: int,
    num_envs: int,
    gamma: float,
    gae_lambda: float,
    seed: int,
) -> None:
    rewards, values, dones, next_values = _case(timesteps, num_envs, seed)

    expected_advantages, expected_returns = compute_gae_numpy(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        next_values=next_values,
    )
    advantages, returns = compute_gae_batched(
        torch.from_numpy(rewards),
        torch.from_numpy(values),
        torch.from_numpy(dones),
        gamma=gamma,
        gae_lambda=gae_lambda,
        next_values=torch.from_numpy(next_values),
    )

    np.testing.assert_allclose(
        advantages.numpy(), expected_advantages, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(returns.numpy(), expected_returns, rtol=RTOL, atol=ATOL)
    assert np.isfinite(advantages.numpy()).all()
    assert np.isfinite(returns.numpy()).all()


def test_gae_reference_handles_terminal_patterns_zero_rewards_and_negative_values() -> (
    None
):
    rewards = np.zeros((4, 2), dtype=np.float32)
    values = -np.asarray(
        [[0.5, 0.25], [0.4, 0.3], [0.1, 0.8], [0.0, 0.2]],
        dtype=np.float32,
    )
    dones = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )

    advantages, returns = compute_gae_numpy(
        rewards,
        values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
        next_values=np.asarray([0.2, -0.3], dtype=np.float32),
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert np.isfinite(advantages).all()
    assert np.isfinite(returns).all()


def test_native_gae_api_rejects_cpu_without_fallback() -> None:
    rewards = torch.ones((2, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA tensors"):
        compute_gae_native(
            rewards,
            rewards,
            torch.zeros_like(rewards),
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_native_gae_available_is_safe_without_extension() -> None:
    assert isinstance(native_gae_available(), bool)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
@pytest.mark.skipif(
    not native_gae_available(), reason="native CUDA GAE extension is unavailable."
)
@pytest.mark.parametrize(
    ("timesteps", "num_envs", "seed"), [(1, 1, 0), (128, 8, 1), (512, 32, 2)]
)
def test_native_gae_matches_numpy_reference(
    timesteps: int,
    num_envs: int,
    seed: int,
) -> None:
    rewards, values, dones, next_values = _case(timesteps, num_envs, seed)
    expected_advantages, expected_returns = compute_gae_numpy(
        rewards,
        values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
        next_values=next_values,
    )

    advantages, returns = compute_gae_native(
        torch.from_numpy(rewards).cuda(),
        torch.from_numpy(values).cuda(),
        torch.from_numpy(dones).cuda(),
        gamma=0.99,
        gae_lambda=0.95,
        next_values=torch.from_numpy(next_values).cuda(),
    )

    np.testing.assert_allclose(
        advantages.cpu().numpy(), expected_advantages, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        returns.cpu().numpy(), expected_returns, rtol=RTOL, atol=ATOL
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
@pytest.mark.skipif(
    not native_gae_available(), reason="native CUDA GAE extension is unavailable."
)
def test_native_gae_rejects_non_contiguous_inputs() -> None:
    base = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    non_contiguous = base[:, ::2]

    with pytest.raises(ValueError, match="contiguous"):
        compute_gae_native(
            non_contiguous,
            non_contiguous,
            torch.zeros_like(non_contiguous),
            gamma=0.99,
            gae_lambda=0.95,
        )


def _case(
    timesteps: int,
    num_envs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=(timesteps, num_envs)).astype(np.float32)
    values = rng.normal(loc=-0.1, size=(timesteps, num_envs)).astype(np.float32)
    dones = np.zeros((timesteps, num_envs), dtype=np.float32)
    dones[-1, :] = 1.0
    if timesteps > 4:
        dones[timesteps // 3, ::2] = 1.0
        dones[(2 * timesteps) // 3, 1::2] = 1.0
    next_values = rng.normal(size=(num_envs,)).astype(np.float32)
    return rewards, values, dones, next_values
