from __future__ import annotations

import torch

from cuda_rl.models import ActorCriticNetwork, PolicyNetwork, QNetwork, ValueNetwork


def test_policy_network_returns_action_distribution_outputs() -> None:
    network = PolicyNetwork(observation_dim=4, action_dim=2, hidden_sizes=(8,))
    observations = torch.zeros((3, 4), dtype=torch.float32)

    actions, log_probabilities, entropies = network.act(
        observations,
        deterministic=False,
    )

    assert actions.shape == (3,)
    assert log_probabilities.shape == (3,)
    assert entropies.shape == (3,)


def test_value_and_q_network_shapes() -> None:
    observations = torch.zeros((5, 4), dtype=torch.float32)

    assert ValueNetwork(4, (8,))(observations).shape == (5,)
    assert QNetwork(4, 2, (8,))(observations).shape == (5, 2)


def test_actor_critic_network_shapes() -> None:
    network = ActorCriticNetwork(observation_dim=4, action_dim=2, hidden_sizes=(8,))
    observations = torch.zeros((2, 4), dtype=torch.float32)

    actions, log_probabilities, entropies, values = network.act(
        observations,
        deterministic=True,
    )

    assert actions.shape == (2,)
    assert log_probabilities.shape == (2,)
    assert entropies.shape == (2,)
    assert values.shape == (2,)
