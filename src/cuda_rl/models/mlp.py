from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def initialize_linear_layer(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    output_gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        linear = nn.Linear(previous_dim, hidden_dim)
        initialize_linear_layer(linear)
        layers.extend((linear, nn.Tanh()))
        previous_dim = hidden_dim
    output_layer = nn.Linear(previous_dim, output_dim)
    initialize_linear_layer(output_layer, gain=output_gain)
    layers.append(output_layer)
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=0.01,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations))

    def distribution(self, observations: Tensor) -> Categorical:
        return Categorical(logits=self.forward(observations))

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(observations)
        if deterministic:
            actions = torch.argmax(distribution.logits, dim=-1)
        else:
            actions = distribution.sample()  # type: ignore[no-untyped-call]
        log_probabilities = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
        entropies = distribution.entropy()  # type: ignore[no-untyped-call]
        return actions, log_probabilities, entropies


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            1,
            output_gain=1.0,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations)).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.policy = PolicyNetwork(observation_dim, action_dim, hidden_sizes)
        self.value = ValueNetwork(observation_dim, hidden_sizes)

    def act(
        self,
        observations: Tensor,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        actions, log_probabilities, entropies = self.policy.act(
            observations,
            deterministic=deterministic,
        )
        values = self.value(observations)
        return actions, log_probabilities, entropies, values


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = build_mlp(
            observation_dim,
            hidden_sizes,
            action_dim,
            output_gain=1.0,
        )

    def forward(self, observations: Tensor) -> Tensor:
        return cast(Tensor, self.network(observations))
