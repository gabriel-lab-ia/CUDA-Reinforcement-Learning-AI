from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class PrioritizedSample:
    transitions: tuple[ReplayTransition, ...]
    indices: tuple[int, ...]
    weights: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.storage: deque[ReplayTransition] = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: ReplayTransition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self.storage):
            raise ValueError("Cannot sample more transitions than currently stored.")
        return self.random.sample(list(self.storage), batch_size)


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with importance-sampling weights."""

    def __init__(
        self,
        capacity: int,
        seed: int,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.storage: list[ReplayTransition] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        transition: ReplayTransition,
        *,
        priority: float | None = None,
    ) -> None:
        max_priority = float(self.priorities[: len(self.storage)].max(initial=1.0))
        assigned_priority = max_priority if priority is None else priority
        if assigned_priority <= 0.0:
            raise ValueError("priority must be positive.")

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.priorities[self.position] = assigned_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> PrioritizedSample:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self.storage):
            raise ValueError("Cannot sample more transitions than currently stored.")

        probabilities = self._probabilities()
        probability_weights = [float(value) for value in probabilities]
        indices = tuple(
            int(
                self.random.choices(
                    range(len(self.storage)),
                    weights=probability_weights,
                )[0]
            )
            for _ in range(batch_size)
        )
        weights = np.asarray(
            [
                (len(self.storage) * probabilities[index]) ** (-self.beta)
                for index in indices
            ],
            dtype=np.float32,
        )
        weights /= weights.max(initial=1.0)
        return PrioritizedSample(
            transitions=tuple(self.storage[index] for index in indices),
            indices=indices,
            weights=weights,
        )

    def update_priorities(
        self,
        indices: tuple[int, ...],
        priorities: tuple[float, ...],
    ) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length.")
        for index, priority in zip(indices, priorities, strict=True):
            if index < 0 or index >= len(self.storage):
                raise IndexError(f"priority index out of range: {index}")
            if priority <= 0.0:
                raise ValueError("priority must be positive.")
            self.priorities[index] = priority + self.epsilon

    def _probabilities(self) -> np.ndarray:
        active_priorities = self.priorities[: len(self.storage)]
        scaled = np.power(active_priorities + self.epsilon, self.alpha)
        total = scaled.sum()
        if total <= 0.0:
            return np.full(len(self.storage), 1.0 / len(self.storage), dtype=np.float64)
        return np.asarray(scaled / total, dtype=np.float64)
