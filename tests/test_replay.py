from __future__ import annotations

import numpy as np

from cuda_rl.replay import PrioritizedReplayBuffer, ReplayBuffer, ReplayTransition


def make_transition(index: int) -> ReplayTransition:
    observation = np.asarray([index, index + 1], dtype=np.float32)
    return ReplayTransition(
        observation=observation,
        action=index % 2,
        reward=float(index),
        next_observation=observation + 1.0,
        terminated=False,
        truncated=False,
    )


def test_replay_buffer_samples_transitions() -> None:
    buffer = ReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.add(make_transition(index))

    sample = buffer.sample(2)

    assert len(sample) == 2
    assert len(buffer) == 4


def test_prioritized_replay_samples_with_weights_and_indices() -> None:
    buffer = PrioritizedReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.add(make_transition(index), priority=float(index + 1))

    sample = buffer.sample(3)

    assert len(sample.transitions) == 3
    assert len(sample.indices) == 3
    assert sample.weights.shape == (3,)
    assert float(sample.weights.max()) <= 1.0


def test_prioritized_replay_updates_priorities() -> None:
    buffer = PrioritizedReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.add(make_transition(index))

    sample = buffer.sample(2)
    buffer.update_priorities(sample.indices, (10.0, 11.0))

    for index in sample.indices:
        assert buffer.priorities[index] > 1.0
