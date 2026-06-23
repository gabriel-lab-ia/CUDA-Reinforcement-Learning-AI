from __future__ import annotations

import json

import pytest

from cuda_rl.config import ConfigError, ExperimentProfile, load_experiment_profile


def test_experiment_profile_builds_from_mapping() -> None:
    profile = ExperimentProfile.from_mapping(
        {
            "name": "cartpole-baseline",
            "algorithm": "dqn",
            "environment_id": "CartPole-v1",
            "seed": 7,
            "tags": ["baseline", "cpu"],
            "schedule": {
                "total_episodes": 10,
                "evaluation_every": 2,
                "checkpoint_every": 5,
            },
            "hyperparameters": {"learning_rate": 0.001, "double_dqn": True},
        }
    )

    assert profile.run_key() == "cartpole-baseline-dqn-CartPole-v1-seed-7-baseline-cpu"
    assert profile.schedule.total_episodes == 10
    assert profile.hyperparameters["double_dqn"] is True


def test_experiment_profile_rejects_invalid_algorithm() -> None:
    with pytest.raises(ConfigError, match="unsupported algorithm"):
        ExperimentProfile.from_mapping(
            {
                "name": "bad",
                "algorithm": "ppo",
                "environment_id": "CartPole-v1",
            }
        )


def test_load_experiment_profile_from_json(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "name": "json-profile",
                "algorithm": "a2c",
                "environment_id": "CartPole-v1",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    profile = load_experiment_profile(path)

    assert profile.name == "json-profile"
    assert profile.algorithm == "a2c"
