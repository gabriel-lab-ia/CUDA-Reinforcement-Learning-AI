from __future__ import annotations

from cuda_rl.config import ExperimentProfile
from cuda_rl.experiments import ExperimentRegistry


def test_experiment_registry_tracks_lifecycle(tmp_path) -> None:
    profile = ExperimentProfile.from_mapping(
        {
            "name": "registry",
            "algorithm": "dqn",
            "environment_id": "CartPole-v1",
            "seed": 11,
        }
    )
    registry = ExperimentRegistry(tmp_path)

    registry.register_profile(profile)
    running = registry.start_run(profile, output_directory=tmp_path / "run")
    completed = registry.finish_run(running, status="completed")

    assert completed.status == "completed"
    assert registry.latest_run() == completed
    assert registry.runs_by_status("completed") == [completed]
