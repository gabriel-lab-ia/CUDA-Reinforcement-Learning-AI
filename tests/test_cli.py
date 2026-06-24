from __future__ import annotations

import json

from cuda_rl.cli import main


def test_cli_profile_outputs_validated_profile(tmp_path, capsys) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "cli-profile",
                "algorithm": "dqn",
                "environment_id": "CartPole-v1",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    main(["profile", str(profile_path)])

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == "cli-profile"
    assert output["algorithm"] == "dqn"


def test_cli_benchmark_campaign_writes_manifest(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "formal.yaml"
    config_path.write_text(
        "\n".join(
            [
                "suite: campaign-formal",
                "benchmark_type: formal_gae",
                "algorithm: gae",
                "backends: [numpy]",
                "seeds: [0]",
                "episodes: 1",
                "mode: formal",
                f"output_directory: {tmp_path / 'benchmarks'}",
                "warmup_repetitions: 0",
                "measured_repetitions: 1",
                "reference_backend: numpy",
                "parameter_grid:",
                "  timesteps: [4]",
                "  num_envs: [1]",
            ]
        ),
        encoding="utf-8",
    )

    try:
        main(["benchmark-campaign", "--configs", str(config_path)])
    except SystemExit as exception:
        assert exception.code == 0

    output = capsys.readouterr().out.strip()
    assert (tmp_path / output / "manifest.json").exists()
