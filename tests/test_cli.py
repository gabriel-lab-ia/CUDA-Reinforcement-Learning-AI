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
