from __future__ import annotations

import json

from cuda_rl.reports import load_training_report


def test_load_training_report_summarizes_episode_metrics(tmp_path) -> None:
    (tmp_path / "episodes.csv").write_text(
        "episode,reward,length,loss,epsilon,elapsed_seconds,global_step,moving_average_reward\n"
        "1,10,20,,1.0,0.1,20,10\n"
        "2,30,40,0.2,0.9,0.2,60,20\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"global_steps": 60}),
        encoding="utf-8",
    )

    report = load_training_report(tmp_path)

    assert report.episode_count == 2
    assert report.reward_summary.mean == 20
    assert "Global steps: 60" in report.to_markdown()
