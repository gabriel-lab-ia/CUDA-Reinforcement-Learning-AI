from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda_rl.metrics.aggregates import ScalarSummary, summarize_scalars


@dataclass(frozen=True, slots=True)
class TrainingReport:
    output_directory: Path
    episode_count: int
    reward_summary: ScalarSummary
    length_summary: ScalarSummary
    final_summary: dict[str, Any]

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"# Training Report: {self.output_directory.name}",
                "",
                f"- Episodes: {self.episode_count}",
                f"- Mean reward: {self.reward_summary.mean:.3f}",
                f"- Median reward: {self.reward_summary.median:.3f}",
                f"- Max reward: {self.reward_summary.maximum:.3f}",
                f"- Mean length: {self.length_summary.mean:.3f}",
                f"- Global steps: {self.final_summary.get('global_steps', 'unknown')}",
            ]
        )


def load_training_report(output_directory: Path | str) -> TrainingReport:
    root = Path(output_directory)
    episodes_path = root / "episodes.csv"
    summary_path = root / "summary.json"
    if not episodes_path.exists():
        raise FileNotFoundError(f"missing episodes.csv: {episodes_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")

    rewards: list[float] = []
    lengths: list[float] = []
    with episodes_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rewards.append(float(row["reward"]))
            lengths.append(float(row["length"]))

    if not rewards:
        raise ValueError("episodes.csv does not contain episode rows.")

    return TrainingReport(
        output_directory=root,
        episode_count=len(rewards),
        reward_summary=summarize_scalars(rewards),
        length_summary=summarize_scalars(lengths),
        final_summary=json.loads(summary_path.read_text(encoding="utf-8")),
    )
