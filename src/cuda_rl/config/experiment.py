from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

Algorithm = Literal["reinforce", "dqn", "a2c"]


class ConfigError(ValueError):
    """Raised when an experiment profile is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    max_wall_time_minutes: int | None = None
    max_gpu_memory_gib: float | None = None
    max_checkpoints: int = 5

    def validate(self) -> None:
        if self.max_wall_time_minutes is not None and self.max_wall_time_minutes <= 0:
            raise ConfigError("max_wall_time_minutes must be positive when set.")
        if self.max_gpu_memory_gib is not None and self.max_gpu_memory_gib <= 0:
            raise ConfigError("max_gpu_memory_gib must be positive when set.")
        if self.max_checkpoints <= 0:
            raise ConfigError("max_checkpoints must be positive.")


@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    total_episodes: int
    evaluation_every: int
    checkpoint_every: int
    reward_window: int = 100

    def validate(self) -> None:
        if self.total_episodes <= 0:
            raise ConfigError("total_episodes must be positive.")
        if self.evaluation_every <= 0:
            raise ConfigError("evaluation_every must be positive.")
        if self.checkpoint_every <= 0:
            raise ConfigError("checkpoint_every must be positive.")
        if self.reward_window <= 0:
            raise ConfigError("reward_window must be positive.")


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    name: str
    algorithm: Algorithm
    environment_id: str
    seed: int
    device: str = "auto"
    tags: tuple[str, ...] = field(default_factory=tuple)
    schedule: TrainingSchedule = field(
        default_factory=lambda: TrainingSchedule(
            total_episodes=500,
            evaluation_every=25,
            checkpoint_every=50,
        )
    )
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    hyperparameters: dict[str, int | float | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("name is required.")
        if self.algorithm not in {"reinforce", "dqn", "a2c"}:
            raise ConfigError(f"unsupported algorithm: {self.algorithm}")
        if not self.environment_id:
            raise ConfigError("environment_id is required.")
        if self.seed < 0:
            raise ConfigError("seed must be non-negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError("device must be one of: auto, cpu, cuda.")
        self.schedule.validate()
        self.resource_budget.validate()

    def run_key(self) -> str:
        tags = "-".join(self.tags) if self.tags else "default"
        return (
            f"{self.name}-{self.algorithm}-{self.environment_id}-"
            f"seed-{self.seed}-{tags}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "environment_id": self.environment_id,
            "seed": self.seed,
            "device": self.device,
            "tags": list(self.tags),
            "schedule": {
                "total_episodes": self.schedule.total_episodes,
                "evaluation_every": self.schedule.evaluation_every,
                "checkpoint_every": self.schedule.checkpoint_every,
                "reward_window": self.schedule.reward_window,
            },
            "resource_budget": {
                "max_wall_time_minutes": self.resource_budget.max_wall_time_minutes,
                "max_gpu_memory_gib": self.resource_budget.max_gpu_memory_gib,
                "max_checkpoints": self.resource_budget.max_checkpoints,
            },
            "hyperparameters": dict(self.hyperparameters),
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExperimentProfile:
        schedule_raw = _mapping(raw.get("schedule", {}), "schedule")
        budget_raw = _mapping(raw.get("resource_budget", {}), "resource_budget")
        profile = cls(
            name=str(raw.get("name", raw.get("environment_id", "experiment"))),
            algorithm=_algorithm(str(raw.get("algorithm", "dqn"))),
            environment_id=str(raw.get("environment_id", raw.get("env", ""))),
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "auto")),
            tags=tuple(str(tag) for tag in raw.get("tags", ())),
            schedule=TrainingSchedule(
                total_episodes=int(
                    schedule_raw.get("total_episodes", raw.get("total_episodes", 500))
                ),
                evaluation_every=int(
                    schedule_raw.get(
                        "evaluation_every",
                        raw.get("evaluation_every", 25),
                    )
                ),
                checkpoint_every=int(
                    schedule_raw.get(
                        "checkpoint_every",
                        raw.get("checkpoint_every", 50),
                    )
                ),
                reward_window=int(
                    schedule_raw.get("reward_window", raw.get("reward_window", 100))
                ),
            ),
            resource_budget=ResourceBudget(
                max_wall_time_minutes=_optional_int(
                    budget_raw.get("max_wall_time_minutes")
                ),
                max_gpu_memory_gib=_optional_float(
                    budget_raw.get("max_gpu_memory_gib")
                ),
                max_checkpoints=int(budget_raw.get("max_checkpoints", 5)),
            ),
            hyperparameters=_hyperparameters(raw.get("hyperparameters", {})),
        )
        profile.validate()
        return profile


def load_experiment_profile(path: Path | str) -> ExperimentProfile:
    source = Path(path)
    if not source.exists():
        raise ConfigError(f"profile does not exist: {source}")
    if source.suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".toml":
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    else:
        raise ConfigError("supported profile formats: .json, .toml")
    return ExperimentProfile.from_mapping(_mapping(raw, "profile"))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _algorithm(value: str) -> Algorithm:
    if value not in {"reinforce", "dqn", "a2c"}:
        raise ConfigError(f"unsupported algorithm: {value}")
    return cast(Algorithm, value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _hyperparameters(value: Any) -> dict[str, int | float | str | bool]:
    raw = _mapping(value, "hyperparameters")
    normalized: dict[str, int | float | str | bool] = {}
    for key, item in raw.items():
        if not isinstance(item, bool | int | float | str):
            raise ConfigError(f"hyperparameter {key!r} must be scalar.")
        normalized[str(key)] = item
    return normalized
