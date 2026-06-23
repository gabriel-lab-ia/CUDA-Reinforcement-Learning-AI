from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

BenchmarkBackend = Literal["cpu", "torch_cpu", "torch_cuda", "native_cuda", "sb3"]
BenchmarkMode = Literal["smoke", "formal"]
BenchmarkStatus = Literal["executed", "skipped", "not_available", "failed"]


@dataclass(frozen=True, slots=True)
class TelemetrySettings:
    enabled: bool = True
    interval_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    suite: str
    benchmark_type: str
    algorithm: str
    environment_id: str | None
    backends: tuple[BenchmarkBackend, ...]
    seeds: tuple[int, ...]
    episodes: int
    device: str = "cpu"
    dtype: str = "float32"
    mode: BenchmarkMode = "smoke"
    output_directory: str = "reports/benchmarks"
    hyperparameters: dict[str, int | float | str | bool] = field(default_factory=dict)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)

    def validate(self) -> None:
        if not self.suite:
            raise ValueError("suite is required.")
        if not self.benchmark_type:
            raise ValueError("benchmark_type is required.")
        if not self.algorithm:
            raise ValueError("algorithm is required.")
        if not self.backends:
            raise ValueError("at least one backend is required.")
        if not self.seeds:
            raise ValueError("at least one seed is required.")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.mode == "formal" and len(self.seeds) < 10:
            raise ValueError("formal benchmark reports require at least 10 seeds.")
        if self.mode == "smoke" and len(self.seeds) < 1:
            raise ValueError("smoke benchmark requires at least one seed.")


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    suite: str
    timestamp: str
    commit_sha: str
    working_tree_status: str
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    platform: str
    cpu: str
    gpu: str | None
    config: dict[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    suite: str
    benchmark_type: str
    algorithm: str
    backend: BenchmarkBackend
    seed: int
    status: BenchmarkStatus
    device: str
    dtype: str
    episodes: int
    environment_steps: int
    training_updates: int
    final_reward: float | None
    mean_reward: float | None
    std_reward: float | None
    reward_window: float | None
    sample_efficiency: float | None
    wall_clock_seconds: float | None
    steps_per_second: float | None
    updates_per_second: float | None
    inference_latency_ms: float | None
    training_latency_ms: float | None
    collection_seconds: float | None
    optimization_seconds: float | None
    evaluation_seconds: float | None
    max_ram_mib: float | None
    max_vram_mib: float | None
    gpu_utilization_mean: float | None
    gpu_utilization_max: float | None
    temperature_celsius: float | None
    power_watts: float | None
    error: str | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    suite: str
    benchmark_type: str
    algorithm: str
    backend: BenchmarkBackend
    status: str
    seed_count: int
    success_rate: float
    mean_reward: float | None
    mean_reward_ci95_low: float | None
    mean_reward_ci95_high: float | None
    mean_wall_clock_seconds: float | None
    mean_steps_per_second: float | None
    variance_between_seeds: float | None
    coefficient_of_variation: float | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
