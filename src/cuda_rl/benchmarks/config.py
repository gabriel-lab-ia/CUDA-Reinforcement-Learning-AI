from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from cuda_rl.benchmarks.schemas import (
    BenchmarkBackend,
    BenchmarkConfig,
    BenchmarkMode,
    GridValue,
    TelemetrySettings,
)


def load_benchmark_config(path: Path | str) -> BenchmarkConfig:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"benchmark config does not exist: {source}")
    if source.suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif source.suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("benchmark config must be .yaml, .yml, or .json")
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must be a mapping.")
    return benchmark_config_from_mapping(raw)


def benchmark_config_from_mapping(raw: dict[str, Any]) -> BenchmarkConfig:
    telemetry_raw = _mapping(raw.get("telemetry", {}), "telemetry")
    config = BenchmarkConfig(
        suite=str(raw["suite"]),
        benchmark_type=str(raw.get("benchmark_type", "microbenchmark")),
        algorithm=str(raw.get("algorithm", "unknown")),
        environment_id=_optional_str(raw.get("environment_id")),
        backends=tuple(_backend(item) for item in raw.get("backends", ("cpu",))),
        seeds=tuple(int(seed) for seed in raw.get("seeds", (0,))),
        episodes=int(raw.get("episodes", 1)),
        device=str(raw.get("device", "cpu")),
        dtype=str(raw.get("dtype", "float32")),
        mode=_mode(str(raw.get("mode", "smoke"))),
        output_directory=str(raw.get("output_directory", "reports/benchmarks")),
        hyperparameters=_hyperparameters(raw.get("hyperparameters", {})),
        warmup_repetitions=int(
            raw.get("warmup_repetitions", raw.get("warmup_iterations", 3))
        ),
        measured_repetitions=int(
            raw.get("measured_repetitions", raw.get("measured_iterations", 10))
        ),
        synchronize_cuda=bool(raw.get("synchronize_cuda", True)),
        measure_end_to_end=bool(raw.get("measure_end_to_end", False)),
        reference_backend=_optional_str(raw.get("reference_backend")),
        fail_on_unavailable_backend=bool(raw.get("fail_on_unavailable_backend", False)),
        record_raw_samples=bool(raw.get("record_raw_samples", True)),
        parameter_grid=_parameter_grid(raw.get("parameter_grid", {})),
        telemetry=TelemetrySettings(
            enabled=bool(telemetry_raw.get("enabled", True)),
            interval_seconds=float(telemetry_raw.get("interval_seconds", 1.0)),
        ),
    )
    config.validate()
    return config


def _backend(value: object) -> BenchmarkBackend:
    normalized = str(value)
    allowed = {"cpu", "numpy", "torch_cpu", "torch_cuda", "native_cuda", "sb3"}
    if normalized not in allowed:
        raise ValueError(f"unsupported benchmark backend: {normalized}")
    return cast(BenchmarkBackend, normalized)


def _mode(value: str) -> BenchmarkMode:
    if value not in {"smoke", "formal"}:
        raise ValueError(f"unsupported benchmark mode: {value}")
    return cast(BenchmarkMode, value)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _hyperparameters(value: object) -> dict[str, int | float | str | bool]:
    raw = _mapping(value, "hyperparameters")
    normalized: dict[str, int | float | str | bool] = {}
    for key, item in raw.items():
        if not isinstance(item, str | int | float | bool):
            raise ValueError(f"hyperparameter {key!r} must be scalar.")
        normalized[str(key)] = item
    return normalized


def _parameter_grid(value: object) -> dict[str, tuple[GridValue, ...]]:
    raw = _mapping(value, "parameter_grid")
    grid: dict[str, tuple[GridValue, ...]] = {}
    for key, item in raw.items():
        values = item if isinstance(item, list | tuple) else [item]
        normalized: list[GridValue] = []
        for scalar in values:
            if not isinstance(scalar, str | int | float | bool):
                raise ValueError(f"parameter_grid {key!r} values must be scalar.")
            normalized.append(scalar)
        grid[str(key)] = tuple(normalized)
    return grid


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
