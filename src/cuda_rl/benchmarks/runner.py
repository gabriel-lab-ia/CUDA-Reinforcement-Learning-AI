from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from cuda_rl.benchmarks.aggregation import aggregate_runs
from cuda_rl.benchmarks.config import load_benchmark_config
from cuda_rl.benchmarks.formal import run_formal_benchmark
from cuda_rl.benchmarks.hardware import capture_benchmark_metadata
from cuda_rl.benchmarks.microbenchmarks import run_microbenchmark
from cuda_rl.benchmarks.reporting import write_comparison_report, write_figures
from cuda_rl.benchmarks.schemas import (
    BenchmarkAggregate,
    BenchmarkConfig,
    BenchmarkMetadata,
    BenchmarkRunResult,
    write_csv,
    write_json,
)


def run_benchmark_config(path: Path | str) -> Path:
    return run_benchmark(load_benchmark_config(path))


def run_benchmark(config: BenchmarkConfig) -> Path:
    config.validate()
    if config.benchmark_type.startswith("formal_"):
        return run_formal_benchmark(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_directory = Path(config.output_directory) / config.suite / timestamp
    metadata = capture_benchmark_metadata(config, timestamp=timestamp)
    runs = _execute(config)
    aggregates = aggregate_runs(runs)
    _write_outputs(output_directory, metadata, runs, aggregates)
    return output_directory


def _execute(config: BenchmarkConfig) -> list[BenchmarkRunResult]:
    if config.benchmark_type in {
        "gae_microbenchmark",
        "ppo_microbenchmark",
        "replay_buffer_benchmark",
    }:
        return run_microbenchmark(config)
    return [
        BenchmarkRunResult(
            suite=config.suite,
            benchmark_type=config.benchmark_type,
            algorithm=config.algorithm,
            backend=backend,
            seed=seed,
            status="not_available",
            device=config.device,
            dtype=config.dtype,
            episodes=config.episodes,
            environment_steps=0,
            training_updates=0,
            final_reward=None,
            mean_reward=None,
            std_reward=None,
            reward_window=None,
            sample_efficiency=None,
            wall_clock_seconds=None,
            steps_per_second=None,
            updates_per_second=None,
            inference_latency_ms=None,
            training_latency_ms=None,
            collection_seconds=None,
            optimization_seconds=None,
            evaluation_seconds=None,
            max_ram_mib=None,
            max_vram_mib=None,
            gpu_utilization_mean=None,
            gpu_utilization_max=None,
            temperature_celsius=None,
            power_watts=None,
            error="End-to-end RL benchmark adapter is scaffolded but not executed.",
        )
        for backend in config.backends
        for seed in config.seeds
    ]


def _write_outputs(
    output_directory: Path,
    metadata: BenchmarkMetadata,
    runs: list[BenchmarkRunResult],
    aggregates: list[BenchmarkAggregate],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "figures").mkdir(exist_ok=True)
    write_json(output_directory / "metadata.json", asdict(metadata))
    write_csv(output_directory / "runs.csv", [run.to_row() for run in runs])
    write_csv(
        output_directory / "aggregate.csv",
        [aggregate.to_row() for aggregate in aggregates],
    )
    write_json(
        output_directory / "summary.json",
        {
            "suite": metadata.suite,
            "run_count": len(runs),
            "executed_count": sum(run.status == "executed" for run in runs),
            "not_available_count": sum(run.status == "not_available" for run in runs),
            "failed_count": sum(run.status == "failed" for run in runs),
        },
    )
    write_comparison_report(
        output_directory / "comparison.md",
        metadata=metadata,
        runs=runs,
        aggregates=aggregates,
    )
    write_figures(output_directory / "figures", aggregates=aggregates)
