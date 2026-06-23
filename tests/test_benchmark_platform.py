from __future__ import annotations

import json

import pytest

from cuda_rl.benchmarks.aggregation import aggregate_runs
from cuda_rl.benchmarks.config import benchmark_config_from_mapping
from cuda_rl.benchmarks.runner import run_benchmark
from cuda_rl.benchmarks.schemas import BenchmarkRunResult
from cuda_rl.benchmarks.statistics import summarize_distribution


def test_benchmark_config_rejects_formal_report_with_too_few_seeds() -> None:
    with pytest.raises(ValueError, match="at least 10 seeds"):
        benchmark_config_from_mapping(
            {
                "suite": "formal",
                "benchmark_type": "gae_microbenchmark",
                "algorithm": "gae",
                "backends": ["torch_cpu"],
                "seeds": [0],
                "episodes": 1,
                "mode": "formal",
            }
        )


def test_bootstrap_summary_contains_confidence_interval() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0], bootstrap_resamples=100)

    assert summary.count == 3
    assert summary.confidence_interval_95_low <= summary.mean
    assert summary.confidence_interval_95_high >= summary.mean


def test_aggregate_runs_summarizes_executed_results() -> None:
    runs = [
        _run(seed=0, mean_reward=10.0, steps_per_second=100.0),
        _run(seed=1, mean_reward=20.0, steps_per_second=200.0),
    ]

    aggregate = aggregate_runs(runs)[0]

    assert aggregate.seed_count == 2
    assert aggregate.success_rate == 1.0
    assert aggregate.mean_reward == 15.0


@pytest.mark.benchmark
def test_benchmark_runner_writes_expected_artifacts(tmp_path) -> None:
    config = benchmark_config_from_mapping(
        {
            "suite": "test-gae",
            "benchmark_type": "gae_microbenchmark",
            "algorithm": "gae",
            "backends": ["torch_cpu"],
            "seeds": [0, 1],
            "episodes": 2,
            "output_directory": str(tmp_path),
            "hyperparameters": {"rollout_steps": 8, "repeats": 2},
        }
    )

    output_directory = run_benchmark(config)

    assert (output_directory / "metadata.json").exists()
    assert (output_directory / "runs.csv").exists()
    assert (output_directory / "aggregate.csv").exists()
    assert (output_directory / "summary.json").exists()
    assert (output_directory / "comparison.md").exists()
    assert (output_directory / "figures" / "steps_per_second.png").exists()
    summary = json.loads((output_directory / "summary.json").read_text())
    assert summary["executed_count"] == 2


def _run(seed: int, mean_reward: float, steps_per_second: float) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        suite="suite",
        benchmark_type="type",
        algorithm="algo",
        backend="cpu",
        seed=seed,
        status="executed",
        device="cpu",
        dtype="float32",
        episodes=1,
        environment_steps=10,
        training_updates=1,
        final_reward=mean_reward,
        mean_reward=mean_reward,
        std_reward=0.0,
        reward_window=mean_reward,
        sample_efficiency=mean_reward / 10.0,
        wall_clock_seconds=1.0,
        steps_per_second=steps_per_second,
        updates_per_second=1.0,
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
    )
