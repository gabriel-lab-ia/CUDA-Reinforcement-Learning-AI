from __future__ import annotations

import json

import pytest

from cuda_rl.benchmarks.aggregation import aggregate_runs
from cuda_rl.benchmarks.config import benchmark_config_from_mapping
from cuda_rl.benchmarks.formal import (
    _time_callable,
    aggregate_raw_samples,
    expand_parameter_grid,
)
from cuda_rl.benchmarks.runner import run_benchmark
from cuda_rl.benchmarks.schemas import BenchmarkRunResult, RawBenchmarkSample
from cuda_rl.benchmarks.statistics import percentile, summarize_distribution


def test_benchmark_config_reads_legacy_and_formal_fields() -> None:
    config = benchmark_config_from_mapping(
        {
            "suite": "formal",
            "benchmark_type": "formal_gae",
            "algorithm": "gae",
            "backends": ["numpy"],
            "seeds": [0],
            "episodes": 1,
            "mode": "formal",
            "warmup_repetitions": 50,
            "measured_repetitions": 200,
            "reference_backend": "numpy",
            "parameter_grid": {"timesteps": [128, 512], "num_envs": [1, 8]},
        }
    )

    assert config.warmup_repetitions == 50
    assert config.measured_repetitions == 200
    assert config.reference_backend == "numpy"
    assert len(expand_parameter_grid(config)) == 4


def test_bootstrap_summary_contains_confidence_interval() -> None:
    summary = summarize_distribution([1.0, 2.0, 3.0], bootstrap_resamples=100)

    assert summary.count == 3
    assert summary.p25 == 1.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert summary.confidence_interval_95_low <= summary.mean
    assert summary.confidence_interval_95_high >= summary.mean


def test_formal_aggregation_excludes_warmup_and_computes_speedup() -> None:
    samples = [
        _sample("numpy", 0, True, 10.0),
        _sample("numpy", 1, False, 10.0),
        _sample("torch_cpu", 0, True, 5.0),
        _sample("torch_cpu", 1, False, 5.0),
    ]

    rows = aggregate_raw_samples(samples, reference_backend="numpy")

    by_backend = {str(row["backend"]): row for row in rows}
    assert by_backend["numpy"]["count"] == 1
    assert by_backend["torch_cpu"]["speedup_vs_reference"] == 2.0


def test_cuda_timing_uses_events_and_synchronizes(monkeypatch) -> None:
    calls: list[str] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            calls.append("record")

        def elapsed_time(self, other: object) -> float:
            assert isinstance(other, FakeEvent)
            return 1.25

    monkeypatch.setattr("torch.cuda.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr("torch.cuda.Event", FakeEvent)

    latency = _time_callable(lambda: calls.append("work"), "torch_cuda", True)

    assert latency == 1.25
    assert calls == ["sync", "record", "work", "record", "sync"]


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


@pytest.mark.benchmark
def test_formal_runner_writes_raw_samples_tables_and_plots(tmp_path) -> None:
    config = benchmark_config_from_mapping(
        {
            "suite": "formal-test-gae",
            "benchmark_type": "formal_gae",
            "algorithm": "gae",
            "backends": ["numpy", "native_cuda"],
            "seeds": [0],
            "episodes": 1,
            "output_directory": str(tmp_path),
            "mode": "formal",
            "warmup_repetitions": 1,
            "measured_repetitions": 2,
            "reference_backend": "numpy",
            "parameter_grid": {"timesteps": [4], "num_envs": [1]},
        }
    )

    output_directory = run_benchmark(config)

    assert (output_directory / "raw_samples.csv").exists()
    assert (output_directory / "raw_runs.jsonl").exists()
    assert (output_directory / "correctness.json").exists()
    assert (output_directory / "failures.json").exists()
    assert (output_directory / "tables" / "latency.md").exists()
    assert (output_directory / "plots" / "latency_vs_workload.png").exists()
    samples = (output_directory / "raw_samples.csv").read_text(encoding="utf-8")
    assert "not_available" in samples


def _sample(
    backend: str,
    repetition: int,
    warmup: bool,
    latency_ms: float,
) -> RawBenchmarkSample:
    return RawBenchmarkSample(
        suite="suite",
        benchmark_type="formal_gae",
        algorithm="gae",
        workload_id="timesteps=4",
        operation="gae",
        backend=backend,
        seed=0,
        repetition=repetition,
        warmup=warmup,
        status="executed",
        latency_ms=latency_ms,
        throughput=1000.0 / latency_ms,
        throughput_unit="elements_per_second",
        device="cpu",
        dtype="float32",
        shape="4x1",
        batch_size=None,
        parameters_json='{"timesteps": 4}',
    )


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
