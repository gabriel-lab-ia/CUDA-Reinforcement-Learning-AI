from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from cuda_rl.algorithms import PPOLossConfig, compute_gae, compute_ppo_loss
from cuda_rl.benchmarks.hardware import capture_benchmark_metadata
from cuda_rl.benchmarks.schemas import (
    BenchmarkConfig,
    BenchmarkMetadata,
    RawBenchmarkSample,
    write_csv,
    write_json,
)
from cuda_rl.benchmarks.statistics import summarize_distribution
from cuda_rl.replay import PrioritizedReplayBuffer, ReplayBuffer, ReplayTransition


def run_formal_benchmark(config: BenchmarkConfig) -> Path:
    config.validate()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_directory = Path(config.output_directory) / config.suite / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "tables").mkdir()
    (output_directory / "plots").mkdir()

    metadata = capture_benchmark_metadata(config, timestamp=timestamp)
    samples: list[RawBenchmarkSample] = []
    correctness: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    write_json(output_directory / "metadata.json", asdict(metadata))
    write_json(output_directory / "environment.json", _environment_payload(metadata))
    _write_resolved_config(output_directory / "config.resolved.yaml", config)

    for parameters in expand_parameter_grid(config):
        for seed in config.seeds:
            try:
                if config.benchmark_type == "formal_gae":
                    _run_gae_workload(config, parameters, seed, samples, correctness)
                elif config.benchmark_type == "formal_ppo_loss":
                    _run_ppo_workload(config, parameters, seed, samples, correctness)
                elif config.benchmark_type == "formal_replay_buffer":
                    _run_replay_workload(config, parameters, seed, samples)
                elif config.benchmark_type in {
                    "formal_dqn_cartpole",
                    "formal_a2c_cartpole",
                }:
                    _record_training_gap(config, parameters, seed, samples, failures)
                else:
                    raise ValueError(
                        f"unsupported formal benchmark: {config.benchmark_type}"
                    )
            except Exception as exception:
                failures.append(
                    {
                        "suite": config.suite,
                        "benchmark_type": config.benchmark_type,
                        "seed": seed,
                        "parameters": parameters,
                        "error": str(exception),
                    }
                )
                if config.fail_on_unavailable_backend:
                    continue

    aggregates = aggregate_raw_samples(
        samples, reference_backend=config.reference_backend
    )
    write_csv(
        output_directory / "raw_samples.csv", [sample.to_row() for sample in samples]
    )
    _write_jsonl(
        output_directory / "raw_runs.jsonl", [sample.to_row() for sample in samples]
    )
    write_csv(output_directory / "aggregate.csv", aggregates)
    write_json(output_directory / "correctness.json", correctness)
    write_json(output_directory / "failures.json", failures)
    write_json(
        output_directory / "summary.json",
        {
            "suite": config.suite,
            "benchmark_type": config.benchmark_type,
            "sample_count": len(samples),
            "measured_sample_count": sum(
                sample.status == "executed" and not sample.warmup for sample in samples
            ),
            "failure_count": len(failures),
            "formal_campaign_executed": True,
            "results_are_formal_evidence": not failures and bool(samples),
        },
    )
    _write_tables(output_directory / "tables", aggregates)
    _write_plots(output_directory / "plots", aggregates)
    return output_directory


def expand_parameter_grid(config: BenchmarkConfig) -> list[dict[str, object]]:
    if not config.parameter_grid:
        return [dict(config.hyperparameters)]
    keys = sorted(config.parameter_grid)
    rows: list[dict[str, object]] = []
    for values in itertools.product(*(config.parameter_grid[key] for key in keys)):
        parameters: dict[str, object] = dict(config.hyperparameters)
        parameters.update(dict(zip(keys, values, strict=True)))
        rows.append(parameters)
    return rows


def aggregate_raw_samples(
    samples: list[RawBenchmarkSample],
    *,
    reference_backend: str | None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[RawBenchmarkSample]] = {}
    for sample in samples:
        if sample.warmup or sample.status != "executed" or sample.latency_ms is None:
            continue
        key = (
            sample.workload_id,
            sample.operation,
            sample.backend,
            sample.dtype,
            sample.parameters_json,
        )
        grouped.setdefault(key, []).append(sample)

    rows: list[dict[str, object]] = []
    latency_means: dict[tuple[str, str, str], float] = {}
    for key, group in sorted(grouped.items()):
        workload_id, operation, backend, dtype, parameters_json = key
        latencies = [
            sample.latency_ms for sample in group if sample.latency_ms is not None
        ]
        throughputs = [
            float(sample.throughput)
            for sample in group
            if sample.throughput is not None and math.isfinite(sample.throughput)
        ]
        latency = summarize_distribution(latencies)
        throughput = summarize_distribution(throughputs) if throughputs else None
        latency_means[(workload_id, operation, backend)] = latency.mean
        rows.append(
            {
                "workload_id": workload_id,
                "operation": operation,
                "backend": backend,
                "dtype": dtype,
                "parameters_json": parameters_json,
                "count": latency.count,
                "mean_latency_ms": latency.mean,
                "median_latency_ms": latency.median,
                "std_latency_ms": latency.standard_deviation,
                "min_latency_ms": latency.minimum,
                "max_latency_ms": latency.maximum,
                "p5_latency_ms": latency.p5,
                "p25_latency_ms": latency.p25,
                "p75_latency_ms": latency.p75,
                "p95_latency_ms": latency.p95,
                "p99_latency_ms": latency.p99,
                "cv_latency": latency.coefficient_of_variation,
                "ci95_low_latency_ms": latency.confidence_interval_95_low,
                "ci95_high_latency_ms": latency.confidence_interval_95_high,
                "mean_throughput": throughput.mean if throughput else None,
                "throughput_unit": group[0].throughput_unit,
                "speedup_vs_reference": None,
            }
        )

    if reference_backend is None:
        return rows
    for row in rows:
        reference = latency_means.get(
            (
                str(row["workload_id"]),
                str(row["operation"]),
                reference_backend,
            )
        )
        mean_latency = row["mean_latency_ms"]
        if (
            isinstance(mean_latency, int | float)
            and reference is not None
            and float(mean_latency) > 0.0
        ):
            row["speedup_vs_reference"] = reference / float(mean_latency)
    return rows


def _run_gae_workload(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    correctness: list[dict[str, object]],
) -> None:
    timesteps = _int_param(
        parameters,
        "timesteps",
        _int_param(parameters, "rollout_steps", 128),
    )
    num_envs = _int_param(parameters, "num_envs", 1)
    gamma = _float_param(parameters, "gamma", 0.99)
    gae_lambda = _float_param(parameters, "gae_lambda", 0.95)
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=(timesteps, num_envs)).astype(np.float32)
    values = rng.normal(size=(timesteps, num_envs)).astype(np.float32)
    dones = np.zeros((timesteps, num_envs), dtype=np.float32)
    dones[-1, :] = 1.0
    reference_advantages, reference_returns = _gae_numpy(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    for backend in config.backends:
        if backend == "native_cuda":
            _record_unavailable(
                config,
                parameters,
                seed,
                samples,
                backend,
                "No Python binding for native CUDA GAE.",
            )
            continue
        if backend == "torch_cuda" and not torch.cuda.is_available():
            _handle_unavailable(
                config, parameters, seed, samples, backend, "CUDA unavailable"
            )
            continue
        if backend == "numpy":
            candidate_advantages, candidate_returns = (
                reference_advantages,
                reference_returns,
            )
        else:
            device = torch.device("cuda" if backend == "torch_cuda" else "cpu")
            candidate_advantages, candidate_returns = _gae_torch(
                rewards,
                values,
                dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
            )
        check = _correctness_payload(
            config,
            parameters,
            seed,
            backend,
            reference_advantages,
            candidate_advantages,
            reference_returns,
            candidate_returns,
        )
        correctness.append(check)
        if not check["allclose"]:
            continue
        if backend == "numpy":
            callable_factory = _gae_numpy_factory(
                rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda
            )
        else:
            device = torch.device("cuda" if backend == "torch_cuda" else "cpu")
            if config.measure_end_to_end:
                callable_factory = _gae_torch_end_to_end_factory(
                    rewards,
                    values,
                    dones,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    device=device,
                )
            else:
                reward_tensor = torch.as_tensor(rewards, device=device)
                value_tensor = torch.as_tensor(values, device=device)
                done_tensor = torch.as_tensor(dones, device=device)
                callable_factory = _gae_torch_tensor_factory(
                    reward_tensor,
                    value_tensor,
                    done_tensor,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )

        _measure_operation(
            config,
            parameters,
            seed,
            samples,
            backend,
            operation="gae",
            units=timesteps * num_envs,
            throughput_unit="elements_per_second",
            shape=f"{timesteps}x{num_envs}",
            batch_size=None,
            callable_factory=callable_factory,
        )


def _run_ppo_workload(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    correctness: list[dict[str, object]],
) -> None:
    batch_size = _int_param(parameters, "batch_size", 64)
    rng = np.random.default_rng(seed)
    arrays = {
        "old": rng.normal(size=batch_size).astype(np.float32),
        "new": rng.normal(size=batch_size).astype(np.float32),
        "advantages": rng.normal(size=batch_size).astype(np.float32),
        "returns": rng.normal(size=batch_size).astype(np.float32),
        "values": rng.normal(size=batch_size).astype(np.float32),
        "entropies": rng.random(size=batch_size).astype(np.float32),
    }
    reference = _ppo_loss_numpy_like_torch(arrays, torch.device("cpu"))
    for backend in config.backends:
        if backend == "torch_cuda" and not torch.cuda.is_available():
            _handle_unavailable(
                config, parameters, seed, samples, backend, "CUDA unavailable"
            )
            continue
        if backend not in {"torch_cpu", "torch_cuda"}:
            _record_unavailable(
                config,
                parameters,
                seed,
                samples,
                backend,
                "PPO loss supports torch_cpu and torch_cuda.",
            )
            continue
        device = torch.device("cuda" if backend == "torch_cuda" else "cpu")
        candidate = _ppo_loss_numpy_like_torch(arrays, device)
        check = {
            "suite": config.suite,
            "benchmark_type": config.benchmark_type,
            "backend": backend,
            "seed": seed,
            "parameters": parameters,
            "max_absolute_error": float(abs(reference - candidate)),
            "mean_absolute_error": float(abs(reference - candidate)),
            "max_relative_error": float(
                abs(reference - candidate) / max(abs(reference), 1e-12)
            ),
            "allclose": bool(np.allclose(reference, candidate, rtol=1e-5, atol=1e-6)),
            "tolerances": {"rtol": 1e-5, "atol": 1e-6},
        }
        correctness.append(check)
        if not check["allclose"]:
            continue
        for operation_name in ("forward", "forward_backward", "forward_backward_step"):
            _measure_operation(
                config,
                parameters,
                seed,
                samples,
                backend,
                operation=operation_name,
                units=batch_size,
                throughput_unit="samples_per_second",
                shape=str(batch_size),
                batch_size=batch_size,
                callable_factory=_ppo_factory(arrays, device, operation_name),
            )


def _run_replay_workload(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
) -> None:
    capacity = _int_param(parameters, "capacity", 10_000)
    batch_size = _int_param(parameters, "batch_size", 32)
    for buffer_kind in ("uniform", "prioritized"):
        local_parameters = dict(parameters)
        local_parameters["buffer"] = buffer_kind
        for operation in ("insert", "sample", "priority_update", "mixed_cycle"):
            if buffer_kind == "uniform" and operation == "priority_update":
                continue
            _measure_operation(
                config,
                local_parameters,
                seed,
                samples,
                "cpu",
                operation=operation,
                units=batch_size if operation != "insert" else capacity,
                throughput_unit=_replay_unit(operation),
                shape=None,
                batch_size=batch_size,
                callable_factory=_replay_factory(
                    capacity,
                    batch_size,
                    seed,
                    buffer_kind,
                    operation,
                ),
            )


def _measure_operation(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    backend: str,
    *,
    operation: str,
    units: int,
    throughput_unit: str,
    shape: str | None,
    batch_size: int | None,
    callable_factory: Callable[[], Callable[[], object]],
) -> None:
    workload_id = _workload_id(parameters)
    repetitions = config.warmup_repetitions + config.measured_repetitions
    for repetition in range(repetitions):
        warmup = repetition < config.warmup_repetitions
        fn = callable_factory()
        try:
            latency_ms = _time_callable(fn, backend, config.synchronize_cuda)
            status = "executed"
            error = None
        except Exception as exception:
            latency_ms = None
            status = "failed"
            error = str(exception)
        samples.append(
            RawBenchmarkSample(
                suite=config.suite,
                benchmark_type=config.benchmark_type,
                algorithm=config.algorithm,
                workload_id=workload_id,
                operation=operation,
                backend=backend,
                seed=seed,
                repetition=repetition,
                warmup=warmup,
                status=status,
                latency_ms=latency_ms,
                throughput=(
                    units / (latency_ms / 1_000.0)
                    if latency_ms and latency_ms > 0
                    else None
                ),
                throughput_unit=throughput_unit,
                device="cuda" if backend in {"torch_cuda", "native_cuda"} else "cpu",
                dtype=config.dtype,
                shape=shape,
                batch_size=batch_size,
                parameters_json=json.dumps(parameters, sort_keys=True),
                error=error,
            )
        )


def _time_callable(
    fn: Callable[[], object], backend: str, synchronize_cuda: bool
) -> float:
    if backend == "torch_cuda":
        if synchronize_cuda:
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    started_at = time.perf_counter_ns()
    fn()
    ended_at = time.perf_counter_ns()
    return (ended_at - started_at) / 1_000_000.0


def _gae_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * non_terminal - values[step]
        running = delta + gamma * gae_lambda * non_terminal * running
        advantages[step] = running
        next_value = values[step]
    return advantages, advantages + values


def _gae_torch(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    reward_tensor = torch.as_tensor(rewards, device=device)
    value_tensor = torch.as_tensor(values, device=device)
    done_tensor = torch.as_tensor(dones, device=device)
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for env in range(rewards.shape[1]):
        advantage, ret = compute_gae(
            reward_tensor[:, env],
            value_tensor[:, env],
            done_tensor[:, env],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages.append(advantage)
        returns.append(ret)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        torch.stack(advantages, dim=1).detach().cpu().numpy(),
        torch.stack(returns, dim=1).detach().cpu().numpy(),
    )


def _gae_torch_from_tensors(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for env in range(rewards.shape[1]):
        advantage, ret = compute_gae(
            rewards[:, env],
            values[:, env],
            dones[:, env],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages.append(advantage)
        returns.append(ret)
    return torch.stack(advantages, dim=1), torch.stack(returns, dim=1)


def _ppo_loss_numpy_like_torch(
    arrays: dict[str, np.ndarray], device: torch.device
) -> float:
    with torch.no_grad():
        result = compute_ppo_loss(
            new_log_probabilities=torch.as_tensor(arrays["new"], device=device),
            old_log_probabilities=torch.as_tensor(arrays["old"], device=device),
            advantages=torch.as_tensor(arrays["advantages"], device=device),
            returns=torch.as_tensor(arrays["returns"], device=device),
            values=torch.as_tensor(arrays["values"], device=device),
            entropies=torch.as_tensor(arrays["entropies"], device=device),
            config=PPOLossConfig(),
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return float(result.loss.detach().cpu().item())


def _ppo_operation(
    arrays: dict[str, np.ndarray], device: torch.device, operation: str
) -> None:
    values = (
        torch.as_tensor(arrays["values"], device=device)
        .clone()
        .detach()
        .requires_grad_(operation != "forward")
    )
    loss = compute_ppo_loss(
        new_log_probabilities=torch.as_tensor(arrays["new"], device=device),
        old_log_probabilities=torch.as_tensor(arrays["old"], device=device),
        advantages=torch.as_tensor(arrays["advantages"], device=device),
        returns=torch.as_tensor(arrays["returns"], device=device),
        values=values,
        entropies=torch.as_tensor(arrays["entropies"], device=device),
        config=PPOLossConfig(),
    ).loss
    if operation == "forward":
        return
    loss.backward()  # type: ignore[no-untyped-call]
    if operation == "forward_backward_step":
        with torch.no_grad():
            if values.grad is not None:
                values -= 0.001 * values.grad


def _replay_operation(
    capacity: int,
    batch_size: int,
    seed: int,
    buffer_kind: str,
    operation: str,
) -> None:
    rng = np.random.default_rng(seed)
    buffer: ReplayBuffer | PrioritizedReplayBuffer = (
        PrioritizedReplayBuffer(capacity=capacity, seed=seed)
        if buffer_kind == "prioritized"
        else ReplayBuffer(capacity=capacity, seed=seed)
    )
    if operation != "insert":
        _fill_buffer(buffer, capacity, rng)
    if operation == "insert":
        _fill_buffer(buffer, capacity, rng)
    elif operation == "sample":
        buffer.sample(batch_size)
    elif operation == "priority_update":
        if not isinstance(buffer, PrioritizedReplayBuffer):
            return
        sample = buffer.sample(batch_size)
        buffer.update_priorities(
            sample.indices, tuple(float(index + 1) for index in sample.indices)
        )
    elif operation == "mixed_cycle":
        transition = _transition(rng, 0)
        if isinstance(buffer, PrioritizedReplayBuffer):
            buffer.add(transition, priority=1.0)
            sample = buffer.sample(batch_size)
            buffer.update_priorities(
                sample.indices, tuple(float(index + 1) for index in sample.indices)
            )
        else:
            buffer.add(transition)
            buffer.sample(batch_size)


def _fill_buffer(
    buffer: ReplayBuffer | PrioritizedReplayBuffer,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    for index in range(capacity):
        transition = _transition(rng, index)
        if isinstance(buffer, PrioritizedReplayBuffer):
            buffer.add(transition, priority=float(index + 1))
        else:
            buffer.add(transition)


def _transition(rng: np.random.Generator, index: int) -> ReplayTransition:
    return ReplayTransition(
        observation=rng.normal(size=(4,)).astype(np.float32),
        action=index % 2,
        reward=float(rng.normal()),
        next_observation=rng.normal(size=(4,)).astype(np.float32),
        terminated=False,
        truncated=False,
    )


def _correctness_payload(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    backend: str,
    reference_advantages: np.ndarray,
    candidate_advantages: np.ndarray,
    reference_returns: np.ndarray,
    candidate_returns: np.ndarray,
) -> dict[str, object]:
    reference = np.concatenate(
        [reference_advantages.ravel(), reference_returns.ravel()]
    )
    candidate = np.concatenate(
        [candidate_advantages.ravel(), candidate_returns.ravel()]
    )
    absolute = np.abs(reference - candidate)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    return {
        "suite": config.suite,
        "benchmark_type": config.benchmark_type,
        "backend": backend,
        "seed": seed,
        "parameters": parameters,
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean() if absolute.size else 0.0),
        "max_relative_error": float(relative.max(initial=0.0)),
        "allclose": bool(np.allclose(reference, candidate, rtol=1e-5, atol=1e-6)),
        "tolerances": {"rtol": 1e-5, "atol": 1e-6},
    }


def _record_unavailable(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    backend: str,
    reason: str,
) -> None:
    samples.append(
        RawBenchmarkSample(
            suite=config.suite,
            benchmark_type=config.benchmark_type,
            algorithm=config.algorithm,
            workload_id=_workload_id(parameters),
            operation="availability",
            backend=backend,
            seed=seed,
            repetition=-1,
            warmup=False,
            status="not_available",
            latency_ms=None,
            throughput=None,
            throughput_unit="",
            device="cuda" if backend in {"torch_cuda", "native_cuda"} else "cpu",
            dtype=config.dtype,
            shape=None,
            batch_size=None,
            parameters_json=json.dumps(parameters, sort_keys=True),
            error=reason,
        )
    )


def _handle_unavailable(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    backend: str,
    reason: str,
) -> None:
    _record_unavailable(config, parameters, seed, samples, backend, reason)
    if config.fail_on_unavailable_backend:
        raise RuntimeError(reason)


def _record_training_gap(
    config: BenchmarkConfig,
    parameters: dict[str, object],
    seed: int,
    samples: list[RawBenchmarkSample],
    failures: list[dict[str, object]],
) -> None:
    for backend in config.backends:
        _record_unavailable(
            config,
            parameters,
            seed,
            samples,
            backend,
            "Formal CartPole training benchmark adapter is documented but not "
            "executed by this harness yet.",
        )
    failures.append(
        {
            "suite": config.suite,
            "benchmark_type": config.benchmark_type,
            "seed": seed,
            "parameters": parameters,
            "error": "training benchmark adapter not implemented",
        }
    )


def _write_tables(directory: Path, aggregates: list[dict[str, object]]) -> None:
    for name, field in (
        ("latency.md", "mean_latency_ms"),
        ("throughput.md", "mean_throughput"),
        ("speedup.md", "speedup_vs_reference"),
    ):
        lines = ["| Workload | Operation | Backend | Value |", "|---|---|---:|---:|"]
        for row in aggregates:
            value = row[field]
            line = (
                f"| {row['workload_id']} | {row['operation']} | "
                f"{row['backend']} | {_fmt(value)} |"
            )
            lines.append(line)
        (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "learning.md").write_text(
        "Formal CartPole learning benchmarks are scaffolded; no learning values "
        "are published until execution.\n",
        encoding="utf-8",
    )


def _write_plots(directory: Path, aggregates: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    for filename, field, ylabel in (
        ("latency_vs_workload.png", "mean_latency_ms", "Latency (ms)"),
        ("throughput_vs_workload.png", "mean_throughput", "Throughput"),
        ("speedup_vs_workload.png", "speedup_vs_reference", "Speedup"),
    ):
        labels = [f"{row['operation']}:{row['backend']}" for row in aggregates]
        values = [_float_object(row[field]) for row in aggregates]
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.bar(range(len(values)), values)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)
    for filename in ("learning_curves.png", "time_to_threshold.png"):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.text(
            0.5, 0.5, "Awaiting formal training execution", ha="center", va="center"
        )
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_resolved_config(path: Path, config: BenchmarkConfig) -> None:
    try:
        import yaml  # type: ignore[import-untyped]

        payload = yaml.safe_dump(asdict(config), sort_keys=True)
    except ImportError:
        payload = json.dumps(asdict(config), indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")


def _environment_payload(metadata: BenchmarkMetadata) -> dict[str, object]:
    payload = cast(dict[str, object], asdict(metadata))
    payload["nvml_warning"] = (
        "NVML telemetry is optional; unavailable fields are recorded as null."
    )
    return payload


def _workload_id(parameters: dict[str, object]) -> str:
    return ",".join(f"{key}={parameters[key]}" for key in sorted(parameters))


def _replay_unit(operation: str) -> str:
    if operation == "insert":
        return "insertions_per_second"
    if operation == "priority_update":
        return "priority_updates_per_second"
    return "samples_per_second"


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return f"{_float_object(value):.6g}"


def _constant_factory(fn: Callable[[], object]) -> Callable[[], Callable[[], object]]:
    return lambda: fn


def _gae_numpy_factory(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Callable[[], Callable[[], object]]:
    def operation() -> object:
        return _gae_numpy(rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda)

    return _constant_factory(operation)


def _gae_torch_end_to_end_factory(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> Callable[[], Callable[[], object]]:
    def operation() -> object:
        return _gae_torch(
            rewards,
            values,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
        )

    return _constant_factory(operation)


def _gae_torch_tensor_factory(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> Callable[[], Callable[[], object]]:
    def operation() -> object:
        return _gae_torch_from_tensors(
            rewards,
            values,
            dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    return _constant_factory(operation)


def _ppo_factory(
    arrays: dict[str, np.ndarray],
    device: torch.device,
    operation: str,
) -> Callable[[], Callable[[], object]]:
    return lambda: lambda: _ppo_operation(arrays, device, operation)


def _replay_factory(
    capacity: int,
    batch_size: int,
    seed: int,
    buffer_kind: str,
    operation: str,
) -> Callable[[], Callable[[], object]]:
    return lambda: (
        lambda: _replay_operation(
            capacity,
            batch_size,
            seed,
            buffer_kind,
            operation,
        )
    )


def _int_param(parameters: dict[str, object], key: str, default: int) -> int:
    value = parameters.get(key, default)
    if not isinstance(value, int | float | str | bool):
        raise TypeError(f"parameter {key!r} must be scalar.")
    return int(value)


def _float_param(parameters: dict[str, object], key: str, default: float) -> float:
    value = parameters.get(key, default)
    if not isinstance(value, int | float | str | bool):
        raise TypeError(f"parameter {key!r} must be scalar.")
    return float(value)


def _float_object(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return 0.0
