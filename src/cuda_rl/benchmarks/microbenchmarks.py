from __future__ import annotations

import time

import numpy as np
import torch

from cuda_rl.algorithms import PPOLossConfig, compute_gae, compute_ppo_loss
from cuda_rl.benchmarks.schemas import (
    BenchmarkConfig,
    BenchmarkRunResult,
    BenchmarkStatus,
)
from cuda_rl.monitoring.telemetry import capture_telemetry
from cuda_rl.replay import PrioritizedReplayBuffer, ReplayBuffer, ReplayTransition


def run_microbenchmark(config: BenchmarkConfig) -> list[BenchmarkRunResult]:
    runs: list[BenchmarkRunResult] = []
    for backend in config.backends:
        for seed in config.seeds:
            if (
                backend in {"torch_cuda", "native_cuda"}
                and not torch.cuda.is_available()
            ):
                runs.append(_skipped_run(config, backend, seed, "CUDA unavailable"))
                continue
            started_at = time.perf_counter()
            telemetry_before = capture_telemetry()
            try:
                if config.benchmark_type == "gae_microbenchmark":
                    steps, updates = _run_gae(config, seed, backend)
                elif config.benchmark_type == "ppo_microbenchmark":
                    steps, updates = _run_ppo(config, seed, backend)
                elif config.benchmark_type == "replay_buffer_benchmark":
                    steps, updates = _run_replay(config, seed, backend)
                else:
                    raise ValueError(
                        f"unsupported microbenchmark: {config.benchmark_type}"
                    )
                status: BenchmarkStatus = "executed"
                error = None
            except Exception as exception:
                steps = 0
                updates = 0
                status = "failed"
                error = str(exception)
            elapsed = time.perf_counter() - started_at
            telemetry_after = capture_telemetry()
            runs.append(
                BenchmarkRunResult(
                    suite=config.suite,
                    benchmark_type=config.benchmark_type,
                    algorithm=config.algorithm,
                    backend=backend,
                    seed=seed,
                    status=status,
                    device=_device_for_backend(backend),
                    dtype=config.dtype,
                    episodes=config.episodes,
                    environment_steps=steps,
                    training_updates=updates,
                    final_reward=None,
                    mean_reward=None,
                    std_reward=None,
                    reward_window=None,
                    sample_efficiency=None,
                    wall_clock_seconds=elapsed,
                    steps_per_second=(steps / elapsed if elapsed > 0.0 else None),
                    updates_per_second=(updates / elapsed if elapsed > 0.0 else None),
                    inference_latency_ms=None,
                    training_latency_ms=(elapsed * 1_000.0 / max(updates, 1)),
                    collection_seconds=None,
                    optimization_seconds=elapsed,
                    evaluation_seconds=None,
                    max_ram_mib=max(
                        telemetry_before.cpu.process_memory_rss_mib,
                        telemetry_after.cpu.process_memory_rss_mib,
                    ),
                    max_vram_mib=telemetry_after.gpu.max_allocated_mib,
                    gpu_utilization_mean=None,
                    gpu_utilization_max=None,
                    temperature_celsius=None,
                    power_watts=None,
                    error=error,
                )
            )
    return runs


def _run_gae(
    config: BenchmarkConfig,
    seed: int,
    backend: str,
) -> tuple[int, int]:
    torch.manual_seed(seed)
    steps = int(config.hyperparameters.get("rollout_steps", 512))
    repeats = int(config.hyperparameters.get("repeats", config.episodes))
    device = torch.device("cuda" if backend == "torch_cuda" else "cpu")
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    rewards = torch.randn(steps, device=device, dtype=dtype)
    values = torch.randn(steps, device=device, dtype=dtype)
    dones = torch.zeros(steps, device=device, dtype=dtype)
    dones[-1] = 1.0
    for _ in range(repeats):
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=0.99,
            gae_lambda=0.95,
        )
        _assert_finite(advantages)
        _assert_finite(returns)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return steps * repeats, repeats


def _run_ppo(
    config: BenchmarkConfig,
    seed: int,
    backend: str,
) -> tuple[int, int]:
    torch.manual_seed(seed)
    batch_size = int(config.hyperparameters.get("batch_size", 512))
    repeats = int(config.hyperparameters.get("repeats", config.episodes))
    device = torch.device("cuda" if backend == "torch_cuda" else "cpu")
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    old_log_probabilities = torch.randn(batch_size, device=device, dtype=dtype)
    new_log_probabilities = old_log_probabilities + 0.01 * torch.randn(
        batch_size,
        device=device,
        dtype=dtype,
    )
    advantages = torch.randn(batch_size, device=device, dtype=dtype)
    returns = torch.randn(batch_size, device=device, dtype=dtype)
    values = torch.randn(batch_size, device=device, dtype=dtype)
    entropies = torch.rand(batch_size, device=device, dtype=dtype)
    for _ in range(repeats):
        result = compute_ppo_loss(
            new_log_probabilities=new_log_probabilities,
            old_log_probabilities=old_log_probabilities,
            advantages=advantages,
            returns=returns,
            values=values,
            entropies=entropies,
            config=PPOLossConfig(),
        )
        _assert_finite(result.loss)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * repeats, repeats


def _run_replay(
    config: BenchmarkConfig,
    seed: int,
    backend: str,
) -> tuple[int, int]:
    if backend not in {"cpu", "torch_cpu"}:
        raise ValueError("replay benchmark currently supports CPU backends only.")
    capacity = int(config.hyperparameters.get("capacity", 10_000))
    batch_size = int(config.hyperparameters.get("batch_size", 128))
    repeats = int(config.hyperparameters.get("repeats", config.episodes))
    prioritized = bool(config.hyperparameters.get("prioritized", True))
    rng = np.random.default_rng(seed)
    buffer = (
        PrioritizedReplayBuffer(capacity=capacity, seed=seed)
        if prioritized
        else ReplayBuffer(capacity=capacity, seed=seed)
    )
    for index in range(capacity):
        transition = ReplayTransition(
            observation=rng.normal(size=(4,)).astype(np.float32),
            action=index % 2,
            reward=float(rng.normal()),
            next_observation=rng.normal(size=(4,)).astype(np.float32),
            terminated=False,
            truncated=False,
        )
        if isinstance(buffer, PrioritizedReplayBuffer):
            buffer.add(transition, priority=float(index + 1))
        else:
            buffer.add(transition)
    for _ in range(repeats):
        buffer.sample(batch_size)
    return capacity + repeats * batch_size, repeats


def _skipped_run(
    config: BenchmarkConfig,
    backend: str,
    seed: int,
    reason: str,
) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        suite=config.suite,
        benchmark_type=config.benchmark_type,
        algorithm=config.algorithm,
        backend=backend,  # type: ignore[arg-type]
        seed=seed,
        status="not_available",
        device=_device_for_backend(backend),
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
        error=reason,
    )


def _device_for_backend(backend: str) -> str:
    return "cuda" if backend in {"torch_cuda", "native_cuda"} else "cpu"


def _assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("benchmark produced NaN or Inf.")
