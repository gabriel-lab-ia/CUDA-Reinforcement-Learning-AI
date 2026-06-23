from __future__ import annotations

import statistics

from cuda_rl.benchmarks.schemas import (
    BenchmarkAggregate,
    BenchmarkBackend,
    BenchmarkRunResult,
)
from cuda_rl.benchmarks.statistics import summarize_distribution


def aggregate_runs(runs: list[BenchmarkRunResult]) -> list[BenchmarkAggregate]:
    groups: dict[tuple[str, str, str, BenchmarkBackend], list[BenchmarkRunResult]] = {}
    for run in runs:
        key = (run.suite, run.benchmark_type, run.algorithm, run.backend)
        groups.setdefault(key, []).append(run)

    aggregates: list[BenchmarkAggregate] = []
    for (suite, benchmark_type, algorithm, backend), group in sorted(groups.items()):
        executed = [run for run in group if run.status == "executed"]
        rewards = [run.mean_reward for run in executed if run.mean_reward is not None]
        wall_times = [
            run.wall_clock_seconds
            for run in executed
            if run.wall_clock_seconds is not None
        ]
        throughputs = [
            run.steps_per_second for run in executed if run.steps_per_second is not None
        ]
        reward_summary = summarize_distribution(rewards) if rewards else None
        wall_summary = summarize_distribution(wall_times) if wall_times else None
        throughput_summary = (
            summarize_distribution(throughputs) if throughputs else None
        )
        aggregates.append(
            BenchmarkAggregate(
                suite=suite,
                benchmark_type=benchmark_type,
                algorithm=algorithm,
                backend=backend,
                status=("executed" if executed else "not_executed"),
                seed_count=len({run.seed for run in group}),
                success_rate=(len(executed) / len(group)) if group else 0.0,
                mean_reward=reward_summary.mean if reward_summary else None,
                mean_reward_ci95_low=(
                    reward_summary.confidence_interval_95_low
                    if reward_summary
                    else None
                ),
                mean_reward_ci95_high=(
                    reward_summary.confidence_interval_95_high
                    if reward_summary
                    else None
                ),
                mean_wall_clock_seconds=wall_summary.mean if wall_summary else None,
                mean_steps_per_second=(
                    throughput_summary.mean if throughput_summary else None
                ),
                variance_between_seeds=(
                    statistics.pvariance(rewards) if len(rewards) > 1 else 0.0
                )
                if rewards
                else None,
                coefficient_of_variation=(
                    reward_summary.coefficient_of_variation if reward_summary else None
                ),
            )
        )
    return aggregates
