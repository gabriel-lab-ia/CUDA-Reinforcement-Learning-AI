from __future__ import annotations

from cuda_rl.benchmarks import BenchmarkCase, BenchmarkSuite


def test_benchmark_suite_runs_cases() -> None:
    counter = {"value": 0}

    def operation() -> None:
        counter["value"] += 1

    suite = BenchmarkSuite(
        [
            BenchmarkCase(
                name="increment",
                operation=operation,
                warmup_iterations=1,
                measured_iterations=2,
            )
        ]
    )

    results = suite.run()

    assert len(results) == 1
    assert results[0].summary.count == 2
    assert counter["value"] == 3
