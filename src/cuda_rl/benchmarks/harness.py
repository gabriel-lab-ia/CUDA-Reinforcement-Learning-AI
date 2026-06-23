from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from cuda_rl.metrics.aggregates import ScalarSummary, summarize_scalars
from cuda_rl.monitoring.telemetry import TelemetrySnapshot, capture_telemetry


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    operation: Callable[[], None]
    warmup_iterations: int = 3
    measured_iterations: int = 10

    def validate(self) -> None:
        if not self.name:
            raise ValueError("benchmark case name is required.")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative.")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be positive.")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    durations_ms: tuple[float, ...]
    summary: ScalarSummary
    telemetry_before: TelemetrySnapshot
    telemetry_after: TelemetrySnapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "durations_ms": list(self.durations_ms),
            "summary": {
                "count": self.summary.count,
                "mean": self.summary.mean,
                "median": self.summary.median,
                "standard_deviation": self.summary.standard_deviation,
                "minimum": self.summary.minimum,
                "maximum": self.summary.maximum,
                "p90": self.summary.p90,
                "p95": self.summary.p95,
            },
            "telemetry_before": self.telemetry_before.to_dict(),
            "telemetry_after": self.telemetry_after.to_dict(),
        }


class BenchmarkSuite:
    def __init__(self, cases: list[BenchmarkCase]) -> None:
        if not cases:
            raise ValueError("at least one benchmark case is required.")
        self.cases = cases

    def run(self) -> list[BenchmarkResult]:
        return [self._run_case(case) for case in self.cases]

    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        case.validate()
        for _ in range(case.warmup_iterations):
            case.operation()

        telemetry_before = capture_telemetry()
        durations_ms: list[float] = []
        for _ in range(case.measured_iterations):
            started_at = time.perf_counter()
            case.operation()
            durations_ms.append((time.perf_counter() - started_at) * 1_000.0)
        telemetry_after = capture_telemetry()

        return BenchmarkResult(
            name=case.name,
            durations_ms=tuple(durations_ms),
            summary=summarize_scalars(durations_ms),
            telemetry_before=telemetry_before,
            telemetry_after=telemetry_after,
        )
