from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StatisticalSummary:
    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float
    confidence_interval_95_low: float
    confidence_interval_95_high: float
    coefficient_of_variation: float


def summarize_distribution(
    values: list[float],
    *,
    bootstrap_resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> StatisticalSummary:
    if not values:
        raise ValueError("at least one value is required.")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    ordered = sorted(float(value) for value in values)
    mean = statistics.fmean(ordered)
    standard_deviation = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    low, high = bootstrap_mean_confidence_interval(
        ordered,
        resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )
    return StatisticalSummary(
        count=len(ordered),
        mean=mean,
        median=statistics.median(ordered),
        standard_deviation=standard_deviation,
        minimum=ordered[0],
        maximum=ordered[-1],
        p5=percentile(ordered, 5.0),
        p25=percentile(ordered, 25.0),
        p75=percentile(ordered, 75.0),
        p95=percentile(ordered, 95.0),
        p99=percentile(ordered, 99.0),
        confidence_interval_95_low=low,
        confidence_interval_95_high=high,
        coefficient_of_variation=(
            0.0 if mean == 0.0 else abs(standard_deviation / mean)
        ),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("at least one value is required.")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be in [0, 100].")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_confidence_interval(
    values: list[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]

    random_generator = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample = [random_generator.choice(values) for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = 1.0 - confidence
    low_index = int((alpha / 2.0) * (len(means) - 1))
    high_index = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return means[low_index], means[high_index]
