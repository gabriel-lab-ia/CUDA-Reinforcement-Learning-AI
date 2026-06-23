from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    p90: float
    p95: float


def summarize_scalars(values: list[float]) -> ScalarSummary:
    if not values:
        raise ValueError("at least one value is required.")

    ordered = sorted(values)
    return ScalarSummary(
        count=len(ordered),
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        standard_deviation=(statistics.pstdev(ordered) if len(ordered) > 1 else 0.0),
        minimum=ordered[0],
        maximum=ordered[-1],
        p90=_percentile(ordered, 90.0),
        p95=_percentile(ordered, 95.0),
    )


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (percentile / 100.0) * (len(ordered_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    weight = rank - lower_index
    return (
        ordered_values[lower_index] * (1.0 - weight)
        + ordered_values[upper_index] * weight
    )
