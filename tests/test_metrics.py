from __future__ import annotations

from cuda_rl.metrics.aggregates import summarize_scalars


def test_summarize_scalars_computes_core_statistics() -> None:
    summary = summarize_scalars([1.0, 2.0, 3.0, 4.0])

    assert summary.count == 4
    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
    assert summary.p90 == 3.7
