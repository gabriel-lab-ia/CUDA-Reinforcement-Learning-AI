# Benchmark Results

No formal benchmark result is committed as a conclusion.

Formal campaign implemented -- awaiting execution on target hardware.

Smoke benchmark infrastructure is implemented and can generate local reports
under:

```text
reports/benchmarks/<suite>/<timestamp>/
```

Committed historical RL reports under `reports/rl/` are training artifacts, not
formal benchmark evidence.

Smoke reports generated during development should be inspected locally. They are
not promoted to formal evidence unless generated from a clean commit with the
benchmark methodology documented.

After a real formal campaign, publication must be generated from artifacts such
as `aggregate.csv`, `correctness.json`, and the files under `tables/`. Do not
manually copy latency, throughput, speedup, or learning values into multiple
documents.
