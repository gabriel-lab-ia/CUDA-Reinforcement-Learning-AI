# Experiment Lifecycle

1. Create or select a versioned config.
2. Validate config schema.
3. Capture metadata, commit SHA, and working tree status.
4. Execute runs for all backends and seeds.
5. Persist `runs.csv`.
6. Aggregate statistics into `aggregate.csv`.
7. Write `metadata.json` and `summary.json`.
8. Generate `comparison.md`.
9. Generate figures.
10. Interpret results with limitations.

Formal reports must not be called conclusive with fewer than ten seeds.
