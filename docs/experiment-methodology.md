# Experiment Methodology

## Purpose

This document defines how experiments are designed, executed, stored, and
reviewed in CUDA Reinforcement Learning AI. The goal is to avoid the most common
failure mode in reinforcement learning projects: impressive-looking results that
cannot be reproduced or explained.

## Experiment profile

Every formal experiment should have a profile with:

- name
- algorithm
- environment id
- seed
- device
- schedule
- resource budget
- hyperparameters
- tags

Profiles are versioned under `configs/experiments/`. JSON and TOML profiles can
be loaded through `cuda_rl.config.load_experiment_profile`. The existing YAML
file remains useful as a human-readable baseline, but the typed loader currently
focuses on formats supported with the standard library.

## Naming

Use descriptive names:

- `cartpole-dqn-baseline`
- `cartpole-a2c-baseline`
- `cartpole-reinforce-entropy-ablation`
- `lunarlander-dqn-prioritized-replay`

Avoid vague names:

- `test`
- `new-run`
- `final`
- `try2`

## Seeds

Use at least three seeds before making claims:

- baseline seed: 42
- alternate seed: 43
- stress seed: 101

A single seed is acceptable for smoke testing only.

## Metrics

Required learning metrics:

- episode reward
- episode length
- moving average reward
- loss when available
- epsilon when available
- evaluation mean reward
- evaluation standard deviation
- global steps

Required systems metrics:

- device
- CUDA availability
- PyTorch version
- CUDA build version
- GPU name when available
- process memory
- wall-clock duration

## Output layout

A training run should write:

```text
reports/rl/<run-name>/
├── episodes.csv
├── evaluations.csv
├── summary.json
├── nosql/
│   └── experiment_summaries.jsonl
└── checkpoints/
```

Generated checkpoints are intentionally ignored by git. Summaries and metrics
can be committed only when they represent a documented benchmark or release
artifact.

## Interpreting reward

Reward should be interpreted with caution:

- final reward can be noisy
- maximum reward can overstate stability
- moving average gives a better learning signal
- evaluation reward is more meaningful than training reward
- standard deviation reveals policy instability

## Comparing algorithms

When comparing REINFORCE, DQN, and A2C:

- use the same environment
- use the same seed set
- use comparable network sizes
- use identical evaluation schedules
- report wall-clock time and global steps
- include failed runs

## Benchmarking CUDA work

CUDA operations should not be benchmarked inside an unstable training loop until
the kernel is validated in isolation. Use standalone benchmarks first, then
integrate the operation into the RL path.

## Regression policy

A change is a regression if:

- a smoke training run no longer completes
- CPU execution breaks
- CUDA request no longer fails clearly on machines without CUDA
- metrics files are not written
- summary metadata is incomplete
- the package cannot be imported

## Review template

Use this template when reviewing experiment results:

```text
Experiment:
Algorithm:
Environment:
Seeds:
Device:
Episodes:
Best evaluation reward:
Final moving average reward:
Observed instability:
Systems notes:
Conclusion:
Next action:
```
