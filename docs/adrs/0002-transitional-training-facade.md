# ADR 0002: Transitional Training Facade

## Status

Accepted

## Context

The repository contains a large training implementation in
`cuda/reinforcement_learning.py`. The package imports historically expected
`cuda_rl.reinforcement_learning`. Removing or moving the monolith abruptly would
create unnecessary breakage while the project is still being modularized.

## Decision

Keep `cuda/reinforcement_learning.py` as the implementation source for now and
provide a compatibility facade at `src/cuda_rl/reinforcement_learning.py`.

## Consequences

Positive:

- package imports work
- command-line training remains stable
- refactoring can happen incrementally
- existing docs and scripts do not need immediate churn

Negative:

- implementation ownership is temporarily split
- package modules re-export some symbols instead of owning them
- mypy coverage is stronger for `src` than for the transitional module

## Exit criteria

The facade can become the implementation when:

- agents live under `src/cuda_rl/agents`
- models live under `src/cuda_rl/models`
- evaluation lives under `src/cuda_rl/evaluation`
- metrics and checkpoints live under `src/cuda_rl/metrics`
- training orchestration lives under `src/cuda_rl/training`
- `cuda/` contains only native CUDA/C++ and optional compatibility wrappers
