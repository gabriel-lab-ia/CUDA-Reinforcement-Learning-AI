# Contributing

Thank you for your interest in CUDA Reinforcement Learning AI.

This repository is developed as a reproducible engineering and research laboratory. Contributions should preserve correctness, measurement quality, clear documentation, and separation between Python orchestration and native CUDA computation.

## Development setup

Create and synchronize the Python environment:

    uv venv --python 3.12
    source .venv/bin/activate
    uv sync

Configure and build the native CUDA layer:

    cmake -S . -B build -G Ninja \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Validation

Before submitting changes, run:

    uv run ruff format .
    uv run ruff check .
    uv run mypy
    uv run pytest -v

When native CUDA files are changed, rebuild and execute the relevant native targets.

## Contribution principles

- Keep experiments reproducible.
- Record seeds, configurations, hardware, and software versions.
- Do not present a single execution as a benchmark.
- Add tests for new behavior.
- Document numerical tolerances.
- Keep generated artifacts out of source directories.
- Prefer small and focused commits.
- Support performance claims with collected evidence.
- Separate educational baselines from optimized implementations.

## Commit messages

Use clear conventional-style commit messages when practical:

    feat: add native CUDA vector addition kernel
    fix: correct CUDA memory cleanup on failure
    test: validate CPU and CUDA numerical agreement
    docs: document benchmark methodology
    refactor: separate telemetry from benchmark execution

## Pull requests

A pull request should describe:

- the problem being solved
- the implementation approach
- validation commands
- collected results when performance is involved
- known limitations
