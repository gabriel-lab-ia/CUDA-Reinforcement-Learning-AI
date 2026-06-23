# Engineering Playbook

## Mission

CUDA Reinforcement Learning AI is an engineering laboratory for reinforcement
learning systems that must be reproducible, measurable, GPU-aware, and easy to
extend. The project is intentionally not a notebook-only exploration. It is a
software system with entry points, automated checks, architectural seams,
experiment records, and explicit operating procedures.

## Engineering standards

1. Every feature must be runnable from the command line.
2. Every persistent output must be discoverable from a deterministic directory.
3. Every formal experiment must produce machine-readable metrics.
4. Every CUDA optimization must have a CPU or PyTorch reference path.
5. Every benchmark must separate correctness validation from performance claims.
6. Every algorithm implementation must expose enough metadata to reproduce a run.
7. Every change to public behavior should have a test or a documented reason.

## Repository responsibilities

The repository is divided into four primary concerns:

- `cuda/`: native CUDA/C++ and the transitional training implementation.
- `src/cuda_rl/`: importable Python package and reusable system components.
- `configs/`: versioned experiment profiles.
- `docs/`: architecture, operations, diagnostics, ADRs, and roadmap.

The current `cuda/reinforcement_learning.py` module is a transitional home for
the training loop. It contains valuable algorithmic implementation work, but it
is intentionally documented as a migration target. The package facade in
`src/cuda_rl/reinforcement_learning.py` keeps imports stable while the code is
split into modules.

## Definition of done

A change is done when the following are true:

- Ruff formatting passes.
- Ruff lint passes.
- Mypy passes in strict mode for `src` and `scripts`.
- Pytest passes on CPU.
- CUDA-specific tests are either passing on GPU hardware or clearly skipped.
- Documentation describes new operational behavior.
- The roadmap reflects materially completed work.

## Local development workflow

Create and sync the environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked
```

Run the quality gate:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -v
```

Run a minimal training smoke test:

```bash
uv run python scripts/train_rl.py \
  --algorithm dqn \
  --episodes 1 \
  --evaluation-episodes 1 \
  --learning-starts 100 \
  --batch-size 4 \
  --checkpoint-every 999 \
  --output-directory /tmp/cuda_rl_smoke \
  --device cpu
```

## Experiment lifecycle

An experiment should move through these states:

1. Planned: profile exists in `configs/`.
2. Running: registry contains a running document.
3. Completed: summary, episode CSV, evaluation CSV, and registry metadata exist.
4. Reviewed: report is inspected and conclusions are recorded.
5. Promoted: configuration becomes a named baseline or benchmark.

The `ExperimentRegistry` persists this lifecycle in JSONL documents. JSONL is
used because it is append-only, diff-friendly, local-first, and compatible with
later migration to MongoDB, DynamoDB, OpenSearch, or object storage.

## Metrics model

Training metrics should answer learning questions:

- Is reward improving?
- Is the agent solving the environment?
- Is the policy stable during evaluation?
- Are losses exploding, collapsing, or oscillating?
- Are episode lengths consistent with reward behavior?

Systems metrics should answer engineering questions:

- What device was used?
- How much memory did the process consume?
- Was CUDA available?
- How long did each operation take?
- Are benchmarks stable across repeated measurements?

## Benchmark discipline

Benchmarks must include:

- warmup iterations
- measured iterations
- mean, median, standard deviation, min, max, p90, and p95
- telemetry before and after execution
- explicit units
- enough metadata to reproduce the run

Never compare CUDA and CPU performance without confirming numerical agreement
first. A fast wrong kernel is not an optimization.

## CUDA engineering principles

Native CUDA work should follow this sequence:

1. Write a clear CPU reference implementation.
2. Write a PyTorch tensor implementation when applicable.
3. Write a native CUDA kernel.
4. Validate numerical agreement across representative inputs.
5. Add CUDA event timing.
6. Benchmark with warmup and repeated measurements.
7. Document memory access patterns and launch parameters.
8. Only then optimize.

## Reinforcement learning engineering principles

RL code is notoriously easy to make impressive and wrong. This project treats
the following as non-negotiable:

- seeds must be explicit
- environment resets must be seeded
- action and observation spaces must be validated
- evaluation must be deterministic when possible
- checkpoints must include config and optimizer state
- reward windows must be reported, not only final reward
- one lucky run is never a scientific result

## Review checklist

Before merging a major RL change, ask:

- Does this change alter training semantics?
- Does it preserve deterministic evaluation?
- Does it preserve CPU execution?
- Does it fail clearly when CUDA is requested but unavailable?
- Does it record enough metadata for reproduction?
- Does it keep generated artifacts out of git?
- Does it have a smoke test that completes quickly?

## Release checklist

Before tagging a release:

- run quality checks locally
- verify CI on GitHub
- run one CPU smoke experiment
- run one CUDA smoke experiment on GPU hardware
- refresh README status
- refresh roadmap completion states
- publish benchmark summary if performance changed

## Known technical debt

- The training loop is still monolithic.
- YAML configs are present but the typed loader currently supports JSON and TOML.
- PPO, SAC, TD3, prioritized replay, and native CUDA RL kernels are roadmap items.
- CI cannot validate real CUDA hardware on the default GitHub-hosted runner.
- Generated report artifacts need stronger retention and indexing policy.

## Next refactoring sequence

The preferred decomposition of `cuda/reinforcement_learning.py` is:

1. Move config and argument parsing into `src/cuda_rl/config`.
2. Move network definitions into `src/cuda_rl/models`.
3. Move replay buffers and agents into `src/cuda_rl/agents`.
4. Move evaluation into `src/cuda_rl/evaluation`.
5. Move metrics writing and checkpoints into `src/cuda_rl/metrics`.
6. Move the training orchestration into `src/cuda_rl/training`.
7. Leave `cuda/reinforcement_learning.py` as a thin compatibility entry point.
