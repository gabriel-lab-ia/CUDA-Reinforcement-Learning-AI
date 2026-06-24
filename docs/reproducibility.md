# Reproducibility

## Environment

Use Python 3.12 and `uv`.

```bash
uv sync --locked --dev
```

## Seeds

Every benchmark config must define explicit seeds. The initial formal campaign
uses seeds `0, 1, 2, 3, 4` and records every repetition. Raw samples are the
source of truth for all aggregate statistics.

## Determinism limits

CUDA, CuDNN, and some PyTorch kernels can remain nondeterministic depending on
hardware, driver, and selected operations. The project records commit SHA,
working tree status, PyTorch version, CUDA build, device, and configuration so
that differences can be investigated.

## Commands

```bash
uv run python scripts/run_benchmarks.py --config configs/benchmarks/gae_microbenchmark.yaml
uv run cuda-rl benchmark --suite gae-smoke --backends torch_cpu --seeds 0,1,2
uv run python scripts/run_benchmark_campaign.py \
  --configs \
  configs/benchmarks/formal_gae.yaml \
  configs/benchmarks/formal_ppo_loss.yaml \
  configs/benchmarks/formal_replay_buffer.yaml \
  configs/benchmarks/formal_dqn_cartpole.yaml \
  configs/benchmarks/formal_a2c_cartpole.yaml
```

Run smoke configs and the non-slow, non-CUDA test suite before launching the
full campaign. Promote a smoke run to formal evidence only by rerunning the
formal config from a known commit and publishing tables extracted from the
resulting artifacts.
