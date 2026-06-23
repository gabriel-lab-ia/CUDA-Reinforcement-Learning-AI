# Reproducibility

## Environment

Use Python 3.12 and `uv`.

```bash
uv sync --locked --dev
```

## Seeds

Every benchmark config must define explicit seeds. Smoke configs use five seeds.
Formal reports require at least ten seeds.

## Determinism limits

CUDA, CuDNN, and some PyTorch kernels can remain nondeterministic depending on
hardware, driver, and selected operations. The project records commit SHA,
working tree status, PyTorch version, CUDA build, device, and configuration so
that differences can be investigated.

## Commands

```bash
uv run python scripts/run_benchmarks.py --config configs/benchmarks/gae_microbenchmark.yaml
uv run cuda-rl benchmark --suite gae-smoke --backends torch_cpu --seeds 0,1,2
```
