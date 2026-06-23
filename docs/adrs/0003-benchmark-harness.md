# ADR 0003: Benchmark Harness

## Status

Accepted

## Context

The project will compare CPU, PyTorch CUDA, and native CUDA operations. Raw
timing snippets are not enough because they do not capture warmup behavior,
variance, percentiles, or machine state.

## Decision

Introduce a small benchmark harness in `src/cuda_rl/benchmarks`.

The harness records:

- warmup iterations
- measured durations
- scalar summaries
- telemetry before execution
- telemetry after execution

## Consequences

Positive:

- consistent benchmark structure
- testable without CUDA
- reusable for future native kernels
- produces serializable result dictionaries

Negative:

- not a substitute for specialized profilers
- wall-clock measurements can be noisy
- CUDA kernels still need synchronization-aware benchmarks

## Future work

Add CUDA event timing for GPU-specific measurements and optional export to the
JSONL document store.
