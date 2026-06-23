# CUDA Kernel Validation

## Target kernel

The first RL-specific CUDA target is Generalized Advantage Estimation.

## Reference formula

```text
delta_t = r_t + gamma * (1 - done_t) * value_{t+1} - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}
return_t = advantage_t + value_t
```

## Current status

- PyTorch reference: implemented in `src/cuda_rl/algorithms/gae.py`.
- Native CUDA kernel scaffold: `cuda/src/gae.cu` and `cuda/include/gae.cuh`.
- Python bindings: not introduced.
- Performance claim: none.

## Validation requirements

Before any CUDA speed claim:

- compare against PyTorch reference with `torch.testing.assert_close`;
- test float32 and float64 tolerances;
- test multiple rollout lengths;
- test terminated and truncated boundaries;
- reject NaN and Inf outputs;
- use CUDA events for timing;
- report hardware metadata.

## Current limitations

The first native CUDA implementation targets one trajectory and uses a single
CUDA thread for the reverse recurrence. This is correct-first scaffolding, not
an optimized segmented-scan implementation. It must not be used for speedup
claims until bindings, correctness tests, CUDA event timing, and multi-size
benchmarks are added.
