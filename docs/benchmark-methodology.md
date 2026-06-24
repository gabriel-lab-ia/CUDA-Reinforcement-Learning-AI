# Benchmark Methodology

## Definition

A benchmark in this repository is a reproducible command-line run that records
configuration, hardware and software metadata, per-repetition raw samples,
aggregate statistics, correctness checks, failures, Markdown tables, and plots.
A smoke benchmark validates that the pipeline works. A formal benchmark is a
campaign configured in `configs/benchmarks/formal_*.yaml`; it is evidence only
after execution on target hardware from a known commit.

## Warm-up and synchronization

Microbenchmarks execute warm-up repetitions and measured repetitions. Warm-up
samples are recorded but excluded from aggregate statistics. CPU timings use
`time.perf_counter_ns()`. PyTorch CUDA timings use
`torch.cuda.Event(enable_timing=True)` with `torch.cuda.synchronize()` before
and after measured regions. CUDA is never silently replaced by CPU; unavailable
requested CUDA backends are recorded as failures or `not_available` according to
the config.

Kernel-only measurements prepare logical inputs outside the timed repetition.
End-to-end measurements may include data movement, graph creation, optimizer
work, or replay-buffer setup when the config explicitly requests it. Small
workloads can be dominated by launch, Python, synchronization, or environment
overhead, so interpretation must separate system overhead from arithmetic
throughput.

## Seeds

Smoke benchmark configs use short repetitions. The initial formal campaign uses
the explicitly configured seeds `0, 1, 2, 3, 4` for GAE, PPO loss, replay buffer,
DQN CartPole, and A2C CartPole. Conclusions must report variance between seeds.

## Confidence intervals

Aggregate statistics include count, mean, median, standard deviation, minimum,
maximum, p5, p25, p75, p95, p99, coefficient of variation, throughput, speedup
against the configured reference backend, and a bootstrap 95% confidence
interval for the mean.

## Interpretation

Do not treat smoke results as performance claims. Speedup claims require raw
samples, aggregate rows, and correctness records from the same campaign. GAE
compares NumPy CPU, PyTorch CPU, PyTorch CUDA, and native CUDA only when native
CUDA has a Python binding. PPO loss compares PyTorch CPU and CUDA. Replay
measures uniform and prioritized buffers separately. CartPole learning curves
must not be interpreted as pure GPU acceleration: environment overhead,
workload size, and algorithmic variance can make CUDA slower while still being
a valid result.
