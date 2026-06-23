# Algorithm Maturity

| Algorithm | Math | Unit tests | End-to-end training | Multi-seed | Baseline comparison | CUDA kernel |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE | Implemented | Partial | Implemented in monolith | Pending | Pending | None |
| DQN | Implemented | Partial | Implemented in monolith | Pending | Scaffolded | None |
| Double DQN | Implemented | Partial | Implemented in monolith | Pending | Scaffolded | None |
| A2C | Implemented | Partial | Implemented in monolith | Pending | Pending | None |
| PPO | Implemented loss/GAE | Yes | Pending | Pending | Scaffolded | None |

This table separates formulas, tested components, trainable agents, multi-seed
benchmarks, baseline comparisons, and CUDA kernels. A checked math component is
not the same thing as a validated training algorithm.
