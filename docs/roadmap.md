# Roadmap

## Phase 1 — Repository foundation

Status: complete

- [x] Initialize Python 3.12 project with `uv`
- [x] Lock project dependencies
- [x] Configure PyTorch CUDA 13.0 packages
- [x] Validate RTX 4050 availability
- [x] Configure Ruff, Mypy, and Pytest
- [x] Add device-selection tests
- [x] Configure native CUDA/C++ build
- [x] Inspect the CUDA device through the Runtime API
- [x] Add project documentation
- [x] Add continuous integration

## Phase 2 — Reproducible compute benchmarks

- [ ] Implement native CUDA vector addition
- [ ] Implement CPU reference vector addition
- [ ] Validate numerical agreement
- [ ] Add CUDA event timing
- [ ] Benchmark NumPy CPU
- [ ] Benchmark PyTorch CPU
- [ ] Benchmark PyTorch CUDA
- [ ] Benchmark native CUDA
- [x] Collect mean, median, standard deviation, and percentiles
- [ ] Record GPU memory and telemetry
- [ ] Export CSV and JSON results
- [ ] Document benchmark methodology and limitations

## Phase 3 — REINFORCE from first principles

- [x] Implement policy network
- [x] Implement stochastic action sampling
- [x] Implement discounted returns
- [x] Implement policy-gradient loss
- [x] Add deterministic evaluation
- [x] Record reward and loss metrics
- [ ] Compare CPU and CUDA training
- [ ] Validate multiple random seeds
- [ ] Document the mathematical derivation
- [ ] Compare against a reference implementation

## Phase 4 — Value-based learning

- [ ] Implement tabular Q-learning baseline
- [x] Implement Deep Q-Network
- [x] Add target network
- [x] Add experience replay
- [x] Add epsilon-greedy scheduling
- [x] Implement Double DQN
- [ ] Implement prioritized experience replay
- [ ] Benchmark replay-buffer operations

## Phase 5 — Actor-critic algorithms

- [x] Implement Advantage Actor-Critic
- [ ] Implement Proximal Policy Optimization
- [ ] Implement Generalized Advantage Estimation
- [ ] Implement Soft Actor-Critic
- [ ] Implement Twin Delayed DDEG
- [ ] Add continuous-control environments
- [ ] Compare learning stability across seeds

## Phase 6 — CUDA operations for reinforcement learning

- [ ] Discounted-return kernel
- [ ] Advantage-normalization kernel
- [ ] GAE kernel
- [ ] Replay-buffer sampling kernel
- [ ] Batch-preprocessing kernels
- [ ] Prioritized-replay operations
- [ ] Metrics-reduction kernels
- [ ] PyTorch C++/CUDA extension integration
- [ ] CPU versus PyTorch CUDA versus native CUDA analysis

## Phase 7 — Experiment and deployment infrastructure

- [x] GitHub Actions quality pipeline
- [x] Structured experiment configuration
- [x] Local NoSQL experiment document store
- [x] Experiment registry and lifecycle metadata
- [x] Telemetry snapshots
- [x] Benchmark harness
- [x] Training report loader
- [ ] TensorBoard integration
- [ ] MLflow integration
- [ ] Containerized development environment
- [ ] Reproducible benchmark reports
- [ ] Model cards
- [ ] Automated documentation publishing
