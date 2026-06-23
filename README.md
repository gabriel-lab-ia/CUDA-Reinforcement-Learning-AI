# CUDA-Reinforcement-Learning-AI
<img width="1774" height="887" alt="ChatGPT Image 23 de jun  de 2026, 16_44_37" src="https://github.com/user-attachments/assets/de5aa2ff-e19c-4214-9fdf-ef5427401d49" />

CUDA-accelerated reinforcement learning laboratory combining native CUDA/C++ kernels, PyTorch agents, reproducible benchmarks, GPU telemetry, and documented experiments.

## Overview

This repository explores reinforcement learning from two complementary engineering layers:

* **Python and PyTorch** for agent implementation, training orchestration, evaluation, experiment tracking, and visualization.
* **Native CUDA/C++** for GPU inspection, custom kernels, low-level memory control, parallel computation, and performance analysis.

The project is being developed as a research-grade engineering lab. The current version combines a reproducible CUDA environment, native CUDA device validation, first-principles deep RL agents, experiment metrics, local NoSQL persistence utilities, and CI/CD quality gates.

## Current capabilities

* Python 3.12 environment managed with `uv`
* Reproducible dependency lockfile
* PyTorch 2.12 with CUDA 13.0 support on Linux
* Native CUDA/C++ compilation with CMake and Ninja
* NVIDIA GPU and compute-capability inspection
* CPU/CUDA device resolution in Python
* REINFORCE, DQN, Double DQN, and A2C training flows for discrete-control Gymnasium environments
* PPO loss and Generalized Advantage Estimation modules for the next training backend
* Uniform replay buffer and Prioritized Experience Replay implementation
* Checkpointing, deterministic evaluation, CSV/JSON experiment reports, benchmark summaries, and scalar metric aggregation
* Local append-only JSONL document store for NoSQL experiment metadata
* Typed experiment profiles, run lifecycle registry, telemetry snapshots, and report loading utilities
* Ruff, Mypy, Pytest, and coverage tooling
* GitHub Actions quality pipeline
* Automated tests for device selection and GPU metadata
* Initial CUDA architecture targeting NVIDIA Ada Lovelace `sm_89`

## Algorithm roadmap

Implemented or scaffolded:

1. REINFORCE
2. Deep Q-Network
3. Double DQN
4. Advantage Actor-Critic

Next advanced algorithms:

1. Proximal Policy Optimization with generalized advantage estimation
2. Prioritized experience replay
3. Soft Actor-Critic
4. Twin Delayed Deep Deterministic Policy Gradient

Implementations will be developed from first principles and compared with established reference implementations when appropriate.

## Planned CUDA applications

Native CUDA work will progress from general computational validation toward reinforcement-learning-specific workloads:

* vector operations and reductions
* matrix operations
* discounted-return calculation
* advantage normalization
* generalized advantage estimation
* replay-buffer sampling
* batch preprocessing
* prioritized replay operations
* metrics reduction and GPU telemetry

## Repository structure

```text
.
├── cuda/
│   ├── include/             # Native CUDA/C++ headers
│   ├── reinforcement_learning.py
│   └── src/                 # Native CUDA/C++ sources
├── configs/                 # Reproducible experiment configs
├── docs/                    # Architecture and roadmap
├── .github/workflows/       # CI/CD quality gates
├── scripts/                 # Environment checks and experiment entry points
├── src/cuda_rl/
│   ├── agents/              # Reinforcement-learning agents
│   ├── algorithms/          # PPO, GAE, and algorithm-specific math
│   ├── benchmarks/          # Benchmark harness and timing summaries
│   ├── config/              # Typed experiment profiles
│   ├── environments/        # Environment construction and wrappers
│   ├── evaluation/          # Policy evaluation
│   ├── experiments/         # Run registry and lifecycle metadata
│   ├── metrics/             # Training and performance metrics
│   ├── models/              # Neural-network models
│   ├── monitoring/          # GPU and system telemetry
│   ├── replay/              # Uniform and prioritized replay buffers
│   ├── reports/             # Training-report loaders and renderers
│   ├── storage/             # Local NoSQL document persistence
│   └── utils/               # Shared utilities
├── tests/                   # Automated Python tests
├── CMakeLists.txt
├── pyproject.toml
└── uv.lock
```

## Requirements

### Python layer

* Python 3.12
* `uv`
* NVIDIA driver compatible with the selected PyTorch CUDA build

### Native CUDA layer

* CUDA Toolkit 13.0
* CMake 3.24 or newer
* Ninja
* C++20-compatible compiler
* NVIDIA GPU

The current default native build targets compute capability `8.9`.

## Python environment

Clone the repository and synchronize the locked environment:

```bash
git clone https://github.com/gabriel-lab-ia/CUDA-Reinforcement-Learning-AI.git
cd CUDA-Reinforcement-Learning-AI

uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Validate the PyTorch CUDA environment:

```bash
uv run python scripts/check_cuda.py
```

## Train an RL agent

Run the script wrapper:

```bash
uv run python scripts/train_rl.py \
  --algorithm dqn \
  --env CartPole-v1 \
  --episodes 500 \
  --device auto
```

Use the higher-level CLI for profile inspection, registry work, reports, and
future training orchestration:

```bash
uv run python scripts/cuda_rl_cli.py profile configs/experiments/cartpole_a2c.json
uv run python scripts/cuda_rl_cli.py report reports/rl/<run-name>
uv run python scripts/cuda_rl_cli.py train -- --algorithm dqn --episodes 10
```

When the project is installed as a package, the console entry point is also
available:

```bash
uv run cuda-rl profile configs/experiments/cartpole_a2c.json
uv run cuda-rl-train --algorithm a2c --episodes 300
```

Experiment outputs are written under `reports/rl/<run-name>/` with `episodes.csv`, `evaluations.csv`, `summary.json`, and checkpoints.

## Native CUDA build

Configure and compile the native CUDA executable:

```bash
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build
./build/cuda_device_check
```

## Quality checks

Run formatting, linting, static typing, and tests:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -v
```

## Experimental principles

Experiments in this repository should be:

* reproducible through fixed configurations and random seeds
* measured across multiple executions
* explicit about CPU and GPU execution
* accompanied by hardware and software metadata
* evaluated using learning and systems metrics
* documented with limitations and interpretation

A single successful CUDA execution is treated as an environment validation, not as a scientific benchmark.

## Documentation

* [Architecture](docs/architecture.md)
* [Engineering playbook](docs/engineering-playbook.md)
* [Experiment methodology](docs/experiment-methodology.md)
* [Technical diagnostic](docs/diagnostic.md)
* [Roadmap](docs/roadmap.md)
* [Contributing](CONTRIBUTING.md)

## Status

The project is currently moving from a strong prototype into a modular research platform.

Implemented:

* reproducible Python environment
* CUDA-enabled PyTorch installation
* native CUDA compilation
* device inspection
* REINFORCE, DQN, Double DQN, and A2C prototype training
* PPO/GAE algorithm math and prioritized replay modules
* local NoSQL document store
* typed experiment profiles, registry, telemetry, benchmark harness, and report loading
* automated tests and CI/CD

Next milestone:

* split the monolithic training implementation into dedicated package modules and add PPO plus native CUDA RL kernels

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
