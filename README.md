# CUDA-Reinforcement-Learning-AI

CUDA-accelerated reinforcement learning laboratory combining native CUDA/C++ kernels, PyTorch agents, reproducible benchmarks, GPU telemetry, and documented experiments.

## Overview

This repository explores reinforcement learning from two complementary engineering layers:

* **Python and PyTorch** for agent implementation, training orchestration, evaluation, experiment tracking, and visualization.
* **Native CUDA/C++** for GPU inspection, custom kernels, low-level memory control, parallel computation, and performance analysis.

The project is being developed incrementally. The current version establishes a reproducible CUDA environment and a native CUDA foundation. Reinforcement-learning agents and GPU benchmarks are part of the next development milestones.

## Current capabilities

* Python 3.12 environment managed with `uv`
* Reproducible dependency lockfile
* PyTorch 2.12 with CUDA 13.0 support on Linux
* Native CUDA/C++ compilation with CMake and Ninja
* NVIDIA GPU and compute-capability inspection
* CPU/CUDA device resolution in Python
* Ruff, Mypy, Pytest, and coverage tooling
* Automated tests for device selection and GPU metadata
* Initial CUDA architecture targeting NVIDIA Ada Lovelace `sm_89`

## Target algorithms

The planned learning sequence is:

1. REINFORCE
2. Deep Q-Network
3. Advantage Actor-Critic
4. Proximal Policy Optimization
5. Soft Actor-Critic
6. Twin Delayed Deep Deterministic Policy Gradient

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
│   └── src/                 # Native CUDA/C++ sources
├── docs/                    # Architecture and roadmap
├── scripts/                 # Environment checks and experiment entry points
├── src/cuda_rl/
│   ├── agents/              # Reinforcement-learning agents
│   ├── environments/        # Environment construction and wrappers
│   ├── evaluation/          # Policy evaluation
│   ├── metrics/             # Training and performance metrics
│   ├── models/              # Neural-network models
│   ├── monitoring/          # GPU and system telemetry
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
* [Roadmap](docs/roadmap.md)
* [Contributing](CONTRIBUTING.md)

## Status

The project is currently in its infrastructure and native-CUDA foundation phase.

Implemented:

* reproducible Python environment
* CUDA-enabled PyTorch installation
* native CUDA compilation
* device inspection
* initial automated tests

Next milestone:

* reproducible CPU, PyTorch CUDA, and native CUDA benchmarks

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
