# Architecture

## Purpose

CUDA Reinforcement Learning AI is structured as a hybrid Python and native CUDA laboratory.

Python provides rapid experimentation, neural-network training, environment integration, metrics, and reporting. Native CUDA provides explicit control over parallel kernels, memory movement, synchronization, and low-level performance characteristics.

## Architectural layers

### Experiment layer

Responsibilities:

- training entry points
- benchmark execution
- configuration loading
- random-seed control
- output persistence
- experiment identification

Primary locations:

    scripts/
    configs/
    reports/
    src/cuda_rl/config/
    src/cuda_rl/experiments/

### Reinforcement-learning layer

Responsibilities:

- policy-gradient agents
- value-based agents
- actor-critic agents
- rollout collection
- replay buffers
- policy evaluation

Primary locations:

    src/cuda_rl/agents/
    src/cuda_rl/models/
    src/cuda_rl/environments/
    src/cuda_rl/evaluation/

### Measurement layer

Responsibilities:

- learning metrics
- scalar aggregate statistics
- timing and throughput metrics
- GPU utilization
- VRAM consumption
- temperature and power telemetry
- result serialization
- benchmark execution
- report loading and markdown rendering

Primary locations:

    src/cuda_rl/metrics/
    src/cuda_rl/monitoring/
    src/cuda_rl/benchmarks/
    src/cuda_rl/reports/
    reports/

### Persistence layer

Responsibilities:

- append-only experiment metadata
- local NoSQL document persistence
- compactable JSONL collections
- storage abstractions that can later be backed by MongoDB, DynamoDB, or object storage

Primary locations:

    src/cuda_rl/storage/
    reports/

### Native CUDA layer

Responsibilities:

- custom CUDA kernels
- explicit device-memory management
- CUDA event timing
- synchronization and error handling
- low-level runtime validation
- optimized reinforcement-learning operations

Primary locations:

    cuda/include/
    cuda/src/

## Execution flow

    configuration
        ↓
    environment construction
        ↓
    agent and model initialization
        ↓
    rollout or replay-buffer collection
        ↓
    CPU, PyTorch CUDA, or native CUDA computation
        ↓
    parameter update and evaluation
        ↓
    metrics and telemetry persistence
        ↓
    document-store indexing

## Native integration strategy

The native CUDA layer will initially use standalone executables for isolated validation and benchmarking.

Later integration options may include PyTorch C++/CUDA extensions, pybind11 bindings, shared native libraries, or custom PyTorch operators.

A binding mechanism should only be introduced after the standalone kernel is correct, tested, and benchmarked.

## Reproducibility

Every formal experiment should record the algorithm, environment, seed, device, precision, hyperparameters, software versions, GPU metadata, execution time, learning metrics, systems metrics, and document-store identifiers.

## Transitional module strategy

The historical training implementation currently remains in `cuda/reinforcement_learning.py`.
The package-level module `src/cuda_rl/reinforcement_learning.py` provides a compatibility facade so imports remain stable while the implementation is progressively split into dedicated `agents`, `models`, `evaluation`, `metrics`, and `storage` modules.

## Runtime metadata flow

    typed profile
        ↓
    experiment registry
        ↓
    training entry point
        ↓
    CSV/JSON metrics
        ↓
    JSONL document store
        ↓
    training report loader

## Validation strategy

1. Static validation through Ruff and Mypy.
2. Python behavior validation through Pytest.
3. Native build and runtime validation through CMake and CUDA.
4. Numerical and performance validation across CPU and GPU implementations.
5. GitHub Actions validation on push and pull request.

Performance improvements are accepted only after numerical correctness has been established.
