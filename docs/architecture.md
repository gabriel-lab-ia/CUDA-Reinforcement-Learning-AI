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
    experiments/
    configs/

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
- timing and throughput metrics
- GPU utilization
- VRAM consumption
- temperature and power telemetry
- result serialization

Primary locations:

    src/cuda_rl/metrics/
    src/cuda_rl/monitoring/
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

## Native integration strategy

The native CUDA layer will initially use standalone executables for isolated validation and benchmarking.

Later integration options may include PyTorch C++/CUDA extensions, pybind11 bindings, shared native libraries, or custom PyTorch operators.

A binding mechanism should only be introduced after the standalone kernel is correct, tested, and benchmarked.

## Reproducibility

Every formal experiment should record the algorithm, environment, seed, device, precision, hyperparameters, software versions, GPU metadata, execution time, learning metrics, and systems metrics.

## Validation strategy

1. Static validation through Ruff and Mypy.
2. Python behavior validation through Pytest.
3. Native build and runtime validation through CMake and CUDA.
4. Numerical and performance validation across CPU and GPU implementations.

Performance improvements are accepted only after numerical correctness has been established.
