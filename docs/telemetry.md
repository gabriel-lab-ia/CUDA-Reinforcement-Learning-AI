# Telemetry

Telemetry is optional and must not break CPU-only environments.

Current telemetry records:

- timestamp;
- host and platform;
- Python version;
- CPU core counts;
- process RSS;
- system memory percentage;
- CUDA availability;
- CUDA build version;
- GPU name when available;
- PyTorch CUDA memory counters.

Formal benchmark metadata also records commit SHA, branch/working-tree state
through the campaign manifest, PyTorch and CUDA runtime versions, operating
system, CPU, GPU name, and process memory. NVML-specific fields such as GPU
utilization, temperature, memory, and power are optional. Missing NVML support
must produce a clear warning or null fields and must not break CPU benchmarks.
