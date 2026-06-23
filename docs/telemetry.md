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

NVML-specific fields such as utilization, temperature, and power are part of
the benchmark schema but currently recorded as unavailable unless implemented
by a future NVML sampler.
