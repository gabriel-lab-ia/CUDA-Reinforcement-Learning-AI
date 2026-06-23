from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import psutil  # type: ignore[import-untyped]
import torch


@dataclass(frozen=True, slots=True)
class CpuTelemetry:
    logical_cores: int
    physical_cores: int | None
    process_memory_rss_mib: float
    system_memory_used_percent: float
    load_average_1m: float | None


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    available: bool
    device_name: str | None
    cuda_version: str | None
    allocated_mib: float | None
    reserved_mib: float | None
    max_allocated_mib: float | None


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    captured_at: float
    hostname: str
    python_version: str
    platform: str
    cpu: CpuTelemetry
    gpu: GpuTelemetry

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return {
            "captured_at": self.captured_at,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_logical_cores": self.cpu.logical_cores,
            "cpu_physical_cores": self.cpu.physical_cores,
            "process_memory_rss_mib": self.cpu.process_memory_rss_mib,
            "system_memory_used_percent": self.cpu.system_memory_used_percent,
            "load_average_1m": self.cpu.load_average_1m,
            "gpu_available": self.gpu.available,
            "gpu_device_name": self.gpu.device_name,
            "cuda_version": self.gpu.cuda_version,
            "gpu_allocated_mib": self.gpu.allocated_mib,
            "gpu_reserved_mib": self.gpu.reserved_mib,
            "gpu_max_allocated_mib": self.gpu.max_allocated_mib,
        }


def capture_telemetry() -> TelemetrySnapshot:
    process = psutil.Process()
    return TelemetrySnapshot(
        captured_at=time.time(),
        hostname=platform.node(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu=CpuTelemetry(
            logical_cores=psutil.cpu_count(logical=True) or 0,
            physical_cores=psutil.cpu_count(logical=False),
            process_memory_rss_mib=_bytes_to_mib(process.memory_info().rss),
            system_memory_used_percent=float(psutil.virtual_memory().percent),
            load_average_1m=_load_average_1m(),
        ),
        gpu=_capture_gpu_telemetry(),
    )


def _capture_gpu_telemetry() -> GpuTelemetry:
    if not torch.cuda.is_available():
        return GpuTelemetry(
            available=False,
            device_name=None,
            cuda_version=torch.version.cuda,
            allocated_mib=None,
            reserved_mib=None,
            max_allocated_mib=None,
        )
    return GpuTelemetry(
        available=True,
        device_name=torch.cuda.get_device_name(0),
        cuda_version=torch.version.cuda,
        allocated_mib=_bytes_to_mib(torch.cuda.memory_allocated(0)),
        reserved_mib=_bytes_to_mib(torch.cuda.memory_reserved(0)),
        max_allocated_mib=_bytes_to_mib(torch.cuda.max_memory_allocated(0)),
    )


def _bytes_to_mib(value: int) -> float:
    return round(value / 1024**2, 3)


def _load_average_1m() -> float | None:
    if not hasattr(os, "getloadavg"):
        return None
    return float(os.getloadavg()[0])
