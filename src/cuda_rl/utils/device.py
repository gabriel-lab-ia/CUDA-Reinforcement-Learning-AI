from __future__ import annotations

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the device used for training and evaluation."""

    normalized = preferred.lower()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-enabled device is available."
        )

    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device {preferred!r}. Expected 'auto', 'cpu', or 'cuda'."
        )

    return torch.device(normalized)


def describe_device(device: torch.device) -> dict[str, str | int | float]:
    """Return serializable hardware information."""

    description: dict[str, str | int | float] = {
        "device_type": device.type,
        "pytorch_version": torch.__version__,
    }

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)

        description.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda or "unknown",
                "compute_capability": (f"{properties.major}.{properties.minor}"),
                "vram_gib": round(properties.total_memory / 1024**3, 2),
                "multiprocessors": properties.multi_processor_count,
            }
        )

    return description
