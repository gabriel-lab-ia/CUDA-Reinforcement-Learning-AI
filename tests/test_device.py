from __future__ import annotations

import pytest
import torch

from cuda_rl.utils.device import describe_device, get_device


def test_cpu_device_can_be_selected() -> None:
    assert get_device("cpu").type == "cpu"


def test_auto_device_selects_available_backend() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert get_device("auto").type == expected


def test_invalid_device_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        get_device("quantum")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_can_be_selected() -> None:
    assert get_device("cuda").type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_description_contains_gpu_metadata() -> None:
    description = describe_device(get_device("cuda"))

    assert description["device_type"] == "cuda"
    assert "device_name" in description
    assert "compute_capability" in description
    assert float(description["vram_gib"]) > 0
