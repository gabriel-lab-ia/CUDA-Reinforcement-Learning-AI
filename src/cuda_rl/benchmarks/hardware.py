from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict

import torch

from cuda_rl.benchmarks.schemas import BenchmarkConfig, BenchmarkMetadata


def capture_benchmark_metadata(
    config: BenchmarkConfig,
    *,
    timestamp: str,
) -> BenchmarkMetadata:
    return BenchmarkMetadata(
        suite=config.suite,
        timestamp=timestamp,
        commit_sha=_git_output(["git", "rev-parse", "HEAD"]),
        working_tree_status=_git_output(["git", "status", "--short"]) or "clean",
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        config=asdict(config),
    )


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()
