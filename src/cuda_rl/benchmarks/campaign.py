from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from cuda_rl.benchmarks.config import load_benchmark_config
from cuda_rl.benchmarks.hardware import _git_output
from cuda_rl.benchmarks.runner import run_benchmark
from cuda_rl.benchmarks.schemas import write_json


def run_benchmark_campaign(config_paths: list[Path]) -> tuple[int, Path]:
    if not config_paths:
        raise ValueError("at least one benchmark config is required.")
    configs = [load_benchmark_config(path) for path in config_paths]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    campaign_directory = Path("reports/benchmarks/campaigns") / timestamp
    campaign_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = campaign_directory / "manifest.json"
    results: list[dict[str, object]] = []
    manifest = {
        "timestamp": timestamp,
        "commit_sha": _git_output(["git", "rev-parse", "HEAD"]),
        "branch": _git_output(["git", "branch", "--show-current"]),
        "working_tree_status": _git_output(["git", "status", "--short"]) or "clean",
        "configs": [str(path) for path in config_paths],
        "resolved_configs": [asdict(config) for config in configs],
        "results": results,
    }
    write_json(manifest_path, manifest)

    exit_code = 0
    for path, config in zip(config_paths, configs, strict=True):
        try:
            output_directory = run_benchmark(config)
            results.append(
                {
                    "config": str(path),
                    "suite": config.suite,
                    "status": "completed",
                    "output_directory": str(output_directory),
                }
            )
        except Exception as exception:
            exit_code = 1
            results.append(
                {
                    "config": str(path),
                    "suite": config.suite,
                    "status": "failed",
                    "error": str(exception),
                }
            )
        write_json(manifest_path, manifest)

    write_json(campaign_directory / "summary.json", manifest)
    return exit_code, campaign_directory
