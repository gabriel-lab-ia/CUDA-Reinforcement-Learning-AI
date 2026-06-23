from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from cuda_rl.benchmarks import run_benchmark_config

    parser = argparse.ArgumentParser(description="Run CUDA RL benchmark configs.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_directory = run_benchmark_config(args.config)
    print(output_directory)


if __name__ == "__main__":
    main()
