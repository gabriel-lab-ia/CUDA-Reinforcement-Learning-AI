from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from cuda_rl.benchmarks.campaign import run_benchmark_campaign

    parser = argparse.ArgumentParser(
        description="Run CUDA RL benchmark campaign configs."
    )
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    exit_code, campaign_directory = run_benchmark_campaign(args.configs)
    print(campaign_directory)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
