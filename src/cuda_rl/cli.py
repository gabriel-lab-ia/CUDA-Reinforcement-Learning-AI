from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cuda_rl.config import load_experiment_profile
from cuda_rl.experiments import ExperimentRegistry, RunStatus
from cuda_rl.reports import load_training_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuda-rl",
        description="CUDA Reinforcement Learning AI command line interface.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Run the legacy training entry point.")
    train.add_argument("training_args", nargs=argparse.REMAINDER)

    validate_profile = subcommands.add_parser(
        "profile",
        help="Validate and print an experiment profile.",
    )
    validate_profile.add_argument("path", type=Path)

    registry = subcommands.add_parser(
        "registry",
        help="Inspect a JSONL experiment registry.",
    )
    registry.add_argument("root", type=Path)
    registry.add_argument("--latest", action="store_true")

    report = subcommands.add_parser(
        "report",
        help="Render a training report from a run directory.",
    )
    report.add_argument("run_directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _run_legacy_training(args.training_args)
    elif args.command == "profile":
        profile = load_experiment_profile(args.path)
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    elif args.command == "registry":
        _run_registry(args.root, latest=args.latest)
    elif args.command == "report":
        print(load_training_report(args.run_directory).to_markdown())
    else:
        raise SystemExit(f"Unsupported command: {args.command}")


def _run_legacy_training(training_args: list[str]) -> None:
    from cuda_rl.reinforcement_learning import main as train_main

    original_argv = sys.argv
    try:
        sys.argv = ["cuda-rl train", *training_args]
        train_main()
    finally:
        sys.argv = original_argv


def _run_registry(root: Path, *, latest: bool) -> None:
    registry = ExperimentRegistry(root)
    if latest:
        run = registry.latest_run()
        print("null" if run is None else json.dumps(run.to_payload(), indent=2))
        return
    statuses: tuple[RunStatus, ...] = ("running", "completed", "failed")
    runs = [
        run.to_payload()
        for status in statuses
        for run in registry.runs_by_status(status)
    ]
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
