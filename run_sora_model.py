"""Create and run a local Sora-style model sample from the repository root.

Usage:
  python run_sora_model.py
  python run_sora_model.py --prepare-only
  python run_sora_model.py --register
  python run_sora_model.py --target ./work/sora-model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from deep_agent.cli import SAMPLE_MODEL_SPECS, create_model_sample


def create_sora_model(target: Path) -> Path:
    return create_model_sample(target, SAMPLE_MODEL_SPECS["sora"])


def run_sample(sample_root: Path, args: argparse.Namespace) -> int:
    command = [sys.executable, "run_model.py"]
    if args.prepare_only:
        command.append("--prepare-only")
    if args.register:
        command.append("--register")
    if args.dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=sample_root, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and run a local Sora-style model sample.")
    parser.add_argument("--target", default="sora_model", help="Directory to create the Sora sample in")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare saved_model/local_model only")
    parser.add_argument("--register", action="store_true", help="Run MLflow logging/register after preparation")
    parser.add_argument("--dry-run", action="store_true", help="Show registration command without importing mlflow")
    parser.add_argument("--create-only", action="store_true", help="Only create sample files and skip run_model.py")
    args = parser.parse_args()

    sample_root = create_sora_model(Path(args.target).resolve())
    print(f"Sora model sample ready: {sample_root}")
    print(f"Artifact: {sample_root / SAMPLE_MODEL_SPECS['sora'].artifact_path}")
    print(f"Runner: {sample_root / 'run_model.py'}")

    if args.create_only:
        return 0
    return run_sample(sample_root, args)


if __name__ == "__main__":
    raise SystemExit(main())
