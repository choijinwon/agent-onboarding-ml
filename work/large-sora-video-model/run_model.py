"""Prepare a local model artifact, then run AI Studio MLflow logging.

Usage:
  python run_model.py
  python run_model.py --model ./model/my-model.onnx
  python run_model.py --env-file ai_studio.env --config config.json --register
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


MODEL_SUFFIXES = {
    ".pkl",
    ".joblib",
    ".onnx",
    ".pt",
    ".pth",
    ".keras",
    ".h5",
    ".safetensors",
}
IGNORED_DIRS = {".aiu", ".git", ".venv", "__pycache__", "registration_packages", "saved_model"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_read_write_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o775)
    except OSError:
        pass
    if os.name == "nt":
        username = os.environ.get("USERNAME") or os.environ.get("USER")
        if username:
            try:
                subprocess.run(
                    ["icacls", str(path), "/grant", f"{username}:(OI)(CI)M", "/T", "/C"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except Exception:
                pass
    return path


def resolve_model_source(config: dict, explicit_model: str = "") -> Path:
    candidates: list[Path] = []
    if explicit_model:
        candidates.append(Path(explicit_model))
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    source_path = str(config_model.get("source_path") or "").strip()
    if source_path:
        candidates.append(Path(source_path))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    root = Path.cwd()
    model_files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            model_files.append(path)
    if model_files:
        return sorted(model_files, key=lambda item: item.stat().st_size, reverse=True)[0].resolve()
    raise FileNotFoundError(
        "로컬 모델 파일을 찾지 못했습니다. --model <경로>를 지정하거나 config.json의 model.source_path를 설정하세요."
    )


def prepare_local_model(source: Path, config: dict) -> Path:
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    save_root = Path(str(config_model.get("save_path") or "saved_model"))
    ensure_read_write_directory(save_root)
    if source.is_dir():
        target = save_root / "local_model"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target.resolve()

    suffix = source.suffix or ".bin"
    target = save_root / f"local_model{suffix}"
    shutil.copy2(source, target)
    return target.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local model and run AI Studio MLflow logging.")
    parser.add_argument("--env-file", default="ai_studio.env", help="AI Studio environment file path")
    parser.add_argument("--config", default="config.json", help="AI Studio config file path")
    parser.add_argument("--model", default="", help="Existing local model file or directory path")
    parser.add_argument("--prepare-only", action="store_true", help="Only create saved_model/local_model and skip MLflow logging")
    parser.add_argument("--register", action="store_true", help="Run MLflow model logging/register after local model preparation")
    parser.add_argument("--dry-run", action="store_true", help="Show the MLflow register command path without importing mlflow")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    config_path = Path(args.config)
    config = load_config(config_path)
    os.environ["AI_STUDIO_CONFIG_PATH"] = str(config_path)

    source_model = resolve_model_source(config, args.model)
    local_model = prepare_local_model(source_model, config)
    os.environ["AI_STUDIO_LOCAL_MODEL_PATH"] = str(local_model)
    print(f"local model prepared: {local_model}")

    if args.prepare_only:
        return
    if args.dry_run:
        print("dry-run register command: python run_model.py --env-file ai_studio.env --register")
        print("mlflow tracking default: file:./mlruns when MLFLOW_TRACKING_URL is empty")
        return
    from mlflow_ai_studio_logging import main as run_mlflow_logging

    run_mlflow_logging()


if __name__ == "__main__":
    main()
