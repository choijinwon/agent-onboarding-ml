"""Run a local model through the AI Studio / MLflow registration flow.

이 파일은 AI Studio에서 직접 실행하는 진입점입니다.
`train.py`는 학습 코드 예시로 두고, 실제 등록/검증 실행은 이 파일을 사용합니다.

기본 실행:
  python run_model.py --prepare-only
  python run_model.py --env-file ai_studio.env --config config.json --register

Sora 샘플 생성 후 실행:
  python run_sora_model.py --create-only
  python run_model.py --model ./sora_model/model/sora-video-sample.onnx --prepare-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


# AI Studio에서 모델 artifact로 인정할 로컬 파일 확장자입니다.
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

# 자동 모델 탐색 시 스캔하지 않을 폴더입니다.
IGNORED_DIRS = {".aiu", ".git", ".venv", "__pycache__", "registration_packages", "saved_model"}
ENV_FIELDS = [
    ("MLFLOW_TRACKING_URL", "MLflow Tracking URL"),
    ("MLFLOW_TRACKING_USERNAME", "MLflow username"),
    ("MLFLOW_TRACKING_PASSWORD", "MLflow password"),
    ("MLFLOW_EXPERIMENT_NAME", "MLflow experiment name"),
    ("MLFLOW_REGISTER_MODEL_NAME", "MLflow registered model name"),
]


def default_ai_studio_config() -> dict[str, object]:
    """Return the minimum config.json shape expected by AI Studio registration."""

    return {
        "mlflow_tracking_url": "${MLFLOW_TRACKING_URL}",
        "mlflow_tracking_username": "${MLFLOW_TRACKING_USERNAME}",
        "mlflow_tracking_password": "${MLFLOW_TRACKING_PASSWORD}",
        "mlflow_experiment_name": "${MLFLOW_EXPERIMENT_NAME}",
        "mlflow_register_model_name": "${MLFLOW_REGISTER_MODEL_NAME}",
        "data": {
            "train_path": "data/train",
            "test_path": "data/test",
            "input_example_path": "input_example.json",
        },
        "model": {
            "source_path": "",
            "save_path": "saved_model",
            "artifact_path": "ai_studio",
            "code_path": ["aiu_custom"],
            "requirements": "requirements.txt",
        },
        "execution": {
            "entrypoint": "run_model.py",
            "blocked_entrypoints": ["train.py"],
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "SGD",
        },
    }


def default_input_example() -> dict[str, object]:
    """Return a small pandas-compatible input example for MLflow pyfunc logging."""

    return {
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[0.1, 0.2, 0.3]],
    }


def default_env_text() -> str:
    return """# AI Studio MLflow environment.
# Fill these values before running: python run_model.py --env-file ai_studio.env --register
# MLFLOW_TRACKING_URL is also mapped to MLFLOW_TRACKING_URI automatically.
MLFLOW_TRACKING_URL=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
MLFLOW_EXPERIMENT_NAME=ai-studio-onboarding
MLFLOW_REGISTER_MODEL_NAME=ai-studio-model
"""


def default_model_wrapper_text() -> str:
    return '''"""AI Studio MLflow pyfunc model wrapper."""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import pandas as pd


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Replace predict() with the real inference logic for your model."""

    def load_context(self, context):
        self.model_path = Path(context.artifacts["model"])
        self.config_path = Path(context.artifacts["config"])

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        return model_input
'''


def default_mlflow_logging_text() -> str:
    return '''"""AI Studio MLflow logging template.

This file is called by run_model.py after the local model is prepared.
Fill prepare_data(), build_model(), and train_model() with project-specific logic
when moving from POC to production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import pandas as pd

from aiu_custom import ModelWrapper


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


def prepare_data(config: dict):
    return {
        "train_path": config["data"]["train_path"],
        "test_path": config["data"]["test_path"],
    }


def build_model(config: dict):
    return {"model_type": "replace-with-real-model"}


def train_model(model, data, config: dict):
    local_model_path = os.getenv("AI_STUDIO_LOCAL_MODEL_PATH", "")
    if local_model_path:
        return Path(local_model_path)
    save_path = Path(config["model"]["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "model.txt").write_text("replace with trained model artifact\\n", encoding="utf-8")
    return save_path


def build_input_example(config: dict):
    input_example_path = Path(config["data"]["input_example_path"])
    if input_example_path.exists():
        payload = json.loads(input_example_path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload["data"], columns=payload["columns"])
    return pd.DataFrame([[0.1, 0.2, 0.3]], columns=["feature_1", "feature_2", "feature_3"])


def main() -> None:
    config_path = Path(os.getenv("AI_STUDIO_CONFIG_PATH", "config.json"))
    config = load_config(str(config_path))

    tracking_url = resolve_env(config["mlflow_tracking_url"]) or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_url)

    username = resolve_env(config["mlflow_tracking_username"])
    password = resolve_env(config["mlflow_tracking_password"])
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    experiment_name = resolve_env(config["mlflow_experiment_name"]) or "ai-studio-onboarding"
    registered_model_name = resolve_env(config["mlflow_register_model_name"]) or "ai-studio-model"
    mlflow.set_experiment(experiment_name)

    data = prepare_data(config)
    model = build_model(config)
    model_path = train_model(model, data, config)
    input_example = build_input_example(config)

    training = config["training"]
    params = {
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "loss_function": "LossFunction",
        "metric_function": "MetricFunction",
        "optimizer": training.get("optimizer", "SGD"),
    }

    metadata_path = Path("training_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "train_data": data["train_path"],
                "test_data": data["test_path"],
                "saved_model": str(model_path),
                "input_example": config["data"]["input_example_path"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\\n",
        encoding="utf-8",
    )

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_text(json.dumps({"model": str(model)}, ensure_ascii=False, indent=2), "model_summary.json")
        mlflow.pyfunc.log_model(
            python_model=ModelWrapper(),
            artifact_path=config["model"]["artifact_path"],
            code_path=config["model"]["code_path"],
            artifacts={
                "model": str(model_path),
                "config": str(config_path),
            },
            registered_model_name=registered_model_name,
            pip_requirements=config["model"]["requirements"],
            input_example=input_example,
        )


if __name__ == "__main__":
    main()
'''


def ensure_ai_studio_process_files(env_file: Path, config_file: Path) -> list[str]:
    """Create missing AI Studio process files without overwriting user edits."""

    created: list[str] = []
    if not env_file.exists():
        env_file.write_text(default_env_text(), encoding="utf-8")
        created.append(str(env_file))
    if not config_file.exists():
        config_file.write_text(json.dumps(default_ai_studio_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created.append(str(config_file))

    input_example_path = Path("input_example.json")
    if not input_example_path.exists():
        input_example_path.write_text(json.dumps(default_input_example(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created.append(str(input_example_path))

    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        requirements_path.write_text("mlflow\ncloudpickle\npandas\nnumpy\n", encoding="utf-8")
        created.append(str(requirements_path))

    custom_dir = ensure_read_write_directory(Path("aiu_custom"))
    init_path = custom_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("from .model_wrapper import ModelWrapper\n", encoding="utf-8")
        created.append(str(init_path))
    wrapper_path = custom_dir / "model_wrapper.py"
    if not wrapper_path.exists():
        wrapper_path.write_text(default_model_wrapper_text(), encoding="utf-8")
        created.append(str(wrapper_path))

    logging_path = Path("mlflow_ai_studio_logging.py")
    if not logging_path.exists():
        logging_path.write_text(default_mlflow_logging_text(), encoding="utf-8")
        created.append(str(logging_path))
    return created


def read_env_values(path: Path) -> dict[str, str]:
    """Read known AI Studio environment keys from an env file."""

    values = {key: "" for key, _ in ENV_FIELDS}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            values[key] = value.strip().strip('"').strip("'")
    return values


def write_env_values(path: Path, values: dict[str, str]) -> None:
    """Write AI Studio env values in a predictable order."""

    lines = [
        "# AI Studio MLflow environment.",
        "# Generated by: python run_model.py --setup-env",
        "# MLFLOW_TRACKING_URL is also mapped to MLFLOW_TRACKING_URI automatically.",
    ]
    for key, _ in ENV_FIELDS:
        lines.append(f"{key}={values.get(key, '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prompt_ai_studio_env(path: Path) -> dict[str, str]:
    """Show a small console setup screen and save values to ai_studio.env."""

    values = read_env_values(path)
    print("")
    print("AI Studio MLflow 환경설정")
    print("값을 입력하고 Enter를 누르세요. 빈 값은 기존 값을 유지합니다.")
    print("")
    for key, label in ENV_FIELDS:
        current = values.get(key, "")
        suffix = f" [{current}]" if current else ""
        entered = input(f"{label} ({key}){suffix}: ").strip()
        if entered:
            values[key] = entered
    write_env_values(path, values)
    print("")
    print(f"saved: {path}")
    print("next: python run_model.py --env-file ai_studio.env --register")
    return values


def load_env_file(path: Path) -> None:
    """Load AI Studio / MLflow environment values from ai_studio.env.

    지원 값:
    - MLFLOW_TRACKING_URL
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD
    - MLFLOW_EXPERIMENT_NAME
    - MLFLOW_REGISTER_MODEL_NAME
    """

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
    tracking_url = os.environ.get("MLFLOW_TRACKING_URL", "").strip()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if tracking_url and not tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_url
    elif tracking_uri and not tracking_url:
        os.environ["MLFLOW_TRACKING_URL"] = tracking_uri


def load_config(path: Path) -> dict:
    """Read config.json.

    config.json에는 데이터 경로, 모델 저장 경로, MLflow artifact path,
    code_path, requirements.txt 위치가 들어갑니다.
    """

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_read_write_directory(path: Path) -> Path:
    """Create a writable directory for Windows 10/11, macOS, and Linux."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o775)
    except OSError:
        pass

    # Windows 10/11 폐쇄망 환경에서는 생성 폴더 권한이 막히는 경우가 있어
    # 현재 사용자에게 수정 권한을 한 번 더 부여합니다.
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
    """Find the model file that should be prepared for AI Studio.

    우선순위:
    1. CLI의 --model 경로
    2. config.json의 model.source_path
    3. 현재 프로젝트 아래에서 가장 큰 모델 artifact 자동 탐색
    """

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
        "로컬 모델 파일을 찾지 못했습니다. --model <경로>를 지정하거나 "
        "config.json의 model.source_path를 설정하세요."
    )


def prepare_local_model(source: Path, config: dict) -> Path:
    """Copy the selected model into saved_model/local_model.*.

    AI Studio 등록 단계에서는 원본 모델을 직접 건드리지 않고,
    saved_model 아래 복사본을 MLflow artifact로 사용합니다.
    """

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
    parser.add_argument("--setup-env", action="store_true", help="Open an interactive AI Studio env setup screen")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    config_path = Path(args.config)

    print("[1/6] AI Studio process files check")
    created = ensure_ai_studio_process_files(env_path, config_path)
    if created:
        print("created: " + ", ".join(created))
    else:
        print("all process files already exist")

    if args.setup_env:
        prompt_ai_studio_env(env_path)
        return

    # 1. AI Studio 환경값을 로딩합니다.
    print("[2/6] Load AI Studio environment")
    load_env_file(env_path)

    # 2. config.json을 읽고 MLflow logging 템플릿이 사용할 경로를 환경변수로 넘깁니다.
    print("[3/6] Load config.json")
    config = load_config(config_path)
    os.environ["AI_STUDIO_CONFIG_PATH"] = str(config_path)

    # 3. 모델 파일을 찾고 saved_model 아래에 실행용 복사본을 만듭니다.
    print("[4/6] Resolve and prepare local model")
    source_model = resolve_model_source(config, args.model)
    local_model = prepare_local_model(source_model, config)
    os.environ["AI_STUDIO_LOCAL_MODEL_PATH"] = str(local_model)
    print(f"local model prepared: {local_model}")

    # 4. 검증만 필요하면 여기서 종료합니다.
    if args.prepare_only:
        print("[5/6] prepare-only mode: skip MLflow logging")
        return

    # 5. MLflow import 없이 등록 명령만 확인할 때 사용합니다.
    if args.dry_run:
        print("[5/6] dry-run mode: skip MLflow import")
        print("dry-run register command: python run_model.py --env-file ai_studio.env --register")
        print("mlflow tracking default: file:./mlruns when MLFLOW_TRACKING_URL is empty")
        return

    # 6. 실제 MLflow logging/register는 별도 템플릿에 위임합니다.
    #    샘플 또는 Wizard 적용 후 생성되는 mlflow_ai_studio_logging.py가 필요합니다.
    if not args.register:
        print("[5/6] register flag was not set")
        print("register flag was not set. Use --register to run MLflow logging.")
        return

    print("[5/6] Run MLflow model logging")
    from mlflow_ai_studio_logging import main as run_mlflow_logging

    run_mlflow_logging()
    print("[6/6] AI Studio MLflow process complete")


if __name__ == "__main__":
    main()
