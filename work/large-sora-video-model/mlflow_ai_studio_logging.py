"""AI Studio MLflow logging template.

Fill the training and inference functions with project-specific logic before production use.
"""

from __future__ import annotations

import json
import os
import subprocess
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


def prepare_data(config: dict):
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    return {"train_path": train_path, "test_path": test_path}


def build_model(config: dict):
    return {"model_type": "replace-with-real-model"}


def train_model(model, data, config: dict):
    local_model_path = os.getenv("AI_STUDIO_LOCAL_MODEL_PATH", "")
    if local_model_path:
        return Path(local_model_path)
    save_path = Path(config["model"]["save_path"])
    ensure_read_write_directory(save_path)
    (save_path / "model.txt").write_text("replace with trained model artifact\n", encoding="utf-8")
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

    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]
    optimizer = config["training"].get("optimizer", "SGD")
    loss_fn = type("LossFunction", (), {})()
    metric_fn = type("MetricFunction", (), {})()

    data = prepare_data(config)
    model = build_model(config)
    model_path = train_model(model, data, config)
    input_example = build_input_example(config)

    training_metadata_path = Path("training_metadata.json")
    training_metadata_path.write_text(
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
        + "\n",
        encoding="utf-8",
    )

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": optimizer,
        }
        mlflow.log_params(params)
        mlflow.log_artifact(str(training_metadata_path))
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
