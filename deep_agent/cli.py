#!/usr/bin/env python3
"""Console POC for AI ML onboarding launch modes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from deep_agent.app_config import (
    AppConfig,
    ensure_local_file_access,
    ensure_read_write_directory,
    ensure_runtime_layout,
    format_config_summary,
)
from deep_agent.path_utils import resolve_filesystem_path
from deep_agent.profile import build_ml_platform_profile, format_profile
from deep_agent.libs import deepagents_libs_as_dict, format_deepagents_libs
from deep_agent.stores.error_log_store import (
    analyze_error_log,
    format_error_analysis,
    format_error_log_list,
    list_error_logs,
    save_error_log,
)
from deep_agent.stores.prompt_store import (
    export_prompt_templates_to_wiki,
    format_prompt_templates,
    load_prompt_templates,
    prompt_templates_as_dict,
)


MODE_BEGINNER = "beginner"
MODE_INTERMEDIATE = "intermediate"
MODE_ADVANCED = "advanced"

MODE_LABELS = {
    MODE_BEGINNER: "초급자 모드",
    MODE_INTERMEDIATE: "중급자 모드",
    MODE_ADVANCED: "고급자 모드",
}

MODE_ALIASES = {
    "1": MODE_BEGINNER,
    "beginner": MODE_BEGINNER,
    "초급자": MODE_BEGINNER,
    "초급": MODE_BEGINNER,
    "2": MODE_INTERMEDIATE,
    "mid": MODE_INTERMEDIATE,
    "middle": MODE_INTERMEDIATE,
    "intermediate": MODE_INTERMEDIATE,
    "미드": MODE_INTERMEDIATE,
    "중급자": MODE_INTERMEDIATE,
    "중급": MODE_INTERMEDIATE,
    "3": MODE_ADVANCED,
    "advanced": MODE_ADVANCED,
    "고급자": MODE_ADVANCED,
    "고급": MODE_ADVANCED,
}

MODE_CHANGE_MESSAGES = {
    MODE_BEGINNER: "이제부터 단계별 Wizard 방식으로 안내합니다.",
    MODE_INTERMEDIATE: "이제부터 Chat + Wizard 혼합 방식으로 안내합니다.",
    MODE_ADVANCED: "이제부터 CLI Command 중심으로 안내합니다.",
}

ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "chrome": "\033[38;2;86;92;112m",
    "window": "\033[38;2;200;204;216m",
    "title": "\033[1m\033[38;2;245;245;245m",
    "panel": "\033[38;2;230;230;230m",
    "normal": "\033[38;2;235;235;235m",
    "muted": "\033[38;2;130;130;130m",
    "accent": "\033[38;2;87;166;255m",
    "input": "\033[38;2;245;245;245m",
    "status": "\033[38;2;170;170;170m",
}
ANSI_BACKGROUND_STYLES = {
    "chrome": "\033[48;2;6;6;7m\033[38;2;86;92;112m",
    "window": "\033[48;2;48;52;72m\033[38;2;200;204;216m",
    "title": "\033[1m\033[48;2;20;20;20m\033[38;2;245;245;245m",
    "panel": "\033[48;2;20;20;20m\033[38;2;230;230;230m",
    "normal": "\033[48;2;6;6;7m\033[38;2;235;235;235m",
    "muted": "\033[48;2;6;6;7m\033[38;2;130;130;130m",
    "accent": "\033[48;2;20;20;20m\033[38;2;87;166;255m",
    "input": "\033[48;2;31;31;31m\033[38;2;245;245;245m",
    "status": "\033[48;2;6;6;7m\033[38;2;170;170;170m",
}
ANSI_INPUT_PANEL_STYLES = {
    "accent": "\033[48;5;235m\033[38;5;75m",
    "text": "\033[48;5;235m\033[38;5;255m",
    "muted": "\033[48;5;235m\033[38;5;245m",
    "cursor": "\033[48;5;235m\033[38;5;255m",
}

_RICH_CONSOLE_ENABLED: bool | None = None
RICH_TUI_WIDTH = 118

MODEL_ARTIFACT_SUFFIXES = {
    ".h5",
    ".joblib",
    ".keras",
    ".mlmodel",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}

HEAVY_SAMPLE_ALIASES = {
    "/sample heavy",
    "/sample-heavy",
    "/샘플 대형모델",
    "/샘플 무거운모델",
    "heavy",
    "heavy model",
    "대형모델",
    "무거운모델",
    "샘플 대형모델",
    "샘플 무거운모델",
}

LARGE10_SAMPLE_ALIASES = {
    "/sample large10",
    "/sample big10",
    "/sample heavy10",
    "/samples large10",
    "/samples big10",
    "/샘플 대형10",
    "/샘플 무거운10",
    "large10",
    "big10",
    "heavy10",
    "대형10",
    "무거운10",
    "샘플 대형10",
    "샘플 무거운10",
}

SAMPLE_KIND_ALIASES = {
    "/sample heavy": "heavy",
    "/sample-heavy": "heavy",
    "/샘플 대형모델": "heavy",
    "/샘플 무거운모델": "heavy",
    "heavy": "heavy",
    "heavy model": "heavy",
    "대형모델": "heavy",
    "무거운모델": "heavy",
    "샘플 대형모델": "heavy",
    "샘플 무거운모델": "heavy",
    "/sample tensorflow": "tensorflow",
    "/sample tf": "tensorflow",
    "/샘플 텐서플로우": "tensorflow",
    "/샘플 텐션플러워": "tensorflow",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "텐서플로우": "tensorflow",
    "텐션플러워": "tensorflow",
    "샘플 텐서플로우": "tensorflow",
    "샘플 텐션플러워": "tensorflow",
    "/sample pytorch": "pytorch",
    "/sample torch": "pytorch",
    "/샘플 파이토치": "pytorch",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "파이토치": "pytorch",
    "샘플 파이토치": "pytorch",
    "/sample sklearn": "sklearn",
    "/sample scikit": "sklearn",
    "/샘플 사이킷런": "sklearn",
    "sklearn": "sklearn",
    "scikit": "sklearn",
    "사이킷런": "sklearn",
    "샘플 사이킷런": "sklearn",
    "/sample onnx": "onnx",
    "/샘플 onnx": "onnx",
    "onnx": "onnx",
    "샘플 onnx": "onnx",
    "/sample sora": "sora",
    "/샘플 소라": "sora",
    "sora": "sora",
    "소라": "sora",
    "소라모델": "sora",
    "샘플 소라": "sora",
    "샘플 소라모델": "sora",
    "/sample sora-error": "sora_error",
    "/sample broken-sora": "sora_error",
    "/샘플 소라오류": "sora_error",
    "sora-error": "sora_error",
    "broken-sora": "sora_error",
    "소라오류": "sora_error",
    "소라에러": "sora_error",
    "샘플 소라오류": "sora_error",
}

DEFAULT_HEAVY_SAMPLE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SampleModelSpec:
    kind: str
    title: str
    directory: str
    artifact_path: str
    artifact_size_bytes: int
    requirements: list[str]
    train_body: str


SAMPLE_MODEL_SPECS = {
    "heavy": SampleModelSpec(
        kind="heavy",
        title="대형 ONNX 모델",
        directory="heavy-model",
        artifact_path="model/heavy-model.onnx",
        artifact_size_bytes=DEFAULT_HEAVY_SAMPLE_BYTES,
        requirements=["mlflow==2.17.0", "scikit-learn==1.5.2", "pandas==2.2.3"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/heavy-model.onnx')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('model_path', args.model_path)\n"
            "        mlflow.log_metric('sample_accuracy', 0.0)\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "tensorflow": SampleModelSpec(
        kind="tensorflow",
        title="TensorFlow Keras 모델",
        directory="tensorflow-model",
        artifact_path="model/tensorflow-sample.keras",
        artifact_size_bytes=24 * 1024 * 1024,
        requirements=["tensorflow==2.17.0", "numpy==1.26.4"],
        train_body=(
            "import argparse\n"
            "\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--epochs', type=int, default=1)\n"
            "    parser.add_argument('--model-path', default='model/tensorflow-sample.keras')\n"
            "    args = parser.parse_args()\n"
            "    print(f'TensorFlow sample model: {args.model_path}, epochs={args.epochs}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "pytorch": SampleModelSpec(
        kind="pytorch",
        title="PyTorch 모델",
        directory="pytorch-model",
        artifact_path="model/pytorch-sample.pt",
        artifact_size_bytes=32 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "torch==2.5.1", "numpy==1.26.4"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--batch-size', type=int, default=8)\n"
            "    parser.add_argument('--model-path', default='model/pytorch-sample.pt')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'pytorch')\n"
            "        mlflow.log_param('batch_size', args.batch_size)\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "sklearn": SampleModelSpec(
        kind="sklearn",
        title="scikit-learn Joblib 모델",
        directory="sklearn-model",
        artifact_path="model/sklearn-sample.joblib",
        artifact_size_bytes=4 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "scikit-learn==1.5.2", "joblib==1.4.2"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/sklearn-sample.joblib')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'scikit-learn')\n"
            "        mlflow.log_metric('sample_score', 0.0)\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "onnx": SampleModelSpec(
        kind="onnx",
        title="ONNX 모델",
        directory="onnx-model",
        artifact_path="model/onnx-sample.onnx",
        artifact_size_bytes=16 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "onnx==1.17.0", "onnxruntime==1.20.1"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/onnx-sample.onnx')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'onnx')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "sora": SampleModelSpec(
        kind="sora",
        title="Sora 스타일 비디오 생성 모델",
        directory="sora-video-model",
        artifact_path="model/sora-video-sample.onnx",
        artifact_size_bytes=64 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "torch==2.5.1", "opencv-python==4.10.0.84"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--prompt', default='a cinematic product demo')\n"
            "    parser.add_argument('--duration-seconds', type=int, default=4)\n"
            "    parser.add_argument('--model-path', default='model/sora-video-sample.onnx')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('model_family', 'sora-style-video-generation')\n"
            "        mlflow.log_param('prompt', args.prompt)\n"
            "        mlflow.log_param('duration_seconds', args.duration_seconds)\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    "sora_error": SampleModelSpec(
        kind="sora_error",
        title="오류 재현용 Sora 스타일 비디오 모델",
        directory="sora-error-model",
        artifact_path="outputs/sora-preview.mp4",
        artifact_size_bytes=8 * 1024 * 1024,
        requirements=["torch==2.5.1", "opencv-python==4.10.0.84"],
        train_body=(
            "import argparse\n"
            "\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--prompt', default='broken registration sample')\n"
            "    parser.add_argument('--preview-path', default='outputs/sora-preview.mp4')\n"
            "    args = parser.parse_args()\n"
            "    print(f'Sora preview only: {args.preview_path}, prompt={args.prompt}')\n"
            "    # Intentionally missing experiment tracking and a supported model artifact.\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
}


LARGE_MODEL_SAMPLE_SPECS = (
    SampleModelSpec(
        kind="large_tensorflow",
        title="대형 TensorFlow Keras 모델",
        directory="large-tensorflow-model",
        artifact_path="model/large-tensorflow.keras",
        artifact_size_bytes=96 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "tensorflow==2.17.0", "numpy==1.26.4"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/large-tensorflow.keras')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'tensorflow')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_pytorch",
        title="대형 PyTorch 체크포인트",
        directory="large-pytorch-checkpoint",
        artifact_path="model/large-pytorch.pt",
        artifact_size_bytes=128 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "torch==2.5.1", "numpy==1.26.4"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/large-pytorch.pt')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'pytorch')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_onnx_transformer",
        title="대형 ONNX Transformer 모델",
        directory="large-onnx-transformer",
        artifact_path="model/large-transformer.onnx",
        artifact_size_bytes=192 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "onnx==1.17.0", "onnxruntime==1.20.1"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/large-transformer.onnx')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('architecture', 'transformer')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_sklearn_bundle",
        title="대형 scikit-learn 번들",
        directory="large-sklearn-bundle",
        artifact_path="model/large-sklearn.joblib",
        artifact_size_bytes=72 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "scikit-learn==1.5.2", "joblib==1.4.2"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/large-sklearn.joblib')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('framework', 'scikit-learn')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_sora_video",
        title="대형 Sora 스타일 비디오 모델",
        directory="large-sora-video-model",
        artifact_path="model/large-sora-video.onnx",
        artifact_size_bytes=256 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "torch==2.5.1", "opencv-python==4.10.0.84"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/large-sora-video.onnx')\n"
            "    parser.add_argument('--duration-seconds', type=int, default=8)\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('model_family', 'sora-style-video-generation')\n"
            "        mlflow.log_param('duration_seconds', args.duration_seconds)\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_vision_embedding",
        title="대형 Vision Embedding 모델",
        directory="large-vision-embedding",
        artifact_path="model/vision-embedding.pt",
        artifact_size_bytes=144 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "torch==2.5.1", "pillow==11.0.0"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/vision-embedding.pt')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('task', 'vision-embedding')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_llm_adapter",
        title="대형 LLM Adapter 모델",
        directory="large-llm-adapter",
        artifact_path="model/llm-adapter.safetensors",
        artifact_size_bytes=160 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "transformers==4.46.3", "safetensors==0.4.5"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/llm-adapter.safetensors')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('task', 'llm-adapter')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_recommender",
        title="대형 추천 모델",
        directory="large-recommender-model",
        artifact_path="model/recommender.pkl",
        artifact_size_bytes=80 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "scikit-learn==1.5.2", "pandas==2.2.3"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/recommender.pkl')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('task', 'recommender')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_timeseries",
        title="대형 시계열 예측 모델",
        directory="large-timeseries-forecast",
        artifact_path="model/timeseries-forecast.onnx",
        artifact_size_bytes=112 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "onnx==1.17.0", "pandas==2.2.3"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/timeseries-forecast.onnx')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('task', 'time-series-forecasting')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
    SampleModelSpec(
        kind="large_tabular_ensemble",
        title="대형 Tabular Ensemble 모델",
        directory="large-tabular-ensemble",
        artifact_path="model/tabular-ensemble.joblib",
        artifact_size_bytes=88 * 1024 * 1024,
        requirements=["mlflow==2.17.0", "xgboost==2.1.2", "joblib==1.4.2"],
        train_body=(
            "import argparse\n"
            "import mlflow\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--model-path', default='model/tabular-ensemble.joblib')\n"
            "    args = parser.parse_args()\n"
            "    with mlflow.start_run():\n"
            "        mlflow.log_param('task', 'tabular-ensemble')\n"
            "        mlflow.log_artifact(args.model_path)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    ),
)


@dataclass(frozen=True)
class ProjectIssue:
    code: str
    severity: str
    title: str
    target: str
    explanation: str
    recommendation: str
    fixable_by_agent: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "target": self.target,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "fixable_by_agent": self.fixable_by_agent,
        }


@dataclass(frozen=True)
class FixPreview:
    code: str
    title: str
    target: str
    action: str
    preview_lines: list[str]
    requires_approval: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "title": self.title,
            "target": self.target,
            "action": self.action,
            "preview_lines": self.preview_lines,
            "requires_approval": self.requires_approval,
        }


@dataclass(frozen=True)
class ApprovalOption:
    key: str
    label: str
    description: str
    will_modify_files: bool
    enabled: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "will_modify_files": self.will_modify_files,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class AppliedChange:
    code: str
    target: str
    status: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "target": self.target,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class FileStat:
    path: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size": format_bytes(self.size_bytes),
        }


@dataclass(frozen=True)
class ProjectScan:
    path: str
    exists: bool
    is_directory: bool
    file_count: int
    directory_count: int
    total_bytes: int
    python_file_count: int
    model_artifacts: list[FileStat]
    largest_files: list[FileStat]
    scan_note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_directory": self.is_directory,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
            "total_size": format_bytes(self.total_bytes),
            "python_file_count": self.python_file_count,
            "model_artifacts": [artifact.as_dict() for artifact in self.model_artifacts],
            "largest_files": [file.as_dict() for file in self.largest_files],
            "scan_note": self.scan_note,
        }


@dataclass(frozen=True)
class RegistrationCheck:
    code: str
    label: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LocalServingPlan:
    status: str
    mode: str
    host: str
    port: int
    health_endpoint: str
    predict_endpoint: str
    checks: list[RegistrationCheck]
    commands: list[str]
    notes: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "host": self.host,
            "port": self.port,
            "health_endpoint": self.health_endpoint,
            "predict_endpoint": self.predict_endpoint,
            "checks": [check.as_dict() for check in self.checks],
            "commands": self.commands,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ProjectAnalysis:
    path: str
    exists: bool
    is_directory: bool
    scan: ProjectScan
    registration_status: str
    registration_checks: list[RegistrationCheck]
    requirements_files: list[str]
    has_mlflow_dependency: bool
    mlflow_usage_files: list[str]
    entrypoint_candidates: list[str]
    model_artifacts: list[str]
    job_template_ready: bool
    local_serving: LocalServingPlan
    issues: list[str]
    issue_details: list[ProjectIssue]
    next_actions: list[str]
    model_parameters: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_directory": self.is_directory,
            "scan": self.scan.as_dict(),
            "registration_status": self.registration_status,
            "registration_checks": [check.as_dict() for check in self.registration_checks],
            "requirements_files": self.requirements_files,
            "has_mlflow_dependency": self.has_mlflow_dependency,
            "mlflow_usage_files": self.mlflow_usage_files,
            "entrypoint_candidates": self.entrypoint_candidates,
            "model_artifacts": self.model_artifacts,
            "job_template_ready": self.job_template_ready,
            "local_serving": self.local_serving.as_dict(),
            "issues": self.issues,
            "issue_details": [issue.as_dict() for issue in self.issue_details],
            "next_actions": self.next_actions,
            "model_parameters": self.model_parameters,
        }


@dataclass(frozen=True)
class CommandResult:
    command: str
    path: str
    status: str
    exit_code: int
    details: list[str]
    result_file: str | None = None
    analysis: ProjectAnalysis | None = None
    fix_previews: list[FixPreview] | None = None
    approval_options: list[ApprovalOption] | None = None
    applied_changes: list[AppliedChange] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "command": self.command,
            "path": self.path,
            "status": self.status,
            "exit_code": self.exit_code,
            "details": self.details,
            "result_file": self.result_file,
        }
        if self.analysis:
            payload["analysis"] = self.analysis.as_dict()
        if self.fix_previews is not None:
            payload["fix_previews"] = [preview.as_dict() for preview in self.fix_previews]
        if self.approval_options is not None:
            payload["approval_options"] = [option.as_dict() for option in self.approval_options]
        if self.applied_changes is not None:
            payload["applied_changes"] = [change.as_dict() for change in self.applied_changes]
        return payload


def rich_console_enabled() -> bool:
    global _RICH_CONSOLE_ENABLED
    if _RICH_CONSOLE_ENABLED is not None:
        return _RICH_CONSOLE_ENABLED
    forced = os.environ.get("FORCE_COLOR", "").strip().lower() in {"1", "true", "yes", "on"}
    if os.environ.get("NO_COLOR") and not forced:
        _RICH_CONSOLE_ENABLED = False
        return _RICH_CONSOLE_ENABLED
    config = AppConfig.load()
    enabled = config.get_bool("ENABLE_RICH_CONSOLE")
    _RICH_CONSOLE_ENABLED = enabled and (forced or sys.stdout.isatty())
    return _RICH_CONSOLE_ENABLED


def tui_style(line: str, role: str = "normal") -> str:
    if not rich_console_enabled():
        return line
    styles = ANSI_BACKGROUND_STYLES if tui_background_enabled() else ANSI_STYLES
    return f"{styles.get(role, styles['normal'])}{line}{ANSI_RESET}"


def tui_segment(text: str, role: str = "normal") -> str:
    if not rich_console_enabled():
        return text
    styles = ANSI_BACKGROUND_STYLES if tui_background_enabled() else ANSI_STYLES
    return f"{styles.get(role, styles['normal'])}{text}{ANSI_RESET}"


def tui_background_enabled() -> bool:
    if os.environ.get("DISABLE_TUI_BACKGROUND"):
        return False
    return AppConfig.load().get_bool("ENABLE_TUI_BACKGROUND")


def tui_input_panel_enabled() -> bool:
    if os.environ.get("DISABLE_TUI_INPUT_PANEL"):
        return False
    return AppConfig.load().get_bool("ENABLE_TUI_INPUT_PANEL")


def style_tui_lines(lines: list[str], roles: list[str]) -> str:
    return "\n".join(tui_style(line, roles[index] if index < len(roles) else "normal") for index, line in enumerate(lines))


def rich_row(text: str = "", role: str = "normal", indent: int = 2, width: int = RICH_TUI_WIDTH) -> str:
    body_width = max(width - indent, 1)
    body = truncate_cell(text, body_width).ljust(body_width)
    return tui_style(" " * indent + body, role)


def rich_card_line(text: str = "", role: str = "panel", accent: bool = False, width: int = RICH_TUI_WIDTH) -> str:
    if not accent:
        return rich_row(text, role=role, indent=4, width=width)
    content_width = width - 5
    content = " " + truncate_cell(text, content_width - 1).ljust(content_width - 1)
    return tui_segment("  |", "accent") + tui_segment(content, role)


def rich_input_panel_line(
    left_text: str,
    middle_text: str = "",
    right_text: str = "",
    width: int = RICH_TUI_WIDTH,
) -> str:
    if not rich_console_enabled() or not tui_input_panel_enabled():
        content = " " + " ".join(part for part in [left_text, middle_text, right_text] if part).strip()
        return rich_card_line(content, role="input", accent=True, width=width)
    content_width = width - 5
    left = left_text.rstrip()
    middle = middle_text.rstrip()
    right = right_text.rstrip()
    used = len(left) + len(middle) + len(right)
    gap_count = 2 if middle and right else 1 if middle or right else 0
    padding = max(content_width - used - gap_count, 0)
    left_style = ANSI_INPUT_PANEL_STYLES["cursor"] if left == "█" else ANSI_INPUT_PANEL_STYLES["accent"]
    middle_gap = " " if middle else ""
    right_gap = " " if right else ""
    return (
        tui_segment("  |", "accent")
        + f"{ANSI_INPUT_PANEL_STYLES['text']} "
        + f"{left_style}{left}"
        + f"{ANSI_INPUT_PANEL_STYLES['text']}{middle_gap}{middle}"
        + f"{ANSI_INPUT_PANEL_STYLES['muted']}{right_gap}{right}"
        + f"{ANSI_INPUT_PANEL_STYLES['text']}{' ' * padding}{ANSI_RESET}"
    )


def render_launch_screen() -> str:
    if rich_console_enabled():
        return render_rich_launch_screen()
    roles = []
    for line in LAUNCH_SCREEN.splitlines():
        if line.startswith("+"):
            roles.append("chrome")
        elif "# Launch" in line:
            roles.append("title")
        elif line.strip().startswith("| >") or "Agents:" in line:
            roles.append("accent")
        elif "esc interrupt" in line:
            roles.append("status")
        elif line.strip() == "|":
            roles.append("normal")
        else:
            roles.append("panel")
    return style_tui_lines(LAUNCH_SCREEN.splitlines(), roles)


def render_rich_launch_screen() -> str:
    rows = [
        rich_row("●  ●  ●   AI ML Onboarding Console ...", role="window"),
        rich_row(),
        rich_card_line("# Launch workflow", role="title", accent=True),
        rich_card_line("사용자 모드를 선택하세요.", role="panel"),
        rich_row(),
        rich_card_line("처음 사용하는 경우에는 1. 초급자 모드를 권장합니다.", role="panel", accent=True),
        rich_row(),
        rich_card_line("1. 초급자 모드    Plan(read-only) -> Build(approval-gated apply)", role="panel"),
        rich_card_line("2. 중급자 모드    Chat + Wizard", role="panel"),
        rich_card_line("3. 고급자 모드    CLI Command", role="panel"),
        rich_row(),
        rich_row(".........  esc interrupt      ctrl+space gap   tab agents   ctrl+p commands", role="status"),
    ]
    return "\n".join(rows)


class ConsoleAssistant:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        clear_fn: Callable[[], None] | None = None,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.clear_fn = clear_fn or clear_console
        self.mode = MODE_BEGINNER

    def run(self) -> None:
        self.show_launch_screen()
        selected = self.read_mode_selection()
        self.set_mode(selected)
        self.show_mode_intro()
        self.run_current_mode()

    def show_launch_screen(self) -> None:
        self.output_fn(render_launch_screen())

    def read_mode_selection(self) -> str:
        while True:
            raw = self.input_fn("선택 > ").strip()
            mode = parse_mode(raw)
            if mode:
                return mode
            self.output_fn(
                "처음 사용하는 경우에는 초급자 모드를 추천합니다.\n"
                "초급자 모드는 파일을 바로 수정하지 않고,\n"
                "분석 결과와 수정안 미리보기를 먼저 보여줍니다."
            )

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def change_mode(self, command: str) -> bool:
        mode = parse_mode_command(command)
        if not mode:
            return False
        self.set_mode(mode)
        self.output_fn(
            f"현재 모드가 {MODE_LABELS[mode]}로 변경되었습니다.\n"
            f"{MODE_CHANGE_MESSAGES[mode]}"
        )
        return True

    def show_mode_intro(self) -> None:
        if self.mode == MODE_BEGINNER:
            self.output_fn(build_beginner_intro())
        elif self.mode == MODE_INTERMEDIATE:
            self.output_fn(INTERMEDIATE_INTRO)
        else:
            self.output_fn(ADVANCED_INTRO)

    def run_current_mode(self) -> None:
        if self.mode == MODE_BEGINNER:
            self.run_beginner_mode()
        elif self.mode == MODE_INTERMEDIATE:
            self.run_intermediate_mode()
        else:
            self.run_advanced_mode()

    def run_beginner_mode(self) -> None:
        project_path = self.input_fn("> ").strip()
        if self.change_mode(project_path):
            self.run_current_mode()
            return
        project_path, message = resolve_beginner_project_input(project_path)
        if message:
            self.output_fn(message)
        self.run_beginner_step_tabs(project_path)

    def run_beginner_step_tabs(self, project_path: str) -> None:
        applied_changes: list[AppliedChange] | None = None
        steps = build_beginner_step_tabs(project_path, applied_changes=applied_changes)
        index = 0
        while 0 <= index < len(steps):
            self.clear_fn()
            self.output_fn(format_beginner_tab(index, len(steps), steps[index]))
            if index == len(steps) - 1:
                return
            prompt = "선택 번호 > " if index in {3, 5} else "다음 > "
            raw = self.input_fn(prompt).strip()
            if self.change_mode(raw):
                self.run_current_mode()
                return
            if index == 3:
                if raw == "1":
                    index = 4
                    continue
                if raw == "2":
                    index = 0
                    continue
                if raw == "3":
                    self.output_fn("초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다.")
                    return
                if raw == "":
                    index += 1
                    continue
                self.output_fn("번호로 선택하세요. 1=수정안 미리보기, 2=프로젝트 경로 확인, 3=취소")
                continue
            if index == 5:
                if raw == "1":
                    analysis = analyze_project(project_path)
                    previews = build_fix_previews(analysis)
                    if not previews:
                        self.output_fn("적용할 수정안이 없습니다. 다음 단계로 이동합니다.")
                        index += 1
                        continue
                    applied_changes = apply_fix_previews(project_path, previews)
                    steps = build_beginner_step_tabs(project_path, applied_changes=applied_changes)
                    self.output_fn(format_beginner_apply_result(applied_changes, analyze_project(project_path)))
                    index += 1
                    continue
                if raw == "2":
                    index = 4
                    continue
                if raw == "3":
                    self.output_fn("초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다.")
                    return
                self.output_fn("번호로 선택하세요. 1=적용, 2=다시 보기, 3=취소")
                continue
            if raw in {"", "n", "next", "다음"}:
                index += 1
                continue
            if raw in {"p", "prev", "previous", "이전"}:
                index = max(0, index - 1)
                continue
            if raw in {"q", "quit", "exit", "종료", "취소"}:
                self.output_fn("초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다.")
                return
            if raw.isdigit() and 1 <= int(raw) <= len(steps):
                index = int(raw) - 1
                continue
            self.output_fn("Enter=다음, 이전, 1~10=탭 이동, 종료 중 하나를 입력하세요.")

    def run_intermediate_mode(self) -> None:
        self.output_fn(INTERMEDIATE_MENU)
        request = self.input_fn("> ").strip()
        if self.change_mode(request):
            self.run_current_mode()
            return
        self.output_fn(handle_intermediate_request(request))

    def run_advanced_mode(self) -> None:
        command = self.input_fn("ml-agent > ").strip()
        if self.change_mode(command):
            self.run_current_mode()
            return
        self.output_fn(handle_advanced_input(command))


def clear_console() -> None:
    print("\033[2J\033[H", end="")


def parse_mode(value: str) -> str | None:
    normalized = value.strip().lower()
    return MODE_ALIASES.get(normalized)


def parse_mode_command(value: str) -> str | None:
    parts = value.strip().split()
    if len(parts) != 2:
        return None
    command, mode_name = parts
    if command not in {"/mode", "/모드"}:
        return None
    return parse_mode(mode_name)


def sample_projects_root(config: AppConfig | None = None) -> Path:
    active_config = AppConfig.load() if config is None else config
    return active_config.root_dir / ".aiu" / "sample_projects"


def work_project_roots(config: AppConfig | None = None) -> list[Path]:
    active_config = AppConfig.load() if config is None else config
    candidates = [
        active_config.root_dir / "work",
        active_config.root_dir.parent / "work",
        Path.cwd() / "work",
        Path.cwd().parent / "work",
    ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            roots.append(resolved)
    return roots


def looks_like_ml_work_project(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    direct_signals = [
        path / "requirements.txt",
        path / "pyproject.toml",
        path / "train.py",
        path / "model",
        path / "models",
    ]
    if any(signal.exists() for signal in direct_signals):
        return True
    artifact_patterns = ("*.pkl", "*.joblib", "*.onnx", "*.pt", "*.pth", "*.keras", "*.h5", "*.safetensors")
    return any(next(path.glob(pattern), None) is not None for pattern in artifact_patterns)


def list_existing_work_projects(roots: list[Path] | None = None) -> list[Path]:
    active_roots = work_project_roots() if roots is None else roots
    projects: list[Path] = []
    seen: set[Path] = set()
    for root in active_roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.iterdir():
            resolved = path.resolve()
            if resolved in seen or not looks_like_ml_work_project(resolved):
                continue
            seen.add(resolved)
            projects.append(resolved)
    return sorted(projects, key=lambda path: path.name.lower())


def resolve_existing_work_project(raw: str, roots: list[Path] | None = None) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    prefixes = ("/work ", "work ", "/워크 ", "워크 ", "/model ", "model ", "/모델 ", "모델 ")
    lowered = value.lower()
    name = value
    for prefix in prefixes:
        if lowered.startswith(prefix):
            name = value[len(prefix):].strip()
            break
    if not name:
        return None
    for project in list_existing_work_projects(roots):
        if name == project.name or name.lower() == project.name.lower():
            return project
    return None


def resolve_beginner_project_input(raw: str) -> tuple[str, str | None]:
    normalized = raw.strip().lower()
    menu_command = beginner_project_menu_command(normalized)
    if menu_command:
        return resolve_beginner_project_input(menu_command)
    custom_sample = resolve_existing_sample_project(raw)
    if custom_sample:
        return (
            str(custom_sample),
            f"기존 샘플 프로젝트를 선택했습니다.\n- 위치: {custom_sample}\n- 초급자 Wizard가 이 경로로 계속 진행합니다.",
        )
    work_project = resolve_existing_work_project(raw)
    if work_project:
        return (
            str(work_project),
            f"work 디렉토리 모델 프로젝트를 선택했습니다.\n- 위치: {work_project}\n- 초급자 Wizard가 이 경로로 계속 진행합니다.",
        )
    if is_beginner_sample_run_command(normalized):
        try:
            payload = run_beginner_sample_run(raw)
        except ValueError as exc:
            return raw, f"샘플 실행 명령을 확인하세요.\n- 이유: {exc}"
        results = [result for result in payload.get("results", []) if isinstance(result, dict)]
        first_project = next((str(result.get("project")) for result in results if result.get("status") == "pass"), "")
        if not first_project and results:
            first_project = str(results[0].get("project") or raw)
        return (
            first_project or raw,
            format_beginner_sample_run_message(payload),
        )
    if normalized in {"/sample matrix", "/samples test", "/샘플 매트릭스", "sample matrix", "샘플 매트릭스"}:
        sample_paths = create_all_model_samples(sample_projects_root())
        return (
            str(sample_paths[0]),
            format_sample_matrix_message(sample_paths),
        )
    if normalized in LARGE10_SAMPLE_ALIASES:
        sample_paths = create_large_model_samples(sample_projects_root())
        return (
            str(sample_paths[0]),
            format_sample_matrix_message(
                sample_paths,
                title="대형 모델 테스트 샘플 10개를 생성하고 Step 1 검증을 완료했습니다.",
            ),
        )
    if normalized.startswith("/sample standard") or normalized.startswith("sample standard") or normalized.startswith("/샘플 표준"):
        parts = normalized.split()
        framework = parts[-1] if len(parts) > 2 else "generic"
        sample_path = create_standard_template_sample(sample_projects_root(), framework)
        return (
            str(sample_path),
            "표준 ML/DL 템플릿 샘플을 생성했습니다.\n"
            f"- 위치: {sample_path}\n"
            "- 지원 흐름: pre-trained 모델 등록 / 직접 학습 후 모델 생성\n"
            "- 초급자 Wizard가 이 경로로 계속 진행합니다.",
        )
    if normalized in {"/sample all", "/samples", "/샘플 전체", "sample all", "samples", "샘플 전체"}:
        sample_paths = create_all_model_samples(sample_projects_root())
        first_path = sample_paths[0]
        return (
            str(first_path),
            "다양한 모델 테스트 샘플을 생성했습니다.\n"
            + "\n".join(f"- {path}" for path in sample_paths)
            + "\n- 초급자 Wizard는 첫 번째 샘플 경로로 계속 진행합니다.",
        )
    sample_kind = SAMPLE_KIND_ALIASES.get(normalized)
    if not sample_kind:
        return raw, None
    spec = SAMPLE_MODEL_SPECS[sample_kind]
    sample_path = create_model_sample(sample_projects_root() / spec.directory, spec)
    sample_note = (
        "- 이 샘플은 오류/수정 흐름 재현을 위해 일부 구성이 빠져 있습니다.\n"
        if sample_kind.endswith("_error")
        else ""
    )
    return (
        str(sample_path),
        f"{spec.title} 테스트 샘플을 생성했습니다.\n"
        f"- 위치: {sample_path}\n"
        "- 실제 외부 모델 다운로드 없이 모델 artifact를 흉내냅니다.\n"
        f"{sample_note}"
        "- 초급자 Wizard가 이 경로로 계속 진행합니다.",
    )


def beginner_project_menu_command(normalized: str) -> str:
    menu = {
        "1": "/sample run --kind all",
        "2": "/sample run --kind standard --framework pytorch --mode train",
        "3": "/sample tensorflow",
        "4": "/sample pytorch",
        "5": "/sample large10",
        "6": "/sample sora-error",
        "여러모델": "/sample run --kind all",
        "여러 모델": "/sample run --kind all",
        "샘플실행": "/sample run --kind all",
        "샘플 실행": "/sample run --kind all",
        "로컬학습": "/sample run --kind standard --framework pytorch --mode train",
        "로컬 학습": "/sample run --kind standard --framework pytorch --mode train",
        "학습샘플": "/sample run --kind standard --framework pytorch --mode train",
        "학습 샘플": "/sample run --kind standard --framework pytorch --mode train",
        "텐서플로우": "/sample tensorflow",
        "파이토치": "/sample pytorch",
        "대형": "/sample large10",
        "오류샘플": "/sample sora-error",
        "오류 샘플": "/sample sora-error",
        "소라오류": "/sample sora-error",
        "소라 오류": "/sample sora-error",
    }
    return menu.get(normalized, "")


def is_beginner_sample_run_command(normalized: str) -> bool:
    return (
        normalized.startswith("/sample run")
        or normalized.startswith("sample run")
        or normalized.startswith("/샘플 실행")
        or normalized.startswith("샘플 실행")
    )


def run_beginner_sample_run(raw: str) -> dict[str, object]:
    value = raw.strip()
    lowered = value.lower()
    if lowered.startswith("/sample "):
        parts = value.split()[1:]
    elif lowered.startswith("sample "):
        parts = value.split()[1:]
    elif lowered.startswith("/샘플 실행"):
        parts = ["run", *value.split()[2:]]
    elif lowered.startswith("샘플 실행"):
        parts = ["run", *value.split()[2:]]
    else:
        parts = ["run", "--kind", "all"]
    if "--kind" not in parts:
        parts.extend(["--kind", "all"])
    return run_sample_projects_command(parts)


def format_beginner_sample_run_message(payload: dict[str, object]) -> str:
    rows = [
        "샘플 모델 실행을 완료했습니다.",
        f"- 실행 대상: {payload.get('kind')}",
        f"- 실행 방식: {payload.get('run_mode')}{' dry-run' if payload.get('dry_run') else ''}",
        f"- 결과: 총 {payload.get('count')}개 / 성공 {payload.get('pass_count')}개 / 실패 {payload.get('fail_count')}개",
        "- 실행 결과:",
    ]
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        rows.append(f"  - [{result.get('status')}] {result.get('name')} (exit={result.get('exit_code')})")
        stderr = str(result.get("stderr") or "").strip()
        if stderr:
            rows.append(f"    - 오류: {stderr.splitlines()[0]}")
    rows.append("- 초급자 Wizard는 첫 번째 성공 샘플 경로로 계속 진행합니다.")
    return "\n".join(rows)


def build_beginner_intro() -> str:
    rows = [
        "초급자 모드가 선택되었습니다.",
        "",
        "이 모드는 AI/ML 프로젝트 등록 절차를 잘 몰라도",
        "단계별 안내에 따라 프로젝트를 점검할 수 있습니다.",
        "",
        "먼저 분석할 프로젝트 경로를 입력하세요.",
        "샘플을 사용하려면 번호만 입력하세요.",
        "",
        "1. 여러 기본 샘플 모델 생성 후 실행",
        "2. 로컬 학습 가능한 표준 PyTorch 샘플 실행",
        "3. TensorFlow 샘플 생성",
        "4. PyTorch 샘플 생성",
        "5. 대형 모델 샘플 10개 생성",
        "6. Sora 오류 샘플 생성",
        "7. 내 프로젝트 경로 직접 입력",
        "",
        "경로를 알고 있다면 그대로 붙여넣어도 됩니다.",
    ]
    existing_samples = list_existing_sample_projects()
    if existing_samples:
        rows.append("")
        rows.append(".aiu/sample_projects/에 있는 기존 샘플:")
        rows.extend(f"- /sample {path.name}  ({path})" for path in existing_samples)
    work_projects = list_existing_work_projects()
    if work_projects:
        rows.append("")
        rows.append("work/에 있는 모델 프로젝트:")
        rows.extend(f"- /work {path.name}  ({path})" for path in work_projects)
        rows.append("- 이름만 입력해도 선택할 수 있습니다. 예: " + work_projects[0].name)
    return "\n".join(rows)


def list_existing_sample_projects(root: Path | None = None) -> list[Path]:
    sample_root = sample_projects_root() if root is None else root
    if not sample_root.exists() or not sample_root.is_dir():
        return []
    return sorted(
        [path for path in sample_root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def resolve_existing_sample_project(raw: str, root: Path | None = None) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    sample_root = sample_projects_root() if root is None else root
    prefixes = ("/sample ", "sample ", "/샘플 ", "샘플 ")
    lowered = value.lower()
    name = value
    for prefix in prefixes:
        if lowered.startswith(prefix):
            name = value[len(prefix):].strip()
            break
    if not name:
        return None
    candidate = (sample_root / name).resolve()
    try:
        candidate.relative_to(sample_root.resolve())
    except ValueError:
        return None
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def create_heavy_model_sample(root: Path, artifact_size_bytes: int = DEFAULT_HEAVY_SAMPLE_BYTES) -> Path:
    spec = SAMPLE_MODEL_SPECS["heavy"]
    custom_spec = SampleModelSpec(
        kind=spec.kind,
        title=spec.title,
        directory=spec.directory,
        artifact_path=spec.artifact_path,
        artifact_size_bytes=artifact_size_bytes,
        requirements=spec.requirements,
        train_body=spec.train_body,
    )
    return create_model_sample(root, custom_spec)


def create_all_model_samples(root: Path) -> list[Path]:
    return [
        create_model_sample(root / spec.directory, spec)
        for spec in SAMPLE_MODEL_SPECS.values()
        if not spec.kind.endswith("_error")
    ]


def create_large_model_samples(
    root: Path,
    artifact_size_bytes: int | None = None,
) -> list[Path]:
    sample_paths = []
    for spec in LARGE_MODEL_SAMPLE_SPECS:
        active_spec = spec
        if artifact_size_bytes is not None:
            active_spec = SampleModelSpec(
                kind=spec.kind,
                title=spec.title,
                directory=spec.directory,
                artifact_path=spec.artifact_path,
                artifact_size_bytes=artifact_size_bytes,
                requirements=spec.requirements,
                train_body=spec.train_body,
            )
        sample_paths.append(create_model_sample(root / spec.directory, active_spec))
    return sample_paths


def format_sample_matrix_message(
    sample_paths: list[Path],
    title: str = "다양한 모델 테스트 샘플을 생성하고 Step 1 검증을 완료했습니다.",
) -> str:
    rows = [title]
    for path in sample_paths:
        analysis = analyze_project(str(path))
        artifact = analysis.scan.model_artifacts[0] if analysis.scan.model_artifacts else None
        artifact_text = f"{artifact.path} ({format_bytes(artifact.size_bytes)})" if artifact else "없음"
        rows.append(
            f"- {path.name}: {analysis.registration_status}, "
            f"artifact={artifact_text}, issues={len(analysis.issues)}"
        )
    rows.append("- 초급자 Wizard는 첫 번째 샘플 경로로 계속 진행합니다.")
    return "\n".join(rows)


def create_model_sample(root: Path, spec: SampleModelSpec) -> Path:
    ensure_read_write_directory(root)
    artifact = root / spec.artifact_path
    ensure_read_write_directory(artifact.parent)
    (root / "requirements.txt").write_text("\n".join(spec.requirements) + "\n", encoding="utf-8")
    (root / "train.py").write_text(spec.train_body, encoding="utf-8")
    if not spec.kind.endswith("_error"):
        ensure_ai_studio_sample_runtime(root)
    (root / "README.md").write_text(
        f"# {spec.title} Wizard Sample\n\n"
        f"Sample kind: `{spec.kind}`\n\n"
        "This project is generated by `ml-agent` for beginner wizard testing.\n"
        "The model artifact is a local dummy file for closed-network POC validation.\n",
        encoding="utf-8",
    )
    ensure_sparse_file(artifact, spec.artifact_size_bytes)
    return root


def ensure_ai_studio_sample_runtime(root: Path) -> None:
    config_path = root / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(default_ai_studio_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ensure_standard_config_files(root)
    ensure_read_write_directory(root / "saved_model")
    env_path = root / "ai_studio.env"
    if not env_path.exists():
        env_path.write_text(ai_studio_env_source(), encoding="utf-8")
    input_example_path = root / "input_example.json"
    if not input_example_path.exists():
        input_example_path.write_text(json.dumps(default_input_example(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    custom_dir = root / "aiu_custom"
    ensure_read_write_directory(custom_dir)
    init_path = custom_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("from .predict import ModelWrapper\n", encoding="utf-8")
    predict_path = custom_dir / "predict.py"
    if not predict_path.exists():
        predict_path.write_text(ai_studio_model_wrapper_source(), encoding="utf-8")
    wrapper_path = custom_dir / "model_wrapper.py"
    if not wrapper_path.exists():
        wrapper_path.write_text("from .predict import ModelWrapper\n", encoding="utf-8")
    logging_path = root / "mlflow_ai_studio_logging.py"
    if not logging_path.exists():
        logging_path.write_text(ai_studio_logging_source(), encoding="utf-8")
    run_model_path = root / "run_model.py"
    if not run_model_path.exists():
        run_model_path.write_text(run_model_source(), encoding="utf-8")


def ensure_sparse_file(path: Path, size_bytes: int) -> None:
    if path.exists() and path.stat().st_size == size_bytes:
        return
    with path.open("wb") as file:
        if size_bytes > 0:
            file.seek(size_bytes - 1)
            file.write(b"\0")


def build_beginner_wizard(project_path: str) -> str:
    return "\n\n".join(build_beginner_step_tabs(project_path))


def build_beginner_step_tabs(
    project_path: str,
    applied_changes: list[AppliedChange] | None = None,
) -> list[str]:
    display_path = project_path or "(프로젝트 경로 미입력)"
    profile = build_ml_platform_profile(MODE_BEGINNER)
    analysis = analyze_project(project_path)
    return [
        "Step 1. 프로젝트 선택\n"
        f"- 선택된 경로: {display_path}",
        "Step 2. 프로젝트 자동 스캔\n"
        f"{format_beginner_scan(analysis.scan, profile.subagents[0].name)}",
        "Step 3. 등록 가능 여부 분석\n"
        f"{format_beginner_analysis(analysis)}",
        "Step 4. 문제 목록 확인\n"
        f"{format_beginner_issues(analysis)}",
        "Step 5. 수정안 미리보기\n"
        f"{format_beginner_fix_preview(analysis)}",
        "Step 6. 사용자 승인\n"
        f"{format_beginner_approval(analysis, profile.approval_policy)}",
        "Step 7. 파일 생성 또는 수정\n"
        f"{format_beginner_apply_step(analysis, applied_changes=applied_changes)}",
        "Step 8. 재검증\n"
        "- 적용 후 MLflow / Job Template 검증을 다시 실행합니다.",
        "Step 9. 로컬 서빙 테스트\n"
        f"{format_beginner_local_serving(analysis)}",
        "Step 10. 분석 리포트 생성\n"
        f"{format_beginner_report(analysis)}",
    ]


def format_beginner_tab(index: int, total: int, body: str) -> str:
    title, _, content = body.partition("\n")
    step_titles = [
        "프로젝트 선택",
        "자동 스캔",
        "등록 분석",
        "문제 확인",
        "수정 미리보기",
        "사용자 승인",
        "파일 적용",
        "재검증",
        "로컬 서빙",
        "리포트",
    ]
    sidebar_rows = []
    for step, label in enumerate(step_titles[:total], start=1):
        if step == index + 1:
            sidebar_rows.append(f"> Step {step:02d}. {label}  [현재]")
        else:
            sidebar_rows.append(f"  Step {step:02d}. {label}")

    if rich_console_enabled():
        return render_rich_beginner_tab(index, total, title, content or body, sidebar_rows)

    header = render_tui_header(index, total, title)
    body_panel = render_tui_body(sidebar_rows, content or body)
    footer = render_tui_footer(index)
    return (
        f"{header}\n"
        f"{body_panel}\n"
        f"{footer}"
    )


def render_rich_beginner_tab(
    index: int,
    total: int,
    title: str,
    content: str,
    sidebar_rows: list[str],
) -> str:
    width = RICH_TUI_WIDTH
    agent = tui_agent_label(index)
    request_line = (content.splitlines() or [""])[0].removeprefix("- ").strip()
    log_lines = build_terminal_log_lines(content.splitlines() or [""])
    mode_line = render_agent_switcher(index)
    active_mode = "Build" if index >= 6 else "Plan"
    inactive_mode = "Plan" if index >= 6 else "Build"
    rows = [
        rich_row("●  ●  ●   AI ML Onboarding Console | ML Platform registration workflow ...", role="window", width=width),
        rich_row(width=width),
        rich_card_line(f"# {title}", role="title", accent=True, width=width),
        rich_card_line(f"Tab {index + 1}/{total}    {agent}", role="muted", width=width),
        rich_row(width=width),
        rich_card_line(request_line, role="panel", accent=True, width=width),
        rich_row(width=width),
    ]
    for line in log_lines[:10]:
        if not line:
            rows.append(rich_row(width=width))
        elif line.startswith("*"):
            rows.append(rich_row(line, role="muted", indent=6, width=width))
        elif line.startswith("~"):
            rows.append(rich_row(line, role="normal", indent=6, width=width))
        else:
            rows.append(rich_row(line, role="normal", indent=6, width=width))
    rows.extend(
        [
            rich_row(width=width),
            rich_row(f"□  {active_mode} · qwen3.6 · AI ML Onboarding", role="normal", indent=6, width=width),
            rich_row(width=width),
            rich_input_panel_line("█", width=width),
            rich_input_panel_line("PLAN", "BUILD", "CHAT", width=width),
            rich_row(".........  esc interrupt      ctrl+space gap   tab agents   ctrl+p commands", role="status", width=width),
            rich_row(f"Current: Tab {index + 1}/{total} | {mode_line} | {title}", role="status", width=width),
        ]
    )
    return "\n".join(rows)


def render_tui_header(index: int, total: int, title: str) -> str:
    width = 112
    current = f"Current: Tab {index + 1}/{total} - {title}"
    agent = tui_agent_label(index)
    meta = f"[Tab {index + 1}] {index + 1}/{total} | {agent} | Deep Agent"
    rows = [
        "+" + "=" * width + "+",
        f"| {'AI ML Onboarding Console':<{width - len(meta) - 1}}{meta} |",
        "+" + "-" * width + "+",
        f"| {('# ' + title):<{width - 1}}|",
        f"| {current:<{width - 1}}|",
        "+" + "-" * width + "+",
    ]
    return style_tui_lines(rows, ["chrome", "chrome", "chrome", "title", "panel", "chrome"])


def render_tui_body(sidebar_rows: list[str], content: str) -> str:
    width = 112
    content_lines = content.splitlines() or [""]
    request_line = content_lines[0].removeprefix("- ").strip()
    log_lines = build_terminal_log_lines(content_lines)
    rows: list[str] = []
    roles: list[str] = []

    def add(line: str, role: str = "normal") -> None:
        rows.append(line)
        roles.append(role)

    add(f"| {'STEPS 1-10':<35}{'CURRENT PANEL':<{width - 36}}|", "title")
    add("|" + " " * width + "|")
    for item in sidebar_rows:
        add(f"| {truncate_cell(item, width - 3).ljust(width - 3)} |", "accent" if item.startswith(">") else "normal")
    add("|" + " " * width + "|")
    add(f"| {'CURRENT STEP':<{width - 1}}|", "title")
    add(f"| {'> ' + truncate_cell(request_line, width - 5):<{width - 1}}|", "accent")
    add("|" + " " * width + "|")
    for line in log_lines:
        role = "accent" if line.startswith("~") else "muted" if line.startswith("*") or line.startswith("  |") else "normal"
        add(f"| {truncate_cell(line, width - 3).ljust(width - 3)} |", role)
    add("+" + "-" * width + "+", "chrome")
    return style_tui_lines(rows, roles)


def build_terminal_log_lines(content_lines: list[str]) -> list[str]:
    rows = [
        "I'll inspect the project state in Plan mode and keep file changes behind approval.",
        "",
    ]
    for line in content_lines:
        if not line.strip():
            rows.append("")
        elif line.startswith("- "):
            rows.append(f"* {line[2:]}")
        elif line.startswith("|"):
            rows.append(f"  {line}")
        else:
            rows.append(line)
    rows.extend(
        [
            "",
            "~ Awaiting next step...",
        ]
    )
    return rows


def render_tui_footer(index: int) -> str:
    command = "선택 번호: 1=적용  2=다시 보기  3=취소" if index == 5 else "Enter=다음  이전=이전 탭  1~10=탭 이동  종료=중단"
    width = 112
    agent = tui_agent_label(index)
    mode_line = render_agent_switcher(index)
    cursor = "█" if rich_console_enabled() else ""
    input_line = "| " + cursor.ljust(width - 2) + " |"
    shortcut_line = f"| {truncate_cell(command, 48).ljust(48)} esc interrupt   ctrl+space gap   tab agents |"
    rows = [
        "+" + "-" * width + "+",
        input_line,
        f"| {mode_line:<{width - 1}}|",
        f"| {'Active agent: ' + agent + ' . Model: qwen3.6 . Workspace: AI ML Onboarding':<{width - 1}}|",
        shortcut_line,
        "+" + "=" * width + "+",
    ]
    return style_tui_lines(
        rows,
        ["chrome", "input", "accent", "status", "status", "chrome"],
    )


def tui_agent_label(index: int) -> str:
    return "Build approval" if index >= 6 else "Plan read-only"


def render_agent_switcher(index: int) -> str:
    if index >= 6:
        return "PLAN | [BUILD] | CHAT"
    return "[PLAN] | BUILD | CHAT"


def analyze_project(project_path: str) -> ProjectAnalysis:
    target = resolve_filesystem_path(project_path or ".")
    if target.exists():
        ensure_local_file_access(target)
    display_path = str(target)
    scan = scan_project(target)

    if not target.exists():
        return ProjectAnalysis(
            path=display_path,
            exists=False,
            is_directory=False,
            scan=scan,
            registration_status="불가",
            registration_checks=build_registration_checks(
                None,
                requirements_files=[],
                has_mlflow_dependency=False,
                mlflow_usage_files=[],
                entrypoint_candidates=[],
                model_artifacts=[],
                scan=scan,
            ),
            requirements_files=[],
            has_mlflow_dependency=False,
            mlflow_usage_files=[],
            entrypoint_candidates=[],
            model_artifacts=[],
            job_template_ready=False,
            local_serving=build_local_serving_plan(
                display_path,
                exists=False,
                is_directory=False,
                requirements_files=[],
                entrypoint_candidates=[],
                model_artifacts=[],
            ),
            issues=["프로젝트 경로를 찾을 수 없습니다."],
            issue_details=[
                build_project_issue(
                    "PROJECT_PATH_NOT_FOUND",
                    "blocker",
                    "프로젝트 경로 없음",
                    display_path,
                    "입력한 위치에 프로젝트 폴더가 없습니다.",
                    "올바른 프로젝트 폴더 경로를 다시 입력하세요.",
                    False,
                )
            ],
            next_actions=["올바른 프로젝트 폴더 경로를 다시 입력하세요."],
        )
    if not target.is_dir():
        return ProjectAnalysis(
            path=display_path,
            exists=True,
            is_directory=False,
            scan=scan,
            registration_status="불가",
            registration_checks=build_registration_checks(
                None,
                requirements_files=[],
                has_mlflow_dependency=False,
                mlflow_usage_files=[],
                entrypoint_candidates=[],
                model_artifacts=[],
                scan=scan,
            ),
            requirements_files=[],
            has_mlflow_dependency=False,
            mlflow_usage_files=[],
            entrypoint_candidates=[],
            model_artifacts=[],
            job_template_ready=False,
            local_serving=build_local_serving_plan(
                display_path,
                exists=True,
                is_directory=False,
                requirements_files=[],
                entrypoint_candidates=[],
                model_artifacts=[],
            ),
            issues=["선택한 경로가 폴더가 아닙니다."],
            issue_details=[
                build_project_issue(
                    "PROJECT_PATH_NOT_DIRECTORY",
                    "blocker",
                    "프로젝트 폴더가 아님",
                    display_path,
                    "선택한 경로가 파일이어서 프로젝트 구조를 스캔할 수 없습니다.",
                    "학습 코드가 들어 있는 프로젝트 폴더를 선택하세요.",
                    False,
                )
            ],
            next_actions=["학습 코드가 들어 있는 프로젝트 폴더를 선택하세요."],
        )

    requirements_files = find_requirements_files(target)
    has_mlflow_dependency = any(file_mentions(path, "mlflow") for path in requirements_files)
    python_files = list_project_files(target, {".py"}, limit=120)
    mlflow_usage_files = [relative_path(path, target) for path in python_files if file_mentions(path, "mlflow")]
    entrypoint_candidates = find_entrypoint_candidates(target, python_files)
    model_artifacts = [
        relative_path(path, target)
        for path in list_project_files(target, MODEL_ARTIFACT_SUFFIXES, limit=80)
    ]
    model_parameters = extract_model_parameters(target, scan)
    registration_checks = build_registration_checks(
        target,
        requirements_files=[relative_path(path, target) for path in requirements_files],
        has_mlflow_dependency=has_mlflow_dependency,
        mlflow_usage_files=mlflow_usage_files,
        entrypoint_candidates=entrypoint_candidates,
        model_artifacts=model_artifacts,
        scan=scan,
    )

    issues: list[str] = []
    issue_details: list[ProjectIssue] = []
    next_actions: list[str] = []
    if not requirements_files:
        issues.append("requirements.txt 또는 pyproject.toml을 찾지 못했습니다.")
        issue_details.append(
            build_project_issue(
                "DEPENDENCY_FILE_MISSING",
                "warning",
                "패키지 목록 파일 없음",
                "requirements.txt 또는 pyproject.toml",
                "플랫폼은 학습 실행 전에 어떤 Python 패키지를 설치해야 하는지 알아야 합니다.",
                "학습에 필요한 패키지 목록을 requirements.txt 또는 pyproject.toml에 정리하세요.",
                True,
            )
        )
        next_actions.append("학습에 필요한 패키지 목록을 requirements.txt 또는 pyproject.toml에 정리하세요.")
    if requirements_files and not has_mlflow_dependency:
        issues.append("패키지 목록에서 mlflow 의존성을 찾지 못했습니다.")
        issue_details.append(
            build_project_issue(
                "MLFLOW_DEPENDENCY_MISSING",
                "warning",
                "MLflow 패키지 누락",
                ", ".join(relative_path(path, target) for path in requirements_files),
                "MLflow가 패키지 목록에 없으면 실행 환경에서 실험 기록 코드가 실패할 수 있습니다.",
                "MLflow를 사용한다면 requirements.txt 또는 pyproject.toml에 mlflow를 추가하세요.",
                True,
            )
        )
        next_actions.append("MLflow를 사용한다면 requirements.txt 또는 pyproject.toml에 mlflow를 추가하세요.")
    if not entrypoint_candidates:
        issues.append("학습 시작 파일 후보를 찾지 못했습니다.")
        issue_details.append(
            build_project_issue(
                "ENTRYPOINT_MISSING",
                "blocker",
                "학습 시작 파일 없음",
                "run_model.py",
                "플랫폼 Job Template은 어떤 파일을 실행해야 하는지 알아야 합니다.",
                "AI Studio 실행 진입점은 run_model.py로 준비하세요. train.py는 직접 실행 후보에서 제외됩니다.",
                False,
            )
        )
        next_actions.append("AI Studio 실행 진입점은 run_model.py로 준비하세요. train.py는 직접 실행 후보에서 제외됩니다.")
    if not mlflow_usage_files:
        issues.append("학습 코드에서 MLflow 사용 흔적을 찾지 못했습니다.")
        issue_details.append(
            build_project_issue(
                "MLFLOW_CODE_MISSING",
                "warning",
                "MLflow 기록 코드 미확인",
                "Python 학습 코드",
                "실험 지표나 모델 artifact를 플랫폼에서 추적하려면 학습 코드 안의 MLflow 기록이 필요합니다.",
                "mlflow.start_run, mlflow.log_metric, mlflow.log_artifact 같은 기록 코드를 확인하세요.",
                True,
            )
        )
        next_actions.append("mlflow.start_run, mlflow.log_metric, mlflow.log_artifact 같은 기록 코드를 확인하세요.")
    if not model_artifacts:
        issues.append("모델 산출물 후보 파일을 찾지 못했습니다.")
        issue_details.append(
            build_project_issue(
                "MODEL_ARTIFACT_MISSING",
                "warning",
                "모델 산출물 없음",
                "model artifact",
                "등록 패키지에는 학습 결과로 만들어진 모델 파일이나 저장 경로가 필요합니다.",
                "학습 후 생성되는 모델 파일 경로를 확인하거나 샘플 artifact를 준비하세요.",
                False,
            )
        )
        next_actions.append("학습 후 생성되는 모델 파일 경로를 확인하거나 샘플 artifact를 준비하세요.")

    job_template_ready = bool(entrypoint_candidates and requirements_files)
    if not job_template_ready:
        next_actions.append("Job Template 초안 생성을 위해 entrypoint와 dependency 정보를 먼저 보완하세요.")

    if not issues:
        registration_status = "등록 가능"
        next_actions.append("dry-run 미리보기에서 Job Template과 등록 패키지 내용을 확인하세요.")
    else:
        registration_status = "보완 필요"

    return ProjectAnalysis(
        path=display_path,
        exists=True,
        is_directory=True,
        scan=scan,
        registration_status=registration_status,
        registration_checks=registration_checks,
        requirements_files=[relative_path(path, target) for path in requirements_files],
        has_mlflow_dependency=has_mlflow_dependency,
        mlflow_usage_files=mlflow_usage_files,
        entrypoint_candidates=entrypoint_candidates,
        model_artifacts=model_artifacts,
        job_template_ready=job_template_ready,
        local_serving=build_local_serving_plan(
            display_path,
            exists=True,
            is_directory=True,
            requirements_files=[relative_path(path, target) for path in requirements_files],
            entrypoint_candidates=entrypoint_candidates,
            model_artifacts=model_artifacts,
        ),
        issues=issues,
        issue_details=issue_details,
        next_actions=dedupe(next_actions),
        model_parameters=model_parameters,
    )


def safe_read_json(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def flatten_model_parameter_values(prefix: str, payload: object, output: dict[str, object]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_model_parameter_values(next_prefix, value, output)
        return
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        output[prefix] = payload


def extract_model_parameters(root: Path, scan: ProjectScan) -> dict[str, object]:
    config = safe_read_json(root / "config.json")
    parameters: dict[str, object] = {}
    for section in ("model", "training", "data"):
        value = config.get(section)
        if isinstance(value, dict):
            flatten_model_parameter_values(section, value, parameters)
    if scan.model_artifacts:
        first = scan.model_artifacts[0]
        parameters["artifact.primary_path"] = first.path
        parameters["artifact.primary_size_bytes"] = first.size_bytes
        parameters["artifact.primary_size"] = format_bytes(first.size_bytes)
        parameters["artifact.count"] = len(scan.model_artifacts)
    else:
        parameters["artifact.count"] = 0
    return parameters


def format_model_parameters(parameters: dict[str, object], limit: int = 10) -> list[str]:
    if not parameters:
        return ["모델 파라미터: 없음"]
    priority = [
        "artifact.primary_path",
        "artifact.primary_size",
        "artifact.count",
        "model.source_path",
        "model.save_path",
        "model.artifact_path",
        "training.epochs",
        "training.learning_rate",
        "training.batch_size",
        "training.optimizer",
    ]
    keys = [key for key in priority if key in parameters]
    keys.extend(key for key in sorted(parameters) if key not in keys)
    rows = []
    for key in keys[:limit]:
        rows.append(f"{key}={parameters[key]}")
    if len(keys) > limit:
        rows.append(f"외 {len(keys) - limit}개")
    return rows


def build_project_issue(
    code: str,
    severity: str,
    title: str,
    target: str,
    explanation: str,
    recommendation: str,
    fixable_by_agent: bool,
) -> ProjectIssue:
    return ProjectIssue(
        code=code,
        severity=severity,
        title=title,
        target=target,
        explanation=explanation,
        recommendation=recommendation,
        fixable_by_agent=fixable_by_agent,
    )


def scan_project(root: Path) -> ProjectScan:
    display_path = str(root)
    if not root.exists():
        return ProjectScan(
            path=display_path,
            exists=False,
            is_directory=False,
            file_count=0,
            directory_count=0,
            total_bytes=0,
            python_file_count=0,
            model_artifacts=[],
            largest_files=[],
            scan_note="프로젝트 경로를 찾을 수 없습니다.",
        )
    if not root.is_dir():
        size = file_size(root)
        file_stat = FileStat(str(root), size)
        return ProjectScan(
            path=display_path,
            exists=True,
            is_directory=False,
            file_count=1,
            directory_count=0,
            total_bytes=size,
            python_file_count=1 if root.suffix.lower() == ".py" else 0,
            model_artifacts=[],
            largest_files=[file_stat],
            scan_note="선택한 경로가 폴더가 아닙니다.",
        )

    file_count = 0
    directory_count = 0
    total_bytes = 0
    python_file_count = 0
    model_artifacts: list[FileStat] = []
    largest_files: list[FileStat] = []
    ignored_dirs = {
        ".aiu",
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "registration_packages",
        "sample_projects",
    }
    for path in root.rglob("*"):
        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue
        if path.is_dir():
            directory_count += 1
            continue
        if not path.is_file():
            continue
        size = file_size(path)
        relative = relative_path(path, root)
        stat = FileStat(relative, size)
        file_count += 1
        total_bytes += size
        if path.suffix.lower() == ".py":
            python_file_count += 1
        if path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES:
            model_artifacts.append(stat)
        largest_files.append(stat)

    largest_files = sorted(largest_files, key=lambda item: item.size_bytes, reverse=True)[:5]
    model_artifacts = sorted(model_artifacts, key=lambda item: item.size_bytes, reverse=True)[:10]
    return ProjectScan(
        path=display_path,
        exists=True,
        is_directory=True,
        file_count=file_count,
        directory_count=directory_count,
        total_bytes=total_bytes,
        python_file_count=python_file_count,
        model_artifacts=model_artifacts,
        largest_files=largest_files,
        scan_note="read-only scan 완료. 파일 내용은 필요한 텍스트 파일만 제한적으로 읽습니다.",
    )


def build_registration_checks(
    root: Path | None,
    requirements_files: list[str],
    has_mlflow_dependency: bool,
    mlflow_usage_files: list[str],
    entrypoint_candidates: list[str],
    model_artifacts: list[str],
    scan: ProjectScan,
) -> list[RegistrationCheck]:
    checks = [
        RegistrationCheck(
            code="project_path",
            label="프로젝트 경로",
            status="pass" if scan.exists and scan.is_directory else "block",
            detail="프로젝트 폴더를 찾았습니다." if scan.exists and scan.is_directory else scan.scan_note,
        ),
        RegistrationCheck(
            code="dependencies",
            label="패키지 목록",
            status="pass" if requirements_files else "warn",
            detail=format_count(requirements_files) if requirements_files else "requirements.txt 또는 pyproject.toml 없음",
        ),
        RegistrationCheck(
            code="mlflow_dependency",
            label="MLflow 의존성",
            status="pass" if has_mlflow_dependency else "warn",
            detail="패키지 목록에서 mlflow 확인" if has_mlflow_dependency else "패키지 목록에 mlflow 없음",
        ),
        RegistrationCheck(
            code="mlflow_code",
            label="MLflow 기록 코드",
            status="pass" if mlflow_usage_files else "warn",
            detail=format_count(mlflow_usage_files) if mlflow_usage_files else "학습 코드에서 mlflow 사용 흔적 없음",
        ),
        RegistrationCheck(
            code="entrypoint",
            label="학습 시작 파일",
            status="pass" if entrypoint_candidates else "block",
            detail=format_count(entrypoint_candidates) if entrypoint_candidates else "run_model.py 후보 없음",
        ),
        RegistrationCheck(
            code="model_artifact",
            label="모델 산출물",
            status="pass" if model_artifacts else "warn",
            detail=format_model_artifact_detail(model_artifacts, scan),
        ),
        RegistrationCheck(
            code="job_template",
            label="Job Template 초안",
            status="pass" if entrypoint_candidates and requirements_files else "warn",
            detail="entrypoint와 dependency 정보 확인" if entrypoint_candidates and requirements_files else "entrypoint/dependency 보완 필요",
        ),
    ]
    if root is not None and scan.exists and scan.is_directory:
        checks.extend(build_mlflow_config_checks(root, requirements_files))
    return checks


RUN_MODEL_REQUIRED_TOKENS = (
    "--env-file",
    "--config",
    "--model",
    "--mode",
    "--register",
    "AI_STUDIO_CONFIG_PATH",
    "AI_STUDIO_LOCAL_MODEL_PATH",
)
LOG_MODEL_REQUIRED_TOKENS = (
    "python_model=ModelWrapper()",
    "artifact_path",
    "code_path",
    "artifacts",
    "registered_model_name",
    "pip_requirements",
    "input_example",
)
AI_STUDIO_ENV_KEYS = (
    "MLFLOW_TRACKING_URL",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTER_MODEL_NAME",
)
SERVING_REQUIREMENT_TOKENS = ("fastapi", "uvicorn")


def build_mlflow_config_checks(root: Path, requirements_files: list[str]) -> list[RegistrationCheck]:
    checks: list[RegistrationCheck] = []
    run_model_path = root / "run_model.py"
    run_model_source_text = safe_read_text(run_model_path)
    if run_model_path.exists():
        missing = [token for token in RUN_MODEL_REQUIRED_TOKENS if token not in run_model_source_text]
        checks.append(
            RegistrationCheck(
                code="run_model_config",
                label="run_model.py 설정 변수",
                status="pass" if not missing else "warn",
                detail="등록 실행 옵션/환경변수 확인" if not missing else "누락 확인: " + ", ".join(missing),
            )
        )
    else:
        checks.append(
            RegistrationCheck(
                code="run_model_config",
                label="run_model.py 설정 변수",
                status="block",
                detail="run_model.py 없음",
            )
        )

    wrapper_path, wrapper_text = find_model_wrapper_source(root)
    if wrapper_text:
        required = ("class ModelWrapper", "predict")
        missing = [token for token in required if token not in wrapper_text]
        if "mlflow.pyfunc.PythonModel" not in wrapper_text:
            missing.append("mlflow.pyfunc.PythonModel")
        if "load_context" not in wrapper_text:
            missing.append("load_context")
        checks.append(
            RegistrationCheck(
                code="model_wrapper",
                label="ModelWrapper 구현",
                status="pass" if not missing else "warn",
                detail=f"{wrapper_path} 확인" if not missing else f"{wrapper_path} 보완 필요: " + ", ".join(missing),
            )
        )
    else:
        checks.append(
            RegistrationCheck(
                code="model_wrapper",
                label="ModelWrapper 구현",
                status="warn",
                detail="predict.py 또는 aiu_custom/model_wrapper.py에서 ModelWrapper를 찾지 못했습니다.",
            )
        )

    log_model_file, log_model_text = find_pyfunc_log_model_source(root)
    if log_model_text:
        missing = [token for token in LOG_MODEL_REQUIRED_TOKENS if token not in compact_source(log_model_text)]
        checks.append(
            RegistrationCheck(
                code="pyfunc_log_model",
                label="mlflow.pyfunc.log_model 파라미터",
                status="pass" if not missing else "warn",
                detail=f"{log_model_file} 확인" if not missing else f"{log_model_file} 누락 확인: " + ", ".join(missing),
            )
        )
    else:
        checks.append(
            RegistrationCheck(
                code="pyfunc_log_model",
                label="mlflow.pyfunc.log_model 파라미터",
                status="warn",
                detail="mlflow.pyfunc.log_model 호출을 찾지 못했습니다.",
            )
        )

    requirements_text = "\n".join(safe_read_text(root / path) for path in requirements_files)
    serving_present = all(token in requirements_text.lower() for token in SERVING_REQUIREMENT_TOKENS)
    checks.append(
        RegistrationCheck(
            code="serving_requirements",
            label="requirements.txt serve/mlflow",
            status="pass" if has_requirement_token(requirements_text, "mlflow") and serving_present else "warn",
            detail=build_serving_requirements_detail(requirements_text),
        )
    )

    env_check = inspect_ai_studio_env(root)
    checks.append(env_check)
    return checks


def compact_source(text: str) -> str:
    return "".join(text.split())


def find_model_wrapper_source(root: Path) -> tuple[str, str]:
    candidates = [root / "predict.py", root / "aiu_custom" / "predict.py", root / "aiu_custom" / "model_wrapper.py"]
    candidates.extend(path for path in list_project_files(root, {".py"}, limit=120) if path.name not in {"predict.py"})
    for path in candidates:
        text = safe_read_text(path)
        if "ModelWrapper" in text:
            return relative_path(path, root), text
    return "", ""


def find_pyfunc_log_model_source(root: Path) -> tuple[str, str]:
    preferred = root / "mlflow_ai_studio_logging.py"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(path for path in list_project_files(root, {".py"}, limit=120) if path != preferred)
    for path in candidates:
        text = safe_read_text(path)
        if "mlflow.pyfunc.log_model" in text:
            return relative_path(path, root), text
    return "", ""


def has_requirement_token(requirements_text: str, token: str) -> bool:
    normalized = requirements_text.lower().replace("_", "-")
    return token.lower().replace("_", "-") in normalized


def build_serving_requirements_detail(requirements_text: str) -> str:
    missing: list[str] = []
    if not has_requirement_token(requirements_text, "mlflow"):
        missing.append("mlflow")
    for token in SERVING_REQUIREMENT_TOKENS:
        if not has_requirement_token(requirements_text, token):
            missing.append(token)
    if not missing:
        return "mlflow, fastapi, uvicorn 확인"
    return "보완 권장: " + ", ".join(missing)


def inspect_ai_studio_env(root: Path) -> RegistrationCheck:
    env_path = root / "ai_studio.env"
    config_path = root / "config.json"
    env_text = safe_read_text(env_path)
    config_text = safe_read_text(config_path)
    combined = env_text + "\n" + config_text
    missing_keys = [key for key in AI_STUDIO_ENV_KEYS if key not in combined]
    empty_keys = [key for key in AI_STUDIO_ENV_KEYS if f"{key}=" in env_text and not env_value(env_text, key)]
    if missing_keys:
        return RegistrationCheck(
            code="mlflow_environment",
            label="기타 환경변수",
            status="warn",
            detail="키 누락: " + ", ".join(missing_keys),
        )
    if empty_keys:
        return RegistrationCheck(
            code="mlflow_environment",
            label="기타 환경변수",
            status="warn",
            detail="원격 등록 값 비어 있음: " + ", ".join(empty_keys) + " / 로컬 file:./mlruns fallback 가능",
        )
    return RegistrationCheck(
        code="mlflow_environment",
        label="기타 환경변수",
        status="pass",
        detail="MLflow 환경변수 키 확인",
    )


def env_value(env_text: str, key: str) -> str:
    for line in env_text.splitlines():
        if line.strip().startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return ""


def format_model_artifact_detail(model_artifacts: list[str], scan: ProjectScan) -> str:
    if not model_artifacts:
        return "모델 파일 후보 없음"
    artifact_sizes = {artifact.path: artifact.size_bytes for artifact in scan.model_artifacts}
    first = model_artifacts[0]
    size = artifact_sizes.get(first)
    if size is None:
        return format_count(model_artifacts)
    if len(model_artifacts) == 1:
        return f"{first} ({format_bytes(size)})"
    return f"{first} ({format_bytes(size)}) 외 {len(model_artifacts) - 1}개"


def build_local_serving_plan(
    project_path: str,
    exists: bool,
    is_directory: bool,
    requirements_files: list[str],
    entrypoint_candidates: list[str],
    model_artifacts: list[str],
) -> LocalServingPlan:
    host = "127.0.0.1"
    port = 8000
    checks: list[RegistrationCheck] = []
    notes = [
        "기본 방식은 FastAPI 호환 로컬 서버 기준입니다.",
        "실제 서버 실행 전에는 dry-run으로 포트와 입력 예시를 먼저 확인합니다.",
    ]
    if not exists:
        checks.append(RegistrationCheck("serving_project", "프로젝트 경로", "block", "프로젝트 폴더를 찾을 수 없습니다."))
    elif not is_directory:
        checks.append(RegistrationCheck("serving_project", "프로젝트 경로", "block", "폴더가 아니어서 서빙할 수 없습니다."))
    else:
        checks.append(RegistrationCheck("serving_project", "프로젝트 경로", "pass", "프로젝트 폴더 확인"))

    checks.append(
        RegistrationCheck(
            "serving_artifact",
            "모델 artifact",
            "pass" if model_artifacts else "block",
            format_count(model_artifacts) if model_artifacts else "서빙할 모델 파일 없음",
        )
    )
    checks.append(
        RegistrationCheck(
            "serving_entrypoint",
            "서빙 entrypoint",
            "pass" if entrypoint_candidates else "warn",
            format_count(entrypoint_candidates) if entrypoint_candidates else "run_model.py 확인 필요",
        )
    )
    checks.append(
        RegistrationCheck(
            "serving_dependencies",
            "서빙 패키지",
            "pass" if requirements_files else "warn",
            format_count(requirements_files) if requirements_files else "requirements.txt 확인 필요",
        )
    )
    checks.append(RegistrationCheck("serving_port", "로컬 포트", "pass", f"{host}:{port} 사용 예정"))

    if any(check.status == "block" for check in checks):
        status = "불가"
    elif any(check.status == "warn" for check in checks):
        status = "보완 필요"
    else:
        status = "준비 가능"

    return LocalServingPlan(
        status=status,
        mode="FastAPI 기본 서버",
        host=host,
        port=port,
        health_endpoint=f"http://{host}:{port}/health",
        predict_endpoint=f"http://{host}:{port}/predict",
        checks=checks,
        commands=[
            f"ml-agent serve {project_path} --dry-run",
            f"python -m uvicorn serving_app:app --host {host} --port {port}",
        ],
        notes=notes,
    )


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def build_fix_previews(analysis: ProjectAnalysis) -> list[FixPreview]:
    if analysis.registration_status == "불가":
        return []

    previews: list[FixPreview] = []
    issue_codes = {issue.code for issue in analysis.issue_details if issue.fixable_by_agent}
    if "DEPENDENCY_FILE_MISSING" in issue_codes:
        previews.append(
            FixPreview(
                code="CREATE_REQUIREMENTS",
                title="requirements.txt 생성",
                target="requirements.txt",
                action="새 패키지 목록 파일을 만듭니다.",
                preview_lines=[
                    "+ mlflow",
                    "+ scikit-learn",
                    "+ pandas",
                ],
            )
        )
    if "MLFLOW_DEPENDENCY_MISSING" in issue_codes:
        target = analysis.requirements_files[0] if analysis.requirements_files else "requirements.txt"
        previews.append(
            FixPreview(
                code="ADD_MLFLOW_DEPENDENCY",
                title="MLflow 의존성 추가",
                target=target,
                action="패키지 목록에 mlflow를 추가합니다.",
                preview_lines=["+ mlflow"],
            )
        )
    scaffold_missing = ai_studio_scaffold_missing(analysis.path)
    if "MLFLOW_CODE_MISSING" in issue_codes and not scaffold_missing:
        target = analysis.entrypoint_candidates[0] if analysis.entrypoint_candidates else "train.py"
        previews.append(
            FixPreview(
                code="ADD_MLFLOW_TRACKING_SNIPPET",
                title="MLflow 기록 코드 추가",
                target=target,
                action="학습 시작 지점 주변에 MLflow 기록 예시를 추가합니다.",
                preview_lines=[
                    "+ import mlflow",
                    "+ with mlflow.start_run():",
                    "+     mlflow.log_param('source', 'ml-agent')",
                    "+     # 기존 학습 코드 실행 후 metric/artifact 기록",
                ],
            )
        )
    if scaffold_missing:
        previews.append(
            FixPreview(
                code="CREATE_AI_STUDIO_MLFLOW_SCAFFOLD",
                title="AI Studio MLflow 등록 스캐폴드 생성",
                target=".",
                action="AI Studio 환경설정, config.json, input example, MLflow model logging 템플릿을 생성합니다.",
                preview_lines=[
                    "+ ai_studio.env",
                    "+ config.json",
                    "+ input_example.json",
                    "+ aiu_custom/model_wrapper.py",
                    "+ mlflow_ai_studio_logging.py",
                    "+ run_model.py",
                    "+ requirements.txt: mlflow, cloudpickle, pandas, numpy 확인",
                    "+ mlflow.pyfunc.log_model(... artifact_path='ai_studio' ...)",
                ],
            )
        )
    return previews


def build_approval_options(analysis: ProjectAnalysis) -> list[ApprovalOption]:
    previews = build_fix_previews(analysis)
    can_apply = bool(previews) and analysis.registration_status != "불가"
    return [
        ApprovalOption(
            key="apply",
            label="적용하기",
            description="Step 5의 미리보기 항목만 적용합니다.",
            will_modify_files=True,
            enabled=can_apply,
        ),
        ApprovalOption(
            key="review",
            label="다시 보기",
            description="파일을 수정하지 않고 Step 5 미리보기를 다시 확인합니다.",
            will_modify_files=False,
        ),
        ApprovalOption(
            key="cancel",
            label="취소하기",
            description="이번 작업을 종료하고 파일을 그대로 둡니다.",
            will_modify_files=False,
        ),
    ]


def apply_fix_previews(project_path: str, previews: list[FixPreview]) -> list[AppliedChange]:
    root = resolve_filesystem_path(project_path or ".")
    if root.exists():
        ensure_local_file_access(root)
    if not root.exists() or not root.is_dir():
        return [
            AppliedChange(
                code="APPLY_BLOCKED",
                target=str(root),
                status="skipped",
                message="프로젝트 폴더가 없어 적용하지 않았습니다.",
            )
        ]

    changes: list[AppliedChange] = []
    for preview in previews:
        target = safe_project_path(root, preview.target)
        if target is None:
            changes.append(
                AppliedChange(
                    code=preview.code,
                    target=preview.target,
                    status="skipped",
                    message="프로젝트 폴더 밖의 경로라 적용하지 않았습니다.",
                )
            )
            continue
        if preview.code == "CREATE_REQUIREMENTS":
            changes.append(apply_create_requirements(target))
        elif preview.code == "ADD_MLFLOW_DEPENDENCY":
            changes.append(apply_add_mlflow_dependency(target))
        elif preview.code == "ADD_MLFLOW_TRACKING_SNIPPET":
            changes.append(apply_add_mlflow_tracking_snippet(target))
        elif preview.code == "CREATE_AI_STUDIO_MLFLOW_SCAFFOLD":
            changes.append(apply_create_ai_studio_mlflow_scaffold(root))
        else:
            changes.append(
                AppliedChange(
                    code=preview.code,
                    target=preview.target,
                    status="skipped",
                    message="지원하지 않는 수정안이라 적용하지 않았습니다.",
                )
            )
    return changes


def ai_studio_scaffold_missing(project_path: str) -> bool:
    root = resolve_filesystem_path(project_path or ".")
    if not root.exists() or not root.is_dir():
        return False
    required_paths = [
        root / "config.json",
        root / "ai_studio.env",
        root / "input_example.json",
        root / "aiu_custom" / "model_wrapper.py",
        root / "mlflow_ai_studio_logging.py",
        root / "run_model.py",
    ]
    return any(not path.exists() for path in required_paths)


def safe_project_path(root: Path, relative: str) -> Path | None:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def apply_create_requirements(target: Path) -> AppliedChange:
    if target.exists():
        return AppliedChange(
            code="CREATE_REQUIREMENTS",
            target=str(target),
            status="skipped",
            message="requirements.txt가 이미 있어 새로 만들지 않았습니다.",
        )
    target.write_text("mlflow\nscikit-learn\npandas\n", encoding="utf-8")
    return AppliedChange(
        code="CREATE_REQUIREMENTS",
        target=str(target),
        status="applied",
        message="requirements.txt를 생성했습니다.",
    )


def apply_add_mlflow_dependency(target: Path) -> AppliedChange:
    content = safe_read_text(target)
    if "mlflow" in content.lower():
        return AppliedChange(
            code="ADD_MLFLOW_DEPENDENCY",
            target=str(target),
            status="skipped",
            message="mlflow가 이미 포함되어 있어 변경하지 않았습니다.",
        )
    separator = "" if not content or content.endswith("\n") else "\n"
    target.write_text(f"{content}{separator}mlflow\n", encoding="utf-8")
    return AppliedChange(
        code="ADD_MLFLOW_DEPENDENCY",
        target=str(target),
        status="applied",
        message="패키지 목록에 mlflow를 추가했습니다.",
    )


def apply_add_mlflow_tracking_snippet(target: Path) -> AppliedChange:
    content = safe_read_text(target)
    if "mlflow" in content.lower():
        return AppliedChange(
            code="ADD_MLFLOW_TRACKING_SNIPPET",
            target=str(target),
            status="skipped",
            message="MLflow 코드가 이미 있어 변경하지 않았습니다.",
        )
    lines = content.splitlines()
    if lines and lines[0].startswith("#!"):
        lines.insert(1, "import mlflow")
    else:
        lines.insert(0, "import mlflow")
    lines.extend(
        [
            "",
            "# ml-agent: MLflow tracking template. Review before production use.",
            "# with mlflow.start_run():",
            "#     mlflow.log_param('source', 'ml-agent')",
            "#     mlflow.log_metric('example_metric', 0.0)",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return AppliedChange(
        code="ADD_MLFLOW_TRACKING_SNIPPET",
        target=str(target),
        status="applied",
        message="학습 코드에 MLflow 기록 템플릿을 추가했습니다.",
    )


AI_STUDIO_REQUIREMENTS = ["mlflow", "cloudpickle", "pandas", "numpy", "scikit-learn", "joblib"]

STANDARD_TEMPLATE_FRAMEWORKS = {
    "generic": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy"],
        "artifact": "saved_model/local_model.bin",
    },
    "tensorflow": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "tensorflow"],
        "artifact": "saved_model/tensorflow_model.keras",
    },
    "pytorch": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "torch"],
        "artifact": "saved_model/pytorch_model.pt",
    },
    "sklearn": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "scikit-learn", "joblib"],
        "artifact": "saved_model/sklearn_model.joblib",
    },
    "xgboost": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "xgboost"],
        "artifact": "saved_model/xgboost_model.json",
    },
    "onnx": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "onnx", "onnxruntime"],
        "artifact": "saved_model/onnx_model.onnx",
    },
    "huggingface": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "transformers", "safetensors"],
        "artifact": "saved_model/huggingface_model.safetensors",
    },
    "sora": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "torch", "opencv-python"],
        "artifact": "saved_model/sora_style_video_model.onnx",
    },
    "llm": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "transformers", "safetensors"],
        "artifact": "saved_model/llm_adapter.safetensors",
    },
    "vision": {
        "requirements": ["mlflow", "cloudpickle", "pandas", "numpy", "torch", "opencv-python"],
        "artifact": "saved_model/vision_embedding.pt",
    },
}

STANDARD_TEMPLATE_ALIASES = {
    "tf": "tensorflow",
    "텐서플로우": "tensorflow",
    "torch": "pytorch",
    "파이토치": "pytorch",
    "사이킷런": "sklearn",
    "scikit-learn": "sklearn",
    "소라": "sora",
    "hf": "huggingface",
    "llm_adapter": "llm",
    "비전": "vision",
}


def normalize_standard_framework(value: str | None) -> str:
    normalized = (value or "generic").strip().lower()
    normalized = STANDARD_TEMPLATE_ALIASES.get(normalized, normalized)
    if normalized in STANDARD_TEMPLATE_FRAMEWORKS:
        return normalized
    return "generic"


def standard_model_config(framework: str = "generic") -> dict[str, object]:
    framework = normalize_standard_framework(framework)
    spec = STANDARD_TEMPLATE_FRAMEWORKS[framework]
    return {
        "framework": framework,
        "model_flow": "pretrained_or_train",
        "pretrained": {
            "description": "Use this when the user already has a model artifact.",
            "source_path": "",
        },
        "direct_train": {
            "description": "Use this when train.py creates the model artifact locally.",
            "entrypoint": "train.py",
            "output_path": spec["artifact"],
        },
        "save_path": "saved_model",
        "artifact_path": "ai_studio",
        "code_path": ["aiu_custom"],
        "requirements": "requirements.txt",
    }


def standard_train_config(framework: str = "generic") -> dict[str, object]:
    return {
        "framework": normalize_standard_framework(framework),
        "data": {
            "train_path": "data/train",
            "test_path": "data/test",
            "input_example_path": "input_example.json",
        },
        "params": {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "SGD",
            "loss_function": "LossFunction",
            "metric_function": "MetricFunction",
        },
    }


def standard_mlflow_config() -> dict[str, object]:
    return {
        "tracking_url": "${MLFLOW_TRACKING_URL}",
        "tracking_username": "${MLFLOW_TRACKING_USERNAME}",
        "tracking_password": "${MLFLOW_TRACKING_PASSWORD}",
        "experiment_name": "${MLFLOW_EXPERIMENT_NAME}",
        "registered_model_name": "${MLFLOW_REGISTER_MODEL_NAME}",
        "local_default_tracking_uri": "file:./mlruns",
    }


def ensure_standard_config_files(root: Path, framework: str = "generic") -> list[str]:
    created: list[str] = []
    config_dir = root / "config"
    ensure_read_write_directory(config_dir)
    files = {
        "config/model_config.json": standard_model_config(framework),
        "config/train_config.json": standard_train_config(framework),
        "config/mlflow_config.json": standard_mlflow_config(),
    }
    for relative, payload in files.items():
        path = root / relative
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            created.append(relative)
    return created


def standard_train_source(framework: str = "generic") -> str:
    framework = normalize_standard_framework(framework)
    artifact = STANDARD_TEMPLATE_FRAMEWORKS[framework]["artifact"]
    return f'''"""Standard AIU training entrypoint.

This file is safe for local POC testing. Replace the dummy training logic with
your TensorFlow/PyTorch/scikit-learn/etc. code when moving to a real model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    framework = os.getenv("AIU_FRAMEWORK", "{framework}")
    output_dir = ensure_dir(Path(os.getenv("AIU_TRAIN_OUTPUT_DIR", "saved_model")))
    artifact_path = Path(os.getenv("AIU_TRAIN_ARTIFACT_PATH", "{artifact}"))
    if not artifact_path.is_absolute():
        artifact_path = Path.cwd() / artifact_path
    ensure_dir(artifact_path.parent)
    artifact_path.write_text(
        "AIU dummy model artifact\\n"
        f"framework={{framework}}\\n",
        encoding="utf-8",
    )
    summary = {{
        "framework": framework,
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "mode": "direct_train",
    }}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"trained model artifact: {{artifact_path}}")


if __name__ == "__main__":
    main()
'''


def ensure_standard_ml_dl_template(root: Path, framework: str = "generic") -> AppliedChange:
    framework = normalize_standard_framework(framework)
    created: list[str] = []
    updated: list[str] = []
    ensure_read_write_directory(root)
    ensure_read_write_directory(root / "aiu_custom")
    ensure_read_write_directory(root / "saved_model")
    created.extend(ensure_standard_config_files(root, framework))

    files = {
        "config.json": json.dumps(default_ai_studio_config(framework), indent=2, ensure_ascii=False) + "\n",
        "ai_studio.env": ai_studio_env_source(),
        "input_example.json": json.dumps(default_input_example(), indent=2, ensure_ascii=False) + "\n",
        "aiu_custom/__init__.py": "from .predict import ModelWrapper\n",
        "aiu_custom/predict.py": ai_studio_model_wrapper_source(),
        "aiu_custom/model_wrapper.py": "from .predict import ModelWrapper\n",
        "mlflow_ai_studio_logging.py": ai_studio_logging_source(),
        "run_model.py": run_model_source(),
        "train.py": standard_train_source(framework),
    }
    for relative, content in files.items():
        path = root / relative
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(relative)
        elif relative == "run_model.py" and is_managed_run_model(path) and "--mode" not in safe_read_text(path):
            path.write_text(content, encoding="utf-8")
            updated.append(relative)

    requirements = list(STANDARD_TEMPLATE_FRAMEWORKS[framework]["requirements"])
    if ensure_requirement_lines(root / "requirements.txt", requirements):
        updated.append("requirements.txt")

    if not created and not updated:
        return AppliedChange(
            code="CREATE_STANDARD_ML_DL_TEMPLATE",
            target=str(root),
            status="skipped",
            message=f"{framework} 표준 템플릿이 이미 준비되어 있습니다.",
        )
    parts = []
    if created:
        parts.append("생성: " + ", ".join(created))
    if updated:
        parts.append("수정: " + ", ".join(updated))
    return AppliedChange(
        code="CREATE_STANDARD_ML_DL_TEMPLATE",
        target=str(root),
        status="applied",
        message=f"{framework} 표준 ML/DL 템플릿을 준비했습니다. " + " / ".join(parts),
    )


def create_standard_template_sample(root: Path, framework: str = "generic") -> Path:
    framework = normalize_standard_framework(framework)
    target = root / f"standard-{framework}-template"
    ensure_standard_ml_dl_template(target, framework)
    (target / "README.md").write_text(
        f"# AIU Standard {framework} Template\n\n"
        "This sample supports both flows:\n\n"
        "- `python run_model.py --mode pretrained --model <artifact> --prepare-only`\n"
        "- `python run_model.py --mode train --prepare-only`\n"
        "- `python run_model.py --mode train --register --dry-run`\n",
        encoding="utf-8",
    )
    return target


def apply_create_ai_studio_mlflow_scaffold(root: Path) -> AppliedChange:
    created: list[str] = []
    updated: list[str] = []

    config_path = root / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(default_ai_studio_config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created.append("config.json")
    for item in ensure_standard_config_files(root):
        created.append(item)
    ensure_read_write_directory(root / "saved_model")

    env_path = root / "ai_studio.env"
    if not env_path.exists():
        env_path.write_text(ai_studio_env_source(), encoding="utf-8")
        created.append("ai_studio.env")

    input_example_path = root / "input_example.json"
    if not input_example_path.exists():
        input_example_path.write_text(json.dumps(default_input_example(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        created.append("input_example.json")

    custom_dir = root / "aiu_custom"
    ensure_read_write_directory(custom_dir)
    init_path = custom_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("from .predict import ModelWrapper\n", encoding="utf-8")
        created.append("aiu_custom/__init__.py")

    predict_path = custom_dir / "predict.py"
    if not predict_path.exists():
        predict_path.write_text(ai_studio_model_wrapper_source(), encoding="utf-8")
        created.append("aiu_custom/predict.py")

    wrapper_path = custom_dir / "model_wrapper.py"
    if not wrapper_path.exists():
        wrapper_path.write_text("from .predict import ModelWrapper\n", encoding="utf-8")
        created.append("aiu_custom/model_wrapper.py")

    logging_path = root / "mlflow_ai_studio_logging.py"
    if not logging_path.exists():
        logging_path.write_text(ai_studio_logging_source(), encoding="utf-8")
        created.append("mlflow_ai_studio_logging.py")
    elif is_managed_ai_studio_logging(logging_path) and "AI_STUDIO_LOCAL_MODEL_PATH" not in safe_read_text(logging_path):
        logging_path.write_text(ai_studio_logging_source(), encoding="utf-8")
        updated.append("mlflow_ai_studio_logging.py")

    run_model_path = root / "run_model.py"
    if not run_model_path.exists():
        run_model_path.write_text(run_model_source(), encoding="utf-8")
        created.append("run_model.py")
    elif is_managed_run_model(run_model_path) and "prepare_local_model" not in safe_read_text(run_model_path):
        run_model_path.write_text(run_model_source(), encoding="utf-8")
        updated.append("run_model.py")

    requirements_path = root / "requirements.txt"
    changed_requirements = ensure_requirement_lines(requirements_path, AI_STUDIO_REQUIREMENTS)
    if changed_requirements:
        updated.append("requirements.txt")

    if not created and not updated:
        return AppliedChange(
            code="CREATE_AI_STUDIO_MLFLOW_SCAFFOLD",
            target=str(root),
            status="skipped",
            message="AI Studio MLflow 스캐폴드가 이미 있어 변경하지 않았습니다.",
        )
    parts = []
    if created:
        parts.append("생성: " + ", ".join(created))
    if updated:
        parts.append("수정: " + ", ".join(updated))
    return AppliedChange(
        code="CREATE_AI_STUDIO_MLFLOW_SCAFFOLD",
        target=str(root),
        status="applied",
        message="AI Studio MLflow 등록 스캐폴드를 적용했습니다. " + " / ".join(parts),
    )


def ensure_requirement_lines(path: Path, requirements: list[str]) -> bool:
    content = safe_read_text(path) if path.exists() else ""
    existing = {line.strip().split("==", 1)[0].split(">=", 1)[0].lower() for line in content.splitlines() if line.strip()}
    missing = [requirement for requirement in requirements if requirement.lower() not in existing]
    if not missing:
        return False
    separator = "" if not content or content.endswith("\n") else "\n"
    path.write_text(f"{content}{separator}" + "\n".join(missing) + "\n", encoding="utf-8")
    return True


def is_managed_run_model(path: Path) -> bool:
    content = safe_read_text(path)
    return "Run AI Studio MLflow model logging with an environment file." in content


def is_managed_ai_studio_logging(path: Path) -> bool:
    content = safe_read_text(path)
    return "AI Studio MLflow logging template." in content


def default_ai_studio_config(framework: str = "generic") -> dict[str, object]:
    framework = normalize_standard_framework(framework)
    return {
        "template": {
            "version": "standard-v1",
            "framework": framework,
            "model_flow": "pretrained_or_train",
        },
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
            "config_path": "config/model_config.json",
        },
        "execution": {
            "entrypoint": "run_model.py",
            "blocked_entrypoints": ["train.py"],
            "supported_modes": ["pretrained", "train"],
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "SGD",
        },
    }


def ai_studio_env_source() -> str:
    return """# AI Studio MLflow environment.
# Fill these values before running: python run_model.py --env-file ai_studio.env
MLFLOW_TRACKING_URL=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
MLFLOW_EXPERIMENT_NAME=ai-studio-onboarding
MLFLOW_REGISTER_MODEL_NAME=ai-studio-model
"""


def default_input_example() -> dict[str, object]:
    return {
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[0.1, 0.2, 0.3]],
    }


def ai_studio_model_wrapper_source() -> str:
    return '''"""AI Studio MLflow pyfunc model wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow.pyfunc
import pandas as pd


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Load a joblib/sklearn model artifact and serve predictions."""

    def load_context(self, context):
        self.model_path = Path(context.artifacts["model"])
        self.config_path = Path(context.artifacts["config"])
        self.model = joblib.load(self.model_path)
        self.config = {}
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        return self.model.predict(model_input)
'''


def ai_studio_logging_source() -> str:
    return '''"""AI Studio MLflow logging template.

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
        + "\\n",
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
'''


def run_model_source() -> str:
    return '''"""Run AI Studio MLflow model logging with an environment file.

This runner supports two safe onboarding flows:
- pretrained: copy an existing model artifact into saved_model/
- train: run train.py locally, then use the generated artifact
- sample: when no artifact is provided, train a local sklearn diabetes model

Usage:
  python run_model.py
  python run_model.py --mode train --prepare-only
  python run_model.py --model ./model/my-model.onnx
  python run_model.py --env-file ai_studio.env --config config.json --register
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
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
LOCAL_MLFLOW_TRACKING_URI = "file:./mlruns"
logging.getLogger("mlflow").setLevel(logging.ERROR)


def configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass


def project_root() -> Path:
    return Path(__file__).resolve().parent


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


def ensure_local_mlflow_store() -> Path:
    mlruns = ensure_read_write_directory(project_root() / "mlruns")
    tracking_url = os.environ.get("MLFLOW_TRACKING_URL", "").strip()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if tracking_url:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_url
    elif not tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = LOCAL_MLFLOW_TRACKING_URI
    return mlruns


def configure_mlflow_environment() -> None:
    tracking_url = os.environ.get("MLFLOW_TRACKING_URL", "").strip()
    tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME", "").strip()
    tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "").strip()
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "TRUE"
    if tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
    if tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password
    if tracking_url:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_url
    else:
        os.environ.setdefault("MLFLOW_TRACKING_URI", LOCAL_MLFLOW_TRACKING_URI)


def compute_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        from sklearn.metrics import root_mean_squared_error

        rmse = root_mean_squared_error(actual, predicted)
    except Exception:
        rmse = mean_squared_error(actual, predicted, squared=False)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


def train_sklearn_diabetes_model(config: dict):
    import joblib
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import train_test_split

    diabetes = load_diabetes(as_frame=True)
    diabetes_df = diabetes.frame
    train_df, test_df = train_test_split(diabetes_df, test_size=0.2, random_state=42)
    train_x = train_df.drop(["target"], axis=1)
    train_y = train_df["target"]
    test_x = test_df.drop(["target"], axis=1)
    test_y = test_df["target"]

    model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
    model.fit(train_x, train_y.values.ravel())
    prediction = model.predict(test_x)
    rmse, mae, r2 = compute_metrics(test_y, prediction)

    model_dir = ensure_read_write_directory(Path(str(config.get("model", {}).get("save_path", "saved_model"))))
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    return model_path.resolve(), train_df, test_df, test_x, {"rmse": rmse, "mae": mae, "r2": r2}


def write_input_example(test_x, batch_size: int = 10) -> dict:
    sample_data = test_x.head(batch_size).to_numpy()
    input_example = {
        "input": [
            {
                "name": "diabetes_example",
                "shape": list(sample_data.shape),
                "datatype": type(sample_data).__name__,
                "data": sample_data.tolist(),
            }
        ]
    }
    Path("input_example.json").write_text(json.dumps(input_example, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    return input_example


def default_input_example() -> dict:
    input_example = {
        "input": [
            {
                "name": "sample_input",
                "shape": [1, 1],
                "datatype": "ndarray",
                "data": [[0.0]],
            }
        ]
    }
    Path("input_example.json").write_text(json.dumps(input_example, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    return input_example


def input_example_to_dataframe(input_example: dict):
    import pandas as pd

    payload = input_example.get("input", [{}])[0]
    data = payload.get("data") or [[0.0]]
    return pd.DataFrame(data)


def write_training_config(params: dict) -> Path:
    config_dir = ensure_read_write_directory(Path("config"))
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(params, indent=4, ensure_ascii=False) + "\\n", encoding="utf-8")
    return config_path


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

    root = project_root()
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


def try_resolve_model_source(config: dict, explicit_model: str = "") -> Path | None:
    try:
        return resolve_model_source(config, explicit_model)
    except FileNotFoundError:
        return None


def unique_target_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    for index in range(2, 1000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"저장 대상 경로를 만들 수 없습니다: {target}")


def prepare_local_model(source: Path, config: dict) -> Path:
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    save_root = Path(str(config_model.get("save_path") or "saved_model"))
    ensure_read_write_directory(save_root)
    if source.is_dir():
        target = unique_target_path(save_root / "local_model")
        shutil.copytree(source, target)
        return target.resolve()

    suffix = source.suffix or ".bin"
    target = unique_target_path(save_root / f"local_model{suffix}")
    shutil.copy2(source, target)
    return target.resolve()


def resolve_train_output(config: dict, framework: str = "") -> Path:
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    save_root = Path(str(config_model.get("save_path") or "saved_model"))
    model_config_path = Path(str(config_model.get("config_path") or "config/model_config.json"))
    output_path = ""
    if model_config_path.exists():
        model_config = load_config(model_config_path)
        direct_train = model_config.get("direct_train", {}) if isinstance(model_config.get("direct_train", {}), dict) else {}
        output_path = str(direct_train.get("output_path") or "")
    if not output_path:
        safe_framework = framework or "generic"
        output_path = str(save_root / f"{safe_framework}_trained_model.bin")
    output = Path(output_path)
    if not output.is_absolute():
        output = project_root() / output
    return output


def run_training(config: dict, framework: str = "") -> Path:
    train_path = Path("train.py")
    if not train_path.exists():
        raise FileNotFoundError("train.py가 없습니다. 직접 학습 모드는 train.py가 필요합니다.")
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    save_root = Path(str(config_model.get("save_path") or "saved_model"))
    ensure_read_write_directory(save_root)
    output_path = resolve_train_output(config, framework)
    ensure_read_write_directory(output_path.parent)
    env = os.environ.copy()
    env["AIU_TRAIN_OUTPUT_DIR"] = str(save_root)
    env["AIU_FRAMEWORK"] = framework or env.get("AIU_FRAMEWORK", "generic")
    env["AIU_TRAIN_ARTIFACT_PATH"] = str(output_path)
    result = subprocess.run(
        [sys.executable, str(train_path)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"train.py 실행 실패: exit={result.returncode}")
    if output_path.exists():
        return output_path.resolve()
    model_files = [
        path
        for path in save_root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    if model_files:
        return sorted(model_files, key=lambda item: item.stat().st_mtime, reverse=True)[0].resolve()
    raise FileNotFoundError(f"학습 산출물을 찾지 못했습니다: {output_path}")


def import_model_wrapper():
    try:
        from aiu_custom.predict import ModelWrapper
    except Exception:
        from aiu_custom import ModelWrapper
    return ModelWrapper


def resolve_experiment_name(config: dict) -> str:
    return (
        os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or str(config.get("mlflow_experiment_name") or "").strip()
        or "ai-studio-onboarding"
    )


def resolve_registered_model_name(config: dict) -> str:
    return (
        os.environ.get("MLFLOW_REGISTER_MODEL_NAME")
        or str(config.get("mlflow_register_model_name") or "").strip()
        or "ai-studio-model"
    )


def run_mlflow_registration(
    model_path: Path,
    config_path: Path,
    config: dict,
    train_df=None,
    test_df=None,
    test_x=None,
    metrics: dict | None = None,
) -> None:
    import mlflow

    ModelWrapper = import_model_wrapper()
    configure_mlflow_environment()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", LOCAL_MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = resolve_experiment_name(config)
    registered_model_name = resolve_registered_model_name(config)
    mlflow.set_experiment(experiment_name)

    params = {"alpha": 0.001, "l1_ratio": 0.5, "random_state": 42}
    generated_config_path = write_training_config(params)
    if test_x is not None:
        input_example = write_input_example(test_x)
        model_input_example = test_x.head(10)
    else:
        input_example = default_input_example()
        model_input_example = input_example_to_dataframe(input_example)

    with mlflow.start_run() as run:
        if train_df is not None and hasattr(mlflow, "data"):
            train_dataset = mlflow.data.from_pandas(train_df, name="Train", targets="target")
            mlflow.log_input(train_dataset, context="training")
        if test_df is not None and hasattr(mlflow, "data"):
            test_dataset = mlflow.data.from_pandas(test_df, name="Test", targets="target")
            mlflow.log_input(test_dataset, context="test")
        if hasattr(mlflow, "set_tag"):
            mlflow.set_tag("data.name", "diabetes(sklearn)" if train_df is not None else "local-artifact")
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics=metrics)
        if hasattr(mlflow, "log_artifact"):
            mlflow.log_artifact(str(generated_config_path))

        mlflow.pyfunc.log_model(
            python_model=ModelWrapper(),
            artifact_path=str(config.get("model", {}).get("artifact_path") or "ai_studio"),
            code_path=["aiu_custom"],
            artifacts={
                "model": str(model_path).replace("\\\\", "/"),
                "config": str(generated_config_path).replace("\\\\", "/"),
            },
            input_example=model_input_example,
            registered_model_name=registered_model_name,
            pip_requirements="requirements.txt",
        )

    run_id = getattr(getattr(run, "info", None), "run_id", "")
    print(f"mlflow run created: {run_id or 'created'}")
    print(f"mlflow tracking uri: {tracking_uri}")
    print(f"registered model name: {registered_model_name}")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Prepare local model and run AI Studio MLflow logging.")
    parser.add_argument("--mode", choices=["pretrained", "train"], default="pretrained", help="Model flow to run")
    parser.add_argument("--framework", default="", help="Framework label for train mode")
    parser.add_argument("--env-file", default="ai_studio.env", help="AI Studio environment file path")
    parser.add_argument("--config", default="config.json", help="AI Studio config file path")
    parser.add_argument("--model", default="", help="Existing local model file or directory path")
    parser.add_argument("--prepare-only", action="store_true", help="Only create saved_model/local_model and skip MLflow logging")
    parser.add_argument("--register", action="store_true", help="Run MLflow model logging/register after local model preparation")
    parser.add_argument("--dry-run", action="store_true", help="Show the MLflow register command path without importing mlflow")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    local_mlflow_store = ensure_local_mlflow_store()
    config_path = Path(args.config)
    config = load_config(config_path)
    os.environ["AI_STUDIO_CONFIG_PATH"] = str(config_path)
    train_df = None
    test_df = None
    test_x = None
    metrics = None

    if args.mode == "train":
        source_model = run_training(config, args.framework)
    else:
        source_model = try_resolve_model_source(config, args.model)
        if source_model is None:
            print("local model artifact not found; creating sklearn diabetes sample model.")
            source_model, train_df, test_df, test_x, metrics = train_sklearn_diabetes_model(config)
    local_model = prepare_local_model(source_model, config)
    os.environ["AI_STUDIO_LOCAL_MODEL_PATH"] = str(local_model)
    print(f"local model prepared: {local_model}")

    if args.prepare_only:
        return
    if args.dry_run:
        print("dry-run register command: python run_model.py --env-file ai_studio.env --register")
        print(f"mlflow tracking default: {LOCAL_MLFLOW_TRACKING_URI} when MLFLOW_TRACKING_URL is empty")
        print(f"local mlruns directory: {local_mlflow_store}")
        return

    run_mlflow_registration(local_model, config_path, config, train_df, test_df, test_x, metrics)


if __name__ == "__main__":
    main()
'''


def find_requirements_files(root: Path) -> list[Path]:
    candidates = [root / "requirements.txt", root / "pyproject.toml"]
    return [path for path in candidates if path.exists() and path.is_file()]


def list_project_files(root: Path, suffixes: set[str], limit: int) -> list[Path]:
    results: list[Path] = []
    ignored_dirs = {".git", ".venv", "__pycache__", "node_modules", "registration_packages"}
    for path in root.rglob("*"):
        if len(results) >= limit:
            break
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            results.append(path)
    return results


def find_entrypoint_candidates(root: Path, python_files: list[Path]) -> list[str]:
    run_model = root / "run_model.py"
    if run_model.exists():
        return ["run_model.py"]
    return []


def file_mentions(path: Path, token: str) -> bool:
    return token.lower() in safe_read_text(path).lower()


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def format_beginner_scan(scan: ProjectScan, scanner_name: str) -> str:
    if not scan.exists:
        return (
            f"- {scanner_name}가 read-only scan을 시도했습니다.\n"
            "- 결과: 프로젝트 경로를 찾을 수 없습니다.\n"
            "- 파일은 수정하지 않았습니다."
        )
    if not scan.is_directory:
        return (
            f"- {scanner_name}가 read-only scan을 시도했습니다.\n"
            "- 결과: 선택한 경로가 폴더가 아닙니다.\n"
            f"- 크기: {format_bytes(scan.total_bytes)}\n"
            "- 파일은 수정하지 않았습니다."
        )
    rows = [
        f"- {scanner_name}가 read-only scan을 완료했습니다.",
        "- 파일은 수정하지 않았습니다.",
        f"- 파일 수: {scan.file_count}개",
        f"- 폴더 수: {scan.directory_count}개",
        f"- 전체 크기: {format_bytes(scan.total_bytes)}",
        f"- Python 파일: {scan.python_file_count}개",
        f"- 모델 artifact 후보: {len(scan.model_artifacts)}개",
    ]
    if scan.model_artifacts:
        rows.append("- 모델 artifact:")
        rows.extend(f"  - {item.path} ({format_bytes(item.size_bytes)})" for item in scan.model_artifacts[:3])
    if scan.largest_files:
        rows.append("- 큰 파일 상위:")
        rows.extend(f"  - {item.path} ({format_bytes(item.size_bytes)})" for item in scan.largest_files[:3])
    return "\n".join(rows)


def format_beginner_analysis(analysis: ProjectAnalysis) -> str:
    rows = [
        f"- 등록 상태: {analysis.registration_status}",
        "- 판정 기준:",
    ]
    for check in analysis.registration_checks:
        rows.append(f"  - {format_check_status(check.status)} {check.label}: {check.detail}")
    blockers = [check for check in analysis.registration_checks if check.status == "block"]
    warnings = [check for check in analysis.registration_checks if check.status == "warn"]
    rows.extend(
        [
            f"- 차단 항목: {len(blockers)}개",
            f"- 보완 권장 항목: {len(warnings)}개",
            f"- Job Template 초안 준비: {'가능' if analysis.job_template_ready else '보완 필요'}",
        ]
    )
    if analysis.model_parameters:
        rows.append("- 모델 파라미터:")
        rows.extend(f"  - {item}" for item in format_model_parameters(analysis.model_parameters, limit=8))
    return "\n".join(rows)


def format_beginner_issues(analysis: ProjectAnalysis) -> str:
    if not analysis.issue_details:
        return (
            "- 문제 수: 0개\n"
            "- 큰 문제를 찾지 못했습니다.\n"
            "- 다음 단계에서 수정 없이 미리보기를 확인할 수 있습니다."
        )
    blocker_count = len([issue for issue in analysis.issue_details if issue.severity == "blocker"])
    warning_count = len([issue for issue in analysis.issue_details if issue.severity == "warning"])
    fixable_count = len([issue for issue in analysis.issue_details if issue.fixable_by_agent])
    issue_rows = [
        f"- 문제 수: {len(analysis.issue_details)}개",
        f"- 필수 확인: {blocker_count}개",
        f"- 보완 권장: {warning_count}개",
        f"- Agent 수정 가능: {fixable_count}개",
        "- 주요 문제:",
    ]
    issue_rows.extend(format_issue_table(analysis.issue_details[:3]))
    if len(analysis.issue_details) > 3:
        issue_rows.append(f"- 나머지 문제 {len(analysis.issue_details) - 3}개는 리포트에 자세히 남깁니다.")
    issue_rows.extend(
        [
            "- 다음 선택:",
            "  1. 수정안 미리보기로 이동",
            "  2. 프로젝트 경로를 다시 확인",
            "  3. 취소",
        ]
    )
    return "\n".join(issue_rows)


def format_issue_table(issues: list[ProjectIssue]) -> list[str]:
    headers = ("번호", "구분", "문제", "대상", "다음 조치", "수정")
    widths = (4, 9, 18, 22, 34, 6)
    rows = [
        "  " + format_table_row(headers, widths),
        "  " + format_table_separator(widths),
    ]
    for index, issue in enumerate(issues, start=1):
        rows.append(
            "  "
            + format_table_row(
                (
                    str(index),
                    format_severity(issue.severity),
                    issue.title,
                    issue.target,
                    issue.recommendation,
                    "가능" if issue.fixable_by_agent else "수동",
                ),
                widths,
            )
        )
    return rows


def format_table_row(values: tuple[str, ...], widths: tuple[int, ...]) -> str:
    cells = [truncate_cell(value, width).ljust(width) for value, width in zip(values, widths)]
    return "| " + " | ".join(cells) + " |"


def format_table_separator(widths: tuple[int, ...]) -> str:
    return "| " + " | ".join("-" * width for width in widths) + " |"


def truncate_cell(value: str, width: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= width:
        return normalized
    if width <= 1:
        return normalized[:width]
    return normalized[: width - 1] + "…"


def format_beginner_fix_preview(analysis: ProjectAnalysis) -> str:
    previews = build_fix_previews(analysis)
    if analysis.registration_status == "불가":
        return (
            "- 현재는 수정안을 만들 수 없습니다.\n"
            "- 먼저 프로젝트 경로나 폴더 구조를 확인해야 합니다.\n"
            "- 파일은 수정하지 않았습니다."
        )
    if not previews:
        return (
            "- 자동 수정이 필요한 항목이 없습니다.\n"
            "- 파일은 수정하지 않았습니다.\n"
            "- 다음 단계에서 리포트 생성 또는 재검증을 진행할 수 있습니다."
        )
    rows = [
        "- dry-run 결과입니다. 아직 파일은 수정하지 않았습니다.",
        f"- 미리보기 항목: {len(previews)}개",
    ]
    for index, preview in enumerate(previews, start=1):
        rows.extend(
            [
                f"- [{index}] {preview.title}",
                f"  대상 파일: {preview.target}",
                f"  작업 내용: {preview.action}",
                "  변경 미리보기:",
            ]
        )
        rows.extend(f"    {line}" for line in preview.preview_lines)
    rows.extend(
        [
            "- 적용하려면 다음 단계에서 1번을 선택합니다.",
            "- 2번 또는 3번을 선택하면 파일을 수정하지 않습니다.",
        ]
    )
    return "\n".join(rows)


def format_beginner_approval(analysis: ProjectAnalysis, approval_policy: str) -> str:
    options = build_approval_options(analysis)
    rows = [
        "- 1번을 선택한 경우에만 변경을 적용합니다.",
        "- 승인 전 상태: 파일은 아직 수정되지 않았습니다.",
        "- 적용 범위: Step 5에 표시된 미리보기 항목으로 제한됩니다.",
        "- 선택 방법: 번호만 입력합니다.",
    ]
    for index, option in enumerate(options, start=1):
        state = "선택 가능" if option.enabled else "선택 불가"
        modifies = "파일 수정 있음" if option.will_modify_files else "파일 수정 없음"
        rows.extend(
            [
                f"- {index}. {option.label} ({state})",
                f"  설명: {option.description}",
                f"  결과: {modifies}",
            ]
        )
    if not options[0].enabled:
        rows.append("- 1번은 수정안이 있을 때만 선택할 수 있습니다.")
    return "\n".join(rows)


def format_beginner_apply_step(
    analysis: ProjectAnalysis,
    applied_changes: list[AppliedChange] | None = None,
) -> str:
    if applied_changes is not None:
        applied_count = len([change for change in applied_changes if change.status == "applied"])
        skipped_count = len([change for change in applied_changes if change.status == "skipped"])
        rows = [
            "- 사용자가 1번을 선택해 적용을 승인했습니다.",
            "- Step 5의 미리보기 항목만 적용했습니다.",
            f"- 적용 완료: {applied_count}개",
            f"- 건너뜀: {skipped_count}개",
        ]
        rows.extend(f"  - {change.message} ({change.target})" for change in applied_changes)
        rows.extend(
            [
                "- 재검증 결과:",
                f"  - 등록 상태: {analysis.registration_status}",
                f"  - 남은 문제: {len(analysis.issue_details)}개",
                "- 삭제 작업은 수행하지 않았습니다.",
                "- AI Studio 환경값을 입력하려면 이 단계에서 1번을 선택하세요.",
                "  - 저장 대상: ai_studio.env",
            ]
        )
        return "\n".join(rows)

    previews = build_fix_previews(analysis)
    if analysis.registration_status == "불가":
        return (
            "- 현재는 적용할 수 없습니다.\n"
            "- 프로젝트 경로 문제가 먼저 해결되어야 합니다.\n"
            "- 삭제 작업은 수행하지 않습니다."
        )
    if not previews:
        return (
            "- 적용할 수정안이 없습니다.\n"
            "- 파일을 생성하거나 수정하지 않습니다.\n"
            "- 삭제 작업은 수행하지 않습니다."
        )
    rows = [
        "- Step 6에서 1번 승인 후에만 아래 파일을 생성하거나 수정합니다.",
        "- 삭제 작업은 수행하지 않습니다.",
        f"- 적용 예정 항목: {len(previews)}개",
    ]
    rows.extend(f"  - {preview.target}: {preview.title}" for preview in previews)
    return "\n".join(rows)


def format_beginner_apply_result(applied_changes: list[AppliedChange], analysis: ProjectAnalysis) -> str:
    applied_count = len([change for change in applied_changes if change.status == "applied"])
    skipped_count = len([change for change in applied_changes if change.status == "skipped"])
    return (
        "1번 적용을 승인했습니다.\n"
        f"- 적용 완료: {applied_count}개\n"
        f"- 건너뜀: {skipped_count}개\n"
        f"- 재검증 등록 상태: {analysis.registration_status}\n"
        f"- 남은 문제: {len(analysis.issue_details)}개"
    )


def format_beginner_local_serving(analysis: ProjectAnalysis) -> str:
    serving = analysis.local_serving
    rows = [
        f"- 상태: {serving.status}",
        f"- 방식: {serving.mode}",
        f"- health 확인: {serving.health_endpoint}",
        f"- predict 테스트: {serving.predict_endpoint}",
        "- 체크 결과:",
    ]
    rows.extend(f"  - {format_check_status(check.status)} {check.label}: {check.detail}" for check in serving.checks)
    rows.extend(
        [
            "- 실행 전 확인 명령:",
            f"  - {serving.commands[0]}",
        ]
    )
    if serving.status == "준비 가능":
        rows.append("- dry-run 확인 후 로컬 서버 실행 명령을 안내합니다.")
    elif serving.status == "보완 필요":
        rows.append("- 보완 항목을 수정한 뒤 다시 로컬 서빙 테스트를 진행합니다.")
    else:
        rows.append("- 차단 항목이 있어 아직 로컬 서빙을 실행하지 않습니다.")
    return "\n".join(rows)


def format_beginner_report(analysis: ProjectAnalysis) -> str:
    report_path = Path(analysis.path or ".") / "ml-agent-report.json"
    register_plan_path = Path(analysis.path or ".") / "mlflow-registration-plan.json"
    rows = [
        "- 최종 결과 요약:",
        f"  - 프로젝트: {analysis.path}",
        f"  - 등록 상태: {analysis.registration_status}",
        f"  - MLflow: {'정상' if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else '보완 필요'}",
        f"  - Job Template: {'준비 가능' if analysis.job_template_ready else '보완 필요'}",
        f"  - 로컬 서빙: {analysis.local_serving.status}",
        f"  - 문제 수: {len(analysis.issue_details)}개",
    ]
    if analysis.model_parameters:
        rows.append("- 모델 파라미터:")
        rows.extend(f"  - {item}" for item in format_model_parameters(analysis.model_parameters, limit=8))
    if analysis.issue_details:
        rows.append("- 남은 문제:")
        for index, issue in enumerate(analysis.issue_details[:3], start=1):
            rows.append(f"  - [{index}] {issue.title}: {issue.recommendation}")
        if len(analysis.issue_details) > 3:
            rows.append(f"  - 외 {len(analysis.issue_details) - 3}개")
    else:
        rows.append("- 남은 문제: 없음")

    rows.append("- 다음 조치:")
    rows.extend(f"  - {action}" for action in analysis.next_actions[:3])
    rows.extend(
        [
            "- MLflow 등록 테스트:",
            f"  - dry-run: ml-agent register {analysis.path} --dry-run",
            f"  - 실행: ml-agent register {analysis.path}",
            "  - MLFLOW_TRACKING_URL이 비어 있으면 file:./mlruns 로컬 저장소를 사용합니다.",
            f"  - 등록 계획 파일: {register_plan_path}",
            "- 리포트 저장:",
            f"  - 저장 경로: {report_path}",
            f"  - 저장 명령: ml-agent report {analysis.path}",
            "- 콘솔에는 위 요약을 표시했고, 파일 저장은 사용자가 리포트 생성을 실행할 때 수행합니다.",
        ]
    )
    return "\n".join(rows)


def format_severity(severity: str) -> str:
    labels = {
        "blocker": "필수 확인",
        "warning": "보완 권장",
        "info": "참고",
    }
    return labels.get(severity, severity)


def format_check_status(status: str) -> str:
    labels = {
        "pass": "[통과]",
        "warn": "[보완]",
        "block": "[차단]",
    }
    return labels.get(status, f"[{status}]")


def format_count(values: list[str]) -> str:
    if not values:
        return "없음"
    preview = ", ".join(values[:3])
    if len(values) > 3:
        preview += f" 외 {len(values) - 3}개"
    return preview


def format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def handle_intermediate_request(request: str) -> str:
    profile = build_ml_platform_profile(MODE_INTERMEDIATE)
    if not request:
        return "분석할 프로젝트 경로나 질문을 입력하세요."
    if "mlflow" in request.lower():
        return (
            "MLflow 설정 검증을 준비합니다.\n"
            f"- 담당 sub-agent: {profile.subagents[1].name}\n"
            "- tracking 설정 확인\n"
            "- experiment/run 기록 코드 확인\n"
            "- model artifact 저장 경로 확인\n"
            "- run_model.py 설정 변수 확인\n"
            "- predict.py 또는 ModelWrapper 구현 확인\n"
            "- mlflow.pyfunc.log_model 파라미터 확인\n"
            "- requirements.txt serve/mlflow 패키지 확인\n"
            "- AI Studio MLflow 환경변수 확인\n"
            "파일 수정 전에는 dry-run 결과를 먼저 제공합니다."
        )
    if "job" in request.lower() or "template" in request.lower() or "초안" in request:
        return (
            "Job Template 초안 생성을 준비합니다.\n"
            f"- 담당 sub-agent: {profile.subagents[2].name}\n"
            "- entrypoint, arguments, requirements를 기준으로 후보를 만듭니다.\n"
            "- 항목별로 선택 적용할 수 있게 미리보기를 제공합니다."
        )
    if "오류" in request or "로그" in request or "error" in request.lower():
        return (
            "오류 로그 분석을 준비합니다.\n"
            f"- 담당 sub-agent: {profile.subagents[3].name}\n"
            "- 실패 위치와 원인을 요약합니다.\n"
            "- MLflow / requirements / arguments 관련 문제를 우선 확인합니다."
        )
    return (
        "전체 등록 가능 여부 분석을 준비합니다.\n"
        "- Deep Agent sub-agent 분담 사용\n"
        "- 프로젝트 구조\n"
        "- MLflow 설정\n"
        "- requirements\n"
        "- arguments\n"
        "- Job Template 후보\n"
        "수정안은 항목별 dry-run으로 먼저 보여드립니다."
    )


def handle_advanced_input(command: str) -> str:
    if not command:
        return ADVANCED_INTRO
    parts = command.split()
    if parts[0] in {"ml-agent", "aiu"}:
        parts = parts[1:]
    if not parts:
        return ADVANCED_INTRO
    if parts[0] == "chat":
        return "chat: Agent 대화 모드 진입"
    if parts[0] == "config":
        return format_config_summary(AppConfig.load())
    if parts[0] == "init":
        return initialize_runtime_layout()
    if parts[0] == "prompts":
        as_json = "--json" in parts
        templates = load_prompt_templates()
        if as_json:
            return json.dumps(prompt_templates_as_dict(templates), ensure_ascii=False, indent=2)
        return format_prompt_templates(templates)
    if parts[0] == "errors":
        return handle_error_command(parts[1:])
    if parts[0] == "profile":
        as_json = "--json" in parts
        profile = build_ml_platform_profile(MODE_ADVANCED)
        if as_json:
            return json.dumps(profile.as_dict(), ensure_ascii=False, indent=2)
        return format_profile(profile)
    if parts[0] == "deepagents":
        as_json = "--json" in parts
        source_zip = option_value(parts, "--source")
        if as_json:
            return json.dumps(deepagents_libs_as_dict(source_zip), ensure_ascii=False, indent=2)
        return format_deepagents_libs(source_zip)
    if parts[0] == "sample":
        return handle_sample_command(parts[1:])
    if parts[0] not in {"analyze", "validate", "fix", "apply", "serve", "report", "register", "verify-run"}:
        return "unknown command. available: analyze, validate, fix, apply, serve, report, register, verify-run, sample, chat, profile, deepagents, config, init, prompts, errors"
    path = parts[1] if len(parts) > 1 else "."
    as_json = "--json" in parts
    result = run_command(parts[0], path, dry_run="--dry-run" in parts, skip_serving="--skip-serving" in parts)
    if as_json:
        return json.dumps(result.as_dict(), ensure_ascii=False, indent=2)
    return format_command_result(result)


def option_value(parts: list[str], option: str) -> str | None:
    if option not in parts:
        return None
    value_index = parts.index(option) + 1
    if value_index >= len(parts):
        return None
    value = parts[value_index]
    if value.startswith("--"):
        return None
    return value


def handle_sample_command(parts: list[str]) -> str:
    as_json = "--json" in parts
    action = next((part for part in parts if not part.startswith("--")), "list")
    root = sample_projects_root()
    if action == "list":
        payload = sample_catalog_as_dict()
        if as_json:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return format_sample_catalog(payload)
    if action == "create":
        kind = option_value(parts, "--kind") or "all"
        framework = option_value(parts, "--framework") or option_value(parts, "-f") or "generic"
        if kind in {"all", "matrix"}:
            paths = create_all_model_samples(root)
        elif kind in {"large10", "big10", "heavy10"}:
            paths = create_large_model_samples(root)
        elif kind in {"standard", "template", "ml-dl-template"}:
            paths = [create_standard_template_sample(root, framework)]
        elif kind in SAMPLE_MODEL_SPECS:
            paths = [create_model_sample(root / SAMPLE_MODEL_SPECS[kind].directory, SAMPLE_MODEL_SPECS[kind])]
        else:
            message = f"unknown sample kind: {kind}"
            if as_json:
                return json.dumps({"status": "error", "message": message, "available_kinds": sorted(SAMPLE_MODEL_SPECS) + ["standard", "large10"]}, ensure_ascii=False, indent=2)
            return message
        payload = {"status": "ok", "created": [str(path) for path in paths], "count": len(paths)}
        if as_json:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return "sample projects created\n" + "\n".join(f"- {path}" for path in paths)
    if action == "run":
        try:
            payload = run_sample_projects_command(parts)
        except ValueError as exc:
            payload = {
                "status": "error",
                "message": str(exc),
                "available_kinds": sorted(SAMPLE_MODEL_SPECS) + ["all", "large10", "standard"],
            }
        if as_json:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if payload.get("status") == "error":
            return str(payload["message"])
        return format_sample_run_result(payload)
    return "unknown sample action. available: list, create, run"


def run_sample_projects_command(parts: list[str]) -> dict[str, object]:
    kind = option_value(parts, "--kind") or "all"
    framework = option_value(parts, "--framework") or option_value(parts, "-f") or "generic"
    register = "--register" in parts
    dry_run = "--dry-run" in parts
    train_mode = "--mode" in parts and option_value(parts, "--mode") == "train"
    timeout = parse_int_option(parts, "--timeout", default=120)
    paths = resolve_sample_run_paths(kind, framework)
    results = [
        run_single_sample_project(path, register=register, dry_run=dry_run, train_mode=train_mode, timeout=timeout)
        for path in paths
    ]
    pass_count = sum(1 for result in results if result["status"] == "pass")
    fail_count = len(results) - pass_count
    return {
        "status": "ok" if fail_count == 0 else "needs_action",
        "command": "sample run",
        "kind": kind,
        "mode": "train" if train_mode else "pretrained",
        "run_mode": "register" if register else "prepare-only",
        "dry_run": dry_run,
        "sample_root": str(sample_projects_root()),
        "count": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "results": results,
    }


def parse_int_option(parts: list[str], option: str, default: int) -> int:
    value = option_value(parts, option)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def resolve_sample_run_paths(kind: str, framework: str = "generic") -> list[Path]:
    root = sample_projects_root()
    if kind in {"all", "matrix"}:
        return create_all_model_samples(root)
    if kind in {"large10", "big10", "heavy10"}:
        return create_large_model_samples(root)
    if kind in {"standard", "template", "ml-dl-template"}:
        return [create_standard_template_sample(root, framework)]
    if kind in SAMPLE_MODEL_SPECS:
        spec = SAMPLE_MODEL_SPECS[kind]
        return [create_model_sample(root / spec.directory, spec)]
    existing = resolve_existing_sample_project(f"/sample {kind}", root)
    if existing:
        return [existing]
    raise ValueError(f"unknown sample kind: {kind}")


def run_single_sample_project(
    path: Path,
    register: bool = False,
    dry_run: bool = False,
    train_mode: bool = False,
    timeout: int = 120,
) -> dict[str, object]:
    run_model_path = path / "run_model.py"
    if not run_model_path.exists():
        return {
            "project": str(path),
            "name": path.name,
            "status": "fail",
            "exit_code": 2,
            "command": "run_model.py missing",
            "stdout": "",
            "stderr": "run_model.py 없음",
        }
    command = [sys.executable, "run_model.py", "--env-file", "ai_studio.env"]
    if train_mode:
        command.extend(["--mode", "train"])
    if register:
        command.append("--register")
        if dry_run:
            command.append("--dry-run")
    else:
        command.append("--prepare-only")
    try:
        result = subprocess.run(
            command,
            cwd=path,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        status = "pass" if result.returncode == 0 else "fail"
        analysis = analyze_project(str(path))
        return {
            "project": str(path),
            "name": path.name,
            "status": status,
            "exit_code": result.returncode,
            "command": " ".join(command),
            "registration_status": analysis.registration_status,
            "artifact_count": len(analysis.scan.model_artifacts),
            "stdout": truncate_output(result.stdout),
            "stderr": truncate_output(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "project": str(path),
            "name": path.name,
            "status": "fail",
            "exit_code": 124,
            "command": " ".join(command),
            "stdout": truncate_output(exc.stdout or ""),
            "stderr": f"timeout after {timeout}s",
        }


def truncate_output(text: str, limit: int = 1200) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>"


def format_sample_run_result(payload: dict[str, object]) -> str:
    rows = [
        "Sample model run result",
        f"- kind: {payload['kind']}",
        f"- mode: {payload['mode']}",
        f"- run: {payload['run_mode']}{' dry-run' if payload['dry_run'] else ''}",
        f"- total: {payload['count']}, pass: {payload['pass_count']}, fail: {payload['fail_count']}",
    ]
    for result in payload["results"]:
        rows.append(
            f"- [{result['status']}] {result['name']} "
            f"(exit={result['exit_code']}, artifacts={result.get('artifact_count', 0)})"
        )
        if result.get("stderr"):
            rows.append(f"  stderr: {result['stderr']}")
    return "\n".join(rows)


def sample_catalog_as_dict() -> dict[str, object]:
    return {
        "sample_root": str(sample_projects_root()),
        "standard_templates": [
            {
                "kind": "standard",
                "framework": framework,
                "artifact_path": str(spec["artifact"]),
                "command": f"aiu sample create --kind standard --framework {framework}",
            }
            for framework, spec in STANDARD_TEMPLATE_FRAMEWORKS.items()
        ],
        "basic": [
            {
                "kind": spec.kind,
                "title": spec.title,
                "directory": spec.directory,
                "artifact_path": spec.artifact_path,
                "artifact_size": format_bytes(spec.artifact_size_bytes),
            }
            for spec in SAMPLE_MODEL_SPECS.values()
        ],
        "large10": [
            {
                "kind": spec.kind,
                "title": spec.title,
                "directory": spec.directory,
                "artifact_path": spec.artifact_path,
                "artifact_size": format_bytes(spec.artifact_size_bytes),
            }
            for spec in LARGE_MODEL_SAMPLE_SPECS
        ],
    }


def format_sample_catalog(payload: dict[str, object]) -> str:
    rows = [
        "Sample model catalog",
        f"- root: {payload['sample_root']}",
        "- create all: aiu sample create --kind all",
        "- create one: aiu sample create --kind tensorflow",
        "- create large10: aiu sample create --kind large10",
        "- create standard: aiu sample create --kind standard --framework pytorch",
        "- run all prepare: aiu sample run --kind all",
        "- run all register dry-run: aiu sample run --kind all --register --dry-run",
        "",
        "Standard templates:",
    ]
    rows.extend(
        f"- {item['framework']}: {item['command']}"
        for item in payload.get("standard_templates", [])
    )
    rows.extend([
        "",
        "Basic samples:",
    ])
    for item in payload["basic"]:
        rows.append(f"- {item['kind']}: {item['title']} ({item['artifact_path']}, {item['artifact_size']})")
    rows.append("")
    rows.append("Large10 samples:")
    for item in payload["large10"]:
        rows.append(f"- {item['kind']}: {item['title']} ({item['artifact_path']}, {item['artifact_size']})")
    return "\n".join(rows)


def run_command(command: str, path: str, dry_run: bool = False, skip_serving: bool = False) -> CommandResult:
    target = Path(path)
    profile = build_ml_platform_profile(MODE_ADVANCED)
    details = []
    status = "ok"
    exit_code = 0
    analysis = analyze_project(path)
    fix_previews = build_fix_previews(analysis) if command in {"fix", "apply"} else None
    approval_options = build_approval_options(analysis) if command in {"fix", "apply"} else None
    applied_changes = None

    if command in {"analyze", "validate", "fix", "apply", "serve", "report", "register", "verify-run"}:
        details.append(f"path={target}")
        details.append(f"agent_profile={profile.name}")
    if command == "fix" and not dry_run:
        details.append("default=dry-run")
        details.append("advanced_apply_required=true")
    if command == "apply":
        details.append("explicit_apply=true")
        applied_changes = apply_fix_previews(path, fix_previews or [])
        details.append(f"applied_changes={len([change for change in applied_changes if change.status == 'applied'])}")
        details.append(f"skipped_changes={len([change for change in applied_changes if change.status == 'skipped'])}")
        analysis = analyze_project(path)
    if command in {"analyze", "validate", "fix", "apply", "serve", "report", "register", "verify-run"}:
        details.append(f"registration_status={analysis.registration_status}")
    if command == "fix":
        details.append(f"preview_items={len(fix_previews or [])}")
        details.append("approval_required=true")
        details.append("apply_choice=explicit")
    if command == "report":
        result_file = str(target / "ml-agent-report.json")
        details.append(f"result_file={result_file}")
    elif command == "register":
        result_file = str(target / "mlflow-registration-plan.json")
        register_plan = build_registration_plan(analysis, dry_run=dry_run)
        write_registration_plan_file(Path(result_file), register_plan)
        details.extend(register_plan["details"])
    elif command == "verify-run":
        result_file = str(target / "mlflow-run-verification.json")
        verification = run_mlflow_verification(target, skip_serving=skip_serving)
        write_verification_file(Path(result_file), verification)
        details.extend(str(item) for item in verification.get("details", []))
        status = str(verification.get("status", "error"))
        exit_code = int(verification.get("exit_code", 1))
    else:
        result_file = None

    details.append(f"mlflow={'ok' if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else 'missing'}")
    details.append(f"job_template={'ready' if analysis.job_template_ready else 'needs_input'}")
    if command in {"serve", "report", "verify-run"}:
        details.append(f"local_serving={analysis.local_serving.status}")
        details.append(f"health={analysis.local_serving.health_endpoint}")
        details.append(f"predict={analysis.local_serving.predict_endpoint}")
    details.append(f"issues={len(analysis.issues)}")
    if command == "verify-run":
        pass
    elif analysis.registration_status == "불가":
        status = "error"
        exit_code = 2
    elif command == "validate" and analysis.issues:
        status = "needs_action"
        exit_code = 1
    elif command == "serve" and analysis.local_serving.status != "준비 가능":
        status = "needs_action"
        exit_code = 1
    elif command == "register" and (analysis.registration_status == "불가" or not analysis.job_template_ready):
        status = "needs_action"
        exit_code = 1
    if command == "report" and result_file:
        write_report_file(Path(result_file), analysis, profile.name, details, status, exit_code)
    return CommandResult(
        command,
        path,
        status,
        exit_code,
        details,
        result_file,
        analysis,
        fix_previews,
        approval_options,
        applied_changes,
    )


def write_report_file(
    path: Path,
    analysis: ProjectAnalysis,
    agent_profile: str,
    details: list[str],
    status: str,
    exit_code: int,
) -> None:
    ensure_read_write_directory(path.parent)
    report = {
        "title": "AI ML 온보딩 분석 리포트",
        "agent_profile": agent_profile,
        "status": status,
        "exit_code": exit_code,
        "summary": {
            "project_path": analysis.path,
            "registration_status": analysis.registration_status,
            "mlflow": "ok" if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else "missing",
            "job_template": "ready" if analysis.job_template_ready else "needs_input",
            "local_serving": analysis.local_serving.status,
            "health_endpoint": analysis.local_serving.health_endpoint,
            "predict_endpoint": analysis.local_serving.predict_endpoint,
            "issue_count": len(analysis.issues),
            "next_actions": analysis.next_actions,
        },
        "details": details,
        "analysis": analysis.as_dict(),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_registration_plan(analysis: ProjectAnalysis, dry_run: bool = False) -> dict[str, object]:
    root = Path(analysis.path or ".")
    env_path = root / "ai_studio.env"
    tracking_uri = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MLFLOW_TRACKING_URL="):
                tracking_uri = line.split("=", 1)[1].strip()
                break
    tracking_mode = "remote" if tracking_uri else "local-file-store"
    effective_tracking_uri = tracking_uri or "file:./mlruns"
    command = "python run_model.py --env-file ai_studio.env --register"
    if dry_run:
        command += " --dry-run"
    details = [
        f"register_mode={'dry-run' if dry_run else 'execute'}",
        f"mlflow_tracking_mode={tracking_mode}",
        f"mlflow_tracking_uri={effective_tracking_uri}",
        f"register_command={command}",
    ]
    if tracking_mode == "local-file-store":
        details.append("local_mlruns_default=true")
    return {
        "project_path": analysis.path,
        "status": analysis.registration_status,
        "dry_run": dry_run,
        "tracking_mode": tracking_mode,
        "tracking_uri": effective_tracking_uri,
        "command": command,
        "details": details,
        "ready": analysis.registration_status == "등록 가능" and analysis.job_template_ready,
    }


def write_registration_plan_file(path: Path, plan: dict[str, object]) -> None:
    ensure_read_write_directory(path.parent)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_mlflow_verification(project_path: Path, skip_serving: bool = False) -> dict[str, object]:
    root = project_path.resolve()
    report: dict[str, object] = {
        "status": "error",
        "exit_code": 1,
        "project_path": str(root),
        "details": [],
        "errors": [],
        "run_model": {},
        "mlflow_run": {},
        "metric_report": {},
        "model_test": {},
        "serving": {"status": "skipped", "reason": "not-run"},
    }
    details: list[str] = report["details"]  # type: ignore[assignment]
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    if not root.exists() or not root.is_dir():
        errors.append("프로젝트 폴더를 찾을 수 없습니다.")
        details.append("verify_run=project_missing")
        return report
    run_model_path = root / "run_model.py"
    if not run_model_path.exists():
        errors.append("run_model.py가 없어 실제 MLflow 실행 검증을 시작할 수 없습니다.")
        details.append("verify_run=run_model_missing")
        return report

    run_result = execute_run_model(root)
    report["run_model"] = run_result
    details.append(f"run_model_exit_code={run_result['exit_code']}")
    if run_result["exit_code"] != 0:
        report["status"] = "error"
        report["exit_code"] = 1
        errors.append("run_model.py 실행 실패")
        return report

    mlflow_result = inspect_latest_mlflow_run(root)
    report["mlflow_run"] = mlflow_result
    details.append(f"mlflow_run_status={mlflow_result['status']}")
    run_id = str(mlflow_result.get("run_id") or "")
    if run_id:
        details.append(f"run_id={run_id}")
    if mlflow_result["status"] != "pass":
        report["status"] = "error"
        report["exit_code"] = 1
        errors.append(str(mlflow_result.get("reason") or "MLflow run 생성 확인 실패"))
        return report

    metrics = mlflow_result.get("metrics") if isinstance(mlflow_result.get("metrics"), dict) else {}
    params = mlflow_result.get("params") if isinstance(mlflow_result.get("params"), dict) else {}
    report["metric_report"] = {
        "status": "pass" if metrics or params else "warn",
        "metrics": metrics,
        "params": params,
        "metric_count": len(metrics),
        "param_count": len(params),
    }
    details.append(f"metrics={len(metrics)}")
    details.append(f"params={len(params)}")

    model_result = load_logged_model_and_predict(root, mlflow_result)
    report["model_test"] = model_result
    details.append(f"model_test_status={model_result['status']}")

    if skip_serving:
        report["serving"] = {"status": "skipped", "reason": "--skip-serving"}
    else:
        serving_result = run_local_serving_smoke_test(root)
        report["serving"] = serving_result
        details.append(f"serving_status={serving_result['status']}")

    required_ok = (
        run_result["exit_code"] == 0
        and mlflow_result["status"] == "pass"
        and (metrics or params)
        and model_result["status"] == "pass"
        and (skip_serving or report["serving"].get("status") == "pass")  # type: ignore[union-attr]
    )
    if required_ok:
        report["status"] = "ok"
        report["exit_code"] = 0
        details.append("completion=run_created")
    else:
        report["status"] = "needs_action"
        report["exit_code"] = 1
        details.append("completion=needs_action")
    return report


def execute_run_model(root: Path) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(root))
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = str(root) + os.pathsep + env["PYTHONPATH"]
    tracking_url = read_ai_studio_env_value(root / "ai_studio.env", "MLFLOW_TRACKING_URL")
    if not tracking_url:
        env["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    command = [sys.executable, "run_model.py", "--env-file", "ai_studio.env", "--register"]
    timeout = AppConfig.load().get_int("DEV_COMMAND_TIMEOUT", 120)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "exit_code": 124,
            "command": " ".join(command),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    return {
        "status": "pass" if completed.returncode == 0 else "error",
        "exit_code": completed.returncode,
        "command": " ".join(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def read_ai_studio_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw_line in safe_read_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def effective_tracking_uri(root: Path) -> str:
    return read_ai_studio_env_value(root / "ai_studio.env", "MLFLOW_TRACKING_URL") or "file:./mlruns"


@dataclass
class WorkingDirectory:
    path: Path
    previous: Path | None = None

    def __enter__(self) -> None:
        self.previous = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.previous is not None:
            os.chdir(self.previous)


def inspect_latest_mlflow_run(root: Path) -> dict[str, object]:
    try:
        import mlflow  # type: ignore
    except Exception as exc:
        fallback = inspect_latest_mlruns_file_store(root)
        if fallback["status"] == "pass":
            fallback["reason"] = "mlflow import unavailable; inspected local mlruns files"
            return fallback
        return {"status": "error", "reason": f"mlflow import failed: {exc}"}

    tracking_uri = effective_tracking_uri(root)
    try:
        with WorkingDirectory(root):
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            runs = []
            for experiment in experiments:
                runs.extend(
                    client.search_runs(
                        [experiment.experiment_id],
                        order_by=["attributes.start_time DESC"],
                        max_results=1,
                    )
                )
        if not runs:
            return {"status": "error", "reason": "MLflow run을 찾지 못했습니다.", "tracking_uri": tracking_uri}
        latest = sorted(runs, key=lambda run: run.info.start_time or 0, reverse=True)[0]
        config = safe_read_json(root / "config.json")
        artifact_path = "ai_studio"
        if isinstance(config.get("model"), dict):
            artifact_path = str(config["model"].get("artifact_path") or artifact_path)  # type: ignore[index]
        return {
            "status": "pass",
            "tracking_uri": tracking_uri,
            "run_id": latest.info.run_id,
            "experiment_id": latest.info.experiment_id,
            "artifact_uri": latest.info.artifact_uri,
            "model_uri": f"runs:/{latest.info.run_id}/{artifact_path}",
            "metrics": dict(latest.data.metrics),
            "params": dict(latest.data.params),
            "tags": dict(latest.data.tags),
        }
    except Exception as exc:
        return {"status": "error", "reason": f"MLflow run inspection failed: {exc}", "tracking_uri": tracking_uri}


def inspect_latest_mlruns_file_store(root: Path) -> dict[str, object]:
    mlruns = root / "mlruns"
    if not mlruns.exists():
        return {"status": "error", "reason": "mlruns 폴더를 찾지 못했습니다."}
    run_dirs = [
        path
        for path in mlruns.rglob("*")
        if path.is_dir() and (path / "meta.yaml").exists() and path.parent.name != "models"
    ]
    if not run_dirs:
        return {"status": "error", "reason": "mlruns 안에서 run meta.yaml을 찾지 못했습니다."}
    latest = sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return {
        "status": "pass",
        "tracking_uri": "file:./mlruns",
        "run_id": latest.name,
        "experiment_id": latest.parent.name,
        "artifact_uri": str(latest / "artifacts"),
        "model_uri": str(latest / "artifacts" / "ai_studio"),
        "metrics": read_mlflow_key_value_dir(latest / "metrics"),
        "params": read_mlflow_key_value_dir(latest / "params"),
        "tags": read_mlflow_key_value_dir(latest / "tags"),
    }


def read_mlflow_key_value_dir(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    if not path.exists():
        return values
    for item in path.iterdir():
        if not item.is_file():
            continue
        text = safe_read_text(item).strip()
        if not text:
            continue
        last = text.splitlines()[-1].split()
        value = last[-1] if last else text
        try:
            values[item.name] = float(value)
        except ValueError:
            values[item.name] = value
    return values


def load_logged_model_and_predict(root: Path, mlflow_result: dict[str, object]) -> dict[str, object]:
    model_uri = str(mlflow_result.get("model_uri") or "")
    if not model_uri:
        return {"status": "error", "reason": "model_uri 없음"}
    try:
        import mlflow.pyfunc  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"mlflow/pandas import failed: {exc}", "model_uri": model_uri}
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        payload = default_input_example()
        input_path = root / "input_example.json"
        if input_path.exists():
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        if "input" in payload:
            input_item = payload.get("input", [{}])[0]
            model_input = pd.DataFrame(input_item.get("data") or [[0.0]])
        else:
            model_input = pd.DataFrame(payload["data"], columns=payload["columns"])
        prediction = model.predict(model_input)
        return {
            "status": "pass",
            "model_uri": model_uri,
            "input_rows": len(model_input),
            "prediction_sample": prediction_to_jsonable(prediction),
        }
    except Exception as exc:
        return {"status": "error", "reason": f"model load/predict failed: {exc}", "model_uri": model_uri}


def prediction_to_jsonable(prediction) -> object:
    if hasattr(prediction, "to_dict"):
        try:
            return prediction.to_dict(orient="records")
        except TypeError:
            return prediction.to_dict()
    if hasattr(prediction, "tolist"):
        return prediction.tolist()
    if isinstance(prediction, (str, int, float, bool)) or prediction is None:
        return prediction
    if isinstance(prediction, (list, tuple, dict)):
        return prediction
    return str(prediction)


def run_local_serving_smoke_test(root: Path) -> dict[str, object]:
    try:
        from fastapi import FastAPI  # type: ignore
        from fastapi.testclient import TestClient  # type: ignore
    except Exception as exc:
        return {"status": "skipped", "reason": f"fastapi serving dependency missing: {exc}"}
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: dict):
        return {"prediction": payload}

    try:
        client = TestClient(app)
        health_response = client.get("/health")
        input_payload = default_input_example()
        input_path = root / "input_example.json"
        if input_path.exists():
            input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        predict_response = client.post("/predict", json=input_payload)
        passed = health_response.status_code == 200 and predict_response.status_code == 200
        return {
            "status": "pass" if passed else "error",
            "health_status": health_response.status_code,
            "health_body": health_response.json(),
            "predict_status": predict_response.status_code,
            "predict_body": predict_response.json(),
        }
    except Exception as exc:
        return {"status": "error", "reason": f"serving smoke test failed: {exc}"}


def write_verification_file(path: Path, verification: dict[str, object]) -> None:
    ensure_read_write_directory(path.parent)
    path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_command_result(result: CommandResult) -> str:
    rows = "\n".join(f"- {detail}" for detail in result.details)
    return (
        f"{result.command}: {result.status}\n"
        f"exit_code: {result.exit_code}\n"
        f"{rows}"
    )


def initialize_runtime_layout() -> str:
    config = AppConfig.load()
    directories = ensure_runtime_layout(config)
    rows = "\n".join(f"- {directory}" for directory in directories)
    migration_notice = format_runtime_migration_notice(config.root_dir)
    notice = f"\n\nmigration notice:\n{migration_notice}" if migration_notice else ""
    return (
        "runtime layout initialized\n"
        f"skill_store_dir: {config.skill_store_dir()}\n"
        "directories:\n"
        f"{rows}"
        f"{notice}"
    )


def ensure_prompt_wiki_export() -> list[Path]:
    return export_prompt_templates_to_wiki(AppConfig.load())


def format_runtime_migration_notice(root: Path) -> str:
    legacy_paths = [
        root / "skills",
        root / "wiki",
        root / "sample_projects",
        root / "sessions",
        root / "chat_errors",
        root / "registration_packages",
        root / "fix_reports",
        root / "deepagents_source",
    ]
    existing = [path for path in legacy_paths if path.exists()]
    if not existing:
        return ""
    rows = [
        "기존 루트 폴더를 발견했습니다. 삭제하지 않았습니다.",
        "새 구조는 deep_agent/와 .aiu/를 사용합니다.",
    ]
    rows.extend(f"- legacy: {path}" for path in existing)
    return "\n".join(rows)


def handle_error_command(parts: list[str]) -> str:
    if not parts or parts[0] == "list":
        return format_error_log_list(list_error_logs())
    if parts[0] == "record":
        message = " ".join(parts[1:]).strip()
        if not message:
            return "error: message is required"
        entry = save_error_log(message)
        return f"error log saved: {entry.id}"
    if parts[0] == "analyze":
        if len(parts) < 2:
            return "error: log id or path is required"
        analysis = analyze_error_log(parts[1])
        return format_error_analysis(analysis)
    return "unknown errors command. available: errors list, errors record <message>, errors analyze <id-or-path>"


def normalize_clipboard_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        char
        for char in normalized
        if char == "\n" or char == "\t" or ord(char) >= 32
    )
    return unicodedata.normalize("NFC", normalized)


def copy_text_to_clipboard(text: str) -> tuple[bool, str]:
    normalized_text = normalize_clipboard_text(text)
    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["pbcopy"])
    elif os.name == "nt":
        commands.append(["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"])
        commands.append(["clip"])
    else:
        commands.extend(
            [
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ]
        )
    errors: list[str] = []
    for command in commands:
        executable = command[0]
        if shutil.which(executable) is None:
            continue
        try:
            subprocess.run(command, input=normalized_text, text=True, encoding="utf-8", check=True, capture_output=True)
            return True, executable
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{executable}: {exc}")
    detail = "; ".join(errors) if errors else "사용 가능한 클립보드 명령을 찾지 못했습니다."
    return False, detail


def handle_logo_command(copy_to_clipboard: bool = False) -> str:
    if not copy_to_clipboard:
        return LAUNCH_SCREEN
    copied, detail = copy_text_to_clipboard(LAUNCH_SCREEN)
    if copied:
        return f"{LAUNCH_SCREEN}\n\nlogo copied to clipboard: {detail}"
    return (
        f"{LAUNCH_SCREEN}\n\n"
        "logo clipboard copy failed.\n"
        f"reason: {detail}\n"
        "위 로고 블록을 콘솔에서 직접 선택해 복사할 수 있습니다."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-agent")
    subparsers = parser.add_subparsers(dest="command")

    for command in ["analyze", "validate", "fix", "apply", "serve", "report", "register", "verify-run"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("path", nargs="?", default=".")
        sub.add_argument("--json", action="store_true")
        sub.add_argument("--dry-run", action="store_true")
        if command == "verify-run":
            sub.add_argument("--skip-serving", action="store_true")

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("action", nargs="?", default="list", choices=["list", "create", "run"])
    sample_parser.add_argument("--kind", default="all")
    sample_parser.add_argument("--framework", "-f", default="generic")
    sample_parser.add_argument("--mode", choices=["pretrained", "train"], default="pretrained")
    sample_parser.add_argument("--register", action="store_true")
    sample_parser.add_argument("--dry-run", action="store_true")
    sample_parser.add_argument("--timeout", type=int)
    sample_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("chat")
    subparsers.add_parser("tui")
    subparsers.add_parser("config")
    subparsers.add_parser("init")
    logo_parser = subparsers.add_parser("logo")
    logo_parser.add_argument("--copy", action="store_true")
    prompts_parser = subparsers.add_parser("prompts")
    prompts_parser.add_argument("--json", action="store_true")
    errors_parser = subparsers.add_parser("errors")
    errors_parser.add_argument("action", nargs="?", default="list", choices=["list", "record", "analyze"])
    errors_parser.add_argument("value", nargs="*")
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--json", action="store_true")
    deepagents_parser = subparsers.add_parser("deepagents")
    deepagents_parser.add_argument("--json", action="store_true")
    deepagents_parser.add_argument("--source")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ensure_prompt_wiki_export()
    if not argv:
        ConsoleAssistant().run()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        ConsoleAssistant().run()
        return 0
    if args.command == "tui":
        from deep_agent.tui import run_tui

        return run_tui()
    if args.command == "config":
        print(format_config_summary(AppConfig.load()))
        return 0
    if args.command == "init":
        print(initialize_runtime_layout())
        return 0
    if args.command == "logo":
        print(handle_logo_command(copy_to_clipboard=args.copy))
        return 0
    if args.command == "prompts":
        ensure_prompt_wiki_export()
        templates = load_prompt_templates()
        if args.json:
            print(json.dumps(prompt_templates_as_dict(templates), ensure_ascii=False, indent=2))
        else:
            print(format_prompt_templates(templates))
        return 0
    if args.command == "errors":
        print(handle_error_command([args.action, *args.value]))
        return 0
    if args.command == "sample":
        parts = [args.action]
        if args.kind:
            parts.extend(["--kind", args.kind])
        if args.framework:
            parts.extend(["--framework", args.framework])
        if args.mode:
            parts.extend(["--mode", args.mode])
        if args.register:
            parts.append("--register")
        if args.dry_run:
            parts.append("--dry-run")
        if args.timeout is not None:
            parts.extend(["--timeout", str(args.timeout)])
        if args.json:
            parts.append("--json")
        print(handle_sample_command(parts))
        return 0
    if args.command == "profile":
        profile = build_ml_platform_profile(MODE_ADVANCED)
        if args.json:
            print(json.dumps(profile.as_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_profile(profile))
        return 0
    if args.command == "deepagents":
        if args.json:
            print(json.dumps(deepagents_libs_as_dict(args.source), ensure_ascii=False, indent=2))
        else:
            print(format_deepagents_libs(args.source))
        return 0
    if not args.command:
        parser.print_help()
        return 2

    result = run_command(args.command, args.path, dry_run=args.dry_run, skip_serving=getattr(args, "skip_serving", False))
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_command_result(result))
    return result.exit_code


LAUNCH_SCREEN = """+================================================================================================+
| AI ML Onboarding Console                                      Deep Agent | Windows 10/11 POC |
+------------------------------------------------------------------------------------------------+
| # Launch workflow                                                                             |
| 사용자 모드를 선택하세요.                                                                       |
|                                                                                                |
| > 처음 사용하는 경우에는 1. 초급자 모드를 권장합니다.                                           |
|                                                                                                |
| Agents: Plan(read-only) -> Build(approval-gated apply)                                         |
|                                                                                                |
| 1. 초급자 모드                                                                                 |
|    - 단계별 Wizard 방식                                                                        |
|    - 선택지만 따라가면 됨                                                                      |
|    - 파일 수정 전 자세한 설명 제공                                                             |
|                                                                                                |
| 2. 중급자 모드                                                                                 |
|    - Chat + Wizard 혼합                                                                        |
|    - 프로젝트 분석 후 수정안 선택                                                              |
|    - MLflow / Job Template 중심 검증                                                           |
|                                                                                                |
| 3. 고급자 모드                                                                                 |
|    - CLI Command 중심                                                                          |
|    - dry-run / apply / validate 직접 실행                                                      |
|    - 자동화 파이프라인 연계 가능                                                               |
+------------------------------------------------------------------------------------------------+
| esc interrupt   /mode beginner   /mode intermediate   /mode advanced                          |
+================================================================================================+"""


INTERMEDIATE_INTRO = """중급자 모드가 선택되었습니다.

프로젝트 경로를 입력하거나,
분석하고 싶은 내용을 자연어로 입력하세요.

예시:
- ./my-project 분석해줘
- MLflow 설정만 확인해줘
- Job Template 초안 만들어줘
- 오류 로그 분석해줘"""


INTERMEDIATE_MENU = """무엇을 하시겠습니까?

1. 전체 등록 가능 여부 분석
2. MLflow 설정 검증
3. Job Template 초안 생성
4. 오류 로그 분석
5. 수정안 미리보기 생성
6. Agent에게 직접 질문
0. 종료"""


ADVANCED_INTRO = """고급자 모드가 선택되었습니다.

사용 가능한 명령어:

analyze    프로젝트 구조 분석
validate   MLflow / Job Template 검증
fix        수정안 생성
apply      승인된 수정안 적용
serve      로컬 서빙 테스트 계획 확인
verify-run MLflow run 생성/모델 로드/추론 실행 검증
report     분석 리포트 생성
chat       Agent 대화 모드 진입
profile    Deep Agent 프로파일 확인
deepagents DeepAgents libs 사용 계획 확인
config     .env 설정 요약
init       런타임/스킬 저장 디렉터리 생성
prompts    저장된 프롬프트 템플릿 확인
errors     에러 로그 저장/목록/분석"""


if __name__ == "__main__":
    raise SystemExit(main())
