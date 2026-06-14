#!/usr/bin/env python3
"""Console POC for AI ML onboarding launch modes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app_config import AppConfig, ensure_runtime_layout, format_config_summary
from deep_agent_profile import build_ml_platform_profile, format_profile
from deepagents_libs import deepagents_libs_as_dict, format_deepagents_libs
from error_log_store import (
    analyze_error_log,
    format_error_analysis,
    format_error_log_list,
    list_error_logs,
    save_error_log,
)
from prompt_store import (
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
    "intermediate": MODE_INTERMEDIATE,
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
}


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
        self.output_fn(LAUNCH_SCREEN)

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
            self.output_fn(BEGINNER_INTRO)
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


def resolve_beginner_project_input(raw: str) -> tuple[str, str | None]:
    normalized = raw.strip().lower()
    if normalized in {"/sample matrix", "/samples test", "/샘플 매트릭스", "sample matrix", "샘플 매트릭스"}:
        sample_paths = create_all_model_samples(Path.cwd() / "sample_projects")
        return (
            str(sample_paths[0]),
            format_sample_matrix_message(sample_paths),
        )
    if normalized in {"/sample all", "/samples", "/샘플 전체", "sample all", "samples", "샘플 전체"}:
        sample_paths = create_all_model_samples(Path.cwd() / "sample_projects")
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
    sample_path = create_model_sample(Path.cwd() / "sample_projects" / spec.directory, spec)
    return (
        str(sample_path),
        f"{spec.title} 테스트 샘플을 생성했습니다.\n"
        f"- 위치: {sample_path}\n"
        "- 실제 외부 모델 다운로드 없이 모델 artifact를 흉내냅니다.\n"
        "- 초급자 Wizard가 이 경로로 계속 진행합니다.",
    )


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
    ]


def format_sample_matrix_message(sample_paths: list[Path]) -> str:
    rows = ["다양한 모델 테스트 샘플을 생성하고 Step 1 검증을 완료했습니다."]
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
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / spec.artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("\n".join(spec.requirements) + "\n", encoding="utf-8")
    (root / "train.py").write_text(spec.train_body, encoding="utf-8")
    (root / "README.md").write_text(
        f"# {spec.title} Wizard Sample\n\n"
        f"Sample kind: `{spec.kind}`\n\n"
        "This project is generated by `ml-agent` for beginner wizard testing.\n"
        "The model artifact is a local dummy file for closed-network POC validation.\n",
        encoding="utf-8",
    )
    ensure_sparse_file(artifact, spec.artifact_size_bytes)
    return root


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
        marker = ">" if step == index + 1 else " "
        sidebar_rows.append(f"{marker} {step:02d} {label}")

    header = render_tui_header(index, total, title)
    body_panel = render_tui_body(sidebar_rows, content or body)
    footer = render_tui_footer(index)
    return (
        f"{header}\n"
        f"{body_panel}\n"
        f"{footer}"
    )


def render_tui_header(index: int, total: int, title: str) -> str:
    width = 104
    tabs = " ".join(
        f"[Tab {step}]" if step == index + 1 else f" Tab {step} "
        for step in range(1, total + 1)
    )
    rows = [
        "┌" + "─" * width + "┐",
        f"│ {'AI ML 온보딩 Assistant':<{width - 1}}│",
        f"│ {'현재 단계: Tab ' + str(index + 1) + '/' + str(total) + ' · ' + title:<{width - 1}}│",
        f"│ {tabs:<{width - 1}}│",
        "└" + "─" * width + "┘",
    ]
    return "\n".join(rows)


def render_tui_body(sidebar_rows: list[str], content: str) -> str:
    left_width = 20
    right_width = 80
    content_lines = content.splitlines() or [""]
    row_count = max(len(sidebar_rows), len(content_lines))
    rows = [
        "┌" + "─" * left_width + "┬" + "─" * right_width + "┐",
        f"│ {'STEPS':<{left_width - 1}}│ {'CURRENT PANEL':<{right_width - 1}}│",
        "├" + "─" * left_width + "┼" + "─" * right_width + "┤",
    ]
    for row_index in range(row_count):
        left = sidebar_rows[row_index] if row_index < len(sidebar_rows) else ""
        right = content_lines[row_index] if row_index < len(content_lines) else ""
        rows.append(
            f"│ {truncate_cell(left, left_width - 2).ljust(left_width - 2)} "
            f"│ {truncate_cell(right, right_width - 2).ljust(right_width - 2)} │"
        )
    rows.append("└" + "─" * left_width + "┴" + "─" * right_width + "┘")
    return "\n".join(rows)


def render_tui_footer(index: int) -> str:
    command = "선택 번호: 1=적용  2=다시 보기  3=취소" if index == 5 else "Enter=다음  이전=이전 탭  1~10=탭 이동  종료=중단"
    return f"명령: {command}"


def analyze_project(project_path: str) -> ProjectAnalysis:
    target = Path(project_path or ".")
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
    registration_checks = build_registration_checks(
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
                "train.py 또는 main.py",
                "플랫폼 Job Template은 어떤 파일을 실행해야 하는지 알아야 합니다.",
                "train.py 또는 main.py처럼 실행 진입점을 확인할 수 있는 파일을 준비하세요.",
                False,
            )
        )
        next_actions.append("train.py 또는 main.py처럼 실행 진입점을 확인할 수 있는 파일을 준비하세요.")
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
    )


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
    ignored_dirs = {".git", ".venv", "__pycache__", "node_modules", "registration_packages", "sample_projects"}
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
    requirements_files: list[str],
    has_mlflow_dependency: bool,
    mlflow_usage_files: list[str],
    entrypoint_candidates: list[str],
    model_artifacts: list[str],
    scan: ProjectScan,
) -> list[RegistrationCheck]:
    return [
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
            detail=format_count(entrypoint_candidates) if entrypoint_candidates else "train.py 또는 main.py 후보 없음",
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
            format_count(entrypoint_candidates) if entrypoint_candidates else "train.py 또는 main.py 확인 필요",
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
    if "MLFLOW_CODE_MISSING" in issue_codes:
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
    root = Path(project_path or ".").resolve()
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
    preferred = [
        root / "train.py",
        root / "main.py",
        root / "src" / "train.py",
        root / "src" / "main.py",
    ]
    candidates = [relative_path(path, root) for path in preferred if path.exists()]
    for path in python_files:
        if len(candidates) >= 8:
            break
        if relative_path(path, root) in candidates:
            continue
        text = safe_read_text(path)
        if "argparse" in text or 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
            candidates.append(relative_path(path, root))
    return candidates


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
    rows = [
        "- 최종 결과 요약:",
        f"  - 프로젝트: {analysis.path}",
        f"  - 등록 상태: {analysis.registration_status}",
        f"  - MLflow: {'정상' if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else '보완 필요'}",
        f"  - Job Template: {'준비 가능' if analysis.job_template_ready else '보완 필요'}",
        f"  - 로컬 서빙: {analysis.local_serving.status}",
        f"  - 문제 수: {len(analysis.issue_details)}개",
    ]
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
    if parts[0] == "ml-agent":
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
        if as_json:
            return json.dumps(deepagents_libs_as_dict(), ensure_ascii=False, indent=2)
        return format_deepagents_libs()
    if parts[0] not in {"analyze", "validate", "fix", "apply", "serve", "report"}:
        return "unknown command. available: analyze, validate, fix, apply, serve, report, chat, profile, deepagents, config, init, prompts, errors"
    path = parts[1] if len(parts) > 1 else "."
    as_json = "--json" in parts
    result = run_command(parts[0], path, dry_run="--dry-run" in parts)
    if as_json:
        return json.dumps(result.as_dict(), ensure_ascii=False, indent=2)
    return format_command_result(result)


def run_command(command: str, path: str, dry_run: bool = False) -> CommandResult:
    target = Path(path)
    profile = build_ml_platform_profile(MODE_ADVANCED)
    details = []
    status = "ok"
    exit_code = 0
    analysis = analyze_project(path)
    fix_previews = build_fix_previews(analysis) if command in {"fix", "apply"} else None
    approval_options = build_approval_options(analysis) if command in {"fix", "apply"} else None
    applied_changes = None

    if command in {"analyze", "validate", "fix", "apply", "serve", "report"}:
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
    if command in {"analyze", "validate", "fix", "apply", "serve", "report"}:
        details.append(f"registration_status={analysis.registration_status}")
    if command == "fix":
        details.append(f"preview_items={len(fix_previews or [])}")
        details.append("approval_required=true")
        details.append("apply_choice=explicit")
    if command == "report":
        result_file = str(target / "ml-agent-report.json")
        details.append(f"result_file={result_file}")
    else:
        result_file = None

    details.append(f"mlflow={'ok' if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else 'missing'}")
    details.append(f"job_template={'ready' if analysis.job_template_ready else 'needs_input'}")
    if command in {"serve", "report"}:
        details.append(f"local_serving={analysis.local_serving.status}")
        details.append(f"health={analysis.local_serving.health_endpoint}")
        details.append(f"predict={analysis.local_serving.predict_endpoint}")
    details.append(f"issues={len(analysis.issues)}")
    if analysis.registration_status == "불가":
        status = "error"
        exit_code = 2
    elif command == "validate" and analysis.issues:
        status = "needs_action"
        exit_code = 1
    elif command == "serve" and analysis.local_serving.status != "준비 가능":
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return (
        "runtime layout initialized\n"
        f"skill_store_dir: {config.skill_store_dir()}\n"
        "directories:\n"
        f"{rows}"
    )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-agent")
    subparsers = parser.add_subparsers(dest="command")

    for command in ["analyze", "validate", "fix", "apply", "serve", "report"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("path", nargs="?", default=".")
        sub.add_argument("--json", action="store_true")
        sub.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("chat")
    subparsers.add_parser("config")
    subparsers.add_parser("init")
    prompts_parser = subparsers.add_parser("prompts")
    prompts_parser.add_argument("--json", action="store_true")
    errors_parser = subparsers.add_parser("errors")
    errors_parser.add_argument("action", nargs="?", default="list", choices=["list", "record", "analyze"])
    errors_parser.add_argument("value", nargs="*")
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--json", action="store_true")
    deepagents_parser = subparsers.add_parser("deepagents")
    deepagents_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        ConsoleAssistant().run()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chat":
        ConsoleAssistant().run()
        return 0
    if args.command == "config":
        print(format_config_summary(AppConfig.load()))
        return 0
    if args.command == "init":
        print(initialize_runtime_layout())
        return 0
    if args.command == "prompts":
        templates = load_prompt_templates()
        if args.json:
            print(json.dumps(prompt_templates_as_dict(templates), ensure_ascii=False, indent=2))
        else:
            print(format_prompt_templates(templates))
        return 0
    if args.command == "errors":
        print(handle_error_command([args.action, *args.value]))
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
            print(json.dumps(deepagents_libs_as_dict(), ensure_ascii=False, indent=2))
        else:
            print(format_deepagents_libs())
        return 0
    if not args.command:
        parser.print_help()
        return 2

    result = run_command(args.command, args.path, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_command_result(result))
    return result.exit_code


LAUNCH_SCREEN = """┌────────────────────────────────────────────┐
│ AI ML 온보딩 Assistant                       │
├────────────────────────────────────────────┤
│ Mode: 폐쇄망 POC                             │
│ Engine: Deep Agent                           │
│ UI: Console Wizard + Chat + CLI              │
└────────────────────────────────────────────┘

사용자 모드를 선택하세요.

1. 초급자 모드
   - 단계별 Wizard 방식
   - 선택지만 따라가면 됨
   - 파일 수정 전 자세한 설명 제공

2. 중급자 모드
   - Chat + Wizard 혼합
   - 프로젝트 분석 후 수정안 선택
   - MLflow / Job Template 중심 검증

3. 고급자 모드
   - CLI Command 중심
   - dry-run / apply / validate 직접 실행
   - 자동화 파이프라인 연계 가능"""


BEGINNER_INTRO = """초급자 모드가 선택되었습니다.

이 모드는 AI/ML 프로젝트 등록 절차를 잘 몰라도
단계별 안내에 따라 프로젝트를 점검할 수 있습니다.

먼저 분석할 프로젝트 경로를 입력하세요.
샘플을 만들려면 다음 중 하나를 입력하세요.
- /sample tensorflow
- /sample pytorch
- /sample sklearn
- /sample onnx
- /sample sora
- /sample heavy
- /sample all
- /sample matrix"""


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
