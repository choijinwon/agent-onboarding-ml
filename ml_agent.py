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


@dataclass(frozen=True)
class ProjectAnalysis:
    path: str
    exists: bool
    is_directory: bool
    registration_status: str
    requirements_files: list[str]
    has_mlflow_dependency: bool
    mlflow_usage_files: list[str]
    entrypoint_candidates: list[str]
    model_artifacts: list[str]
    job_template_ready: bool
    issues: list[str]
    next_actions: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_directory": self.is_directory,
            "registration_status": self.registration_status,
            "requirements_files": self.requirements_files,
            "has_mlflow_dependency": self.has_mlflow_dependency,
            "mlflow_usage_files": self.mlflow_usage_files,
            "entrypoint_candidates": self.entrypoint_candidates,
            "model_artifacts": self.model_artifacts,
            "job_template_ready": self.job_template_ready,
            "issues": self.issues,
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
        return payload


class ConsoleAssistant:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
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
        self.output_fn(build_beginner_wizard(project_path))

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


def build_beginner_wizard(project_path: str) -> str:
    display_path = project_path or "(프로젝트 경로 미입력)"
    profile = build_ml_platform_profile(MODE_BEGINNER)
    analysis = analyze_project(project_path)
    return (
        "Step 1. 프로젝트 선택\n"
        f"- 선택된 경로: {display_path}\n\n"
        "Step 2. 프로젝트 자동 스캔\n"
        f"- {profile.subagents[0].name}가 현재는 read-only scan만 수행합니다.\n\n"
        "Step 3. 등록 가능 여부 분석\n"
        f"{format_beginner_analysis(analysis)}\n\n"
        "Step 4. 문제 목록 확인\n"
        f"{format_beginner_issues(analysis)}\n\n"
        "Step 5. 수정안 미리보기\n"
        "- 파일 수정 전 dry-run 결과를 먼저 보여줍니다.\n"
        "- 화면에는 명령어 대신 선택지를 제공합니다.\n\n"
        "Step 6. 사용자 승인\n"
        f"- {profile.approval_policy}\n"
        "- 1. 적용하기\n"
        "- 2. 다시 보기\n"
        "- 3. 취소하기\n\n"
        "Step 7. 파일 생성 또는 수정\n"
        "- 사용자가 '적용하기'를 선택한 경우에만 파일을 생성하거나 수정합니다.\n"
        "- 삭제 작업은 수행하지 않습니다.\n\n"
        "Step 8. 재검증\n"
        "- 적용 후 MLflow / Job Template 검증을 다시 실행합니다.\n\n"
        "Step 9. 분석 리포트 생성\n"
        "- 최종 결과와 다음 조치를 리포트로 남깁니다."
    )


def analyze_project(project_path: str) -> ProjectAnalysis:
    target = Path(project_path or ".")
    display_path = str(target)

    if not target.exists():
        return ProjectAnalysis(
            path=display_path,
            exists=False,
            is_directory=False,
            registration_status="불가",
            requirements_files=[],
            has_mlflow_dependency=False,
            mlflow_usage_files=[],
            entrypoint_candidates=[],
            model_artifacts=[],
            job_template_ready=False,
            issues=["프로젝트 경로를 찾을 수 없습니다."],
            next_actions=["올바른 프로젝트 폴더 경로를 다시 입력하세요."],
        )
    if not target.is_dir():
        return ProjectAnalysis(
            path=display_path,
            exists=True,
            is_directory=False,
            registration_status="불가",
            requirements_files=[],
            has_mlflow_dependency=False,
            mlflow_usage_files=[],
            entrypoint_candidates=[],
            model_artifacts=[],
            job_template_ready=False,
            issues=["선택한 경로가 폴더가 아닙니다."],
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

    issues: list[str] = []
    next_actions: list[str] = []
    if not requirements_files:
        issues.append("requirements.txt 또는 pyproject.toml을 찾지 못했습니다.")
        next_actions.append("학습에 필요한 패키지 목록을 requirements.txt 또는 pyproject.toml에 정리하세요.")
    if requirements_files and not has_mlflow_dependency:
        issues.append("패키지 목록에서 mlflow 의존성을 찾지 못했습니다.")
        next_actions.append("MLflow를 사용한다면 requirements.txt 또는 pyproject.toml에 mlflow를 추가하세요.")
    if not entrypoint_candidates:
        issues.append("학습 시작 파일 후보를 찾지 못했습니다.")
        next_actions.append("train.py 또는 main.py처럼 실행 진입점을 확인할 수 있는 파일을 준비하세요.")
    if not mlflow_usage_files:
        issues.append("학습 코드에서 MLflow 사용 흔적을 찾지 못했습니다.")
        next_actions.append("mlflow.start_run, mlflow.log_metric, mlflow.log_artifact 같은 기록 코드를 확인하세요.")
    if not model_artifacts:
        issues.append("모델 산출물 후보 파일을 찾지 못했습니다.")
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
        registration_status=registration_status,
        requirements_files=[relative_path(path, target) for path in requirements_files],
        has_mlflow_dependency=has_mlflow_dependency,
        mlflow_usage_files=mlflow_usage_files,
        entrypoint_candidates=entrypoint_candidates,
        model_artifacts=model_artifacts,
        job_template_ready=job_template_ready,
        issues=issues,
        next_actions=dedupe(next_actions),
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


def format_beginner_analysis(analysis: ProjectAnalysis) -> str:
    return "\n".join(
        [
            f"- 등록 상태: {analysis.registration_status}",
            f"- 패키지 파일: {format_count(analysis.requirements_files)}",
            f"- MLflow 의존성: {'확인됨' if analysis.has_mlflow_dependency else '미확인'}",
            f"- MLflow 코드 사용: {format_count(analysis.mlflow_usage_files)}",
            f"- 학습 시작 파일 후보: {format_count(analysis.entrypoint_candidates)}",
            f"- 모델 산출물 후보: {format_count(analysis.model_artifacts)}",
            f"- Job Template 초안 준비: {'가능' if analysis.job_template_ready else '보완 필요'}",
        ]
    )


def format_beginner_issues(analysis: ProjectAnalysis) -> str:
    if not analysis.issues:
        return "- 큰 문제를 찾지 못했습니다.\n- 다음 단계에서 수정 없이 미리보기를 확인할 수 있습니다."
    issue_rows = [f"- {issue}" for issue in analysis.issues[:5]]
    if analysis.next_actions:
        issue_rows.append("- 다음 조치: " + analysis.next_actions[0])
    return "\n".join(issue_rows)


def format_count(values: list[str]) -> str:
    if not values:
        return "없음"
    preview = ", ".join(values[:3])
    if len(values) > 3:
        preview += f" 외 {len(values) - 3}개"
    return preview


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
    if parts[0] not in {"analyze", "validate", "fix", "apply", "report"}:
        return "unknown command. available: analyze, validate, fix, apply, report, chat, profile, config, init, prompts, errors"
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

    if command in {"analyze", "validate", "fix", "apply", "report"}:
        details.append(f"path={target}")
        details.append(f"agent_profile={profile.name}")
        details.append(f"registration_status={analysis.registration_status}")
    if command == "fix" and not dry_run:
        details.append("default=dry-run")
        details.append("advanced_apply_required=true")
    if command == "apply":
        details.append("explicit_apply=true")
        details.append("approved changes would be applied in a full implementation")
    if command == "report":
        result_file = str(target / "ml-agent-report.json")
        details.append(f"result_file={result_file}")
    else:
        result_file = None

    details.append(f"mlflow={'ok' if analysis.has_mlflow_dependency or analysis.mlflow_usage_files else 'missing'}")
    details.append(f"job_template={'ready' if analysis.job_template_ready else 'needs_input'}")
    details.append(f"issues={len(analysis.issues)}")
    if analysis.registration_status == "불가":
        status = "error"
        exit_code = 2
    elif command == "validate" and analysis.issues:
        status = "needs_action"
        exit_code = 1
    return CommandResult(command, path, status, exit_code, details, result_file, analysis)


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

    for command in ["analyze", "validate", "fix", "apply", "report"]:
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

먼저 분석할 프로젝트 경로를 입력하세요."""


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
report     분석 리포트 생성
chat       Agent 대화 모드 진입
profile    Deep Agent 프로파일 확인
config     .env 설정 요약
init       런타임/스킬 저장 디렉터리 생성
prompts    저장된 프롬프트 템플릿 확인
errors     에러 로그 저장/목록/분석"""


if __name__ == "__main__":
    raise SystemExit(main())
