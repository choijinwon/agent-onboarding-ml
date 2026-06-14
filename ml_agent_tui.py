from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
import os
import shlex
from urllib.parse import unquote, urlparse

from app_config import AppConfig

from ml_agent import (
    AppliedChange,
    MODE_CHANGE_MESSAGES,
    MODE_LABELS,
    analyze_project,
    apply_fix_previews,
    build_beginner_step_tabs,
    build_fix_previews,
    format_beginner_apply_result,
    format_beginner_tab,
    parse_mode_command,
    resolve_beginner_project_input,
)
from deepagents_runtime import DeepAgentsRuntime
from qwen_chat import QwenChatConfig


EXIT_COMMANDS = {"/exit", "exit", "quit", "q", "종료"}


class LogView:
    pass


class CommandInput:
    pass


class StatusBar:
    pass


class AIOnboardingTuiApp:
    pass


def textual_available() -> bool:
    return find_spec("textual") is not None


def missing_textual_message() -> str:
    return (
        "Textual TUI를 실행하려면 Textual 패키지가 필요합니다.\n\n"
        "설치 방법:\n"
        '  pip install ".[tui,deepagents]"\n'
        "또는\n"
        "  pip install textual deepagents langchain-openai\n\n"
        "Windows 10/11에서는 Windows Terminal, WezTerm, Alacritty 사용을 권장합니다."
    )


def command_placeholder_for_mode(agent_mode: str, model: str = "qwen3.6") -> str:
    if agent_mode == "Build":
        return f"[Build DeepAgents · {model}] 승인 후 수정 가능 - /path, /model, 질문"
    return f"[Plan DeepAgents · {model}] 읽기 전용 - /path 경로, 드롭/붙여넣기, 다음"


def available_models_from_config(config: AppConfig) -> list[str]:
    models = [model.strip() for model in config.get("QWEN_MODELS").split(",") if model.strip()]
    if config.get("QWEN_MODEL") and config.get("QWEN_MODEL") not in models:
        models.insert(0, config.get("QWEN_MODEL"))
    return models or ["qwen3.6"]


def parse_model_command(command: str) -> str | None:
    parts = command.strip().split()
    if not parts or parts[0] not in {"/model", "/모델"}:
        return None
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def is_fix_request(command: str) -> bool:
    lowered = command.lower()
    keywords = (
        "수정",
        "고쳐",
        "fix",
        "apply",
        "자동",
        "반영",
        "패치",
        "patch",
    )
    return any(keyword in lowered for keyword in keywords)


def is_wizard_navigation(command: str, total: int) -> bool:
    if command in {"다음", "next", "n", "이전", "prev", "previous", "p"}:
        return True
    return command.isdigit() and 1 <= int(command) <= total


def strip_path_command(raw: str) -> tuple[str, bool]:
    value = raw.strip()
    lowered = value.lower()
    for prefix in ("/path", "/경로"):
        if lowered == prefix:
            return "", True
        if lowered.startswith(prefix + " "):
            return value[len(prefix) :].strip(), True
    return value, False


def path_candidates_from_input(raw: str) -> list[str]:
    value, _ = strip_path_command(raw)
    candidates: list[str] = []
    for line in value.replace("\r", "\n").split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        for prefix in (">", "경로:", "path:", "프로젝트:", "project:"):
            if candidate.lower().startswith(prefix.lower()):
                candidate = candidate[len(prefix) :].strip()
        if candidate:
            candidates.append(candidate)
    return candidates or [value.strip()]


def normalize_input_path(raw: str) -> Path | None:
    for candidate in path_candidates_from_input(raw):
        value = candidate.strip().strip('"').strip("'")
        if not value:
            continue
        if value.startswith("file://"):
            parsed = urlparse(value)
            value = unquote(parsed.path or value.removeprefix("file://"))
        else:
            value = unquote(value)
        try:
            parts = shlex.split(value)
        except ValueError:
            parts = []
        if len(parts) == 1:
            value = parts[0]
        expanded = os.path.expandvars(os.path.expanduser(value))
        path = Path(expanded).resolve()
        if path.exists():
            if path.is_file():
                return path.parent
            if path.is_dir():
                return path
    return None


@dataclass
class BeginnerTuiController:
    project_input: str = ""
    agent_mode: str = "Plan"
    index: int = 0
    applied_changes: list[AppliedChange] | None = None
    exited: bool = False
    log_lines: list[str] = field(default_factory=list)
    qwen_config: QwenChatConfig | None = None
    deepagents_runtime: DeepAgentsRuntime | None = None

    def __post_init__(self) -> None:
        self.project_path = ""
        self.sample_message: str | None = None
        self.app_config = AppConfig.load()
        self.available_models = available_models_from_config(self.app_config)
        if self.qwen_config is None:
            self.qwen_config = QwenChatConfig.from_app_config(self.app_config)
        if self.deepagents_runtime is None:
            self.deepagents_runtime = DeepAgentsRuntime(self.app_config)
        self.set_project(self.project_input)
        self.log_lines.extend(
            [
                "# AI ML Onboarding workflow",
                "초급자 Wizard TUI가 시작되었습니다.",
                "하단 Chat 입력창의 모든 요청은 DeepAgents runtime으로 처리합니다.",
            ]
        )
        if self.sample_message:
            self.log_lines.append(self.sample_message)

    @property
    def total(self) -> int:
        return len(self.steps)

    def set_project(self, raw: str) -> None:
        self.project_path, self.sample_message = resolve_beginner_project_input(raw)
        self.applied_changes = None
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        self.index = 0

    def select_project_path(self, path: Path) -> str:
        self.project_path = str(path)
        self.sample_message = None
        self.applied_changes = None
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        self.index = 0
        message = f"프로젝트 경로를 선택했습니다.\n- 위치: {path}"
        self.log_lines.append(message)
        return self.current_screen()

    def current_screen(self) -> str:
        return format_beginner_tab(self.index, len(self.steps), self.steps[self.index])

    def render_log(self) -> str:
        recent = "\n".join(self.log_lines[-12:])
        return f"{recent}\n\n{self.current_screen()}"

    @property
    def qwen_model(self) -> str:
        return self.qwen_config.model if self.qwen_config else "qwen3.6"

    def toggle_agent(self) -> str:
        self.agent_mode = "Build" if self.agent_mode == "Plan" else "Plan"
        return f"현재 Agent 모드: {self.agent_mode}"

    def select_model(self, model: str) -> str:
        if not model:
            message = "선택 가능한 모델: " + ", ".join(self.available_models)
            self.log_lines.append(message)
            return message
        if model not in self.available_models:
            message = f"지원하지 않는 모델입니다: {model}\n선택 가능한 모델: " + ", ".join(self.available_models)
            self.log_lines.append(message)
            return message
        self.qwen_config = replace(self.qwen_config or QwenChatConfig.from_app_config(self.app_config), model=model)
        if self.deepagents_runtime is not None:
            self.deepagents_runtime.qwen_config = self.qwen_config
        message = f"현재 모델이 {model}로 변경되었습니다."
        self.log_lines.append(message)
        return message

    def submit(self, raw: str) -> str:
        command = raw.strip()
        if not command:
            command = "다음"

        if command in EXIT_COMMANDS:
            self.exited = True
            message = "초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다."
            self.log_lines.append(message)
            return message

        mode = parse_mode_command(command)
        if mode:
            message = f"현재 모드가 {MODE_LABELS[mode]}로 변경되었습니다.\n{MODE_CHANGE_MESSAGES[mode]}"
            self.log_lines.append(message)
            return message

        model = parse_model_command(command)
        if model is not None:
            return self.select_model(model)

        path_value, is_path_command = strip_path_command(command)
        if is_path_command and not path_value:
            message = "경로를 함께 입력하세요. 예: /path /Users/me/my-model"
            self.log_lines.append(message)
            return message

        if command.startswith("/sample ") or command.startswith("/샘플 "):
            self.set_project(command)
            message = self.sample_message or "샘플 프로젝트를 선택했습니다."
            self.log_lines.append(message)
            return self.current_screen()

        selected_path = normalize_input_path(command)
        if selected_path is not None:
            return self.select_project_path(selected_path)
        if is_path_command:
            message = f"경로를 찾을 수 없습니다: {path_value}"
            self.log_lines.append(message)
            return message

        if self.index == 3:
            return self._handle_issue_choice(command)
        if self.index == 5 and command in {"1", "2", "3"}:
            return self._handle_approval_choice(command)
        if is_wizard_navigation(command, self.total):
            return self._handle_navigation(command)
        return self._handle_chat(command)

    def _handle_chat(self, command: str) -> str:
        self.log_lines.append(f"나: {command}")
        result = self._invoke_deepagents(command)
        if result.used_deepagents:
            self.log_lines.append(f"DeepAgents {self.qwen_model}: {result.content}")
            return result.content
        if is_fix_request(command):
            return self._handle_chat_fix_request(command, unavailable_reason=result.content)
        self.log_lines.append(f"DeepAgents: {result.content}")
        return result.content

    def _invoke_deepagents(self, command: str):
        runtime = self.deepagents_runtime or DeepAgentsRuntime(self.app_config)
        return runtime.invoke(command, project_path=self.project_path, agent_mode=self.agent_mode)

    def _handle_chat_fix_request(self, command: str, unavailable_reason: str = "") -> str:
        analysis = analyze_project(self.project_path)
        previews = build_fix_previews(analysis)
        if self.agent_mode != "Build":
            message = (
                "DeepAgents runtime이 현재 사용할 수 없어 로컬 dry-run 미리보기만 표시합니다. "
                "Plan 모드라 파일을 수정하지 않았습니다."
            )
            if unavailable_reason:
                message += f"\n사유: {unavailable_reason}"
            if previews:
                preview_titles = ", ".join(preview.title for preview in previews)
                message += f"\n미리보기: {preview_titles}"
            else:
                message += "\n현재 자동 수정할 항목이 없습니다."
            self.log_lines.append(f"DeepAgents: {message}")
            return message
        message = "DeepAgents runtime이 현재 사용할 수 없어 채팅 자동 수정은 실행하지 않았습니다."
        if unavailable_reason:
            message += f"\n사유: {unavailable_reason}"
        if previews:
            preview_titles = ", ".join(preview.title for preview in previews)
            message += f"\n로컬 미리보기: {preview_titles}"
        else:
            message += "\n현재 로컬 기준 자동 수정할 항목도 없습니다."
        self.log_lines.append(f"DeepAgents: {message}")
        return message

    def _handle_issue_choice(self, command: str) -> str:
        if command == "1":
            self.index = 4
            return self.current_screen()
        if command == "2":
            self.index = 0
            return self.current_screen()
        if command == "3":
            self.exited = True
            message = "초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다."
            self.log_lines.append(message)
            return message
        if command in {"다음", "next", "n"}:
            self.index += 1
            return self.current_screen()
        message = "번호로 선택하세요. 1=수정안 미리보기, 2=프로젝트 경로 확인, 3=취소"
        self.log_lines.append(message)
        return self.current_screen()

    def _handle_approval_choice(self, command: str) -> str:
        if command == "1":
            if self.agent_mode != "Build":
                message = "Build 모드에서만 변경을 적용할 수 있습니다. Tab으로 Build 모드로 전환하세요."
                self.log_lines.append(message)
                return message
            analysis = analyze_project(self.project_path)
            previews = build_fix_previews(analysis)
            if not previews:
                self.index += 1
                message = "적용할 수정안이 없습니다. 다음 단계로 이동합니다."
                self.log_lines.append(message)
                return self.current_screen()
            self.applied_changes = apply_fix_previews(self.project_path, previews)
            self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
            result = format_beginner_apply_result(self.applied_changes, analyze_project(self.project_path))
            self.index += 1
            self.log_lines.append(result)
            return self.current_screen()
        if command == "2":
            self.index = 4
            return self.current_screen()
        if command == "3":
            self.exited = True
            message = "초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다."
            self.log_lines.append(message)
            return message
        message = "번호로 선택하세요. 1=승인, 2=다시 보기, 3=취소"
        self.log_lines.append(message)
        return self.current_screen()

    def _handle_navigation(self, command: str) -> str:
        if command in {"다음", "next", "n"}:
            self.index = min(len(self.steps) - 1, self.index + 1)
            return self.current_screen()
        if command in {"이전", "prev", "previous", "p"}:
            self.index = max(0, self.index - 1)
            return self.current_screen()
        if command.isdigit() and 1 <= int(command) <= len(self.steps):
            self.index = int(command) - 1
            return self.current_screen()
        message = "Enter=다음, 이전, 1~10=탭 이동, /exit 중 하나를 입력하세요."
        self.log_lines.append(message)
        return self.current_screen()


def run_tui(project_path: str = "") -> int:
    if not textual_available():
        print(missing_textual_message())
        return 2

    global AIOnboardingTuiApp, CommandInput, LogView, StatusBar

    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import Input, Static

    class LogView(Static):
        pass

    class CommandInput(Input):
        def on_mount(self) -> None:
            self.focus()

        def on_paste(self, event: events.Paste) -> None:
            event.stop()
            self.insert_text_at_cursor(event.text.strip())

        def on_key(self, event) -> None:
            if event.key == "tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app.action_toggle_agent()

    class StatusBar(Static):
        pass

    class AIOnboardingTuiApp(App[None]):
        AUTO_FOCUS = "#command"
        CSS = """
        Screen {
            background: #080808;
            color: #e8e8e8;
        }
        #shell {
            height: 100%;
            padding: 1 2;
            background: #080808;
        }
        #title {
            height: 3;
            padding: 1 2;
            background: #151515;
            border-left: solid #3f3f46;
            text-style: bold;
        }
        #log {
            height: 1fr;
            padding: 1 2;
            background: #080808;
            color: #e8e8e8;
        }
        #command {
            height: 3;
            padding: 0 1;
            background: #202020;
            color: #ffffff;
            border-left: solid #58a6ff;
        }
        #command:focus {
            background: #242424;
            border-left: solid #58a6ff;
        }
        #command.build {
            background: #2a2418;
            border-left: solid #f0b429;
        }
        #command.build:focus {
            background: #302817;
            border-left: solid #f0b429;
        }
        #status {
            height: 1;
            color: #9a9a9a;
            background: #080808;
        }
        """
        BINDINGS = [
            Binding("tab", "toggle_agent", "agents", show=True, priority=True),
            Binding("escape", "quit", "interrupt", show=True),
        ]

        def __init__(self, initial_project_path: str = "") -> None:
            super().__init__()
            self.controller = BeginnerTuiController(initial_project_path)

        def compose(self) -> ComposeResult:
            with Vertical(id="shell"):
                yield Static("AI ML Onboarding Console | ML Platform registration workflow ...", id="title")
                yield LogView("", id="log")
                yield CommandInput(placeholder=command_placeholder_for_mode("Plan", self.controller.qwen_model), id="command")
                yield StatusBar("", id="status")

        def on_mount(self) -> None:
            self._refresh()
            self._focus_command()

        def action_toggle_agent(self) -> None:
            self.controller.toggle_agent()
            self._refresh()
            self._focus_command()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value
            command = self.query_one(CommandInput)
            command.value = ""
            self.controller.submit(value)
            if self.controller.exited:
                self.exit()
                return
            self._refresh()
            self._focus_command()

        def on_click(self) -> None:
            self._focus_command()

        def on_paste(self, event: events.Paste) -> None:
            command = self.query_one(CommandInput)
            if self.focused is command:
                return
            event.stop()
            self._focus_command()
            command.insert_text_at_cursor(event.text.strip())

        def on_key(self, event) -> None:
            command = self.query_one(CommandInput)
            if event.key == "tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.action_toggle_agent()
                return
            if self.focused is command:
                return
            if event.key == "enter":
                event.stop()
                value = command.value
                command.value = ""
                self.controller.submit(value)
                if self.controller.exited:
                    self.exit()
                    return
                self._refresh()
                self._focus_command()
                return
            if event.character:
                event.stop()
                self._focus_command()
                command.insert_text_at_cursor(event.character)

        def _refresh(self) -> None:
            command = self.query_one(CommandInput)
            command.placeholder = command_placeholder_for_mode(self.controller.agent_mode, self.controller.qwen_model)
            command.set_class(self.controller.agent_mode == "Build", "build")
            self.query_one(LogView).update(self.controller.render_log())
            self.query_one(StatusBar).update(
                f"Current: Tab {self.controller.index + 1}/{self.controller.total}  |  "
                f"{self.controller.agent_mode}  |  {self.controller.qwen_model}  |  esc interrupt   tab agents"
            )

        def _focus_command(self) -> None:
            self.set_focus(self.query_one(CommandInput))

    AIOnboardingTuiApp(project_path).run()
    return 0


__all__ = [
    "AIOnboardingTuiApp",
    "BeginnerTuiController",
    "available_models_from_config",
    "CommandInput",
    "LogView",
    "StatusBar",
    "is_fix_request",
    "is_wizard_navigation",
    "normalize_input_path",
    "path_candidates_from_input",
    "parse_model_command",
    "strip_path_command",
    "missing_textual_message",
    "run_tui",
    "textual_available",
]
