from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
import os
import shlex
from urllib.parse import unquote, urlparse

from deep_agent.app_config import AppConfig
from deep_agent.stores.chat_session_store import append_chat_session_event

from deep_agent.cli import (
    AppliedChange,
    MODE_CHANGE_MESSAGES,
    MODE_LABELS,
    analyze_project,
    apply_fix_previews,
    build_beginner_step_tabs,
    build_fix_previews,
    format_beginner_apply_result,
    format_beginner_tab,
    format_beginner_fix_preview,
    parse_mode_command,
    resolve_beginner_project_input,
)
from deep_agent.runtime import DeepAgentsRuntime
from deep_agent.qwen_chat import QwenChatConfig


EXIT_COMMANDS = {"/exit", "exit", "quit", "q", "종료"}
AGENT_MODES = ("Plan", "Build", "Chatbot")
AGENT_MODE_ALIASES = {
    "plan": "Plan",
    "플랜": "Plan",
    "계획": "Plan",
    "build": "Build",
    "빌드": "Build",
    "수정": "Build",
    "chat": "Chatbot",
    "chatbot": "Chatbot",
    "챗봇": "Chatbot",
    "쳇봇": "Chatbot",
    "대화": "Chatbot",
}


class LogView:
    pass


class CommandInput:
    pass


class ModeSelector:
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
    if agent_mode == "Chatbot":
        return f"[Chatbot DeepAgents · {model}] 자연어로 분석/자동수정 요청"
    return f"[Plan DeepAgents · {model}] 읽기 전용 - /path 경로, 드롭/붙여넣기, 다음"


def format_agent_mode_selector(agent_mode: str) -> str:
    parts = []
    for mode in AGENT_MODES:
        parts.append(f"[ {mode}* ]" if mode == agent_mode else f"[ {mode} ]")
    return " ".join(parts)


def parse_agent_mode_command(command: str) -> str | None:
    parts = command.strip().split()
    if not parts or parts[0] not in {"/agent", "/에이전트"}:
        return None
    if len(parts) == 1:
        return ""
    return AGENT_MODE_ALIASES.get(parts[1].strip().lower())


def model_selection_placeholder(models: list[str]) -> str:
    if not models:
        return "[Model Select] 모델명을 입력하세요"
    return f"[Model Select] Tab/화살표 선택, Enter 확정, 1-{len(models)} 번호 가능"


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


def format_model_choices(models: list[str], current_model: str) -> str:
    lines = ["모델을 선택하세요."]
    for index, model in enumerate(models, start=1):
        marker = " (현재)" if model == current_model else ""
        lines.append(f"{index}. {model}{marker}")
    lines.append("번호를 입력하거나 /model <모델명>으로 선택할 수 있습니다.")
    return "\n".join(lines)


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
    awaiting_model_selection: bool = False
    model_selection_index: int = 0

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

    @property
    def highlighted_model(self) -> str:
        if not self.available_models:
            return self.qwen_model
        if self.qwen_model in self.available_models and not self.awaiting_model_selection:
            self.model_selection_index = self.available_models.index(self.qwen_model)
        self.model_selection_index %= len(self.available_models)
        return self.available_models[self.model_selection_index]

    def toggle_agent(self) -> str:
        index = AGENT_MODES.index(self.agent_mode) if self.agent_mode in AGENT_MODES else 0
        self.agent_mode = AGENT_MODES[(index + 1) % len(AGENT_MODES)]
        return f"현재 Agent 모드: {self.agent_mode}"

    def previous_agent(self) -> str:
        index = AGENT_MODES.index(self.agent_mode) if self.agent_mode in AGENT_MODES else 0
        self.agent_mode = AGENT_MODES[(index - 1) % len(AGENT_MODES)]
        return f"현재 Agent 모드: {self.agent_mode}"

    def select_agent_mode(self, mode: str) -> str:
        if mode not in AGENT_MODES:
            message = "지원하지 않는 Agent 모드입니다. Plan, Build, Chatbot 중 하나를 선택하세요."
            self.log_lines.append(message)
            return message
        self.agent_mode = mode
        message = f"현재 Agent 모드: {self.agent_mode}"
        self.log_lines.append(message)
        return message

    def start_model_selection(self) -> str:
        self.awaiting_model_selection = True
        if self.qwen_model in self.available_models:
            self.model_selection_index = self.available_models.index(self.qwen_model)
        message = format_model_choices(self.available_models, self.qwen_model)
        self.log_lines.append(message)
        return message

    def cycle_model_selection(self, delta: int = 1) -> str:
        self.awaiting_model_selection = True
        if self.available_models:
            self.model_selection_index = (self.model_selection_index + delta) % len(self.available_models)
        return self.highlighted_model

    def select_model(self, model: str) -> str:
        if not model:
            if self.awaiting_model_selection:
                model = self.highlighted_model
            else:
                return self.start_model_selection()
        if model.isdigit() and 1 <= int(model) <= len(self.available_models):
            model = self.available_models[int(model) - 1]
        if model not in self.available_models:
            message = f"지원하지 않는 모델입니다: {model}\n선택 가능한 모델: " + ", ".join(self.available_models)
            self.log_lines.append(message)
            return message
        self.qwen_config = replace(self.qwen_config or QwenChatConfig.from_app_config(self.app_config), model=model)
        if self.deepagents_runtime is not None:
            self.deepagents_runtime.qwen_config = self.qwen_config
        self.awaiting_model_selection = False
        message = f"현재 모델이 {model}로 변경되었습니다."
        self.log_lines.append(message)
        return message

    def submit(self, raw: str) -> str:
        command = raw.strip()
        if not command and self.awaiting_model_selection:
            return self.select_model("")
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

        agent_mode = parse_agent_mode_command(command)
        if agent_mode is not None:
            if not agent_mode:
                message = format_agent_mode_selector(self.agent_mode)
                self.log_lines.append(message)
                return message
            return self.select_agent_mode(agent_mode)

        model = parse_model_command(command)
        if model is not None:
            return self.select_model(model)
        if self.awaiting_model_selection:
            return self.select_model(command)

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
        if self.agent_mode != "Chatbot":
            return self._handle_non_chatbot_text(command)
        return self.handle_chat_message(command)

    def handle_chat_message(self, command: str) -> str:
        self.log_lines.append(f"나: {command}")
        result = self._invoke_deepagents(command, agent_mode="AutoFix")
        applied_changes: list[AppliedChange] = []
        final_analysis = analyze_project(self.project_path)
        if result.used_deepagents:
            applied_changes = self._apply_fixable_issues_after_chat()
            final_analysis = analyze_project(self.project_path)
            response = self._format_chatbot_response(result.content, applied_changes, final_analysis)
            self.log_lines.append(f"Agent: {response}")
            self._save_chat_session(command, response, applied_changes, final_analysis)
            return response
        response = f"{result.content}\n파일은 수정하지 않았습니다."
        self.log_lines.append(f"Agent: {response}")
        self._save_chat_session(command, response, applied_changes, final_analysis)
        return response

    def _handle_non_chatbot_text(self, command: str) -> str:
        analysis = analyze_project(self.project_path)
        self.log_lines.append(f"나: {command}")
        if self.agent_mode == "Build":
            message = (
                "Build 모드는 승인된 수정안을 적용하는 모드입니다. "
                "자연어 대화와 자동 수정은 Chatbot 모드에서 실행하세요."
            )
        else:
            message = (
                "Plan 모드는 읽기 전용입니다. 자연어 대화와 자동 수정은 Chatbot 모드에서 실행하세요."
            )
        previews = build_fix_previews(analysis)
        if previews:
            message += "\n현재 미리보기 가능한 수정안: " + ", ".join(preview.title for preview in previews)
        else:
            message += f"\n현재 등록 상태: {analysis.registration_status}"
        self.log_lines.append(f"Agent: {message}")
        return message

    def _invoke_deepagents(self, command: str, agent_mode: str | None = None):
        runtime = self.deepagents_runtime or DeepAgentsRuntime(self.app_config)
        return runtime.invoke(command, project_path=self.project_path, agent_mode=agent_mode or self.agent_mode)

    def _apply_fixable_issues_after_chat(self) -> list[AppliedChange]:
        analysis = analyze_project(self.project_path)
        previews = build_fix_previews(analysis)
        if not previews:
            return []
        self.applied_changes = apply_fix_previews(self.project_path, previews)
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        self.index = min(6, len(self.steps) - 1)
        return self.applied_changes

    def _format_chatbot_response(
        self,
        agent_content: str,
        applied_changes: list[AppliedChange],
        final_analysis,
    ) -> str:
        rows = [agent_content]
        if applied_changes:
            applied_count = len([change for change in applied_changes if change.status == "applied"])
            rows.append("")
            rows.append(f"자동 수정 결과: {applied_count}/{len(applied_changes)}개 적용")
            rows.extend(f"- {change.target}: {change.message}" for change in applied_changes)
            if final_analysis.issue_details:
                rows.append("남은 문제:")
                rows.extend(f"- {issue.title}: {issue.recommendation}" for issue in final_analysis.issue_details[:5])
        else:
            preview = format_beginner_fix_preview(final_analysis)
            rows.append("")
            rows.append("자동 수정 결과: 적용 가능한 항목이 없습니다.")
            if final_analysis.issue_details:
                rows.append("남은 문제:")
                rows.extend(f"- {issue.title}: {issue.recommendation}" for issue in final_analysis.issue_details[:5])
            elif "미리보기 항목: 0개" not in preview:
                rows.append(preview)
        rows.append("")
        rows.append(f"최종 등록 상태: {final_analysis.registration_status}")
        return "\n".join(rows)

    def _save_chat_session(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
        final_analysis,
    ) -> None:
        append_chat_session_event(
            self.app_config,
            {
                "project_path": self.project_path,
                "user_message": user_message,
                "selected_model": self.qwen_model,
                "analysis_status": final_analysis.registration_status,
                "applied_changes": [change.as_dict() for change in applied_changes],
                "agent_response": agent_response,
                "remaining_issues": [issue.as_dict() for issue in final_analysis.issue_details],
            },
        )

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

    global AIOnboardingTuiApp, CommandInput, LogView, ModeSelector, StatusBar

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
            if event.key == "shift+tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app.action_previous_agent()

    class ModeSelector(Static):
        pass

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
        #command.chatbot {
            background: #1c2630;
            border-left: solid #3fb950;
        }
        #command.chatbot:focus {
            background: #203040;
            border-left: solid #3fb950;
        }
        #mode-selector {
            height: 1;
            color: #9a9a9a;
            background: #080808;
        }
        #mode-selector.plan {
            color: #58a6ff;
        }
        #mode-selector.build {
            color: #f0b429;
        }
        #mode-selector.chatbot {
            color: #3fb950;
        }
        #status {
            height: 1;
            color: #9a9a9a;
            background: #080808;
        }
        """
        BINDINGS = [
            Binding("tab", "toggle_agent", "agents", show=True, priority=True),
            Binding("shift+tab", "previous_agent", "prev agent", show=False, priority=True),
            Binding("escape", "quit", "interrupt", show=True),
        ]

        def __init__(self, initial_project_path: str = "") -> None:
            super().__init__()
            self.controller = BeginnerTuiController(initial_project_path)

        def compose(self) -> ComposeResult:
            with Vertical(id="shell"):
                yield Static("AI ML Onboarding Console | ML Platform registration workflow ...", id="title")
                yield LogView("", id="log")
                yield ModeSelector("", id="mode-selector")
                yield CommandInput(placeholder=command_placeholder_for_mode("Plan", self.controller.qwen_model), id="command")
                yield StatusBar("", id="status")

        def on_mount(self) -> None:
            self._refresh()
            self._focus_command()

        def action_toggle_agent(self) -> None:
            if self.controller.awaiting_model_selection:
                self.controller.cycle_model_selection(1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            self.controller.toggle_agent()
            self._refresh()
            self._focus_command()

        def action_previous_agent(self) -> None:
            if self.controller.awaiting_model_selection:
                self.controller.cycle_model_selection(-1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            self.controller.previous_agent()
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
            if self.controller.awaiting_model_selection and event.key in {"up", "left", "shift+tab"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_model_selection(-1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_model_selection and event.key in {"down", "right"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_model_selection(1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            if event.key == "tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.action_toggle_agent()
                return
            if event.key == "shift+tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.action_previous_agent()
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

        def _refresh(self, force_model_value: bool = False) -> None:
            command = self.query_one(CommandInput)
            if self.controller.awaiting_model_selection:
                command.placeholder = model_selection_placeholder(self.controller.available_models)
                if force_model_value or not command.value:
                    command.value = self.controller.highlighted_model
                    command.cursor_position = len(command.value)
            else:
                command.placeholder = command_placeholder_for_mode(self.controller.agent_mode, self.controller.qwen_model)
            command.set_class(self.controller.agent_mode == "Build", "build")
            command.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            self.query_one(LogView).update(self.controller.render_log())
            selector = self.query_one(ModeSelector)
            selector.update(format_agent_mode_selector(self.controller.agent_mode))
            selector.set_class(self.controller.agent_mode == "Plan", "plan")
            selector.set_class(self.controller.agent_mode == "Build", "build")
            selector.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            input_state = "Model Select" if self.controller.awaiting_model_selection else self.controller.agent_mode
            self.query_one(StatusBar).update(
                f"Current: Tab {self.controller.index + 1}/{self.controller.total}  |  "
                f"{input_state}  |  {self.controller.qwen_model}  |  esc interrupt   tab agents"
            )

        def _focus_command(self) -> None:
            self.set_focus(self.query_one(CommandInput))

    AIOnboardingTuiApp(project_path).run()
    return 0


__all__ = [
    "AIOnboardingTuiApp",
    "AGENT_MODES",
    "BeginnerTuiController",
    "available_models_from_config",
    "CommandInput",
    "ModeSelector",
    "LogView",
    "StatusBar",
    "format_agent_mode_selector",
    "format_model_choices",
    "is_fix_request",
    "is_wizard_navigation",
    "model_selection_placeholder",
    "normalize_input_path",
    "path_candidates_from_input",
    "parse_agent_mode_command",
    "parse_model_command",
    "strip_path_command",
    "missing_textual_message",
    "run_tui",
    "textual_available",
]
