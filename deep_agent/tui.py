from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
import os
import re
import shlex
from threading import Thread
from urllib.parse import unquote, urlparse

from deep_agent.app_config import AppConfig
from deep_agent.stores.chat_session_store import append_chat_session_event

from deep_agent.cli import (
    AppliedChange,
    ADVANCED_INTRO,
    INTERMEDIATE_MENU,
    MODE_ADVANCED,
    MODE_CHANGE_MESSAGES,
    MODE_INTERMEDIATE,
    MODE_LABELS,
    MODE_BEGINNER,
    analyze_project,
    apply_fix_previews,
    build_beginner_step_tabs,
    build_fix_previews,
    format_beginner_apply_result,
    format_beginner_tab,
    format_beginner_fix_preview,
    handle_advanced_input,
    handle_intermediate_request,
    parse_mode,
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
    "chbot": "Chatbot",
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
    return ""


def format_agent_mode_selector(agent_mode: str) -> str:
    return "plan build chatbot"


def parse_agent_mode_command(command: str) -> str | None:
    parts = command.strip().split()
    if not parts:
        return None
    if parts[0] in {"/agent", "/에이전트"}:
        if len(parts) == 1:
            return ""
        return AGENT_MODE_ALIASES.get(parts[1].strip().lower().removesuffix("모드"))
    if len(parts) == 1:
        return AGENT_MODE_ALIASES.get(parts[0].strip().lower().removesuffix("모드"))
    return None


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


def parse_folder_command(command: str) -> str | None:
    value = command.strip()
    lowered = value.lower()
    for prefix in ("/folder", "/폴더", "/dir", "/디렉토리"):
        if lowered == prefix:
            return ""
        if lowered.startswith(prefix + " "):
            return value[len(prefix) :].strip()
    return None


def format_model_choices(models: list[str], current_model: str) -> str:
    lines = ["모델을 선택하세요."]
    for index, model in enumerate(models, start=1):
        marker = " (현재)" if model == current_model else ""
        lines.append(f"{index}. {model}{marker}")
    lines.append("번호를 입력하거나 /model <모델명>으로 선택할 수 있습니다.")
    return "\n".join(lines)


def folder_selection_placeholder(folders: list[Path]) -> str:
    if not folders:
        return "[Folder Select] 폴더 경로를 입력하세요"
    return f"[Folder Select] Tab/화살표 선택, Enter 확정, 1-{len(folders)} 번호 가능"


def format_folder_choices(folders: list[Path], current_folder: Path | None = None) -> str:
    lines = ["폴더를 선택하세요."]
    for index, folder in enumerate(folders, start=1):
        marker = " (선택)" if current_folder is not None and folder == current_folder else ""
        lines.append(f"{index}. {folder}{marker}")
    lines.append("번호를 입력하거나 /folder <기준경로>로 후보를 다시 불러올 수 있습니다.")
    return "\n".join(lines)


def format_tui_launch_mode_screen(message: str = "") -> str:
    rows = [
        "AI ML Onboarding Console",
        "",
        "사용자 모드를 선택하세요.",
        "",
        "1. 초급자 모드",
        "   Step 1~10 Wizard",
        "",
        "2. 중급자 모드",
        "   Chat + Wizard",
        "",
        "3. 고급자 모드",
        "   CLI Command",
    ]
    if message:
        rows.extend(["", message])
    return "\n".join(rows)


def format_tui_intermediate_screen(message: str = "") -> str:
    if message:
        return f"{message}\n\n{INTERMEDIATE_MENU}"
    return INTERMEDIATE_MENU


def format_tui_advanced_screen(message: str = "") -> str:
    if message:
        return f"{message}\n\n{ADVANCED_INTRO}"
    return ADVANCED_INTRO


def format_tui_chatbot_screen(project_path: str, model: str, launch_mode: str | None = None) -> str:
    mode_label = MODE_LABELS.get(launch_mode or "", "TUI")
    project_text = project_path or "(프로젝트 경로 미선택)"
    return "\n".join(
        [
            "Chatbot Mode",
            "",
            f"- 실행 모드: {mode_label}",
            f"- 프로젝트: {project_text}",
            f"- 모델: {model}",
            "- 처리 방식: DeepAgents runtime + AutoFix 정책",
            "",
            "input 창에 자연어로 입력하세요.",
            "- 이 프로젝트 분석해줘",
            "- 오류 로그 보고 고쳐줘",
            "- 등록 가능하게 수정해줘",
            "",
            "명령:",
            "- /folder : 폴더 선택",
            "- /model : 모델 선택",
            "- plan 또는 build : 모드 전환",
        ]
    )


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


def is_greeting(command: str) -> bool:
    normalized = command.strip().lower()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "하이",
        "안녕",
        "안녕하세요",
        "반가워",
    }


def greeting_response() -> str:
    return (
        "안녕하세요. AI ML 온보딩 Agent입니다.\n"
        "프로젝트 분석, 오류 로그 점검, MLflow 설정 확인, 수정안 미리보기를 도와드릴 수 있습니다."
    )


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


WINDOWS_DRIVE_PATH_RE = re.compile(r"^/?[A-Za-z]:[\\/]")
WINDOWS_ENV_RE = re.compile(r"%([^%]+)%")
POWERSHELL_ENV_RE = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def is_windows_style_path(value: str) -> bool:
    return bool(WINDOWS_DRIVE_PATH_RE.match(value)) or value.startswith(("\\\\", "//"))


def expand_cross_platform_vars(value: str) -> str:
    def replace_percent(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    def replace_powershell(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    expanded = WINDOWS_ENV_RE.sub(replace_percent, value)
    expanded = POWERSHELL_ENV_RE.sub(replace_powershell, expanded)
    return os.path.expandvars(os.path.expanduser(expanded))


def normalize_path_text(value: str) -> str:
    normalized = value.strip().strip('"').strip("'")
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        normalized = unquote(parsed.path or normalized.removeprefix("file://"))
    else:
        normalized = unquote(normalized)
    if WINDOWS_DRIVE_PATH_RE.match(normalized):
        normalized = normalized.lstrip("/")
    normalized = expand_cross_platform_vars(normalized)
    if not is_windows_style_path(normalized):
        try:
            parts = shlex.split(normalized)
        except ValueError:
            parts = []
        if len(parts) == 1:
            normalized = parts[0]
    return normalized


def normalize_input_path(raw: str) -> Path | None:
    for candidate in path_candidates_from_input(raw):
        value = normalize_path_text(candidate)
        if not value:
            continue
        path = Path(value).resolve()
        if path.exists():
            if path.is_file():
                return path.parent
            if path.is_dir():
                return path
    return None


def folder_has_project_signals(path: Path) -> bool:
    signals = [
        path / "requirements.txt",
        path / "pyproject.toml",
        path / "run_model.py",
        path / "train.py",
        path / "model",
        path / "models",
    ]
    if any(signal.exists() for signal in signals):
        return True
    artifact_patterns = ("*.pkl", "*.joblib", "*.onnx", "*.pt", "*.pth", "*.keras", "*.h5", "*.safetensors")
    return any(next(path.glob(pattern), None) is not None for pattern in artifact_patterns)


def discover_selectable_folders(base: str = "", limit: int = 30) -> list[Path]:
    config = AppConfig.load()
    roots: list[Path] = []
    if base.strip():
        selected = normalize_input_path(base)
        if selected is not None:
            roots.append(selected)
        else:
            normalized = normalize_path_text(base)
            if normalized:
                path = Path(normalized).resolve()
                if path.exists():
                    roots.append(path.parent if path.is_file() else path)
    else:
        roots.extend(
            [
                Path.cwd(),
                Path.cwd() / "work",
                Path.cwd().parent / "work",
                config.root_dir / "work",
                config.root_dir / ".aiu" / "sample_projects",
            ]
        )
    folders: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        candidates = [root] if folder_has_project_signals(root) else []
        try:
            candidates.extend(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        except OSError:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            folders.append(resolved)
            if len(folders) >= limit:
                return folders
    return folders


@dataclass
class BeginnerTuiController:
    project_input: str = ""
    selected_launch_mode: str | None = None
    agent_mode: str = "Plan"
    index: int = 0
    applied_changes: list[AppliedChange] | None = None
    exited: bool = False
    log_lines: list[str] = field(default_factory=list)
    qwen_config: QwenChatConfig | None = None
    deepagents_runtime: DeepAgentsRuntime | None = None
    awaiting_model_selection: bool = False
    model_selection_index: int = 0
    awaiting_folder_selection: bool = False
    folder_selection_index: int = 0
    folder_options: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_path = ""
        self.sample_message: str | None = None
        self.app_config = AppConfig.load()
        self.available_models = available_models_from_config(self.app_config)
        if self.qwen_config is None:
            self.qwen_config = QwenChatConfig.from_app_config(self.app_config)
        if self.deepagents_runtime is None:
            self.deepagents_runtime = DeepAgentsRuntime(self.app_config)
        self.steps: list[str] = []
        self.latest_message = ""
        if self.selected_launch_mode:
            self.activate_launch_mode(self.selected_launch_mode)

    @property
    def total(self) -> int:
        return len(self.steps) if self.steps else 0

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
        message = ""
        self.latest_message = message
        return self.current_screen()

    def current_screen(self) -> str:
        if self.selected_launch_mode is None:
            return format_tui_launch_mode_screen(self.latest_message)
        if self.agent_mode == "Chatbot":
            return format_tui_chatbot_screen(self.project_path, self.qwen_model, self.selected_launch_mode)
        if self.selected_launch_mode == MODE_INTERMEDIATE:
            return format_tui_intermediate_screen(self.latest_message)
        if self.selected_launch_mode == MODE_ADVANCED:
            return format_tui_advanced_screen(self.latest_message)
        return format_beginner_tab(self.index, len(self.steps), self.steps[self.index])

    def render_log(self) -> str:
        log_text = "\n\n".join(self.log_lines[-12:])
        if self.agent_mode == "Chatbot" and self.selected_launch_mode is not None:
            screen = self.current_screen()
            if log_text:
                return f"{log_text}\n\n{screen}"
            if self.latest_message and self.latest_message != screen:
                return f"{self.latest_message}\n\n{screen}"
            return screen
        if self.selected_launch_mode in {MODE_INTERMEDIATE, MODE_ADVANCED}:
            screen = self.current_screen()
            return f"{log_text}\n\n{screen}" if log_text else screen
        if self.selected_launch_mode == MODE_BEGINNER and self.latest_message:
            parts = [part for part in [log_text, self.latest_message, self.current_screen()] if part]
            return "\n\n".join(parts)
        if log_text:
            return f"{log_text}\n\n{self.current_screen()}"
        return self.current_screen()

    def activate_launch_mode(self, mode: str) -> str:
        self.selected_launch_mode = mode
        self.latest_message = ""
        if mode == MODE_BEGINNER:
            self.agent_mode = "Plan"
            self.set_project(self.project_input)
            if self.sample_message:
                self.latest_message = self.sample_message
            return self.current_screen()
        if mode == MODE_INTERMEDIATE:
            self.agent_mode = "Chatbot"
            return self.current_screen()
        if mode == MODE_ADVANCED:
            self.agent_mode = "Build"
            return self.current_screen()
        self.selected_launch_mode = None
        self.latest_message = "지원하지 않는 모드입니다. 1, 2, 3 중 하나를 선택하세요."
        return self.current_screen()

    def should_show_thinking(self, raw: str) -> bool:
        command = raw.strip()
        if not command or command in EXIT_COMMANDS or self.awaiting_model_selection:
            return False
        if self.selected_launch_mode is None:
            return False
        if self.selected_launch_mode == MODE_ADVANCED and self.agent_mode != "Chatbot":
            return False
        if (
            parse_mode_command(command)
            or parse_agent_mode_command(command) is not None
            or parse_model_command(command) is not None
            or parse_folder_command(command) is not None
        ):
            return False
        if self.selected_launch_mode in {MODE_INTERMEDIATE, MODE_ADVANCED}:
            return self.agent_mode == "Chatbot"
        if self.selected_launch_mode != MODE_BEGINNER or self.agent_mode != "Chatbot":
            return False
        path_value, is_path_command = strip_path_command(command)
        if is_path_command or command.startswith("/sample ") or command.startswith("/샘플 "):
            return False
        if is_wizard_navigation(command, self.total):
            return False
        return True

    def set_thinking(self, raw: str) -> None:
        command = raw.strip()
        self.latest_message = f"생각중...\n\n나: {command}"
        self._append_or_replace_chat_log(command, "생각중...", [])

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
            self.latest_message = message
            return message
        self.agent_mode = mode
        message = self.current_screen() if mode == "Chatbot" else ""
        self.latest_message = message
        return message

    def start_model_selection(self) -> str:
        self.awaiting_model_selection = True
        if self.qwen_model in self.available_models:
            self.model_selection_index = self.available_models.index(self.qwen_model)
        message = format_model_choices(self.available_models, self.qwen_model)
        self.latest_message = message
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
            self.latest_message = message
            return message
        self.qwen_config = replace(self.qwen_config or QwenChatConfig.from_app_config(self.app_config), model=model)
        if self.deepagents_runtime is not None:
            self.deepagents_runtime.qwen_config = self.qwen_config
        self.awaiting_model_selection = False
        message = ""
        self.latest_message = message
        return message

    @property
    def highlighted_folder(self) -> Path | None:
        if not self.folder_options:
            return None
        self.folder_selection_index %= len(self.folder_options)
        return self.folder_options[self.folder_selection_index]

    def start_folder_selection(self, base: str = "") -> str:
        self.folder_options = discover_selectable_folders(base)
        self.folder_selection_index = 0
        self.awaiting_folder_selection = True
        if not self.folder_options:
            message = "선택 가능한 폴더를 찾지 못했습니다. /folder <상위폴더경로> 형태로 다시 입력하세요."
            self.latest_message = message
            return message
        message = format_folder_choices(self.folder_options, self.highlighted_folder)
        self.latest_message = message
        return message

    def cycle_folder_selection(self, delta: int = 1) -> str:
        self.awaiting_folder_selection = True
        if self.folder_options:
            self.folder_selection_index = (self.folder_selection_index + delta) % len(self.folder_options)
        highlighted = self.highlighted_folder
        return str(highlighted) if highlighted else ""

    def select_folder(self, value: str) -> str:
        candidate = value.strip()
        selected: Path | None = None
        if not candidate:
            selected = self.highlighted_folder
        elif candidate.isdigit() and 1 <= int(candidate) <= len(self.folder_options):
            selected = self.folder_options[int(candidate) - 1]
        else:
            selected = normalize_input_path(candidate)
        if selected is None:
            message = "폴더를 선택하지 못했습니다. 번호를 입력하거나 폴더 경로를 입력하세요."
            self.latest_message = message
            return message
        self.awaiting_folder_selection = False
        self.folder_options = []
        return self.select_project_path(selected)

    def submit(self, raw: str) -> str:
        command = raw.strip()
        if self.selected_launch_mode is None:
            if command in EXIT_COMMANDS:
                self.exited = True
                return ""
            mode = parse_mode(command)
            if mode:
                return self.activate_launch_mode(mode)
            self.latest_message = "먼저 모드를 선택하세요. 1=초급자, 2=중급자, 3=고급자"
            return self.current_screen()

        if self.selected_launch_mode == MODE_INTERMEDIATE:
            return self._submit_intermediate(command)
        if self.selected_launch_mode == MODE_ADVANCED:
            return self._submit_advanced(command)

        if not command and self.awaiting_folder_selection:
            return self.select_folder("")
        if not command and self.awaiting_model_selection:
            return self.select_model("")
        if not command:
            command = "다음"

        if command in EXIT_COMMANDS:
            self.exited = True
            message = ""
            self.latest_message = message
            return message

        mode = parse_mode_command(command)
        if mode:
            message = f"현재 모드가 {MODE_LABELS[mode]}로 변경되었습니다.\n{MODE_CHANGE_MESSAGES[mode]}"
            self.latest_message = message
            return message

        agent_mode = parse_agent_mode_command(command)
        if agent_mode is not None:
            if not agent_mode:
                message = format_agent_mode_selector(self.agent_mode)
                self.latest_message = message
                return message
            return self.select_agent_mode(agent_mode)

        model = parse_model_command(command)
        if model is not None:
            return self.select_model(model)
        if self.awaiting_model_selection:
            return self.select_model(command)

        folder_base = parse_folder_command(command)
        if folder_base is not None:
            return self.start_folder_selection(folder_base)
        if self.awaiting_folder_selection:
            return self.select_folder(command)

        path_value, is_path_command = strip_path_command(command)
        if is_path_command and not path_value:
            message = "경로를 함께 입력하세요. 예: /path /Users/me/my-model"
            self.latest_message = message
            return message

        if command.startswith("/sample ") or command.startswith("/샘플 "):
            self.set_project(command)
            message = self.sample_message or ""
            self.latest_message = message
            return self.current_screen()

        selected_path = normalize_input_path(command)
        if selected_path is not None:
            return self.select_project_path(selected_path)
        if is_path_command:
            message = f"경로를 찾을 수 없습니다: {path_value}"
            self.latest_message = message
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

    def _submit_intermediate(self, command: str) -> str:
        if command in EXIT_COMMANDS:
            self.exited = True
            return ""
        mode = parse_mode_command(command)
        if mode:
            return self.activate_launch_mode(mode)
        agent_mode = parse_agent_mode_command(command)
        if agent_mode is not None:
            if not agent_mode:
                self.latest_message = format_agent_mode_selector(self.agent_mode)
                return self.current_screen()
            self.select_agent_mode(agent_mode)
            return self.current_screen()
        if self.agent_mode == "Chatbot":
            response = self.handle_chat_message(command)
            return self.render_log() if response else response
        if is_greeting(command):
            self.latest_message = greeting_response()
            return self.current_screen()
        self.latest_message = handle_intermediate_request(command)
        return self.current_screen()

    def _submit_advanced(self, command: str) -> str:
        if command in EXIT_COMMANDS:
            self.exited = True
            return ""
        mode = parse_mode_command(command)
        if mode:
            return self.activate_launch_mode(mode)
        agent_mode = parse_agent_mode_command(command)
        if agent_mode is not None:
            if not agent_mode:
                self.latest_message = format_agent_mode_selector(self.agent_mode)
                return self.current_screen()
            self.select_agent_mode(agent_mode)
            return self.current_screen()
        if self.agent_mode == "Chatbot":
            response = self.handle_chat_message(command)
            return self.render_log() if response else response
        self.latest_message = handle_advanced_input(command)
        return self.current_screen()

    def handle_chat_message(self, command: str) -> str:
        if is_greeting(command):
            response = greeting_response()
            self.latest_message = response
            final_analysis = analyze_project(self.project_path)
            self._append_or_replace_chat_log(command, response, [])
            self._save_chat_session(command, response, [], final_analysis)
            return response
        result = self._invoke_deepagents(command, agent_mode="AutoFix")
        applied_changes: list[AppliedChange] = []
        final_analysis = analyze_project(self.project_path)
        if result.used_deepagents:
            applied_changes = self._apply_fixable_issues_after_chat()
            final_analysis = analyze_project(self.project_path)
            response = self._format_chatbot_response(result.content, applied_changes, final_analysis)
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, applied_changes)
            self._save_chat_session(command, response, applied_changes, final_analysis)
            return response
        response = f"{result.content}\n파일은 수정하지 않았습니다."
        self.latest_message = response
        self._append_or_replace_chat_log(command, response, applied_changes)
        self._save_chat_session(command, response, applied_changes, final_analysis)
        return response

    def _handle_non_chatbot_text(self, command: str) -> str:
        return self.current_screen()

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

    def _append_chat_log(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
    ) -> None:
        self.log_lines.append(self._format_chat_log(user_message, agent_response, applied_changes))

    def _append_or_replace_chat_log(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
    ) -> None:
        prefix = f"나: {user_message}\nAgent: "
        replacement = self._format_chat_log(user_message, agent_response, applied_changes)
        for index in range(len(self.log_lines) - 1, -1, -1):
            if self.log_lines[index].startswith(prefix):
                self.log_lines[index] = replacement
                return
        self.log_lines.append(replacement)

    def _append_chat_error(self, user_message: str, error: Exception) -> None:
        response = f"Chatbot 응답 처리 중 오류가 발생했습니다: {error}"
        self.latest_message = response
        self._append_or_replace_chat_log(user_message, response, [])

    def _format_chat_log(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
    ) -> str:
        rows = [f"나: {user_message}", f"Agent: {agent_response}"]
        if applied_changes:
            rows.append("수정사항:")
            rows.extend(f"- {change.target}: {change.message}" for change in applied_changes)
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
            self.latest_message = message
            return message
        message = "DeepAgents runtime이 현재 사용할 수 없어 채팅 자동 수정은 실행하지 않았습니다."
        if unavailable_reason:
            message += f"\n사유: {unavailable_reason}"
        if previews:
            preview_titles = ", ".join(preview.title for preview in previews)
            message += f"\n로컬 미리보기: {preview_titles}"
        else:
            message += "\n현재 로컬 기준 자동 수정할 항목도 없습니다."
        self.latest_message = message
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
            message = ""
            self.latest_message = message
            return message
        if command in {"다음", "next", "n"}:
            self.index += 1
            return self.current_screen()
        message = "번호로 선택하세요. 1=수정안 미리보기, 2=프로젝트 경로 확인, 3=취소"
        self.latest_message = message
        return message

    def _handle_approval_choice(self, command: str) -> str:
        if command == "1":
            self.agent_mode = "Build"
            analysis = analyze_project(self.project_path)
            previews = build_fix_previews(analysis)
            if not previews:
                self.index += 1
                message = ""
                self.latest_message = message
                return self.current_screen()
            self.applied_changes = apply_fix_previews(self.project_path, previews)
            self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
            result = format_beginner_apply_result(self.applied_changes, analyze_project(self.project_path))
            self.index += 1
            self.latest_message = result
            return self.current_screen()
        if command == "2":
            self.index = 4
            return self.current_screen()
        if command == "3":
            self.exited = True
            message = ""
            self.latest_message = message
            return message
        message = "번호로 선택하세요. 1=승인, 2=다시 보기, 3=취소"
        self.latest_message = message
        return message

    def _handle_navigation(self, command: str) -> str:
        if command in {"다음", "next", "n"}:
            self.index = 0 if self.index >= len(self.steps) - 1 else self.index + 1
            return self.current_screen()
        if command in {"이전", "prev", "previous", "p"}:
            self.index = max(0, self.index - 1)
            return self.current_screen()
        if command.isdigit() and 1 <= int(command) <= len(self.steps):
            self.index = int(command) - 1
            return self.current_screen()
        message = "Enter=다음, 이전, 1~10=탭 이동, /exit 중 하나를 입력하세요."
        self.latest_message = message
        return message


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
                return
            if event.key == "shift+tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app.action_previous_agent()
                return
            if event.key == "ctrl+space":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.insert_text_at_cursor("    ")
                return

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
            text-style: bold;
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
            display: none;
            height: 0;
            color: #9a9a9a;
            background: #080808;
        }
        """
        BINDINGS = [
            Binding("tab", "toggle_agent", "agents", show=True, priority=True),
            Binding("shift+tab", "previous_agent", "prev agent", show=False, priority=True),
            Binding("ctrl+space", "insert_input_gap", "input gap", show=False, priority=True),
            Binding("escape", "quit", "interrupt", show=True),
        ]

        def __init__(self, initial_project_path: str = "") -> None:
            super().__init__()
            self.controller = BeginnerTuiController(initial_project_path)
            self._chatbot_busy = False

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
            if self.controller.selected_launch_mode is None:
                self.controller.latest_message = "먼저 모드를 선택하세요. 1=초급자, 2=중급자, 3=고급자"
                self._refresh()
                self._focus_command()
                return
            if self.controller.awaiting_model_selection:
                self.controller.cycle_model_selection(1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_folder_selection:
                self.controller.cycle_folder_selection(1)
                self._refresh(force_folder_value=True)
                self._focus_command()
                return
            self.controller.toggle_agent()
            self._refresh()
            self._focus_command()

        def action_previous_agent(self) -> None:
            if self.controller.selected_launch_mode is None:
                self.controller.latest_message = "먼저 모드를 선택하세요. 1=초급자, 2=중급자, 3=고급자"
                self._refresh()
                self._focus_command()
                return
            if self.controller.awaiting_model_selection:
                self.controller.cycle_model_selection(-1)
                self._refresh(force_model_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_folder_selection:
                self.controller.cycle_folder_selection(-1)
                self._refresh(force_folder_value=True)
                self._focus_command()
                return
            self.controller.previous_agent()
            self._refresh()
            self._focus_command()

        def action_insert_input_gap(self) -> None:
            command = self.query_one(CommandInput)
            self.set_focus(command)
            command.insert_text_at_cursor("    ")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value
            command = self.query_one(CommandInput)
            command.value = ""
            self._submit_or_queue(value)

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
            if self.controller.awaiting_folder_selection and event.key in {"up", "left", "shift+tab"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_folder_selection(-1)
                self._refresh(force_folder_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_folder_selection and event.key in {"down", "right"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_folder_selection(1)
                self._refresh(force_folder_value=True)
                self._focus_command()
                return
            if event.key == "ctrl+space":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                command.insert_text_at_cursor("    ")
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
                self._submit_or_queue(value)
                return

        def _submit_or_queue(self, value: str) -> None:
            if self.controller.should_show_thinking(value):
                if self._chatbot_busy:
                    self.controller.latest_message = "이전 Chatbot 요청을 처리 중입니다. 잠시만 기다려 주세요."
                    self._refresh()
                    self._focus_command()
                    return
                self._chatbot_busy = True
                self.controller.set_thinking(value)
                self._refresh()
                self._focus_command()
                self.set_timer(0.05, lambda: self._start_submit_worker(value), name="chatbot-submit")
                return
            self._submit_value(value)

        def _start_submit_worker(self, value: str) -> None:
            Thread(target=self._submit_value_in_thread, args=(value,), daemon=True).start()

        def _submit_value_in_thread(self, value: str) -> None:
            try:
                self.controller.submit(value)
            except Exception as exc:  # pragma: no cover - UI safety boundary
                self.controller._append_chat_error(value, exc)
            self.call_from_thread(self._finish_submit)

        def _finish_submit(self) -> None:
            self._chatbot_busy = False
            if self.controller.exited:
                self.exit()
                return
            self._refresh()
            self._focus_command()

        def _submit_value(self, value: str) -> None:
            self.controller.submit(value)
            if self.controller.exited:
                self.exit()
                return
            self._refresh()
            self._focus_command()

        def _refresh(self, force_model_value: bool = False, force_folder_value: bool = False) -> None:
            command = self.query_one(CommandInput)
            if self.controller.awaiting_model_selection:
                command.placeholder = model_selection_placeholder(self.controller.available_models)
                if force_model_value or not command.value:
                    command.value = self.controller.highlighted_model
                    command.cursor_position = len(command.value)
            elif self.controller.awaiting_folder_selection:
                command.placeholder = folder_selection_placeholder(self.controller.folder_options)
                highlighted = self.controller.highlighted_folder
                if highlighted is not None and (force_folder_value or not command.value):
                    command.value = str(highlighted)
                    command.cursor_position = len(command.value)
            else:
                command.placeholder = command_placeholder_for_mode(self.controller.agent_mode, self.controller.qwen_model)
            command.set_class(self.controller.agent_mode == "Build", "build")
            command.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            self.query_one(LogView).update(self.controller.render_log())
            selector = self.query_one(ModeSelector)
            if self.controller.selected_launch_mode is None:
                selector.update("beginner intermediate advanced")
            else:
                selector.update(format_agent_mode_selector(self.controller.agent_mode))
            selector.set_class(self.controller.agent_mode == "Plan", "plan")
            selector.set_class(self.controller.agent_mode == "Build", "build")
            selector.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            self.query_one(StatusBar).update("")

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
    "discover_selectable_folders",
    "folder_selection_placeholder",
    "format_folder_choices",
    "ModeSelector",
    "LogView",
    "StatusBar",
    "format_agent_mode_selector",
    "format_model_choices",
    "format_tui_chatbot_screen",
    "is_fix_request",
    "is_greeting",
    "is_wizard_navigation",
    "model_selection_placeholder",
    "normalize_input_path",
    "normalize_path_text",
    "path_candidates_from_input",
    "parse_agent_mode_command",
    "parse_folder_command",
    "parse_model_command",
    "strip_path_command",
    "missing_textual_message",
    "run_tui",
    "textual_available",
]
