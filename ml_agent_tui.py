from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

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
        '  pip install ".[tui]"\n'
        "또는\n"
        "  pip install textual\n\n"
        "Windows 10/11에서는 Windows Terminal, WezTerm, Alacritty 사용을 권장합니다."
    )


@dataclass
class BeginnerTuiController:
    project_input: str = ""
    agent_mode: str = "Plan"
    index: int = 0
    applied_changes: list[AppliedChange] | None = None
    exited: bool = False
    log_lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_path = ""
        self.sample_message: str | None = None
        self.set_project(self.project_input)
        self.log_lines.extend(
            [
                "# AI ML Onboarding workflow",
                "초급자 Wizard TUI가 시작되었습니다.",
                "하단 입력 박스에 프로젝트 경로, /sample tensorflow, 다음, 1 같은 값을 입력하세요.",
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

    def current_screen(self) -> str:
        return format_beginner_tab(self.index, len(self.steps), self.steps[self.index])

    def render_log(self) -> str:
        recent = "\n".join(self.log_lines[-12:])
        return f"{recent}\n\n{self.current_screen()}"

    def toggle_agent(self) -> str:
        self.agent_mode = "Build" if self.agent_mode == "Plan" else "Plan"
        message = f"현재 Agent 모드: {self.agent_mode}"
        self.log_lines.append(message)
        return message

    def submit(self, raw: str) -> str:
        command = raw.strip()
        if not command:
            command = "다음"

        self.log_lines.append(f"> {command}")

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

        if command.startswith("/sample ") or command.startswith("/샘플 "):
            self.set_project(command)
            message = self.sample_message or "샘플 프로젝트를 선택했습니다."
            self.log_lines.append(message)
            return self.current_screen()

        if self.index == 3:
            return self._handle_issue_choice(command)
        if self.index == 5:
            return self._handle_approval_choice(command)
        return self._handle_navigation(command)

    def _handle_issue_choice(self, command: str) -> str:
        if command == "1":
            self.index = 4
            message = "수정안 미리보기 단계로 이동합니다."
        elif command == "2":
            self.index = 0
            message = "프로젝트 경로 확인 단계로 이동합니다."
        elif command == "3":
            self.exited = True
            message = "초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다."
        elif command in {"다음", "next", "n"}:
            self.index += 1
            message = "다음 단계로 이동합니다."
        else:
            message = "번호로 선택하세요. 1=수정안 미리보기, 2=프로젝트 경로 확인, 3=취소"
        self.log_lines.append(message)
        return message if self.exited else self.current_screen()

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
            message = "수정안 미리보기 단계로 돌아갑니다."
        elif command == "3":
            self.exited = True
            message = "초급자 Wizard를 종료합니다. 파일은 추가로 수정하지 않았습니다."
        else:
            message = "번호로 선택하세요. 1=승인, 2=다시 보기, 3=취소"
        self.log_lines.append(message)
        return message if self.exited else self.current_screen()

    def _handle_navigation(self, command: str) -> str:
        if command in {"다음", "next", "n"}:
            self.index = min(len(self.steps) - 1, self.index + 1)
            message = "다음 단계로 이동합니다."
        elif command in {"이전", "prev", "previous", "p"}:
            self.index = max(0, self.index - 1)
            message = "이전 단계로 이동합니다."
        elif command.isdigit() and 1 <= int(command) <= len(self.steps):
            self.index = int(command) - 1
            message = f"Tab {command} 단계로 이동합니다."
        else:
            message = "Enter=다음, 이전, 1~10=탭 이동, /exit 중 하나를 입력하세요."
        self.log_lines.append(message)
        return self.current_screen()


def run_tui(project_path: str = "") -> int:
    if not textual_available():
        print(missing_textual_message())
        return 2

    global AIOnboardingTuiApp, CommandInput, LogView, StatusBar

    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import Input, Static

    class LogView(Static):
        pass

    class CommandInput(Input):
        def on_mount(self) -> None:
            self.focus()

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
                yield CommandInput(placeholder="프로젝트 경로, /sample tensorflow, 다음, 1, /exit", id="command")
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
            self.query_one(LogView).update(self.controller.render_log())
            self.query_one(StatusBar).update(
                f"Current: Tab {self.controller.index + 1}/{self.controller.total}  |  "
                f"{self.controller.agent_mode}  |  esc interrupt   tab agents"
            )

        def _focus_command(self) -> None:
            self.set_focus(self.query_one(CommandInput))

    AIOnboardingTuiApp(project_path).run()
    return 0


__all__ = [
    "AIOnboardingTuiApp",
    "BeginnerTuiController",
    "CommandInput",
    "LogView",
    "StatusBar",
    "missing_textual_message",
    "run_tui",
    "textual_available",
]
