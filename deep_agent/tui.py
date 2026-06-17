from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
import re
import textwrap
from threading import Thread
import time
import unicodedata

from deep_agent.app_config import AppConfig
from deep_agent.path_utils import (
    existing_filesystem_path,
    expand_cross_platform_vars,
    is_windows_style_path,
    normalize_path_text,
)
from deep_agent.stores.chat_session_store import (
    append_chat_session_event,
    load_chat_context_summary,
    save_chat_context_summary,
)
from deep_agent.stores.prompt_store import (
    append_used_prompt_to_wiki,
    export_prompt_templates_to_wiki,
    format_wiki_recent_prompt_for_tui,
    format_wiki_today_for_tui,
)

from deep_agent.cli import (
    AppliedChange,
    ADVANCED_INTRO,
    FixPreview,
    INTERMEDIATE_MENU,
    MODE_ADVANCED,
    MODE_CHANGE_MESSAGES,
    MODE_INTERMEDIATE,
    MODE_LABELS,
    MODE_BEGINNER,
    analyze_project,
    apply_fix_previews,
    apply_serving_fix_previews,
    build_beginner_intro,
    build_beginner_step_tabs,
    build_fix_previews,
    build_serving_fix_previews,
    format_bytes,
    format_beginner_apply_result,
    format_beginner_tab,
    format_beginner_fix_preview,
    format_serving_apply_result,
    format_model_parameters,
    handle_advanced_input,
    handle_intermediate_request,
    run_beginner_mlflow_verification,
    ensure_standard_ml_dl_template,
    list_existing_work_projects,
    copy_text_to_clipboard,
    normalize_standard_framework,
    normalize_clipboard_text,
    repair_clipboard_mojibake,
    parse_mode,
    parse_mode_command,
    resolve_beginner_project_input,
)
from deep_agent.runtime import DeepAgentsRuntime, build_deepagents_system_prompt
from deep_agent.qwen_chat import QwenChatConfig, chat_with_qwen


EXIT_COMMANDS = {"/exit", "exit", "quit", "q", "종료"}
HELP_COMMANDS = {"/help", "help", "도움말", "/도움말", "?"}
COMPACT_COMMANDS = {"/compact", "/요약", "/context compact", "context compact", "컨텍스트 압축"}
WIKI_COMMANDS = {"/wiki", "/위키", "wiki", "위키"}
WIKI_LAST_COMMANDS = {"/wiki last", "/위키 최근", "wiki last", "위키 최근"}
AGENT_MODES = ("Plan", "Build", "Chatbot")
AGENT_MODE_DISPLAY = {
    "Plan": "PLAN",
    "Build": "BUILD",
    "Chatbot": "CHAT",
}
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
AI_STUDIO_ENV_FIELDS = [
    ("MLFLOW_TRACKING_URL", "MLflow Tracking URL"),
    ("MLFLOW_TRACKING_USERNAME", "MLflow username"),
    ("MLFLOW_TRACKING_PASSWORD", "MLflow password"),
    ("MLFLOW_EXPERIMENT_NAME", "MLflow experiment name"),
    ("MLFLOW_REGISTER_MODEL_NAME", "MLflow registered model name"),
]
PASTE_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")
PASTE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
AGENT_RESPONSE_CHOICE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[(\d+)\]|(\d+)[.)]|(\d+)\s*[-:])\s+(.+?)\s*$")
LONG_PASTE_CHAR_LIMIT = 12000
LONG_PASTE_LINE_LIMIT = 250
PASTE_DEDUP_SECONDS = 0.75
MAX_CHAT_LOG_ENTRIES = 8
MAX_CHAT_RENDER_CHARS = 7000
MAX_CHAT_RENDER_LINES = 180
DEFAULT_CONTEXT_COMPACT_AFTER = 12
DEFAULT_CONTEXT_RECENT_MESSAGES = 4
DEFAULT_CONTEXT_MAX_CHARS = 6000
MAX_TUI_RENDER_CHARS = 24000
SAFE_CHAT_FIX_CODES = {
    "CREATE_REQUIREMENTS",
    "ADD_MLFLOW_DEPENDENCY",
    "CREATE_AI_STUDIO_MLFLOW_SCAFFOLD",
}
REVIEW_REQUIRED_CHAT_FIX_CODES = {
    "ADD_MLFLOW_TRACKING_SNIPPET",
}


class LogView:
    pass


class CommandInput:
    pass


class ModeSelector:
    pass


class StatusBar:
    pass


class SendButton:
    pass


class FileButton:
    pass


class SampleButton:
    pass


class CancelButton:
    pass


class ClearButton:
    pass


class MultiAgentButton:
    pass


class RunModelButton:
    pass


class AIOnboardingTuiApp:
    pass


def textual_available() -> bool:
    return find_spec("textual") is not None


def is_right_click_event(event) -> bool:
    button = getattr(event, "button", None)
    button_name = str(getattr(event, "button_name", "")).lower()
    return button == 3 or button_name == "right"


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


def format_agent_mode_label(agent_mode: str) -> str:
    return AGENT_MODE_DISPLAY.get(agent_mode, agent_mode.upper())


def format_agent_mode_selector(agent_mode: str) -> str:
    labels = []
    for mode in AGENT_MODES:
        label = format_agent_mode_label(mode)
        labels.append(f"[{label}]" if mode == agent_mode else label)
    return " | ".join(labels)


def format_agent_mode_title(agent_mode: str) -> str:
    return f"{format_agent_mode_label(agent_mode)} MODE"


def format_chat_card(
    user_message: str,
    agent_response: str,
    applied_changes: list[AppliedChange] | None = None,
) -> str:
    changes = applied_changes or []
    width = 76
    divider = "  " + "-" * width
    display_user_message = truncate_for_tui(user_message, max_chars=900, max_lines=18)
    display_agent_response = truncate_for_tui(
        agent_response,
        max_chars=MAX_CHAT_RENDER_CHARS,
        max_lines=MAX_CHAT_RENDER_LINES,
    )
    rows = [
        divider,
        f"  YOU    {display_user_message}",
        "",
        "  AGENT  response",
        *indent_block(display_agent_response, "         "),
    ]
    if changes:
        rows.extend(
            [
                "",
                "  BUILD  changes",
                *[f"         - {change.target}: {change.message}" for change in changes],
            ]
        )
    rows.append(divider)
    return "\n".join(rows)


def format_thinking_animation(elapsed_seconds: int) -> str:
    elapsed = max(0, int(elapsed_seconds))
    width = 8
    fill = (elapsed % width) + 1
    bar = "=" * fill + " " * (width - fill)
    return f"AI thinking [{bar}] {elapsed}s"


def truncate_for_tui(text: str, *, max_chars: int, max_lines: int) -> str:
    if len(text) <= max_chars and text.count("\n") < max_lines:
        return text
    lines = text.splitlines()
    clipped_by_lines = len(lines) > max_lines
    if clipped_by_lines:
        text = "\n".join(lines[:max_lines])
    clipped_by_chars = len(text) > max_chars
    if clipped_by_chars:
        text = text[:max_chars].rstrip()
    omitted_lines = max(0, len(lines) - max_lines) if clipped_by_lines else 0
    note = "... 화면 성능을 위해 일부 응답을 접었습니다. 전체 내용은 sessions/wiki 로그에 저장됩니다."
    if omitted_lines:
        note += f" (숨긴 줄: {omitted_lines})"
    return f"{text}\n{note}"


def compact_chat_entries(
    entries: list[dict[str, str]],
    previous_summary: str = "",
    *,
    max_chars: int = 2400,
) -> str:
    rows: list[str] = []
    if previous_summary:
        rows.append(previous_summary.strip())
    for entry in entries:
        user_message = normalize_context_line(entry.get("user_message", ""))
        agent_response = normalize_context_line(entry.get("agent_response", ""))
        if user_message:
            rows.append(f"사용자: {user_message}")
        if agent_response:
            rows.append(f"Agent: {agent_response}")
    summary = "\n".join(row for row in rows if row).strip()
    if len(summary) > max_chars:
        summary = summary[-max_chars:].lstrip()
        summary = "이전 대화 요약 일부 생략...\n" + summary
    return summary


def normalize_context_line(text: str, limit: int = 360) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= limit:
        return compacted
    return compacted[:limit].rstrip() + "..."


def build_compacted_runtime_prompt(
    command: str,
    summary: str,
    recent_entries: list[dict[str, str]],
    *,
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> str:
    if not summary and not recent_entries:
        return command
    rows = [
        "아래는 이전 대화 컨텍스트입니다. 반복 설명은 줄이고 현재 요청에 바로 답하세요.",
    ]
    if summary:
        rows.extend(["", "[압축 요약]", summary.strip()])
    if recent_entries:
        rows.append("")
        rows.append("[최근 대화]")
        for entry in recent_entries:
            user_message = normalize_context_line(entry.get("user_message", ""))
            agent_response = normalize_context_line(entry.get("agent_response", ""))
            if user_message:
                rows.append(f"- 사용자: {user_message}")
            if agent_response:
                rows.append(f"  Agent: {agent_response}")
    rows.extend(["", "[현재 요청]", command])
    prompt = "\n".join(rows)
    if len(prompt) > max_chars:
        return prompt[-max_chars:].lstrip()
    return prompt


def trim_tui_render_text(text: str, max_chars: int = MAX_TUI_RENDER_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return (
        "... 이전 화면 일부를 생략했습니다. 전체 대화는 sessions/wiki 로그에 저장됩니다.\n\n"
        + text[-max_chars:]
    )


def indent_block(text: str, prefix: str) -> list[str]:
    lines = str(text).splitlines() or [""]
    return [f"{prefix}{line}" if line else prefix.rstrip() for line in lines]


def format_tui_help_screen(
    launch_mode: str | None = None,
    agent_mode: str = "Plan",
    project_path: str = "",
    model: str = "qwen3.6",
) -> str:
    mode_label = MODE_LABELS.get(launch_mode or "", "모드 미선택")
    project_text = project_path or "(프로젝트 경로 미선택)"
    rows = [
        "  ----------------------------------------------------------------------------",
        "  HELP   AI ML Onboarding Console",
        "",
        f"  MODE   {mode_label} / {format_agent_mode_selector(agent_mode)}",
        f"  MODEL  {model}",
        f"  PATH   {project_text}",
        "",
        "  BASIC",
        "         /help                  도움말 표시",
        "         /exit                  종료",
        "         Enter                  입력 제출",
        "         Shift+Enter            input 줄바꿈",
        "         Ctrl+Enter             입력 제출",
        "         CLEAR 버튼             input 전체 삭제",
        "         Ctrl+L / Ctrl+U        input 전체 삭제",
        "         MULTI ON/OFF 버튼      멀티에이전트 사용 전환",
        "         /wiki                 오늘 저장된 Agent 응답 목록",
        "         /wiki last            최근 Agent 응답 화면 표시",
        "         /mode beginner         초급자 모드 전환",
        "         /mode intermediate     중급자 모드 전환",
        "         /mode advanced         고급자 모드 전환",
        "",
        "  AGENT",
        "         plan                   읽기 전용 계획/검토",
        "         build                  승인 후 수정/적용",
        "         chat                   질문/응답/자동수정 대화",
        "         /agent                 현재 Agent 모드 표시",
        "",
        "  PROJECT",
        "         SAMPLE 버튼           샘플 모델 목록에서 선택",
        "         /path <경로>           프로젝트 경로 직접 입력",
        "         /open [상위경로]       파일/폴더 열기",
        "         /folder [상위경로]     폴더 목록에서 선택",
        "         /file [상위경로]       파일 위치 기준으로 선택",
        "         /sample tensorflow     TensorFlow 샘플 생성/선택",
        "         /sample pytorch        PyTorch 샘플 생성/선택",
        "         /sample large10        대형 모델 샘플 10개 생성",
        "         /sample all            기본 샘플 생성",
        "",
        "  MODEL",
        "         /model                 모델 목록 표시",
        "         /model qwen3.6         Agent 모델 선택",
        "",
        "  WIZARD",
        "         다음 / 이전            Step 이동",
        "         1~10                   Step 번호 이동",
        "         Step 6에서 1           승인 후 파일 생성/수정",
        "  ----------------------------------------------------------------------------",
    ]
    return "\n".join(rows)


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
    for prefix in ("/folder", "/폴더", "/dir", "/디렉토리", "/open", "/열기", "/file", "/파일"):
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
        return "[Open Folder] 폴더 경로를 입력하세요"
    return f"[Open Folder] Tab/화살표 선택, Enter 확정, 1-{len(folders)} 번호 가능"


def format_folder_choices(folders: list[Path], current_folder: Path | None = None) -> str:
    lines = ["파일/폴더를 열 프로젝트 위치로 선택하세요."]
    for index, folder in enumerate(folders, start=1):
        marker = " (선택)" if current_folder is not None and folder == current_folder else ""
        lines.append(f"{index}. {folder}{marker}")
    lines.append("번호를 입력하거나 /open <기준경로>로 후보를 다시 불러올 수 있습니다.")
    return "\n".join(lines)


SAMPLE_CHOICES = (
    ("여러 기본 샘플 모델 생성 후 실행", "/sample run --kind all"),
    ("로컬 학습 가능한 표준 PyTorch 샘플 실행", "/sample run --kind standard --framework pytorch --mode train"),
    ("TensorFlow", "/sample tensorflow"),
    ("PyTorch", "/sample pytorch"),
    ("scikit-learn", "/sample sklearn"),
    ("ONNX", "/sample onnx"),
    ("Sora", "/sample sora"),
    ("Standard PyTorch", "/sample standard pytorch"),
    ("Standard TensorFlow", "/sample standard tensorflow"),
    ("Large 10", "/sample large10"),
    ("All Basic", "/sample all"),
)


def sample_selection_placeholder(choices: tuple[tuple[str, str], ...] = SAMPLE_CHOICES) -> str:
    return f"[Sample Select] Tab/화살표 선택, Enter 확정, 1-{len(choices)} 번호 가능"


def format_sample_choices(
    choices: tuple[tuple[str, str], ...] = SAMPLE_CHOICES,
    selected_index: int = 0,
) -> str:
    lines = [
        "샘플 모델을 선택하세요.",
        "선택한 샘플은 자동 생성되고 Step 1 프로젝트 경로로 설정됩니다.",
    ]
    for index, (label, command) in enumerate(choices, start=1):
        marker = " (선택)" if index - 1 == selected_index else ""
        lines.append(f"{index}. {label}{marker}")
    lines.append("번호를 입력하거나 Tab/화살표로 이동 후 Enter를 누르세요.")
    return "\n".join(lines)


def extract_agent_response_choices(text: str, limit: int = 9) -> list[str]:
    choices: list[str] = []
    for line in text.splitlines():
        match = AGENT_RESPONSE_CHOICE_RE.match(line)
        if not match:
            continue
        label = match.group(4).strip()
        if not label:
            continue
        choices.append(label)
        if len(choices) >= limit:
            break
    return choices


def format_agent_response_choices(choices: list[str], current_index: int = 0) -> str:
    lines = [
        "Agent 응답 선택",
        "- 번호를 입력하면 해당 항목을 실행합니다.",
        "- 원하는 항목이 없으면 직접 다시 메시지를 입력하세요.",
        "",
    ]
    for index, choice in enumerate(choices, start=1):
        marker = " (선택)" if index - 1 == current_index else ""
        lines.append(f"{index}. {choice}{marker}")
    return "\n".join(lines)


def agent_response_choice_placeholder(choices: list[str]) -> str:
    if not choices:
        return "원하는 내용을 다시 입력하세요"
    return f"[Agent Choices] 1-{len(choices)} 번호 선택, 또는 직접 질문 입력"


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


def format_tui_model_info(project_path: str) -> list[str]:
    if not project_path:
        return ["- 프로젝트 모델: (프로젝트 경로 미선택)"]
    analysis = analyze_project(project_path)
    if not analysis.exists or not analysis.is_directory:
        return ["- 프로젝트 모델: 프로젝트 경로 확인 필요"]
    if not analysis.scan.model_artifacts:
        return ["- 프로젝트 모델: 모델 artifact 없음"]
    first = analysis.scan.model_artifacts[0]
    rows = [
        f"- 프로젝트 모델: {first.path}",
        f"- 모델 크기: {format_bytes(first.size_bytes)}",
        f"- 모델 후보: {len(analysis.scan.model_artifacts)}개",
    ]
    if len(analysis.scan.model_artifacts) > 1:
        rows.append(
            "- 추가 모델: "
            + ", ".join(item.path for item in analysis.scan.model_artifacts[1:4])
        )
    parameter_rows = format_model_parameters(analysis.model_parameters, limit=12)
    rows.append("- 모델 파라미터:")
    rows.extend(f"  - {item}" for item in parameter_rows)
    return rows


def format_tui_chatbot_screen(project_path: str, model: str, launch_mode: str | None = None) -> str:
    mode_label = MODE_LABELS.get(launch_mode or "", "TUI")
    project_text = project_path or "(프로젝트 경로 미선택)"
    model_info = format_tui_model_info(project_path)
    return "\n".join(
        [
            format_agent_mode_title("Chatbot"),
            "",
            f"- 실행 모드: {mode_label}",
            f"- 프로젝트: {project_text}",
            f"- Agent 모델: {model}",
            *model_info,
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
            "- PLAN / BUILD / CHAT : 모드 전환",
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


def is_chat_coding_request(command: str) -> bool:
    lowered = command.lower()
    coding_keywords = (
        "코딩",
        "개발",
        "구현",
        "기능 추가",
        "기능 수정",
        "파일 수정",
        "파일 생성",
        "코드 작성",
        "코드 수정",
        "디렉토리 분석",
        "폴더 분석",
        "코드베이스",
        "coding",
        "develop",
        "implement",
        "implementation",
        "edit file",
        "write code",
        "modify code",
        "add feature",
        "codebase",
    )
    return any(keyword in lowered for keyword in coding_keywords)


def should_use_autofix_chat(command: str) -> bool:
    lowered = command.lower()
    keywords = (
        "분석",
        "검증",
        "수정",
        "고쳐",
        "오류",
        "에러",
        "등록",
        "자동",
        "프로젝트",
        "mlflow",
        "requirements",
        "requirement",
        "job template",
        "template",
        "run_model",
        "train.py",
        "config",
        "서빙",
        "모델",
        "artifact",
    )
    return any(keyword in lowered for keyword in keywords)


def is_chat_apply_approved(command: str) -> bool:
    lowered = command.lower()
    approval_keywords = (
        "승인",
        "적용",
        "반영",
        "진행",
        "apply",
        "approved",
        "go ahead",
    )
    approved = any(keyword in lowered for keyword in approval_keywords)
    return approved and (is_fix_request(command) or is_standard_template_request(command))


def chat_blocked_reasons(command: str) -> list[str]:
    lowered = command.lower()
    reasons: list[str] = []
    if any(keyword in lowered for keyword in ("삭제", "지워", "delete", "remove file", "rm ")):
        reasons.append("파일 삭제 요청은 차단됩니다.")
    if any(keyword in lowered for keyword in ("프로젝트 외부", "system32", "/etc/", "다른 프로젝트", "outside project")):
        reasons.append("선택한 프로젝트 폴더 밖의 쓰기 작업은 차단됩니다.")
    if any(keyword in lowered for keyword in ("artifact 삭제", "artifact 교체", "모델 교체", "모델 삭제", "replace model")):
        reasons.append("모델 artifact 삭제/교체는 차단됩니다.")
    if any(keyword in lowered for keyword in ("api key", "apikey", "password", "credential", "secret", "비밀번호", "토큰")):
        reasons.append("credential/API key를 코드에 직접 삽입하는 요청은 차단됩니다.")
    if any(keyword in lowered for keyword in ("bucket 업로드", "원격 업로드", "remote upload")) and not any(
        keyword in lowered for keyword in ("설정", "env", "환경")
    ):
        reasons.append("Bucket/MLflow 원격 업로드는 설정값 확인 없이 실행하지 않습니다.")
    return reasons


@dataclass(frozen=True)
class ChatCodePolicyPlan:
    safe: list[FixPreview]
    review_required: list[FixPreview]
    blocked: list[str]

    @property
    def has_applicable(self) -> bool:
        return bool(self.safe or self.review_required)


def classify_chat_fix_previews(previews: list[FixPreview], command: str = "") -> ChatCodePolicyPlan:
    safe: list[FixPreview] = []
    review_required: list[FixPreview] = []
    blocked = chat_blocked_reasons(command)
    for preview in previews:
        if preview.code in SAFE_CHAT_FIX_CODES:
            safe.append(preview)
        elif preview.code in REVIEW_REQUIRED_CHAT_FIX_CODES:
            review_required.append(preview)
        else:
            review_required.append(preview)
    return ChatCodePolicyPlan(safe=safe, review_required=review_required, blocked=blocked)


def format_chat_code_policy_plan(plan: ChatCodePolicyPlan) -> str:
    rows = ["수정 정책: 승인 기반 자동수정"]
    if plan.safe:
        rows.append("")
        rows.append("승인하면 자동 적용 가능한 항목")
        rows.extend(f"- {preview.target}: {preview.title}" for preview in plan.safe)
    if plan.review_required:
        rows.append("")
        rows.append("검토 필요")
        rows.extend(f"- {preview.target}: {preview.title}" for preview in plan.review_required)
    if plan.blocked:
        rows.append("")
        rows.append("수정 차단")
        rows.extend(f"- {reason}" for reason in plan.blocked)
    if plan.has_applicable:
        rows.extend(
            [
                "",
                "선택하세요. 번호만 입력하면 됩니다.",
                "1. safe 항목 적용",
                "2. 검토 필요 항목 미리보기",
                "3. safe + 검토 필요 항목 적용",
                "4. 취소",
            ]
        )
    else:
        rows.extend(["", "- 적용 가능한 수정 항목이 없습니다. 파일은 수정하지 않았습니다."])
    return "\n".join(rows)


def is_standard_template_request(command: str) -> bool:
    lowered = command.lower()
    keywords = (
        "표준 템플릿",
        "학습 템플릿",
        "템플릿 만들어",
        "템플릿 생성",
        "ml/dl",
        "ml dl",
        "framework template",
        "training template",
    )
    return any(keyword in lowered for keyword in keywords)


def extract_template_framework(command: str) -> str:
    lowered = command.lower()
    candidates = (
        "tensorflow",
        "텐서플로우",
        "pytorch",
        "파이토치",
        "sklearn",
        "사이킷런",
        "xgboost",
        "onnx",
        "huggingface",
        "hf",
        "sora",
        "소라",
        "llm",
        "vision",
        "비전",
    )
    for candidate in candidates:
        if candidate in lowered:
            return normalize_standard_framework(candidate)
    return "generic"


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
        if is_shell_prompt_line(candidate):
            continue
        for prefix in (">", "경로:", "path:", "프로젝트:", "project:"):
            if candidate.lower().startswith(prefix.lower()):
                candidate = candidate[len(prefix) :].strip()
        candidate = strip_shell_path_prefix(candidate)
        for expanded in expand_drag_drop_path_candidates(candidate):
            if expanded:
                candidates.append(expanded)
    return candidates or [value.strip()]


def is_shell_prompt_line(candidate: str) -> bool:
    if re.match(r"^(PS\s+)?[A-Za-z]:\\.*>\s*$", candidate):
        return True
    if re.match(r"^[^@\s]+@[^:\s]+:.*[$#]\s*$", candidate):
        return True
    return False


def expand_drag_drop_path_candidates(candidate: str) -> list[str]:
    value = candidate.strip().strip("\u200e\u200f")
    if not value:
        return []
    variants = [value]
    stripped = value.strip().strip('"').strip("'")
    variants.append(stripped)
    if stripped.endswith(("/", "\\")) and len(stripped) > 1:
        variants.append(stripped.rstrip("/\\"))
    quoted_matches = re.findall(r"""['"]([^'"]+)['"]""", value)
    variants.extend(match.strip() for match in quoted_matches)
    file_matches = re.findall(r"file://[^\s'\"<>]+", value)
    variants.extend(file_matches)
    if "file://localhost/" in value:
        variants.append(value.replace("file://localhost", "file://"))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def strip_shell_path_prefix(candidate: str) -> str:
    value = candidate.strip()
    lowered = value.lower()
    for prefix in ("cd ", "dir ", "ls ", "open ", "explorer ", "start "):
        if lowered.startswith(prefix):
            return value[len(prefix) :].strip()
    if value.startswith("& "):
        return value[2:].strip()
    if value.startswith("@"):
        return value[1:].strip()
    return value


def normalize_pasted_input(text: str) -> str:
    normalized = repair_clipboard_mojibake(text)
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = PASTE_ESCAPE_RE.sub("", normalized)
    normalized = PASTE_CONTROL_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    normalized = textwrap.dedent(normalized).strip("\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    compacted: list[str] = []
    blank_seen = False
    for line in lines:
        if not line.strip():
            if not blank_seen:
                compacted.append("")
            blank_seen = True
            continue
        compacted.append(line)
        blank_seen = False
    return unicodedata.normalize("NFC", "\n".join(compacted).strip())


def is_long_paste(text: str) -> bool:
    return len(text) > LONG_PASTE_CHAR_LIMIT or text.count("\n") + 1 > LONG_PASTE_LINE_LIMIT


def paste_status_message(text: str, *, duplicate: bool = False) -> str:
    if duplicate:
        return "같은 붙여넣기가 감지되어 중복 입력을 무시했습니다."
    if is_long_paste(text):
        return "긴 입력이 정리되었습니다. Enter로 전송하세요."
    return ""


def read_ai_studio_env(path: Path) -> dict[str, str]:
    values = {key: "" for key, _ in AI_STUDIO_ENV_FIELDS}
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


def write_ai_studio_env(path: Path, values: dict[str, str]) -> None:
    rows = [
        "# AI Studio MLflow environment.",
        "# Saved from beginner Wizard Step 7.",
        "# MLFLOW_TRACKING_URL is mapped to MLFLOW_TRACKING_URI by run_model.py.",
    ]
    for key, _ in AI_STUDIO_ENV_FIELDS:
        rows.append(f"{key}={values.get(key, '')}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def normalize_input_path(raw: str) -> Path | None:
    for candidate in path_candidates_from_input(raw):
        path = existing_filesystem_path(candidate)
        if path is None:
            continue
        if path.exists():
            if path.is_file():
                return path.parent
            if path.is_dir():
                return path
    return None


def choose_folder_with_dialog(initial_path: str = "") -> Path | None:
    initial_dir = normalize_input_path(initial_path) if initial_path.strip() else None
    try:
        from tkinter import Tk, filedialog
    except Exception:
        return None
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=str(initial_dir) if initial_dir else None)
        root.destroy()
    except Exception:
        return None
    if not selected:
        return None
    path = existing_filesystem_path(selected)
    if path is None:
        return None
    return path if path.is_dir() else path.parent


def looks_like_path_input(raw: str) -> bool:
    for candidate in path_candidates_from_input(raw):
        if not candidate:
            continue
        normalized = normalize_path_text(candidate)
        if is_windows_style_path(normalized):
            return True
        if normalized.startswith(("/", "~", "file://")):
            return True
        if re.match(r"^[A-Za-z]:[\\/]", normalized):
            return True
    return False


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
    folders: list[Path] = []
    seen: set[Path] = set()

    def add_folder(candidate: Path) -> bool:
        resolved = candidate.resolve()
        if resolved in seen:
            return False
        seen.add(resolved)
        folders.append(resolved)
        return len(folders) >= limit

    if base.strip():
        selected = normalize_input_path(base)
        if selected is not None:
            roots.append(selected)
        else:
            path = existing_filesystem_path(base)
            if path is not None:
                roots.append(path.parent if path.is_file() else path)
    else:
        for project in list_existing_work_projects():
            if add_folder(project):
                return folders
        roots.extend(
            [
                Path.cwd() / "work",
                Path.cwd().parent / "work",
                config.root_dir / "work",
                config.root_dir.parent / "work",
                Path.cwd(),
                config.root_dir / ".aiu" / "sample_projects",
            ]
        )
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        candidates = [root] if folder_has_project_signals(root) else []
        try:
            candidates.extend(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        except OSError:
            continue
        for candidate in candidates:
            if add_folder(candidate):
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
    awaiting_sample_selection: bool = False
    sample_selection_index: int = 0
    awaiting_agent_response_choice: bool = False
    agent_response_choice_index: int = 0
    agent_response_choices: list[str] = field(default_factory=list)
    last_agent_response: str = ""
    awaiting_chat_code_policy: bool = False
    pending_chat_code_policy: ChatCodePolicyPlan | None = None
    pending_chat_command: str = ""
    pending_chat_agent_content: str = ""
    awaiting_run_model_repair: bool = False
    awaiting_ai_studio_env: bool = False
    ai_studio_env_index: int = 0
    ai_studio_env_values: dict[str, str] = field(default_factory=dict)
    chat_context_entries: list[dict[str, str]] = field(default_factory=list)
    chat_context_summary: str = ""
    chat_context_compacted_count: int = 0

    def __post_init__(self) -> None:
        self.project_path = ""
        self.sample_message: str | None = None
        self.app_config = AppConfig.load()
        export_prompt_templates_to_wiki(self.app_config)
        saved_summary = load_chat_context_summary(self.app_config)
        self.chat_context_summary = str(saved_summary.get("summary") or "")
        self.chat_context_compacted_count = int(saved_summary.get("message_count") or 0)
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
    def multi_agent_enabled(self) -> bool:
        return self.app_config.get_bool("ENABLE_MULTI_AGENT")

    def toggle_multi_agent(self) -> str:
        enabled = not self.multi_agent_enabled
        self.app_config.values["ENABLE_MULTI_AGENT"] = "true" if enabled else "false"
        self.deepagents_runtime = DeepAgentsRuntime(self.app_config)
        message = (
            "멀티에이전트가 ON으로 변경되었습니다. 복잡한 분석은 더 깊게 처리하지만 응답이 느릴 수 있습니다."
            if enabled
            else "멀티에이전트가 OFF로 변경되었습니다. CHAT 일반 질문은 Qwen 단일 호출로 빠르게 응답합니다."
        )
        self.latest_message = message
        return message

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
        if not self.steps:
            return build_beginner_intro()
        return format_beginner_tab(self.index, len(self.steps), self.steps[self.index])

    def render_log(self) -> str:
        log_text = "\n\n".join(self.log_lines[-MAX_CHAT_LOG_ENTRIES:])
        if self.agent_mode == "Chatbot" and self.selected_launch_mode is not None:
            screen = self.current_screen()
            if log_text:
                return trim_tui_render_text(f"{log_text}\n\n{screen}")
            if self.latest_message and self.latest_message != screen:
                return trim_tui_render_text(f"{self.latest_message}\n\n{screen}")
            return screen
        if self.selected_launch_mode in {MODE_INTERMEDIATE, MODE_ADVANCED}:
            screen = self.current_screen()
            return trim_tui_render_text(f"{log_text}\n\n{screen}") if log_text else screen
        if self.selected_launch_mode == MODE_BEGINNER and self.latest_message:
            parts = [part for part in [log_text, self.latest_message, self.current_screen()] if part]
            return trim_tui_render_text("\n\n".join(parts))
        if log_text:
            return trim_tui_render_text(f"{log_text}\n\n{self.current_screen()}")
        return self.current_screen()

    def activate_launch_mode(self, mode: str) -> str:
        self.selected_launch_mode = mode
        self.latest_message = ""
        if mode == MODE_BEGINNER:
            self.agent_mode = "Plan"
            if self.project_input.strip():
                self.set_project(self.project_input)
                if self.sample_message:
                    self.latest_message = self.sample_message
            else:
                self.project_path = ""
                self.sample_message = None
                self.applied_changes = None
                self.steps = []
                self.index = 0
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
        if (
            not command
            or command in EXIT_COMMANDS
            or command in HELP_COMMANDS
            or command in COMPACT_COMMANDS
            or self.awaiting_model_selection
            or self.awaiting_sample_selection
            or self.awaiting_ai_studio_env
            or self.awaiting_chat_code_policy
        ):
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
        path_value, is_path_command = strip_path_command(command)
        if (
            is_path_command
            or command.startswith("/sample ")
            or command.startswith("/샘플 ")
            or looks_like_path_input(command)
            or normalize_input_path(command) is not None
        ):
            return False
        if self.selected_launch_mode in {MODE_INTERMEDIATE, MODE_ADVANCED}:
            return self.agent_mode == "Chatbot"
        if self.selected_launch_mode != MODE_BEGINNER or self.agent_mode != "Chatbot":
            return False
        if is_wizard_navigation(command, self.total):
            return False
        return True

    def set_thinking(self, raw: str, elapsed_seconds: int = 0) -> None:
        command = raw.strip()
        response = format_thinking_animation(elapsed_seconds)
        self.latest_message = format_chat_card(command, response)
        self._append_or_replace_chat_log(command, response, [])

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
            message = "선택 가능한 폴더를 찾지 못했습니다. /open <상위폴더경로> 형태로 다시 입력하세요."
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

    @property
    def highlighted_sample_command(self) -> str:
        self.sample_selection_index %= len(SAMPLE_CHOICES)
        return SAMPLE_CHOICES[self.sample_selection_index][1]

    def start_sample_selection(self) -> str:
        self.awaiting_sample_selection = True
        self.sample_selection_index = 0
        message = format_sample_choices(SAMPLE_CHOICES, self.sample_selection_index)
        self.latest_message = message
        return message

    def cycle_sample_selection(self, delta: int = 1) -> str:
        self.awaiting_sample_selection = True
        self.sample_selection_index = (self.sample_selection_index + delta) % len(SAMPLE_CHOICES)
        message = format_sample_choices(SAMPLE_CHOICES, self.sample_selection_index)
        self.latest_message = message
        return self.highlighted_sample_command

    def select_sample(self, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            command = self.highlighted_sample_command
        elif candidate.isdigit() and 1 <= int(candidate) <= len(SAMPLE_CHOICES):
            command = SAMPLE_CHOICES[int(candidate) - 1][1]
        elif candidate.startswith("/sample ") or candidate.startswith("/샘플 "):
            command = candidate
        else:
            lowered = candidate.lower()
            matched = next((sample_command for label, sample_command in SAMPLE_CHOICES if lowered in label.lower()), "")
            if not matched:
                message = "샘플을 선택하지 못했습니다. 번호를 입력하거나 SAMPLE 버튼으로 다시 선택하세요."
                self.latest_message = message
                return message
            command = matched
        self.awaiting_sample_selection = False
        self.set_project(command)
        self.latest_message = self.sample_message or ""
        return self.current_screen()

    @property
    def highlighted_agent_response_choice(self) -> str:
        if not self.agent_response_choices:
            return ""
        self.agent_response_choice_index %= len(self.agent_response_choices)
        return self.agent_response_choices[self.agent_response_choice_index]

    def cycle_agent_response_choice(self, delta: int = 1) -> str:
        self.awaiting_agent_response_choice = True
        if self.agent_response_choices:
            self.agent_response_choice_index = (self.agent_response_choice_index + delta) % len(self.agent_response_choices)
        return self.highlighted_agent_response_choice

    def _capture_agent_response_choices(self, agent_response: str) -> None:
        choices = extract_agent_response_choices(agent_response)
        self.last_agent_response = agent_response
        self.agent_response_choices = choices
        self.agent_response_choice_index = 0
        self.awaiting_agent_response_choice = bool(choices)
        if choices:
            message = format_agent_response_choices(choices, self.agent_response_choice_index)
            self.latest_message = f"{agent_response}\n\n{message}"
            self.log_lines.append(message)

    def select_agent_response_choice(self, value: str) -> str:
        prepared = self.prepare_agent_response_choice_submission(value)
        return self.handle_chat_message(prepared[0])

    def prepare_agent_response_choice_submission(self, value: str) -> tuple[str, str]:
        candidate = value.strip()
        if not self.agent_response_choices:
            self.awaiting_agent_response_choice = False
            return candidate, candidate
        if not candidate:
            choice = self.highlighted_agent_response_choice
        elif candidate.isdigit() and 1 <= int(candidate) <= len(self.agent_response_choices):
            choice = self.agent_response_choices[int(candidate) - 1]
        else:
            self.awaiting_agent_response_choice = False
            self.agent_response_choices = []
            return candidate, candidate
        self.awaiting_agent_response_choice = False
        self.agent_response_choices = []
        followup = (
            "아래 Agent 응답 선택지를 실행해줘.\n"
            f"선택 항목: {choice}\n\n"
            f"이전 Agent 응답:\n{self.last_agent_response}"
        )
        return followup, f"선택: {choice}"

    @property
    def ai_studio_env_path(self) -> Path:
        return Path(self.project_path) / "ai_studio.env"

    def start_ai_studio_env_setup(self) -> str:
        if not self.project_path:
            message = "먼저 Step 1에서 프로젝트 경로를 선택하세요."
            self.latest_message = message
            return message
        self.awaiting_ai_studio_env = True
        self.ai_studio_env_index = 0
        self.ai_studio_env_values = read_ai_studio_env(self.ai_studio_env_path)
        message = self._format_ai_studio_env_prompt()
        self.latest_message = message
        return message

    def continue_ai_studio_env_setup(self, value: str) -> str:
        key, _ = AI_STUDIO_ENV_FIELDS[self.ai_studio_env_index]
        entered = value.strip()
        if entered:
            self.ai_studio_env_values[key] = entered
        self.ai_studio_env_index += 1
        if self.ai_studio_env_index >= len(AI_STUDIO_ENV_FIELDS):
            write_ai_studio_env(self.ai_studio_env_path, self.ai_studio_env_values)
            self.awaiting_ai_studio_env = False
            message = (
                "AI Studio 환경설정이 저장되었습니다.\n"
                f"- 저장 대상: {self.ai_studio_env_path}\n"
                "- MLFLOW_TRACKING_URL은 run_model.py 실행 시 MLFLOW_TRACKING_URI로 자동 매핑됩니다.\n"
                "- 다음 단계: python run_model.py --env-file ai_studio.env --register"
            )
            self.latest_message = message
            return message
        message = self._format_ai_studio_env_prompt()
        self.latest_message = message
        return message

    def _format_ai_studio_env_prompt(self) -> str:
        key, label = AI_STUDIO_ENV_FIELDS[self.ai_studio_env_index]
        current = self.ai_studio_env_values.get(key, "")
        current_label = f"현재값: {current}" if current else "현재값: (비어 있음)"
        return (
            "AI Studio 환경설정 입력\n"
            f"- {self.ai_studio_env_index + 1}/{len(AI_STUDIO_ENV_FIELDS)} {label}\n"
            f"- 키: {key}\n"
            f"- {current_label}\n"
            "- 값을 입력하고 Enter를 누르세요. 빈 값은 현재값을 유지합니다."
        )

    def submit(self, raw: str) -> str:
        command = raw.strip()
        if command in HELP_COMMANDS:
            message = format_tui_help_screen(
                self.selected_launch_mode,
                self.agent_mode,
                self.project_path,
                self.qwen_model,
            )
            self.latest_message = message
            return message
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

        if self.awaiting_ai_studio_env:
            return self.continue_ai_studio_env_setup(raw)
        if not command and self.awaiting_folder_selection:
            return self.select_folder("")
        if not command and self.awaiting_sample_selection:
            return self.select_sample("")
        if not command and self.awaiting_agent_response_choice:
            return self.select_agent_response_choice("")
        if not command and self.awaiting_chat_code_policy:
            return self.select_chat_code_policy("1")
        if not command and self.awaiting_run_model_repair:
            return self.repair_and_rerun_local_model()
        if not command and self.awaiting_model_selection:
            return self.select_model("")
        if not command and self.selected_launch_mode == MODE_BEGINNER and not self.steps:
            self.latest_message = ""
            return self.current_screen()
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
        if self.awaiting_chat_code_policy and command in {"1", "2", "3", "4"}:
            return self.select_chat_code_policy(command)
        if self.awaiting_run_model_repair and command in {"1", "2", "3"}:
            return self.select_run_model_repair(command)
        if self.awaiting_agent_response_choice and command.isdigit():
            return self.select_agent_response_choice(command)

        folder_base = parse_folder_command(command)
        if folder_base is not None:
            return self.start_folder_selection(folder_base)
        if self.awaiting_folder_selection:
            return self.select_folder(command)
        if self.awaiting_sample_selection:
            return self.select_sample(command)
        if self.awaiting_chat_code_policy:
            return self.select_chat_code_policy(command)
        if self.awaiting_run_model_repair:
            return self.select_run_model_repair(command)
        if self.awaiting_agent_response_choice:
            return self.select_agent_response_choice(command)

        path_result = self._try_select_project_path_from_input(command, announce=self.agent_mode == "Chatbot")
        if path_result is not None:
            return path_result

        if command.startswith("/sample ") or command.startswith("/샘플 "):
            self.set_project(command)
            message = self.sample_message or ""
            self.latest_message = message
            return self.current_screen()

        if self.selected_launch_mode == MODE_BEGINNER and not self.steps and self.agent_mode != "Chatbot":
            self.set_project(command)
            if self.sample_message:
                self.latest_message = self.sample_message
            return self.render_log() if self.latest_message else self.current_screen()

        if self.index == 3:
            return self._handle_issue_choice(command)
        if self.index == 5 and command in {"1", "2", "3"}:
            return self._handle_approval_choice(command)
        if self.index == 6 and command in {"1", "/env", "/환경", "환경설정", "ai studio", "aistudio"}:
            return self.start_ai_studio_env_setup()
        if self.index == 8 and command in {"1", "2"}:
            return self._handle_serving_choice(command)
        if self.index == 9 and command == "1":
            return self._handle_report_choice(command)
        if is_wizard_navigation(command, self.total):
            return self._handle_navigation(command)
        if self.agent_mode != "Chatbot":
            return self._handle_non_chatbot_text(command)
        return self.handle_chat_message(command)

    def _submit_intermediate(self, command: str) -> str:
        if command in HELP_COMMANDS:
            self.latest_message = format_tui_help_screen(
                self.selected_launch_mode,
                self.agent_mode,
                self.project_path,
                self.qwen_model,
            )
            return self.current_screen()
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
        path_result = self._try_select_project_path_from_input(command, announce=True)
        if path_result is not None:
            return path_result
        if self.agent_mode == "Chatbot":
            if self.awaiting_chat_code_policy:
                response = self.select_chat_code_policy(command)
                return self.render_log() if response else response
            if self.awaiting_agent_response_choice:
                response = self.select_agent_response_choice(command)
                return self.render_log() if response else response
            response = self.handle_chat_message(command)
            return self.render_log() if response else response
        if is_greeting(command):
            self.latest_message = greeting_response()
            return self.current_screen()
        self.latest_message = handle_intermediate_request(command)
        return self.current_screen()

    def _submit_advanced(self, command: str) -> str:
        if command in HELP_COMMANDS:
            self.latest_message = format_tui_help_screen(
                self.selected_launch_mode,
                self.agent_mode,
                self.project_path,
                self.qwen_model,
            )
            return self.current_screen()
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
        path_result = self._try_select_project_path_from_input(command, announce=True)
        if path_result is not None:
            return path_result
        if self.agent_mode == "Chatbot":
            if self.awaiting_chat_code_policy:
                response = self.select_chat_code_policy(command)
                return self.render_log() if response else response
            if self.awaiting_agent_response_choice:
                response = self.select_agent_response_choice(command)
                return self.render_log() if response else response
            response = self.handle_chat_message(command)
            return self.render_log() if response else response
        self.latest_message = handle_advanced_input(command)
        return self.current_screen()

    def _try_select_project_path_from_input(self, command: str, announce: bool = False) -> str | None:
        path_value, is_path_command = strip_path_command(command)
        if is_path_command and not path_value:
            message = "경로를 함께 입력하세요. 예: /path C:\\Users\\me\\my-model"
            self.latest_message = message
            return self.render_log() if self.agent_mode == "Chatbot" else message
        selected_path = normalize_input_path(command)
        if selected_path is not None:
            self.project_path = str(selected_path)
            self.sample_message = None
            self.applied_changes = None
            self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
            self.index = 0
            if announce:
                self.latest_message = (
                    "프로젝트 폴더를 선택했습니다.\n"
                    f"- 위치: {self.project_path}\n"
                    "- 이제 Chatbot 질문, 분석, 코드 수정 요청에 이 경로를 사용합니다."
                )
                return self.render_log() if self.agent_mode == "Chatbot" else self.current_screen()
            self.latest_message = ""
            return self.current_screen()
        if is_path_command:
            message = f"경로를 찾을 수 없습니다: {path_value}"
            self.latest_message = message
            return self.render_log() if self.agent_mode == "Chatbot" else message
        return None

    def handle_chat_message(self, command: str) -> str:
        normalized_command = " ".join(command.strip().lower().split())
        if normalized_command in WIKI_LAST_COMMANDS:
            response = format_wiki_recent_prompt_for_tui(self.app_config)
            self.latest_message = response
            return response
        if normalized_command in WIKI_COMMANDS:
            response = format_wiki_today_for_tui(self.app_config)
            self.latest_message = response
            return response
        if command.strip() in COMPACT_COMMANDS:
            response = self.compact_chat_context(reason="manual")
            self.latest_message = response
            self._append_chat_log(command, response, [])
            return response
        if is_greeting(command):
            response = greeting_response()
            self._save_chat_session(command, response, [], None)
            response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode="Chat"))
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, [])
            return response
        chat_apply_approved = is_chat_apply_approved(command)
        template_requested = is_standard_template_request(command)
        coding_requested = is_chat_coding_request(command)
        blocked_reasons = chat_blocked_reasons(command)
        fix_requested = is_fix_request(command) or coding_requested or template_requested or bool(blocked_reasons)
        if fix_requested and not self.project_path:
            response = "코드 수정 기능을 사용하려면 먼저 프로젝트 폴더를 선택하세요. FILE 버튼 또는 /folder 명령으로 선택할 수 있습니다."
            self._save_chat_session(command, response, [], None)
            response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode="Build"))
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, [])
            return response
        use_autofix = should_use_autofix_chat(command) or chat_apply_approved
        runtime_mode = "Build" if (chat_apply_approved or coding_requested) else ("AutoFix" if use_autofix else "Chat")
        if not self.multi_agent_enabled and runtime_mode == "Chat":
            response = chat_with_qwen(
                self._runtime_prompt_with_context(command),
                config=self.qwen_config,
                project_path=self.project_path,
                agent_mode="Chat",
            )
            self._save_chat_session(command, response, [], None)
            response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode="Chat"))
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, [])
            self._capture_agent_response_choices(response)
            return response
        result = self._invoke_deepagents(command, agent_mode=runtime_mode)
        applied_changes: list[AppliedChange] = []
        if template_requested and chat_apply_approved:
            applied_changes.append(ensure_standard_ml_dl_template(Path(self.project_path), extract_template_framework(command)))
        if fix_requested:
            if chat_apply_approved:
                applied_changes.extend(self._apply_chat_code_policy(command, result.content, include_review=False))
                final_analysis = analyze_project(self.project_path)
                response = self._format_chatbot_response(result.content, applied_changes, final_analysis)
                self._save_chat_session(command, response, applied_changes, final_analysis)
                response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode=runtime_mode))
                self.latest_message = response
                self._append_or_replace_chat_log(command, response, applied_changes)
                self._capture_agent_response_choices(response)
                return response
            policy_response = self._start_chat_code_policy(command, result.content)
            self._save_chat_session(command, policy_response, [], analyze_project(self.project_path))
            policy_response = self._with_wiki_notice(policy_response, self._save_used_prompt(command, policy_response, agent_mode=runtime_mode))
            self.latest_message = policy_response
            self._append_or_replace_chat_log(command, policy_response, [])
            return policy_response
        if (result.used_deepagents and use_autofix) or (chat_apply_approved and not result.used_deepagents):
            applied_changes.extend(self._apply_fixable_issues_after_chat())
            final_analysis = analyze_project(self.project_path)
            response = self._format_chatbot_response(result.content, applied_changes, final_analysis)
            self._save_chat_session(command, response, applied_changes, final_analysis)
            response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode=runtime_mode))
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, applied_changes)
            self._capture_agent_response_choices(response)
            return response
        if result.used_deepagents:
            response = result.content
            self._save_chat_session(command, response, applied_changes, None)
            response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode=runtime_mode))
            self.latest_message = response
            self._append_or_replace_chat_log(command, response, applied_changes)
            self._capture_agent_response_choices(response)
            return response
        response = f"{result.content}\n파일은 수정하지 않았습니다."
        self._save_chat_session(command, response, applied_changes, analyze_project(self.project_path) if use_autofix else None)
        response = self._with_wiki_notice(response, self._save_used_prompt(command, response, agent_mode=runtime_mode))
        self.latest_message = response
        self._append_or_replace_chat_log(command, response, applied_changes)
        self._capture_agent_response_choices(response)
        return response

    def _handle_non_chatbot_text(self, command: str) -> str:
        return self.current_screen()

    def _invoke_deepagents(self, command: str, agent_mode: str | None = None):
        runtime = self.deepagents_runtime or DeepAgentsRuntime(self.app_config)
        self._auto_compact_chat_context()
        prompt = self._runtime_prompt_with_context(command)
        return runtime.invoke(prompt, project_path=self.project_path, agent_mode=agent_mode or self.agent_mode)

    def _runtime_prompt_with_context(self, command: str) -> str:
        recent_count = self.app_config.get_int("CHAT_CONTEXT_RECENT_MESSAGES", DEFAULT_CONTEXT_RECENT_MESSAGES)
        max_chars = self.app_config.get_int("CHAT_CONTEXT_MAX_CHARS", DEFAULT_CONTEXT_MAX_CHARS)
        recent_entries = self.chat_context_entries[-recent_count:] if recent_count > 0 else []
        return build_compacted_runtime_prompt(
            command,
            self.chat_context_summary,
            recent_entries,
            max_chars=max_chars,
        )

    def _auto_compact_chat_context(self) -> None:
        compact_after = self.app_config.get_int("CHAT_CONTEXT_COMPACT_AFTER", DEFAULT_CONTEXT_COMPACT_AFTER)
        recent_count = self.app_config.get_int("CHAT_CONTEXT_RECENT_MESSAGES", DEFAULT_CONTEXT_RECENT_MESSAGES)
        if compact_after <= 0 or len(self.chat_context_entries) < compact_after:
            return
        if len(self.chat_context_entries) <= recent_count:
            return
        self.compact_chat_context(reason="auto")

    def compact_chat_context(self, reason: str = "manual") -> str:
        recent_count = self.app_config.get_int("CHAT_CONTEXT_RECENT_MESSAGES", DEFAULT_CONTEXT_RECENT_MESSAGES)
        if not self.chat_context_entries and not self.chat_context_summary:
            return "압축할 챗봇 컨텍스트가 아직 없습니다."
        keep_count = max(0, recent_count)
        if reason == "manual" and len(self.chat_context_entries) <= keep_count:
            keep_count = 0
        compact_targets = self.chat_context_entries[:-keep_count] if keep_count else list(self.chat_context_entries)
        recent_entries = self.chat_context_entries[-keep_count:] if keep_count else []
        if compact_targets:
            self.chat_context_summary = compact_chat_entries(compact_targets, self.chat_context_summary)
            self.chat_context_compacted_count += len(compact_targets)
            save_chat_context_summary(self.app_config, self.chat_context_summary, self.chat_context_compacted_count)
            self.chat_context_entries = recent_entries
        elif self.chat_context_summary:
            save_chat_context_summary(self.app_config, self.chat_context_summary, self.chat_context_compacted_count)
        return (
            "챗봇 컨텍스트를 압축했습니다.\n"
            f"- 방식: {'자동' if reason == 'auto' else '수동'}\n"
            f"- 압축 누적 메시지: {self.chat_context_compacted_count}개\n"
            f"- 최근 유지 메시지: {len(self.chat_context_entries)}개"
        )

    def _apply_fixable_issues_after_chat(self) -> list[AppliedChange]:
        analysis = analyze_project(self.project_path)
        previews = build_fix_previews(analysis)
        if not previews:
            return []
        self.applied_changes = apply_fix_previews(self.project_path, previews)
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        self.index = min(6, len(self.steps) - 1)
        return self.applied_changes

    def _start_chat_code_policy(self, command: str, agent_content: str) -> str:
        analysis = analyze_project(self.project_path)
        plan = self._build_chat_code_policy_plan(analysis, command)
        self.pending_chat_code_policy = plan
        self.pending_chat_command = command
        self.pending_chat_agent_content = agent_content
        self.awaiting_chat_code_policy = plan.has_applicable
        response = f"{agent_content}\n\n{format_chat_code_policy_plan(plan)}"
        self.latest_message = response
        return response

    def select_chat_code_policy(self, value: str) -> str:
        command = value.strip()
        plan = self.pending_chat_code_policy
        if plan is None:
            self.awaiting_chat_code_policy = False
            message = "대기 중인 수정안이 없습니다."
            self.latest_message = message
            return message
        if command == "1":
            self.awaiting_chat_code_policy = False
            return self._apply_pending_chat_code_policy(include_review=False)
        if command == "2":
            message = self._format_review_required_preview(plan)
            self.latest_message = message
            return message
        if command == "3":
            self.awaiting_chat_code_policy = False
            return self._apply_pending_chat_code_policy(include_review=True)
        if command == "4":
            self.awaiting_chat_code_policy = False
            self.pending_chat_code_policy = None
            message = "수정을 취소했습니다. 파일은 수정하지 않았습니다."
            self.latest_message = message
            self._append_or_replace_chat_log(self.pending_chat_command or "수정 취소", message, [])
            return message
        message = "번호로 선택하세요. 1=safe 적용, 2=검토 필요 미리보기, 3=전체 승인 적용, 4=취소"
        self.latest_message = message
        return message

    def _format_review_required_preview(self, plan: ChatCodePolicyPlan) -> str:
        if not plan.review_required:
            return "검토 필요 항목이 없습니다. 1번을 선택하면 safe 항목만 적용합니다."
        rows = ["검토 필요 항목 미리보기"]
        for preview in plan.review_required:
            rows.extend(
                [
                    f"- 대상: {preview.target}",
                    f"  항목: {preview.title}",
                    f"  조치: {preview.action}",
                ]
            )
            rows.extend(f"  {line}" for line in preview.preview_lines)
        rows.extend(["", "3번을 선택하면 위 검토 필요 항목까지 승인 적용합니다."])
        return "\n".join(rows)

    def _apply_pending_chat_code_policy(self, *, include_review: bool) -> str:
        plan = self.pending_chat_code_policy
        if plan is None:
            message = "대기 중인 수정안이 없습니다."
            self.latest_message = message
            return message
        pending_command = self.pending_chat_command
        pending_agent_content = self.pending_chat_agent_content
        self.awaiting_chat_code_policy = False
        self.awaiting_agent_response_choice = False
        self.agent_response_choices = []
        previews = list(plan.safe)
        if include_review:
            previews.extend(plan.review_required)
        try:
            applied_changes = self._apply_chat_previews(previews)
            final_analysis = analyze_project(self.project_path)
        except Exception as exc:
            self.agent_mode = "Plan"
            self._append_chat_error(pending_command, exc)
            message = (
                "수정 적용 중 오류가 발생했습니다.\n"
                "- 파일 적용은 중단했고 Plan 모드로 돌아왔습니다.\n"
                f"- 오류: {exc}"
            )
            self.pending_chat_code_policy = None
            self.latest_message = message
            return message
        rows = [
            "수정 정책: 승인 기반 자동수정",
            "- Build 모드로 전환해 승인된 항목만 적용했습니다.",
            "- 수정이 완료되어 Plan 모드로 돌아왔습니다.",
        ]
        if plan.blocked:
            rows.append("수정 차단")
            rows.extend(f"- {reason}" for reason in plan.blocked)
        rows.append("")
        rows.append(self._format_chatbot_response(pending_agent_content, applied_changes, final_analysis))
        response = "\n".join(rows)
        self.pending_chat_code_policy = None
        self.latest_message = response
        self._append_or_replace_chat_log(pending_command, response, applied_changes)
        self._save_chat_session(pending_command, response, applied_changes, final_analysis)
        self._save_used_prompt(pending_command, response, agent_mode="Build")
        return response

    def _apply_chat_code_policy(
        self,
        command: str,
        agent_content: str,
        *,
        include_review: bool,
    ) -> list[AppliedChange]:
        analysis = analyze_project(self.project_path)
        plan = self._build_chat_code_policy_plan(analysis, command)
        if plan.blocked and not plan.has_applicable:
            return []
        previews = list(plan.safe)
        if include_review:
            previews.extend(plan.review_required)
        return self._apply_chat_previews(previews)

    def _build_chat_code_policy_plan(self, analysis, command: str) -> ChatCodePolicyPlan:
        plan = classify_chat_fix_previews(build_fix_previews(analysis), command)
        extra_review = self._manual_review_previews_from_command(command)
        if not extra_review:
            return plan
        return ChatCodePolicyPlan(
            safe=plan.safe,
            review_required=[*plan.review_required, *extra_review],
            blocked=plan.blocked,
        )

    def _manual_review_previews_from_command(self, command: str) -> list[FixPreview]:
        lowered = command.lower()
        if not any(keyword in lowered for keyword in ("train.py", "학습 코드", "기록 코드", "로직", "modelwrapper", "predict.py")):
            return []
        root = Path(self.project_path)
        target = "train.py" if (root / "train.py").exists() else ""
        if not target and (root / "aiu_custom" / "predict.py").exists():
            target = "aiu_custom/predict.py"
        if not target:
            return []
        return [
            FixPreview(
                code="CHAT_REVIEW_EXISTING_CODE_CHANGE",
                title="MLflow 기록 코드 추가" if "기록 코드" in lowered or "mlflow" in lowered else "기존 코드 로직 변경 검토",
                target=target,
                action="기존 학습/추론 코드는 바로 변경하지 않고 미리보기와 사용자 승인을 요구합니다.",
                preview_lines=[
                    "+ 변경 후보를 검토한 뒤 3번을 선택한 경우에만 적용 경로로 진입합니다.",
                    "+ 대규모 로직 변경, 모델 로딩 변경, MLflow 코드 삽입은 검토 필요 항목입니다.",
                ],
            )
        ]

    def _apply_chat_previews(self, previews: list[FixPreview]) -> list[AppliedChange]:
        if not previews:
            return []
        previous_mode = self.agent_mode
        self.agent_mode = "Build"
        self.applied_changes = apply_fix_previews(self.project_path, previews)
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        self.index = min(6, len(self.steps) - 1)
        self.agent_mode = "Plan" if previous_mode == "Chatbot" else previous_mode
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
        self._prune_chat_log_lines()

    def _append_or_replace_chat_log(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
    ) -> None:
        marker = f"  YOU    {user_message}"
        replacement = self._format_chat_log(user_message, agent_response, applied_changes)
        for index in range(len(self.log_lines) - 1, -1, -1):
            if marker in self.log_lines[index]:
                self.log_lines[index] = replacement
                self._prune_chat_log_lines()
                return
        self.log_lines.append(replacement)
        self._prune_chat_log_lines()

    def _prune_chat_log_lines(self) -> None:
        if len(self.log_lines) > MAX_CHAT_LOG_ENTRIES:
            self.log_lines = self.log_lines[-MAX_CHAT_LOG_ENTRIES:]

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
        return format_chat_card(user_message, agent_response, applied_changes)

    def _save_chat_session(
        self,
        user_message: str,
        agent_response: str,
        applied_changes: list[AppliedChange],
        final_analysis,
    ) -> None:
        self.chat_context_entries.append(
            {
                "user_message": user_message,
                "agent_response": agent_response,
                "agent_mode": self.agent_mode,
            }
        )
        append_chat_session_event(
            self.app_config,
            {
                "project_path": self.project_path,
                "user_message": user_message,
                "selected_model": self.qwen_model,
                "analysis_status": final_analysis.registration_status if final_analysis else "not_analyzed",
                "applied_changes": [change.as_dict() for change in applied_changes],
                "agent_response": agent_response,
                "remaining_issues": [issue.as_dict() for issue in final_analysis.issue_details] if final_analysis else [],
            },
        )

    def _save_used_prompt(self, user_message: str, agent_response: str, *, agent_mode: str) -> list[Path]:
        return append_used_prompt_to_wiki(
            self.app_config,
            {
                "project_path": self.project_path,
                "user_prompt": user_message,
                "system_prompt": build_deepagents_system_prompt(self.project_path, agent_mode),
                "agent_mode": agent_mode,
                "launch_mode": self.selected_launch_mode or "",
                "selected_model": self.qwen_model,
                "response_summary": agent_response,
                "agent_response": agent_response,
            },
        )

    def _with_wiki_notice(self, response: str, paths: list[Path]) -> str:
        dated_markdown = next((path for path in paths if path.suffix == ".md" and path.parent.name == "used"), None)
        markdown_path = dated_markdown or next((path for path in paths if path.suffix == ".md"), None)
        if markdown_path is None:
            return response
        return (
            f"{response}\n\n"
            "Wiki 저장됨\n"
            f"- 파일: {markdown_path}\n"
            "- 화면에서 다시 보기: /wiki last"
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
            try:
                analysis = analyze_project(self.project_path)
                previews = build_fix_previews(analysis)
                if not previews:
                    self.index += 1
                    message = "자동 수정할 항목이 없어 Step 6을 스킵했습니다. 파일은 수정하지 않았습니다."
                    self.latest_message = message
                    return self.current_screen()
                self.applied_changes = apply_fix_previews(self.project_path, previews)
                self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
                result = format_beginner_apply_result(self.applied_changes, analyze_project(self.project_path))
                self.index += 1
            except Exception as exc:
                self.index = 5
                self._append_chat_error("Step 6 승인 적용", exc)
                self.latest_message = (
                    "Step 6 수정 적용 중 오류가 발생했습니다.\n"
                    "- 파일 적용을 중단하고 Plan 모드로 돌아왔습니다.\n"
                    f"- 오류: {exc}"
                )
                return self.current_screen()
            finally:
                self.agent_mode = "Plan"
            result = f"{result}\n- Agent 모드: Build에서 수정 적용 후 Plan으로 자동 전환했습니다."
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
        message = "번호로 선택하세요. 1=승인 후 생성/수정, 2=수정안 다시 보기, 3=취소"
        self.latest_message = message
        return message

    def _handle_serving_choice(self, command: str) -> str:
        if command == "2":
            self.index = min(self.index + 1, self.total - 1)
            message = "Step 9 자동 보완을 건너뛰고 다음 단계로 이동합니다."
            self.latest_message = message
            return self.current_screen()
        analysis = analyze_project(self.project_path)
        previews = build_serving_fix_previews(analysis)
        if not previews:
            self.index = min(self.index + 1, self.total - 1)
            message = "Step 9에서 자동 보완할 항목이 없어 스킵했습니다."
            self.latest_message = message
            return self.current_screen()
        self.agent_mode = "Build"
        applied_changes = apply_serving_fix_previews(self.project_path, previews)
        refreshed = analyze_project(self.project_path)
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        result = format_serving_apply_result(applied_changes, refreshed)
        self.agent_mode = "Plan"
        result = f"{result}\n- Agent 모드: Build에서 Step 9 보완 적용 후 Plan으로 자동 전환했습니다."
        self.latest_message = result
        return self.current_screen()

    def _handle_report_choice(self, command: str) -> str:
        if self.awaiting_run_model_repair and command in {"1", "2", "3"}:
            return self.select_run_model_repair(command)
        return self.run_local_model_training()

    def run_local_model_training(self, *, after_repair: bool = False) -> str:
        if not self.project_path:
            message = "먼저 Step 1에서 샘플 또는 프로젝트 폴더를 선택하세요."
            self.latest_message = message
            return self.render_log()
        self.awaiting_run_model_repair = False
        self.agent_mode = "Build"
        try:
            result = run_beginner_mlflow_verification(self.project_path)
        except Exception as exc:
            result = (
                "로컬 모델 학습 실행 중 오류가 발생했습니다.\n"
                f"- 오류: {exc}\n"
                "- 파일 적용은 중단했고 Plan 모드로 돌아왔습니다."
            )
        finally:
            self.agent_mode = "Plan"
        result = self._append_run_model_repair_options(result, after_repair=after_repair)
        self.latest_message = f"{result}\n- Agent 모드: Build에서 MLflow 실행 검증 후 Plan으로 자동 전환했습니다."
        self.steps = build_beginner_step_tabs(self.project_path, applied_changes=self.applied_changes)
        if self.steps:
            self.index = min(9, len(self.steps) - 1)
        return self.render_log()

    def _append_run_model_repair_options(self, result: str, *, after_repair: bool = False) -> str:
        if after_repair:
            return result
        if not self._run_model_failed(result):
            return result
        analysis = analyze_project(self.project_path)
        fix_previews = build_fix_previews(analysis)
        serving_previews = build_serving_fix_previews(analysis)
        if not fix_previews and not serving_previews:
            self.awaiting_run_model_repair = False
            return (
                f"{result}\n\n"
                "오류 자동 보완\n"
                "- 현재 자동 수정 가능한 항목을 찾지 못했습니다.\n"
                "- 오류 내용을 확인한 뒤 run_model.py, requirements.txt, ai_studio.env를 점검하세요."
            )
        self.awaiting_run_model_repair = True
        rows = [
            result,
            "",
            "오류 자동 보완",
            "- RUN MODEL 실행 중 오류가 발생해 수정 가능한 항목을 다시 분석했습니다.",
            "- 삭제나 모델 artifact 교체는 수행하지 않습니다.",
        ]
        if fix_previews:
            rows.append("- 등록 보완 항목:")
            rows.extend(f"  - {preview.target}: {preview.title}" for preview in fix_previews)
        if serving_previews:
            rows.append("- 로컬 서빙 보완 항목:")
            rows.extend(f"  - {preview.target}: {preview.title}" for preview in serving_previews)
        rows.extend(
            [
                "- 선택:",
                "  1. 자동 보완 후 다시 실행",
                "  2. 오류만 보기",
                "  3. 취소",
            ]
        )
        return "\n".join(rows)

    def _run_model_failed(self, result: str) -> bool:
        lowered = result.lower()
        return any(
            token in lowered
            for token in (
                "status: error",
                "status: needs_action",
                "run_model.py 실행 실패",
                "오류가 발생",
                "error",
                "failed",
            )
        )

    def select_run_model_repair(self, command: str) -> str:
        if command == "1":
            return self.repair_and_rerun_local_model()
        if command == "2":
            self.awaiting_run_model_repair = False
            message = "오류 내용을 유지합니다. 파일은 수정하지 않았습니다."
            self.latest_message = message
            return self.current_screen()
        if command == "3":
            self.awaiting_run_model_repair = False
            message = "RUN MODEL 자동 보완을 취소했습니다. 파일은 수정하지 않았습니다."
            self.latest_message = message
            return self.current_screen()
        message = "번호로 선택하세요. 1=자동 보완 후 다시 실행, 2=오류만 보기, 3=취소"
        self.latest_message = message
        return message

    def repair_and_rerun_local_model(self) -> str:
        if not self.project_path:
            self.awaiting_run_model_repair = False
            message = "먼저 Step 1에서 샘플 또는 프로젝트 폴더를 선택하세요."
            self.latest_message = message
            return self.render_log()
        self.awaiting_run_model_repair = False
        self.agent_mode = "Build"
        try:
            analysis = analyze_project(self.project_path)
            fix_previews = build_fix_previews(analysis)
            serving_previews = build_serving_fix_previews(analysis)
            applied_changes = []
            if fix_previews:
                applied_changes.extend(apply_fix_previews(self.project_path, fix_previews))
            if serving_previews:
                applied_changes.extend(apply_serving_fix_previews(self.project_path, serving_previews))
            self.applied_changes = applied_changes
            repair_summary = self._format_run_model_repair_summary(applied_changes)
        except Exception as exc:
            self.agent_mode = "Plan"
            self.latest_message = (
                "RUN MODEL 자동 보완 중 오류가 발생했습니다.\n"
                f"- 오류: {exc}\n"
                "- 파일 적용은 중단했고 Plan 모드로 돌아왔습니다."
            )
            return self.current_screen()
        finally:
            self.agent_mode = "Plan"
        self.run_local_model_training(after_repair=True)
        self.latest_message = f"{repair_summary}\n\n{self.latest_message}"
        return self.render_log()

    def _format_run_model_repair_summary(self, applied_changes: list[AppliedChange]) -> str:
        applied_count = len([change for change in applied_changes if change.status == "applied"])
        skipped_count = len([change for change in applied_changes if change.status == "skipped"])
        if not applied_changes:
            return "RUN MODEL 자동 보완\n- 적용 가능한 항목이 없어 파일을 수정하지 않았습니다."
        rows = [
            "RUN MODEL 자동 보완",
            f"- 적용 완료: {applied_count}개",
            f"- 건너뜀: {skipped_count}개",
            "- 다시 RUN MODEL을 실행했습니다.",
        ]
        rows.extend(f"  - {change.target}: {change.message}" for change in applied_changes[:8])
        return "\n".join(rows)

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

    global AIOnboardingTuiApp, CancelButton, ClearButton, CommandInput, FileButton, LogView, ModeSelector, MultiAgentButton, RunModelButton, SampleButton, SendButton, StatusBar

    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, RichLog, Static, TextArea

    class LogView(RichLog):
        _last_rendered_text = ""

        def replace_text(self, text: str) -> None:
            if text == self._last_rendered_text:
                return
            self._last_rendered_text = text
            self.clear()
            self.write(text)
            self.scroll_end(animate=False)

    class CommandInput(TextArea):
        @property
        def value(self) -> str:
            return self.text

        @value.setter
        def value(self, text: str) -> None:
            self.load_text(text)

        @property
        def cursor_position(self) -> int:
            return len(self.text)

        @cursor_position.setter
        def cursor_position(self, position: int) -> None:
            self.cursor_location = self.document.get_location_from_index(position)

        def insert_text_at_cursor(self, text: str) -> None:
            self.insert(text)

        def on_mount(self) -> None:
            self.focus()

        def on_paste(self, event: events.Paste) -> None:
            event.stop()
            if hasattr(event, "prevent_default"):
                event.prevent_default()
            self.app._insert_pasted_input(event.text)

        def _handle_submit_keys(self, event) -> bool:
            if event.key in {"ctrl+l", "ctrl+u"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app._clear_command_input()
                return True
            if event.key in {"ctrl+enter", "ctrl+j"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app._submit_command_input()
                return True
            if event.key == "shift+enter":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.insert_text_at_cursor("\n")
                return True
            if event.key == "enter":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app._submit_command_input()
                return True
            if event.key == "tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app.action_toggle_agent()
                return True
            if event.key == "shift+tab":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.app.action_previous_agent()
                return True
            if event.key == "ctrl+space":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.insert_text_at_cursor("    ")
                return True
            return False

        async def _on_key(self, event: events.Key) -> None:
            if self._handle_submit_keys(event):
                return
            await super()._on_key(event)

    class ModeSelector(Static):
        pass

    class StatusBar(Static):
        pass

    class SendButton(Button):
        pass

    class FileButton(Button):
        pass

    class SampleButton(Button):
        pass

    class CancelButton(Button):
        pass

    class ClearButton(Button):
        pass

    class MultiAgentButton(Button):
        pass

    class RunModelButton(Button):
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
            overflow-y: auto;
        }
        #input-area {
            dock: bottom;
            height: 7;
            background: #080808;
        }
        #command {
            height: 4;
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
        #actions {
            height: 3;
            margin-top: 0;
        }
        #file {
            width: 14;
            height: 3;
            margin-right: 1;
            background: #2d333b;
            color: #ffffff;
            text-style: bold;
        }
        #file.build {
            background: #4a3720;
        }
        #file.chatbot {
            background: #223a2b;
        }
        #sample {
            width: 18;
            height: 3;
            margin-right: 1;
            background: #273447;
            color: #ffffff;
            text-style: bold;
        }
        #sample.build {
            background: #4a3720;
        }
        #sample.chatbot {
            background: #223a2b;
        }
        #clear {
            width: 14;
            height: 3;
            margin-right: 1;
            background: #3a3a3a;
            color: #ffffff;
            text-style: bold;
        }
        #clear.build {
            background: #4a3720;
        }
        #clear.chatbot {
            background: #223a2b;
        }
        #multi-agent {
            width: 18;
            height: 3;
            margin-right: 1;
            background: #30363d;
            color: #ffffff;
            text-style: bold;
        }
        #multi-agent.off {
            background: #1f2a36;
            color: #58a6ff;
        }
        #multi-agent.on {
            background: #3a2f18;
            color: #f0b429;
        }
        #run-model {
            width: 20;
            height: 3;
            margin-right: 1;
            background: #1f6f43;
            color: #ffffff;
            text-style: bold;
        }
        #run-model.build {
            background: #b7791f;
        }
        #run-model.chatbot {
            background: #238636;
        }
        #cancel {
            width: 14;
            height: 3;
            margin-right: 1;
            background: #5f1d1d;
            color: #ffffff;
            text-style: bold;
        }
        #cancel:disabled {
            background: #2a2a2a;
            color: #777777;
        }
        #send {
            width: 1fr;
            height: 3;
            background: #1f6feb;
            color: #ffffff;
            text-style: bold;
        }
        #send.build {
            background: #b7791f;
        }
        #send.chatbot {
            background: #238636;
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
            self._chatbot_request_id = 0
            self._active_chatbot_request_id: int | None = None
            self._active_chatbot_value = ""
            self._cancelled_chatbot_requests: set[int] = set()
            self._thinking_started_at = 0.0
            self._thinking_timer = None
            self._input_status = ""
            self._last_paste_signature = ""
            self._last_paste_at = 0.0

        def compose(self) -> ComposeResult:
            with Vertical(id="shell"):
                yield Static("AI ML Onboarding Console | ML Platform registration workflow ...", id="title")
                yield LogView(id="log", wrap=True, markup=False, auto_scroll=True)
                yield ModeSelector("", id="mode-selector")
                with Vertical(id="input-area"):
                    yield CommandInput(id="command")
                    with Horizontal(id="actions"):
                        yield FileButton("FILE", id="file")
                        yield SampleButton("SAMPLE", id="sample")
                        yield ClearButton("CLEAR", id="clear")
                        yield RunModelButton("RUN MODEL", id="run-model")
                        yield MultiAgentButton("MULTI ON", id="multi-agent")
                        yield CancelButton("CANCEL", id="cancel", disabled=True)
                        yield SendButton("SEND  Enter", id="send")
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
            if self.controller.awaiting_sample_selection:
                self.controller.cycle_sample_selection(1)
                self._refresh(force_sample_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_agent_response_choice:
                self.controller.cycle_agent_response_choice(1)
                self._refresh(force_agent_choice_value=True)
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
            if self.controller.awaiting_sample_selection:
                self.controller.cycle_sample_selection(-1)
                self._refresh(force_sample_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_agent_response_choice:
                self.controller.cycle_agent_response_choice(-1)
                self._refresh(force_agent_choice_value=True)
                self._focus_command()
                return
            self.controller.previous_agent()
            self._refresh()
            self._focus_command()

        def action_insert_input_gap(self) -> None:
            command = self.query_one(CommandInput)
            self.set_focus(command)
            command.insert_text_at_cursor("    ")

        def action_submit_input(self) -> None:
            self._submit_command_input()

        def _submit_command_input(self) -> None:
            command = self.query_one(CommandInput)
            value = command.value
            command.value = ""
            self._input_status = ""
            self._submit_or_queue(value)

        def _clear_command_input(self) -> None:
            command = self.query_one(CommandInput)
            command.value = ""
            self._input_status = ""
            self._focus_command()

        def _insert_pasted_input(self, raw_text: str) -> None:
            command = self.query_one(CommandInput)
            normalized = normalize_pasted_input(raw_text)
            now = time.monotonic()
            signature = f"{len(normalized)}:{hash(normalized)}"
            if signature == self._last_paste_signature and now - self._last_paste_at <= PASTE_DEDUP_SECONDS:
                self._input_status = paste_status_message(normalized, duplicate=True)
                self.query_one(StatusBar).update(self._input_status)
                self._focus_command()
                return
            self._last_paste_signature = signature
            self._last_paste_at = now
            command.insert_text_at_cursor(normalized)
            command.cursor_position = len(command.value)
            self._input_status = paste_status_message(normalized)
            self.query_one(StatusBar).update(self._input_status)
            self._focus_command()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "file":
                event.stop()
                self._open_folder_picker()
                return
            if event.button.id == "sample":
                event.stop()
                self._open_sample_picker()
                return
            if event.button.id == "clear":
                event.stop()
                self._clear_command_input()
                return
            if event.button.id == "multi-agent":
                event.stop()
                self._toggle_multi_agent()
                return
            if event.button.id == "run-model":
                event.stop()
                self._run_local_model_training()
                return
            if event.button.id == "cancel":
                event.stop()
                self._cancel_chatbot_request()
                return
            if event.button.id != "send":
                return
            event.stop()
            self._submit_command_input()

        def _toggle_multi_agent(self) -> None:
            message = self.controller.toggle_multi_agent()
            self._input_status = message
            self._refresh()
            self._focus_command()

        def _run_local_model_training(self) -> None:
            self._input_status = "RUN MODEL 실행 중..."
            self.query_one(StatusBar).update(self._input_status)
            self.controller.run_local_model_training()
            self._refresh()
            self._focus_command()

        def _cancel_chatbot_request(self) -> None:
            request_id = self._active_chatbot_request_id
            if not self._chatbot_busy or request_id is None:
                self.controller.latest_message = "취소할 CHAT 요청이 없습니다."
                self._refresh()
                self._focus_command()
                return
            self._cancelled_chatbot_requests.add(request_id)
            self._chatbot_busy = False
            self._stop_thinking_animation()
            message = "CHAT 요청을 취소했습니다. 늦게 도착한 응답은 표시하지 않습니다."
            self.controller.latest_message = message
            if self._active_chatbot_value:
                self.controller._append_or_replace_chat_log(self._active_chatbot_value, message, [])
            self._refresh()
            self._focus_command()

        def _open_folder_picker(self) -> None:
            command = self.query_one(CommandInput)
            base = command.value.strip()
            command.value = ""
            if self.controller.selected_launch_mode is None:
                self.controller.latest_message = "먼저 모드를 선택하세요. 1=초급자, 2=중급자, 3=고급자"
                self._refresh()
                self._focus_command()
                return
            selected = choose_folder_with_dialog(base)
            if selected is not None:
                self.controller.select_project_path(selected)
                self.controller.latest_message = (
                    "폴더 선택창에서 프로젝트 폴더를 선택했습니다.\n"
                    f"- 위치: {self.controller.project_path}"
                )
                self._refresh()
                self._focus_command()
                return
            self.controller.start_folder_selection(base)
            self._refresh(force_folder_value=True)
            self._focus_command()

        def _open_sample_picker(self) -> None:
            command = self.query_one(CommandInput)
            command.value = ""
            if self.controller.selected_launch_mode is None:
                self.controller.latest_message = "먼저 모드를 선택하세요. 1=초급자, 2=중급자, 3=고급자"
                self._refresh()
                self._focus_command()
                return
            self.controller.start_sample_selection()
            self._refresh(force_sample_value=True)
            self._focus_command()

        def on_click(self) -> None:
            self._focus_command()

        def on_mouse_down(self, event: events.MouseDown) -> None:
            if not is_right_click_event(event):
                return
            event.stop()
            if hasattr(event, "prevent_default"):
                event.prevent_default()
            self.action_copy_current_screen()

        def on_paste(self, event: events.Paste) -> None:
            command = self.query_one(CommandInput)
            if self.focused is command:
                return
            event.stop()
            if hasattr(event, "prevent_default"):
                event.prevent_default()
            self._focus_command()
            self._insert_pasted_input(event.text)

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
            if self.controller.awaiting_sample_selection and event.key in {"up", "left", "shift+tab"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_sample_selection(-1)
                self._refresh(force_sample_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_agent_response_choice and event.key in {"up", "left", "shift+tab"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_agent_response_choice(-1)
                self._refresh(force_agent_choice_value=True)
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
            if self.controller.awaiting_sample_selection and event.key in {"down", "right"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_sample_selection(1)
                self._refresh(force_sample_value=True)
                self._focus_command()
                return
            if self.controller.awaiting_agent_response_choice and event.key in {"down", "right"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self.controller.cycle_agent_response_choice(1)
                self._refresh(force_agent_choice_value=True)
                self._focus_command()
                return
            if event.key == "ctrl+space":
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                command.insert_text_at_cursor("    ")
                self._focus_command()
                return
            if event.key in {"ctrl+l", "ctrl+u"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self._clear_command_input()
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
            if event.key in {"ctrl+enter", "ctrl+j"}:
                event.stop()
                if hasattr(event, "prevent_default"):
                    event.prevent_default()
                self._submit_command_input()
                return

        def action_copy_current_screen(self) -> None:
            text = normalize_clipboard_text(self.controller.render_log())
            copied, detail = copy_text_to_clipboard(text)
            if copied:
                self.controller.latest_message = f"현재 화면을 클립보드에 복사했습니다. ({detail}, UTF-8)"
            else:
                try:
                    self.copy_to_clipboard(text)
                    self.controller.latest_message = "현재 화면을 클립보드에 복사했습니다. (Textual fallback)"
                except Exception:
                    self.controller.latest_message = f"클립보드 복사에 실패했습니다: {detail}"
            self._refresh()
            self._focus_command()

        def _submit_or_queue(self, value: str) -> None:
            display_value = value
            submit_value = value
            if self.controller.awaiting_agent_response_choice:
                submit_value, display_value = self.controller.prepare_agent_response_choice_submission(value)
            if self.controller.should_show_thinking(submit_value):
                if self._chatbot_busy:
                    self.controller.latest_message = (
                        "이전 CHAT 요청을 정리하는 중입니다. 잠시 후 다시 입력하거나 Esc로 종료하세요."
                    )
                    self._refresh()
                    self._focus_command()
                    return
                self._chatbot_busy = True
                self._chatbot_request_id += 1
                request_id = self._chatbot_request_id
                self._active_chatbot_request_id = request_id
                self._active_chatbot_value = display_value.strip()
                self.controller.set_thinking(display_value)
                self._start_thinking_animation()
                self._refresh()
                self._focus_command()
                self.set_timer(
                    0.05,
                    lambda: self._start_submit_worker(submit_value, request_id),
                    name="chatbot-submit",
                )
                return
            self._submit_value(submit_value)

        def _start_submit_worker(self, value: str, request_id: int) -> None:
            if request_id in self._cancelled_chatbot_requests:
                self._finish_submit(request_id, value)
                return
            try:
                Thread(target=self._submit_value_in_thread, args=(value, request_id), daemon=True).start()
            except Exception as exc:  # pragma: no cover - UI safety boundary
                self.controller._append_chat_error(value, exc)
                self._finish_submit(request_id, value)

        def _submit_value_in_thread(self, value: str, request_id: int) -> None:
            try:
                if request_id not in self._cancelled_chatbot_requests:
                    self.controller.submit(value)
            except BaseException as exc:  # pragma: no cover - UI safety boundary
                if request_id not in self._cancelled_chatbot_requests:
                    self.controller._append_chat_error(value, exc)
            finally:
                try:
                    self.call_from_thread(lambda: self._finish_submit(request_id, value))
                except RuntimeError:
                    if self._active_chatbot_request_id == request_id:
                        self._chatbot_busy = False

        def _finish_submit(self, request_id: int | None = None, value: str = "") -> None:
            if request_id is not None and request_id in self._cancelled_chatbot_requests:
                self._cancelled_chatbot_requests.discard(request_id)
                self._stop_thinking_animation()
                self.controller.latest_message = "CHAT 요청을 취소했습니다."
                if value.strip():
                    self.controller._append_or_replace_chat_log(value.strip(), "CHAT 요청을 취소했습니다.", [])
                if self._active_chatbot_request_id == request_id:
                    self._active_chatbot_request_id = None
                    self._active_chatbot_value = ""
                self._refresh()
                self._focus_command()
                return
            if request_id is not None and self._active_chatbot_request_id not in {None, request_id}:
                return
            self._chatbot_busy = False
            self._stop_thinking_animation()
            self._active_chatbot_request_id = None
            self._active_chatbot_value = ""
            if self.controller.exited:
                self.exit()
                return
            self._refresh()
            self._focus_command()

        def _start_thinking_animation(self) -> None:
            self._stop_thinking_animation()
            self._thinking_started_at = time.monotonic()
            self._thinking_timer = self.set_interval(1.0, self._tick_thinking_animation, name="chatbot-thinking")

        def _stop_thinking_animation(self) -> None:
            timer = self._thinking_timer
            self._thinking_timer = None
            if timer is not None:
                try:
                    timer.stop()
                except AttributeError:
                    try:
                        timer.pause()
                    except AttributeError:
                        pass

        def _tick_thinking_animation(self) -> None:
            if not self._chatbot_busy or not self._active_chatbot_value:
                self._stop_thinking_animation()
                return
            elapsed = int(time.monotonic() - self._thinking_started_at)
            self.controller.set_thinking(self._active_chatbot_value, elapsed_seconds=elapsed)
            self._refresh()

        def _submit_value(self, value: str) -> None:
            self.controller.submit(value)
            if self.controller.exited:
                self.exit()
                return
            self._refresh()
            self._focus_command()

        def _refresh(
            self,
            force_model_value: bool = False,
            force_folder_value: bool = False,
            force_sample_value: bool = False,
            force_agent_choice_value: bool = False,
        ) -> None:
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
            elif self.controller.awaiting_sample_selection:
                command.placeholder = sample_selection_placeholder(SAMPLE_CHOICES)
                if force_sample_value or not command.value:
                    command.value = self.controller.highlighted_sample_command
                    command.cursor_position = len(command.value)
            elif self.controller.awaiting_agent_response_choice:
                command.placeholder = agent_response_choice_placeholder(self.controller.agent_response_choices)
                if force_agent_choice_value or not command.value:
                    command.value = str(self.controller.agent_response_choice_index + 1)
                    command.cursor_position = len(command.value)
            elif self.controller.awaiting_chat_code_policy:
                command.placeholder = "1=safe 적용, 2=검토 미리보기, 3=전체 승인, 4=취소"
            else:
                command.placeholder = command_placeholder_for_mode(self.controller.agent_mode, self.controller.qwen_model)
            command.set_class(self.controller.agent_mode == "Build", "build")
            command.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            send = self.query_one(SendButton)
            send.set_class(self.controller.agent_mode == "Build", "build")
            send.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            cancel = self.query_one(CancelButton)
            cancel.disabled = not self._chatbot_busy
            file_button = self.query_one(FileButton)
            file_button.set_class(self.controller.agent_mode == "Build", "build")
            file_button.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            sample_button = self.query_one(SampleButton)
            sample_button.set_class(self.controller.agent_mode == "Build", "build")
            sample_button.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            clear_button = self.query_one(ClearButton)
            clear_button.set_class(self.controller.agent_mode == "Build", "build")
            clear_button.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            multi_button = self.query_one(MultiAgentButton)
            multi_button.label = "MULTI ON" if self.controller.multi_agent_enabled else "MULTI OFF"
            multi_button.set_class(self.controller.multi_agent_enabled, "on")
            multi_button.set_class(not self.controller.multi_agent_enabled, "off")
            run_button = self.query_one(RunModelButton)
            run_button.set_class(self.controller.agent_mode == "Build", "build")
            run_button.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            self.query_one(LogView).replace_text(self.controller.render_log())
            selector = self.query_one(ModeSelector)
            if self.controller.selected_launch_mode is None:
                selector.update("beginner intermediate advanced")
            else:
                selector.update(format_agent_mode_selector(self.controller.agent_mode))
            selector.set_class(self.controller.agent_mode == "Plan", "plan")
            selector.set_class(self.controller.agent_mode == "Build", "build")
            selector.set_class(self.controller.agent_mode == "Chatbot", "chatbot")
            self.query_one(StatusBar).update(self._input_status)

        def _focus_command(self) -> None:
            self.set_focus(self.query_one(CommandInput))

    AIOnboardingTuiApp(project_path).run()
    return 0


__all__ = [
    "AIOnboardingTuiApp",
    "AGENT_MODES",
    "BeginnerTuiController",
    "agent_response_choice_placeholder",
    "available_models_from_config",
    "CommandInput",
    "CancelButton",
    "ClearButton",
    "FileButton",
    "SampleButton",
    "discover_selectable_folders",
    "extract_agent_response_choices",
    "folder_selection_placeholder",
    "format_agent_response_choices",
    "format_folder_choices",
    "format_sample_choices",
    "ModeSelector",
    "MultiAgentButton",
    "RunModelButton",
    "SendButton",
    "LogView",
    "StatusBar",
    "format_agent_mode_selector",
    "format_model_choices",
    "format_thinking_animation",
    "format_tui_model_info",
    "format_tui_chatbot_screen",
    "format_tui_help_screen",
    "classify_chat_fix_previews",
    "build_compacted_runtime_prompt",
    "compact_chat_entries",
    "format_chat_code_policy_plan",
    "is_fix_request",
    "is_chat_apply_approved",
    "is_chat_coding_request",
    "is_greeting",
    "is_long_paste",
    "is_right_click_event",
    "is_wizard_navigation",
    "model_selection_placeholder",
    "sample_selection_placeholder",
    "normalize_input_path",
    "normalize_pasted_input",
    "normalize_path_text",
    "path_candidates_from_input",
    "paste_status_message",
    "parse_agent_mode_command",
    "parse_folder_command",
    "parse_model_command",
    "should_use_autofix_chat",
    "strip_path_command",
    "missing_textual_message",
    "run_tui",
    "textual_available",
]
