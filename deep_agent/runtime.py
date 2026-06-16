"""DeepAgents runtime bridge for the AI ML onboarding TUI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_agent.app_config import AppConfig, ensure_local_file_access
from deep_agent.cli import (
    analyze_project,
    apply_fix_previews,
    build_fix_previews,
    format_beginner_analysis,
    format_beginner_apply_result,
    format_beginner_fix_preview,
)
from deep_agent.path_utils import resolve_filesystem_path
from deep_agent.qwen_chat import QwenChatConfig


LOCAL_DEEPAGENTS_LIB = (
    Path(__file__).resolve().parent / "vendor" / "deepagents" / "deepagents-main" / "libs" / "deepagents"
)


def _ensure_local_deepagents_on_path() -> None:
    if LOCAL_DEEPAGENTS_LIB.exists():
        path = str(LOCAL_DEEPAGENTS_LIB)
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass(frozen=True)
class DeepAgentsRunResult:
    content: str
    used_deepagents: bool
    error: str | None = None


class DeepAgentsRuntime:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.app_config = config or AppConfig.load()
        self.qwen_config = QwenChatConfig.from_app_config(self.app_config)
        self._agent_cache: dict[tuple[str, str, str, str], Any] = {}

    def is_configured(self) -> bool:
        return self.qwen_config.is_configured()

    def invoke(self, prompt: str, *, project_path: str = "", agent_mode: str = "Plan") -> DeepAgentsRunResult:
        if not self.is_configured():
            return DeepAgentsRunResult(
                content=(
                    "DeepAgents runtime은 Qwen 연결 설정이 필요합니다. "
                    ".env에 QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL 값을 설정하세요."
                ),
                used_deepagents=False,
                error="qwen_not_configured",
            )
        try:
            agent = self._create_agent(project_path=project_path, agent_mode=agent_mode)
            result = agent.invoke({"messages": prompt})
            return DeepAgentsRunResult(content=extract_deepagents_content(result), used_deepagents=True)
        except Exception as exc:  # pragma: no cover - dependency/provider boundary
            return DeepAgentsRunResult(
                content=f"DeepAgents runtime 실행 실패: {exc}",
                used_deepagents=False,
                error=type(exc).__name__,
            )

    def _create_agent(self, *, project_path: str, agent_mode: str):
        use_cache = agent_mode != "Chat"
        resolved_project = resolve_filesystem_path(project_path) if project_path else None
        if resolved_project is not None and resolved_project.exists():
            ensure_local_file_access(resolved_project)
            if resolved_project.is_file():
                resolved_project = resolved_project.parent
        resolved_project_path = str(resolved_project) if resolved_project is not None else ""
        cache_key = (
            resolved_project_path,
            agent_mode,
            self.qwen_config.model,
            self.qwen_config.base_url.rstrip("/"),
        )
        if use_cache and cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        _ensure_local_deepagents_on_path()
        from deepagents import create_deep_agent  # type: ignore
        from deepagents.backends import FilesystemBackend  # type: ignore
        from deepagents.middleware.filesystem import FilesystemPermission  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore

        model = ChatOpenAI(
            model=self.qwen_config.model,
            api_key=self.qwen_config.api_key,
            base_url=self.qwen_config.base_url.rstrip("/"),
            temperature=0.2,
            timeout=self.app_config.get_int("DEV_COMMAND_TIMEOUT", default=120),
        )
        tools = [] if agent_mode == "Chat" else [analyze_ml_project, preview_ml_fixes, apply_ml_fixes]
        backend = None
        permissions = None
        if resolved_project is not None and resolved_project.exists() and resolved_project.is_dir():
            backend = FilesystemBackend(root_dir=str(resolved_project), virtual_mode=True)
            write_mode = "allow" if agent_mode in {"Build", "AutoFix"} else "deny"
            permissions = [
                FilesystemPermission(operations=["read"], paths=["/**"], mode="allow"),
                FilesystemPermission(operations=["write"], paths=["/**"], mode=write_mode),
            ]
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=build_deepagents_system_prompt(resolved_project_path or project_path, agent_mode),
            backend=backend,
            permissions=permissions,
            name="ai-ml-onboarding-deepagent",
        )
        if use_cache:
            self._agent_cache[cache_key] = agent
        return agent


def build_deepagents_system_prompt(project_path: str, agent_mode: str) -> str:
    if agent_mode == "AutoFix":
        apply_policy = (
            "AutoFix mode: first call analyze_ml_project. If fixable issues are found, call preview_ml_fixes "
            "and apply_ml_fixes automatically. Do not create unsupported artifacts, delete files, or perform "
            "large structural rewrites. Explain manual-only issues after applying supported fixes."
        )
    elif agent_mode == "Build":
        apply_policy = "Build mode: if the user asks to fix, modify, patch, or apply changes, you may call apply_ml_fixes."
    elif agent_mode == "Chat":
        apply_policy = (
            "Chat mode: answer directly and concisely. Do not modify files. "
            "Do not run project analysis unless the user explicitly asks for analysis, fixes, validation, or registration readiness."
        )
    else:
        apply_policy = "Plan mode: do not modify files. Use analyze_ml_project and preview_ml_fixes only."
    return (
        "You are AI ML Onboarding Deep Agent for a closed-network ML Platform POC. "
        "Answer in Korean. Use the provided ML onboarding tools before giving conclusions. "
        "Focus on MLflow, requirements, entrypoints, job templates, serving, and registration readiness. "
        f"Current project path: {project_path or '(not selected)'}. {apply_policy} "
        "Never claim files changed unless apply_ml_fixes reports applied changes."
    )


def analyze_ml_project(project_path: str) -> str:
    """Analyze an ML project registration readiness without modifying files."""
    return format_beginner_analysis(analyze_project(project_path))


def preview_ml_fixes(project_path: str) -> str:
    """Return dry-run ML onboarding fix previews without modifying files."""
    return format_beginner_fix_preview(analyze_project(project_path))


def apply_ml_fixes(project_path: str) -> str:
    """Apply supported ML onboarding fixes, then re-analyze the project."""
    analysis = analyze_project(project_path)
    previews = build_fix_previews(analysis)
    applied = apply_fix_previews(project_path, previews)
    return format_beginner_apply_result(applied, analyze_project(project_path))


def extract_deepagents_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            if isinstance(content, list):
                return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
            if content is not None:
                return str(content)
        if "output" in result:
            return str(result["output"])
    return str(result)


__all__ = [
    "DeepAgentsRunResult",
    "DeepAgentsRuntime",
    "analyze_ml_project",
    "apply_ml_fixes",
    "build_deepagents_system_prompt",
    "extract_deepagents_content",
    "preview_ml_fixes",
]
