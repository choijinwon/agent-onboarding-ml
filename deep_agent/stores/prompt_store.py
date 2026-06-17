"""Prompt template storage helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deep_agent.app_config import AppConfig, ensure_read_write_directory
from deep_agent.stores.chat_session_store import mask_sensitive_payload


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    description: str
    prompt: str
    tags: list[str]


def load_prompt_templates(config: AppConfig | None = None) -> list[PromptTemplate]:
    active_config = AppConfig.load() if config is None else config
    path = active_config.root_dir / active_config.get("PROMPT_STORE_PATH")
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    templates = payload.get("templates", [])
    return [
        PromptTemplate(
            name=item["name"],
            description=item.get("description", ""),
            prompt=item.get("prompt", ""),
            tags=list(item.get("tags", [])),
        )
        for item in templates
    ]


def format_prompt_templates(templates: list[PromptTemplate]) -> str:
    if not templates:
        return "Prompt templates: none"
    rows = "\n".join(
        f"- {template.name}: {template.description} [{', '.join(template.tags)}]"
        for template in templates
    )
    return f"Prompt templates: {len(templates)}\n{rows}"


def format_prompt_templates_markdown(templates: list[PromptTemplate]) -> str:
    lines = [
        "# Prompt Templates",
        "",
        "AI ML Onboarding Console에서 사용하는 기본 프롬프트 모음입니다.",
        "",
        f"Total: {len(templates)}",
        "",
    ]
    for template in templates:
        tags = ", ".join(template.tags) if template.tags else "-"
        lines.extend(
            [
                f"## {template.name}",
                "",
                f"- Description: {template.description or '-'}",
                f"- Tags: {tags}",
                "",
                "```text",
                template.prompt.strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def export_prompt_templates_to_wiki(config: AppConfig | None = None) -> list[Path]:
    active_config = AppConfig.load() if config is None else config
    templates = load_prompt_templates(active_config)
    prompt_dir = active_config.root_dir / active_config.get("WIKI_PROMPT_DIR")
    ensure_read_write_directory(prompt_dir)

    markdown_path = prompt_dir / "prompt_templates.md"
    json_path = prompt_dir / "prompt_templates.json"
    markdown_path.write_text(format_prompt_templates_markdown(templates), encoding="utf-8")
    json_path.write_text(
        json.dumps(prompt_templates_as_dict(templates), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [markdown_path, json_path]


def used_prompts_paths(config: AppConfig) -> tuple[Path, Path]:
    prompt_dir = config.root_dir / config.get("WIKI_PROMPT_DIR")
    return prompt_dir / "used_prompts.jsonl", prompt_dir / "used_prompts.md"


def dated_used_prompts_paths(config: AppConfig, timestamp: datetime) -> tuple[Path, Path]:
    prompt_dir = config.root_dir / config.get("WIKI_PROMPT_DIR") / "used"
    date_name = timestamp.astimezone().date().isoformat()
    return prompt_dir / f"{date_name}.jsonl", prompt_dir / f"{date_name}.md"


def append_used_prompt_to_wiki(
    config: AppConfig,
    event: dict[str, Any],
    now: datetime | None = None,
) -> list[Path]:
    timestamp = now or datetime.now(timezone.utc)
    jsonl_path, markdown_path = used_prompts_paths(config)
    dated_jsonl_path, dated_markdown_path = dated_used_prompts_paths(config, timestamp)
    ensure_read_write_directory(jsonl_path.parent)
    ensure_read_write_directory(dated_jsonl_path.parent)
    payload = mask_sensitive_payload({"timestamp": timestamp.isoformat(), **event}, config)
    for path in (jsonl_path, dated_jsonl_path):
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    for path, title in (
        (markdown_path, "# Used Prompts\n\nAI ML Onboarding Console에서 실제 사용한 프롬프트 기록입니다.\n"),
        (dated_markdown_path, f"# Used Prompts {timestamp.astimezone().date().isoformat()}\n\n날짜별 Agent 프롬프트/응답 기록입니다.\n"),
    ):
        if not path.exists():
            path.write_text(title, encoding="utf-8")
        with path.open("a", encoding="utf-8") as file:
            file.write(format_used_prompt_markdown(payload))
    return [jsonl_path, markdown_path, dated_jsonl_path, dated_markdown_path]


def format_used_prompt_markdown(payload: dict[str, Any]) -> str:
    user_prompt = str(payload.get("user_prompt", "")).strip()
    system_prompt = str(payload.get("system_prompt", "")).strip()
    response_summary = str(payload.get("response_summary", "")).strip()
    agent_response = str(payload.get("agent_response", "")).strip()
    lines = [
        "",
        f"## {payload.get('timestamp', '')}",
        "",
        f"- Mode: {payload.get('agent_mode', '-')}",
        f"- Launch Mode: {payload.get('launch_mode', '-')}",
        f"- Model: {payload.get('selected_model', '-')}",
        f"- Project: {payload.get('project_path', '-')}",
        "",
        "### User Prompt",
        "",
        "```text",
        user_prompt,
        "```",
        "",
    ]
    if system_prompt:
        lines.extend(
            [
                "### System Prompt",
                "",
                "```text",
                system_prompt,
                "```",
                "",
            ]
        )
    if response_summary:
        lines.extend(
            [
                "### Response Summary",
                "",
                "```text",
                response_summary[:2000],
                "```",
                "",
            ]
        )
    if agent_response:
        lines.extend(
            [
                "### Agent Response",
                "",
                "```text",
                agent_response,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def load_recent_used_prompt(config: AppConfig, today_only: bool = False) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    jsonl_path = dated_used_prompts_paths(config, now)[0] if today_only else used_prompts_paths(config)[0]
    if not jsonl_path.exists():
        return None
    lines = [line.strip() for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def format_wiki_recent_prompt_for_tui(config: AppConfig) -> str:
    payload = load_recent_used_prompt(config)
    if payload is None:
        return "Wiki에 저장된 Agent 응답이 아직 없습니다."
    dated_markdown = dated_used_prompts_paths(config, datetime.fromisoformat(str(payload["timestamp"])))[1]
    response = str(payload.get("agent_response") or payload.get("response_summary") or "").strip()
    preview_lines = response.splitlines()[:12]
    preview = "\n".join(preview_lines) if preview_lines else "(응답 내용 없음)"
    return (
        "Wiki 최근 Agent 응답\n"
        f"- 저장 파일: {dated_markdown}\n"
        f"- 저장 시간: {payload.get('timestamp', '-')}\n"
        f"- 모델: {payload.get('selected_model', '-')}\n"
        f"- 프로젝트: {payload.get('project_path', '-')}\n"
        "- 응답 미리보기:\n"
        f"{preview}"
    )


def format_wiki_today_for_tui(config: AppConfig) -> str:
    now = datetime.now(timezone.utc)
    jsonl_path, markdown_path = dated_used_prompts_paths(config, now)
    if not jsonl_path.exists():
        return f"오늘 저장된 Wiki 응답이 없습니다.\n- 예상 파일: {markdown_path}"
    rows = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "Wiki 오늘 저장 목록",
        f"- 저장 파일: {markdown_path}",
        f"- 저장 건수: {len(rows)}",
    ]
    for index, line in enumerate(rows[-5:], start=max(1, len(rows) - 4)):
        payload = json.loads(line)
        prompt = str(payload.get("user_prompt", "")).replace("\n", " ")
        lines.append(f"  {index}. {payload.get('timestamp', '-')} | {prompt[:80]}")
    return "\n".join(lines)


def prompt_templates_as_dict(templates: list[PromptTemplate]) -> dict[str, object]:
    return {
        "templates": [
            {
                "name": template.name,
                "description": template.description,
                "prompt": template.prompt,
                "tags": template.tags,
            }
            for template in templates
        ]
    }


__all__ = [
    "PromptTemplate",
    "append_used_prompt_to_wiki",
    "dated_used_prompts_paths",
    "export_prompt_templates_to_wiki",
    "format_wiki_recent_prompt_for_tui",
    "format_wiki_today_for_tui",
    "format_prompt_templates",
    "format_prompt_templates_markdown",
    "format_used_prompt_markdown",
    "load_recent_used_prompt",
    "load_prompt_templates",
    "prompt_templates_as_dict",
    "used_prompts_paths",
]
