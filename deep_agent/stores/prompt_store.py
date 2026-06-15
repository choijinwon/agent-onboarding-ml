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


def append_used_prompt_to_wiki(
    config: AppConfig,
    event: dict[str, Any],
    now: datetime | None = None,
) -> list[Path]:
    jsonl_path, markdown_path = used_prompts_paths(config)
    ensure_read_write_directory(jsonl_path.parent)
    timestamp = now or datetime.now(timezone.utc)
    payload = mask_sensitive_payload({"timestamp": timestamp.isoformat(), **event}, config)
    with jsonl_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    if not markdown_path.exists():
        markdown_path.write_text("# Used Prompts\n\nAI ML Onboarding Console에서 실제 사용한 프롬프트 기록입니다.\n", encoding="utf-8")
    with markdown_path.open("a", encoding="utf-8") as file:
        file.write(format_used_prompt_markdown(payload))
    return [jsonl_path, markdown_path]


def format_used_prompt_markdown(payload: dict[str, Any]) -> str:
    user_prompt = str(payload.get("user_prompt", "")).strip()
    system_prompt = str(payload.get("system_prompt", "")).strip()
    response_summary = str(payload.get("response_summary", "")).strip()
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
    "export_prompt_templates_to_wiki",
    "format_prompt_templates",
    "format_prompt_templates_markdown",
    "format_used_prompt_markdown",
    "load_prompt_templates",
    "prompt_templates_as_dict",
    "used_prompts_paths",
]
