"""Prompt template storage helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app_config import AppConfig


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
    prompt_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = prompt_dir / "prompt_templates.md"
    json_path = prompt_dir / "prompt_templates.json"
    markdown_path.write_text(format_prompt_templates_markdown(templates), encoding="utf-8")
    json_path.write_text(
        json.dumps(prompt_templates_as_dict(templates), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [markdown_path, json_path]


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
