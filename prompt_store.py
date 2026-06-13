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
