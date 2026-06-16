"""Chat session persistence for the TUI chatbot."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deep_agent.app_config import AppConfig, ensure_read_write_directory


SENSITIVE_KEYS = ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "BASE_URL")


def mask_sensitive_text(value: str, config: AppConfig) -> str:
    if not config.get_bool("MASK_SENSITIVE_LOGS"):
        return value
    masked = value
    for key in SENSITIVE_KEYS:
        secret = config.get(f"QWEN_{key}") if key in {"API_KEY", "BASE_URL"} else config.get(key)
        if secret and secret not in {"your-internal-qwen-key", "http://xxx.xxx.xxx.xxx:port/v1"}:
            masked = masked.replace(secret, "***")
    return masked


def mask_sensitive_payload(payload: Any, config: AppConfig) -> Any:
    if isinstance(payload, str):
        return mask_sensitive_text(payload, config)
    if isinstance(payload, list):
        return [mask_sensitive_payload(item, config) for item in payload]
    if isinstance(payload, dict):
        return {key: mask_sensitive_payload(value, config) for key, value in payload.items()}
    return payload


def chat_session_path(config: AppConfig, now: datetime | None = None) -> Path:
    timestamp = now or datetime.now(timezone.utc)
    session_dir = config.root_dir / (config.get("SESSION_DIR") or "sessions")
    return session_dir / f"chat-session-{timestamp:%Y%m%d}.jsonl"


def chat_context_summary_path(config: AppConfig) -> Path:
    session_dir = config.root_dir / (config.get("SESSION_DIR") or "sessions")
    return session_dir / "chat-context-summary.json"


def load_chat_context_summary(config: AppConfig) -> dict[str, Any]:
    path = chat_context_summary_path(config)
    if not path.exists():
        return {"summary": "", "message_count": 0, "updated_at": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"summary": "", "message_count": 0, "updated_at": ""}
    if not isinstance(payload, dict):
        return {"summary": "", "message_count": 0, "updated_at": ""}
    return payload


def save_chat_context_summary(config: AppConfig, summary: str, message_count: int) -> Path:
    path = chat_context_summary_path(config)
    ensure_read_write_directory(path.parent)
    payload = mask_sensitive_payload(
        {
            "summary": summary,
            "message_count": message_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        config,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_chat_session_event(config: AppConfig, event: dict[str, Any], now: datetime | None = None) -> Path:
    path = chat_session_path(config, now=now)
    ensure_read_write_directory(path.parent)
    timestamp = now or datetime.now(timezone.utc)
    payload = {
        "timestamp": timestamp.isoformat(),
        **event,
    }
    payload = mask_sensitive_payload(payload, config)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


__all__ = [
    "append_chat_session_event",
    "chat_context_summary_path",
    "chat_session_path",
    "load_chat_context_summary",
    "mask_sensitive_payload",
    "mask_sensitive_text",
    "save_chat_context_summary",
]
