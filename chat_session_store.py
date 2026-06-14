"""Chat session persistence for the TUI chatbot."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app_config import AppConfig


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


def append_chat_session_event(config: AppConfig, event: dict[str, Any], now: datetime | None = None) -> Path:
    path = chat_session_path(config, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    "chat_session_path",
    "mask_sensitive_payload",
    "mask_sensitive_text",
]
