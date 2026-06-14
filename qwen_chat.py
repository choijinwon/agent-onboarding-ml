"""OpenAI-compatible Qwen chat helper for the TUI."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from app_config import AppConfig


UrlOpen = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class QwenChatConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "QwenChatConfig":
        return cls(
            api_key=config.get("QWEN_API_KEY"),
            base_url=config.get("QWEN_BASE_URL"),
            model=config.get("QWEN_MODEL") or "qwen3.6",
        )

    def is_configured(self) -> bool:
        key = self.api_key.strip()
        base_url = self.base_url.strip()
        return bool(key and base_url) and not key.startswith("your-") and "xxx.xxx" not in base_url

    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


def build_qwen_system_prompt(project_path: str, agent_mode: str) -> str:
    return (
        "You are AI ML Onboarding Assistant for a closed-network ML Platform POC. "
        "Answer in Korean. Keep responses concise and practical. "
        "Focus on MLflow, requirements, train entrypoints, job templates, serving, and registration readiness. "
        "Do not claim files were modified unless a tool path actually applied changes. "
        f"Current project path: {project_path or '(not selected)'}. Current mode: {agent_mode}."
    )


def chat_with_qwen(
    prompt: str,
    *,
    config: QwenChatConfig,
    project_path: str = "",
    agent_mode: str = "Plan",
    urlopen: UrlOpen | None = None,
) -> str:
    if not config.is_configured():
        return (
            "Qwen 3.6 연결 설정이 아직 샘플 값입니다.\n"
            "`.env`에 QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL=qwen3.6 값을 넣으면 "
            "이 입력창에서 바로 Qwen에게 질문할 수 있습니다."
        )

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": build_qwen_system_prompt(project_path, agent_mode)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        config.endpoint(),
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = urlopen or urllib.request.urlopen
    try:
        with opener(request, config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return f"Qwen 연결 실패: {exc.reason}"
    except TimeoutError:
        return "Qwen 연결 시간이 초과되었습니다. QWEN_BASE_URL과 네트워크를 확인하세요."
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        return f"Qwen 응답 처리 실패: {exc}"

    choices = data.get("choices") or []
    if not choices:
        return "Qwen 응답이 비어 있습니다. 모델명과 endpoint를 확인하세요."
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return str(content).strip() or "Qwen 응답 내용이 비어 있습니다."


__all__ = [
    "QwenChatConfig",
    "build_qwen_system_prompt",
    "chat_with_qwen",
]
