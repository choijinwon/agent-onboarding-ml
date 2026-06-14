"""DeepAgents libs manifest and runtime availability helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass


DEEPAGENTS_LIBS_REFERENCE = "https://github.com/langchain-ai/deepagents/tree/main/libs"


@dataclass(frozen=True)
class DeepAgentsLibSpec:
    name: str
    path: str
    purpose: str
    poc_usage: str
    required_now: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DEEPAGENTS_LIBS = [
    DeepAgentsLibSpec(
        name="deepagents",
        path="libs/deepagents",
        purpose="Core Deep Agents runtime, create_deep_agent harness, subagents, skills, context engineering.",
        poc_usage="실제 LLM runtime 연결 시 ai-ml-onboarding profile을 create_deep_agent 설정으로 옮기는 대상.",
        required_now=True,
    ),
    DeepAgentsLibSpec(
        name="deepagents-acp",
        path="libs/acp",
        purpose="Agent Client Protocol integration.",
        poc_usage="향후 외부 agent/client 연계가 필요할 때 선택 적용.",
    ),
    DeepAgentsLibSpec(
        name="deepagents-evals",
        path="libs/evals",
        purpose="Deep Agent behavior and regression evaluation utilities.",
        poc_usage="Wizard 수정 전후 회귀 테스트와 agent 품질 평가에 연결.",
    ),
    DeepAgentsLibSpec(
        name="deepagents-code",
        path="libs/code",
        purpose="Textual-style code/TUI assistant surface.",
        poc_usage="초급자 Wizard TUI 화면의 참고 구현 축.",
    ),
    DeepAgentsLibSpec(
        name="deepagents-cli",
        path="libs/cli",
        purpose="Command-line interface for Deep Agents workflows.",
        poc_usage="고급자 모드 CLI 명령 체계와 자동화 파이프라인 참고.",
    ),
    DeepAgentsLibSpec(
        name="deepagents-partners",
        path="libs/partners",
        purpose="Partner integrations and extension packages.",
        poc_usage="내부 LLM/Qwen provider 연결 확장 시 참고.",
    ),
]


def deepagents_runtime_available() -> bool:
    return importlib.util.find_spec("deepagents") is not None


def deepagents_libs_as_dict() -> dict[str, object]:
    return {
        "reference": DEEPAGENTS_LIBS_REFERENCE,
        "runtime_import": "deepagents",
        "runtime_available": deepagents_runtime_available(),
        "install_hint": "pip install '.[deepagents]' 또는 폐쇄망 wheelhouse에서 deepagents wheel 설치",
        "libs": [spec.as_dict() for spec in DEEPAGENTS_LIBS],
    }


def format_deepagents_libs() -> str:
    runtime = "available" if deepagents_runtime_available() else "missing"
    rows = [
        "DeepAgents Libs",
        f"- reference: {DEEPAGENTS_LIBS_REFERENCE}",
        f"- runtime_import: deepagents ({runtime})",
        "- install_hint: pip install '.[deepagents]' 또는 폐쇄망 wheelhouse에서 deepagents wheel 설치",
        "",
        "libs:",
    ]
    for spec in DEEPAGENTS_LIBS:
        required = "required" if spec.required_now else "optional"
        rows.extend(
            [
                f"- {spec.name} ({required})",
                f"  path: {spec.path}",
                f"  purpose: {spec.purpose}",
                f"  poc_usage: {spec.poc_usage}",
            ]
        )
    return "\n".join(rows)
