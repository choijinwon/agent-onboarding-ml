"""Deep Agent profile for the AI ML onboarding POC.

This module is inspired by langchain-ai/deepagents. It keeps the POC
dependency-free while preserving the important harness concepts:
sub-agents, scoped filesystem permissions, skills, memory, summarization,
and human approval before writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


AgentMode = Literal["beginner", "intermediate", "advanced"]
PermissionMode = Literal["allow", "deny", "interrupt"]
FilesystemOperation = Literal["read", "write"]


DEEPAGENTS_REFERENCE = "https://github.com/langchain-ai/deepagents"


@dataclass(frozen=True)
class PermissionRule:
    operations: list[FilesystemOperation]
    paths: list[str]
    mode: PermissionMode = "allow"


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[str]


@dataclass(frozen=True)
class DeepAgentProfile:
    name: str
    mode: AgentMode
    model_policy: str
    system_prompt: str
    subagents: list[SubAgentSpec]
    permissions: list[PermissionRule]
    skills: list[str]
    memory: list[str]
    tools: list[str]
    approval_policy: str
    context_policy: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


BASE_SYSTEM_PROMPT = """You are an AI ML Onboarding Assistant.
Help users register ML projects safely in a closed-network POC.
Prefer read-only analysis first, explain risk before writes, and verify after apply."""


MODE_SYSTEM_PROMPTS = {
    "beginner": (
        "Use short wizard steps. Explain difficult AI/ML onboarding terms in plain Korean. "
        "Show previews before any file creation or edit."
    ),
    "intermediate": (
        "Use summary-first answers. Focus on MLflow, requirements, arguments, "
        "and Job Template validation. Offer itemized dry-run fixes."
    ),
    "advanced": (
        "Use concise command-oriented output. Prefer machine-readable results, "
        "exit codes, result files, dry-run, and explicit apply."
    ),
}


def build_ml_platform_profile(mode: AgentMode = "beginner") -> DeepAgentProfile:
    """Build the closed-network Deep Agent profile for a launch mode."""

    return DeepAgentProfile(
        name="ai-ml-onboarding-assistant",
        mode=mode,
        model_policy="model-agnostic; closed-network compatible tool-calling model",
        system_prompt=f"{BASE_SYSTEM_PROMPT}\n{MODE_SYSTEM_PROMPTS[mode]}",
        subagents=[
            SubAgentSpec(
                name="project-scanner",
                description="Scans project structure, entrypoints, requirements, and config files.",
                system_prompt="Return a concise project inventory without modifying files.",
                tools=["ls", "read_file", "glob", "grep"],
            ),
            SubAgentSpec(
                name="mlflow-validator",
                description="Checks MLflow tracking, experiment/run logging, and artifact paths.",
                system_prompt="Validate MLflow readiness and explain missing registration requirements.",
                tools=["read_file", "grep"],
            ),
            SubAgentSpec(
                name="job-template-planner",
                description="Drafts Job Template candidates from entrypoint, arguments, and dependencies.",
                system_prompt="Produce dry-run template changes and require approval before writes.",
                tools=["read_file", "grep", "write_file"],
            ),
            SubAgentSpec(
                name="log-analyzer",
                description="Explains error logs and maps failures to requirements, arguments, or MLflow.",
                system_prompt="Summarize root cause, evidence, and next validation step.",
                tools=["read_file", "grep"],
            ),
        ],
        permissions=[
            PermissionRule(["read"], ["/project/**"], "allow"),
            PermissionRule(["write"], ["/project/**"], "interrupt"),
            PermissionRule(["write"], ["/project/**/.git/**"], "deny"),
            PermissionRule(["write"], ["/project/**/secrets/**", "/project/**/.env"], "deny"),
        ],
        skills=[
            "mlflow-registration-check",
            "job-template-draft",
            "closed-network-validation",
        ],
        memory=[
            "/memory/ai-ml-onboarding-registration-rules.md",
            "/memory/team-job-template-conventions.md",
        ],
        tools=[
            "write_todos",
            "ls",
            "read_file",
            "glob",
            "grep",
            "write_file",
            "edit_file",
            "execute",
            "task",
        ],
        approval_policy="Human-in-the-loop required for write_file/edit_file/apply.",
        context_policy="Summarize long scans and offload detailed evidence into the report artifact.",
    )


def format_profile(profile: DeepAgentProfile) -> str:
    """Render a compact console summary of the Deep Agent profile."""

    subagent_rows = "\n".join(
        f"- {agent.name}: {agent.description}" for agent in profile.subagents
    )
    permission_rows = "\n".join(
        f"- {','.join(rule.operations)} {','.join(rule.paths)} -> {rule.mode}"
        for rule in profile.permissions
    )
    return (
        f"Deep Agent Profile: {profile.name}\n"
        f"mode: {profile.mode}\n"
        f"reference: {DEEPAGENTS_REFERENCE}\n"
        f"model_policy: {profile.model_policy}\n"
        f"approval_policy: {profile.approval_policy}\n"
        f"context_policy: {profile.context_policy}\n\n"
        "subagents:\n"
        f"{subagent_rows}\n\n"
        "permissions:\n"
        f"{permission_rows}\n\n"
        f"skills: {', '.join(profile.skills)}\n"
        f"memory: {', '.join(profile.memory)}"
    )
