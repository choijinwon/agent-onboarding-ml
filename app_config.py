"""Environment configuration and runtime directory setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENV = {
    "QWEN_API_KEY": "your-internal-qwen-key",
    "QWEN_BASE_URL": "http://xxx.xxx.xxx.xxx:port/v1",
    "QWEN_MODEL": "qwen3.5",
    "QWEN_MODELS": "qwen3.5,gpt20,gamma",
    "ENABLE_MULTI_AGENT": "true",
    "ENABLE_HARNESS_SKILLS": "true",
    "PROMPT_STORE_PATH": "prompt_templates.json",
    "CHAT_WORKSPACE_DIR": "agent_workspace",
    "REGISTERED_WORKSPACE_DIR": "agent_workspace/registered",
    "CHAT_ERROR_DIR": "chat_errors",
    "MASK_SENSITIVE_LOGS": "true",
    "REGISTRATION_PACKAGE_DIR": "registration_packages",
    "FIX_REPORT_DIR": "fix_reports",
    "SKILL_STORE_DIR": "skills",
}

DIRECTORY_KEYS = (
    "CHAT_WORKSPACE_DIR",
    "REGISTERED_WORKSPACE_DIR",
    "CHAT_ERROR_DIR",
    "REGISTRATION_PACKAGE_DIR",
    "FIX_REPORT_DIR",
    "SKILL_STORE_DIR",
)


@dataclass(frozen=True)
class AppConfig:
    values: dict[str, str]
    root_dir: Path

    @classmethod
    def load(cls, env_file: str = ".env", root_dir: Path | None = None) -> "AppConfig":
        root = Path.cwd() if root_dir is None else root_dir
        values = dict(DEFAULT_ENV)
        values.update(_read_env_file(root / env_file))
        for key in DEFAULT_ENV:
            if key in os.environ:
                values[key] = os.environ[key]
        return cls(values=values, root_dir=root)

    def get(self, key: str) -> str:
        return self.values.get(key, "")

    def get_bool(self, key: str) -> bool:
        return self.get(key).strip().lower() in {"1", "true", "yes", "on"}

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def runtime_directories(self) -> list[Path]:
        dirs = [self.root_dir / self.get(key) for key in DIRECTORY_KEYS if self.get(key)]
        if self.get_bool("ENABLE_HARNESS_SKILLS"):
            dirs.append(self.skill_store_dir())
        return _dedupe_paths(dirs)

    def skill_store_dir(self) -> Path:
        return self.root_dir / self.get("SKILL_STORE_DIR")


def ensure_runtime_layout(config: AppConfig) -> list[Path]:
    created_or_existing = []
    for directory in config.runtime_directories():
        directory.mkdir(parents=True, exist_ok=True)
        created_or_existing.append(directory)
    _ensure_skill_readme(config.skill_store_dir())
    return created_or_existing


def format_config_summary(config: AppConfig) -> str:
    directories = "\n".join(f"- {path}" for path in config.runtime_directories())
    models = ", ".join(model.strip() for model in config.get("QWEN_MODELS").split(","))
    return (
        "Environment Config\n"
        f"- qwen_base_url={config.get('QWEN_BASE_URL')}\n"
        f"- qwen_model={config.get('QWEN_MODEL')}\n"
        f"- qwen_models={models}\n"
        f"- multi_agent={config.get_bool('ENABLE_MULTI_AGENT')}\n"
        f"- harness_skills={config.get_bool('ENABLE_HARNESS_SKILLS')}\n"
        f"- skill_store_dir={config.skill_store_dir()}\n\n"
        "runtime_directories:\n"
        f"{directories}"
    )


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _ensure_skill_readme(skill_dir: Path) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    readme = skill_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "# Agent Skills\n\n"
        "Deep Agent harness skills are stored here.\n\n"
        "Each skill should live in its own directory with a `SKILL.md` file.\n",
        encoding="utf-8",
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    deduped = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
