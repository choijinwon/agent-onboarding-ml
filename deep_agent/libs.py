"""DeepAgents libs manifest and runtime availability helpers."""

from __future__ import annotations

import importlib.util
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


DEEPAGENTS_LIBS_REFERENCE = "https://github.com/langchain-ai/deepagents/tree/main/libs"
DEEPAGENTS_SOURCE_ENV = "DEEPAGENTS_SOURCE_ZIP"
DEEPAGENTS_SOURCE_DIR_ENV = "DEEPAGENTS_SOURCE_DIR"


POC_USAGE_BY_PATH = {
    "libs/deepagents": "실제 LLM runtime 연결 시 ai-ml-onboarding profile을 create_deep_agent 설정으로 옮기는 대상.",
    "libs/code": "초급자 Wizard TUI 화면의 참고 구현 축.",
    "libs/cli": "고급자 모드 CLI 명령 체계와 자동화 파이프라인 참고.",
    "libs/evals": "Wizard 수정 전후 회귀 테스트와 agent 품질 평가에 연결.",
    "libs/acp": "향후 외부 agent/client 연계가 필요할 때 선택 적용.",
    "libs/talon": "채널/런타임 확장 방식 참고.",
    "libs/partners/daytona": "Daytona sandbox provider 연계가 필요할 때 참고.",
    "libs/partners/modal": "Modal sandbox provider 연계가 필요할 때 참고.",
    "libs/partners/quickjs": "QuickJS 기반 sandbox/subagent 실행 모델 참고.",
    "libs/partners/runloop": "Runloop sandbox provider 연계가 필요할 때 참고.",
    "libs/partners/vercel": "Vercel sandbox provider 연계가 필요할 때 참고.",
}


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


def _extract_project_value(pyproject_text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*['\"]([^'\"]+)['\"]", pyproject_text)
    if not match:
        return None
    return match.group(1).strip()


def _default_source_candidates() -> list[Path]:
    package_root = Path(__file__).resolve().parent
    return [
        package_root / "vendor" / "deepagents" / "deepagents-main",
        package_root / "vendor" / "deepagents",
        Path.cwd() / "deepagents_source" / "deepagents-main",
        Path.cwd() / "deepagents_source",
        Path.cwd() / "deepagents-main.zip",
        Path.home() / "Downloads" / "deepagents-main.zip",
    ]


def _detect_source(candidate: Path) -> tuple[Path, str] | None:
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        return candidate, "zip"
    if candidate.is_dir():
        if any(candidate.glob("libs/**/pyproject.toml")):
            return candidate, "directory"
        if any(candidate.glob("deepagents-main/libs/**/pyproject.toml")):
            return candidate, "directory"
    return None


def resolve_deepagents_source(source: str | None = None) -> tuple[Path | None, str | None]:
    candidates: list[Path] = []
    if source:
        candidates.append(Path(source).expanduser())
    env_dir = os.environ.get(DEEPAGENTS_SOURCE_DIR_ENV, "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    env_zip = os.environ.get(DEEPAGENTS_SOURCE_ENV, "").strip()
    if env_zip:
        candidates.append(Path(env_zip).expanduser())
    candidates.extend(_default_source_candidates())
    for candidate in candidates:
        detected = _detect_source(candidate)
        if detected:
            return detected
    return None, None


def read_deepagents_libs_from_zip(source_zip: str | Path) -> list[DeepAgentsLibSpec]:
    source_path = Path(source_zip).expanduser()
    specs: list[DeepAgentsLibSpec] = []
    with zipfile.ZipFile(source_path) as archive:
        pyprojects = sorted(
            name
            for name in archive.namelist()
            if re.match(r"^[^/]+/libs/.+/pyproject\.toml$", name)
        )
        for pyproject in pyprojects:
            parts = pyproject.split("/")
            lib_path = "/".join(parts[1:-1])
            raw_text = archive.read(pyproject).decode("utf-8", errors="replace")
            package_name = _extract_project_value(raw_text, "name") or lib_path.rsplit("/", 1)[-1]
            description = _extract_project_value(raw_text, "description")
            purpose = description or f"DeepAgents package from {lib_path}."
            specs.append(
                DeepAgentsLibSpec(
                    name=package_name,
                    path=lib_path,
                    purpose=purpose,
                    poc_usage=POC_USAGE_BY_PATH.get(lib_path, "DeepAgents 확장 패키지로 필요 시 선택 적용."),
                    required_now=lib_path == "libs/deepagents" or package_name == "deepagents",
                )
            )
    return specs


def read_deepagents_libs_from_directory(source_dir: str | Path) -> list[DeepAgentsLibSpec]:
    source_path = Path(source_dir).expanduser()
    pyprojects = sorted(source_path.glob("libs/**/pyproject.toml"))
    if not pyprojects:
        pyprojects = sorted(source_path.glob("deepagents-main/libs/**/pyproject.toml"))
    specs: list[DeepAgentsLibSpec] = []
    for pyproject in pyprojects:
        relative_parts = pyproject.relative_to(source_path).parts
        if "libs" not in relative_parts:
            continue
        libs_index = relative_parts.index("libs")
        lib_path = "/".join(relative_parts[libs_index:-1])
        raw_text = pyproject.read_text(encoding="utf-8", errors="replace")
        package_name = _extract_project_value(raw_text, "name") or lib_path.rsplit("/", 1)[-1]
        description = _extract_project_value(raw_text, "description")
        purpose = description or f"DeepAgents package from {lib_path}."
        specs.append(
            DeepAgentsLibSpec(
                name=package_name,
                path=lib_path,
                purpose=purpose,
                poc_usage=POC_USAGE_BY_PATH.get(lib_path, "DeepAgents 확장 패키지로 필요 시 선택 적용."),
                required_now=lib_path == "libs/deepagents" or package_name == "deepagents",
            )
        )
    return specs


def deepagents_runtime_available() -> bool:
    return importlib.util.find_spec("deepagents") is not None


def _libs_for_source(source: str | None = None) -> tuple[list[DeepAgentsLibSpec], Path | None, str | None, str | None]:
    source_path, source_type = resolve_deepagents_source(source)
    if not source_path:
        return DEEPAGENTS_LIBS, None, None, None
    try:
        if source_type == "directory":
            return read_deepagents_libs_from_directory(source_path), source_path, source_type, None
        return read_deepagents_libs_from_zip(source_path), source_path, source_type, None
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        return DEEPAGENTS_LIBS, source_path, source_type, f"{type(exc).__name__}: {exc}"


def deepagents_libs_as_dict(source_zip: str | None = None) -> dict[str, object]:
    libs, resolved_source, source_type, source_error = _libs_for_source(source_zip)
    return {
        "reference": DEEPAGENTS_LIBS_REFERENCE,
        "source_path": str(resolved_source) if resolved_source else None,
        "source_type": source_type,
        "source_zip": str(resolved_source) if resolved_source and source_type == "zip" else None,
        "source_zip_found": resolved_source is not None and source_type == "zip",
        "source_error": source_error,
        "libs_source": source_type if resolved_source and not source_error else "fallback_manifest",
        "runtime_import": "deepagents",
        "runtime_available": deepagents_runtime_available(),
        "install_hint": "pip install '.[deepagents]' 또는 폐쇄망 wheelhouse에서 deepagents wheel 설치",
        "libs": [spec.as_dict() for spec in libs],
    }


def format_deepagents_libs(source_zip: str | None = None) -> str:
    libs, resolved_source, source_type, source_error = _libs_for_source(source_zip)
    runtime = "available" if deepagents_runtime_available() else "missing"
    rows = [
        "DeepAgents Libs",
        f"- reference: {DEEPAGENTS_LIBS_REFERENCE}",
        f"- source_path: {resolved_source if resolved_source else 'not found'}",
        f"- source_type: {source_type if source_type else 'none'}",
        f"- libs_source: {source_type if resolved_source and not source_error else 'fallback_manifest'}",
        f"- runtime_import: deepagents ({runtime})",
        "- install_hint: pip install '.[deepagents]' 또는 폐쇄망 wheelhouse에서 deepagents wheel 설치",
        "",
        "libs:",
    ]
    if source_error:
        rows.extend(["", f"source_error: {source_error}", ""])
    for spec in libs:
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
