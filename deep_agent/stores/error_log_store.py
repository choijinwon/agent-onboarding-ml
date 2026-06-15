"""Error log storage and lightweight analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from deep_agent.app_config import AppConfig, ensure_read_write_directory


@dataclass(frozen=True)
class ErrorLogEntry:
    id: str
    created_at: str
    source: str
    project_path: str
    message: str
    tags: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorAnalysis:
    error_id: str
    summary: str
    likely_causes: list[str]
    retry_actions: list[str]
    recommended_command: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def save_error_log(
    message: str,
    *,
    source: str = "manual",
    project_path: str = ".",
    config: AppConfig | None = None,
) -> ErrorLogEntry:
    active_config = AppConfig.load() if config is None else config
    directory = error_log_dir(active_config)
    ensure_read_write_directory(directory)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry = ErrorLogEntry(
        id=f"error-{created_at}",
        created_at=created_at,
        source=source,
        project_path=project_path,
        message=_mask_sensitive(message, active_config),
        tags=_detect_tags(message),
    )
    path = directory / f"{entry.id}.json"
    path.write_text(json.dumps(entry.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def list_error_logs(config: AppConfig | None = None) -> list[ErrorLogEntry]:
    active_config = AppConfig.load() if config is None else config
    directory = error_log_dir(active_config)
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.glob("error-*.json")):
        entries.append(_read_error_entry(path))
    return entries


def analyze_error_log(path: str | Path, config: AppConfig | None = None) -> ErrorAnalysis:
    active_config = AppConfig.load() if config is None else config
    entry_path = resolve_error_log_path(path, active_config)
    entry = _read_error_entry(entry_path)
    causes = _likely_causes(entry.message)
    actions = _retry_actions(entry)
    return ErrorAnalysis(
        error_id=entry.id,
        summary=_summarize_error(entry.message),
        likely_causes=causes,
        retry_actions=actions,
        recommended_command=f"ml-agent fix {entry.project_path} --dry-run",
    )


def format_error_log_list(entries: list[ErrorLogEntry]) -> str:
    if not entries:
        return "error logs: none"
    rows = "\n".join(
        f"- {entry.id} source={entry.source} project={entry.project_path} tags={','.join(entry.tags)}"
        for entry in entries
    )
    return f"error logs: {len(entries)}\n{rows}"


def format_error_analysis(analysis: ErrorAnalysis) -> str:
    causes = "\n".join(f"- {cause}" for cause in analysis.likely_causes)
    actions = "\n".join(f"- {action}" for action in analysis.retry_actions)
    return (
        f"error_id: {analysis.error_id}\n"
        f"summary: {analysis.summary}\n"
        "likely_causes:\n"
        f"{causes}\n"
        "retry_actions:\n"
        f"{actions}\n"
        f"recommended_command: {analysis.recommended_command}"
    )


def error_log_dir(config: AppConfig) -> Path:
    return config.root_dir / config.get("CHAT_ERROR_DIR")


def resolve_error_log_path(path: str | Path, config: AppConfig) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    candidate = error_log_dir(config) / str(path)
    if candidate.exists():
        return candidate
    candidate = error_log_dir(config) / f"{path}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"error log not found: {path}")


def _read_error_entry(path: Path) -> ErrorLogEntry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ErrorLogEntry(
        id=payload["id"],
        created_at=payload["created_at"],
        source=payload.get("source", "unknown"),
        project_path=payload.get("project_path", "."),
        message=payload.get("message", ""),
        tags=list(payload.get("tags", [])),
    )


def _mask_sensitive(message: str, config: AppConfig) -> str:
    if not config.get_bool("MASK_SENSITIVE_LOGS"):
        return message
    masked = message
    for marker in ("api_key=", "token=", "password=", "secret="):
        lower = masked.lower()
        index = lower.find(marker)
        while index != -1:
            end = masked.find(" ", index)
            if end == -1:
                end = len(masked)
            masked = f"{masked[:index]}{marker}***{masked[end:]}"
            lower = masked.lower()
            index = lower.find(marker, index + len(marker) + 3)
    return masked


def _detect_tags(message: str) -> list[str]:
    lower = message.lower()
    tags = []
    for tag, keywords in {
        "mlflow": ("mlflow", "tracking_uri", "experiment", "artifact"),
        "requirements": ("modulenotfounderror", "importerror", "requirements", "package"),
        "arguments": ("argument", "argparse", "unrecognized arguments"),
        "job-template": ("queue", "gpu", "cpu", "memory", "job template"),
        "qwen": ("qwen", "base_url", "/v1", "connection"),
    }.items():
        if any(keyword in lower for keyword in keywords):
            tags.append(tag)
    return tags or ["general"]


def _likely_causes(message: str) -> list[str]:
    tags = _detect_tags(message)
    causes = []
    if "mlflow" in tags:
        causes.append("MLflow tracking, experiment, artifact 경로 설정 누락 가능성")
    if "requirements" in tags:
        causes.append("requirements 누락 또는 폐쇄망 패키지 반입 누락 가능성")
    if "arguments" in tags:
        causes.append("학습 entrypoint arguments와 Job Template arguments 불일치 가능성")
    if "job-template" in tags:
        causes.append("queue, gpu, cpu, memory 등 Job Template resource 설정 오류 가능성")
    if "qwen" in tags:
        causes.append("Qwen base URL, model name, 내부망 연결 설정 오류 가능성")
    return causes or ["상세 로그 기반 추가 분석 필요"]


def _retry_actions(entry: ErrorLogEntry) -> list[str]:
    actions = [
        f"관련 프로젝트 재분석: ml-agent analyze {entry.project_path}",
        f"검증 재실행: ml-agent validate {entry.project_path}",
        f"수정안 재생성: ml-agent fix {entry.project_path} --dry-run",
    ]
    if "mlflow" in entry.tags:
        actions.insert(1, "MLflow tracking URI와 artifact logging 코드 확인")
    if "job-template" in entry.tags or "arguments" in entry.tags:
        actions.insert(1, "Job Template arguments와 resource 기본값 확인")
    return actions


def _summarize_error(message: str) -> str:
    clean = " ".join(message.split())
    if len(clean) <= 160:
        return clean
    return f"{clean[:157]}..."
