from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse


WINDOWS_DRIVE_PATH_RE = re.compile(r"^/?([A-Za-z]):[\\/](.*)")
WINDOWS_ENV_RE = re.compile(r"%([^%]+)%")
POWERSHELL_ENV_RE = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def is_windows_style_path(value: str) -> bool:
    return bool(WINDOWS_DRIVE_PATH_RE.match(value)) or value.startswith(("\\\\", "//"))


def expand_cross_platform_vars(value: str) -> str:
    def replace_percent(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    def replace_powershell(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    expanded = WINDOWS_ENV_RE.sub(replace_percent, value)
    expanded = POWERSHELL_ENV_RE.sub(replace_powershell, expanded)
    return os.path.expandvars(os.path.expanduser(expanded))


def normalize_path_text(value: str) -> str:
    normalized = value.strip().strip('"').strip("'")
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        normalized = unquote(parsed.path or normalized.removeprefix("file://"))
    else:
        normalized = unquote(normalized)
    if WINDOWS_DRIVE_PATH_RE.match(normalized):
        normalized = normalized.lstrip("/")
    normalized = expand_cross_platform_vars(normalized)
    if not is_windows_style_path(normalized):
        try:
            parts = shlex.split(normalized)
        except ValueError:
            parts = []
        if len(parts) == 1:
            normalized = parts[0]
    return normalized


def windows_drive_candidates(value: str) -> list[Path]:
    match = WINDOWS_DRIVE_PATH_RE.match(value)
    if not match:
        return []
    drive = match.group(1).upper()
    rest = match.group(2).replace("\\", "/")
    env_root = os.environ.get(f"AIU_WINDOWS_DRIVE_{drive}") or os.environ.get("AIU_WINDOWS_DRIVE_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser() / rest)
    candidates.extend([Path(f"/mnt/{drive.lower()}") / rest, Path(f"/{drive.lower()}") / rest])
    return candidates


def windows_unc_candidates(value: str) -> list[Path]:
    if not value.startswith(("\\\\", "//")):
        return []
    cleaned = value.replace("\\", "/").lstrip("/")
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) < 2:
        return []
    server, share, *rest = parts
    tail = Path(*rest) if rest else Path()
    env_specific = os.environ.get(f"AIU_UNC_{server.upper()}_{share.upper()}".replace("-", "_"))
    env_root = os.environ.get("AIU_UNC_ROOT")
    candidates: list[Path] = []
    if env_specific:
        candidates.append(Path(env_specific).expanduser() / tail)
    if env_root:
        candidates.append(Path(env_root).expanduser() / server / share / tail)
    candidates.extend([Path("/mnt") / server / share / tail, Path("/Volumes") / share / tail])
    return candidates


def filesystem_path_candidates(value: str) -> list[Path]:
    normalized = normalize_path_text(value)
    if not normalized:
        return []
    candidates = [Path(normalized).expanduser()]
    if os.name != "nt":
        candidates.extend(windows_drive_candidates(normalized))
        candidates.extend(windows_unc_candidates(normalized))
    return candidates


def resolve_filesystem_path(value: str) -> Path:
    candidates = filesystem_path_candidates(value)
    if not candidates:
        return Path(value or ".")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def existing_filesystem_path(value: str) -> Path | None:
    for candidate in filesystem_path_candidates(value):
        if candidate.exists():
            return candidate.resolve()
    return None


__all__ = [
    "existing_filesystem_path",
    "expand_cross_platform_vars",
    "filesystem_path_candidates",
    "is_windows_style_path",
    "normalize_path_text",
    "resolve_filesystem_path",
    "windows_unc_candidates",
    "windows_drive_candidates",
]
