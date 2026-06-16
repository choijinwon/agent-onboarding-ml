from __future__ import annotations

import os
import re
import shlex
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlparse


WINDOWS_DRIVE_PATH_RE = re.compile(r"^/?([A-Za-z]):(?:[\\/](.*))?$")
WINDOWS_SLOPPY_DRIVE_PATH_RE = re.compile(r"^[\\/]+([A-Za-z]):(?:[\\/](.*))?$")
WINDOWS_EXTENDED_DRIVE_RE = re.compile(r"^\\\\\?\\([A-Za-z]):\\?(.*)$")
WINDOWS_EXTENDED_UNC_RE = re.compile(r"^\\\\\?\\UNC\\(.+)$", re.IGNORECASE)
WINDOWS_ENV_RE = re.compile(r"%([^%]+)%")
POWERSHELL_ENV_RE = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def is_windows_style_path(value: str) -> bool:
    return (
        bool(WINDOWS_DRIVE_PATH_RE.match(value))
        or bool(WINDOWS_EXTENDED_DRIVE_RE.match(value))
        or bool(WINDOWS_EXTENDED_UNC_RE.match(value))
        or value.startswith(("\\\\", "//"))
    )


def is_windows_absolute_path(value: str) -> bool:
    normalized = normalize_windows_namespace(value.strip().strip('"').strip("'"))
    return bool(WINDOWS_DRIVE_PATH_RE.match(normalized)) or normalized.startswith(("\\\\", "//"))


def normalize_windows_namespace(value: str) -> str:
    sloppy_drive_match = WINDOWS_SLOPPY_DRIVE_PATH_RE.match(value)
    if sloppy_drive_match:
        drive, rest = sloppy_drive_match.groups()
        if rest is None:
            return f"{drive}:"
        separator = "\\" if "\\" in value else "/"
        return f"{drive}:{separator}{rest}".rstrip("\\/")
    drive_match = WINDOWS_EXTENDED_DRIVE_RE.match(value)
    if drive_match:
        drive, rest = drive_match.groups()
        return f"{drive}:\\{rest}".rstrip("\\")
    unc_match = WINDOWS_EXTENDED_UNC_RE.match(value)
    if unc_match:
        return "\\\\" + unc_match.group(1)
    return value


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
        if parsed.netloc and parsed.netloc.lower() not in ("localhost", ""):
            normalized = f"//{parsed.netloc}{unquote(parsed.path)}"
        else:
            normalized = unquote(parsed.path or normalized.removeprefix("file://"))
    else:
        normalized = unquote(normalized)
    normalized = normalize_windows_namespace(normalized)
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
    rest = (match.group(2) or "").replace("\\", "/")
    rest_parts = [part for part in rest.split("/") if part]
    env_root = os.environ.get(f"AIU_WINDOWS_DRIVE_{drive}") or os.environ.get("AIU_WINDOWS_DRIVE_ROOT")
    candidates: list[Path] = []
    if env_root:
        root = Path(env_root).expanduser()
        candidates.append(root / Path(*rest_parts))
        candidates.append(root / drive / Path(*rest_parts))
    candidates.extend(
        [
            Path(f"/mnt/{drive.lower()}") / Path(*rest_parts),
            Path(f"/{drive.lower()}") / Path(*rest_parts),
            Path(f"/{drive}") / Path(*rest_parts),
        ]
    )
    return _dedupe_paths(candidates)


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
    candidates.extend(
        [
            Path("/mnt") / server / share / tail,
            Path("/Volumes") / share / tail,
            Path("//") / server / share / tail,
        ]
    )
    return _dedupe_paths(candidates)


def native_windows_path(value: str) -> Path:
    pure = PureWindowsPath(value)
    return Path(str(pure))


def filesystem_path_candidates(value: str) -> list[Path]:
    normalized = normalize_path_text(value)
    if not normalized:
        return []
    if os.name == "nt" and is_windows_style_path(normalized):
        return [native_windows_path(normalized).expanduser()]
    candidates = [Path(normalized).expanduser()]
    if os.name != "nt":
        candidates.extend(windows_drive_candidates(normalized))
        candidates.extend(windows_unc_candidates(normalized))
    return _dedupe_paths(candidates)


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
    "is_windows_absolute_path",
    "is_windows_style_path",
    "normalize_path_text",
    "resolve_filesystem_path",
    "windows_unc_candidates",
    "windows_drive_candidates",
]
