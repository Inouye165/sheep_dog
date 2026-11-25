"""Atomic file-write helpers shared across the training and persistence layers.

Writing JSON/text with a temp file + rename keeps readers from ever observing a
half-written file, and means an interrupted process (e.g. an overnight reboot)
leaves the previous complete version intact rather than a truncated one.
"""

from __future__ import annotations

import contextlib
import errno
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_MIN_FREE_DISK_BYTES = 512 * 1024 * 1024
DEFAULT_WARN_FREE_DISK_BYTES = 2 * 1024 * 1024 * 1024


class DiskSpaceError(OSError):
    """Raised when free disk space drops below the configured safety threshold."""


def get_min_free_disk_bytes(default_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES) -> int:
    """Return the minimum required free disk bytes, honoring ``SHEEPDOG_MIN_FREE_DISK_MB``."""
    raw = os.environ.get("SHEEPDOG_MIN_FREE_DISK_MB")
    if raw is None or not raw.strip():
        return default_bytes
    try:
        mb = float(raw.strip())
    except (ValueError, OverflowError):
        return default_bytes
    if not math.isfinite(mb) or mb < 0:
        return default_bytes
    if mb == 0.0:
        return 0
    return int(mb * 1024 * 1024)


def get_free_disk_bytes(path: str | Path) -> int | None:
    """Return free disk bytes on the filesystem containing *path*, or None on OSError."""
    try:
        resolved = Path(path).resolve()
        for candidate in (resolved, *resolved.parents):
            if candidate.exists():
                return int(shutil.disk_usage(candidate).free)
    except OSError:
        return None
    return None


def ensure_free_disk_space(
    path: str | Path,
    min_free_bytes: int | None = None,
    *,
    context: str = "",
) -> int | None:
    """Verify free disk space meets the safety threshold, raising DiskSpaceError if not."""
    threshold = (
        get_min_free_disk_bytes()
        if min_free_bytes is None
        else max(0, int(min_free_bytes))
    )
    free = get_free_disk_bytes(path)
    if free is not None and threshold > 0 and free < threshold:
        ctx_suffix = f" ({context})" if context else ""
        raise DiskSpaceError(
            errno.ENOSPC,
            (
                f"Insufficient free disk space{ctx_suffix}: "
                f"{free / (1024 * 1024):.1f} MB free, at least "
                f"{threshold / (1024 * 1024):.1f} MB required"
            ),
            str(path),
        )
    return free


def cleanup_orphaned_tmp_files(
    roots: Iterable[str | Path],
    *,
    min_age_seconds: float = 300.0,
) -> int:
    """Delete orphaned ``*.tmp`` files older than *min_age_seconds* under *roots*."""
    now = time.time()
    removed = 0
    for root in roots:
        root_path = Path(root)
        try:
            if not root_path.is_dir():
                continue
            candidates = list(root_path.rglob("*.tmp"))
        except OSError:
            continue
        for candidate in candidates:
            try:
                if not candidate.is_file():
                    continue
                if now - candidate.stat().st_mtime >= min_age_seconds:
                    candidate.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


def atomic_replace(tmp: Path, dest: Path) -> None:
    """Rename *tmp* to *dest*, retrying briefly on Windows file-lock errors."""
    try:
        for attempt in range(30):
            try:
                os.replace(tmp, dest)
                return
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(min(0.2, 0.03 * (attempt + 1)))
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _atomic_tmp_path(path: Path) -> Path:
    """Build a unique sibling ``.tmp`` path for atomic replacement of *path*."""
    return path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")


def atomic_write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write *data* as JSON to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            if indent is None:
                json.dump(data, fh, separators=(",", ":"))
            else:
                json.dump(data, fh, indent=indent)
        atomic_replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        atomic_replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write *payload* bytes to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        tmp.write_bytes(payload)
        atomic_replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
