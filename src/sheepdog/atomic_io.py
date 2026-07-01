"""Atomic file-write helpers shared across the training and persistence layers.

Writing JSON/text with a temp file + rename keeps readers from ever observing a
half-written file, and means an interrupted process (e.g. an overnight reboot)
leaves the previous complete version intact rather than a truncated one.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def atomic_replace(tmp: Path, dest: Path) -> None:
    """Rename *tmp* to *dest*, retrying briefly on Windows file-lock errors."""
    for attempt in range(6):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    atomic_replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    atomic_replace(tmp, path)
