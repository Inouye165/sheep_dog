"""Replay persistence utilities."""

from __future__ import annotations

import gzip
import json
import uuid
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_replace


class ReplayStore:
    """Persist and load replay frames to a JSON or compressed JSON.gz file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, payload: Any, compress: bool = True) -> Path:
        """Write *payload* atomically to *name* (optionally gzipped)."""
        self.root.mkdir(parents=True, exist_ok=True)
        filename = name
        if compress and not filename.endswith(".gz"):
            filename = f"{filename}.gz"
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp")
        if filename.endswith(".gz"):
            with gzip.open(tmp, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
        else:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        atomic_replace(tmp, path)
        return path

    def read(self, name_or_path: str | Path) -> dict[str, Any]:
        """Read a replay payload from a JSON or JSON.gz file."""
        target = Path(name_or_path)
        if not target.is_absolute():
            target = self.root / name_or_path
        if not target.exists() and not str(target).endswith(".gz"):
            gz_target = target.with_suffix(target.suffix + ".gz")
            if gz_target.exists():
                target = gz_target

        if str(target).endswith(".gz"):
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
