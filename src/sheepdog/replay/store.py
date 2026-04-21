"""Replay persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReplayStore:
    """Persist replay frames to a JSON file."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, payload: Any) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path
