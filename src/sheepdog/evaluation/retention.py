"""Evaluation replay retention management.

Implements retention policy for evaluation episode replays:
- Always keeps Evaluation #1 (baseline)
- Retains milestone evaluations every 25 evaluations (#25, #50, #75, ...)
- Keeps the rolling latest evaluation (#N) for instant viewing
- Retains any evaluation explicitly pinned by the user
- Safely prunes all other intermediate replays to prevent disk bloat
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationRetentionPolicy:
    """Retention rules for evaluation replay bundles."""

    save_first: bool = True
    milestone_interval: int = 25
    keep_latest: bool = True


class EvaluationReplayRetentionManager:
    """Manages disk lifecycle and pruning of evaluation replays."""

    def __init__(
        self,
        output_dir: str | Path,
        policy: EvaluationRetentionPolicy | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.policy = policy or EvaluationRetentionPolicy()
        self.pinned_file = self.output_dir / "pinned_evaluations.json"
        self.index_file = self.output_dir / "eval_retention_index.json"

    def get_pinned_evaluation_ids(self) -> set[str]:
        """Return the set of explicitly pinned evaluation IDs."""
        if not self.pinned_file.exists():
            return set()
        try:
            with self.pinned_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    return set(str(item) for item in data)
                if isinstance(data, dict):
                    return set(str(k) for k, v in data.items() if v)
        except Exception as exc:
            logger.warning("Failed to read pinned evaluations: %s", exc)
        return set()

    def pin_evaluation(self, evaluation_id: str, pinned: bool = True) -> bool:
        """Pin or unpin an evaluation to prevent its replays from being pruned."""
        pinned_ids = self.get_pinned_evaluation_ids()
        clean_id = str(evaluation_id).strip()
        if not clean_id:
            return False

        if pinned:
            pinned_ids.add(clean_id)
        else:
            pinned_ids.discard(clean_id)

        try:
            atomic_write_json(self.pinned_file, sorted(list(pinned_ids)))
            return True
        except Exception as exc:
            logger.warning("Failed to update pinned evaluations: %s", exc)
            return False

    def is_pinned(self, evaluation_id: str) -> bool:
        """Check if an evaluation ID is pinned."""
        return str(evaluation_id).strip() in self.get_pinned_evaluation_ids()

    def get_retention_status(
        self,
        evaluation_id: str,
        evaluation_index: int,
        is_latest: bool = False,
    ) -> tuple[bool, str]:
        """Determine whether an evaluation replay set should be retained.

        Returns (should_retain, reason).
        """
        if self.is_pinned(evaluation_id):
            return True, "pinned"
        if self.policy.save_first and evaluation_index == 1:
            return True, "first"
        if (
            self.policy.milestone_interval > 0
            and evaluation_index > 0
            and evaluation_index % self.policy.milestone_interval == 0
        ):
            return True, "milestone"
        if self.policy.keep_latest and is_latest:
            return True, "latest"
        return False, "unretained"

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the retention index ledger."""
        if not self.index_file.exists():
            return {}
        try:
            with self.index_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to load retention index: %s", exc)
        return {}

    def _save_index(self, index: dict[str, dict[str, Any]]) -> None:
        """Atomically persist the retention index ledger."""
        try:
            atomic_write_json(self.index_file, index)
        except Exception as exc:
            logger.warning("Failed to save retention index: %s", exc)

    def register_and_prune(
        self,
        evaluation_id: str,
        evaluation_index: int,
        checkpoint_episode: int,
        replay_paths: list[str | Path],
    ) -> dict[str, Any]:
        """Register newly saved evaluation replays and prune older unretained replays."""
        index = self._load_index()
        str_paths = [str(p) for p in replay_paths if p]

        index[evaluation_id] = {
            "evaluation_id": evaluation_id,
            "evaluation_index": evaluation_index,
            "checkpoint_episode": checkpoint_episode,
            "replay_paths": str_paths,
            "pruned": False,
        }

        # Determine latest evaluation across entries with replays
        latest_eval_id: str | None = None
        max_index = -1
        for eid, entry in index.items():
            e_idx = int(entry.get("evaluation_index", 0))
            if e_idx >= max_index:
                max_index = e_idx
                latest_eval_id = eid

        pruned_count = 0
        retained_count = 0

        for eid, entry in list(index.items()):
            e_idx = int(entry.get("evaluation_index", 0))
            is_latest = (eid == latest_eval_id)
            should_retain, reason = self.get_retention_status(eid, e_idx, is_latest=is_latest)

            entry["retention_status"] = reason
            if should_retain:
                retained_count += 1
                entry["pruned"] = False
            else:
                # Prune replay files
                entry_paths = entry.get("replay_paths", [])
                if entry_paths and not entry.get("pruned", False):
                    for path_str in entry_paths:
                        file_path = Path(path_str)
                        if not file_path.is_absolute():
                            file_path = self.output_dir / file_path
                        if file_path.exists():
                            try:
                                file_path.unlink()
                                pruned_count += 1
                            except Exception as exc:
                                logger.warning("Failed to delete pruned replay file %s: %s", file_path, exc)
                    entry["pruned"] = True

        self._save_index(index)
        return {
            "evaluation_id": evaluation_id,
            "evaluation_index": evaluation_index,
            "retained_count": retained_count,
            "pruned_count": pruned_count,
            "status": index[evaluation_id].get("retention_status", "unknown"),
        }
