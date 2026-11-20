"""Persistent summary cache for evaluation metadata JSON files.

Promotion gate evaluation in ``server.py`` reads a small set of summary fields and
per-seed record outcomes from every ``eval_*.json`` and ``evaluation-checkpoint-*.json``
file in the evaluations directory. Long runs accumulate tens of thousands of these files,
and re-parsing all of them whenever a new evaluation lands causes ~100-second pauses.

This module keeps a small sidecar cache of just the needed fields, keyed by each file's
size and modification time. Only new or changed files are parsed; everything else comes
from the cache. The cache is strictly best-effort: if it is missing, corrupt, or written
by a different version, it is rebuilt from the evaluation files, which remain the source
of truth.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

CACHE_VERSION = 2

SUMMARY_FIELDS: tuple[str, ...] = (
    "average_completion_steps",
    "average_reward",
    "checkpoint_episode",
    "checkpoint_id",
    "curriculum_stage",
    "environment_config_hash",
    "evaluation_id",
    "evaluation_mode",
    "evaluation_seed_set_id",
    "observation_schema_hash",
    "promotion_eligible",
    "run_id",
    "success_rate",
)

RECORD_FIELDS: tuple[str, ...] = (
    "seed",
    "steps",
    "success",
    "timeout",
)

ProgressCallback = Callable[[int, int], None]


def default_eval_cache_path(output_root: Path, journey: str | None = None) -> Path:
    """Location of the evaluation summary cache for an artifacts output root."""
    root = Path(output_root)
    if journey:
        return root / "archive" / f"journey-{journey}" / "cache" / "evaluation-summaries.json"
    return root / "cache" / "evaluation-summaries.json"


def _is_evaluation_filename(name: str) -> bool:
    if not name.endswith(".json") or name.endswith(".inprogress.json"):
        return False
    return name.startswith("eval_") or name.startswith("evaluation-checkpoint-")


def _summarize_evaluation(path: Path) -> dict[str, Any] | None:
    """Extract summary fields from one evaluation file, or None if it is unreadable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    summary: dict[str, Any] = {field: data[field] for field in SUMMARY_FIELDS if field in data}
    raw_records = data.get("records")
    if isinstance(raw_records, list):
        summary["records"] = [
            {field: rec[field] for field in RECORD_FIELDS if field in rec}
            for rec in raw_records
            if isinstance(rec, dict)
        ]
    return summary


def _read_cache(cache_path: Path) -> dict[str, Any]:
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable evaluation summary cache %s: %s", cache_path, exc)
        return {}
    if (
        not isinstance(cache, dict)
        or cache.get("version") != CACHE_VERSION
        or cache.get("fields") != list(SUMMARY_FIELDS)
        or cache.get("record_fields") != list(RECORD_FIELDS)
        or not isinstance(cache.get("entries"), dict)
    ):
        return {}
    return cache["entries"]


def load_evaluation_summaries(
    eval_dir: Path,
    cache_path: Path,
    progress: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[dict[str, Any]]:
    """Return deduplicated summary dicts for readable evaluation files in *eval_dir*.

    Matches ``_load_all_persisted_evaluations`` filtering and deduplication semantics.
    Unreadable or non-object files are cached as ``None`` and skipped.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        return []

    cached_entries = _read_cache(cache_path)
    entries: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Path, int, int]] = []
    order: list[str] = []

    try:
        with os.scandir(eval_dir) as listing:
            for item in listing:
                name = item.name
                if not _is_evaluation_filename(name):
                    continue
                try:
                    if not item.is_file():
                        continue
                    stat = item.stat()
                except OSError:
                    continue
                order.append(name)
                cached = cached_entries.get(name)
                if (
                    isinstance(cached, dict)
                    and cached.get("size") == stat.st_size
                    and cached.get("mtime_ns") == stat.st_mtime_ns
                ):
                    entries[name] = cached
                else:
                    pending.append((name, Path(item.path), stat.st_size, stat.st_mtime_ns))
    except OSError:
        return []

    for index, (name, path, size, mtime_ns) in enumerate(pending, start=1):
        entries[name] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "summary": _summarize_evaluation(path),
        }
        if progress is not None and (index % progress_every == 0 or index == len(pending)):
            progress(index, len(pending))

    removed = set(cached_entries) - set(entries)
    if pending or removed:
        try:
            atomic_write_json(
                cache_path,
                {
                    "version": CACHE_VERSION,
                    "fields": list(SUMMARY_FIELDS),
                    "record_fields": list(RECORD_FIELDS),
                    "entries": entries,
                },
            )
        except OSError as exc:
            logger.warning("Could not write evaluation summary cache %s: %s", cache_path, exc)

    evaluations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for name in order:
        summary = entries[name].get("summary")
        if isinstance(summary, dict):
            eval_id = summary.get("evaluation_id") or summary.get("checkpoint_id") or name
            if eval_id not in seen_ids:
                seen_ids.add(eval_id)
                evaluations.append(dict(summary))
    return evaluations
