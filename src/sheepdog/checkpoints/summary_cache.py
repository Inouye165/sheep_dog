"""Persistent summary cache for checkpoint metadata JSON files.

Startup recovery needs a handful of fields from every ``checkpoint-*.json`` file (schema
hashes, stage, run id, episode, ...). Long runs accumulate tens of thousands of these files
(~2 GB), and opening + parsing all of them on every server start took minutes.

This module keeps a small sidecar cache of just the needed fields, keyed by each file's
size and modification time. Only new or changed files are parsed; everything else comes
from the cache. The cache is strictly best-effort: if it is missing, corrupt, or written
by a different version, it is rebuilt from the checkpoint files, which remain the source
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

CACHE_VERSION = 1

# Every field that startup recovery reads from checkpoint metadata. If recovery starts
# reading a new field, add it here; the cache is invalidated automatically because the
# field list is stored alongside the entries.
SUMMARY_FIELDS: tuple[str, ...] = (
    "action_space_hash",
    "checkpoint_episode",
    "checkpoint_id",
    "curriculum_stage",
    "environment_config_hash",
    "last_evaluation_time",
    "last_policy_update_time",
    "observation_schema_hash",
    "policy_mode",
    "policy_state_path",
    "policy_type",
    "policy_version",
    "ppo_update_count",
    "promotion_seed_set_id",
    "reward_schema_version",
    "run_id",
    "success_rate",
    "trainer_type",
)

ProgressCallback = Callable[[int, int], None]


def default_cache_path(output_root: Path) -> Path:
    """Location of the summary cache for an artifacts output root."""
    return Path(output_root) / "cache" / "checkpoint-summaries.json"


def _summarize(path: Path) -> dict[str, Any] | None:
    """Extract summary fields from one checkpoint file, or None if it is unreadable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {field: data[field] for field in SUMMARY_FIELDS if field in data}


def _read_cache(cache_path: Path) -> dict[str, Any]:
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable checkpoint summary cache %s: %s", cache_path, exc)
        return {}
    if (
        not isinstance(cache, dict)
        or cache.get("version") != CACHE_VERSION
        or cache.get("fields") != list(SUMMARY_FIELDS)
        or not isinstance(cache.get("entries"), dict)
    ):
        return {}
    return cache["entries"]


def load_checkpoint_summaries(
    checkpoints_dir: Path,
    cache_path: Path,
    progress: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[dict[str, Any]]:
    """Return a summary dict for every readable ``checkpoint-*.json`` in *checkpoints_dir*.

    Summaries are returned in directory-listing order (the same order ``Path.glob`` yields).
    Unreadable or non-object files are skipped, matching the previous inline behavior.
    *progress* is called as ``progress(parsed, to_parse)`` while uncached files are parsed.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return []

    cached_entries = _read_cache(cache_path)
    entries: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Path, int, int]] = []
    order: list[str] = []

    with os.scandir(checkpoints_dir) as listing:
        for item in listing:
            name = item.name
            if not (name.startswith("checkpoint-") and name.endswith(".json")):
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

    for index, (name, path, size, mtime_ns) in enumerate(pending, start=1):
        # Unreadable files are cached as None so they are not re-parsed every start;
        # a rewrite changes size/mtime and triggers a fresh parse.
        entries[name] = {"size": size, "mtime_ns": mtime_ns, "summary": _summarize(path)}
        if progress is not None and (index % progress_every == 0 or index == len(pending)):
            progress(index, len(pending))

    removed = set(cached_entries) - set(entries)
    if pending or removed:
        try:
            atomic_write_json(
                cache_path,
                {"version": CACHE_VERSION, "fields": list(SUMMARY_FIELDS), "entries": entries},
            )
        except OSError as exc:
            logger.warning("Could not write checkpoint summary cache %s: %s", cache_path, exc)

    summaries: list[dict[str, Any]] = []
    for name in order:
        summary = entries[name].get("summary")
        if isinstance(summary, dict):
            summaries.append(dict(summary))
    return summaries
