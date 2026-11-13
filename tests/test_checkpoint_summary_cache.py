"""Tests for the persistent checkpoint metadata summary cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from sheepdog.checkpoints import summary_cache
from sheepdog.checkpoints.summary_cache import (
    SUMMARY_FIELDS,
    default_cache_path,
    load_checkpoint_summaries,
)


def _write_checkpoint(directory: Path, episode: int, **fields: object) -> Path:
    payload = {
        "checkpoint_episode": episode,
        "checkpoint_id": f"chk-{episode}",
        "curriculum_stage": 2,
        "observation_schema_hash": "obs",
        "action_space_hash": "act",
        # Large field that recovery does not need; must not be copied into the cache.
        "training_scenario_coverage": ["x" * 50] * 20,
        **fields,
    }
    path = directory / f"checkpoint-{episode:06d}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    return checkpoints, default_cache_path(tmp_path)


def test_summaries_contain_only_recovery_fields(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1, run_id="run-a", success_rate=0.5)

    summaries = load_checkpoint_summaries(checkpoints, cache_path)

    assert summaries == [
        {
            "action_space_hash": "act",
            "checkpoint_episode": 1,
            "checkpoint_id": "chk-1",
            "curriculum_stage": 2,
            "observation_schema_hash": "obs",
            "run_id": "run-a",
            "success_rate": 0.5,
        }
    ]
    assert set(summaries[0]) <= set(SUMMARY_FIELDS)
    assert cache_path.exists()


def test_unchanged_files_are_served_from_cache(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    for episode in range(1, 4):
        _write_checkpoint(checkpoints, episode)
    first = load_checkpoint_summaries(checkpoints, cache_path)

    with patch.object(summary_cache, "_summarize", side_effect=AssertionError("re-parsed")):
        second = load_checkpoint_summaries(checkpoints, cache_path)

    assert second == first
    assert len(second) == 3


def test_only_new_or_changed_files_are_parsed(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1)
    changed = _write_checkpoint(checkpoints, 2)
    load_checkpoint_summaries(checkpoints, cache_path)

    _write_checkpoint(checkpoints, 2, curriculum_stage=5, run_id="longer-run-id-changes-size")
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    _write_checkpoint(checkpoints, 3)

    parsed: list[str] = []
    real_summarize = summary_cache._summarize

    def tracking_summarize(path: Path):
        parsed.append(path.name)
        return real_summarize(path)

    progress_calls: list[tuple[int, int]] = []
    with patch.object(summary_cache, "_summarize", side_effect=tracking_summarize):
        summaries = load_checkpoint_summaries(
            checkpoints, cache_path, progress=lambda done, total: progress_calls.append((done, total))
        )

    assert sorted(parsed) == ["checkpoint-000002.json", "checkpoint-000003.json"]
    assert progress_calls[-1] == (2, 2)
    by_episode = {s["checkpoint_episode"]: s for s in summaries}
    assert by_episode[2]["curriculum_stage"] == 5
    assert set(by_episode) == {1, 2, 3}


def test_deleted_files_are_dropped(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1)
    doomed = _write_checkpoint(checkpoints, 2)
    load_checkpoint_summaries(checkpoints, cache_path)

    doomed.unlink()
    summaries = load_checkpoint_summaries(checkpoints, cache_path)

    assert [s["checkpoint_episode"] for s in summaries] == [1]
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(cached["entries"]) == {"checkpoint-000001.json"}


def test_unreadable_checkpoints_are_skipped_and_not_reparsed(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1)
    (checkpoints / "checkpoint-000002.json").write_text("{not json", encoding="utf-8")
    (checkpoints / "checkpoint-000003.json").write_text("[1, 2, 3]", encoding="utf-8")
    (checkpoints / "notes.json").write_text("{}", encoding="utf-8")

    summaries = load_checkpoint_summaries(checkpoints, cache_path)
    assert [s["checkpoint_episode"] for s in summaries] == [1]

    with patch.object(summary_cache, "_summarize", side_effect=AssertionError("re-parsed")):
        assert load_checkpoint_summaries(checkpoints, cache_path) == summaries


def test_corrupt_or_outdated_cache_is_rebuilt(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{corrupt", encoding="utf-8")
    assert [s["checkpoint_episode"] for s in load_checkpoint_summaries(checkpoints, cache_path)] == [1]

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["fields"] = ["checkpoint_episode"]  # written by an older field list
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    parsed: list[str] = []
    real_summarize = summary_cache._summarize
    with patch.object(
        summary_cache,
        "_summarize",
        side_effect=lambda path: parsed.append(path.name) or real_summarize(path),
    ):
        load_checkpoint_summaries(checkpoints, cache_path)
    assert parsed == ["checkpoint-000001.json"]


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_checkpoint_summaries(tmp_path / "nope", default_cache_path(tmp_path)) == []


def test_cache_write_failure_is_not_fatal(tmp_path: Path) -> None:
    checkpoints, cache_path = _setup(tmp_path)
    _write_checkpoint(checkpoints, 1)
    with patch.object(summary_cache, "atomic_write_json", side_effect=OSError("disk full")):
        summaries = load_checkpoint_summaries(checkpoints, cache_path)
    assert [s["checkpoint_episode"] for s in summaries] == [1]
