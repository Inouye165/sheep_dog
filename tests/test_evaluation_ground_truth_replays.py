"""Tests for authentic evaluation ground-truth replay capture and replay store."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import InstinctOnlyPolicy
from sheepdog.replay.store import ReplayStore


def test_replay_store_write_compressed_and_read(tmp_path: Path) -> None:
    """ReplayStore writes gzipped JSON by default and transparently reads it back."""
    store = ReplayStore(tmp_path / "replays")
    payload = {
        "seed": 11,
        "steps": 45,
        "frames": [{"step": 1, "snapshot": {"penned_count": 0}}],
    }

    saved_path = store.write("test_replay.json", payload, compress=True)
    assert saved_path.exists()
    assert str(saved_path).endswith(".json.gz")

    # Verify physical file is valid gzip
    with gzip.open(saved_path, "rt", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    assert raw_data["seed"] == 11
    assert raw_data["steps"] == 45

    # Verify store.read transparently reads without needing .gz extension
    loaded_direct = store.read("test_replay.json")
    assert loaded_direct == payload

    loaded_with_gz = store.read("test_replay.json.gz")
    assert loaded_with_gz == payload


def test_replay_store_write_uncompressed_and_read(tmp_path: Path) -> None:
    """ReplayStore can write plain JSON when compression is disabled."""
    store = ReplayStore(tmp_path / "replays")
    payload = {"seed": 23, "steps": 10}

    saved_path = store.write("uncompressed.json", payload, compress=False)
    assert saved_path.exists()
    assert str(saved_path).endswith(".json")
    assert not str(saved_path).endswith(".gz")

    loaded = store.read(saved_path)
    assert loaded == payload


def test_evaluator_captures_ground_truth_replays(tmp_path: Path) -> None:
    """Evaluator captures authentic frame history matching episode stats when capture_replays=True."""
    config = LabConfig()
    evaluator = Evaluator(config, output_root=tmp_path / "eval_out")
    policy = InstinctOnlyPolicy()

    seeds = (11, 23)
    summary, json_path, _ = evaluator.evaluate(
        policy,
        seeds,
        checkpoint_episode=100,
        capture_replays=True,
        evaluation_mode="confidence",
    )

    assert json_path.exists()
    assert len(summary.records) == 2

    for record in summary.records:
        assert record.replay_path != ""
        rep_file = Path(record.replay_path)
        assert rep_file.exists()
        assert str(rep_file).endswith(".json.gz")

        # Load the recorded replay frames
        with gzip.open(rep_file, "rt", encoding="utf-8") as fh:
            bundle = json.load(fh)

        assert bundle["seed"] == record.seed
        assert "frames" in bundle
        assert len(bundle["frames"]) == record.steps
        # The final frame snapshot matches the record stats
        last_frame_snapshot = bundle["frames"][-1]["snapshot"]
        assert last_frame_snapshot["penned_count"] == record.sheep_penned


def test_server_candidate_dir_resolves_gzipped_replay(tmp_path: Path) -> None:
    """Server replay lookup correctly finds and extracts .json.gz replay files."""
    replays_dir = tmp_path / "artifacts" / "evaluations" / "replays"
    replays_dir.mkdir(parents=True)

    replay_id = "checkpoint-000123-seed-000011"
    store = ReplayStore(replays_dir)
    data = {"seed": 11, "steps": 50, "frames": [{"step": 1}]}
    gz_file = store.write(f"{replay_id}.json", data, compress=True)
    assert gz_file.name == f"{replay_id}.json.gz"
    assert gz_file.exists()

    # Emulate server candidate_dirs resolution
    candidate_dirs = [
        replays_dir,
    ]
    target_file = None
    for c_dir in candidate_dirs:
        gz_path = c_dir / f"{replay_id}.json.gz"
        json_path = c_dir / f"{replay_id}.json"
        if gz_path.exists():
            target_file = gz_path
            break
        if json_path.exists():
            target_file = json_path
            break

    assert target_file is not None
    assert str(target_file).endswith(".gz")
    with gzip.open(target_file, "rt", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == data

