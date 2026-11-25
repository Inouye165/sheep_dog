from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path

import pytest

from sheepdog import server
from sheepdog.checkpoints.store import compute_seed_set_id
from sheepdog.config import LabConfig
from sheepdog.tracking.registry import CheckpointRecord, EvaluationRecordEntry, TrackingRegistry
from sheepdog.training.episode_store import EpisodeStore
from sheepdog.training.runtime import TrainingRuntimeTracker


def test_stage_timing_is_scoped_and_durable_without_paused_or_offline_time(tmp_path: Path) -> None:
    seconds = 0.0

    def monotonic() -> float:
        return seconds

    def utc_now() -> datetime.datetime:
        return datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(seconds=seconds)

    path = tmp_path / "runtime.json"
    options = {
        "monotonic": monotonic,
        "utc_now": utc_now,
        "start_heartbeat_thread": False,
    }
    tracker = TrainingRuntimeTracker(path, **options)
    tracker.start_session("run-a", stage=3)
    seconds = 10
    tracker.transition("evaluation")
    seconds = 14
    tracker.transition("paused")
    seconds = 100
    tracker.start_session("run-a", stage=4)
    tracker.transition("training")
    seconds = 110
    tracker.heartbeat()
    seconds = 900

    # Treat the previous session as crashed and recover from its persisted heartbeat.
    restarted = TrainingRuntimeTracker(path, **options)
    restarted._process_is_alive = lambda _: False
    restarted._close_stale_session()
    restarted.start_session("run-a", stage=4)
    seconds = 907
    restarted.start_session("run-b", stage=4)
    seconds = 910
    restarted.end_session("stopped")
    assert restarted.stage_totals() == [
        {
            "run_id": "run-a", "curriculum_stage": 3, "training_seconds": 10,
            "evaluation_seconds": 4, "active_seconds": 14,
        },
        {
            "run_id": "run-a", "curriculum_stage": 4, "training_seconds": 17,
            "evaluation_seconds": 0, "active_seconds": 17,
        },
        {
            "run_id": "run-b", "curriculum_stage": 4, "training_seconds": 3,
            "evaluation_seconds": 0, "active_seconds": 3,
        },
    ]
    assert restarted.snapshot()["paused_seconds"] == 86


def test_stage_report_separates_runs_stages_and_evaluation_suites(tmp_path: Path, monkeypatch) -> None:
    config = LabConfig()
    config = replace(config, training=replace(config.training, output_dir=str(tmp_path)))
    monkeypatch.setattr(server, "LabConfig", lambda: config)
    db = tmp_path / "training-telemetry.sqlite"
    store = EpisodeStore(db)
    with store._get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO training_episodes (event_key, run_id, curriculum_stage, seed, success, result)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("a", "run-a", 2, 53, 0, "timeout"),
                ("b", "run-a", 2, 53, 1, "success"),
                ("c", "run-a", 2, 98, 0, "stopped"),
                ("d", "run-a", 3, 53, 0, "timeout"),
                ("e", "run-b", 2, 53, 0, "timeout"),
            ],
        )
        conn.commit()
    store.close()
    registry = TrackingRegistry(db)
    for ck, run_id, stage in (("chk-a", "run-a", 2), ("chk-b", "run-b", 2)):
        registry.record_checkpoint(CheckpointRecord(
            checkpoint_record_id=f"chk_rec_{ck}", checkpoint_id=ck,
            run_id=run_id, curriculum_stage=stage, checkpoint_label_episode=21,
        ))
        for suite, manifest in (
            ("confidence", compute_seed_set_id(config.training.evaluation_seeds)),
            ("custom", "unclassified"),
            ("holdout", "holdout-manifest"),
        ):
            registry.record_evaluation(EvaluationRecordEntry(
                registry_eval_id=f"{ck}-{suite}", legacy_evaluation_id=f"id-{ck}-{suite}",
                checkpoint_id=ck, evaluated_at="2026-11-24T12:00:00Z",
                seed_set_id=manifest, environment_config_hash="test-config",
                metadata={"seed_set_role": "held_out"} if suite == "holdout" else None,
                evaluation_mode=suite, per_seed_results=[{
                    "seed": 53, "success": False, "stop_reason": "timeout",
                    "sheep_penned": 5, "replay_path": "replays/seed53.json",
                }],
            ))
    manager = object.__new__(server.TrainingManager)
    manager.runtime_tracker = TrainingRuntimeTracker(
        tmp_path / "training-runtime.json", start_heartbeat_thread=False
    )
    report = manager.get_stage_report(2, "run-a")
    assert report["training"]["sampled_episodes"] == 3
    seed53 = report["training"]["seeds"][0]
    assert (seed53["seed"], seed53["attempts"], seed53["failures"]) == (53, 2, 1)
    assert seed53["latest"]["termination_reason"] == "success"
    suites = {suite["evaluation_mode"]: suite for suite in report["evaluations"]}
    assert len(suites) == 3
    assert all(suite["checkpoint_id"] == "chk-a" for suite in suites.values())
    assert suites["confidence"]["seed_set_role"] == "development"
    assert suites["custom"]["seed_set_role"] == "unknown"
    assert suites["holdout"]["seed_set_role"] == "held_out"
    assert suites["confidence"]["environment_config_hash"] == "test-config"
    assert suites["confidence"]["seeds"][0]["sheep_penned"] == 5


def test_stage_report_can_read_an_archived_journey(tmp_path: Path, monkeypatch) -> None:
    config = LabConfig()
    config = replace(config, training=replace(config.training, output_dir=str(tmp_path)))
    monkeypatch.setattr(server, "LabConfig", lambda: config)
    journey = "20261124-120000"
    archived = tmp_path / "archive" / f"journey-{journey}" / "training-runtime.json"
    seconds = 0.0

    def monotonic() -> float:
        return seconds

    tracker = TrainingRuntimeTracker(archived, monotonic=monotonic, start_heartbeat_thread=False)
    tracker.start_session("archived-run", stage=5)
    seconds = 10
    tracker.end_session("archived")
    manager = object.__new__(server.TrainingManager)
    manager.runtime_tracker = TrainingRuntimeTracker(
        tmp_path / "training-runtime.json", start_heartbeat_thread=False
    )
    monkeypatch.setattr(manager, "_load_cached_json_dict", server._load_json)
    with pytest.raises(ValueError, match="Invalid archived journey"):
        manager.get_stage_report(5, journey="../outside")
    with pytest.raises(FileNotFoundError, match="Archived journey not found"):
        manager.get_stage_report(5, journey="20261124-130000")
    content = archived.read_bytes()
    assert manager.get_stage_report(5, "archived-run")["timing"] is None
    report = manager.get_stage_report(5, "archived-run", journey=journey)
    assert report["journey"] == journey
    assert report["timing"]["training_seconds"] == 10
    assert archived.read_bytes() == content
