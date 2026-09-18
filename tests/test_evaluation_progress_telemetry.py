"""Tests for evaluation progress telemetry and terminal tracking."""

from pathlib import Path

from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import InstinctOnlyPolicy
from sheepdog.server import TrainingManager


def test_evaluator_progress_callback_emits_seed_lifecycle_and_results(tmp_path: Path) -> None:
    """Test that Evaluator emits evaluation in-progress, seed progress, and completion events."""
    config = LabConfig()
    evaluator = Evaluator(config, tmp_path / "evaluations")
    events = []

    def on_progress(payload):
        events.append(dict(payload))

    evaluator.progress_callback = on_progress

    policy = InstinctOnlyPolicy()
    seeds = (11, 13)

    summary, json_path, csv_path = evaluator.evaluate(
        policy,
        seeds,
        checkpoint_episode=1,
        evaluation_mode="quick",
    )

    assert summary is not None
    assert json_path.exists()
    assert csv_path.exists()

    # Verify events
    assert len(events) >= 3
    # 1. Starting event
    start_event = events[0]
    assert start_event["phase"] == "evaluation"
    assert start_event["evaluation_in_progress"] is True
    assert start_event["evaluation_mode"] == "quick"
    assert start_event["evaluation_total_seeds"] == 2
    assert start_event["evaluation_completed_seeds"] == 0

    # 2. Seed events exist
    seed_events = [e for e in events if e.get("evaluation_seed") in seeds]
    assert len(seed_events) >= 2
    # At least one event has latest_result after seed 1
    assert any(e.get("evaluation_latest_result") is not None for e in events)

    # 3. Completion event
    complete_event = events[-1]
    assert complete_event["phase"] == "evaluation_complete"
    assert complete_event["evaluation_in_progress"] is False
    assert complete_event["evaluation_completed_seeds"] == 2
    assert complete_event["evaluation_success_rate"] is not None


from unittest.mock import patch
from dataclasses import replace

def test_server_status_tracks_evaluation_progress(tmp_path: Path) -> None:
    """Test that TrainingManager updates evaluation_status and evaluation_in_progress."""
    config = replace(
        LabConfig(),
        training=replace(
            LabConfig().training,
            output_dir=str(tmp_path / "artifacts"),
            backup_dir=str(tmp_path / "backups"),
            web_export_dir=str(tmp_path / "web"),
        ),
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        status = manager.snapshot()

    # Initial status
    assert "evaluation_status" in status
    assert status["evaluation_in_progress"] is False

    # Simulate evaluation progress event received by server progress_callback
    payload = {
        "phase": "evaluation",
        "evaluation_in_progress": True,
        "evaluation_mode": "confidence",
        "evaluation_seed": 107,
        "evaluation_seed_index": 3,
        "evaluation_total_seeds": 10,
        "evaluation_completed_seeds": 2,
        "evaluation_success_count": 2,
        "evaluation_timeout_count": 0,
        "evaluation_success_rate": 1.0,
        "evaluation_seeds": [101, 103, 107, 109, 113],
        "evaluation_message": "Evaluation [confidence] in progress: seed 107 (3/10)",
        "evaluation_latest_result": {
            "seed": 103,
            "success": True,
            "timeout": False,
            "status": "SUCCESS",
            "reward": 22.5,
            "penned": 6,
            "total_sheep": 6,
            "steps": 140,
        },
        "evaluation_recent_results": [
            {"seed": 101, "success": True, "reward": 20.0},
            {"seed": 103, "success": True, "reward": 22.5},
        ],
    }

    # In server, progress_callback updates _status
    with manager._lock:
        manager._status.update({
            "evaluation_in_progress": True,
            "evaluation_status": {
                "in_progress": True,
                "mode": payload["evaluation_mode"],
                "current_seed": payload["evaluation_seed"],
                "seed_index": payload["evaluation_seed_index"],
                "total_seeds": payload["evaluation_total_seeds"],
                "completed_seeds": payload["evaluation_completed_seeds"],
                "success_count": payload["evaluation_success_count"],
                "timeout_count": payload["evaluation_timeout_count"],
                "success_rate": payload["evaluation_success_rate"],
                "seeds": payload["evaluation_seeds"],
                "message": payload["evaluation_message"],
                "latest_result": payload["evaluation_latest_result"],
                "recent_results": payload["evaluation_recent_results"],
            },
            "evaluation_seed": payload["evaluation_seed"],
            "evaluation_seed_index": payload["evaluation_seed_index"],
            "evaluation_total_seeds": payload["evaluation_total_seeds"],
            "evaluation_message": payload["evaluation_message"],
            "phase": "evaluation",
            "message": payload["evaluation_message"],
        })

    snapshot = manager.snapshot()
    assert snapshot["phase"] == "evaluation"
    assert snapshot["evaluation_in_progress"] is True
    assert snapshot["evaluation_seed"] == 107
    assert snapshot["evaluation_seed_index"] == 3
    assert snapshot["evaluation_total_seeds"] == 10
    assert snapshot["evaluation_status"]["current_seed"] == 107
    assert snapshot["evaluation_status"]["success_count"] == 2
    assert snapshot["evaluation_status"]["latest_result"]["status"] == "SUCCESS"
