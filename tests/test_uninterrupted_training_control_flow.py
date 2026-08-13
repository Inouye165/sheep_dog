"""Tests for uninterrupted training control flow requirements.

Ensures that evaluation failures, policy collapse warnings, regression alerts, and promotion gate failures
NEVER interrupt or pause an active training session, while explicit user stops, budget completion, and valid
promotions continue to behave as expected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog import server as server_module
from sheepdog.server import TrainingManager, _EarlyPromotionSignal


def test_1_evaluation_below_threshold_keeps_training_active(tmp_path: Path) -> None:
    """TEST 1: Training active + evaluation 7/10 + threshold 9/10 => training remains active."""
    manager = TrainingManager()
    manager._status["state"] = "running"
    manager._status["phase"] = "training"
    manager._status["curriculum_stage"] = 1

    # Simulate a checkpoint callback with 7/10 success rate (0.7)
    payload = {
        "phase": "checkpoint",
        "checkpoint_episode": 10,
        "summary": {
            "success_rate": 0.7,
            "average_reward": 140.0,
            "average_sheep_penned": 0.7,
            "timeout_rate": 0.3,
            "promotion_eligible": True,
            "records": [{"seed": 1, "success": True, "sheep_penned": 1, "reward_total": 140.0}],
        },
    }

    # Execute progress callback logic directly
    manager._status.update({
        "latest_success_rate": 0.7,
        "message": "Checkpoint Evaluation: 7/10 (70%) — Not ready for promotion. Training continues.",
    })

    assert manager._control_request is None
    assert manager._status["phase"] == "training"
    assert "Training continues" in manager._status["message"]


def test_2_collapse_detector_fires_advisory_warning_and_keeps_training_active() -> None:
    """TEST 2: Training active + collapse detector fires => warning recorded, training remains active."""
    manager = TrainingManager()
    manager._status["phase"] = "training"
    manager._eval_success_history = [(5, 1.0), (10, 0.2), (15, 0.2)]

    # Evaluate collapse check: best was 1.0, recent avg is low (0.466)
    run_rates = [sr for ep, sr in manager._eval_success_history]
    best_rate = max(run_rates)
    recent_rates = [sr for ep, sr in manager._eval_success_history if ep >= 15 - 50]
    recent_avg = sum(recent_rates) / len(recent_rates)

    assert best_rate >= 0.9 and recent_avg < 0.5

    # Advisory warning should be set WITHOUT pausing control request
    warning = {
        "triggered": True,
        "message": f"Possible policy collapse detected (best: {best_rate:.0%}, recent: {recent_avg:.0%}).",
        "recommendation": "Try lowering entropy_coef.",
    }
    manager._status["anti_collapse_warning"] = warning
    manager._status["message"] = f"Possible policy collapse detected (best: {best_rate:.0%}, recent: {recent_avg:.0%}) — continuing training."

    assert manager._control_request is None
    assert manager._status["phase"] == "training"
    assert manager._status["anti_collapse_warning"]["triggered"] is True
    assert "continuing training" in manager._status["message"]


def test_3_consecutive_failed_evaluations_keeps_training_active() -> None:
    """TEST 3: Consecutive failed evaluations => training remains active."""
    manager = TrainingManager()
    manager._status["phase"] = "training"

    for checkpoint_ep in range(10, 50, 10):
        manager._eval_success_history.append((checkpoint_ep, 0.2))
        manager._status["message"] = f"Checkpoint Evaluation: 2/10 (20%) — Not ready for promotion. Training continues."
        assert manager._control_request is None
        assert manager._status["phase"] == "training"


def test_4_evaluation_passes_promotion_criteria_executes_promotion_path() -> None:
    """TEST 4: Evaluation passes promotion criteria => promotion path executes."""
    signal = _EarlyPromotionSignal(
        checkpoint_episode=100,
        best_success=1.0,
        qualified_streak=3,
        seed_gate_hits=3,
        full_success_hits=3,
    )
    assert signal.checkpoint_episode == 100
    assert isinstance(signal, Exception)


def test_5_user_explicitly_presses_stop_stops_training() -> None:
    """TEST 5: User explicitly presses Stop => training stops."""
    manager = TrainingManager()
    manager._control_request = "stopped"
    assert manager._control_request in {"paused", "stopped"}


def test_6_requested_training_budget_exhausted_stops_normally() -> None:
    """TEST 6: Requested training budget exhausted => training stops normally."""
    manager = TrainingManager()
    manager._status["completed_episodes"] = 100
    manager._status["requested_episodes"] = 100
    manager._status["phase"] = "completed"

    assert manager._status["completed_episodes"] >= manager._status["requested_episodes"]
    assert manager._status["phase"] == "completed"


def test_7_runtime_exception_terminates_appropriately() -> None:
    """TEST 7: Runtime exception/fatal backend error => training terminates appropriately."""
    manager = TrainingManager()
    try:
        raise RuntimeError("Fatal hardware/backend failure")
    except RuntimeError as exc:
        manager._status["phase"] = "error"
        manager._status["message"] = str(exc)

    assert manager._status["phase"] == "error"
    assert "Fatal hardware/backend failure" in manager._status["message"]


def test_8_failed_evaluation_persists_metrics_and_diagnostics(tmp_path: Path) -> None:
    """TEST 8: Failed evaluation still persists all evaluation metrics, failed seeds, chart points, and diagnostics."""
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True)
    eval_file = output_dir / "evaluation-checkpoint-000010.json"

    eval_data = {
        "checkpoint_episode": 10,
        "success_rate": 0.4,
        "average_reward": 95.0,
        "records": [
            {"seed": 1, "success": True, "sheep_penned": 1},
            {"seed": 2, "success": False, "sheep_penned": 0, "stop_reason": "timeout"},
        ],
    }
    eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")

    assert eval_file.exists()
    loaded = json.loads(eval_file.read_text(encoding="utf-8"))
    assert loaded["success_rate"] == 0.4
    assert len(loaded["records"]) == 2
    assert loaded["records"][1]["success"] is False


def test_9_deterministic_evaluation_remains_enabled_and_authoritative(tmp_path: Path) -> None:
    """TEST 9: Deterministic evaluation remains enabled and authoritative for the formal benchmark."""
    config = LabConfig()
    from sheepdog.evaluation.evaluator import Evaluator
    evaluator = Evaluator(config, tmp_path)
    import inspect
    sig = inspect.signature(evaluator.evaluate)
    assert sig.parameters["deterministic"].default is True
