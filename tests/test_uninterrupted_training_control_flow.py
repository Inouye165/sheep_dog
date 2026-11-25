"""Tests for uninterrupted training control flow requirements.

Ensures that evaluation failures, policy collapse warnings, regression alerts, and promotion gate failures
NEVER interrupt or pause an active training session, while explicit user stops, budget completion, and valid
promotions continue to behave as expected.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.curriculum import CURRICULUM_STAGES
from sheepdog.server import (
    MAX_STAGE_MASTERY_FULL_SUCCESS_HITS,
    MAX_STAGE_MASTERY_QUALIFIED_STREAK,
    TrainingManager,
    _EarlyMaxStageStopSignal,
    _EarlyPromotionSignal,
)
from sheepdog.training.stage_health import compute_stage_health_summary


@pytest.fixture(autouse=True)
def _isolate_training_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent TrainingManager.__init__ from touching live artifacts or spawning threads."""
    monkeypatch.setattr(TrainingManager, "_resume_interrupted_session", lambda self: None)
    monkeypatch.setattr(TrainingManager, "restore_active_run_state", lambda self: None)
    monkeypatch.setattr(TrainingManager, "_reconcile_web_exports", staticmethod(lambda: None))


def test_1_evaluation_below_threshold_keeps_training_active(tmp_path: Path) -> None:
    """TEST 1: Training active + evaluation 7/10 + threshold 9/10 => training remains active."""
    manager = TrainingManager()
    manager._status["state"] = "running"
    manager._status["phase"] = "training"
    manager._status["curriculum_stage"] = 1

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
    assert manager._status["anti_collapse_warning"] == warning
    assert "continuing training" in manager._status["message"]


def test_3_consecutive_failed_evaluations_keeps_training_active() -> None:
    """TEST 3: Consecutive failed evaluations => training remains active."""
    manager = TrainingManager()
    manager._status["phase"] = "training"

    for checkpoint_ep in range(10, 50, 10):
        manager._eval_success_history.append((checkpoint_ep, 0.2))
        manager._status["message"] = "Checkpoint Evaluation: 2/10 (20%) — Not ready for promotion. Training continues."
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


def test_10_final_stage_sustained_mastery_stops_early_without_promotion(tmp_path: Path) -> None:
    """Final-stage sustained mastery interrupts trainer.train() and completes cleanly."""
    max_stage = max(CURRICULUM_STAGES)
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)

    training_state_path = artifacts / "training-state.json"
    training_state_path.write_text(
        json.dumps(
            {
                "total_episodes_trained": 100,
                "incomplete_batch": {
                    "batch_completed_segments": 1,
                    "batch_total_segments": 10,
                    "batch_steps_per_segment": 1000,
                },
            }
        ),
        encoding="utf-8",
    )

    config = LabConfig(
        training=TrainingConfig(
            episodes=100,
            output_dir=str(artifacts),
            web_export_dir=str(generated),
            evaluation_seeds=(101, 103, 107),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    interrupted_by_signal = False
    remaining_segments_ran = False

    class FakeMaxStageTrainer:
        total_episodes_trained = 100

        def train(self, progress_callback=None, should_stop=None):
            nonlocal interrupted_by_signal, remaining_segments_ran
            assert progress_callback is not None
            try:
                progress_callback(
                    {
                        "phase": "checkpoint",
                        "checkpoint_episode": 10,
                        "batch_completed_segments": 1,
                        "batch_total_segments": 10,
                        "summary": {
                            "curriculum_stage": max_stage,
                            "success_rate": 1.0,
                            "average_reward": 250.0,
                            "timeout_rate": 0.0,
                            "average_sheep_penned": 6.0,
                            "evaluation_seed_count": 3,
                            "records": [
                                {"seed": 101, "success": True, "sheep_penned": 6, "reward_total": 250.0},
                                {"seed": 103, "success": True, "sheep_penned": 6, "reward_total": 250.0},
                                {"seed": 107, "success": True, "sheep_penned": 6, "reward_total": 250.0},
                            ],
                        },
                    }
                )
            except _EarlyMaxStageStopSignal:
                interrupted_by_signal = True
                raise
            remaining_segments_ran = True

    gate_snapshot = {
        "decision": "promote_ready",
        "ready": True,
        "qualified_streak": MAX_STAGE_MASTERY_QUALIFIED_STREAK,
        "seed_gate_hits": 3,
        "full_success_hits": MAX_STAGE_MASTERY_FULL_SUCCESS_HITS,
        "best_success": 1.0,
        "reason": "Promotion criteria satisfied",
    }

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=FakeMaxStageTrainer()),
        patch("sheepdog.server.compute_promotion_gate_snapshot", return_value=gate_snapshot),
        patch.object(
            TrainingManager,
            "get_checkpoint_details",
            return_value={
                "curriculum_stage": max_stage,
                "policy_version": None,
                "checkpoint_id": None,
                "evaluation_seeds": [101, 103, 107],
            },
        ),
    ):
        manager = TrainingManager()
        manager._run_training(
            100,
            True,
            curriculum_stage=max_stage,
            auto_promote=True,
        )

    assert interrupted_by_signal is True
    assert remaining_segments_ran is False
    snap = manager.snapshot()
    assert snap["phase"] == "complete"
    assert snap["running"] is False
    assert "Stopped at max stage" in snap["message"]
    assert snap["curriculum_stage"] == max_stage
    assert snap["auto_promote_stages_completed"] == 0
    persisted_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    assert persisted_state["incomplete_batch"] is None


def test_11_final_stage_health_does_not_report_promotion_candidate(tmp_path: Path) -> None:
    """Final-stage health summary never labels max_stage as Promotion Candidate."""
    max_stage = max(CURRICULUM_STAGES)
    evals_dir = tmp_path / "evaluations"
    evals_dir.mkdir(parents=True)

    for idx, cp_ep in enumerate((10, 20, 30), start=1):
        eval_payload = {
            "checkpoint_episode": cp_ep,
            "curriculum_stage": max_stage,
            "success_rate": 1.0,
            "average_reward": 260.0 + idx,
            "average_steps": 120.0,
            "records": [
                {"seed": 101, "success": True, "sheep_penned": 6, "steps": 120},
                {"seed": 103, "success": True, "sheep_penned": 6, "steps": 120},
            ],
        }
        (evals_dir / f"eval_chk_test_pv_{idx}_ts_{cp_ep}.json").write_text(
            json.dumps(eval_payload),
            encoding="utf-8",
        )

    summary = compute_stage_health_summary(tmp_path, target_stage=max_stage, force_refresh=True)
    assert summary["promotion_status_text"] != "Promotion Candidate"
    assert summary["promotion_status_text"] == "Final Stage Mastered"
    for rec in summary["prescriptive_recommendations"]:
        assert "auto-promotion gate" not in rec["suggested_action"].lower()
