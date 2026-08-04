"""Regression tests verifying backend/frontend stage synchronization and Stage 9 remediation fork."""

from __future__ import annotations
import json
import pytest
from http import HTTPStatus
from pathlib import Path
from sheepdog.config import LabConfig
from sheepdog.server import TrainingManager, resolve_workspace_path


def test_remediation_fork_atomic_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify Stage 9 remediation fork updates all persisted state sources atomically."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy Stage 8 best-model.zip
    best_model_path = models_dir / "best-model.zip"
    best_model_path.write_bytes(b"PK_DUMMY_ZIP_DATA")

    # Initial state: Stage 8 with active batch
    initial_run_state = {
        "active_curriculum_stage": 8,
        "active_stage_name": "Stage 8",
        "run_id": "run_stage8_legacy",
        "batch_total_timesteps": 22500,
        "batch_completed_timesteps": 1500,
    }
    (out_dir / "run-state.json").write_text(json.dumps(initial_run_state), encoding="utf-8")

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)
    monkeypatch.setattr("sheepdog.server._read_promotion_history", lambda path: [{"to_stage": 8}])

    manager = TrainingManager()

    # 1. Start request for Stage 9 BEFORE fork should fail because active stage is 8
    try:
        manager.start(requested_episodes=20, curriculum_stage=9)
        assert False, "Should have failed validation before remediation fork"
    except ValueError as err:
        assert "does not match active stage" in str(err)

    # 2. Execute Stage 9 remediation fork
    res, status = manager.remediation_fork(target_stage=9, canary_episodes=20)
    assert status == HTTPStatus.OK
    assert res["active_curriculum_stage"] == 9
    assert res["running"] is False
    assert res["batch_completed_timesteps"] == 0

    # 3. Verify every persisted state file updated to Stage 9
    run_state = json.loads((out_dir / "run-state.json").read_text(encoding="utf-8"))
    assert run_state["active_curriculum_stage"] == 9
    assert run_state["batch_completed_timesteps"] == 0
    assert run_state["source_stage"] == 8

    settings = json.loads((out_dir / "training-settings.json").read_text(encoding="utf-8"))
    assert settings["curriculum_stage"] == 9

    eff_cfg = json.loads((out_dir / "effective-training-config.json").read_text(encoding="utf-8"))
    assert eff_cfg["environment"]["curriculum_stage"] == 9
    assert eff_cfg["environment"]["spawn_mix"]["wall_recovery"] == 0.70

    # 4. Verify Stage 8 best-model zip remains unchanged
    assert best_model_path.exists()
    assert best_model_path.read_bytes() == b"PK_DUMMY_ZIP_DATA"


def test_remediation_fork_rejects_when_running():
    """Verify remediation fork fails with 409 if training is active."""
    manager = TrainingManager()
    with manager._lock:
        class DummyThread:
            def is_alive(self):
                return True
        manager._thread = DummyThread()

    res, status = manager.remediation_fork(target_stage=9)
    assert status == HTTPStatus.CONFLICT
    assert "Cannot create Stage 9 remediation run while training is active" in res["error"]
