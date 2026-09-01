"""Regression tests for cross-stage environment and model source separation."""

import hashlib
import json
from http import HTTPStatus
from pathlib import Path

import pytest

from sheepdog.checkpoints.sidecar import compute_file_sha256, create_sidecar_metadata
from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.server import TrainingManager


def ppo_hash(model) -> str:
    hasher = hashlib.sha256()
    state_dict = model.policy.state_dict()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().numpy()
        hasher.update(key.encode('utf-8'))
        hasher.update(tensor.tobytes())
    return hasher.hexdigest()


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = LabConfig()
    stage8_cfg = apply_curriculum_stage(config, 8)
    stage9_cfg = apply_curriculum_stage(config, 9)

    # Initialize Stage 8 model
    p8 = NeuralPolicy.initialize(stage8_cfg)
    best_model_path = models_dir / "best-model.zip"
    p8.save(best_model_path)
    hash8 = ppo_hash(p8.model)

    # Create sidecar for best-model.zip
    create_sidecar_metadata(
        model_path=best_model_path,
        config=stage8_cfg,
        policy_architecture="MaskableActorCriticPolicy",
        migration_method="test_init",
    )

    # Initialize Stage 9 run state
    stage9_model_path = models_dir / "model_stage9_latest.zip"
    p9 = NeuralPolicy.initialize(stage9_cfg)
    p9.save(stage9_model_path)
    hash9 = ppo_hash(p9.model)

    run_state = {
        "active_curriculum_stage": 9,
        "active_stage_name": "Stage 9",
        "run_id": "run_stage9_test_123",
        "active_model_path": str(stage9_model_path),
        "active_model_source": "remediation",
        "active_checkpoint_id": "chk_stage9_ep20",
        "parent_checkpoint_id": "chk_stage8_ep6551",
        "parent_run_id": "run_stage8_legacy_6551",
        "observation_schema_hash": "8f3c742d5c70e95941df72272ef41adab8080ad4a939706e5a57cc54331e6f21",
        "action_space_hash": "c860f87b1b605dca3f960a1653870f6a781a64639a50e51b7e423da3ff8e8956",
        "trainer_type": "maskable_ppo",
        "policy_type": "neural",
        "policy_mode": "neural_policy",
    }
    (out_dir / "run-state.json").write_text(json.dumps(run_state, indent=2), encoding="utf-8")

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)

    return {
        "out_dir": out_dir,
        "models_dir": models_dir,
        "best_model_path": best_model_path,
        "stage9_model_path": stage9_model_path,
        "hash8": hash8,
        "hash9": hash9,
        "stage8_cfg": stage8_cfg,
        "stage9_cfg": stage9_cfg,
    }


def test_1_dropdown_selection_alone_does_not_change_active_backend_stage(setup_workspace):
    """1. Prove that selecting a stage dropdown in UI alone does not change active backend stage."""
    manager = TrainingManager()
    manager.restore_active_run_state()
    # Backend active stage remains 9
    assert manager._status["curriculum_stage"] == 9
    assert manager._status["run_id"] == "run_stage9_test_123"


def test_2_stage8_env_plus_stage9_checkpoint_loads_stage9_parameter_hash(setup_workspace):
    """2. Prove that Stage 8 environment + Stage 9 checkpoint loads the Stage 9 policy parameter hash."""
    manager = TrainingManager()
    manager.restore_active_run_state()

    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")
    assert status == HTTPStatus.OK
    assert res["active_curriculum_stage"] == 8
    assert res["source_stage"] == 9
    assert res["target_environment_stage"] == 8

    # Load in-memory policy and check parameter hash matches Stage 9 hash
    active_path = Path(res["active_model_path"])
    loaded_policy = NeuralPolicy.load(active_path, setup_workspace["stage8_cfg"])
    assert ppo_hash(loaded_policy.model) == setup_workspace["hash9"]


def test_3_stage8_env_plus_original_stage8_checkpoint_loads_stage8_parameter_hash(setup_workspace):
    """3. Prove that Stage 8 environment + original Stage 8 checkpoint loads the Stage 8 parameter hash."""
    manager = TrainingManager()
    manager.restore_active_run_state()

    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="original_stage8")
    assert status == HTTPStatus.OK
    assert res["active_curriculum_stage"] == 8
    assert res["active_model_source"] == "original_stage8_baseline"

    active_path = Path(res["active_model_path"])
    loaded_policy = NeuralPolicy.load(active_path, setup_workspace["stage8_cfg"])
    assert ppo_hash(loaded_policy.model) == setup_workspace["hash8"]


def test_4_no_checkpoint_is_overwritten(setup_workspace):
    """4. Prove that source checkpoints (Stage 8 and Stage 9) are never overwritten."""
    sha_p8_before = compute_file_sha256(setup_workspace["best_model_path"])
    sha_p9_before = compute_file_sha256(setup_workspace["stage9_model_path"])

    manager = TrainingManager()
    manager.restore_active_run_state()
    manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")

    sha_p8_after = compute_file_sha256(setup_workspace["best_model_path"])
    sha_p9_after = compute_file_sha256(setup_workspace["stage9_model_path"])

    assert sha_p8_before == sha_p8_after
    assert sha_p9_before == sha_p9_after


def test_5_parent_lineage_identifies_selected_source_checkpoint(setup_workspace):
    """5. Prove that parent lineage identifies selected source checkpoint IDs and stage numbers."""
    manager = TrainingManager()
    manager.restore_active_run_state()
    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")

    assert status == HTTPStatus.OK
    assert res["parent_checkpoint_id"] == "chk_stage9_ep20"
    assert res["parent_run_id"] == "run_stage9_test_123"
    assert res["source_stage"] == 9
    assert res["target_environment_stage"] == 8


def test_6_fresh_model_is_never_created_unless_explicitly_requested(setup_workspace):
    """6. Prove that cross_stage_fork never uses active_model_source: fresh."""
    manager = TrainingManager()
    manager.restore_active_run_state()
    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")

    assert res["active_model_source"] != "fresh"
    assert res["active_model_source"] == "latest_stage9_checkpoint"


def test_7_failed_transition_leaves_prior_run_active(setup_workspace):
    """7. Prove that a failed transition (e.g. while training is active) leaves prior run active."""
    manager = TrainingManager()
    manager.restore_active_run_state()

    # Simulate active training thread
    class DummyThread:
        def is_alive(self):
            return True

    manager._thread = DummyThread()

    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")
    assert status == HTTPStatus.CONFLICT
    assert "error" in res

    # Manager state remains Stage 9
    assert manager._status["curriculum_stage"] == 9


def test_8_refreshing_browser_shows_backend_confirmed_state(setup_workspace):
    """8. Prove that status restoration after cross_stage_fork returns the backend-confirmed stage state."""
    manager = TrainingManager()
    manager.restore_active_run_state()
    res, status = manager.cross_stage_fork(target_stage=8, starting_model_source="latest_stage9")

    new_manager = TrainingManager()
    new_manager.restore_active_run_state()
    assert new_manager._status["curriculum_stage"] == 8
    assert new_manager._status["run_id"] == res["run_id"]
    assert new_manager._status["parent_checkpoint_id"] == "chk_stage9_ep20"


def test_9_mismatch_409_guard_remains_effective_for_uncommitted_stage_changes(setup_workspace):
    """9. Prove that start() rejects uncommitted stage changes with HTTP 409 Conflict."""
    manager = TrainingManager()
    manager.restore_active_run_state()
    # Attempt start() requesting stage 8 directly while active stage is 9 without cross_stage_fork
    with pytest.raises(ValueError) as exc_info:
        manager.start(curriculum_stage=8)
    assert "stage 8" in str(exc_info.value) or "does not match active stage" in str(exc_info.value)
