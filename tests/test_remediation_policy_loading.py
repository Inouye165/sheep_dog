"""Regression test verifying Stage 9 remediation run loads Stage 8 policy and optimizer state."""

from __future__ import annotations
import hashlib
import json
from http import HTTPStatus
from pathlib import Path
import pytest
import dataclasses
from sb3_contrib import MaskablePPO
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.server import TrainingManager, resolve_workspace_path, _build_training_job_config
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


def ppo_hash(model: MaskablePPO) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(param.cpu().numpy().tobytes())
    return h.hexdigest()


def test_remediation_fork_loads_policy_and_optimizer_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify remediation fork populates signature & lineage, restoring Stage 8 policy weights & optimizer."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)

    config = LabConfig()

    # 1. Create source Stage 8 NeuralPolicy & save to best-model.zip
    source_policy = NeuralPolicy.initialize(config)
    source_hash = ppo_hash(source_policy.model)
    best_model_path = models_dir / "best-model.zip"
    source_policy.save(best_model_path)

    assert best_model_path.exists()
    saved_model = MaskablePPO.load(str(best_model_path))
    assert ppo_hash(saved_model) == source_hash

    # 2. Setup initial Stage 8 state files
    initial_run_state = {
        "active_curriculum_stage": 8,
        "active_stage_name": "Stage 8",
        "run_id": "run_stage8_legacy_1234",
        "active_checkpoint_id": "chk_stage8_ep6551",
        "batch_total_timesteps": 500000,
        "batch_completed_timesteps": 500000,
        "episodes_completed": 6551,
        "policy_version": 2559,
    }
    (out_dir / "run-state.json").write_text(json.dumps(initial_run_state), encoding="utf-8")

    initial_summary = {
        "total_episodes_trained": 6551
    }
    (out_dir / "training-summary.json").write_text(json.dumps(initial_summary), encoding="utf-8")

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)
    monkeypatch.setattr("sheepdog.server._read_promotion_history", lambda path: [{"to_stage": 8}])

    manager = TrainingManager()

    # 3. Execute Stage 9 remediation fork
    res, status = manager.remediation_fork(target_stage=9, canary_episodes=20)
    assert status == HTTPStatus.OK
    assert res["active_curriculum_stage"] == 9
    assert res["running"] is False
    assert res["parent_run_id"] == "run_stage8_legacy_1234"
    assert res["parent_checkpoint_id"] == "chk_stage8_ep6551"
    assert res["active_model_source"] == "remediation"

    # 4. Verify training-state.json includes training_signature and lineage
    training_state = json.loads((out_dir / "training-state.json").read_text(encoding="utf-8"))
    assert training_state["parent_run_id"] == "run_stage8_legacy_1234"
    assert training_state["parent_checkpoint_id"] == "chk_stage8_ep6551"
    assert training_state["active_model_source"] == "remediation"
    assert isinstance(training_state.get("training_signature"), dict)

    # 5. Verify MaskablePPOTrainer sees compatible policy state and loads source policy
    job_config = _build_training_job_config(20, fast_mode=True, curriculum_stage=9)
    job_config = dataclasses.replace(
        job_config,
        training=dataclasses.replace(job_config.training, output_dir=str(out_dir)),
    )

    trainer = MaskablePPOTrainer(job_config, out_dir)
    assert trainer.has_compatible_policy_state() is True

    # 6. Verify policy loading restores identical parameter hash & optimizer state
    resume_path = trainer._loaded_state.get("best_model_path") or trainer._loaded_state.get("policy_state_path")
    assert resume_path is not None

    loaded_policy = NeuralPolicy.load(resume_path, job_config, trainer._loaded_state.get("policy_config"))
    loaded_hash = ppo_hash(loaded_policy.model)

    assert loaded_hash == source_hash, f"Expected {source_hash}, got {loaded_hash}"
    assert loaded_policy.model.policy.optimizer is not None


def test_summary_export_failure_does_not_prevent_checkpoint_survival(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that a failure during summary export does not prevent the Stage 9 model checkpoint from surviving and reloading."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = LabConfig()
    source_policy = NeuralPolicy.initialize(config)
    best_model_path = models_dir / "best-model.zip"
    source_policy.save(best_model_path)

    initial_run_state = {
        "active_curriculum_stage": 9,
        "active_stage_name": "Stage 9",
        "run_id": "run_stage9_remediation",
        "active_checkpoint_id": "chk_stage8_source",
        "batch_total_timesteps": 100,
        "batch_completed_timesteps": 0,
        "episodes_completed": 0,
        "policy_version": 10,
    }
    (out_dir / "run-state.json").write_text(json.dumps(initial_run_state), encoding="utf-8")

    job_config = _build_training_job_config(20, fast_mode=True, curriculum_stage=9)
    job_config = dataclasses.replace(
        job_config,
        training=dataclasses.replace(
            job_config.training,
            output_dir=str(out_dir),
            checkpoint_episodes=[1],
            total_timesteps=64,
        ),
    )

    trainer = MaskablePPOTrainer(job_config, out_dir)

    def failing_export(*args, **kwargs):
        raise RuntimeError("Simulated serialization crash during summary export")

    monkeypatch.setattr(trainer, "_export_neural_training_summary", failing_export)

    summary = trainer.train()
    assert summary is not None

    # Checkpoint zip file must exist and reload cleanly
    state = json.loads((out_dir / "training-state.json").read_text(encoding="utf-8"))
    saved_path = state.get("policy_state_path")
    assert saved_path is not None
    assert Path(saved_path).exists()

    reloaded_policy = NeuralPolicy.load(saved_path, job_config)
    assert reloaded_policy is not None
    assert reloaded_policy.model is not None


def test_remediation_fork_lineage_preserves_stage8_parent_across_stage9_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify remediation fork metadata identifies Stage 8 source parent IDs and maintains them across multiple forks."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)

    config = LabConfig()
    source_policy = NeuralPolicy.initialize(config)
    best_model_path = models_dir / "best-model.zip"
    source_policy.save(best_model_path)

    # 1. Initial Stage 8 state
    initial_run_state = {
        "active_curriculum_stage": 8,
        "active_stage_name": "Stage 8",
        "run_id": "run_stage8_legacy_6551",
        "active_checkpoint_id": "chk_stage8_ep6551",
        "episodes_completed": 6551,
        "policy_version": 2559,
    }
    (out_dir / "run-state.json").write_text(json.dumps(initial_run_state), encoding="utf-8")

    manager = TrainingManager()

    # 2. First Stage 9 remediation fork
    res1, status1 = manager.remediation_fork(target_stage=9, canary_episodes=20)
    assert status1 == HTTPStatus.OK
    assert res1["active_model_source"] == "remediation"
    assert res1["active_model_path"] == str(best_model_path)
    assert res1["active_checkpoint_id"] == "chk_stage8_ep6551"
    assert res1["parent_checkpoint_id"] == "chk_stage8_ep6551"
    assert res1["parent_run_id"] == "run_stage8_legacy_6551"
    assert res1["source_stage"] == 8
    assert res1["active_curriculum_stage"] == 9

    # 3. Second Stage 9 remediation fork on the already Stage-9 workspace
    res2, status2 = manager.remediation_fork(target_stage=9, canary_episodes=20)
    assert status2 == HTTPStatus.OK
    assert res2["active_model_source"] == "remediation"
    assert res2["active_model_path"] == str(best_model_path)
    assert res2["active_checkpoint_id"] == "chk_stage8_ep6551"
    assert res2["parent_checkpoint_id"] == "chk_stage8_ep6551"
    assert res2["parent_run_id"] == "run_stage8_legacy_6551"
    assert res2["source_stage"] == 8
    assert res2["active_curriculum_stage"] == 9


