"""Tests for clean-sheet emergent_gate server config hyperparam loading and stage backups."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sheepdog.checkpoints.store import get_action_space_hash, get_observation_schema_hash
from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.server import (
    TrainingManager,
    _apply_user_hyperparams,
    _expected_observation_size,
    _read_explicit_user_hyperparams,
)
from sheepdog.training.backup import TrainingBackupManager


def test_apply_user_hyperparams_clean_sheet_emergent_gate(tmp_path: Path) -> None:
    """Verify _apply_user_hyperparams converts strides to tuple and builds 208-input obs across stages."""
    hyperparams_payload = {
        "training": {
            "trainer_type": "maskable_ppo",
            "policy_type": "neural",
            "observation_mode": "emergent_gate",
            "observation_history_strides": [0, 4, 8, 16],
            "include_loop_flags_in_observation": False,
            "learning_rate": 3e-4,
            "learning_rate_final": 1e-4,
            "ppo_env_workers": 8,
            "backup_enabled": True,
            "replay_export_on_promotion": True,
            "replay_export_on_new_best": True,
            "replay_export_every_n_checkpoints": 10,
            "auto_promote": True,
        }
    }
    (tmp_path / "user-hyperparams.json").write_text(
        json.dumps(hyperparams_payload, indent=2), encoding="utf-8"
    )

    base = LabConfig()
    explicit = _read_explicit_user_hyperparams(tmp_path, base)
    cfg = _apply_user_hyperparams(base, explicit)

    assert cfg.training.observation_history_strides == (0, 4, 8, 16)
    assert isinstance(cfg.training.observation_history_strides, tuple)
    assert cfg.training.include_loop_flags_in_observation is False
    assert cfg.training.observation_mode == "emergent_gate"
    assert cfg.training.learning_rate == pytest.approx(3e-4)
    assert cfg.training.learning_rate_final == pytest.approx(1e-4)
    assert cfg.training.ppo_env_workers == 8

    for stage in (1, 5, 16, 34):
        profiled = apply_training_profile(
            LabConfig(),
            enable_instinct_rewards=False,
            curriculum_stage=stage,
            debug_reward_breakdown=False,
        )
        stage_cfg = _apply_user_hyperparams(profiled, explicit)
        env = SheepdogEnvironment(stage_cfg)
        env.reset(seed=42)
        obs = env.build_observation_for_dog(0)
        assert len(obs.values) == 208, f"Expected 208-input obs at stage {stage}, got {len(obs.values)}"
        assert obs.metadata["observation_history_strides"] == (0, 4, 8, 16)


def test_training_backup_manager_backup_completed_stage(tmp_path: Path) -> None:
    """Verify TrainingBackupManager.backup_completed_stage writes milestone dir, manifest, and .zip copy."""
    backup_root = tmp_path / "backups"
    src_model = tmp_path / "best-model.zip"
    src_model.write_bytes(b"dummy-zip-model-bytes")

    manager = TrainingBackupManager(backup_root=backup_root, output_root=tmp_path)
    manifest = manager.backup_completed_stage(
        stage=1,
        model_path=src_model,
        checkpoint_payload={"checkpoint_id": "chk_1", "policy_version": 1, "success_rate": 0.9},
        config_dict={"training": {"observation_mode": "emergent_gate"}},
        metrics={"success_rate": 0.9},
    )

    stage_dir = backup_root / "stages" / "stage_1"
    manifest_path = stage_dir / "stage_1_manifest.json"
    copied_zip = stage_dir / "stage_1_best_model.zip"

    assert stage_dir.is_dir()
    assert manifest_path.is_file()
    assert copied_zip.is_file()
    assert copied_zip.read_bytes() == b"dummy-zip-model-bytes"
    assert manifest["stage"] == 1
    assert manifest["has_model"] is True
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["checkpoint_id"] == "chk_1"


def test_resolve_precedence_state_applies_emergent_gate_user_hyperparams(tmp_path: Path) -> None:
    """Verify _resolve_precedence_state applies user-hyperparams.json before computing obs_hash."""
    hyperparams_payload = {
        "training": {
            "observation_mode": "emergent_gate",
            "observation_history_strides": [0, 4, 8, 16],
            "include_loop_flags_in_observation": False,
        }
    }
    (tmp_path / "user-hyperparams.json").write_text(
        json.dumps(hyperparams_payload, indent=2), encoding="utf-8"
    )

    base = LabConfig()
    emergent_cfg = _apply_user_hyperparams(base, hyperparams_payload)
    emergent_obs_hash = get_observation_schema_hash(emergent_cfg)
    assert emergent_obs_hash != get_observation_schema_hash(base)

    run_state_payload = {
        "observation_schema_hash": emergent_obs_hash,
        "action_space_hash": get_action_space_hash(),
        "active_curriculum_stage": 1,
        "policy_type": "neural",
    }
    (tmp_path / "run-state.json").write_text(
        json.dumps(run_state_payload, indent=2), encoding="utf-8"
    )

    manager = TrainingManager.__new__(TrainingManager)
    state = manager._resolve_precedence_state(tmp_path)
    assert state["recovery_status"] == "success"
    assert state["observation_schema_hash"] == emergent_obs_hash


def test_expected_observation_size_respects_mode_strides_and_loop_flags() -> None:
    """Verify _expected_observation_size accounts for mode, strides, and loop flags."""
    default_cfg = LabConfig()
    assert _expected_observation_size(default_cfg) == 54

    gate_treatment_cfg = replace(
        default_cfg,
        training=replace(default_cfg.training, observation_mode="guided_gate_treatment"),
    )
    assert _expected_observation_size(gate_treatment_cfg) == 58

    emergent_gate_cfg = replace(
        default_cfg,
        training=replace(
            default_cfg.training,
            observation_mode="emergent_gate",
            observation_history_strides=(0, 4, 8, 16),
            include_loop_flags_in_observation=False,
        ),
    )
    assert _expected_observation_size(emergent_gate_cfg) == 208
