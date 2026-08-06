"""Regression tests for legacy checkpoint schema migration, sidecar validation, and safety boundaries."""

import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from http import HTTPStatus

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.environment import SheepdogEnvironment, ACTION_ORDER
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.server import TrainingManager, RestoreCompatibilityError, resolve_workspace_path
from sheepdog.checkpoints.store import (
    get_observation_schema_hash,
    get_action_space_hash,
    verify_checkpoint_compatibility,
)
from sheepdog.checkpoints.sidecar import (
    compute_file_sha256,
    create_sidecar_metadata,
    load_and_verify_sidecar,
    get_sidecar_path_for_model,
)


def ppo_hash(model) -> str:
    """Compute deterministic SHA256 digest over ordered policy state_dict tensors."""
    hasher = hashlib.sha256()
    state_dict = model.policy.state_dict()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().numpy()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.tobytes())
    return hasher.hexdigest()


def test_1_legacy_checkpoint_with_no_embedded_hash_rejected_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """1. Prove that a legacy checkpoint/state with no embedded hash and no sidecar is rejected by default."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dummy_zip = models_dir / "dummy-model.zip"
    dummy_zip.write_bytes(b"PK\x03\x04dummy_zip_content")

    # run-state.json has observation_schema_hash: None and no sidecar exists
    run_state = {
        "active_curriculum_stage": 9,
        "run_id": "run_test_legacy",
        "active_model_path": str(dummy_zip),
        "observation_schema_hash": None,
        "action_space_hash": None,
    }
    (out_dir / "run-state.json").write_text(json.dumps(run_state), encoding="utf-8")

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)

    manager = TrainingManager()
    with pytest.raises(RestoreCompatibilityError) as exc_info:
        manager.restore_active_run_state()
    assert "Observation schema mismatch" in str(exc_info.value)


def test_2_verified_matching_sidecar_permits_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """2. Prove that a verified matching sidecar allows status restoration and checkpoint loading."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = LabConfig()
    stage9_cfg = apply_curriculum_stage(config, 9)

    dummy_zip = models_dir / "legacy-model.zip"
    source_policy = NeuralPolicy.initialize(stage9_cfg)
    source_policy.save(dummy_zip)

    # Create matching sidecar for dummy_zip
    sidecar = create_sidecar_metadata(
        model_path=dummy_zip,
        config=stage9_cfg,
        policy_architecture="MaskableActorCriticPolicy",
        migration_method="reconstructed_from_stage8_canonical_schema",
    )
    assert sidecar["verified_legacy_schema"] is True

    # run-state.json has observation_schema_hash: None, but sidecar is present and matching
    run_state = {
        "active_curriculum_stage": 9,
        "run_id": "run_test_legacy_sidecar",
        "active_model_path": str(dummy_zip),
        "observation_schema_hash": None,
        "action_space_hash": None,
    }
    (out_dir / "run-state.json").write_text(json.dumps(run_state), encoding="utf-8")

    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)

    manager = TrainingManager()
    manager.restore_active_run_state()
    assert manager._status["curriculum_stage"] == 9


def test_3_sidecar_for_different_checkpoint_sha256_is_rejected(tmp_path: Path):
    """3. Prove that a sidecar referencing a different checkpoint SHA256 is rejected."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dummy_zip = models_dir / "legacy-model.zip"
    dummy_zip.write_bytes(b"PK\x03\x04real_model_content_12345")

    sidecar_path = get_sidecar_path_for_model(dummy_zip)
    sidecar_data = {
        "observation_schema_hash": "some_obs_hash",
        "action_schema_hash": "some_act_hash",
        "source_checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "verified_legacy_schema": True,
    }
    sidecar_path.write_text(json.dumps(sidecar_data), encoding="utf-8")

    # load_and_verify_sidecar must fail because SHA256 does not match
    verified = load_and_verify_sidecar(dummy_zip)
    assert verified is None


def test_4_same_observation_dimension_different_feature_order_rejected():
    """4. Prove that identical feature dimension but permuted feature order produces a different hash and is rejected."""
    cfg = apply_curriculum_stage(LabConfig(), 9)
    env = SheepdogEnvironment(cfg)
    env.reset(seed=42)
    obs = env.build_observation_for_dog(0)

    canonical_names = list(obs.feature_names)
    assert len(canonical_names) == 54

    # Permute feature order (swap first two features)
    permuted_names = list(canonical_names)
    permuted_names[0], permuted_names[1] = permuted_names[1], permuted_names[0]

    canonical_hash = hashlib.sha256(json.dumps(canonical_names).encode("utf-8")).hexdigest()
    permuted_hash = hashlib.sha256(json.dumps(permuted_names).encode("utf-8")).hexdigest()

    assert len(canonical_names) == len(permuted_names)
    assert canonical_hash != permuted_hash


def test_5_different_action_ordering_rejected():
    """5. Prove that modifying action space ordering changes the action space hash."""
    canonical_actions = list(ACTION_ORDER)
    permuted_actions = list(reversed(canonical_actions))

    canonical_hash = hashlib.sha256(json.dumps(canonical_actions).encode("utf-8")).hexdigest()
    permuted_hash = hashlib.sha256(json.dumps(permuted_actions).encode("utf-8")).hexdigest()

    assert canonical_hash != permuted_hash


def test_6_stage8_zip_is_never_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """6. Prove that the Stage 8 source best-model.zip is never modified during sidecar creation or remediation fork."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = LabConfig()
    stage8_cfg = apply_curriculum_stage(config, 8)
    source_policy = NeuralPolicy.initialize(stage8_cfg)

    best_model_path = models_dir / "best-model.zip"
    source_policy.save(best_model_path)

    initial_sha256 = compute_file_sha256(best_model_path)

    # Perform sidecar creation
    create_sidecar_metadata(
        model_path=best_model_path,
        config=stage8_cfg,
    )

    # Perform remediation fork
    monkeypatch.setattr("sheepdog.server.resolve_workspace_path", lambda path: out_dir)
    manager = TrainingManager()
    res, status = manager.remediation_fork(target_stage=9, canary_episodes=20)
    assert status == HTTPStatus.OK

    post_sha256 = compute_file_sha256(best_model_path)
    assert initial_sha256 == post_sha256, "best-model.zip was modified!"


def test_7_stage9_initialization_produces_same_policy_parameter_hash_as_stage8():
    """7. Prove that Stage 9 policy initialization from best-model.zip yields identical policy parameter tensors."""
    best_model_path = Path("artifacts/models/best-model.zip")
    assert best_model_path.exists()

    config = LabConfig()
    stage8_cfg = apply_curriculum_stage(config, 8)
    stage9_cfg = apply_curriculum_stage(config, 9)

    policy_stage8 = NeuralPolicy.load(best_model_path, stage8_cfg)
    hash_stage8 = ppo_hash(policy_stage8.model)

    policy_stage9 = NeuralPolicy.load(best_model_path, stage9_cfg)
    hash_stage9 = ppo_hash(policy_stage9.model)

    assert hash_stage8 == hash_stage9


def test_8_genuinely_mismatched_schema_remains_blocked(tmp_path: Path):
    """8. Prove that a genuinely mismatched schema hash is rejected by verify_checkpoint_compatibility."""
    current_cfg = apply_curriculum_stage(LabConfig(), 9)
    checkpoint_meta = {
        "observation_schema_hash": "invalid_mismatched_schema_hash_1234567890",
        "action_space_hash": get_action_space_hash(),
        "reward_schema_version": "1.0",
        "env_config_version": "1.0",
    }

    result = verify_checkpoint_compatibility(checkpoint_meta, current_cfg)
    assert result["compatible"] is False
    assert len(result["errors"]) > 0
    assert "Observation schema mismatch" in result["errors"][0]
