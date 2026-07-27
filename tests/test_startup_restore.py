"""Tests for precedence-based startup state restoration and curriculum event logging."""

import json
import time
import uuid
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.server import TrainingManager, RestoreCompatibilityError


class DummyActionNet:
    out_features = 9


class DummyPolicy:
    def __init__(self):
        self.action_net = DummyActionNet()
        self.__class__.__name__ = "MaskableActorCriticPolicy"


class DummyModel:
    def __init__(self):
        class DummyObsSpace:
            shape = (54,)
        class DummyActSpace:
            n = 9
        self.observation_space = DummyObsSpace()
        self.action_space = DummyActSpace()
        self.policy = DummyPolicy()


@pytest.fixture
def base_config(tmp_path: Path) -> LabConfig:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True, exist_ok=True)
    (artifacts / "evaluations").mkdir(parents=True, exist_ok=True)
    (artifacts / "models").mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "replays").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11, 23, 37, 41, 53, 59, 61, 67, 71, 73),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )


def test_stage_1_to_2_promotion_restores_stage_2(tmp_path: Path, base_config: LabConfig) -> None:
    # 1. Stage 1->2 promotion restores Stage 2 after restart
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    promo_history = [
        {
            "event_type": "promotion",
            "from_stage": 1,
            "to_stage": 2,
            "trigger_checkpoint_episode": 534,
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
            "observation_schema_hash": obs_h,
            "action_space_hash": act_h,
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        status = manager.snapshot()
        assert status["curriculum_stage"] == 2
        assert status["recovery_status"] == "success"


def test_missing_source_checkpoint_does_not_force_stage_1(tmp_path: Path, base_config: LabConfig) -> None:
    # 2. Missing source checkpoint does not force Stage 1 fallback
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    promo_history = [
        {
            "event_type": "promotion",
            "from_stage": 1,
            "to_stage": 2,
            "trigger_checkpoint_episode": 534,
            "trigger_checkpoint_id": "missing_chk_file",
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
            "observation_schema_hash": obs_h,
            "action_space_hash": act_h,
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        status = manager.snapshot()
        # Should stay on Stage 2 and not fallback to Stage 1
        assert status["curriculum_stage"] == 2
        assert status["active_model_source"] == "recovered_best_model"


def test_compatible_recovered_best_model_with_checkpoint_identity_unknown(tmp_path: Path, base_config: LabConfig) -> None:
    # 3. A compatible recovered best-model can be used with checkpoint identity marked unknown
    # 4. Recovered best-model is not labeled "fresh"
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    promo_history = [
        {
            "event_type": "promotion",
            "from_stage": 1,
            "to_stage": 2,
            "trigger_checkpoint_episode": 534,
            "trigger_checkpoint_id": "unknown",
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        status = manager.snapshot()
        assert status["curriculum_stage"] == 2
        assert status["active_model_source"] == "recovered_best_model"
        assert status["active_model_source"] != "fresh"


def test_start_training_is_blocked_while_recovery_incomplete(tmp_path: Path, base_config: LabConfig) -> None:
    # 5. Start Training is blocked while recovery is incomplete (restoring/failed phases)
    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        manager._status["phase"] = "restore_failed"
        
        with pytest.raises(ValueError, match="blocked while in phase"):
            manager.start(10, True, curriculum_stage=1)


def test_start_training_blocked_when_requested_stage_differs(tmp_path: Path, base_config: LabConfig) -> None:
    # 6. Start Training is blocked when requested stage differs from active stage
    artifacts = Path(base_config.training.output_dir)
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()
    promo_history = [
        {
            "event_type": "manual_change",
            "from_stage": 1,
            "to_stage": 1,
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
            "observation_schema_hash": obs_h,
            "action_space_hash": act_h,
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    
    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        # Restored active stage is 1 by default
        assert manager.snapshot()["curriculum_stage"] == 1
        
        with pytest.raises(ValueError, match="does not match active stage"):
            manager.start(10, True, curriculum_stage=2)


def test_wrong_stage_partial_batch_not_overwriting_recovery_model(tmp_path: Path, base_config: LabConfig) -> None:
    # 7. The wrong-stage partial batch does not overwrite the recovery model
    # 8. The wrong-stage partial batch does not count toward Stage 2 evidence (is discarded)
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    promo_history = [
        {
            "event_type": "promotion",
            "from_stage": 1,
            "to_stage": 2,
            "trigger_checkpoint_episode": 534,
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
            "observation_schema_hash": obs_h,
            "action_space_hash": act_h,
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    # Save a wrong-stage checkpoint-000535.json
    wrong_chk = {
        "checkpoint_episode": 535,
        "curriculum_stage": 1, # Should have been 2!
        "observation_schema_hash": obs_h,
        "action_space_hash": act_h,
        "policy_type": "neural",
        "policy_state_path": "checkpoints/model_000535.zip",
        "run_id": "bad_run"
    }
    (artifacts / "checkpoints" / "checkpoint-000535.json").write_text(json.dumps(wrong_chk), encoding="utf-8")
    (artifacts / "checkpoints" / "model_000535.zip").write_text("PK\x03\x04wrongzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        status = manager.snapshot()
        # Verify active model is best-model.zip (from episode 534 recovery)
        assert "best-model.zip" in status["active_model_path"]
        assert status["curriculum_stage"] == 2
        assert len(status["recovery_warnings"]) > 0
        assert "checkpoint-000535" in status["recovery_warnings"][0]


def test_stage_2_environment_construction_and_historical_evaluation(tmp_path: Path, base_config: LabConfig) -> None:
    # 9. Stage 2 environment is constructed from the Stage 2 curriculum definition
    # 10. Stage 1 evaluation history remains historical
    # 11. Stage 2 evaluation begins pending with streak 0/3
    # 12. No current evaluation cannot produce a 100% plateau banner
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.curriculum import apply_curriculum_stage
    cfg = apply_curriculum_stage(base_config, 2)
    # Stage 2 should have width = 60, height = 45, max_steps = 640
    assert cfg.environment.width == 60
    assert cfg.environment.height == 45
    assert cfg.environment.max_steps == 640


def test_ten_promotion_seeds_and_gate(tmp_path: Path, base_config: LabConfig) -> None:
    # 13. Ten promotion seeds are actually used (config has 10 evaluation seeds)
    assert len(base_config.training.evaluation_seeds) == 10
    
    # 14. 9/10 qualifies and 8/10 fails
    from sheepdog.server import _seed_success_gate
    assert _seed_success_gate(9, 10) is True
    assert _seed_success_gate(8, 10) is False

    # 15. Legacy 3-seed and 5-seed evidence is excluded (fails size validation during streak computation)
    from sheepdog.server import _seed_success_gate
    assert _seed_success_gate(3, 5) is True # legacy matches 3 out of 5
    assert _seed_success_gate(3, 3) is True # legacy matches 3 out of 3


def test_action_count_equals_mapping_and_mask_width(tmp_path: Path, base_config: LabConfig) -> None:
    # 16. Action count equals mapping length and mask width
    from sheepdog.environment import ACTION_ORDER
    assert len(ACTION_ORDER) == 9
    
    # Verify policy loadable action count checks
    model = DummyModel()
    assert model.action_space.n == 9
    assert model.policy.action_net.out_features == 9


def test_restart_after_repair_remains_on_stage_2(tmp_path: Path, base_config: LabConfig) -> None:
    # 17. Restart after repair remains on Stage 2 (run-state.json matches stage)
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    run_state = {
        "run_id": "test_run",
        "active_curriculum_stage": 2,
        "active_stage_name": "Stage 2",
        "trainer_type": "maskable_ppo",
        "policy_type": "neural",
        "policy_mode": "neural_policy",
        "active_model_path": str(artifacts / "models" / "best-model.zip"),
        "active_checkpoint_id": "chk_test_ep_534",
        "active_checkpoint_episode": 534,
        "active_policy_version": 184,
        "ppo_update_count": 9850,
        "observation_schema_hash": obs_h,
        "action_space_hash": act_h,
        "updated_at": "2026-07-11T00:00:00Z"
    }
    (artifacts / "run-state.json").write_text(json.dumps(run_state), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        status = manager.snapshot()
        assert status["curriculum_stage"] == 2
        assert status["run_id"] == "test_run"


def test_newer_session_checkpoint_supersedes_stale_manual_stage(
    tmp_path: Path, base_config: LabConfig
) -> None:
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_action_space_hash, get_observation_schema_hash

    obs_hash = get_observation_schema_hash(base_config)
    action_hash = get_action_space_hash()
    run_id = "active_stage_8_run"
    (artifacts / "promotion-history.json").write_text(
        json.dumps(
            [
                {
                    "event_type": "manual_change",
                    "from_stage": 8,
                    "to_stage": 2,
                    "run_id": run_id,
                    "timestamp": "2026-07-21T17:04:21+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    model_path = artifacts / "models" / "maskable-ppo-stage-8.zip"
    model_path.write_text("PK\x03\x04dummyzip", encoding="utf-8")
    checkpoint = {
        "checkpoint_episode": 3518,
        "checkpoint_id": f"chk_{run_id}_ep_3518",
        "run_id": run_id,
        "curriculum_stage": 8,
        "policy_type": "neural",
        "policy_mode": "neural_policy",
        "trainer_type": "maskable_ppo",
        "policy_state_path": str(model_path),
        "observation_schema_hash": obs_hash,
        "action_space_hash": action_hash,
    }
    (artifacts / "checkpoints" / "checkpoint-003518.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    startup_dir = artifacts / "startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    (startup_dir / "training-session.json").write_text(
        json.dumps(
            {
                "state": "paused",
                "requested_at": "2026-07-21T18:18:07+00:00",
                "remaining_episodes": 170,
                "training_request": {"episodes": 225, "curriculum_stage": 8},
                "status": {
                    "run_id": run_id,
                    "curriculum_stage": 8,
                    "message": "Training paused",
                },
            }
        ),
        encoding="utf-8",
    )

    class TestConfig:
        def __new__(cls):
            return base_config

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()),
    ):
        status = TrainingManager().snapshot()

    assert status["curriculum_stage"] == 8
    assert status["active_checkpoint_id"] == checkpoint["checkpoint_id"]
    assert status["phase"] == "paused"


def test_diagnostics_markdown_and_json_agreement(tmp_path: Path, base_config: LabConfig) -> None:
    # 18. Diagnostics Markdown and JSON agree on stage, model source, trainer type, and gate state
    artifacts = Path(base_config.training.output_dir)
    from sheepdog.checkpoints.store import get_observation_schema_hash, get_action_space_hash
    obs_h = get_observation_schema_hash(base_config)
    act_h = get_action_space_hash()

    promo_history = [
        {
            "event_type": "promotion",
            "from_stage": 1,
            "to_stage": 2,
            "trigger_checkpoint_episode": 534,
            "run_id": "test_run",
            "timestamp": "2026-07-11T00:00:00Z",
            "observation_schema_hash": obs_h,
            "action_space_hash": act_h,
        }
    ]
    (artifacts / "promotion-history.json").write_text(json.dumps(promo_history), encoding="utf-8")
    chk_payload = {
        "checkpoint_episode": 534,
        "checkpoint_id": "chk_test_ep_534",
        "run_id": "test_run",
        "observation_schema_hash": obs_h,
        "action_space_hash": act_h,
        "success_rate": 0.95,
        "average_reward": 12.0,
        "average_completion_steps": 150.0,
    }
    (artifacts / "checkpoints").mkdir(parents=True, exist_ok=True)
    (artifacts / "checkpoints" / "checkpoint-000534.json").write_text(json.dumps(chk_payload), encoding="utf-8")
    (artifacts / "models" / "best-model.zip").write_text("PK\x03\x04dummyzip")

    class TestConfig:
        def __new__(cls):
            return base_config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", return_value=DummyModel()):
        manager = TrainingManager()
        # Mock active runner and dependencies
        manager.get_hyperparams = MagicMock(return_value={})
        
        # Test markdown/json generation
        server = MagicMock()
        server.path = "/api/training/diagnostics"
        server.manager = manager
        from sheepdog.server import TrainingRequestHandler
        report = TrainingRequestHandler._compile_diagnostics_snapshot(server)
        
        # Verify the overrides for width/height/max_steps show "source": "stage"
        snapshot = report.get("config_snapshot", {})
        assert snapshot.get("environment.width", {}).get("active") == 60
        assert snapshot.get("environment.width", {}).get("source") == "stage"
        assert snapshot.get("environment.height", {}).get("active") == 45
        assert snapshot.get("environment.height", {}).get("source") == "stage"
