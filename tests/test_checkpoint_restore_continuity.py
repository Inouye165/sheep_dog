"""Regression tests ensuring restored checkpoints preserve model lineage and continue learning."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from sheepdog.checkpoints.store import get_action_space_hash, get_observation_schema_hash
from sheepdog.config import LabConfig, RewardConfig, TrainingConfig
from sheepdog.server import TrainingManager
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


def test_restore_checkpoint_populates_signature(tmp_path: Path) -> None:
    """Ensure restore_checkpoint writes a valid training_signature, timesteps, and stage to training-state.json."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    checkpoints_dir = artifacts / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True)

    dummy_model = models_dir / "maskable-ppo-stage15-87477882.zip"
    dummy_model.write_bytes(b"PK\x03\x04dummy")

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(tmp_path / "web"),
        )
    )
    obs_hash = get_observation_schema_hash(config)
    act_hash = get_action_space_hash()

    # Create a checkpoint file without a training_signature (legacy/audit checkpoint)
    chk_payload = {
        "checkpoint_id": "chk_run_20260817_063453_91f4_pv_3579_ts_87477882",
        "checkpoint_episode": 20697,
        "total_training_episodes": 20697,
        "environment_episodes_total": 156483,
        "success_rate": 1.0,
        "average_reward": 409.77,
        "average_completion_steps": 272.9,
        "policy_state_path": str(dummy_model),
        "policy_config": {
            "hidden_sizes": [128, 128, 128],
            "observation_size": 54,
            "action_size": 9,
            "env_workers": 1,
        },
        "observation_schema_hash": obs_hash,
        "action_space_hash": act_hash,
        "environment_config": {"curriculum_stage": 15},
        "run_id": "run_20260817_063453_91f4",
    }

    chk_file = checkpoints_dir / "checkpoint-020697.json"
    chk_file.write_text(json.dumps(chk_payload), encoding="utf-8")

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig), \
         patch("sb3_contrib.MaskablePPO.load", MagicMock()):
        manager = TrainingManager()
        res, status_code = manager.restore_checkpoint(20697)

        assert status_code == 200
        assert res["status"] == "success"

        # Verify state payload
        state_file = artifacts / "training-state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text(encoding="utf-8"))

        assert state["total_episodes_trained"] == 20697
        assert state["total_timesteps"] == 87477882
        assert state["best_model_curriculum_stage"] == 15
        assert isinstance(state.get("training_signature"), dict)
        assert state["training_signature"]["action_size"] == 9
        assert state["training_signature"]["observation_mode"] == "guided"


def test_maskable_ppo_trainer_recognizes_restored_state_as_compatible(tmp_path: Path) -> None:
    """Ensure MaskablePPOTrainer treats the restored state as compatible and resumes from it."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True)

    dummy_model = models_dir / "best-model.zip"
    dummy_model.write_bytes(b"PK\x03\x04dummy")

    state_payload = {
        "total_episodes_trained": 20697,
        "total_environment_episodes": 156483,
        "total_timesteps": 87477882,
        "policy_state_path": str(dummy_model),
        "best_model_path": str(dummy_model),
        "best_model_curriculum_stage": 15,
        "best_success_rate": 1.0,
        "policy_config": {
            "hidden_sizes": [128, 128, 128],
            "observation_size": 54,
            "action_size": 9,
            "env_workers": 1,
        },
    }
    state_file = artifacts / "training-state.json"
    state_file.write_text(json.dumps(state_payload), encoding="utf-8")

    config = LabConfig()
    trainer = MaskablePPOTrainer(config, artifacts)

    assert trainer.total_episodes_trained == 20697
    assert trainer.has_compatible_policy_state() is True


def test_trainer_compatible_across_reward_scale_modifications(tmp_path: Path) -> None:
    """Tuning reward scales must never invalidate an existing trained model."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True)

    dummy_model = models_dir / "best-model.zip"
    dummy_model.write_bytes(b"PK\x03\x04dummy")

    state_payload = {
        "total_episodes_trained": 20697,
        "policy_state_path": str(dummy_model),
        "best_model_path": str(dummy_model),
        "policy_config": {
            "hidden_sizes": [128, 128, 128],
            "observation_size": 54,
            "action_size": 9,
        },
        "training_signature": {
            "action_size": 9,
            "observation_mode": "guided",
            "rewards": {
                "farthest_sheep_progress_scale": 0.5,
                "stray_ignore_penalty_scale": 0.005,
            },
        },
    }
    state_file = artifacts / "training-state.json"
    state_file.write_text(json.dumps(state_payload), encoding="utf-8")

    # Current config has modified reward scales
    modified_rewards = RewardConfig(
        farthest_sheep_progress_scale=0.99,
        stray_ignore_penalty_scale=0.05,
    )
    config = LabConfig(rewards=modified_rewards)

    trainer = MaskablePPOTrainer(config, artifacts)
    assert trainer.has_compatible_policy_state() is True


def test_policy_load_called_instead_of_initialize_when_resuming(tmp_path: Path) -> None:
    """Ensure training calls POLICY_CLASS.load rather than initialize when resuming a restored checkpoint."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True)

    dummy_model = models_dir / "best-model.zip"
    dummy_model.write_bytes(b"PK\x03\x04dummy")

    state_payload = {
        "total_episodes_trained": 20697,
        "policy_state_path": str(dummy_model),
        "best_model_path": str(dummy_model),
        "policy_config": {
            "hidden_sizes": [128, 128, 128],
            "observation_size": 54,
            "action_size": 9,
        },
    }
    state_file = artifacts / "training-state.json"
    state_file.write_text(json.dumps(state_payload), encoding="utf-8")

    training_cfg = TrainingConfig(
        checkpoint_episodes=(1,),
        total_timesteps=100,
        output_dir=str(artifacts),
        web_export_dir=str(tmp_path / "web"),
    )
    config = LabConfig(training=training_cfg)

    trainer = MaskablePPOTrainer(config, artifacts)

    mock_policy = MagicMock()
    mock_policy.config.to_dict.return_value = {"hidden_sizes": [128, 128, 128]}
    with patch.object(MaskablePPOTrainer, "POLICY_CLASS") as mock_policy_cls:
        mock_policy_cls.load.return_value = mock_policy

        # Stop immediately after starting the training loop
        trainer.train(should_stop=lambda: True)

        # Assert POLICY_CLASS.load was called with the model, NOT initialize()
        mock_policy_cls.load.assert_called_once()
        mock_policy_cls.initialize.assert_not_called()

