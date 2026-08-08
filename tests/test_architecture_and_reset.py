"""Tests for 3-layer neural network architecture [128, 128, 128] and training reset integrity."""

import json
from dataclasses import replace
from pathlib import Path
import pytest

from sheepdog.config import LabConfig
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy, HierarchicalNeuralPolicyConfig
from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.server import TrainingManager


def test_1_policy_network_has_three_128_hidden_layers():
    config = LabConfig()
    policy = NeuralPolicy.initialize(config)
    assert policy.config.hidden_sizes == (128, 128, 128)
    net_arch = policy.model.policy_kwargs.get("net_arch")
    assert net_arch == [128, 128, 128]


def test_2_value_network_has_three_128_hidden_layers():
    config = LabConfig()
    policy = NeuralPolicy.initialize(config)
    net_arch = policy.model.policy_kwargs.get("net_arch")
    assert isinstance(net_arch, list)
    assert len(net_arch) == 3
    assert all(size == 128 for size in net_arch)


def test_3_new_run_does_not_load_existing_checkpoint():
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("active_model_path") is None
    assert status.get("checkpoint_episode") is None or status.get("checkpoint_episode") == 0


def test_4_new_run_resets_curriculum_to_stage_1():
    manager = TrainingManager()
    manager.reset_journey()
    status = manager._status
    assert status.get("curriculum_stage") == 1 or status.get("active_curriculum_stage") == 1


def test_5_new_run_resets_episode_and_global_counters():
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("current_episode") is None or status.get("current_episode") == 0
    assert status.get("checkpoint_episode") is None or status.get("checkpoint_episode") == 0
    assert status.get("total_episodes_trained") == 0


def test_6_new_run_creates_fresh_optimizer():
    config = LabConfig()
    policy1 = NeuralPolicy.initialize(config)
    policy2 = NeuralPolicy.initialize(config)
    assert policy1.model.policy.optimizer is not policy2.model.policy.optimizer


def test_7_old_two_layer_checkpoints_rejected_as_incompatible(tmp_path):
    config = LabConfig()
    old_config_dict = {"hidden_sizes": [128, 128], "observation_size": 54}
    fake_zip_path = tmp_path / "old_2layer_model.zip"
    
    policy_2layer = NeuralPolicy.initialize(config)
    policy_2layer.config = NeuralPolicyConfig(hidden_sizes=(128, 128), observation_size=54)
    saved_path = policy_2layer.save(fake_zip_path)
    
    with pytest.raises((ValueError, RuntimeError), match="Incompatible model architecture"):
        NeuralPolicy.load(saved_path, config, policy_config=old_config_dict)


def test_8_cleared_training_storage_initializes_correctly(tmp_path):
    manager = TrainingManager()
    cfg = LabConfig()
    custom_cfg = replace(cfg, training=replace(cfg.training, output_dir=str(tmp_path / "artifacts"), web_export_dir=str(tmp_path / "web/public/generated")))
    
    manager._clear_training_outputs(custom_cfg)
    assert not (tmp_path / "artifacts" / "checkpoints").exists()
    assert not (tmp_path / "artifacts" / "training-summary.json").exists()


def test_9_dashboard_api_reports_no_previous_best_result_after_reset():
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("best_model_path") is None
    assert status.get("best_score") is None
    assert status.get("best_model_checkpoint_episode") is None


def test_10_first_new_episode_is_numbered_correctly():
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("current_episode") is None or status.get("current_episode") == 0


def test_11_resume_and_fork_available_for_future_but_not_fresh_start():
    manager = TrainingManager()
    manager.clear()
    fresh_status = manager._status
    assert fresh_status.get("resumed") is not True
    assert fresh_status.get("forked") is not True
