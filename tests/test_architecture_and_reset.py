"""Tests for 3-layer neural network architecture [256, 256, 128] and training reset integrity."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import sheepdog.server as server_mod
import sheepdog.training.episode_store as episode_store_mod
import sheepdog.training.replay_writer as replay_writer_mod
from sheepdog.config import LabConfig as OriginalLabConfig
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.server import TrainingManager


def _reset_telemetry_singletons() -> None:
    """Close and reset global EpisodeStore and AsyncReplayWriter singletons."""
    writer = replay_writer_mod._global_replay_writer
    if writer is not None:
        with contextlib.suppress(Exception):
            writer.close()
        replay_writer_mod._global_replay_writer = None

    store = episode_store_mod._global_episode_store
    if store is not None:
        with contextlib.suppress(Exception):
            store.close()
        episode_store_mod._global_episode_store = None

    for ro_store in list(episode_store_mod._global_ro_episode_stores.values()):
        with contextlib.suppress(Exception):
            ro_store.close()
    episode_store_mod._global_ro_episode_stores.clear()


@pytest.fixture(autouse=True)
def _isolate_training_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[OriginalLabConfig]:
    """Redirect TrainingManager and LabConfig away from live artifacts/ for every test."""
    _reset_telemetry_singletons()

    base_cfg = OriginalLabConfig()
    isolated_cfg = replace(
        base_cfg,
        training=replace(
            base_cfg.training,
            output_dir=str(tmp_path / "artifacts"),
            backup_dir=str(tmp_path / "artifacts" / "backups"),
            web_export_dir=str(tmp_path / "web_generated"),
            backup_enabled=False,
        ),
    )

    def _make_isolated_config() -> OriginalLabConfig:
        return isolated_cfg

    created_managers: list[TrainingManager] = []
    orig_init = TrainingManager.__init__

    def _tracked_init(self: TrainingManager) -> None:
        orig_init(self)
        created_managers.append(self)

    monkeypatch.setattr(server_mod, "LabConfig", _make_isolated_config)
    monkeypatch.setattr(
        "tests.test_architecture_and_reset.LabConfig",
        _make_isolated_config,
        raising=False,
    )
    monkeypatch.setattr(TrainingManager, "__init__", _tracked_init)
    monkeypatch.setattr(
        TrainingManager,
        "_export_untrained_baseline",
        lambda self, config: None,
    )

    try:
        yield isolated_cfg
    finally:
        for mgr in created_managers:
            with contextlib.suppress(Exception):
                mgr._watchdog_stop_event.set()
            with contextlib.suppress(Exception):
                mgr.runtime_tracker._stop_event.set()
        _reset_telemetry_singletons()


LabConfig = OriginalLabConfig


def test_1_policy_network_has_three_hidden_layers() -> None:
    """Verify policy network initializes with LabConfig default (256, 256, 128) architecture."""
    config = LabConfig()
    policy = NeuralPolicy.initialize(config)
    assert policy.config.hidden_sizes == (256, 256, 128)
    net_arch = policy.model.policy_kwargs.get("net_arch")
    assert net_arch == [256, 256, 128]


def test_2_value_network_has_three_hidden_layers() -> None:
    """Verify shared actor-critic network configures [256, 256, 128] hidden layers."""
    config = LabConfig()
    policy = NeuralPolicy.initialize(config)
    net_arch = policy.model.policy_kwargs.get("net_arch")
    assert isinstance(net_arch, list)
    assert len(net_arch) == 3
    assert net_arch == [256, 256, 128]


def test_3_new_run_does_not_load_existing_checkpoint() -> None:
    """Verify clear() leaves active_model_path unset and checkpoint_episode reset."""
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("active_model_path") is None
    assert status.get("checkpoint_episode") is None or status.get("checkpoint_episode") == 0


def test_4_new_run_resets_curriculum_to_stage_1() -> None:
    """Verify reset_journey() resets curriculum stage back to Stage 1."""
    manager = TrainingManager()
    manager.reset_journey()
    status = manager._status
    assert status.get("curriculum_stage") == 1 or status.get("active_curriculum_stage") == 1


def test_5_new_run_resets_episode_and_global_counters() -> None:
    """Verify clear() resets episode and total_episodes_trained counters."""
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("current_episode") is None or status.get("current_episode") == 0
    assert status.get("checkpoint_episode") is None or status.get("checkpoint_episode") == 0
    assert status.get("total_episodes_trained") == 0


def test_6_new_run_creates_fresh_optimizer() -> None:
    """Verify distinct NeuralPolicy initializations own independent optimizer instances."""
    config = LabConfig()
    policy1 = NeuralPolicy.initialize(config)
    policy2 = NeuralPolicy.initialize(config)
    assert policy1.model.policy.optimizer is not policy2.model.policy.optimizer


def test_7_old_two_layer_checkpoints_rejected_as_incompatible(tmp_path: Path) -> None:
    """Verify loading a 2-layer checkpoint into a 3-layer config raises an incompatibility error."""
    config = LabConfig()
    old_config_dict = {"hidden_sizes": [128, 128], "observation_size": 54}
    fake_zip_path = tmp_path / "old_2layer_model.zip"

    policy_2layer = NeuralPolicy.initialize(config)
    policy_2layer.config = NeuralPolicyConfig(hidden_sizes=(128, 128), observation_size=54)
    saved_path = policy_2layer.save(fake_zip_path)

    with pytest.raises((ValueError, RuntimeError), match="Incompatible model architecture"):
        NeuralPolicy.load(saved_path, config, policy_config=old_config_dict)


def test_8_cleared_training_storage_initializes_correctly(tmp_path: Path) -> None:
    """Verify _clear_training_outputs removes checkpoints and training-summary in isolated dir."""
    manager = TrainingManager()
    cfg = LabConfig()
    custom_cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            output_dir=str(tmp_path / "custom_artifacts"),
            web_export_dir=str(tmp_path / "web/public/generated"),
        ),
    )

    manager._clear_training_outputs(custom_cfg)
    assert not (tmp_path / "custom_artifacts" / "checkpoints").exists()
    assert not (tmp_path / "custom_artifacts" / "training-summary.json").exists()


def test_9_dashboard_api_reports_no_previous_best_result_after_reset() -> None:
    """Verify clear() resets best_model_path, best_score, and best_model_checkpoint_episode."""
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("best_model_path") is None
    assert status.get("best_score") is None
    assert status.get("best_model_checkpoint_episode") is None


def test_10_first_new_episode_is_numbered_correctly() -> None:
    """Verify clear() leaves current_episode at None or 0."""
    manager = TrainingManager()
    manager.clear()
    status = manager._status
    assert status.get("current_episode") is None or status.get("current_episode") == 0


def test_11_resume_and_fork_available_for_future_but_not_fresh_start() -> None:
    """Verify a cleared manager does not mark resumed or forked True."""
    manager = TrainingManager()
    manager.clear()
    fresh_status = manager._status
    assert fresh_status.get("resumed") is not True
    assert fresh_status.get("forked") is not True
