"""Unit tests for main-process PyTorch thread cap configuration and enforcement."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.policies.neural import NeuralPolicy, _make_rl_adapter
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


class _StopAfterThreadSetError(RuntimeError):
    """Sentinel raised to short-circuit heavy setup after torch.set_num_threads."""


def test_training_config_torch_num_threads_default_and_roundtrip() -> None:
    default_cfg = TrainingConfig()
    assert default_cfg.torch_num_threads == 4

    lab = LabConfig(training=TrainingConfig(torch_num_threads=2))
    payload = lab.to_dict()
    assert payload["training"]["torch_num_threads"] == 2

    restored = LabConfig.from_dict(payload)
    assert restored.training.torch_num_threads == 2

    # Backward compatibility: omitted key defaults to 4.
    legacy_payload = lab.to_dict()
    del legacy_payload["training"]["torch_num_threads"]
    restored_legacy = LabConfig.from_dict(legacy_payload)
    assert restored_legacy.training.torch_num_threads == 4


@pytest.mark.parametrize("bad_value", [0, -1, -8])
def test_training_config_torch_num_threads_rejects_non_positive(bad_value: int) -> None:
    with pytest.raises(ValueError, match="torch_num_threads"):
        TrainingConfig(torch_num_threads=bad_value)

    with pytest.raises(ValueError, match="torch_num_threads"):
        LabConfig.from_dict({"training": {"torch_num_threads": bad_value}})


def test_neural_policy_initialize_sets_torch_num_threads() -> None:
    config = LabConfig(training=TrainingConfig(torch_num_threads=3, ppo_env_workers=1))
    with (
        patch("sheepdog.policies.neural.torch.set_num_threads") as mock_set_threads,
        patch(
            "sheepdog.policies.neural._build_vec_env",
            side_effect=_StopAfterThreadSetError("stop"),
        ),
        pytest.raises(_StopAfterThreadSetError),
    ):
        NeuralPolicy.initialize(config)

    mock_set_threads.assert_called_once_with(3)


def test_neural_policy_load_sets_torch_num_threads(tmp_path: Path) -> None:
    config = LabConfig(training=TrainingConfig(torch_num_threads=2, ppo_env_workers=1))
    with (
        patch("sheepdog.policies.neural.torch.set_num_threads") as mock_set_threads,
        patch(
            "sheepdog.policies.neural._build_vec_env",
            side_effect=_StopAfterThreadSetError("stop"),
        ),
        pytest.raises(_StopAfterThreadSetError),
    ):
        NeuralPolicy.load(tmp_path / "model.zip", config)

    mock_set_threads.assert_called_once_with(2)


def test_maskable_ppo_trainer_train_sets_torch_num_threads(tmp_path: Path) -> None:
    config = LabConfig(
        training=TrainingConfig(
            trainer_type="maskable_ppo",
            policy_type="neural",
            torch_num_threads=5,
            ppo_env_workers=1,
            output_dir=str(tmp_path / "out"),
            web_export_dir=str(tmp_path / "web"),
        )
    )
    trainer = MaskablePPOTrainer(config=config, output_root=tmp_path / "out")

    with (
        patch("sheepdog.training.maskable_ppo.torch.set_num_threads") as mock_set_threads,
        patch.object(
            Path,
            "mkdir",
            side_effect=_StopAfterThreadSetError("stop"),
        ),
        pytest.raises(_StopAfterThreadSetError),
    ):
        trainer.train()

    mock_set_threads.assert_called_once_with(5)


def test_make_rl_adapter_keeps_single_thread_for_workers() -> None:
    config_payload = LabConfig(training=TrainingConfig(torch_num_threads=6)).to_dict()
    with (
        patch("sheepdog.policies.neural.torch.set_num_threads") as mock_set_threads,
        patch("sheepdog.policies.neural.SheepdogRLAdapter") as mock_adapter_cls,
    ):
        _make_rl_adapter(config_payload, worker_index=0, workers=4)

    mock_set_threads.assert_called_once_with(1)
    mock_adapter_cls.assert_called_once()
