"""Tests for multi-worker SubprocVecEnv seeding, replay safety, and fallbacks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig, _build_vec_env
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


def _make_config(tmp_path: Path, *, ppo_env_workers: int = 4) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2),
        rewards=RewardConfig(),
        training=TrainingConfig(
            trainer_type="maskable_ppo",
            policy_type="neural",
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            candidate_evaluation_seeds=(91,),
            rollout_steps=16,
            batch_size=16,
            total_timesteps=128,
            train_seed=7,
            ppo_env_workers=ppo_env_workers,
            backup_enabled=False,
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web"),
        ),
    )


def test_subproc_vec_env_unique_seeds_and_learn(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SHEEPDOG_PPO_NUM_ENVS", raising=False)
    config = _make_config(tmp_path, ppo_env_workers=4)

    vec_env, workers, backend = _build_vec_env(config)
    try:
        assert workers == 4
        assert backend == "subproc"

        worker_seeds = vec_env.get_attr("_latest_seed")
        assert worker_seeds == [7, 100_007, 200_007, 300_007]
        assert len(set(worker_seeds)) == 4

        vec_env.reset()
        for _ in range(64):
            actions = np.zeros(workers, dtype=np.int64)
            vec_env.step(actions)

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=16,
            batch_size=16,
            seed=config.training.train_seed,
            verbose=0,
        )
        model.learn(total_timesteps=128)
    finally:
        vec_env.close()


def test_env_override_forces_dummy_vec_env(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SHEEPDOG_PPO_NUM_ENVS", "1")
    config = _make_config(tmp_path, ppo_env_workers=4)

    vec_env, workers, backend = _build_vec_env(config)
    try:
        assert workers == 1
        assert backend == "dummy"
        assert vec_env.num_envs == 1
    finally:
        vec_env.close()


def test_subproc_oserror_falls_back_to_dummy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SHEEPDOG_PPO_NUM_ENVS", raising=False)
    config = _make_config(tmp_path, ppo_env_workers=4)

    with patch(
        "sheepdog.policies.neural.SubprocVecEnv",
        side_effect=OSError("spawn failed"),
    ):
        vec_env, workers, backend = _build_vec_env(config)

    try:
        assert workers == 1
        assert backend == "dummy"
        assert vec_env.num_envs == 1
    finally:
        vec_env.close()


def test_subproc_memoryerror_falls_back_to_dummy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SHEEPDOG_PPO_NUM_ENVS", raising=False)
    config = _make_config(tmp_path, ppo_env_workers=4)

    with patch(
        "sheepdog.policies.neural.SubprocVecEnv",
        side_effect=MemoryError("virtual memory exhausted"),
    ):
        vec_env, workers, backend = _build_vec_env(config)

    try:
        assert workers == 1
        assert backend == "dummy"
        assert vec_env.num_envs == 1
    finally:
        vec_env.close()


def test_neural_policy_close_closes_vec_env_once(tmp_path: Path) -> None:
    mock_env = MagicMock()
    mock_model = SimpleNamespace(env=mock_env, gamma=0.99)
    policy = NeuralPolicy(
        model=mock_model,  # type: ignore[arg-type]
        model_path=tmp_path / "model.zip",
        config=NeuralPolicyConfig(hidden_sizes=(32, 32), observation_size=10),
    )

    policy.close()
    policy.close()

    mock_env.close.assert_called_once()


def test_neural_policy_load_closes_vec_env_on_validation_error(tmp_path: Path) -> None:
    mock_vec_env = MagicMock()
    mock_vec_env.observation_space.shape = (10,)
    mock_model = SimpleNamespace(observation_space=SimpleNamespace(shape=(999,)))
    config = _make_config(tmp_path, ppo_env_workers=1)

    with (
        patch(
            "sheepdog.policies.neural._build_vec_env",
            return_value=(mock_vec_env, 1, "dummy"),
        ),
        patch("sheepdog.policies.neural.MaskablePPO.load", return_value=mock_model),
        pytest.raises(ValueError, match="Incompatible observation size"),
    ):
        NeuralPolicy.load(tmp_path / "missing.zip", config)

    mock_vec_env.close.assert_called_once()


def test_maskable_ppo_trainer_train_closes_policy_and_evaluator_on_early_stop(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, ppo_env_workers=1)
    trainer = MaskablePPOTrainer(config, tmp_path)

    with (
        patch.object(NeuralPolicy, "close", autospec=True, wraps=NeuralPolicy.close) as policy_close,
        patch.object(
            trainer.evaluator,
            "close",
            autospec=True,
            wraps=trainer.evaluator.close,
        ) as evaluator_close,
    ):
        trainer.train(should_stop=lambda: True)

    policy_close.assert_called_once()
    evaluator_close.assert_called_once()
