"""Regression tests for true team-step PPO transitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.observations import HERD_DOG_SLOTS
from sheepdog.policies.factory import create_policy_from_name, load_playable_policy
from sheepdog.policies.joint_team import JointTeamPolicy
from sheepdog.training.factory import create_trainer
from sheepdog.training.joint_team_ppo import JointTeamPPOTrainer
from sheepdog.training.team_rl_env import TeamActionRLEnv


def _config(tmp_path: Path, *, dogs: int = 3) -> LabConfig:
    """Build a small joint-action test configuration."""
    return LabConfig(
        environment=EnvironmentConfig(dogs=dogs, sheep=2, max_steps=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            trainer_type="joint_maskable_ppo",
            policy_type="neural",
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            quick_evaluation_seed_count=1,
            rollout_steps=8,
            batch_size=4,
            total_timesteps=8,
            ppo_env_workers=1,
            output_dir=str(tmp_path / "artifacts"),
            web_export_dir=str(tmp_path / "web"),
        ),
    )


def test_team_adapter_has_fixed_centralized_spaces(tmp_path: Path) -> None:
    """Observation and action shapes remain stable across dog counts."""
    one_dog = TeamActionRLEnv(_config(tmp_path, dogs=1))
    three_dogs = TeamActionRLEnv(_config(tmp_path, dogs=3))

    one_observation, _ = one_dog.reset(seed=7)
    three_observation, _ = three_dogs.reset(seed=7)

    assert one_observation.shape == three_observation.shape
    assert one_observation.shape == one_dog.observation_space.shape
    assert tuple(one_dog.action_space.nvec) == (9,) * HERD_DOG_SLOTS
    assert np.count_nonzero(one_observation[one_observation.size // HERD_DOG_SLOTS :]) == 0


def test_team_adapter_advances_world_once_and_returns_immediate_reward(
    tmp_path: Path,
) -> None:
    """One external action produces one complete world transition."""
    adapter = TeamActionRLEnv(_config(tmp_path))
    adapter.reset(seed=11)
    wait = ACTION_ORDER.index("wait")

    _observation, reward, terminated, truncated, info = adapter.step(
        np.full(HERD_DOG_SLOTS, wait, dtype=np.int64)
    )

    assert adapter._environment.step_count == 1  # pylint: disable=protected-access
    assert reward == adapter._episode_reward  # pylint: disable=protected-access
    assert reward != 0.0
    assert not terminated
    assert not truncated
    assert info["team_step_completed"] is True
    assert info["world_step_count"] == 1


def test_team_adapter_returns_flat_masks_with_inactive_slots(tmp_path: Path) -> None:
    """Each MultiDiscrete head receives one legal-action mask."""
    adapter = TeamActionRLEnv(_config(tmp_path, dogs=1))
    adapter.reset(seed=13)
    mask = adapter.action_masks().reshape(HERD_DOG_SLOTS, len(ACTION_ORDER))

    assert mask.shape == (HERD_DOG_SLOTS, len(ACTION_ORDER))
    for inactive_mask in mask[1:]:
        assert np.flatnonzero(inactive_mask).tolist() == [ACTION_ORDER.index("wait")]


def test_team_adapter_rejects_malformed_actions(tmp_path: Path) -> None:
    """Malformed joint actions fail before mutating the environment."""
    adapter = TeamActionRLEnv(_config(tmp_path))
    adapter.reset(seed=17)

    with pytest.raises(ValueError, match="Expected 3 team actions"):
        adapter.step(np.asarray([0, 1], dtype=np.int64))

    assert adapter._environment.step_count == 0  # pylint: disable=protected-access


def test_joint_policy_initializes_and_selects_all_active_actions(tmp_path: Path) -> None:
    """The centralized model returns one legal action per active dog."""
    config = _config(tmp_path, dogs=2)
    policy = JointTeamPolicy.initialize(config)
    environment = SheepdogEnvironment(config)
    environment.reset(seed=19)

    actions = policy.select_actions(environment)

    assert len(actions) == 2
    assert set(actions).issubset(set(ACTION_ORDER))
    assert tuple(policy.model.action_space.nvec) == (9,) * HERD_DOG_SLOTS


def test_joint_policy_save_load_and_factory_round_trip(tmp_path: Path) -> None:
    """Joint checkpoints retain distinct architecture metadata and remain playable."""
    config = _config(tmp_path, dogs=2)
    policy = create_policy_from_name("joint_team_policy", config=config)
    assert isinstance(policy, JointTeamPolicy)
    model_path = policy.save(tmp_path / "joint-model")

    loaded = JointTeamPolicy.load(model_path, config, policy.config.to_dict())
    environment = SheepdogEnvironment(config)
    environment.reset(seed=23)

    assert len(loaded.select_actions(environment)) == 2


def test_joint_trainer_factory_runs_tiny_update_with_isolated_artifacts(
    tmp_path: Path,
) -> None:
    """A short rollout saves state and checkpoints outside legacy PPO paths."""
    config = _config(tmp_path)
    trainer = create_trainer(config, config.training.output_dir)

    summary = trainer.train()

    output_root = Path(config.training.output_dir)
    assert isinstance(trainer, JointTeamPPOTrainer)
    assert summary.checkpoints
    assert Path(summary.final_model_path).is_relative_to(output_root / "models" / "joint_team")
    assert (output_root / "joint-team-training-state.json").exists()
    assert (output_root / "joint-team-training-summary.json").exists()
    assert (output_root / "checkpoints" / "joint_team" / "checkpoint-000000.json").exists()
    assert not (output_root / "training-state.json").exists()

    loaded = load_playable_policy(config, policy_mode="joint_team_policy")
    assert isinstance(loaded, JointTeamPolicy)
