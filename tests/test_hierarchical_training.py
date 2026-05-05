"""Tests for HierarchicalMaskablePPOTrainer and ShepherdNeuralDogPolicy.

SB3 / sb3_contrib are optional.  All tests in this module are skipped when
those packages are not installed.
"""

from __future__ import annotations

import pytest

sb3_contrib = pytest.importorskip("sb3_contrib")


from sheepdog.config import LabConfig
from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy
from sheepdog.training.joint_rl_env import JointActionRLEnv


# ---------------------------------------------------------------------------
# JointActionRLEnv as a valid Gymnasium env
# ---------------------------------------------------------------------------


def test_joint_env_reset_returns_array():
    import numpy as np  # noqa: PLC0415

    env = JointActionRLEnv(LabConfig())
    obs, info = env.reset(seed=0)
    assert hasattr(obs, "shape")
    assert obs.dtype.name == "float32"


def test_joint_env_step_runs():
    env = JointActionRLEnv(LabConfig())
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(8)  # wait
    assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# ShepherdNeuralDogPolicy.initialize()
# ---------------------------------------------------------------------------


def test_initialize_returns_policy():
    policy = ShepherdNeuralDogPolicy.initialize(LabConfig())
    assert isinstance(policy, ShepherdNeuralDogPolicy)


def test_select_actions_returns_correct_count():
    from sheepdog.environment import SheepdogEnvironment  # noqa: PLC0415

    config = LabConfig()
    policy = ShepherdNeuralDogPolicy.initialize(config)
    env = SheepdogEnvironment(config)
    env.reset(seed=0)
    actions = policy.select_actions(env)
    assert len(actions) == len(env.dogs)


def test_select_actions_returns_valid_action_type():
    from sheepdog.environment import SheepdogEnvironment  # noqa: PLC0415
    from sheepdog.environment import ACTION_ORDER  # noqa: PLC0415

    config = LabConfig()
    policy = ShepherdNeuralDogPolicy.initialize(config)
    env = SheepdogEnvironment(config)
    env.reset(seed=0)
    actions = policy.select_actions(env)
    for action in actions:
        assert action in ACTION_ORDER


def test_policy_name():
    assert ShepherdNeuralDogPolicy.name == "shepherd_neural_dogs"


def test_policy_trainer_type():
    assert ShepherdNeuralDogPolicy.trainer_type == "hierarchical_maskable_ppo"


# ---------------------------------------------------------------------------
# Save / load round trip (tmp dir)
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    from sheepdog.environment import SheepdogEnvironment  # noqa: PLC0415

    config = LabConfig()
    policy = ShepherdNeuralDogPolicy.initialize(config)
    save_path = policy.save(tmp_path / "model.zip")
    loaded = ShepherdNeuralDogPolicy.load(save_path, config)
    assert isinstance(loaded, ShepherdNeuralDogPolicy)

    env = SheepdogEnvironment(config)
    env.reset(seed=1)
    actions = loaded.select_actions(env)
    assert len(actions) == len(env.dogs)


# ---------------------------------------------------------------------------
# HierarchicalMaskablePPOTrainer: smoke test 10 timesteps
# ---------------------------------------------------------------------------


def test_trainer_runs_minimal_steps(tmp_path):
    from dataclasses import replace  # noqa: PLC0415

    from sheepdog.training.hierarchical_trainer import HierarchicalMaskablePPOTrainer  # noqa: PLC0415

    config = LabConfig(
        training=replace(
            LabConfig().training,
            total_timesteps=50,
            output_dir=str(tmp_path),
        )
    )
    trainer = HierarchicalMaskablePPOTrainer(config, str(tmp_path))
    result = trainer.train()
    assert isinstance(result, dict)
