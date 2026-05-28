"""Tests for HierarchicalMaskablePPOTrainer and ShepherdNeuralDogPolicy.

SB3 / sb3_contrib are optional.  All tests in this module are skipped when
those packages are not installed.
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel,wrong-import-position
from __future__ import annotations

import pytest

pytest.importorskip("sb3_contrib")


from sheepdog.config import LabConfig  # noqa: E402
from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy  # noqa: E402
from sheepdog.training.joint_rl_env import JointActionRLEnv  # noqa: E402

# ---------------------------------------------------------------------------
# JointActionRLEnv as a valid Gymnasium env
# ---------------------------------------------------------------------------


def test_joint_env_reset_returns_array():
    env = JointActionRLEnv(LabConfig())
    obs, _info = env.reset(seed=0)
    assert hasattr(obs, "shape")
    assert obs.dtype.name == "float32"


def test_joint_env_step_runs():
    env = JointActionRLEnv(LabConfig())
    env.reset(seed=0)
    _obs, reward, _terminated, _truncated, _info = env.step(8)  # wait
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
    from sheepdog.environment import (
        ACTION_ORDER,  # noqa: PLC0415
        SheepdogEnvironment,  # noqa: PLC0415
    )

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

    from sheepdog.training.hierarchical_trainer import (
        HierarchicalMaskablePPOTrainer,  # noqa: PLC0415
    )

    config = LabConfig(
        training=replace(
            LabConfig().training,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            total_timesteps=10,
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web"),
        )
    )
    trainer = HierarchicalMaskablePPOTrainer(config, str(tmp_path))
    result = trainer.train()
    assert isinstance(result, dict)
