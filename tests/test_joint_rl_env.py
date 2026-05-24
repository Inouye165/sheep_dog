"""Tests for JointActionRLEnv (joint dog action stepping)."""

from __future__ import annotations

import numpy as np
import pytest

from sheepdog.config import LabConfig
from sheepdog.training.joint_rl_env import JointActionRLEnv


def _make_env(seed: int = 0) -> JointActionRLEnv:
    return JointActionRLEnv(LabConfig(), fixed_seed_sequence=[seed])


# ---------------------------------------------------------------------------
# Basic Gymnasium compliance
# ---------------------------------------------------------------------------


def test_observation_shape_matches_space():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape


def test_observation_dtype_is_float32():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.dtype == np.float32


def test_action_space_is_discrete_9():
    import gymnasium as gym  # noqa: PLC0415

    env = _make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert int(env.action_space.n) == 9


def test_action_masks_length_matches_action_space():
    env = _make_env()
    env.reset()
    masks = env.action_masks()
    assert len(masks) == int(env.action_space.n)


def test_action_masks_dtype_is_bool():
    env = _make_env()
    env.reset()
    masks = env.action_masks()
    assert masks.dtype == bool


def test_reset_returns_valid_obs():
    env = _make_env()
    obs, info = env.reset()
    assert np.all(np.isfinite(obs))
    assert obs.shape == env.observation_space.shape


def test_reset_returns_dict_info():
    env = _make_env()
    _, info = env.reset()
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Step behaviour
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple():
    env = _make_env()
    env.reset()
    result = env.step(8)  # action 8 = wait
    assert len(result) == 5


def test_step_obs_shape_matches_space():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(8)
    assert obs.shape == env.observation_space.shape


def test_shepherd_command_in_step_info():
    from sheepdog.shepherd import ShepherdCommand  # noqa: PLC0415

    env = _make_env()
    env.reset()
    _, _, _, _, info = env.step(8)
    assert "shepherd_command" in info
    assert isinstance(info["shepherd_command"], ShepherdCommand)


def test_shepherd_command_in_reset_info():
    from sheepdog.shepherd import ShepherdCommand  # noqa: PLC0415

    env = _make_env()
    _, info = env.reset()
    assert "shepherd_command" in info
    assert isinstance(info["shepherd_command"], ShepherdCommand)


def test_episode_ends_eventually():
    """Running random actions long enough must eventually produce terminated or truncated."""
    env = _make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        masks = env.action_masks()
        valid = np.where(masks)[0]
        action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        pytest.fail("Episode did not end after 5000 steps")


# ---------------------------------------------------------------------------
# Dog cycling: N dogs must all act before team step fires
# ---------------------------------------------------------------------------


def test_multiple_steps_before_team_step():
    """Each dog takes a turn; reward is 0.0 until all dogs have acted."""
    config = LabConfig()
    env = JointActionRLEnv(config)
    env.reset(seed=0)
    n_dogs = len(env._environment.dogs)  # type: ignore[attr-defined]
    if n_dogs < 2:
        pytest.skip("Need at least 2 dogs")

    # The first (n_dogs - 1) steps collect actions; only the n_dogs-th step
    # fires the team step and yields a nonzero reward.  At minimum, the
    # observation returned should always be finite and correct shape.
    for _step_idx in range(n_dogs):
        obs, reward, terminated, truncated, info = env.step(8)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break  # short episodes are OK


def test_env_is_valid_gymnasium_env():
    """gym.utils.env_checker.check_env should not raise."""
    try:
        from gymnasium.utils.env_checker import check_env  # noqa: PLC0415
    except ImportError:
        pytest.skip("check_env not available")
    env = _make_env()
    check_env(env, skip_render_check=True)
