"""Unit tests for timeout vs true-termination terminal observations in SheepdogRLAdapter."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_config(
    *,
    dogs: int = 1,
    sheep: int = 1,
    curriculum_stage: int = 1,
    max_steps: int = 3,
    no_progress_window: int = 100,
    observation_mode: str = "guided",
    strides: tuple[int, ...] = (0,),
) -> LabConfig:
    payload = LabConfig().to_dict()
    payload["environment"]["dogs"] = dogs
    payload["environment"]["sheep"] = sheep
    payload["environment"]["curriculum_stage"] = curriculum_stage
    payload["environment"]["max_steps"] = max_steps
    payload["environment"]["no_progress_window"] = no_progress_window
    payload["training"]["observation_mode"] = observation_mode
    payload["training"]["observation_history_strides"] = list(strides)
    return LabConfig.from_dict(payload)


def _first_legal_action(mask: np.ndarray) -> int:
    """Pick the first legal non-wait action if available, else any legal action."""
    wait_idx = ACTION_ORDER.index("wait")
    legal_indices = [int(i) for i, allowed in enumerate(mask) if allowed]
    for idx in legal_indices:
        if idx != wait_idx:
            return idx
    return legal_indices[0]


def test_timeout_returns_real_observation_direct_and_dummy_vec_env() -> None:
    """Timeout (truncated=True, terminated=False) emits real s_T observation for bootstrapping."""
    cfg = _make_config(dogs=1, sheep=1, max_steps=3, no_progress_window=100)

    # 1. Direct SheepdogRLAdapter check
    direct_env = SheepdogRLAdapter(cfg, fixed_seed_sequence=[42, 99])
    obs, _ = direct_env.reset()
    actions_taken: list[int] = []
    terminated = False
    truncated = False
    for _ in range(cfg.environment.max_steps):
        action = _first_legal_action(direct_env.action_masks())
        actions_taken.append(action)
        obs, _, terminated, truncated, _ = direct_env.step(action)

    assert terminated is False
    assert truncated is True
    assert direct_env._environment.step_count == 3
    assert np.all(np.isfinite(obs))
    assert np.any(obs != 0.0)
    expected_final_obs = np.array(
        direct_env._environment.build_observation_for_dog(0).values,
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs, expected_final_obs)

    # 2. SB3 DummyVecEnv check (captures terminal_observation before auto-reset)
    vec_env = DummyVecEnv([lambda: SheepdogRLAdapter(cfg, fixed_seed_sequence=[42, 99])])
    post_reset_obs = vec_env.reset()
    dones = np.array([False])
    infos: list[dict] = [{}]
    for action in actions_taken:
        post_reset_obs, _, dones, infos = vec_env.step([action])

    assert bool(dones[0]) is True
    assert infos[0].get("TimeLimit.truncated") is True
    terminal_obs = infos[0]["terminal_observation"]
    assert np.all(np.isfinite(terminal_obs))
    assert np.any(terminal_obs != 0.0)
    np.testing.assert_allclose(terminal_obs, expected_final_obs)
    assert not np.allclose(terminal_obs, post_reset_obs[0])


def test_timeout_with_emergent_gate_strided_history() -> None:
    """Timeout at step_count==max_steps works cleanly with emergent_gate strided history."""
    cfg = _make_config(
        dogs=3,
        sheep=6,
        curriculum_stage=34,
        max_steps=5,
        no_progress_window=100,
        observation_mode="emergent_gate",
        strides=(0, 4, 8, 16),
    )

    direct_env = SheepdogRLAdapter(cfg, fixed_seed_sequence=[53, 77])
    direct_env.reset()
    actions_taken: list[int] = []
    terminated = False
    truncated = False
    direct_final_obs = np.zeros(direct_env.observation_space.shape, dtype=np.float32)
    for _ in range(cfg.environment.max_steps * cfg.environment.dogs):
        action = _first_legal_action(direct_env.action_masks())
        actions_taken.append(action)
        direct_final_obs, _, terminated, truncated, _ = direct_env.step(action)

    assert terminated is False
    assert truncated is True
    assert direct_env._environment.step_count == 5
    assert np.all(np.isfinite(direct_final_obs))
    assert np.any(direct_final_obs != 0.0)

    vec_env = DummyVecEnv([lambda: SheepdogRLAdapter(cfg, fixed_seed_sequence=[53, 77])])
    post_reset_obs = vec_env.reset()
    dones = np.array([False])
    infos: list[dict] = [{}]
    for action in actions_taken:
        post_reset_obs, _, dones, infos = vec_env.step([action])

    assert bool(dones[0]) is True
    assert infos[0].get("TimeLimit.truncated") is True
    terminal_obs = infos[0]["terminal_observation"]
    assert terminal_obs.shape == direct_env.observation_space.shape
    assert np.all(np.isfinite(terminal_obs))
    assert np.any(terminal_obs != 0.0)
    np.testing.assert_allclose(terminal_obs, direct_final_obs)
    assert not np.allclose(terminal_obs, post_reset_obs[0])


def test_true_termination_early_stop_returns_zeros() -> None:
    """True termination (terminated=True, e.g. no_progress_window) still returns zeros."""
    cfg = _make_config(dogs=1, sheep=1, max_steps=20, no_progress_window=2)
    wait_action = ACTION_ORDER.index("wait")

    # 1. Direct SheepdogRLAdapter check
    direct_env = SheepdogRLAdapter(cfg, fixed_seed_sequence=[42, 99])
    direct_env.reset()
    obs = np.ones(direct_env.observation_space.shape, dtype=np.float32)
    terminated = False
    truncated = False
    for _ in range(cfg.environment.max_steps):
        obs, _, terminated, truncated, _ = direct_env.step(wait_action)
        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    np.testing.assert_array_equal(
        obs, np.zeros(direct_env.observation_space.shape, dtype=np.float32)
    )

    # 2. DummyVecEnv check
    vec_env = DummyVecEnv([lambda: SheepdogRLAdapter(cfg, fixed_seed_sequence=[42, 99])])
    vec_env.reset()
    dones = np.array([False])
    infos: list[dict] = [{}]
    for _ in range(cfg.environment.max_steps):
        _, _, dones, infos = vec_env.step([wait_action])
        if bool(dones[0]):
            break

    assert bool(dones[0]) is True
    assert not infos[0].get("TimeLimit.truncated", False)
    np.testing.assert_array_equal(
        infos[0]["terminal_observation"],
        np.zeros(direct_env.observation_space.shape, dtype=np.float32),
    )
