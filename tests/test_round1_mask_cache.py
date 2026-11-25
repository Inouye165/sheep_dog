"""Unit tests for the per-(world_step, dog_index) action-mask cache in SheepdogRLAdapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_config(
    *,
    dogs: int = 3,
    sheep: int = 6,
    max_steps: int = 100,
    no_progress_window: int = 100,
) -> LabConfig:
    payload = LabConfig().to_dict()
    payload["environment"]["dogs"] = dogs
    payload["environment"]["sheep"] = sheep
    payload["environment"]["max_steps"] = max_steps
    payload["environment"]["no_progress_window"] = no_progress_window
    return LabConfig.from_dict(payload)


def test_mask_cache_matches_fresh_and_calls_at_most_once_per_dog_step() -> None:
    """action_masks() matches fresh action_mask_for_dog and computes at most once per (step, dog)."""
    cfg = _make_config()
    seeds = [7, 13, 29, 41, 59]

    for seed in seeds:
        adapter = SheepdogRLAdapter(cfg, fixed_seed_sequence=[seed])
        obs, reset_info = adapter.reset()
        assert obs.shape == adapter.observation_space.shape
        assert "action_mask" not in reset_info

        orig_fn = adapter._environment.action_mask_for_dog
        call_counts: Counter[tuple[int, int]] = Counter()

        def tracked_action_mask_for_dog(
            dog_index: int,
            *,
            reserved_positions: set | None = None,
            policy_mode: str | None = None,
            _orig=orig_fn,
            _counts=call_counts,
            _adapter=adapter,
        ):
            key = (int(_adapter._environment.step_count), int(dog_index))
            _counts[key] += 1
            return _orig(
                dog_index,
                reserved_positions=reserved_positions,
                policy_mode=policy_mode,
            )

        adapter._environment.action_mask_for_dog = tracked_action_mask_for_dog  # type: ignore[method-assign]

        rng = np.random.default_rng(seed)
        for _ in range(60):
            key = (int(adapter._environment.step_count), int(adapter._current_dog_index))
            mask1 = adapter.action_masks()
            mask2 = adapter.action_masks()
            assert mask1 is mask2
            assert mask1.dtype == np.bool_
            assert not mask1.flags.writeable

            reserved = {
                adapter._environment.dogs[i].position
                for i in range(adapter._current_dog_index)
            }
            fresh_map = orig_fn(
                adapter._current_dog_index,
                reserved_positions=reserved,
            )
            fresh_arr = np.array(
                [fresh_map.get(action, False) for action in ACTION_ORDER],
                dtype=bool,
            )
            assert np.array_equal(mask1, fresh_arr)

            legal = np.flatnonzero(mask1)
            action = int(rng.choice(legal))
            _, _, terminated, truncated, info = adapter.step(action)
            assert "action_mask" not in info
            assert call_counts[key] == 1

            if terminated or truncated:
                break


def test_mask_cache_bit_identical_rollout_vs_uncached() -> None:
    """Cached rollout produces bit-identical observations, masks, rewards, and flags."""
    cfg = _make_config()
    rng = np.random.default_rng(1234)
    actions = [int(rng.integers(0, len(ACTION_ORDER))) for _ in range(60)]

    cached_env = SheepdogRLAdapter(cfg, fixed_seed_sequence=[42])
    uncached_env = SheepdogRLAdapter(cfg, fixed_seed_sequence=[42])

    obs_c, info_c = cached_env.reset()
    obs_u, info_u = uncached_env.reset()
    assert "action_mask" not in info_c
    assert "action_mask" not in info_u
    assert np.array_equal(obs_c, obs_u)

    for action in actions:
        mask_c = cached_env.action_masks()
        uncached_env._invalidate_action_mask_cache()
        mask_u = uncached_env.action_masks()
        assert np.array_equal(mask_c, mask_u)

        obs_c, rew_c, term_c, trunc_c, info_c = cached_env.step(action)
        uncached_env._invalidate_action_mask_cache()
        obs_u, rew_u, term_u, trunc_u, info_u = uncached_env.step(action)

        assert np.array_equal(obs_c, obs_u)
        assert rew_c == rew_u
        assert term_c == term_u
        assert trunc_c == trunc_u
        assert "action_mask" not in info_c
        assert "action_mask" not in info_u

        if term_c or trunc_c:
            break


def test_mask_cache_invalidates_on_reset_and_update_env_config() -> None:
    """Cache clears on update_env_config() and reset()."""
    cfg = _make_config()
    adapter = SheepdogRLAdapter(cfg, fixed_seed_sequence=[11, 22])
    adapter.reset()

    mask_before = adapter.action_masks()
    assert adapter._cached_action_mask_key == (0, 0)
    assert adapter._cached_action_mask is mask_before

    new_env_cfg = replace(cfg.environment, max_steps=120)
    adapter.update_env_config(new_env_cfg)
    assert adapter._cached_action_mask_key is None
    assert adapter._cached_action_mask is None

    # Populate cache again, advance one dog, then reset to a new seed.
    mask_step0 = adapter.action_masks()
    legal = np.flatnonzero(mask_step0)
    adapter.step(int(legal[0]))
    _ = adapter.action_masks()
    assert adapter._cached_action_mask is not None

    _, reset_info = adapter.reset(seed=99)
    assert "action_mask" not in reset_info
    assert adapter._cached_action_mask_key is None
    assert adapter._cached_action_mask is None

    mask_after_reset = adapter.action_masks()
    fresh_map = adapter._environment.action_mask_for_dog(0, reserved_positions=set())
    fresh_arr = np.array(
        [fresh_map.get(action, False) for action in ACTION_ORDER],
        dtype=bool,
    )
    assert np.array_equal(mask_after_reset, fresh_arr)
