"""Gymnasium adapter for training a shared dog policy with PPO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.training.scenario_sampler import ScenarioSampler


class SheepdogRLAdapter(gym.Env[np.ndarray, int]):
    """Single-agent sequential wrapper around the multi-dog team environment.

    One RL step selects the action for the current dog. After the final dog acts,
    the underlying team environment advances once and emits the shared team reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: LabConfig,
        fixed_seed_sequence: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.fixed_seed_sequence = tuple(fixed_seed_sequence or ())
        self._fixed_seed_index = 0
        self._episode_counter = 0
        self._environment = SheepdogEnvironment(config)
        initial_snapshot = self._environment.reset(seed=config.training.train_seed)
        del initial_snapshot
        observation_size = len(self._environment.build_observation_for_dog(0).values)
        self.action_space = spaces.Discrete(len(ACTION_ORDER))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self._pending_actions: list[str] = []
        self._current_dog_index = 0
        self._latest_seed = config.training.train_seed
        self._episode_reward = 0.0
        # Scenario sampler for mixing difficult training scenarios
        self._scenario_sampler = ScenarioSampler(
            config.training,
            config.environment,
        )

        # Training Scenario Coverage & Exposure Trackers
        self._last_curriculum_stage = config.rewards.instincts.curriculum_stage
        self._stage_unique_seeds = set()
        self._stage_unique_configs = set()
        self._starting_sheep_to_pen_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
        self._starting_dog_to_sheep_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
        self._similarity_episodes = {11: 0, 23: 0, 37: 0, 41: 0, 53: 0}
        self._similarity_successes = {11: 0, 23: 0, 37: 0, 41: 0, 53: 0}
        self._evaluation_layouts = {}
        self._precompute_evaluation_layouts()
        self._current_episode_similarity_match = None

    def _precompute_evaluation_layouts(self) -> None:
        """Pre-generate initial positions for standard evaluation seeds to check training similarity."""
        try:
            temp_env = SheepdogEnvironment(self.config)
            for seed in [11, 23, 37, 41, 53]:
                temp_env.reset(seed=seed)
                self._evaluation_layouts[seed] = {
                    "dog_positions": [(dog.position.x, dog.position.y) for dog in temp_env.dogs],
                    "sheep_positions": [
                        (sheep.position.x, sheep.position.y) for sheep in temp_env.sheep
                    ],
                    "pen_position": (temp_env.pen.origin.x, temp_env.pen.origin.y),
                }
        except (RuntimeError, ValueError):
            self._evaluation_layouts.clear()

    def get_coverage_stats(self) -> dict[str, Any]:
        """Expose training exposure statistics."""
        return {
            "seeds_seen_list": list(self._stage_unique_seeds),
            "configs_seen_list": list(self._stage_unique_configs),
            "min_sheep_to_pen": float(self._starting_sheep_to_pen_stats["min"]) if self._starting_sheep_to_pen_stats["count"] > 0 else 0.0,
            "max_sheep_to_pen": float(self._starting_sheep_to_pen_stats["max"]) if self._starting_sheep_to_pen_stats["count"] > 0 else 0.0,
            "sum_sheep_to_pen": float(self._starting_sheep_to_pen_stats["sum"]),
            "count_sheep_to_pen": int(self._starting_sheep_to_pen_stats["count"]),
            "min_dog_to_sheep": float(self._starting_dog_to_sheep_stats["min"]) if self._starting_dog_to_sheep_stats["count"] > 0 else 0.0,
            "max_dog_to_sheep": float(self._starting_dog_to_sheep_stats["max"]) if self._starting_dog_to_sheep_stats["count"] > 0 else 0.0,
            "sum_dog_to_sheep": float(self._starting_dog_to_sheep_stats["sum"]),
            "count_dog_to_sheep": int(self._starting_dog_to_sheep_stats["count"]),
            "similarity_episodes": self._similarity_episodes,
            "similarity_successes": self._similarity_successes,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None:
            seed = self._next_seed()
        self._latest_seed = int(seed)
        self._pending_actions = []
        self._current_dog_index = 0
        self._episode_reward = 0.0

        # Use scenario sampler if scenario training is enabled
        if self.config.training.scenario_training_enabled:
            selection = self._scenario_sampler.sample(self._episode_counter)
            if selection.scenario is not None:
                # Use predefined scenario
                self._environment.reset_from_scenario(selection.scenario)
            else:
                # Use normal random reset
                self._environment.reset(seed=self._latest_seed)
        else:
            # Scenario training disabled: use normal random reset
            self._environment.reset(seed=self._latest_seed)

        # Stage checks and trackers
        curr_stage = self.config.rewards.instincts.curriculum_stage
        if curr_stage != self._last_curriculum_stage:
            self._last_curriculum_stage = curr_stage
            self._stage_unique_seeds.clear()
            self._stage_unique_configs.clear()
            self._starting_sheep_to_pen_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
            self._starting_dog_to_sheep_stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
            for ev_s in self._similarity_episodes:
                self._similarity_episodes[ev_s] = 0
                self._similarity_successes[ev_s] = 0

        self._stage_unique_seeds.add(self._latest_seed)
        
        # Hash initial positions configuration
        dogs = self._environment.dogs
        sheep = self._environment.sheep
        pen = self._environment.pen
        dog_positions = tuple(sorted((dog.position.x, dog.position.y) for dog in dogs))
        sheep_positions = tuple(sorted((item.position.x, item.position.y) for item in sheep))
        pen_origin = (pen.origin.x, pen.origin.y)
        config_hash = hash((dog_positions, sheep_positions, pen_origin))
        self._stage_unique_configs.add(config_hash)

        # Distances
        import math
        sheep_dists = [
            math.hypot(item.position.x - pen.origin.x, item.position.y - pen.origin.y)
            for item in sheep
        ]
        avg_sheep_to_pen = float(np.mean(sheep_dists)) if sheep_dists else 0.0
        
        dog_dists = []
        for dog in dogs:
            for item in sheep:
                dog_dists.append(
                    math.hypot(dog.position.x - item.position.x, dog.position.y - item.position.y)
                )
        avg_dog_to_sheep = float(np.mean(dog_dists)) if dog_dists else 0.0

        # Update stats
        def _update_stat_dict(d, val):
            if val < d["min"]: d["min"] = val
            if val > d["max"]: d["max"] = val
            d["sum"] += val
            d["count"] += 1

        _update_stat_dict(self._starting_sheep_to_pen_stats, avg_sheep_to_pen)
        _update_stat_dict(self._starting_dog_to_sheep_stats, avg_dog_to_sheep)

        # Check similarity to evaluation layouts
        self._current_episode_similarity_match = None
        if not dog_positions or not sheep_positions:
            observation = self._current_observation()
            return observation, self._info()
        cur_mean_dog = tuple(np.mean(dog_positions, axis=0))
        cur_mean_sheep = tuple(np.mean(sheep_positions, axis=0))
        for ev_seed, ev_layout in self._evaluation_layouts.items():
            if not ev_layout["dog_positions"] or not ev_layout["sheep_positions"]:
                continue
            ev_mean_dog = tuple(np.mean(ev_layout["dog_positions"], axis=0))
            ev_mean_sheep = tuple(np.mean(ev_layout["sheep_positions"], axis=0))

            if (pen_origin == ev_layout["pen_position"] and
                math.hypot(cur_mean_dog[0] - ev_mean_dog[0], cur_mean_dog[1] - ev_mean_dog[1]) < 6.0 and
                math.hypot(cur_mean_sheep[0] - ev_mean_sheep[0], cur_mean_sheep[1] - ev_mean_sheep[1]) < 6.0):
                self._current_episode_similarity_match = ev_seed
                break

        observation = self._current_observation()
        return observation, self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_name = ACTION_ORDER[int(action)]
        mask = self.action_masks()
        if not bool(mask[int(action)]):
            action_name = "wait"
        self._pending_actions.append(action_name)
        terminated = False
        truncated = False
        reward = 0.0
        if len(self._pending_actions) >= self._environment.dog_count:
            snapshot, breakdown = self._environment.step(self._pending_actions)
            reward = float(breakdown.total)
            self._episode_reward += reward
            terminated = bool(snapshot.success or snapshot.stopped)
            truncated = bool(snapshot.timeout)
            info = self._info()
            info["team_step_completed"] = True
            info["final_snapshot"] = snapshot.to_dict()
            self._pending_actions = []
            self._current_dog_index = 0

            # Increment similarity counters if episode finished
            if terminated or truncated:
                penned = getattr(snapshot, "penned_count", sum(1 for s in snapshot.sheep if getattr(s, "penned", False)))
                total_sheep = len(snapshot.sheep)
                status_str = "SUCCESS" if snapshot.success else ("TIMEOUT" if snapshot.timeout else "STOPPED")
                info["episode"] = {
                    "r": float(self._episode_reward),
                    "l": int(getattr(snapshot, "step", 0)),
                    "success": bool(snapshot.success),
                    "penned": int(penned),
                    "total_sheep": int(total_sheep),
                    "status": status_str,
                    "seed": int(self._latest_seed),
                }

                if self._current_episode_similarity_match is not None:
                    self._similarity_episodes[self._current_episode_similarity_match] += 1
                    if snapshot.success:
                        self._similarity_successes[self._current_episode_similarity_match] += 1
                    self._current_episode_similarity_match = None
        else:
            info = self._info(action_mask=mask)
            self._current_dog_index += 1
            info["team_step_completed"] = False
        observation = (
            self._current_observation()
            if not (terminated or truncated)
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Rendering is not supported; this environment is headless."""

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask for the current dog's legal actions."""
        mask_map = self._environment.action_mask_for_dog(
            self._current_dog_index,
            reserved_positions={
                self._environment.project_dog_action(index, action)
                for index, action in enumerate(self._pending_actions)
            },
        )
        return np.asarray([mask_map[action] for action in ACTION_ORDER], dtype=bool)

    def _current_observation(self) -> np.ndarray:
        return np.asarray(
            self._environment.build_observation_for_dog(self._current_dog_index).values,
            dtype=np.float32,
        )

    def _info(self, action_mask: np.ndarray | None = None) -> dict[str, Any]:
        return {
            "seed": self._latest_seed,
            "current_dog_index": self._current_dog_index,
            "action_mask": (
                action_mask if action_mask is not None else self.action_masks()
            ).tolist(),
            "pending_actions": list(self._pending_actions),
        }

    def _next_seed(self) -> int:
        if self.fixed_seed_sequence:
            seed = self.fixed_seed_sequence[self._fixed_seed_index % len(self.fixed_seed_sequence)]
            self._fixed_seed_index += 1
            return int(seed)
        seed = self.config.training.train_seed + self._episode_counter
        self._episode_counter += 1
        return int(seed)

    def get_scenario_usage_summary(self) -> dict[str, Any]:
        """Return scenario usage statistics for observability."""
        if not self.config.training.scenario_training_enabled:
            return {"scenario_training_enabled": False}
        return self._scenario_sampler.get_usage_summary()

    @property
    def episode_counter(self) -> int:
        """Return the count of training episodes completed in this environment instance."""
        return self._episode_counter
