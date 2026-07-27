"""Joint-action Gymnasium environment for hierarchical shepherd + neural dog training.

This adapter differs from ``SheepdogRLAdapter`` in one critical way:

    ``SheepdogRLAdapter`` (sequential)
        RL takes one action per dog in sequence; the team step fires only after
        the last dog acts.  Credit assignment is harder because early-dog rewards
        are zero and only the final step carries the full team reward.

    ``JointActionRLEnv`` (joint / synchronous)
        One ``env.step(actions)`` call consumes *all* dog actions together,
        advances the world once, and returns the shared reward immediately.
        This is better for multi-agent credit assignment and matches the
        conceptual model of dogs acting in parallel.

Observation
-----------
Each call to ``step`` returns the observation for **dog 0** (the next dog the
shared policy should act for).  The env cycles through dogs 0 … N-1 across
consecutive calls to ``step``, collecting actions, then fires the team step
when all N actions are in.

To train all dogs with a single policy through standard PPO, the caller drives
the loop externally via the MultiDogVecEnv wrapper (or any MARL library that
supports this pattern).  For SB3 single-agent training, ``JointActionRLEnv``
behaves identically to ``SheepdogRLAdapter`` from the SB3 perspective – the
difference is which observation builder is used and that *all* dog actions are
submitted simultaneously.

Observation space
-----------------
The observation is a flat float32 vector built by
``HierarchicalObservationBuilder``, which includes:

  base role-aware features (from RoleAwareObservationBuilder)
  + shepherd command one-hot (8 values)
  + dog identity: normalized index, normalized count, one-hot slot (MAX_DOG_SLOTS values)

Action space
------------
Discrete(9) – same as ``SheepdogRLAdapter``.

Shepherd
--------
A ``ScriptedShepherd`` is instantiated once and queried on every *team* step.
Its command is baked into each dog's observation.  This is Phase-A; in Phase-B
the shepherd can be replaced by a learned policy via dependency injection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.observations import HierarchicalObservationBuilder
from sheepdog.shepherd import ScriptedShepherd, ShepherdCommand


class JointActionRLEnv(gym.Env[np.ndarray, int]):
    """Single-policy multi-dog environment with hierarchical shepherd commands.

    From SB3's point of view this looks like a standard single-agent env.
    Internally it cycles through all dogs, collecting one action per dog,
    then submits all actions to the underlying team environment at once.

    The observation for each dog slot includes:
      - the base role-aware feature vector (same as SheepdogRLAdapter)
      - the current shepherd command (one-hot)
      - the dog's identity (normalized index + one-hot slot)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: LabConfig,
        fixed_seed_sequence: Sequence[int] | None = None,
        shepherd: ScriptedShepherd | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.fixed_seed_sequence = tuple(fixed_seed_sequence or ())
        self._fixed_seed_index = 0
        self._episode_counter = 0
        self._environment = SheepdogEnvironment(config)
        self._shepherd = shepherd if shepherd is not None else ScriptedShepherd()
        self._obs_builder = HierarchicalObservationBuilder()

        # Probe observation size with a dummy reset so the space is stable.
        self._environment.reset(seed=config.training.train_seed)
        self._current_command: ShepherdCommand = self._shepherd.issue_command(self._environment)
        probe = self._obs_builder.build_hierarchical(self._environment, 0, self._current_command)
        observation_size = len(probe.values)

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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed()
        self._latest_seed = int(seed)
        self._pending_actions = []
        self._current_dog_index = 0
        self._environment.reset(seed=self._latest_seed)
        self._current_command = self._shepherd.issue_command(self._environment)

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
            return self._current_observation(), self._info()
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

        return self._current_observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Accept one dog action; fire the team step when all dogs have acted."""
        action_name = ACTION_ORDER[int(action)]
        mask = self.action_masks()
        if not bool(mask[int(action)]):
            action_name = "wait"

        self._pending_actions.append(action_name)

        terminated = False
        truncated = False
        reward = 0.0

        if len(self._pending_actions) >= self._environment.dog_count:
            # All dogs have chosen – advance the world.
            snapshot, breakdown = self._environment.step(self._pending_actions)
            reward = float(breakdown.total)
            terminated = bool(snapshot.success or snapshot.stopped)
            truncated = bool(snapshot.timeout)
            # Update shepherd command for the next round of observations.
            if not (terminated or truncated):
                self._current_command = self._shepherd.issue_command(self._environment)
            info = self._info()
            info["team_step_completed"] = True
            info["shepherd_command"] = self._current_command
            info["final_snapshot"] = snapshot.to_dict()
            self._pending_actions = []
            self._current_dog_index = 0

            # Increment similarity counters if episode finished
            if (terminated or truncated) and self._current_episode_similarity_match is not None:
                self._similarity_episodes[self._current_episode_similarity_match] += 1
                if snapshot.success:
                    self._similarity_successes[self._current_episode_similarity_match] += 1
                self._current_episode_similarity_match = None
        else:
            info = self._info(action_mask=mask)
            self._current_dog_index += 1
            info["team_step_completed"] = False
            info["shepherd_command"] = self._current_command

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_observation(self) -> np.ndarray:
        obs = self._obs_builder.build_hierarchical(
            self._environment,
            self._current_dog_index,
            self._current_command,
        )
        return np.asarray(obs.values, dtype=np.float32)

    def _info(self, action_mask: np.ndarray | None = None) -> dict[str, Any]:
        return {
            "seed": self._latest_seed,
            "current_dog_index": self._current_dog_index,
            "action_mask": (
                action_mask if action_mask is not None else self.action_masks()
            ).tolist(),
            "pending_actions": list(self._pending_actions),
            "shepherd_command": self._current_command,
        }

    def _next_seed(self) -> int:
        if self.fixed_seed_sequence:
            seed = self.fixed_seed_sequence[self._fixed_seed_index % len(self.fixed_seed_sequence)]
            self._fixed_seed_index += 1
            return int(seed)
        seed = self.config.training.train_seed + self._episode_counter
        self._episode_counter += 1
        return int(seed)

    @property
    def episode_counter(self) -> int:
        """Return the count of training episodes completed in this environment instance."""
        return self._episode_counter
