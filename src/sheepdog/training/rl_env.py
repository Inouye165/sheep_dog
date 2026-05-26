"""Gymnasium adapter for training a shared dog policy with PPO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment


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
        self._environment.reset(seed=self._latest_seed)
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
        info = self._info()
        if len(self._pending_actions) >= self._environment.dog_count:
            snapshot, breakdown = self._environment.step(self._pending_actions)
            reward = float(breakdown.total)
            terminated = bool(snapshot.success or snapshot.stopped)
            truncated = bool(snapshot.timeout)
            info = self._info()
            info["team_step_completed"] = True
            info["final_snapshot"] = snapshot.to_dict()
            self._pending_actions = []
            self._current_dog_index = 0
        else:
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

    def _info(self) -> dict[str, Any]:
        return {
            "seed": self._latest_seed,
            "current_dog_index": self._current_dog_index,
            "action_mask": self.action_masks().tolist(),
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
