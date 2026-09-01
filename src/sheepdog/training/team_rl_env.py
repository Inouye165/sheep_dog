"""True team-step Gymnasium adapter for joint sheepdog actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.observations import HERD_DOG_SLOTS
from sheepdog.training.rl_env import SheepdogRLAdapter


def build_team_observation(environment: SheepdogEnvironment) -> np.ndarray:
    """Return fixed-width observations for every supported dog slot."""
    per_dog_size = len(environment.build_observation_for_dog(0).values)
    observations: list[np.ndarray] = []
    for dog_index in range(HERD_DOG_SLOTS):
        if dog_index < environment.dog_count:
            values = environment.build_observation_for_dog(dog_index).values
            observations.append(np.asarray(values, dtype=np.float32))
        else:
            observations.append(np.zeros(per_dog_size, dtype=np.float32))
    return np.concatenate(observations)


def build_team_action_mask(environment: SheepdogEnvironment) -> np.ndarray:
    """Return concatenated action masks for every supported dog slot."""
    masks: list[np.ndarray] = []
    wait_index = ACTION_ORDER.index("wait")
    for dog_index in range(HERD_DOG_SLOTS):
        if dog_index < environment.dog_count:
            mask_map = environment.action_mask_for_dog(dog_index)
            masks.append(np.asarray([mask_map[action] for action in ACTION_ORDER], dtype=bool))
        else:
            inactive_mask = np.zeros(len(ACTION_ORDER), dtype=bool)
            inactive_mask[wait_index] = True
            masks.append(inactive_mask)
    return np.concatenate(masks)


class TeamActionRLEnv(SheepdogRLAdapter):
    """Expose one complete dog-team decision as one PPO transition."""

    def __init__(
        self,
        config: LabConfig,
        fixed_seed_sequence: Sequence[int] | None = None,
    ) -> None:
        if config.environment.dogs > HERD_DOG_SLOTS:
            raise ValueError(
                f"TeamActionRLEnv supports at most {HERD_DOG_SLOTS} dogs, "
                f"got {config.environment.dogs}"
            )
        super().__init__(config, fixed_seed_sequence=fixed_seed_sequence)
        observation_size = len(self._environment.build_observation_for_dog(0).values)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size * HERD_DOG_SLOTS,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(HERD_DOG_SLOTS, len(ACTION_ORDER), dtype=np.int64)
        )

    def step(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply all active dog actions and advance the world exactly once."""
        action_indices = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_indices.shape != (HERD_DOG_SLOTS,):
            raise ValueError(
                f"Expected {HERD_DOG_SLOTS} team actions, got {action_indices.size}"
            )
        if np.any(action_indices < 0) or np.any(action_indices >= len(ACTION_ORDER)):
            raise ValueError("Team action index is outside the action space")

        transition: tuple[np.ndarray, float, bool, bool, dict[str, Any]] | None = None
        for dog_index in range(self._environment.dog_count):
            transition = super().step(int(action_indices[dog_index]))
        if transition is None:
            raise RuntimeError("Cannot execute a team action without active dogs")

        observation, reward, terminated, truncated, info = transition
        info["team_step_completed"] = True
        info["team_action_indices"] = action_indices.tolist()
        info["world_step_count"] = self._environment.step_count
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the flattened mask expected by MaskablePPO MultiDiscrete actions."""
        return build_team_action_mask(self._environment)

    def _current_observation(self) -> np.ndarray:
        """Return the centralized fixed-width team observation."""
        return build_team_observation(self._environment)
