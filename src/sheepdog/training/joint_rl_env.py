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
        probe = self._obs_builder.build_hierarchical(
            self._environment, 0, self._current_command
        )
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
        info = self._info()

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
        else:
            self._current_dog_index += 1
            info["team_step_completed"] = False
            info["shepherd_command"] = self._current_command

        observation = (
            self._current_observation()
            if not (terminated or truncated)
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return observation, reward, terminated, truncated, info

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

    def _info(self) -> dict[str, Any]:
        return {
            "seed": self._latest_seed,
            "current_dog_index": self._current_dog_index,
            "action_mask": self.action_masks().tolist(),
            "pending_actions": list(self._pending_actions),
            "shepherd_command": self._current_command,
        }

    def _next_seed(self) -> int:
        if self.fixed_seed_sequence:
            seed = self.fixed_seed_sequence[
                self._fixed_seed_index % len(self.fixed_seed_sequence)
            ]
            self._fixed_seed_index += 1
            return int(seed)
        seed = self.config.training.train_seed + self._episode_counter
        self._episode_counter += 1
        return int(seed)
