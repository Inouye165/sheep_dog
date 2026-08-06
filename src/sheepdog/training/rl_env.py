"""Gymnasium adapter for training a shared dog policy with PPO."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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
        self.env_index: int = getattr(config, "env_index", 0)
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
        self._active_trajectory_buffer: list[dict[str, Any]] = []

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

    def _next_seed(self) -> int:
        if self.fixed_seed_sequence:
            seed = self.fixed_seed_sequence[self._fixed_seed_index % len(self.fixed_seed_sequence)]
            self._fixed_seed_index += 1
            return int(seed)
        self._episode_counter += 1
        return int(self.config.training.train_seed + self._episode_counter)

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
        self._active_trajectory_buffer = []

        # Use scenario sampler if scenario training is enabled
        if self.config.training.scenario_training_enabled:
            selection = self._scenario_sampler.sample(self._episode_counter)
            if selection.scenario is not None:
                self._environment.reset_from_scenario(selection.scenario)
            else:
                self._environment.reset(seed=self._latest_seed)
        else:
            self._environment.reset(seed=self._latest_seed)

        # Buffer step 0 initial state snapshot
        init_snap = self._environment.get_state_snapshot()
        self._active_trajectory_buffer.append({
            "step": 0,
            "actions": [],
            "snapshot": init_snap.to_dict(),
            "reward": {
                "progress_to_pen": 0.0,
                "sheep_penned": 0.0,
                "flock_cohesion": 0.0,
                "scatter_penalty": 0.0,
                "time_penalty": 0.0,
                "no_progress_penalty": 0.0,
                "wall_pressure_penalty": 0.0,
                "wait_penalty": 0.0,
                "terminal_success": 0.0,
                "terminal_failure": 0.0,
                "total": 0.0,
            },
        })

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

            # Append frame to active trajectory buffer
            self._active_trajectory_buffer.append({
                "step": int(snapshot.step),
                "actions": list(self._pending_actions),
                "snapshot": snapshot.to_dict(),
                "reward": breakdown.to_dict(),
            })

            team_actions = list(self._pending_actions)
            self._pending_actions = []
            self._current_dog_index = 0

            # Handle episode completion & selective trajectory capture
            if terminated or truncated:
                penned = getattr(snapshot, "penned_count", sum(1 for s in snapshot.sheep if getattr(s, "penned", False)))
                total_sheep = len(snapshot.sheep)
                status_str = "SUCCESS" if snapshot.success else ("TIMEOUT" if snapshot.timeout else "STOPPED")
                stage = self.config.rewards.instincts.curriculum_stage

                from sheepdog.training.replay_writer import (
                    get_global_capture_policy,
                    get_replay_writer,
                    ReplayWriteJob,
                )
                from sheepdog.training.episode_store import get_episode_store

                policy = get_global_capture_policy()
                should_capture, capture_reason = policy.should_capture(
                    stage=stage,
                    success=bool(snapshot.success),
                    status=status_str,
                    reward=float(self._episode_reward),
                )

                replay_id = None
                replay_path_str = None

                if should_capture:
                    import time
                    ep_num = int(self._episode_counter)
                    seed_val = int(self._latest_seed)
                    ts_suffix = int(time.time())
                    replay_id = f"diag_stage{stage}_ep{ep_num}_seed{seed_val}_t{ts_suffix}"
                    event_key = f"ep_{ep_num}_stage_{stage}_seed_{seed_val}_env_{self.env_index}_t{ts_suffix}"

                    output_dir = Path("artifacts/replays")
                    output_path = output_dir / f"{replay_id}.json.gz"
                    replay_path_str = str(output_path)

                    stats_dict = self._environment._stats.to_dict() if hasattr(self._environment, "_stats") else {}

                    payload = {
                        "seed": seed_val,
                        "policy_name": f"ppo_stage_{stage}",
                        "trainer_type": "maskable_ppo",
                        "policy_type": "team_shared",
                        "replay_mode": "training_diagnostic",
                        "replay_source": "training-diagnostic",
                        "capture_reason": capture_reason,
                        "capture_status": "queued",
                        "environment": self.config.to_dict()["environment"],
                        "final_snapshot": snapshot.to_dict(),
                        "stats": stats_dict,
                        "frames": list(self._active_trajectory_buffer),
                    }

                    job = ReplayWriteJob(
                        replay_id=replay_id,
                        event_key=event_key,
                        payload=payload,
                        output_path=output_path,
                        capture_reason=capture_reason,
                        replay_source="training-diagnostic",
                        use_gzip=True,
                    )

                    writer = get_replay_writer(output_dir=output_dir, episode_store=get_episode_store())
                    writer.enqueue(job)

                info["episode"] = {
                    "r": float(self._episode_reward),
                    "l": int(getattr(snapshot, "step", 0)),
                    "success": bool(snapshot.success),
                    "penned": int(penned),
                    "total_sheep": int(total_sheep),
                    "status": status_str,
                    "seed": int(self._latest_seed),
                    "replay_available": 1 if should_capture else 0,
                    "replay_id": replay_id,
                    "replay_path": replay_path_str,
                    "replay_source": "training-diagnostic" if should_capture else None,
                    "capture_reason": capture_reason if should_capture else "not_requested",
                    "capture_status": "queued" if should_capture else "not_requested",
                }

                if self._current_episode_similarity_match is not None:
                    self._similarity_episodes[self._current_episode_similarity_match] += 1
                    if snapshot.success:
                        self._similarity_successes[self._current_episode_similarity_match] += 1
                    self._current_episode_similarity_match = None

                self._active_trajectory_buffer = []
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
                self._environment.dogs[i].position
                for i in range(self._current_dog_index)
            },
        )
        return np.array(
            [mask_map.get(action, False) for action in ACTION_ORDER],
            dtype=bool,
        )

    def _info(
        self, action_mask: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Build metadata dict emitted alongside step observations."""
        if action_mask is None:
            action_mask = self.action_masks()
        return {
            "current_dog_index": self._current_dog_index,
            "action_mask": action_mask,
            "simulated_seconds": self._environment.simulated_seconds,
            "step_count": self._environment.step_count,
            "latest_seed": self._latest_seed,
        }

    def _current_observation(self) -> np.ndarray:
        """Extract observation vector for the active dog."""
        obs = self._environment.build_observation_for_dog(self._current_dog_index)
        return np.array(obs.values, dtype=np.float32)
