"""Gymnasium adapter for training a shared dog policy with PPO."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.observations import build_observation_builder
from sheepdog.training.replay_writer import (
    get_global_capture_policy,
    validate_telemetry_config,
)
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
        *,
        telemetry_mode: str | None = None,
        telemetry_head_steps: int | None = None,
        telemetry_tail_steps: int | None = None,
        max_telemetry_steps: int | None = None,
        max_telemetry_steps_per_episode: int | None = None,
    ) -> None:
        super().__init__()
        effective_max = (
            max_telemetry_steps_per_episode
            if max_telemetry_steps_per_episode is not None
            else max_telemetry_steps
        )
        validate_telemetry_config(
            telemetry_mode=telemetry_mode,
            max_telemetry_steps=effective_max,
            telemetry_head_steps=telemetry_head_steps,
            telemetry_tail_steps=telemetry_tail_steps,
        )
        self._telemetry_mode: str | None = telemetry_mode
        self._telemetry_head_steps: int | None = telemetry_head_steps
        self._telemetry_tail_steps: int | None = telemetry_tail_steps
        self._max_telemetry_steps: int | None = effective_max
        self.config = config
        self.fixed_seed_sequence = tuple(fixed_seed_sequence or ())
        self._fixed_seed_index = 0
        self._episode_counter = 0
        self.env_index: int = getattr(config, "env_index", 0)
        self._environment = SheepdogEnvironment(config)
        self._environment._observation_builder = build_observation_builder(
            config.training.observation_mode
        )
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
        self._pending_decisions: list[dict[str, Any]] = []
        self._captured_telemetry_steps = 0
        self._telemetry_truncated = False
        self._telemetry_omitted = False
        tail_len = (
            telemetry_tail_steps
            if telemetry_tail_steps is not None and telemetry_tail_steps > 0
            else (500 if telemetry_tail_steps is None else 1)
        )
        self._tail_telemetry_buffer: deque[tuple[int, dict[str, Any]]] = deque(
            maxlen=tail_len
        )
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
        seeds = getattr(getattr(config, "training", None), "evaluation_seeds", (11, 23, 37, 41, 53, 59, 61, 67, 71, 73))
        self._similarity_episodes = {s: 0 for s in seeds}
        self._similarity_successes = {s: 0 for s in seeds}
        self._evaluation_layouts = {}
        self._precompute_evaluation_layouts()
        self._current_episode_similarity_match = None

    def _check_mid_episode_telemetry_mutation(self) -> None:
        if hasattr(self, "_environment") and not self._environment._terminated and self._environment._step_count > 0:
            raise RuntimeError("Cannot change telemetry configuration during an active episode")

    @property
    def telemetry_mode(self) -> str:
        if self._telemetry_mode is not None:
            return self._telemetry_mode
        return getattr(get_global_capture_policy(), "telemetry_mode", "prefix")

    @telemetry_mode.setter
    def telemetry_mode(self, value: str) -> None:
        self._check_mid_episode_telemetry_mutation()
        validate_telemetry_config(telemetry_mode=value)
        self._telemetry_mode = value

    @property
    def telemetry_head_steps(self) -> int:
        if self._telemetry_head_steps is not None:
            return self._telemetry_head_steps
        return getattr(get_global_capture_policy(), "telemetry_head_steps", 200)

    @telemetry_head_steps.setter
    def telemetry_head_steps(self, value: int) -> None:
        self._check_mid_episode_telemetry_mutation()
        validate_telemetry_config(telemetry_head_steps=value)
        self._telemetry_head_steps = value

    @property
    def telemetry_tail_steps(self) -> int:
        if self._telemetry_tail_steps is not None:
            return self._telemetry_tail_steps
        return getattr(get_global_capture_policy(), "telemetry_tail_steps", 500)

    @telemetry_tail_steps.setter
    def telemetry_tail_steps(self, value: int) -> None:
        self._check_mid_episode_telemetry_mutation()
        validate_telemetry_config(telemetry_tail_steps=value)
        self._telemetry_tail_steps = value
        expected_tail_len = value if value > 0 else 1
        self._tail_telemetry_buffer = deque(
            self._tail_telemetry_buffer, maxlen=expected_tail_len
        )

    @property
    def max_telemetry_steps(self) -> int:
        if self._max_telemetry_steps is not None:
            return self._max_telemetry_steps
        return getattr(get_global_capture_policy(), "max_telemetry_steps_per_episode", 1000)

    @max_telemetry_steps.setter
    def max_telemetry_steps(self, value: int) -> None:
        self._check_mid_episode_telemetry_mutation()
        validate_telemetry_config(max_telemetry_steps=value)
        self._max_telemetry_steps = value

    @property
    def max_telemetry_steps_per_episode(self) -> int:
        return self.max_telemetry_steps

    @max_telemetry_steps_per_episode.setter
    def max_telemetry_steps_per_episode(self, value: int) -> None:
        self.max_telemetry_steps = value

    def export_active_trajectory(self, *, finalize_tail: bool = False) -> dict[str, Any]:
        """Export the current active trajectory buffer.

        If *finalize_tail* is True and an episode is interrupted mid-run before
        normal termination, the current contents of the rolling tail buffer are
        stitched into the trajectory copy and an omission marker is placed at step H+1
        (if a gap exists relative to the current step count).
        If *finalize_tail* is False, frames in the active trajectory buffer are
        returned as-is (with intermediate steps > H having step_telemetry=None
        and recent steps buffered in _tail_telemetry_buffer).
        """
        import copy
        frames = copy.deepcopy(self._active_trajectory_buffer)
        current_step = int(self._environment.step_count)
        capture_policy = get_global_capture_policy()
        telemetry_enabled = bool(capture_policy.capture_step_telemetry)
        mode = self.telemetry_mode
        head_steps = self.telemetry_head_steps
        tail_steps = self.telemetry_tail_steps

        if finalize_tail and len(frames) > 1 and telemetry_enabled and mode == "head_tail":
            T = current_step
            tail_start = max(1, T - tail_steps + 1) if tail_steps > 0 else T + 1
            if tail_start > head_steps + 1:
                omitted_start = head_steps + 1
                omitted_end = tail_start - 1
                omitted_count = tail_start - 1 - head_steps
                omission_marker = {
                    "telemetry_omitted": True,
                    "omitted_reason": "intentionally_omitted_window",
                    "telemetry_mode": "head_tail",
                    "head_steps": head_steps,
                    "tail_steps": tail_steps,
                    "omitted_range": f"steps_{omitted_start}_to_{omitted_end}",
                    "omitted_steps_start": omitted_start,
                    "omitted_steps_end": omitted_end,
                    "omitted_steps_count": omitted_count,
                    "total_world_steps": T,
                }
                marker_idx = omitted_start
                if 0 <= marker_idx < len(frames):
                    frames[marker_idx]["step_telemetry"] = omission_marker

            for step_num, payload in self._tail_telemetry_buffer:
                if 0 <= step_num < len(frames):
                    frames[step_num]["step_telemetry"] = payload

        return {
            "step_count": current_step,
            "total_frames": len(frames),
            "frames": frames,
            "tail_buffer_count": len(self._tail_telemetry_buffer),
        }

    def _precompute_evaluation_layouts(self) -> None:
        """Pre-generate initial positions for standard evaluation seeds to check training similarity."""
        try:
            temp_env = SheepdogEnvironment(self.config)
            seeds = getattr(getattr(self.config, "training", None), "evaluation_seeds", (11, 23, 37, 41, 53, 59, 61, 67, 71, 73))
            for seed in seeds:
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

    def update_failure_weights(self, weights: dict[str, float]) -> None:
        """Update failure scenario weights on the underlying scenario sampler."""
        self._scenario_sampler.set_failure_weights(weights)

    def update_env_config(self, env_config: EnvironmentConfig) -> None:
        """Update environment config on the environment and underlying scenario sampler."""
        self._environment.env_config = env_config
        self._scenario_sampler.update_env_config(env_config)

    def get_scenario_sampler_summary(self) -> dict[str, Any]:
        """Return the scenario sampler usage and telemetry summary."""
        return self._scenario_sampler.get_usage_summary()

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
        self._pending_decisions = []
        self._captured_telemetry_steps = 0
        self._telemetry_truncated = False
        self._telemetry_omitted = False
        self._tail_telemetry_buffer.clear()
        self._current_dog_index = 0
        self._episode_reward = 0.0
        self._active_trajectory_buffer = []
        self._latest_scenario_id: str | None = None

        # Use scenario sampler if scenario training or failure-directed training is enabled
        if self.config.training.scenario_training_enabled or getattr(self.config.training, "failure_directed_training_enabled", False):
            selection = self._scenario_sampler.sample(self._episode_counter)
            if selection.scenario is not None:
                self._latest_scenario_id = getattr(
                    selection, "scenario_id", getattr(selection.scenario, "scenario_id", None)
                )
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
            if val < d["min"]:
                d["min"] = val
            if val > d["max"]:
                d["max"] = val
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
        from sheepdog.training.replay_writer import get_global_capture_policy

        dog_idx = self._current_dog_index
        raw_action_name = ACTION_ORDER[int(action)]
        mask = self._current_dog_action_mask()
        is_legal = bool(mask[int(action)])
        executed_action = raw_action_name if is_legal else "wait"
        replacement_reason = None if is_legal else "action_masked_illegal"
        self._pending_actions.append(executed_action)

        capture_policy = get_global_capture_policy()
        telemetry_enabled = bool(capture_policy.capture_step_telemetry)
        mode = self.telemetry_mode
        budget = self.max_telemetry_steps
        head_steps = self.telemetry_head_steps
        tail_steps = self.telemetry_tail_steps
        expected_tail_len = tail_steps if tail_steps > 0 else 1
        if self._tail_telemetry_buffer.maxlen != expected_tail_len:
            self._tail_telemetry_buffer = deque(
                self._tail_telemetry_buffer, maxlen=expected_tail_len
            )

        current_world_step = int(self._environment.step_count) + 1
        if not telemetry_enabled:
            capture_active = False
        elif mode == "prefix":
            capture_active = self._captured_telemetry_steps < budget
        elif mode == "head_tail":
            capture_active = (tail_steps > 0) or (current_world_step <= head_steps)
        else:
            capture_active = False

        if capture_active:
            dog_obs = self._current_observation().tolist()
            decision_mask_dict = {
                act: bool(mask[idx]) for idx, act in enumerate(ACTION_ORDER)
            }
            physical_mask = self._environment.action_mask_for_dog(
                dog_idx,
                policy_mode="neural_policy",
                reserved_positions={
                    self._environment.dogs[i].position
                    for i in range(dog_idx)
                },
            )
            physical_mask_dict = {
                act: bool(physical_mask.get(act, False)) for act in ACTION_ORDER
            }
            self._pending_decisions.append({
                "dog_index": dog_idx,
                "world_step": int(self._environment.step_count),
                "observation": dog_obs,
                "decision_action_mask": decision_mask_dict,
                "recomputed_physical_mask": physical_mask_dict,
                "selected_action": raw_action_name,
                "executed_action": executed_action,
                "replacement_reason": replacement_reason,
                "action_probabilities": None,
                "logits": None,
                "value_estimate": None,
                "decision_mode": "sampled",
                "policy_version": getattr(self, "policy_version", None),
                "checkpoint_id": getattr(self, "checkpoint_id", None),
                "mask_type": "actual_decision_time",
            })

        terminated = False
        truncated = False
        reward = 0.0
        if len(self._pending_actions) >= self._environment.dog_count:
            step_telemetry: dict[str, Any] | None = None
            if telemetry_enabled:
                if mode == "prefix":
                    if self._captured_telemetry_steps < budget:
                        self._captured_telemetry_steps += 1
                        from sheepdog.training.diagnostics.temporal_interaction import (
                            build_step_instrumentation_payload,
                        )

                        step_telemetry = build_step_instrumentation_payload(
                            self._environment,
                            decisions=self._pending_decisions,
                            raw_actions=self._pending_actions,
                        )
                    elif not self._telemetry_truncated:
                        self._telemetry_truncated = True
                        step_telemetry = {
                            "telemetry_truncated": True,
                            "truncation_reason": "per_episode_telemetry_budget_exhausted",
                            "telemetry_limit_steps": budget,
                            "affected_range": f"steps_{self._environment.step_count}_to_end",
                        }
                elif mode == "head_tail" and capture_active:
                    from sheepdog.training.diagnostics.temporal_interaction import (
                        build_step_instrumentation_payload,
                    )

                    payload = build_step_instrumentation_payload(
                        self._environment,
                        decisions=self._pending_decisions,
                        raw_actions=self._pending_actions,
                    )
                    if current_world_step <= head_steps:
                        step_telemetry = payload
                        self._captured_telemetry_steps += 1
                    else:
                        if tail_steps > 0:
                            self._tail_telemetry_buffer.append((current_world_step, payload))
                        step_telemetry = None

            snapshot, breakdown = self._environment.step(self._pending_actions)
            reward = float(breakdown.total)
            self._episode_reward += reward
            terminated = bool(snapshot.success or snapshot.stopped)
            truncated = bool(snapshot.timeout)
            info = self._info()
            info["team_step_completed"] = True
            info["final_snapshot"] = snapshot.to_dict()

            # Append frame to active trajectory buffer
            frame_entry: dict[str, Any] = {
                "step": int(snapshot.step),
                "actions": list(self._pending_actions),
                "snapshot": snapshot.to_dict(),
                "reward": breakdown.to_dict(),
            }
            if step_telemetry is not None:
                frame_entry["step_telemetry"] = step_telemetry
            self._active_trajectory_buffer.append(frame_entry)

            self._pending_actions = []
            self._pending_decisions = []
            self._current_dog_index = 0

            # Handle episode completion & selective trajectory capture
            if terminated or truncated:
                if telemetry_enabled and mode == "head_tail":
                    T = int(snapshot.step)
                    tail_start = max(1, T - tail_steps + 1) if tail_steps > 0 else T + 1
                    if tail_start > head_steps + 1:
                        self._telemetry_omitted = True
                        omitted_start = head_steps + 1
                        omitted_end = tail_start - 1
                        omitted_count = tail_start - 1 - head_steps
                        omission_marker = {
                            "telemetry_omitted": True,
                            "omitted_reason": "intentionally_omitted_window",
                            "telemetry_mode": "head_tail",
                            "head_steps": head_steps,
                            "tail_steps": tail_steps,
                            "omitted_range": f"steps_{omitted_start}_to_{omitted_end}",
                            "omitted_steps_start": omitted_start,
                            "omitted_steps_end": omitted_end,
                            "omitted_steps_count": omitted_count,
                            "total_world_steps": T,
                        }
                        marker_idx = omitted_start
                        if 0 <= marker_idx < len(self._active_trajectory_buffer):
                            self._active_trajectory_buffer[marker_idx]["step_telemetry"] = omission_marker

                    for step_num, payload in self._tail_telemetry_buffer:
                        if 0 <= step_num < len(self._active_trajectory_buffer):
                            self._active_trajectory_buffer[step_num]["step_telemetry"] = payload

                penned = getattr(snapshot, "penned_count", sum(1 for s in snapshot.sheep if getattr(s, "penned", False)))
                total_sheep = len(snapshot.sheep)
                status_str = "SUCCESS" if snapshot.success else ("TIMEOUT" if snapshot.timeout else "STOPPED")
                stage = self.config.rewards.instincts.curriculum_stage

                from sheepdog.training.episode_store import get_episode_store
                from sheepdog.training.replay_writer import (
                    ReplayWriteJob,
                    get_global_capture_policy,
                    get_replay_writer,
                )

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

                    output_dir = Path(self.config.training.output_dir) / "replays"
                    output_path = output_dir / f"{replay_id}.json.gz"
                    replay_path_str = str(output_path)

                    environment_stats = getattr(self._environment, "_stats", None)
                    stats_dict = (
                        environment_stats.to_dict()
                        if environment_stats is not None
                        else {}
                    )

                    T = int(snapshot.step)
                    if mode == "head_tail":
                        tail_start = max(1, T - tail_steps + 1) if tail_steps > 0 else T + 1
                        has_gap = tail_start > head_steps + 1
                        omitted_steps = (tail_start - 1 - head_steps) if has_gap else 0
                        detailed_steps = (T - omitted_steps) if telemetry_enabled else 0
                        telemetry_coverage = {
                            "mode": mode,
                            "total_steps": T,
                            "captured_detailed_steps": detailed_steps,
                            "omitted_steps_count": omitted_steps,
                            "retained_head_range": [1, min(head_steps, T)] if (head_steps > 0 and telemetry_enabled) else None,
                            "retained_tail_range": [tail_start, T] if (tail_steps > 0 and tail_start <= T and telemetry_enabled) else None,
                            "omitted_range": [head_steps + 1, tail_start - 1] if has_gap else None,
                            "telemetry_truncated": False,
                            "complete_decision_coverage": (tail_start <= head_steps + 1) and telemetry_enabled,
                        }
                    else:
                        is_truncated = self._telemetry_truncated
                        detailed_steps = min(T, budget) if telemetry_enabled else 0
                        omitted_steps = (T - detailed_steps) if telemetry_enabled else 0
                        telemetry_coverage = {
                            "mode": mode,
                            "total_steps": T,
                            "captured_detailed_steps": detailed_steps,
                            "omitted_steps_count": omitted_steps,
                            "retained_head_range": [1, min(budget, T)] if (budget > 0 and telemetry_enabled) else None,
                            "retained_tail_range": None,
                            "omitted_range": [budget + 1, T] if is_truncated else None,
                            "telemetry_truncated": is_truncated,
                            "complete_decision_coverage": (budget >= T) and telemetry_enabled,
                        }

                    payload = {
                        "seed": seed_val,
                        "scenario_id": getattr(self, "_latest_scenario_id", None),
                        "policy_name": f"ppo_stage_{stage}",
                        "trainer_type": "maskable_ppo",
                        "policy_type": "team_shared",
                        "replay_mode": "training_diagnostic",
                        "replay_source": "training-diagnostic",
                        "strategy_version": "1.0.0",
                        "discount_timebase": "sequential_shared_dog",
                        "discount_factor": float(self.config.training.gamma),
                        "dog_count": int(self._environment.dog_count),
                        "capture_reason": capture_reason,
                        "capture_status": "queued",
                        "environment": self.config.to_dict()["environment"],
                        "final_snapshot": snapshot.to_dict(),
                        "stats": stats_dict,
                        "telemetry_mode": mode,
                        "telemetry_head_steps": head_steps if mode == "head_tail" else None,
                        "telemetry_tail_steps": tail_steps if mode == "head_tail" else None,
                        "telemetry_truncated": self._telemetry_truncated,
                        "telemetry_limit_steps": budget if (mode == "prefix" and telemetry_enabled) else None,
                        "telemetry_coverage": telemetry_coverage,
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

                    writer = get_replay_writer(
                        output_dir=output_dir,
                        episode_store=get_episode_store(
                            Path(self.config.training.output_dir) / "training-telemetry.sqlite"
                        ),
                    )
                    writer.enqueue(job)

                dog_cnt = self._environment.dog_count
                spawn_mode = getattr(snapshot, "spawn_mode", getattr(self._environment, "_spawn_mode", ""))
                pen = getattr(snapshot, "pen", getattr(self._environment, "_pen", None))
                parts = [f"{dog_cnt}d/{total_sheep}s"]
                if spawn_mode:
                    parts.append(f"Spawn: {spawn_mode}")
                if pen and hasattr(pen, "origin"):
                    pen_ox, pen_oy = int(pen.origin.x), int(pen.origin.y)
                    opening = getattr(pen, "opening", "")
                    if opening:
                        parts.append(f"Pen: ({pen_ox},{pen_oy},{opening})")
                    else:
                        parts.append(f"Pen: ({pen_ox},{pen_oy})")
                field_setup_str = " | ".join(parts)
                env_stats = getattr(self._environment, "_stats", None)
                info["episode"] = {
                    "r": float(self._episode_reward),
                    "l": int(getattr(snapshot, "step", 0)),
                    "success": bool(snapshot.success),
                    "penned": int(penned),
                    "total_sheep": int(total_sheep),
                    "status": status_str,
                    "seed": int(self._latest_seed),
                    "field_setup": field_setup_str,
                    "replay_available": 1 if should_capture else 0,
                    "replay_id": replay_id,
                    "replay_path": replay_path_str,
                    "replay_source": "training-diagnostic" if should_capture else None,
                    "capture_reason": capture_reason if should_capture else "not_requested",
                    "capture_status": "queued" if should_capture else "not_requested",
                    "reward_breakdown": dict(getattr(self._environment, "_episode_reward_breakdown", {})),
                    "pen_zone": getattr(env_stats, "pen_zone", ""),
                    "spawn_mode": getattr(env_stats, "spawn_mode", spawn_mode),
                    "initial_sheep_zone": getattr(env_stats, "initial_sheep_zone", ""),
                    "final_sheep_zone": getattr(env_stats, "final_sheep_zone", ""),
                    "corner_time_pct": getattr(env_stats, "corner_time_pct", 0.0),
                    "wall_time_pct": getattr(env_stats, "wall_time_pct", 0.0),
                    "corner_stuck_at_end": getattr(env_stats, "corner_stuck_at_end", False),
                    "spatial_metrics": getattr(env_stats, "spatial_metrics", {}),
                }

                if self._current_episode_similarity_match is not None:
                    match_seed = self._current_episode_similarity_match
                    self._similarity_episodes[match_seed] = self._similarity_episodes.get(match_seed, 0) + 1
                    if snapshot.success:
                        self._similarity_successes[match_seed] = self._similarity_successes.get(match_seed, 0) + 1
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
        return self._current_dog_action_mask()

    def _current_dog_action_mask(self) -> np.ndarray:
        """Return the legal-action mask for the active dog."""
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
