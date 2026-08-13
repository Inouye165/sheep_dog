"""MaskablePPO trainer for the experimental neural-policy path."""

from __future__ import annotations

import functools
import contextlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from sheepdog.atomic_io import atomic_write_json
from sheepdog.checkpoints.store import (
    CheckpointMetadata,
    get_action_space_hash,
    get_observation_schema_hash,
)
from sheepdog.environment import ACTION_ORDER, ENV_CONFIG_VERSION
from sheepdog.policies.neural import NeuralPolicy, tensorboard_available
from sheepdog.rewards import REWARD_SCHEMA_VERSION
from sheepdog.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NeuralTrainingRunSummary:
    """Metadata for a neural PPO training run."""

    checkpoints: list[dict[str, Any]]
    final_model_path: str
    policy_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "checkpoints": self.checkpoints,
            "final_model_path": self.final_model_path,
            "policy_config": self.policy_config,
        }


class _TrainingProgressCallback(BaseCallback):
    """Relay PPO timestep progress back to the training manager."""

    def __init__(
        self,
        emit: Callable[[dict[str, Any]], None],
        *,
        should_stop: Callable[[], bool] | None,
        report_interval: int,
        total_timesteps: int,
        starting_total_episodes: int,
        batch_total_episodes: int,
        batch_total_timesteps: int,
        completed_timesteps: int,
        completed_segments: int,
        segment_index: int,
        policy_version: int,
        starting_total: int,
    ) -> None:
        super().__init__()
        self._emit = emit
        self._starting_total = starting_total
        self._should_stop = should_stop
        self._report_interval = max(1, report_interval)
        self._total_timesteps = max(1, total_timesteps)
        self._starting_total_episodes = starting_total_episodes
        self._batch_total_episodes = batch_total_episodes
        self._batch_total_timesteps = batch_total_timesteps
        self._completed_timesteps = completed_timesteps
        self._completed_segments = completed_segments
        self._segment_index = segment_index
        self.policy_version = policy_version
        self._last_reported_steps = 0

    def _on_step(self) -> bool:
        if self._should_stop is not None and self._should_stop():
            return False

        # Emit individual episode completion logs as they finish
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done and idx < len(infos):
                    ep_info = infos[idx].get("episode") if isinstance(infos[idx], dict) else None
                    if isinstance(ep_info, dict):
                        actual_eps = 0
                        try:
                            if self.model is not None and self.model.get_env() is not None:
                                curr_counters = self.model.get_env().get_attr("_episode_counter")
                                actual_eps = int(sum(curr_counters))
                        except Exception:
                            actual_eps = 0
                        ep_num = self._starting_total + actual_eps
                        current_global_ts = self._completed_timesteps + int(self.num_timesteps)
                        self._emit(
                            {
                                "phase": "episode_complete",
                                "episode": ep_num,
                                "current_episode": ep_num,
                                "total_episodes_trained": ep_num,
                                "actual_completed_episodes": actual_eps,
                                "global_timestep": current_global_ts,
                                "reward": float(ep_info.get("r", 0.0)),
                                "length": int(ep_info.get("l", 0)),
                                "success": bool(ep_info.get("success", False)),
                                "penned": int(ep_info.get("penned", 0)),
                                "total_sheep": int(ep_info.get("total_sheep", 0)),
                                "status": str(ep_info.get("status", "UNKNOWN")),
                                "seed": ep_info.get("seed"),
                                "field_setup": ep_info.get("field_setup"),
                                "replay_available": ep_info.get("replay_available"),
                                "replay_id": ep_info.get("replay_id"),
                                "replay_path": ep_info.get("replay_path"),
                                "replay_source": ep_info.get("replay_source"),
                                "capture_reason": ep_info.get("capture_reason"),
                                "capture_status": ep_info.get("capture_status"),
                                "reward_breakdown": ep_info.get("reward_breakdown"),
                            }
                        )

        num_timesteps = int(self.num_timesteps)
        if (
            num_timesteps < self._total_timesteps
            and num_timesteps - self._last_reported_steps < self._report_interval
        ):
            return True
        self._last_reported_steps = num_timesteps
        completion = min(1.0, num_timesteps / self._total_timesteps)
        fractional_completed = self._completed_segments + completion
        ppo_update_count = getattr(self.model, "_n_updates", 0)
        current_policy_version = self.policy_version + ppo_update_count
        last_policy_update_time = datetime.now(UTC).isoformat() if ppo_update_count > 0 else None
        
        actual_completed_episodes = 0
        try:
            if self.model is not None and self.model.get_env() is not None:
                curr_counters = self.model.get_env().get_attr("_episode_counter")
                actual_completed_episodes = int(sum(curr_counters))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            actual_completed_episodes = 0

        self._emit(
            {
                "phase": "learning",
                "batch_completed_episodes": fractional_completed,
                "actual_completed_episodes": actual_completed_episodes,
                "current_episode": self._starting_total + actual_completed_episodes,
                "total_episodes_trained": self._starting_total + actual_completed_episodes,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"Learning neural policy: "
                    f"{min(num_timesteps, self._total_timesteps)}/{self._total_timesteps} "
                    f"timesteps ({completion:.0%})"
                ),
                "batch_total_episodes": self._batch_total_episodes,
                "batch_total_timesteps": self._batch_total_timesteps,
                "batch_completed_timesteps": self._completed_timesteps
                + min(num_timesteps, self._total_timesteps),
                "policy_version": current_policy_version,
                "ppo_update_count": ppo_update_count,
                "last_policy_update_time": last_policy_update_time,
            }
        )
        return True


def finish_wandb_run() -> None:
    """Finish an active W&B run when the optional SDK is available."""
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is not None and hasattr(wandb, "finish"):
        try:
            wandb.finish()
        except Exception as exc:
            print(f"Warning: wandb.finish() raised {exc}", flush=True)


def _wandb_finish_on_exit(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            finish_wandb_run()
    return wrapper


class MaskablePPOTrainer(Trainer):
    """Train the shared role-aware neural policy with MaskablePPO."""

    MODEL_DIRNAME = "models"

    def has_compatible_policy_state(self) -> bool:
        """Return whether the persisted policy state can resume this trainer."""
        return self._has_compatible_policy_state()

    def _training_signature(self) -> dict[str, Any]:
        # Deliberately excludes ``dogs`` and ``sheep`` counts: the observation
        # vector is now padded to fixed MAX_SHEEP_SLOTS / HERD_DOG_SLOTS slots so
        # the architecture is stable across curriculum promotions.  Speeds and
        # sprint multiplier are kept because they change training dynamics.
        return {
            "action_size": len(ACTION_ORDER),
            "observation_mode": self.config.training.observation_mode,
            "rewards": asdict(self.config.rewards),
            "environment": {
                "dog_speed": self.config.environment.dog_speed,
                "dog_sprint_multiplier": self.config.environment.dog_sprint_multiplier,
                "sheep_speed": self.config.environment.sheep_speed,
            },
        }

    @staticmethod
    def _strip_non_architectural_fields(sig: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with stage/toggle metadata and curriculum overrides removed before comparison.

        ``curriculum_stage``, ``debug_reward_breakdown``, and
        ``enable_instinct_rewards`` are runtime toggles that do not affect the
        neural-network architecture; stripping them and curriculum-dependent
        speed/reward overrides lets stage promotions reuse the trained model
        and accumulate ``total_episodes_trained`` correctly.
        """
        import copy
        sig = copy.deepcopy(sig)

        from sheepdog.curriculum import CURRICULUM_REWARD_OVERRIDES

        # 1. Strip instinct toggles (runtime flags, not architecture).
        if "rewards" in sig and isinstance(sig["rewards"], dict):
            instincts = sig["rewards"].get("instincts")
            if isinstance(instincts, dict):
                for key in ("curriculum_stage", "debug_reward_breakdown", "enable_instinct_rewards"):
                    instincts.pop(key, None)

        # 2. Strip curriculum-controlled environment speeds symmetrically.
        # These are tuned per stage and do not affect the network architecture,
        # so they must be dropped from every signature regardless of the stage
        # embedded in it.
        if "environment" in sig and isinstance(sig["environment"], dict):
            for key in ("dog_speed", "sheep_speed"):
                sig["environment"].pop(key, None)

        # 3. Strip every curriculum reward-override key symmetrically.
        # Stripping only the current stage's override keys is asymmetric: the
        # first stage that introduces overrides (stage 7) would never match the
        # prior stage's stored signature, forcing a needless from-scratch model
        # reset on promotion. Dropping the union of all override keys from every
        # signature keeps the comparison stable across the whole curriculum.
        if "rewards" in sig and isinstance(sig["rewards"], dict):
            override_keys = {
                key
                for overrides in CURRICULUM_REWARD_OVERRIDES.values()
                for key in overrides
            }
            for key in override_keys:
                sig["rewards"].pop(key, None)

        return sig

    def _has_compatible_policy_state(self) -> bool:
        stored_signature = self._loaded_state.get("training_signature")
        if not isinstance(stored_signature, dict):
            return False
        normalize = self._strip_non_architectural_fields
        return normalize(stored_signature) == normalize(self._training_signature())

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "total_episodes_trained": 0,
                "total_environment_episodes": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
                "incomplete_batch": None,
                "run_id": None,
                "parent_run_id": None,
                "parent_checkpoint_id": None,
            }
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "total_episodes_trained": 0,
                "total_environment_episodes": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
                "incomplete_batch": None,
                "run_id": None,
                "parent_run_id": None,
                "parent_checkpoint_id": None,
            }
        return {
            "total_episodes_trained": int(payload.get("total_episodes_trained", 0)),
            "total_environment_episodes": int(
                payload.get("total_environment_episodes", 0)
            ),
            "total_timesteps": int(payload.get("total_timesteps", 0)),
            "policy_state_path": payload.get("policy_state_path"),
            "policy_config": payload.get("policy_config"),
            "training_signature": payload.get("training_signature"),
            "best_model_path": payload.get("best_model_path"),
            "best_model_curriculum_stage": payload.get("best_model_curriculum_stage"),
            "best_success_rate": payload.get("best_success_rate"),
            "best_average_reward": payload.get("best_average_reward"),
            "best_completion_steps": payload.get("best_completion_steps"),
            "incomplete_batch": payload.get("incomplete_batch"),
            "run_id": payload.get("run_id"),
            "parent_run_id": payload.get("parent_run_id"),
            "parent_checkpoint_id": payload.get("parent_checkpoint_id"),
            "policy_version": payload.get("policy_version"),
            "training_scenario_coverage": payload.get("training_scenario_coverage"),
        }

    @_wandb_finish_on_exit
    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> NeuralTrainingRunSummary:  # type: ignore[override]
        train_config = self.config.training
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)
        model_root = self.output_root / self.MODEL_DIRNAME
        model_root.mkdir(parents=True, exist_ok=True)
        active_stage = self.config.rewards.instincts.curriculum_stage
        if active_stage >= 9:
            stage_model_root = self.output_root / "checkpoints" / f"stage{active_stage}"
        else:
            stage_model_root = model_root
        stage_model_root.mkdir(parents=True, exist_ok=True)
        resuming_policy = (
            bool(self._loaded_state.get("policy_state_path"))
            and self.has_compatible_policy_state()
        )
        starting_total = self.total_episodes_trained if resuming_policy else 0
        starting_environment_episodes = (
            int(self._loaded_state.get("total_environment_episodes", 0))
            if resuming_policy
            else 0
        )
        loaded_p_ver = self._loaded_state.get("policy_version")
        policy_version = int(loaded_p_ver) if (resuming_policy and loaded_p_ver is not None) else 0
        batch_total = max(1, len(train_config.checkpoint_episodes))

        # Initialize Weights & Biases if enabled
        wandb_enabled = False
        if getattr(train_config, "wandb_enabled", False) or os.getenv("SHEEPDOG_WANDB_ENABLED", "").lower() in ("true", "1"):
            wandb_enabled = True

        if wandb_enabled:
            try:
                import wandb
                if getattr(wandb, "run", None) is None:
                    wandb_config = {
                        "learning_rate": train_config.learning_rate,
                        "learning_rate_final": train_config.learning_rate_final,
                        "total_timesteps": train_config.total_timesteps,
                        "batch_total": batch_total,
                        "environment": self.config.to_dict().get("environment", {}),
                        "rewards": self.config.to_dict().get("rewards", {}),
                    }
                    wandb.init(
                        project="sheep_dog_herding",
                        config=wandb_config,
                        sync_tensorboard=tensorboard_available(),
                    )
            # W&B exposes several SDK-specific exception types across versions.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logging.getLogger(__name__).warning(
                    "Could not initialize W&B: %s. Running without W&B.", exc
                )
                wandb_enabled = False
        n_checkpoints = batch_total
        configured_timestep_targets = tuple(train_config.checkpoint_timesteps)
        if configured_timestep_targets:
            if (
                len(configured_timestep_targets) != n_checkpoints
                or configured_timestep_targets[-1] != train_config.total_timesteps
                or any(
                    current <= previous
                    for previous, current in zip(
                        (0, *configured_timestep_targets[:-1]),
                        configured_timestep_targets,
                    )
                )
            ):
                raise ValueError(
                    "checkpoint_timesteps must be increasing, match checkpoint count, "
                    "and end at total_timesteps"
                )
            timestep_targets = configured_timestep_targets
        else:
            steps_per_segment = max(1, train_config.total_timesteps // n_checkpoints)
            timestep_targets = tuple(
                steps_per_segment * segment for segment in range(1, n_checkpoints + 1)
            )
        segment_timesteps = tuple(
            current - previous
            for previous, current in zip((0, *timestep_targets[:-1]), timestep_targets)
        )
        uniform_segment_timesteps = (
            segment_timesteps[0] if len(set(segment_timesteps)) == 1 else None
        )
        starting_total_timesteps = (
            int(self._loaded_state.get("total_timesteps", 0)) if resuming_policy else 0
        )
        # Resume an interrupted batch when the new job has the same segment count.
        _incomplete = self._loaded_state.get("incomplete_batch") if resuming_policy else None
        skip_segments: int = 0
        if (
            isinstance(_incomplete, dict)
            and int(_incomplete.get("batch_total_segments", 0)) == n_checkpoints
            and (
                _incomplete.get("batch_timestep_targets") == list(timestep_targets)
                or (
                    not configured_timestep_targets
                    and int(_incomplete.get("batch_steps_per_segment", 0))
                    == uniform_segment_timesteps
                )
            )
            and 0 < int(_incomplete.get("batch_completed_segments", 0)) < n_checkpoints
        ):
            skip_segments = int(_incomplete["batch_completed_segments"])
        if skip_segments > 0 and isinstance(_incomplete, dict):
            stored_batch_start = _incomplete.get("batch_starting_checkpoint_episode")
            if stored_batch_start is not None:
                batch_starting_checkpoint_episode = int(stored_batch_start)
            else:
                completed_slot = int(train_config.checkpoint_episodes[skip_segments - 1])
                batch_starting_checkpoint_episode = starting_total - completed_slot
        else:
            batch_starting_checkpoint_episode = starting_total + (1 if resuming_policy else 0)
        best_success_rate: float = (
            float(self._loaded_state["best_success_rate"])
            if resuming_policy and self._loaded_state.get("best_success_rate") is not None
            else -1.0
        )
        best_average_reward: float = (
            float(self._loaded_state["best_average_reward"])
            if resuming_policy and self._loaded_state.get("best_average_reward") is not None
            else float("-inf")
        )
        best_completion_steps: float = (
            float(self._loaded_state["best_completion_steps"])
            if resuming_policy and self._loaded_state.get("best_completion_steps") is not None
            else float("inf")
        )
        tracked_best_model_path: Path | None = (
            Path(self._loaded_state["best_model_path"])
            if resuming_policy and self._loaded_state.get("best_model_path")
            else None
        )
        best_model_curriculum_stage: int = (
            int(self._loaded_state["best_model_curriculum_stage"])
            if resuming_policy and self._loaded_state.get("best_model_curriculum_stage") is not None
            else self.config.rewards.instincts.curriculum_stage
        )

        def emit(payload: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            payload.setdefault("batch_total_episodes", batch_total)
            payload.setdefault("starting_total_episodes", starting_total)
            progress_callback(payload)

        emit(
            {
                "phase": "starting",
                "batch_completed_episodes": skip_segments,
                "current_episode": None,
                "total_episodes_trained": starting_total,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"Resuming neural training from {starting_total} checkpoints"
                    if resuming_policy
                    else "Starting fresh neural training run"
                ),
            }
        )

        # Prefer the best-scoring model when resuming so each new training batch
        # continues from the highest-quality weights rather than the most recent
        # (potentially degraded) checkpoint.  Fall back to policy_state_path when
        # no best model has been recorded yet (e.g. first batch ever run).
        best_model_path_str = self._loaded_state.get("best_model_path")
        resume_path = best_model_path_str or self._loaded_state.get("policy_state_path")
        if resuming_policy and resume_path:
            policy = NeuralPolicy.load(
                resume_path,
                self.config,
                self._loaded_state.get("policy_config"),
            )
        else:
            policy = NeuralPolicy.initialize(self.config)

        checkpoint_payloads: list[dict[str, Any]] = (
            list(self._load_summary_checkpoints()) if resuming_policy else []
        )
        saved_model_path: Path | None = None
        interrupted = False

        # Stamp an in-progress batch marker before the loop so a crash mid-batch
        # is resumable on the next call with the same episode count.
        _batch_marker_pre: dict[str, Any] = {
            "batch_completed_segments": skip_segments,
            "batch_total_segments": n_checkpoints,
            "batch_steps_per_segment": uniform_segment_timesteps,
            "batch_timestep_targets": list(timestep_targets),
            "batch_starting_checkpoint_episode": batch_starting_checkpoint_episode,
        }
        _pre_loop_state = dict(self._loaded_state)
        _pre_loop_state["incomplete_batch"] = _batch_marker_pre
        atomic_write_json(self._state_path, _pre_loop_state)

        for completed_checkpoints, _checkpoint_slot in enumerate(
            train_config.checkpoint_episodes,
            start=1,
        ):
            if should_stop is not None and should_stop():
                interrupted = True
                break
            if completed_checkpoints <= skip_segments:
                continue
            new_segments = completed_checkpoints - skip_segments
            segment_steps = segment_timesteps[completed_checkpoints - 1]
            resumed_target = timestep_targets[skip_segments - 1] if skip_segments else 0
            cumulative_ts = (
                starting_total_timesteps
                + timestep_targets[completed_checkpoints - 1]
                - resumed_target
            )
            emit(
                {
                    "phase": "learning",
                    "batch_completed_episodes": completed_checkpoints - 1,
                    "current_episode": None,
                    "total_episodes_trained": starting_total + new_segments - 1,
                    "checkpoint_episode": None,
                    "best_score": None,
                    "message": (
                        f"Learning neural policy: segment {completed_checkpoints}/{n_checkpoints}"
                        f" (0/{segment_steps} timesteps)"
                    ),
                }
            )
            progress_reporter = _TrainingProgressCallback(
                emit,
                should_stop=should_stop,
                # Emit progress frequently enough that long PPO segments do not
                # look stalled in the UI. Cap the update rate to avoid chatty logs.
                report_interval=max(250, min(5_000, segment_steps // 100)),
                total_timesteps=segment_steps,
                starting_total_episodes=starting_total + new_segments - 1,
                batch_total_episodes=batch_total,
                batch_total_timesteps=timestep_targets[-1],
                completed_timesteps=(
                    starting_total_timesteps
                    + (
                        timestep_targets[completed_checkpoints - 2]
                        if completed_checkpoints > 1
                        else 0
                    )
                ),
                completed_segments=completed_checkpoints - 1,
                segment_index=completed_checkpoints - 1,
                policy_version=policy_version,
                starting_total=starting_environment_episodes,
            )
            from stable_baselines3.common.callbacks import CallbackList
            callbacks_to_use = [progress_reporter]
            if wandb_enabled:
                try:
                    from wandb.integration.sb3 import WandbCallback
                    callbacks_to_use.append(WandbCallback(verbose=0))
                # The optional integration may fail with version-specific SDK errors.
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logging.getLogger(__name__).warning(
                        "Failed to create W&B callback: %s", exc
                    )

            callback_list = CallbackList(callbacks_to_use)

            # Linear LR annealing across the batch: full LR at segment 0,
            # learning_rate_final at the last segment.  This keeps updates
            # aggressive early (fast cliff recovery) and conservative late
            # (prevents policy collapse near the end of a run).
            _batch_done = completed_checkpoints - skip_segments - 1
            _batch_span = max(1, n_checkpoints - skip_segments - 1)
            _batch_progress = _batch_done / _batch_span
            
            # Enforce 5e-5 floor on learning rate annealing
            policy.model.learning_rate = max(
                5e-5,
                train_config.learning_rate
                + (train_config.learning_rate_final - train_config.learning_rate) * _batch_progress
            )
            # Apply training overrides for exploration and advantage estimation
            policy.model.ent_coef = train_config.entropy_coef
            policy.model.gae_lambda = train_config.gae_lambda
            
            training_phase = (
                self.runtime_tracker.phase("training")
                if self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            if hasattr(policy.model, "tensorboard_log") and policy.model.tensorboard_log:
                try:
                    Path(policy.model.tensorboard_log).mkdir(parents=True, exist_ok=True)
                except Exception:
                    policy.model.tensorboard_log = None

            with training_phase:
                try:
                    policy.model.learn(
                        total_timesteps=segment_steps,
                        reset_num_timesteps=True,
                        progress_bar=False,
                        callback=callback_list,
                    )
                except (FileNotFoundError, OSError) as tb_err:
                    logger.warning("Tensorboard logging failed (%s). Continuing rollout without Tensorboard logging.", tb_err)
                    policy.model.tensorboard_log = None
                    if hasattr(policy.model, "_logger"):
                        policy.model._logger = None
                    policy.model.learn(
                        total_timesteps=segment_steps,
                        reset_num_timesteps=False,
                        progress_bar=False,
                        callback=callback_list,
                    )
            
            # Increment policy update version
            policy_version += 1
            policy.policy_version = policy_version

            environment_episodes_since_run_start = 0
            try:
                if hasattr(policy.model, "get_env") and policy.model.get_env() is not None:
                    environment_episodes_since_run_start = int(
                        sum(policy.model.get_env().get_attr("_episode_counter"))
                    )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                environment_episodes_since_run_start = 0
            environment_episodes_total = (
                starting_environment_episodes + environment_episodes_since_run_start
            )

            # Retrieve training scenario coverage stats from env
            eval_seeds = getattr(getattr(self.config, "training", None), "evaluation_seeds", (11, 23, 37, 41, 53, 59, 61, 67, 71, 73))
            curr_coverage = self._loaded_state.get("training_scenario_coverage")
            if not isinstance(curr_coverage, dict):
                curr_coverage = {
                    "seeds_seen": [],
                    "configs_seen": [],
                    "min_sheep_to_pen": float("inf"),
                    "max_sheep_to_pen": float("-inf"),
                    "sum_sheep_to_pen": 0.0,
                    "count_sheep_to_pen": 0,
                    "min_dog_to_sheep": float("inf"),
                    "max_dog_to_sheep": float("-inf"),
                    "sum_dog_to_sheep": 0.0,
                    "count_dog_to_sheep": 0,
                    "similarity_episodes": {str(k): 0 for k in eval_seeds},
                    "similarity_successes": {str(k): 0 for k in eval_seeds},
                }
            curr_coverage["similarity_episodes"] = {str(k): v for k, v in curr_coverage.get("similarity_episodes", {}).items()}
            curr_coverage["similarity_successes"] = {str(k): v for k, v in curr_coverage.get("similarity_successes", {}).items()}

            if hasattr(policy.model, "get_env") and policy.model.get_env() is not None:
                venv = policy.model.get_env()
                try:
                    if hasattr(venv, "env_method"):
                        workers_stats = venv.env_method("get_coverage_stats")
                        for ws in workers_stats:
                            if not isinstance(ws, dict):
                                continue
                            seeds_list = set(curr_coverage.get("seeds_seen", []))
                            seeds_list.update(ws.get("seeds_seen_list", []))
                            curr_coverage["seeds_seen"] = list(seeds_list)

                            configs_list = set(curr_coverage.get("configs_seen", []))
                            configs_list.update(ws.get("configs_seen_list", []))
                            curr_coverage["configs_seen"] = list(configs_list)

                            if ws.get("count_sheep_to_pen", 0) > 0:
                                curr_coverage["min_sheep_to_pen"] = min(curr_coverage["min_sheep_to_pen"], ws["min_sheep_to_pen"])
                                curr_coverage["max_sheep_to_pen"] = max(curr_coverage["max_sheep_to_pen"], ws["max_sheep_to_pen"])
                                curr_coverage["sum_sheep_to_pen"] += ws["sum_sheep_to_pen"]
                                curr_coverage["count_sheep_to_pen"] += ws["count_sheep_to_pen"]

                            if ws.get("count_dog_to_sheep", 0) > 0:
                                curr_coverage["min_dog_to_sheep"] = min(curr_coverage["min_dog_to_sheep"], ws["min_dog_to_sheep"])
                                curr_coverage["max_dog_to_sheep"] = max(curr_coverage["max_dog_to_sheep"], ws["max_dog_to_sheep"])
                                curr_coverage["sum_dog_to_sheep"] += ws["sum_dog_to_sheep"]
                                curr_coverage["count_dog_to_sheep"] += ws["count_dog_to_sheep"]

                            ws_sim_episodes = ws.get("similarity_episodes", {})
                            ws_sim_successes = ws.get("similarity_successes", {})
                            all_keys = set(eval_seeds) | {int(k) for k in ws_sim_episodes.keys() if str(k).isdigit()}
                            for k in all_keys:
                                k_str = str(k)
                                ep_val = ws_sim_episodes.get(k, ws_sim_episodes.get(k_str, 0))
                                suc_val = ws_sim_successes.get(k, ws_sim_successes.get(k_str, 0))
                                curr_coverage["similarity_episodes"][k_str] = curr_coverage["similarity_episodes"].get(k_str, 0) + ep_val
                                curr_coverage["similarity_successes"][k_str] = curr_coverage["similarity_successes"].get(k_str, 0) + suc_val
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    logging.getLogger(__name__).debug(
                        "PPO environment coverage statistics are unavailable",
                        exc_info=True,
                    )

            self._loaded_state["training_scenario_coverage"] = curr_coverage
            
            # Extract PPO diagnostics from model logger
            logger_obj = getattr(policy.model, "logger", None)
            approx_kl = 0.0
            clip_fraction = 0.0
            explained_variance = 0.0
            policy_gradient_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0
            loss = 0.0
            if logger_obj is not None:
                name_to_value = getattr(logger_obj, "name_to_value", {})
                approx_kl = float(name_to_value.get("train/approx_kl", 0.0))
                clip_fraction = float(name_to_value.get("train/clip_fraction", 0.0))
                explained_variance = float(name_to_value.get("train/explained_variance", 0.0))
                policy_gradient_loss = float(name_to_value.get("train/policy_gradient_loss", 0.0))
                value_loss = float(name_to_value.get("train/value_loss", 0.0))
                entropy_loss = float(name_to_value.get("train/entropy_loss", 0.0))
                loss = float(name_to_value.get("train/loss", 0.0))
                
            if should_stop is not None and should_stop():
                interrupted = True
                break
            model_save_phase = (
                self.runtime_tracker.phase("checkpoint_save")
                if self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            with model_save_phase:
                active_stage = self.config.rewards.instincts.curriculum_stage
                model_name = f"maskable-ppo-stage{active_stage}-{cumulative_ts:08d}" if active_stage >= 9 else f"maskable-ppo-{cumulative_ts:08d}"
                saved_model_path = policy.save(stage_model_root / model_name)
            total_eps_this_checkpoint = batch_starting_checkpoint_episode + _checkpoint_slot
            run_id = self._loaded_state.get("run_id")
            chk_id = f"chk_{run_id}_pv_{policy_version}_ts_{cumulative_ts}"
            recorded_time = datetime.now(UTC).isoformat()
            active_stage = self.config.rewards.instincts.curriculum_stage

            from sheepdog.checkpoints.store import (
                compute_env_config_hash,
                compute_seed_set_id,
            )

            self.evaluator.runtime_tracker = self.runtime_tracker
            quick_seed_count = max(1, int(train_config.quick_evaluation_seed_count))
            quick_seeds = tuple(train_config.evaluation_seeds[:quick_seed_count])
            evaluation_phase = (
                self.runtime_tracker.phase("evaluation")
                if self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            with evaluation_phase:
                quick_summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                    policy,
                    quick_seeds,
                    checkpoint_episode=total_eps_this_checkpoint,
                    capture_replays=False,
                    evaluation_mode="quick",
                    run_id=run_id,
                    checkpoint_id=chk_id,
                    policy_version=policy_version,
                    curriculum_stage=active_stage,
                )
                is_final_checkpoint = completed_checkpoints == n_checkpoints
                confidence_candidate = (
                    quick_summary.success_rate
                    >= float(train_config.confidence_candidate_success_rate)
                )
                if is_final_checkpoint or confidence_candidate or len(quick_summary.records) < len(train_config.evaluation_seeds):
                    summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                        policy,
                        tuple(train_config.evaluation_seeds),
                        checkpoint_episode=total_eps_this_checkpoint,
                        capture_replays=False,
                        evaluation_mode="confidence",
                        run_id=run_id,
                        checkpoint_id=chk_id,
                        policy_version=policy_version,
                        curriculum_stage=active_stage,
                    )
                else:
                    summary = quick_summary

            current_stage = self.config.rewards.instincts.curriculum_stage
            is_new_best = summary.promotion_eligible and (
                current_stage > best_model_curriculum_stage
                or (
                    current_stage == best_model_curriculum_stage
                    and (
                        summary.success_rate > best_success_rate
                        or (
                            summary.success_rate == best_success_rate
                            and summary.average_reward > best_average_reward
                        )
                        or (
                            summary.success_rate == best_success_rate
                            and summary.average_reward == best_average_reward
                            and summary.average_completion_steps < best_completion_steps
                        )
                    )
                )
            )
            periodic_replay = int(train_config.replay_export_every_n_checkpoints)
            should_export_replay = summary.promotion_eligible and (
                (is_new_best and train_config.replay_export_on_new_best)
                or (is_final_checkpoint and train_config.replay_export_on_final)
                or (
                    summary.success_rate == 0
                    and train_config.replay_export_on_failed_diagnostic
                )
                or (periodic_replay > 0 and completed_checkpoints % periodic_replay == 0)
            )
            representative_replay_path: Path | None = None
            if should_export_replay:
                representative_replay_path = self.evaluator.export_replay(
                    policy,
                    int(summary.records[0].seed),
                    total_eps_this_checkpoint,
                )
                records = list(summary.records)
                records[0] = replace(
                    records[0], replay_path=str(representative_replay_path)
                )
                summary = replace(summary, records=tuple(records))

            runtime_snapshot = (
                self.runtime_tracker.episode_snapshot()
                if self.runtime_tracker is not None
                else {}
            )
            metadata = CheckpointMetadata(
                checkpoint_episode=total_eps_this_checkpoint,
                total_training_episodes=total_eps_this_checkpoint,
                policy_name=policy.name,
                trainer_type=policy.trainer_type,
                policy_type=policy.policy_type,
                seed=train_config.train_seed,
                success_rate=summary.success_rate,
                average_completion_steps=summary.average_completion_steps,
                timeout_rate=summary.timeout_rate,
                average_sheep_penned=summary.average_sheep_penned,
                average_reward=summary.average_reward,
                environment_config=self.config.to_dict()["environment"],
                reward_config=self.config.to_dict()["rewards"],
                policy_state_path=str(saved_model_path),
                policy_config=policy.config.to_dict(),
                evaluation_replay_path=(
                    str(representative_replay_path)
                    if representative_replay_path is not None
                    else None
                ),
                run_id=run_id,
                checkpoint_id=chk_id,
                environment_episodes_total=environment_episodes_total,
                environment_episodes_since_run_start=environment_episodes_since_run_start,
                parent_run_id=self._loaded_state.get("parent_run_id"),
                parent_checkpoint_id=self._loaded_state.get("parent_checkpoint_id"),
                global_timestep=cumulative_ts,
                observation_schema_hash=get_observation_schema_hash(self.config),
                action_space_hash=get_action_space_hash(),
                reward_schema_version=REWARD_SCHEMA_VERSION,
                env_config_version=ENV_CONFIG_VERSION,
                created_timestamp=recorded_time,
                deterministic_evaluation=True,
                evaluation_seeds=[record.seed for record in summary.records],
                policy_version=policy_version,
                policy_gradient_loss=policy_gradient_loss,
                value_loss=value_loss,
                entropy_loss=entropy_loss,
                loss=loss,
                approx_kl=approx_kl,
                clip_fraction=clip_fraction,
                explained_variance=explained_variance,
                training_scenario_coverage=curr_coverage,
                curriculum_stage=active_stage,
                evaluation_seed_set_id=compute_seed_set_id(
                    [record.seed for record in summary.records]
                ),
                evaluation_seed_count=len(summary.records),
                environment_config_hash=compute_env_config_hash(self.config.to_dict()["environment"]),
                evaluation_timestamp=recorded_time,
                evaluation_id=summary.evaluation_id,
                evaluation_mode=summary.evaluation_mode,
                promotion_eligible=summary.promotion_eligible,
                **runtime_snapshot,
            )
            if is_new_best:
                best_success_rate = summary.success_rate
                best_average_reward = summary.average_reward
                best_completion_steps = summary.average_completion_steps
                best_model_curriculum_stage = current_stage
                best_model_name = f"stage{current_stage}-best-model" if current_stage >= 9 else "best-model"
                tracked_best_model_path = policy.save(stage_model_root / best_model_name)
            checkpoint_write_phase = (
                self.runtime_tracker.phase("checkpoint_save")
                if self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            with checkpoint_write_phase:
                checkpoint_path = self.checkpoint_store.write(metadata)
            checkpoint_payload = {
                "checkpoint_episode": total_eps_this_checkpoint,
                "recorded_at": recorded_time,
                "checkpoint": checkpoint_path.name,
                "evaluation": evaluation_json.name,
                "replay": (
                    str(representative_replay_path)
                    if representative_replay_path is not None
                    else None
                ),
                "policy_name": policy.name,
                "trainer_type": policy.trainer_type,
                "policy_type": policy.policy_type,
                "policy_mode": policy.name,
                "replay_mode": "neural_ppo",
                "total_training_episodes": total_eps_this_checkpoint,
                "policy_state_path": str(saved_model_path),
                "success_rate": summary.success_rate,
                "timeout_rate": summary.timeout_rate,
                "average_completion_steps": summary.average_completion_steps,
                "average_completion_seconds": summary.average_completion_seconds,
                "average_sheep_penned": summary.average_sheep_penned,
                "average_reward": summary.average_reward,
                "average_distance_to_pen": summary.average_distance_to_pen,
                "average_flock_spread": summary.average_flock_spread,
                "environment_config": self.config.to_dict()["environment"],
                "reward_config": self.config.to_dict()["rewards"],
                "records": [record.to_dict() for record in summary.records],
                "run_id": run_id,
                "checkpoint_id": chk_id,
                "environment_episodes_total": environment_episodes_total,
                "environment_episodes_since_run_start": environment_episodes_since_run_start,
                "parent_run_id": self._loaded_state.get("parent_run_id"),
                "parent_checkpoint_id": self._loaded_state.get("parent_checkpoint_id"),
                "global_timestep": cumulative_ts,
                "observation_schema_hash": get_observation_schema_hash(self.config),
                "action_space_hash": get_action_space_hash(),
                "reward_schema_version": REWARD_SCHEMA_VERSION,
                "env_config_version": ENV_CONFIG_VERSION,
                "created_timestamp": recorded_time,
                "deterministic_evaluation": True,
                "evaluation_seeds": [record.seed for record in summary.records],
                "policy_version": policy_version,
                "policy_gradient_loss": policy_gradient_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "loss": loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "explained_variance": explained_variance,
                "training_scenario_coverage": curr_coverage,
                "curriculum_stage": active_stage,
                "evaluation_seed_set_id": compute_seed_set_id(
                    [record.seed for record in summary.records]
                ),
                "evaluation_seed_count": len(summary.records),
                "environment_config_hash": compute_env_config_hash(self.config.to_dict()["environment"]),
                "evaluation_timestamp": recorded_time,
                "evaluation_id": summary.evaluation_id,
                "evaluation_mode": summary.evaluation_mode,
                "promotion_eligible": summary.promotion_eligible,
                **runtime_snapshot,
            }
            checkpoint_payloads = self._merge_checkpoint(checkpoint_payloads, checkpoint_payload)
            # Persist state after every checkpoint so progress survives a
            # restart or reboot mid-run.  Without this, only a fully-completed
            # batch is durable; every in-flight episode is lost on crash.
            intermediate_state: dict[str, Any] = {
                "total_episodes_trained": total_eps_this_checkpoint,
                "total_environment_episodes": environment_episodes_total,
                "total_timesteps": cumulative_ts,
                "policy_state_path": str(saved_model_path),
                "best_model_path": str(tracked_best_model_path)
                if tracked_best_model_path
                else None,
                "best_model_curriculum_stage": best_model_curriculum_stage,
                "best_success_rate": best_success_rate,
                "best_average_reward": best_average_reward,
                "best_completion_steps": best_completion_steps,
                "policy_config": policy.config.to_dict(),
                "training_signature": self._training_signature(),
                "incomplete_batch": {
                    "batch_completed_segments": completed_checkpoints,
                    "batch_total_segments": n_checkpoints,
                    "batch_steps_per_segment": uniform_segment_timesteps,
                    "batch_timestep_targets": list(timestep_targets),
                    "batch_starting_checkpoint_episode": batch_starting_checkpoint_episode,
                },
                "run_id": self._loaded_state.get("run_id"),
                "parent_run_id": self._loaded_state.get("parent_run_id"),
                "parent_checkpoint_id": self._loaded_state.get("parent_checkpoint_id"),
                "policy_version": policy_version,
                "training_scenario_coverage": curr_coverage,
            }
            state_export_phase = (
                self.runtime_tracker.phase("checkpoint_save")
                if self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            with state_export_phase:
                atomic_write_json(self._state_path, intermediate_state)
                self._loaded_state = intermediate_state
                # Keep history current so a resumed run retains every checkpoint.
                try:
                    self._export_neural_training_summary(
                        checkpoint_payloads,
                        str(saved_model_path),
                        total_eps_this_checkpoint,
                        policy.config.to_dict(),
                    )
                    self._export_web_assets(
                        web_export_dir,
                        checkpoint_payloads,
                        summary,
                        representative_replay_path,
                        checkpoint_path,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Optional summary/web asset export failed: %s", exc, exc_info=True
                    )
            if self.should_evaluate_saved_scenarios(
                completed_checkpoints, n_checkpoints
            ):
                self._evaluate_saved_scenarios(policy, total_eps_this_checkpoint)
            emit(
                {
                    "phase": "checkpoint",
                    "batch_completed_episodes": completed_checkpoints,
                    "current_episode": total_eps_this_checkpoint,
                    "total_episodes_trained": total_eps_this_checkpoint,
                    "checkpoint_episode": total_eps_this_checkpoint,
                    "checkpoint_path": str(checkpoint_path),
                    "replay_path": (
                        str(representative_replay_path)
                        if representative_replay_path is not None
                        else None
                    ),
                    "summary": summary.to_dict(),
                    "best_score": summary.average_reward,
                    "message": f"Checkpoint {total_eps_this_checkpoint} exported",
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "explained_variance": explained_variance,
                    "total_timesteps": cumulative_ts,
                    "batch_total_timesteps": timestep_targets[-1],
                    "batch_completed_timesteps": timestep_targets[
                        completed_checkpoints - 1
                    ],
                    "environment_episodes_total": environment_episodes_total,
                    "environment_episodes_since_run_start": (
                        environment_episodes_since_run_start
                    ),
                    "policy_version": policy_version,
                    "last_policy_update_time": datetime.now(UTC).isoformat(),
                    "last_evaluation_time": datetime.now(UTC).isoformat(),
                    "run_id": self._loaded_state.get("run_id"),
                }
            )

        if interrupted:
            return NeuralTrainingRunSummary(
                checkpoints=checkpoint_payloads,
                final_model_path=str(saved_model_path) if saved_model_path is not None else "",
                policy_config=policy.config.to_dict(),
            )

        if checkpoint_payloads and checkpoint_payloads[-1].get("total_training_episodes", 0) > 0:
            total_episodes_trained = int(checkpoint_payloads[-1]["total_training_episodes"])
        else:
            last_ep = (
                train_config.checkpoint_episodes[-1]
                if train_config.checkpoint_episodes
                else 0
            )
            if last_ep > 0:
                skipped_ep = (
                    train_config.checkpoint_episodes[skip_segments - 1]
                    if (
                        skip_segments > 0
                        and skip_segments <= len(train_config.checkpoint_episodes)
                    )
                    else 0
                )
                total_episodes_trained = starting_total + (last_ep - skipped_ep)
            else:
                total_episodes_trained = starting_total + max(1, batch_total - skip_segments)

        final_model_path_str = str(saved_model_path) if saved_model_path is not None else ""
        total_environment_episodes = (
            int(checkpoint_payloads[-1].get("environment_episodes_total", 0))
            if checkpoint_payloads
            else starting_environment_episodes
        )
        state_payload = {
            "total_episodes_trained": total_episodes_trained,
            "total_environment_episodes": total_environment_episodes,
            "total_timesteps": starting_total_timesteps
            + timestep_targets[-1]
            - (timestep_targets[skip_segments - 1] if skip_segments else 0),
            "policy_state_path": final_model_path_str,
            "best_model_path": str(tracked_best_model_path) if tracked_best_model_path else None,
            "best_model_curriculum_stage": best_model_curriculum_stage,
            "best_success_rate": best_success_rate,
            "best_average_reward": best_average_reward,
            "best_completion_steps": best_completion_steps,
            "policy_config": policy.config.to_dict(),
            "training_signature": self._training_signature(),
            "incomplete_batch": None,
            "run_id": self._loaded_state.get("run_id"),
            "parent_run_id": self._loaded_state.get("parent_run_id"),
            "parent_checkpoint_id": self._loaded_state.get("parent_checkpoint_id"),
            "policy_version": policy_version,
        }
        atomic_write_json(self._state_path, state_payload)
        self._loaded_state = state_payload
        try:
            self._export_neural_training_summary(
                checkpoint_payloads,
                final_model_path_str,
                total_episodes_trained,
                policy.config.to_dict(),
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Optional summary export at batch completion failed: %s", exc, exc_info=True
            )
        emit(
            {
                "phase": "complete",
                "batch_completed_episodes": train_config.checkpoint_episodes[-1]
                if train_config.checkpoint_episodes
                else batch_total,
                "current_episode": checkpoint_payloads[-1]["checkpoint_episode"]
                if checkpoint_payloads
                else None,
                "total_episodes_trained": total_episodes_trained,
                "cumulative_environment_episodes": total_environment_episodes,
                "checkpoint_episode": checkpoint_payloads[-1]["checkpoint_episode"]
                if checkpoint_payloads
                else None,
                "checkpoint_path": None,
                "replay_path": checkpoint_payloads[-1]["replay"] if checkpoint_payloads else None,
                "summary": None,
                "best_score": checkpoint_payloads[-1]["average_reward"]
                if checkpoint_payloads
                else None,
                "message": "Training complete",
            }
        )
        return NeuralTrainingRunSummary(
            checkpoints=checkpoint_payloads,
            final_model_path=final_model_path_str,
            policy_config=policy.config.to_dict(),
        )

    def _export_neural_training_summary(
        self,
        checkpoints: list[dict[str, Any]],
        final_model_path: str,
        total_episodes_trained: int,
        policy_config: dict[str, Any],
    ) -> None:
        records_start = max(0, len(checkpoints) - 3)
        lightweight_checkpoints = []
        for idx, cp in enumerate(checkpoints):
            cp_copy = dict(cp)
            if idx < records_start:
                cp_copy.pop("records", None)
                cp_copy.pop("training_scenario_coverage", None)
            lightweight_checkpoints.append(cp_copy)

        payload = {
            "checkpoints": lightweight_checkpoints,
            "final_model_path": final_model_path,
            "policy_config": policy_config,
            "trainer_type": "maskable_ppo",
            "policy_type": "neural",
            "policy_mode": "neural_policy",
            "replay_mode": "neural_ppo",
            "total_episodes_trained": total_episodes_trained,
        }
        atomic_write_json(self.output_root / "training-summary.json", payload)
