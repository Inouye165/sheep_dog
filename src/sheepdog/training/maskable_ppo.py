"""MaskablePPO trainer for the experimental neural-policy path."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from sheepdog.checkpoints.store import CheckpointMetadata
from sheepdog.environment import ACTION_ORDER
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.rl_env import SheepdogRLAdapter
from sheepdog.training.trainer import Trainer


@dataclass(frozen=True, slots=True)
class NeuralTrainingRunSummary:
    """Metadata for a neural PPO training run."""

    checkpoints: list[dict[str, Any]]
    final_model_path: str
    policy_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
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
        report_interval: int,
        total_timesteps: int,
        starting_total_episodes: int,
        batch_total_episodes: int,
        completed_segments: int,
        segment_index: int,
    ) -> None:
        super().__init__()
        self._emit = emit
        self._report_interval = max(1, report_interval)
        self._total_timesteps = max(1, total_timesteps)
        self._starting_total_episodes = starting_total_episodes
        self._batch_total_episodes = batch_total_episodes
        self._completed_segments = completed_segments
        self._segment_index = segment_index
        self._last_reported_steps = 0

    def _on_step(self) -> bool:
        num_timesteps = int(self.num_timesteps)
        if (
            num_timesteps < self._total_timesteps
            and num_timesteps - self._last_reported_steps < self._report_interval
        ):
            return True
        self._last_reported_steps = num_timesteps
        completion = min(1.0, num_timesteps / self._total_timesteps)
        fractional_completed = self._completed_segments + completion
        self._emit(
            {
                "phase": "learning",
                "batch_completed_episodes": fractional_completed,
                "current_episode": self._segment_index,
                "total_episodes_trained": self._starting_total_episodes,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"Learning neural policy: {num_timesteps}/{self._total_timesteps} "
                    f"timesteps ({completion:.0%})"
                ),
                "batch_total_episodes": self._batch_total_episodes,
            }
        )
        return True


class MaskablePPOTrainer(Trainer):
    """Train the shared role-aware neural policy with MaskablePPO."""

    MODEL_DIRNAME = "models"

    def _training_signature(self) -> dict[str, Any]:
        # Deliberately excludes ``dogs`` and ``sheep`` counts: the observation
        # vector is now padded to fixed MAX_SHEEP_SLOTS / HERD_DOG_SLOTS slots so
        # the architecture is stable across curriculum promotions.  Speeds and
        # sprint multiplier are kept because they change training dynamics.
        return {
            "action_size": len(ACTION_ORDER),
            "rewards": asdict(self.config.rewards),
            "environment": {
                "dog_speed": self.config.environment.dog_speed,
                "dog_sprint_multiplier": self.config.environment.dog_sprint_multiplier,
                "sheep_speed": self.config.environment.sheep_speed,
            },
        }

    @staticmethod
    def _strip_non_architectural_fields(sig: dict[str, Any]) -> dict[str, Any]:
        """Return a copy with stage/toggle metadata removed before comparison.

        ``curriculum_stage``, ``debug_reward_breakdown``, and
        ``enable_instinct_rewards`` are runtime toggles that do not affect the
        neural-network architecture; stripping them lets stage promotions reuse
        the trained model and accumulate ``total_episodes_trained`` correctly.
        """
        sig = dict(sig)
        if "rewards" in sig:
            rewards = dict(sig["rewards"])
            instincts = dict(rewards.get("instincts", {}))
            for key in ("curriculum_stage", "debug_reward_breakdown", "enable_instinct_rewards"):
                instincts.pop(key, None)
            rewards["instincts"] = instincts
            sig["rewards"] = rewards
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
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
                "incomplete_batch": None,
            }
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "total_episodes_trained": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
                "incomplete_batch": None,
            }
        return {
            "total_episodes_trained": int(payload.get("total_episodes_trained", 0)),
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
        }

    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> NeuralTrainingRunSummary:  # type: ignore[override]
        train_config = self.config.training
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)
        model_root = self.output_root / self.MODEL_DIRNAME
        model_root.mkdir(parents=True, exist_ok=True)
        resuming_policy = (
            bool(self._loaded_state.get("policy_state_path"))
            and self._has_compatible_policy_state()
        )
        starting_total = self.total_episodes_trained if resuming_policy else 0
        batch_total = max(1, len(train_config.checkpoint_episodes))
        n_checkpoints = batch_total
        steps_per_segment = max(1, train_config.total_timesteps // n_checkpoints)
        starting_total_timesteps = (
            int(self._loaded_state.get("total_timesteps", 0)) if resuming_policy else 0
        )
        # Resume an interrupted batch when the new job has the same segment count.
        _incomplete = self._loaded_state.get("incomplete_batch") if resuming_policy else None
        skip_segments: int = 0
        if (
            isinstance(_incomplete, dict)
            and int(_incomplete.get("batch_total_segments", 0)) == n_checkpoints
            and int(_incomplete.get("batch_steps_per_segment", 0)) == steps_per_segment
            and 0 < int(_incomplete.get("batch_completed_segments", 0)) < n_checkpoints
        ):
            skip_segments = int(_incomplete["batch_completed_segments"])
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
        training_env = SheepdogRLAdapter(self.config)
        policy.model.set_env(training_env)

        checkpoint_payloads: list[dict[str, Any]] = (
            list(self._load_summary_checkpoints()) if resuming_policy else []
        )
        saved_model_path: Path | None = None

        # Stamp an in-progress batch marker before the loop so a crash mid-batch
        # is resumable on the next call with the same episode count.
        _batch_marker_pre: dict[str, Any] = {
            "batch_completed_segments": skip_segments,
            "batch_total_segments": n_checkpoints,
            "batch_steps_per_segment": steps_per_segment,
        }
        _pre_loop_state = dict(self._loaded_state)
        _pre_loop_state["incomplete_batch"] = _batch_marker_pre
        self._state_path.write_text(json.dumps(_pre_loop_state, indent=2), encoding="utf-8")

        for completed_checkpoints, _checkpoint_slot in enumerate(
            train_config.checkpoint_episodes,
            start=1,
        ):
            if completed_checkpoints <= skip_segments:
                continue
            new_segments = completed_checkpoints - skip_segments
            cumulative_ts = starting_total_timesteps + new_segments * steps_per_segment
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
                        f" (0/{steps_per_segment} timesteps)"
                    ),
                }
            )
            progress_reporter = _TrainingProgressCallback(
                emit,
                report_interval=max(1, steps_per_segment // 4),
                total_timesteps=steps_per_segment,
                starting_total_episodes=starting_total + new_segments - 1,
                batch_total_episodes=batch_total,
                completed_segments=completed_checkpoints - 1,
                segment_index=completed_checkpoints - 1,
            )
            # Linear LR annealing across the batch: full LR at segment 0,
            # learning_rate_final at the last segment.  This keeps updates
            # aggressive early (fast cliff recovery) and conservative late
            # (prevents policy collapse near the end of a run).
            _batch_done = completed_checkpoints - skip_segments - 1
            _batch_span = max(1, n_checkpoints - skip_segments - 1)
            _batch_progress = _batch_done / _batch_span
            policy.model.learning_rate = (
                train_config.learning_rate
                + (train_config.learning_rate_final - train_config.learning_rate) * _batch_progress
            )
            policy.model.learn(
                total_timesteps=steps_per_segment,
                reset_num_timesteps=True,
                progress_bar=False,
                callback=progress_reporter,
            )
            saved_model_path = policy.save(model_root / f"maskable-ppo-{cumulative_ts:08d}")
            total_eps_this_checkpoint = starting_total + _checkpoint_slot
            summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                policy,
                train_config.evaluation_seeds,
                checkpoint_episode=total_eps_this_checkpoint,
            )
            representative_replay_path = Path(summary.records[0].replay_path)
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
                evaluation_replay_path=str(representative_replay_path),
            )
            current_stage = self.config.rewards.instincts.curriculum_stage
            is_new_best = (
                # A higher curriculum stage always beats a lower one.
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
            if is_new_best:
                best_success_rate = summary.success_rate
                best_average_reward = summary.average_reward
                best_completion_steps = summary.average_completion_steps
                best_model_curriculum_stage = current_stage
                tracked_best_model_path = policy.save(model_root / "best-model")
            checkpoint_path = self.checkpoint_store.write(metadata)
            checkpoint_payload = {
                "checkpoint_episode": total_eps_this_checkpoint,
                "recorded_at": datetime.now(UTC).isoformat(),
                "checkpoint": checkpoint_path.name,
                "evaluation": evaluation_json.name,
                "replay": str(representative_replay_path),
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
            }
            checkpoint_payloads = self._merge_checkpoint(checkpoint_payloads, checkpoint_payload)
            # Persist state after every checkpoint so progress survives a
            # restart or reboot mid-run.  Without this, only a fully-completed
            # batch is durable; every in-flight episode is lost on crash.
            intermediate_state: dict[str, Any] = {
                "total_episodes_trained": total_eps_this_checkpoint,
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
                    "batch_steps_per_segment": steps_per_segment,
                },
            }
            self._state_path.write_text(json.dumps(intermediate_state, indent=2), encoding="utf-8")
            self._loaded_state = intermediate_state
            # Also keep training-summary.json current so that checkpoint
            # history is complete when a run resumes after a crash/reboot.
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
            emit(
                {
                    "phase": "checkpoint",
                    "batch_completed_episodes": completed_checkpoints,
                    "current_episode": total_eps_this_checkpoint,
                    "total_episodes_trained": total_eps_this_checkpoint,
                    "checkpoint_episode": total_eps_this_checkpoint,
                    "checkpoint_path": str(checkpoint_path),
                    "replay_path": str(representative_replay_path),
                    "summary": summary.to_dict(),
                    "best_score": summary.average_reward,
                    "message": f"Checkpoint {total_eps_this_checkpoint} exported",
                }
            )

        total_episodes_trained = starting_total + (batch_total - skip_segments)
        final_model_path_str = str(saved_model_path) if saved_model_path is not None else ""
        state_payload = {
            "total_episodes_trained": total_episodes_trained,
            "total_timesteps": starting_total_timesteps
            + (batch_total - skip_segments) * steps_per_segment,
            "policy_state_path": final_model_path_str,
            "best_model_path": str(tracked_best_model_path) if tracked_best_model_path else None,
            "best_model_curriculum_stage": best_model_curriculum_stage,
            "best_success_rate": best_success_rate,
            "best_average_reward": best_average_reward,
            "best_completion_steps": best_completion_steps,
            "policy_config": policy.config.to_dict(),
            "training_signature": self._training_signature(),
            "incomplete_batch": None,
        }
        self._state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        self._loaded_state = state_payload
        self._export_neural_training_summary(
            checkpoint_payloads,
            final_model_path_str,
            total_episodes_trained,
            policy.config.to_dict(),
        )
        emit(
            {
                "phase": "complete",
                "batch_completed_episodes": batch_total,
                "current_episode": checkpoint_payloads[-1]["checkpoint_episode"]
                if checkpoint_payloads
                else None,
                "total_episodes_trained": total_episodes_trained,
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
        payload = {
            "checkpoints": checkpoints,
            "final_model_path": final_model_path,
            "policy_config": policy_config,
            "trainer_type": "maskable_ppo",
            "policy_type": "neural",
            "policy_mode": "neural_policy",
            "replay_mode": "neural_ppo",
            "total_episodes_trained": total_episodes_trained,
        }
        path = self.output_root / "training-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
