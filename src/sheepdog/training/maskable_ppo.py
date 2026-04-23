"""MaskablePPO trainer for the experimental neural-policy path."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
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
    ) -> None:
        super().__init__()
        self._emit = emit
        self._report_interval = max(1, report_interval)
        self._total_timesteps = max(1, total_timesteps)
        self._starting_total_episodes = starting_total_episodes
        self._batch_total_episodes = batch_total_episodes
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
        self._emit(
            {
                "phase": "learning",
                "batch_completed_episodes": 0,
                "current_episode": None,
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

    def _has_compatible_policy_state(self) -> bool:
        policy_config = self._loaded_state.get("policy_config")
        if not isinstance(policy_config, dict):
            return False
        return int(policy_config.get("action_size", 0)) == len(ACTION_ORDER)

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "total_episodes_trained": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
            }
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "total_episodes_trained": 0,
                "total_timesteps": 0,
                "policy_state_path": None,
                "policy_config": None,
            }
        return {
            "total_episodes_trained": int(payload.get("total_episodes_trained", 0)),
            "total_timesteps": int(payload.get("total_timesteps", 0)),
            "policy_state_path": payload.get("policy_state_path"),
            "policy_config": payload.get("policy_config"),
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
        resuming_policy = bool(self._loaded_state.get("policy_state_path")) and self._has_compatible_policy_state()
        starting_total = self.total_episodes_trained if resuming_policy else 0
        batch_total = max(1, len(train_config.checkpoint_episodes))

        def emit(payload: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            payload.setdefault("batch_total_episodes", batch_total)
            payload.setdefault("starting_total_episodes", starting_total)
            progress_callback(payload)

        emit(
            {
                "phase": "starting",
                "batch_completed_episodes": 0,
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

        policy_state_path = self._loaded_state.get("policy_state_path")
        if resuming_policy and policy_state_path:
            policy = NeuralPolicy.load(
                policy_state_path,
                self.config,
                self._loaded_state.get("policy_config"),
            )
        else:
            policy = NeuralPolicy.initialize(self.config)
        training_env = SheepdogRLAdapter(self.config)
        policy.model.set_env(training_env)
        emit(
            {
                "phase": "learning",
                "batch_completed_episodes": 0,
                "current_episode": None,
                "total_episodes_trained": starting_total,
                "checkpoint_episode": None,
                "best_score": None,
                "message": (
                    f"Learning neural policy: 0/{train_config.total_timesteps} timesteps (0%)"
                ),
            }
        )
        progress_reporter = _TrainingProgressCallback(
            emit,
            report_interval=max(1, train_config.total_timesteps // 4),
            total_timesteps=train_config.total_timesteps,
            starting_total_episodes=starting_total,
            batch_total_episodes=batch_total,
        )
        policy.model.learn(
            total_timesteps=train_config.total_timesteps,
            progress_bar=False,
            callback=progress_reporter,
        )

        checkpoint_payloads: list[dict[str, Any]] = (
            list(self._load_summary_checkpoints()) if resuming_policy else []
        )
        final_model_path = model_root / f"maskable-ppo-{train_config.total_timesteps:08d}"
        saved_model_path = policy.save(final_model_path)

        for completed_checkpoints, checkpoint_episode in enumerate(
            train_config.checkpoint_episodes,
            start=1,
        ):
            summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                policy,
                train_config.evaluation_seeds,
                checkpoint_episode=checkpoint_episode,
            )
            representative_replay_path = Path(summary.records[0].replay_path)
            metadata = CheckpointMetadata(
                checkpoint_episode=checkpoint_episode,
                total_training_episodes=(starting_total + len(train_config.checkpoint_episodes)),
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
            checkpoint_path = self.checkpoint_store.write(metadata)
            checkpoint_payload = {
                "checkpoint_episode": checkpoint_episode,
                "checkpoint": checkpoint_path.name,
                "evaluation": evaluation_json.name,
                "replay": str(representative_replay_path),
                "policy_name": policy.name,
                "trainer_type": policy.trainer_type,
                "policy_type": policy.policy_type,
                "policy_mode": policy.name,
                "replay_mode": "neural_ppo",
                "total_training_episodes": (starting_total + len(train_config.checkpoint_episodes)),
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
                    "current_episode": checkpoint_episode,
                    "total_episodes_trained": starting_total + completed_checkpoints,
                    "checkpoint_episode": checkpoint_episode,
                    "checkpoint_path": str(checkpoint_path),
                    "replay_path": str(representative_replay_path),
                    "summary": summary.to_dict(),
                    "best_score": summary.average_reward,
                    "message": f"Checkpoint {checkpoint_episode} exported",
                }
            )

        total_episodes_trained = starting_total + batch_total
        starting_total_timesteps = int(self._loaded_state.get("total_timesteps", 0)) if resuming_policy else 0
        state_payload = {
            "total_episodes_trained": total_episodes_trained,
            "total_timesteps": starting_total_timesteps + train_config.total_timesteps,
            "policy_state_path": str(saved_model_path),
            "policy_config": policy.config.to_dict(),
        }
        self._state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        self._loaded_state = state_payload
        self._export_neural_training_summary(
            checkpoint_payloads,
            str(saved_model_path),
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
                "best_score": checkpoint_payloads[-1]["average_reward"] if checkpoint_payloads else None,
                "message": "Training complete",
            }
        )
        return NeuralTrainingRunSummary(
            checkpoints=checkpoint_payloads,
            final_model_path=str(saved_model_path),
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