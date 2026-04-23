"""MaskablePPO trainer for the experimental neural-policy path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sheepdog.checkpoints.store import CheckpointMetadata
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


class MaskablePPOTrainer(Trainer):
    """Train the shared role-aware neural policy with MaskablePPO."""

    MODEL_DIRNAME = "models"

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

    def train(self, progress_callback=None) -> NeuralTrainingRunSummary:  # type: ignore[override]
        del progress_callback
        train_config = self.config.training
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)
        model_root = self.output_root / self.MODEL_DIRNAME
        model_root.mkdir(parents=True, exist_ok=True)
        policy_state_path = self._loaded_state.get("policy_state_path")
        if policy_state_path:
            policy = NeuralPolicy.load(
                policy_state_path,
                self.config,
                self._loaded_state.get("policy_config"),
            )
        else:
            policy = NeuralPolicy.initialize(self.config)
        training_env = SheepdogRLAdapter(self.config)
        policy.model.set_env(training_env)
        policy.model.learn(total_timesteps=train_config.total_timesteps, progress_bar=False)

        checkpoint_payloads: list[dict[str, Any]] = list(self._load_summary_checkpoints())
        final_model_path = model_root / f"maskable-ppo-{train_config.total_timesteps:08d}"
        saved_model_path = policy.save(final_model_path)

        for checkpoint_episode in train_config.checkpoint_episodes:
            summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                policy,
                train_config.evaluation_seeds,
                checkpoint_episode=checkpoint_episode,
            )
            representative_replay_path = Path(summary.records[0].replay_path)
            metadata = CheckpointMetadata(
                checkpoint_episode=checkpoint_episode,
                total_training_episodes=(
                    self.total_episodes_trained + len(train_config.checkpoint_episodes)
                ),
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
                "total_training_episodes": (
                    self.total_episodes_trained + len(train_config.checkpoint_episodes)
                ),
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

        total_episodes_trained = self.total_episodes_trained + len(train_config.checkpoint_episodes)
        state_payload = {
            "total_episodes_trained": total_episodes_trained,
            "total_timesteps": (
                int(self._loaded_state.get("total_timesteps", 0))
                + train_config.total_timesteps
            ),
            "policy_state_path": str(saved_model_path),
            "policy_config": policy.config.to_dict(),
        }
        self._state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        self._loaded_state = state_payload
        self._export_training_summary(
            checkpoint_payloads,
            str(saved_model_path),
            total_episodes_trained,
            policy.config.to_dict(),
        )
        return NeuralTrainingRunSummary(
            checkpoints=checkpoint_payloads,
            final_model_path=str(saved_model_path),
            policy_config=policy.config.to_dict(),
        )

    def _export_training_summary(
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