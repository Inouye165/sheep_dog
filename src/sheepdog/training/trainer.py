"""Hill-climbing trainer for the shared dog policy."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.checkpoints.store import CheckpointMetadata, CheckpointStore
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy


def _checkpoint_replay_mode(policy_name: str, total_training_episodes: int) -> str:
    """Return a truthfulness label for exported checkpoints."""

    if policy_name == "neural_policy":
        return "neural_ppo"
    if policy_name == "trained_policy" and total_training_episodes > 0:
        return "trained_linear"
    return "baseline"


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    """Metadata for a training run."""

    checkpoints: list[dict[str, Any]]
    final_weights: PolicyWeights

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoints": self.checkpoints, "final_weights": asdict(self.final_weights)}


@dataclass(frozen=True, slots=True)
class CandidateEvaluationSummary:
    """Aggregate score components for one candidate policy."""

    seeds: tuple[int, ...]
    average_reward: float
    success_rate: float
    timeout_rate: float
    stopped_rate: float
    average_sheep_penned: float
    average_distance_to_pen: float
    average_flock_spread: float

    @property
    def score(self) -> float:
        return (
            self.average_reward
            + self.success_rate * 20.0
            + self.average_sheep_penned * 2.5
            - self.timeout_rate * 8.0
            - self.stopped_rate * 5.0
            - self.average_distance_to_pen * 0.12
            - self.average_flock_spread * 0.35
        )


class Trainer:
    """Train and evaluate checkpoints with honest, deterministic metrics."""

    STATE_FILENAME = "training-state.json"

    def __init__(self, config: LabConfig, output_root: str | Path) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_store = CheckpointStore(self.output_root / "checkpoints")
        self.evaluator = Evaluator(config, self.output_root / "evaluations")
        self._state_path = self.output_root / self.STATE_FILENAME
        self._loaded_state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"total_episodes_trained": 0, "weights": None, "best_score": None}
        try:
            with self._state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"total_episodes_trained": 0, "weights": None, "best_score": None}
        return {
            "total_episodes_trained": int(payload.get("total_episodes_trained", 0)),
            "weights": payload.get("weights"),
            "best_score": payload.get("best_score"),
        }

    def _save_state(self, total: int, weights: PolicyWeights, best_score: float) -> None:
        payload = {
            "total_episodes_trained": total,
            "weights": asdict(weights),
            "best_score": best_score,
        }
        with self._state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @property
    def total_episodes_trained(self) -> int:
        return self._loaded_state.get("total_episodes_trained", 0)

    def _candidate_evaluation_seeds(self) -> tuple[int, ...]:
        configured = tuple(self.config.training.candidate_evaluation_seeds)
        if configured:
            return configured
        return (self.config.training.evaluation_seed,)

    def _evaluate_candidate(self, policy: TrainableLinearPolicy) -> CandidateEvaluationSummary:
        results = [
            SheepdogEnvironment(self.config).run_policy(
                policy,
                seed=seed,
                capture_replay=False,
            )
            for seed in self._candidate_evaluation_seeds()
        ]
        return CandidateEvaluationSummary(
            seeds=tuple(result.seed for result in results),
            average_reward=fmean(result.stats.reward_total for result in results),
            success_rate=fmean(1.0 if result.stats.success else 0.0 for result in results),
            timeout_rate=fmean(1.0 if result.stats.timeout else 0.0 for result in results),
            stopped_rate=fmean(1.0 if result.stats.stopped else 0.0 for result in results),
            average_sheep_penned=fmean(result.stats.sheep_penned for result in results),
            average_distance_to_pen=fmean(
                result.final_snapshot.average_distance_to_pen for result in results
            ),
            average_flock_spread=fmean(result.final_snapshot.flock_spread for result in results),
        )

    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TrainingRunSummary:
        train_config: TrainingConfig = self.config.training
        rng = random.Random(train_config.train_seed + self.total_episodes_trained)
        starting_total = self.total_episodes_trained
        starting_weights = PolicyWeights.from_dict(self._loaded_state.get("weights"))
        best_policy = TrainableLinearPolicy(weights=starting_weights)
        best_score: float | None = self._loaded_state.get("best_score")
        if best_score is None:
            best_score = self._evaluate_candidate(best_policy).score
        checkpoint_payloads: list[dict[str, Any]] = list(self._load_summary_checkpoints())
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)
        batch_total = train_config.episodes + 1
        candidate_pool_size = train_config.candidate_pool_size

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
                "checkpoint_path": None,
                "replay_path": None,
                "best_score": best_score,
                "message": f"Resuming from {starting_total} episodes",
            }
        )

        for episode in range(batch_total):
            cumulative_episode = starting_total + episode
            if episode > 0:
                candidate_policies = [
                    TrainableLinearPolicy(
                        weights=best_policy.weights.mutated(rng, train_config.mutation_scale)
                    )
                    for _ in range(candidate_pool_size)
                ]
                for candidate_policy in candidate_policies:
                    candidate_score = self._evaluate_candidate(candidate_policy).score
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_policy = candidate_policy

            emit(
                {
                    "phase": "training",
                    "batch_completed_episodes": episode + 1,
                    "current_episode": cumulative_episode,
                    "total_episodes_trained": cumulative_episode,
                    "checkpoint_episode": None,
                    "best_score": best_score,
                    "message": f"Episode {cumulative_episode} (batch {episode + 1}/{batch_total})",
                }
            )

            if episode in train_config.checkpoint_episodes:
                summary, evaluation_json, _csv_path = self.evaluator.evaluate(
                    best_policy,
                    train_config.evaluation_seeds,
                    checkpoint_episode=cumulative_episode,
                )
                representative_replay_path = Path(summary.records[0].replay_path)
                metadata = CheckpointMetadata(
                    checkpoint_episode=cumulative_episode,
                    total_training_episodes=cumulative_episode,
                    policy_name=best_policy.name,
                    trainer_type="hill_climb",
                    policy_type="linear",
                    seed=train_config.train_seed,
                    success_rate=summary.success_rate,
                    average_completion_steps=summary.average_completion_steps,
                    timeout_rate=summary.timeout_rate,
                    average_sheep_penned=summary.average_sheep_penned,
                    average_reward=summary.average_reward,
                    environment_config=asdict(self.config.environment),
                    reward_config=asdict(self.config.rewards),
                    policy_weights=asdict(best_policy.weights),
                    evaluation_replay_path=str(representative_replay_path),
                )
                checkpoint_path = self.checkpoint_store.write(metadata)
                checkpoint_payload = {
                    "checkpoint_episode": cumulative_episode,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": checkpoint_path.name,
                    "evaluation": evaluation_json.name,
                    "replay": str(representative_replay_path),
                    "policy_name": best_policy.name,
                    "trainer_type": "hill_climb",
                    "policy_type": "linear",
                    "policy_mode": best_policy.name,
                    "replay_mode": _checkpoint_replay_mode(best_policy.name, cumulative_episode),
                    "total_training_episodes": cumulative_episode,
                    "success_rate": summary.success_rate,
                    "timeout_rate": summary.timeout_rate,
                    "average_completion_steps": summary.average_completion_steps,
                    "average_completion_seconds": summary.average_completion_seconds,
                    "average_sheep_penned": summary.average_sheep_penned,
                    "average_reward": summary.average_reward,
                    "average_distance_to_pen": summary.average_distance_to_pen,
                    "average_flock_spread": summary.average_flock_spread,
                    "environment_config": asdict(self.config.environment),
                    "reward_config": asdict(self.config.rewards),
                    "records": [record.to_dict() for record in summary.records],
                }
                checkpoint_payloads = self._merge_checkpoint(
                    checkpoint_payloads, checkpoint_payload
                )
                self._save_state(cumulative_episode, best_policy.weights, best_score)
                self._loaded_state = {
                    "total_episodes_trained": cumulative_episode,
                    "weights": asdict(best_policy.weights),
                    "best_score": best_score,
                }
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
                        "batch_completed_episodes": episode + 1,
                        "current_episode": cumulative_episode,
                        "total_episodes_trained": cumulative_episode,
                        "checkpoint_episode": cumulative_episode,
                        "checkpoint_path": str(checkpoint_path),
                        "replay_path": str(representative_replay_path),
                        "summary": summary.to_dict(),
                        "best_score": best_score,
                        "message": f"Checkpoint {cumulative_episode} exported",
                    }
                )

        final_total = starting_total + max(0, batch_total - 1)
        self._save_state(final_total, best_policy.weights, best_score)
        self._loaded_state = {
            "total_episodes_trained": final_total,
            "weights": asdict(best_policy.weights),
            "best_score": best_score,
        }
        self._export_training_summary(checkpoint_payloads, best_policy.weights, final_total)
        emit(
            {
                "phase": "complete",
                "batch_completed_episodes": batch_total,
                "current_episode": final_total,
                "total_episodes_trained": final_total,
                "checkpoint_episode": checkpoint_payloads[-1]["checkpoint_episode"]
                if checkpoint_payloads
                else None,
                "checkpoint_path": None,
                "replay_path": checkpoint_payloads[-1]["replay"] if checkpoint_payloads else None,
                "best_score": best_score,
                "message": "Training complete",
            }
        )
        return TrainingRunSummary(
            checkpoints=checkpoint_payloads, final_weights=best_policy.weights
        )

    def _load_summary_checkpoints(self) -> list[dict[str, Any]]:
        path = self.output_root / "training-summary.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return list(payload.get("checkpoints", []))

    def _merge_checkpoint(
        self,
        checkpoints: list[dict[str, Any]],
        new_entry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        filtered = [
            entry
            for entry in checkpoints
            if entry.get("checkpoint_episode") != new_entry["checkpoint_episode"]
        ]
        filtered.append(new_entry)
        filtered.sort(key=lambda entry: entry.get("checkpoint_episode", 0))
        return filtered

    def _export_training_summary(
        self,
        checkpoints: list[dict[str, Any]],
        weights: PolicyWeights,
        total_episodes_trained: int,
    ) -> None:
        payload = {
            "checkpoints": checkpoints,
            "final_weights": asdict(weights),
            "trainer_type": "hill_climb",
            "policy_type": "linear",
            "policy_mode": "trained_policy" if total_episodes_trained > 0 else "instinct_only",
            "replay_mode": "trained_linear" if total_episodes_trained > 0 else "baseline",
            "total_episodes_trained": total_episodes_trained,
        }
        path = self.output_root / "training-summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _export_web_assets(
        self,
        web_export_dir: Path,
        checkpoint_payloads: list[dict[str, Any]],
        summary: object,
        replay_path: Path,
        checkpoint_path: Path,
    ) -> None:
        summary_payload = summary.to_dict()
        replay_output_dir = web_export_dir / "replays"
        replay_output_dir.mkdir(parents=True, exist_ok=True)

        def _export_record(record: dict[str, Any]) -> dict[str, Any]:
            source_path = Path(record["replay_path"])
            target_path = replay_output_dir / source_path.name
            if source_path.exists():
                target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            exported_record = dict(record)
            exported_record["replay_path"] = f"/generated/replays/{target_path.name}"
            return exported_record

        exported_summary_records = [_export_record(record) for record in summary_payload["records"]]
        summary_payload["records"] = exported_summary_records

        exported_checkpoints: list[dict[str, Any]] = []
        for checkpoint_payload in checkpoint_payloads:
            exported_payload = dict(checkpoint_payload)
            exported_payload["records"] = [
                _export_record(record) for record in checkpoint_payload["records"]
            ]
            exported_checkpoints.append(exported_payload)

        with (web_export_dir / "latest-checkpoint.json").open("w", encoding="utf-8") as handle:
            json.dump({"checkpoint": checkpoint_path.name}, handle, indent=2)
        with (web_export_dir / "latest-evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
        with (web_export_dir / "checkpoint-index.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"checkpoints": exported_checkpoints, "latest": summary_payload}, handle, indent=2
            )
        replay_target = web_export_dir / "latest-replay.json"
        replay_target.write_text(replay_path.read_text(encoding="utf-8"), encoding="utf-8")
