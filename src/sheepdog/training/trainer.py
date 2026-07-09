"""Hill-climbing trainer for the shared dog policy."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.atomic_io import atomic_write_json, atomic_write_text
from sheepdog.checkpoints.store import CheckpointMetadata, CheckpointStore
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.evaluator import EvaluationSummary, Evaluator
from sheepdog.evaluation.scenario_evaluator import evaluate_checkpoint_on_scenarios
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
        """Return a JSON-serializable summary payload."""
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
    average_steps: float

    @property
    def score(self) -> float:
        """Return the scalar hill-climbing score for this candidate."""
        # Priority 1: success_rate dominates — mirrors is_better_formal criterion.
        # With 5 candidate seeds, the minimum non-zero rate delta is 0.2 (= 2 000 pts),
        # which swamps all other terms combined.
        # Priority 2: fewer steps — faster herding is strictly better (same logic as
        # is_better_formal tiebreaker).
        # Remaining terms supply an exploratory gradient when success_rate == 0
        # across every candidate; average_reward is included at 0.1× so a
        # 100-pt reward swing (≈ 10 pts) cannot override a single-success difference.
        return (
            self.success_rate * 10_000.0
            - self.average_steps * 0.5
            + self.average_reward * 0.1
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
            "hill_climb_curriculum_stage": payload.get("hill_climb_curriculum_stage"),
            "best_formal_episode": payload.get("best_formal_episode"),
            "best_formal_success_rate": payload.get("best_formal_success_rate"),
            "best_formal_avg_reward": payload.get("best_formal_avg_reward"),
            "best_formal_avg_steps": payload.get("best_formal_avg_steps"),
            "best_formal_curriculum_stage": payload.get("best_formal_curriculum_stage"),
            "best_formal_weights": payload.get("best_formal_weights"),
        }

    def _save_state(
        self,
        total: int,
        weights: PolicyWeights,
        best_score: float,
        *,
        hill_climb_curriculum_stage: int | None = None,
        best_formal_episode: int | None = None,
        best_formal_success_rate: float | None = None,
        best_formal_avg_reward: float | None = None,
        best_formal_avg_steps: float | None = None,
        best_formal_curriculum_stage: int | None = None,
        best_formal_weights: PolicyWeights | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "total_episodes_trained": total,
            "weights": asdict(weights),
            "best_score": best_score,
        }
        if hill_climb_curriculum_stage is not None:
            payload["hill_climb_curriculum_stage"] = hill_climb_curriculum_stage
        if best_formal_episode is not None:
            payload["best_formal_episode"] = best_formal_episode
            payload["best_formal_success_rate"] = best_formal_success_rate
            payload["best_formal_avg_reward"] = best_formal_avg_reward
            payload["best_formal_avg_steps"] = best_formal_avg_steps
            payload["best_formal_curriculum_stage"] = best_formal_curriculum_stage
            payload["best_formal_weights"] = (
                asdict(best_formal_weights) if best_formal_weights is not None else None
            )
        atomic_write_json(self._state_path, payload)

    @property
    def total_episodes_trained(self) -> int:
        """Return the number of episodes persisted in trainer state."""
        return self._loaded_state.get("total_episodes_trained", 0)

    def override_start_weights(self, weights_dict: dict[str, Any]) -> None:
        """Replace the loaded weights with *weights_dict* and clear best_score.

        Called by the server when promoting to a new stage with an explicit
        best-formal checkpoint so the trainer starts from that policy rather
        than whatever the hill-climber last wrote to state.
        """
        self._loaded_state["weights"] = weights_dict
        self._loaded_state["best_score"] = None

    @staticmethod
    def _is_strictly_better(
        new_stage: int,
        new_rate: float,
        new_steps: float,
        stored_stage: int | None,
        stored_rate: float | None,
        stored_steps: float | None,
    ) -> bool:
        """Return True if (new_stage, new_rate, new_steps) strictly beats stored.

        Mirrors the frontend isStrictlyBetterCheckpoint ordering:
        higher curriculum stage > higher success_rate > fewer completion steps.
        None stored values are treated as worst-possible so any real result wins.
        """
        if stored_stage is None:
            return True
        if new_stage != stored_stage:
            return new_stage > stored_stage
        _stored_rate = stored_rate if stored_rate is not None else float("-inf")
        _stored_steps = stored_steps if stored_steps is not None else float("inf")
        if new_rate != _stored_rate:
            return new_rate > _stored_rate
        return new_steps < _stored_steps

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
            average_steps=fmean(result.stats.steps for result in results),
        )

    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> TrainingRunSummary:
        """Run hill-climbing training and export evaluated checkpoints."""
        train_config: TrainingConfig = self.config.training
        current_stage: int = self.config.rewards.instincts.curriculum_stage
        # Load checkpoint payloads and reconcile best_formal_weights BEFORE
        # selecting starting weights so any stale state on disk is corrected first.
        checkpoint_payloads: list[dict[str, Any]] = list(self._load_summary_checkpoints())
        self._reconcile_formal_best(current_stage, checkpoint_payloads)
        # Detect stage promotion: if stored weights came from a different stage,
        # seed the new stage from the best formally-evaluated checkpoint rather
        # than the hill-climbing tail, which may have overfit to a small seed set.
        stored_hill_climb_stage: int | None = self._loaded_state.get("hill_climb_curriculum_stage")
        stage_changed = (
            stored_hill_climb_stage is not None
            and stored_hill_climb_stage != current_stage
            and self._loaded_state.get("best_formal_weights") is not None
        )
        rng = random.Random(train_config.train_seed + self.total_episodes_trained)
        starting_total = self.total_episodes_trained
        # Always start from the formally-evaluated best policy when one exists.
        # This prevents the hill-climbing tail from drifting away from the
        # validated best across batches, ensuring each run is anchored to the
        # true best (by rate → fewer-steps) rather than the last mutation.
        formal_weights_dict = self._loaded_state.get("best_formal_weights")
        if stage_changed or formal_weights_dict is not None:
            starting_weights = PolicyWeights.from_dict(self._loaded_state["best_formal_weights"])
        else:
            starting_weights = PolicyWeights.from_dict(self._loaded_state.get("weights"))
        best_policy = TrainableLinearPolicy(weights=starting_weights)
        # Re-evaluate the starting policy so the hill-climbing baseline score is
        # consistent with the actual starting weights (formal best may differ from
        # the tail that produced the stored best_score).
        best_score: float | None = (
            None
            if (stage_changed or formal_weights_dict is not None)
            else self._loaded_state.get("best_score")
        )
        if best_score is None:
            best_score = self._evaluate_candidate(best_policy).score
        # Stochastic restart: when resuming from a formal best, explore a small
        # neighbourhood before episode 0 so consecutive batches don't hill-climb
        # from the exact same weights (local-optima escape).  Only activates on
        # subsequent batches (starting_total > 0) so the very first run gets a
        # clean baseline checkpoint.  The formal best is replaced only if a
        # neighbour genuinely scores higher on candidate seeds.
        if formal_weights_dict is not None and starting_total > 0:
            for _ in range(train_config.candidate_pool_size):
                restart_candidate = TrainableLinearPolicy(
                    weights=best_policy.weights.mutated(rng, train_config.mutation_scale)
                )
                restart_score = self._evaluate_candidate(restart_candidate).score
                if restart_score > best_score:
                    best_score = restart_score
                    best_policy = restart_candidate
        # Carry forward best-formal tracking from persisted state.
        best_formal_episode: int | None = self._loaded_state.get("best_formal_episode")
        best_formal_success_rate: float | None = self._loaded_state.get("best_formal_success_rate")
        best_formal_avg_reward: float | None = self._loaded_state.get("best_formal_avg_reward")
        best_formal_avg_steps: float | None = self._loaded_state.get("best_formal_avg_steps")
        best_formal_curriculum_stage: int | None = self._loaded_state.get(
            "best_formal_curriculum_stage"
        )
        best_formal_weights: PolicyWeights | None = (
            PolicyWeights.from_dict(self._loaded_state["best_formal_weights"])
            if self._loaded_state.get("best_formal_weights") is not None
            else None
        )
        web_export_dir = Path(train_config.web_export_dir)
        web_export_dir.mkdir(parents=True, exist_ok=True)
        batch_total = train_config.episodes + 1
        candidate_pool_size = train_config.candidate_pool_size
        executed_iterations = 0

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
                "seed_episode": best_formal_episode,
                "message": f"Resuming from {starting_total} episodes",
            }
        )

        for episode in range(batch_total):
            if should_stop is not None and should_stop():
                break
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
                    "recorded_at": datetime.now(UTC).isoformat(),
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
                # Update the best-formal pointer when this checkpoint is strictly
                # better, using the same ordering as the frontend's
                # isStrictlyBetterCheckpoint: stage > success_rate > fewer_steps.
                # average_reward is intentionally excluded — fewer steps = strictly better.
                formal_stage = current_stage
                is_better_formal = self._is_strictly_better(
                    formal_stage,
                    summary.success_rate,
                    summary.average_completion_steps,
                    best_formal_curriculum_stage,
                    best_formal_success_rate,
                    best_formal_avg_steps,
                )
                if is_better_formal:
                    best_formal_episode = cumulative_episode
                    best_formal_success_rate = summary.success_rate
                    best_formal_avg_reward = summary.average_reward
                    best_formal_avg_steps = summary.average_completion_steps
                    best_formal_curriculum_stage = formal_stage
                    best_formal_weights = best_policy.weights
                self._save_state(
                    cumulative_episode,
                    best_policy.weights,
                    best_score,
                    hill_climb_curriculum_stage=current_stage,
                    best_formal_episode=best_formal_episode,
                    best_formal_success_rate=best_formal_success_rate,
                    best_formal_avg_reward=best_formal_avg_reward,
                    best_formal_avg_steps=best_formal_avg_steps,
                    best_formal_curriculum_stage=best_formal_curriculum_stage,
                    best_formal_weights=best_formal_weights,
                )
                self._loaded_state = {
                    "total_episodes_trained": cumulative_episode,
                    "weights": asdict(best_policy.weights),
                    "best_score": best_score,
                    "hill_climb_curriculum_stage": current_stage,
                    "best_formal_episode": best_formal_episode,
                    "best_formal_success_rate": best_formal_success_rate,
                    "best_formal_avg_reward": best_formal_avg_reward,
                    "best_formal_avg_steps": best_formal_avg_steps,
                    "best_formal_curriculum_stage": best_formal_curriculum_stage,
                    "best_formal_weights": (
                        asdict(best_formal_weights) if best_formal_weights is not None else None
                    ),
                }
                self._export_web_assets(
                    web_export_dir,
                    checkpoint_payloads,
                    summary,
                    representative_replay_path,
                    checkpoint_path,
                )
                self._evaluate_saved_scenarios(best_policy, cumulative_episode)
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
            executed_iterations += 1

        final_total = starting_total + max(0, batch_total - 1)
        if executed_iterations < batch_total:
            final_total = starting_total + max(0, executed_iterations - 1)
        self._save_state(
            final_total,
            best_policy.weights,
            best_score,
            hill_climb_curriculum_stage=current_stage,
            best_formal_episode=best_formal_episode,
            best_formal_success_rate=best_formal_success_rate,
            best_formal_avg_reward=best_formal_avg_reward,
            best_formal_avg_steps=best_formal_avg_steps,
            best_formal_curriculum_stage=best_formal_curriculum_stage,
            best_formal_weights=best_formal_weights,
        )
        self._loaded_state = {
            "total_episodes_trained": final_total,
            "weights": asdict(best_policy.weights),
            "best_score": best_score,
            "hill_climb_curriculum_stage": current_stage,
            "best_formal_episode": best_formal_episode,
            "best_formal_success_rate": best_formal_success_rate,
            "best_formal_avg_reward": best_formal_avg_reward,
            "best_formal_avg_steps": best_formal_avg_steps,
            "best_formal_curriculum_stage": best_formal_curriculum_stage,
            "best_formal_weights": (
                asdict(best_formal_weights) if best_formal_weights is not None else None
            ),
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

    def _reconcile_formal_best(
        self,
        current_stage: int,
        checkpoint_payloads: list[dict[str, Any]],
    ) -> None:
        """Re-derive best_formal_weights from checkpoint files if stale.

        Scans all checkpoint payloads for *current_stage* using the canonical
        rate → fewer-steps ordering and updates the in-memory loaded-state if a
        better result is found than what is currently stored.  This corrects
        state written under any older scoring criterion (e.g. reward tiebreaker).
        """
        stored_stage = self._loaded_state.get("best_formal_curriculum_stage")
        if stored_stage is not None and stored_stage != current_stage:
            # Formal best is from a different stage; leave it for stage_changed path.
            return
        stored_best_rate: float = self._loaded_state.get("best_formal_success_rate") or -1.0
        stored_best_steps: float = self._loaded_state.get("best_formal_avg_steps") or float("inf")
        best_payload: dict[str, Any] | None = None
        best_rate = stored_best_rate
        best_steps = stored_best_steps
        for payload in checkpoint_payloads:
            payload_stage = (
                payload.get("reward_config", {}).get("instincts", {}).get("curriculum_stage", 0)
            )
            if payload_stage != current_stage:
                continue
            rate: float = payload.get("success_rate") or 0.0
            steps: float = payload.get("average_completion_steps") or float("inf")
            if self._is_strictly_better(
                current_stage, rate, steps, current_stage, best_rate, best_steps
            ):
                best_rate = rate
                best_steps = steps
                best_payload = payload
        if best_payload is None:
            return
        checkpoint_name: str | None = best_payload.get("checkpoint")
        if not checkpoint_name:
            return
        checkpoint_file = self.output_root / "checkpoints" / checkpoint_name
        try:
            data: dict[str, Any] = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        policy_weights = data.get("policy_weights")
        if policy_weights is None:
            return
        self._loaded_state["best_formal_episode"] = best_payload.get("checkpoint_episode")
        self._loaded_state["best_formal_success_rate"] = best_rate
        self._loaded_state["best_formal_avg_steps"] = best_steps
        self._loaded_state["best_formal_avg_reward"] = best_payload.get("average_reward")
        self._loaded_state["best_formal_curriculum_stage"] = current_stage
        self._loaded_state["best_formal_weights"] = policy_weights

    def _merge_checkpoint(
        self,
        checkpoints: list[dict[str, Any]],
        new_entry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        def _entry_key(entry: dict[str, Any]) -> tuple[int | None, int | None]:
            stage_raw = (
                (entry.get("environment_config") or {}).get("curriculum_stage")
                if isinstance(entry.get("environment_config"), dict)
                else None
            )
            try:
                stage = int(stage_raw) if stage_raw is not None else None
            except (TypeError, ValueError):
                stage = None
            return entry.get("checkpoint_episode"), stage

        new_key = _entry_key(new_entry)
        filtered = [
            entry
            for entry in checkpoints
            if _entry_key(entry) != new_key
        ]
        filtered.append(new_entry)
        def _stage_value(entry: dict[str, Any]) -> int:
            stage_raw = (
                (entry.get("environment_config") or {}).get("curriculum_stage")
                if isinstance(entry.get("environment_config"), dict)
                else 0
            )
            try:
                return int(stage_raw)
            except (TypeError, ValueError):
                return 0

        filtered.sort(
            key=lambda entry: (
                entry.get("checkpoint_episode", 0),
                _stage_value(entry),
            )
        )
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
        atomic_write_json(self.output_root / "training-summary.json", payload)

    def _export_web_assets(
        self,
        web_export_dir: Path,
        checkpoint_payloads: list[dict[str, Any]],
        summary: EvaluationSummary,
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
            exported_payload["journey"] = "current"
            exported_payload["records"] = [
                _export_record(record) for record in checkpoint_payload["records"]
            ]
            exported_checkpoints.append(exported_payload)

        # Prepend archived journey checkpoints so the full timeline is visible.
        archived_checkpoints = self._load_archived_checkpoints()
        all_checkpoints = archived_checkpoints + exported_checkpoints

        atomic_write_json(
            web_export_dir / "latest-checkpoint.json", {"checkpoint": checkpoint_path.name}
        )
        atomic_write_json(web_export_dir / "latest-evaluation.json", summary_payload)
        atomic_write_json(
            web_export_dir / "checkpoint-index.json",
            {"checkpoints": all_checkpoints, "latest": summary_payload},
        )
        replay_target = web_export_dir / "latest-replay.json"
        atomic_write_text(replay_target, replay_path.read_text(encoding="utf-8"))

    def _load_archived_checkpoints(self) -> list[dict[str, Any]]:
        """Load checkpoints from all archived journeys.

        Scans ``artifacts/archive/journey-*/checkpoint-index.json`` (preferred)
        or ``training-summary.json`` (fallback) for each archived journey and
        returns their checkpoint entries tagged with ``"journey"`` metadata.
        Records are stripped to keep the payload lightweight — archived replays
        are not expected to be playable from the live UI.
        """
        archive_root = self.output_root / "archive"
        if not archive_root.is_dir():
            return []

        all_archived: list[dict[str, Any]] = []
        journey_dirs = sorted(archive_root.iterdir())
        for journey_dir in journey_dirs:
            if not journey_dir.is_dir() or not journey_dir.name.startswith("journey-"):
                continue
            journey_label = journey_dir.name
            # Prefer checkpoint-index.json (already web-export shaped) over
            # training-summary.json (which contains raw replay paths).
            candidates = [
                journey_dir / "checkpoint-index.json",
                journey_dir / "training-summary.json",
            ]
            loaded: list[dict[str, Any]] = []
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    loaded = list(payload.get("checkpoints", []))
                except (OSError, json.JSONDecodeError):
                    continue
                break  # use the first file that works

            for entry in loaded:
                entry["journey"] = journey_label
                # Strip heavyweight records — archived replays are unavailable
                # from the live web server, so carrying hundreds of KB of per-seed
                # data would bloat the checkpoint-index for no benefit.
                entry.pop("records", None)
            all_archived.extend(loaded)

        # Sort by recorded_at for a consistent timeline across journeys.
        all_archived.sort(
            key=lambda e: e.get("recorded_at", ""),
        )
        return all_archived

    def reconcile_web_exports(self, web_export_dir: str | Path) -> int | None:
        """Rebuild the web export when it lags behind ``training-summary.json``.

        An interrupted run (e.g. an overnight reboot) can leave the trainer's
        ``training-summary.json`` ahead of the web ``checkpoint-index.json`` that
        the UI reads, which shows up as "lost progress"/regression in the chart.
        On startup we compare the two and, if the summary is ahead, re-export the
        web assets from the summary so the UI catches up without re-training.

        Returns the checkpoint episode the web export was advanced to, or ``None``
        when no reconciliation was needed.
        """
        web_dir = Path(web_export_dir)
        summary_path = self.output_root / "training-summary.json"
        index_path = web_dir / "checkpoint-index.json"

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        checkpoints = summary.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            return None

        def _episode(payload: Any) -> int:
            if isinstance(payload, dict):
                value = payload.get("checkpoint_episode")
                if isinstance(value, (int, float)):
                    return int(value)
            return -1

        summary_latest = max((_episode(item) for item in checkpoints), default=-1)
        if summary_latest < 0:
            return None

        web_latest = -1
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            web_latest = _episode(index.get("latest"))
            if web_latest < 0:
                web_latest = max(
                    (_episode(item) for item in index.get("checkpoints", [])),
                    default=-1,
                )
        except (OSError, json.JSONDecodeError):
            web_latest = -1

        if web_latest >= summary_latest:
            return None

        replay_output_dir = web_dir / "replays"
        replay_output_dir.mkdir(parents=True, exist_ok=True)

        def _export_record(record: dict[str, Any]) -> dict[str, Any]:
            source_path = Path(record["replay_path"])
            target_path = replay_output_dir / source_path.name
            if source_path.exists() and not target_path.exists():
                target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            exported_record = dict(record)
            exported_record["replay_path"] = f"/generated/replays/{target_path.name}"
            return exported_record

        exported_checkpoints: list[dict[str, Any]] = []
        for checkpoint_payload in checkpoints:
            exported_payload = dict(checkpoint_payload)
            exported_payload["journey"] = "current"
            exported_payload["records"] = [
                _export_record(record) for record in checkpoint_payload.get("records", [])
            ]
            exported_checkpoints.append(exported_payload)

        # Prepend archived journey checkpoints so the full timeline is visible.
        archived_checkpoints = self._load_archived_checkpoints()
        all_checkpoints = archived_checkpoints + exported_checkpoints

        latest_payload = exported_checkpoints[-1]
        atomic_write_json(
            web_dir / "latest-checkpoint.json", {"checkpoint": latest_payload.get("checkpoint")}
        )
        atomic_write_json(web_dir / "latest-evaluation.json", latest_payload)
        atomic_write_json(
            web_dir / "checkpoint-index.json",
            {"checkpoints": all_checkpoints, "latest": latest_payload},
        )
        latest_records = latest_payload.get("records") or []
        if latest_records:
            replay_rel = latest_records[0].get("replay_path", "")
            replay_src = web_dir / replay_rel.removeprefix("/generated/")
            if replay_src.exists():
                atomic_write_text(
                    web_dir / "latest-replay.json", replay_src.read_text(encoding="utf-8")
                )
        return summary_latest

    def _evaluate_saved_scenarios(
        self, policy: TrainableLinearPolicy, checkpoint_episode: int
    ) -> None:
        """Run saved hard-case scenarios when any exist (does not affect global best)."""

        import contextlib
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            evaluate_checkpoint_on_scenarios(
                self.config,
                policy,
                checkpoint_episode,
            )
