"""Deterministic checkpoint evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.environment import EpisodeResult, SheepdogEnvironment
from sheepdog.policies.base import Policy
from sheepdog.replay.store import ReplayStore


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Per-seed evaluation results."""

    seed: int
    success: bool
    timeout: bool
    stopped: bool
    steps: int
    simulated_seconds: float
    sheep_penned: int
    final_sheep_distance_to_pen: float
    final_flock_spread: float
    no_progress_steps: int
    reward_total: float
    role_switches: int
    collector_activations: int
    blocker_activations: int
    reward_breakdown: dict[str, float]
    replay_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate checkpoint evaluation output."""

    checkpoint_episode: int
    policy_name: str
    records: tuple[EvaluationRecord, ...]
    success_rate: float
    timeout_rate: float
    average_completion_steps: float
    average_completion_seconds: float
    average_sheep_penned: float
    average_reward: float
    trainer_type: str = "hill_climb"
    policy_type: str = "linear"
    average_distance_to_pen: float = 0.0
    average_flock_spread: float = 0.0
    stopped_rate: float = 0.0
    average_role_switches: float = 0.0
    average_collector_activations: float = 0.0
    average_blocker_activations: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator:
    """Run checkpoint comparisons on fixed evaluation seeds."""

    def __init__(self, config: LabConfig, output_root: str | Path) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.replay_store = ReplayStore(self.output_root / "replays")

    def evaluate(
        self,
        policy: Policy,
        seeds: tuple[int, ...],
        checkpoint_episode: int,
    ) -> tuple[EvaluationSummary, Path, Path]:
        results: list[EpisodeResult] = []
        records: list[EvaluationRecord] = []

        for seed in seeds:
            environment = SheepdogEnvironment(self.config)
            result = environment.run_policy(policy, seed, capture_replay=True)
            results.append(result)
            replay_path = self.replay_store.write(
                f"checkpoint-{checkpoint_episode:06d}-seed-{seed:06d}.json",
                {
                    "seed": result.seed,
                    "policy_name": result.policy_name,
                    "trainer_type": getattr(policy, "trainer_type", "hill_climb"),
                    "policy_type": getattr(policy, "policy_type", "linear"),
                    "final_snapshot": result.final_snapshot.to_dict(),
                    "stats": asdict(result.stats),
                    "frames": [frame.to_dict() for frame in result.replay],
                },
            )
            records.append(
                EvaluationRecord(
                    **{
                        **self._record_from_result(result).to_dict(),
                        "replay_path": str(replay_path),
                    }
                )
            )

        summary = EvaluationSummary(
            checkpoint_episode=checkpoint_episode,
            policy_name=policy.name,
            trainer_type=getattr(policy, "trainer_type", "hill_climb"),
            policy_type=getattr(policy, "policy_type", "linear"),
            records=tuple(records),
            success_rate=fmean(1.0 if record.success else 0.0 for record in records),
            timeout_rate=fmean(1.0 if record.timeout else 0.0 for record in records),
            average_completion_steps=fmean(record.steps for record in records),
            average_completion_seconds=fmean(record.simulated_seconds for record in records),
            average_sheep_penned=fmean(record.sheep_penned for record in records),
            average_reward=fmean(record.reward_total for record in records),
            average_distance_to_pen=fmean(
                record.final_sheep_distance_to_pen for record in records
            ),
            average_flock_spread=fmean(record.final_flock_spread for record in records),
            stopped_rate=fmean(1.0 if record.stopped else 0.0 for record in records),
            average_role_switches=fmean(result.stats.role_switches for result in results),
            average_collector_activations=fmean(
                result.stats.collector_activations for result in results
            ),
            average_blocker_activations=fmean(
                result.stats.blocker_activations for result in results
            ),
        )

        json_path = self.output_root / f"evaluation-checkpoint-{checkpoint_episode:06d}.json"
        csv_path = self.output_root / f"evaluation-checkpoint-{checkpoint_episode:06d}.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict().keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

        return summary, json_path, csv_path

    def _record_from_result(self, result: EpisodeResult) -> EvaluationRecord:
        snapshot = result.final_snapshot
        return EvaluationRecord(
            seed=result.seed,
            success=result.stats.success,
            timeout=result.stats.timeout,
            stopped=result.stats.stopped,
            steps=result.stats.steps,
            simulated_seconds=result.stats.simulated_seconds,
            sheep_penned=result.stats.sheep_penned,
            final_sheep_distance_to_pen=snapshot.average_distance_to_pen,
            final_flock_spread=snapshot.flock_spread,
            no_progress_steps=result.stats.no_progress_steps,
            reward_total=result.stats.reward_total,
            role_switches=result.stats.role_switches,
            collector_activations=result.stats.collector_activations,
            blocker_activations=result.stats.blocker_activations,
            reward_breakdown=result.stats.final_reward_breakdown,
            replay_path="",
        )
