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
    no_progress_steps: int
    reward_total: float
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
        records: list[EvaluationRecord] = []

        for seed in seeds:
            environment = SheepdogEnvironment(self.config)
            result = environment.run_policy(policy, seed, capture_replay=True)
            replay_path = self.replay_store.write(
                f"checkpoint-{checkpoint_episode:06d}-seed-{seed:06d}.json",
                {
                    "seed": result.seed,
                    "policy_name": result.policy_name,
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
            records=tuple(records),
            success_rate=fmean(1.0 if record.success else 0.0 for record in records),
            timeout_rate=fmean(1.0 if record.timeout else 0.0 for record in records),
            average_completion_steps=fmean(record.steps for record in records),
            average_completion_seconds=fmean(record.simulated_seconds for record in records),
            average_sheep_penned=fmean(record.sheep_penned for record in records),
            average_reward=fmean(record.reward_total for record in records),
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
            no_progress_steps=result.stats.no_progress_steps,
            reward_total=result.stats.reward_total,
            reward_breakdown=result.stats.final_reward_breakdown,
            replay_path="",
        )
