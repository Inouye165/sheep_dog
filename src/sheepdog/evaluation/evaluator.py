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


def _policy_metadata(policy_name: str, *, trainer_type: str | None, policy_type: str | None) -> tuple[str, str, str]:
    """Return normalized trainer, policy, and replay-mode labels for replay export."""

    normalized_trainer = trainer_type or "baseline"
    normalized_policy_type = policy_type or "instinct"
    replay_mode = "baseline"
    if policy_name == "neural_policy" or normalized_trainer == "maskable_ppo":
        normalized_trainer = "maskable_ppo"
        normalized_policy_type = "neural"
        replay_mode = "neural_ppo"
    elif policy_name == "trained_policy":
        normalized_trainer = "hill_climb"
        normalized_policy_type = "linear"
        replay_mode = "trained_linear"
    elif policy_name == "heuristic_expert":
        normalized_trainer = "baseline"
        normalized_policy_type = "heuristic"
    elif policy_name in {"random_untrained", "random_policy"}:
        normalized_trainer = "baseline"
        normalized_policy_type = "random"
    else:
        normalized_trainer = "baseline"
        normalized_policy_type = "instinct"
    return normalized_trainer, normalized_policy_type, replay_mode


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
    cumulative_gate_progress: float
    controlled_stall_steps: int
    left_flank_occupancy_steps: int
    right_flank_occupancy_steps: int
    gate_corridor_occupancy_peak: float
    gate_corridor_failure_steps: int
    dog_role_occupancy: dict[str, dict[str, int]]
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
    average_gate_progress: float = 0.0
    average_controlled_stall_steps: float = 0.0
    average_left_flank_occupancy_steps: float = 0.0
    average_right_flank_occupancy_steps: float = 0.0
    average_gate_corridor_occupancy_peak: float = 0.0
    average_gate_corridor_failure_steps: float = 0.0

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
            trainer_type, policy_type, replay_mode = _policy_metadata(
                result.policy_name,
                trainer_type=getattr(policy, "trainer_type", None),
                policy_type=getattr(policy, "policy_type", None),
            )
            replay_path = self.replay_store.write(
                f"checkpoint-{checkpoint_episode:06d}-seed-{seed:06d}.json",
                {
                    "seed": result.seed,
                    "policy_name": result.policy_name,
                    "trainer_type": trainer_type,
                    "policy_type": policy_type,
                    "policy_mode": result.policy_name,
                    "replay_mode": replay_mode,
                    "environment": {
                        "dogs": self.config.environment.dogs,
                        "sheep": self.config.environment.sheep,
                        "width": self.config.environment.width,
                        "height": self.config.environment.height,
                        "curriculum_stage": self.config.rewards.instincts.curriculum_stage,
                        "enable_instinct_rewards": self.config.rewards.instincts.enable_instinct_rewards,
                    },
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

        summary_trainer_type, summary_policy_type, _summary_replay_mode = _policy_metadata(
            policy.name,
            trainer_type=getattr(policy, "trainer_type", None),
            policy_type=getattr(policy, "policy_type", None),
        )

        summary = EvaluationSummary(
            checkpoint_episode=checkpoint_episode,
            policy_name=policy.name,
            trainer_type=summary_trainer_type,
            policy_type=summary_policy_type,
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
            average_gate_progress=fmean(
                result.stats.cumulative_gate_progress for result in results
            ),
            average_controlled_stall_steps=fmean(
                result.stats.controlled_stall_steps for result in results
            ),
            average_left_flank_occupancy_steps=fmean(
                result.stats.left_flank_occupancy_steps for result in results
            ),
            average_right_flank_occupancy_steps=fmean(
                result.stats.right_flank_occupancy_steps for result in results
            ),
            average_gate_corridor_occupancy_peak=fmean(
                result.stats.gate_corridor_occupancy_peak for result in results
            ),
            average_gate_corridor_failure_steps=fmean(
                result.stats.gate_corridor_failure_steps for result in results
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
            cumulative_gate_progress=result.stats.cumulative_gate_progress,
            controlled_stall_steps=result.stats.controlled_stall_steps,
            left_flank_occupancy_steps=result.stats.left_flank_occupancy_steps,
            right_flank_occupancy_steps=result.stats.right_flank_occupancy_steps,
            gate_corridor_occupancy_peak=result.stats.gate_corridor_occupancy_peak,
            gate_corridor_failure_steps=result.stats.gate_corridor_failure_steps,
            dog_role_occupancy=result.stats.dog_role_occupancy,
            reward_breakdown=result.stats.final_reward_breakdown,
            replay_path="",
        )
