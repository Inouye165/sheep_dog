"""Benchmark harness for comparing baseline and experimental policies."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.base import Policy
from sheepdog.replay.store import ReplayStore


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Comparable aggregate metrics for one policy run group."""

    label: str
    policy_name: str
    trainer_type: str
    policy_type: str
    seeds: tuple[int, ...]
    success_rate: float
    average_sheep_penned: float
    average_episode_reward: float
    timeout_rate: float
    stopped_rate: float
    average_flock_spread: float
    average_distance_to_pen: float
    average_role_switches: float
    average_collector_activations: float
    average_blocker_activations: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkHarness:
    """Evaluate multiple policy modes on the same fixed seeds."""

    def __init__(self, config: LabConfig, output_root: str | Path) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.replay_store = ReplayStore(self.output_root / "replays")

    def evaluate_policy(
        self,
        label: str,
        policy: Policy,
        seeds: tuple[int, ...],
    ) -> BenchmarkResult:
        results = [
            SheepdogEnvironment(self.config).run_policy(policy, seed=seed, capture_replay=False)
            for seed in seeds
        ]
        return BenchmarkResult(
            label=label,
            policy_name=policy.name,
            trainer_type=getattr(policy, "trainer_type", "scripted"),
            policy_type=getattr(policy, "policy_type", "scripted"),
            seeds=seeds,
            success_rate=fmean(1.0 if result.stats.success else 0.0 for result in results),
            average_sheep_penned=fmean(result.stats.sheep_penned for result in results),
            average_episode_reward=fmean(result.stats.reward_total for result in results),
            timeout_rate=fmean(1.0 if result.stats.timeout else 0.0 for result in results),
            stopped_rate=fmean(1.0 if result.stats.stopped else 0.0 for result in results),
            average_flock_spread=fmean(result.final_snapshot.flock_spread for result in results),
            average_distance_to_pen=fmean(
                result.final_snapshot.average_distance_to_pen for result in results
            ),
            average_role_switches=fmean(result.stats.role_switches for result in results),
            average_collector_activations=fmean(
                result.stats.collector_activations for result in results
            ),
            average_blocker_activations=fmean(
                result.stats.blocker_activations for result in results
            ),
        )

    def compare(
        self,
        entries: list[tuple[str, Policy]],
        seeds: tuple[int, ...],
        output_stem: str = "benchmark-comparison",
    ) -> tuple[list[BenchmarkResult], Path, Path, Path]:
        results = [self.evaluate_policy(label, policy, seeds) for label, policy in entries]
        json_path = self.output_root / f"{output_stem}.json"
        csv_path = self.output_root / f"{output_stem}.csv"
        summary_path = self.output_root / f"{output_stem}.md"
        json_path.write_text(
            json.dumps({"results": [result.to_dict() for result in results]}, indent=2),
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].to_dict().keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
        summary_lines = [
            "# Benchmark Comparison",
            "",
            f"Seeds: {', '.join(str(seed) for seed in seeds)}",
            "",
        ]
        for result in results:
            summary_lines.extend(
                [
                    f"## {result.label}",
                    f"- policy: {result.policy_name}",
                    f"- trainer: {result.trainer_type}",
                    f"- success_rate: {result.success_rate:.3f}",
                    f"- average_episode_reward: {result.average_episode_reward:.3f}",
                    f"- average_sheep_penned: {result.average_sheep_penned:.3f}",
                    f"- timeout_rate: {result.timeout_rate:.3f}",
                    f"- stopped_rate: {result.stopped_rate:.3f}",
                    f"- average_flock_spread: {result.average_flock_spread:.3f}",
                    f"- average_distance_to_pen: {result.average_distance_to_pen:.3f}",
                    "",
                ]
            )
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        return results, json_path, csv_path, summary_path