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
    average_gate_progress: float
    average_controlled_stall_steps: float
    average_left_flank_occupancy_steps: float
    average_right_flank_occupancy_steps: float
    average_gate_corridor_occupancy_peak: float
    average_gate_corridor_failure_steps: float

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
                    f"- average_gate_progress: {result.average_gate_progress:.3f}",
                    "- average_controlled_stall_steps: "
                    f"{result.average_controlled_stall_steps:.3f}",
                    "- average_gate_corridor_peak: "
                    f"{result.average_gate_corridor_occupancy_peak:.3f}",
                    "- average_gate_corridor_failures: "
                    f"{result.average_gate_corridor_failure_steps:.3f}",
                    "",
                ]
            )
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        return results, json_path, csv_path, summary_path


# ---------------------------------------------------------------------------
# Herding evaluation report: compare baseline + hierarchical neural policies
# ---------------------------------------------------------------------------


def run_herding_eval_report(
    config: LabConfig,
    output_dir: str | Path,
    seeds: tuple[int, ...] | None = None,
    *,
    hierarchical_model_path: str | Path | None = None,
    hierarchical_checkpoint_episode: int | None = None,
    baseline_checkpoint_episode: int | None = None,
) -> tuple[Path, Path]:
    """Compare random, scripted baseline, and hierarchical neural dog policies.

    This is the key proof-of-learning report.  Running it at different training
    milestones shows whether the neural policy is improving over the baselines.

    Parameters
    ----------
    config:
        Lab configuration.
    output_dir:
        Directory to write the report files.
    seeds:
        Evaluation seeds.  Defaults to ``config.training.evaluation_seeds``.
    hierarchical_model_path:
        Path to a trained ``ShepherdNeuralDogPolicy`` ``.zip`` model file.
        When ``None`` the hierarchical entry is an untrained (random-weight)
        policy and labelled ``hierarchical_checkpoint_0``.
    hierarchical_checkpoint_episode:
        Checkpoint episode number for labelling (default: 0 if no model given).
    baseline_checkpoint_episode:
        Checkpoint episode to load for ``trained_policy`` / ``neural_policy``
        baseline comparison.  When ``None`` the scripted baselines are used only.

    Returns
    -------
    (json_path, markdown_path)
        Paths to the generated report files.
    """
    import datetime

    from sheepdog.policies.factory import create_policy_from_name, load_playable_policy
    from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
    from sheepdog.policies.random_policy import RandomPolicy

    eval_seeds = seeds or config.training.evaluation_seeds
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    harness = BenchmarkHarness(config, out / "replays")

    entries: list[tuple[str, Any]] = [
        ("random_policy", RandomPolicy()),
        ("instinct_only", InstinctOnlyPolicy()),
        ("heuristic_expert", HeuristicExpertPolicy()),
    ]

    # Optionally add the trained baseline policy from a checkpoint.
    if baseline_checkpoint_episode is not None:
        try:
            baseline_policy = load_playable_policy(
                config, checkpoint_episode=baseline_checkpoint_episode
            )
            entries.append(
                (f"baseline_checkpoint_{baseline_checkpoint_episode}", baseline_policy)
            )
        except FileNotFoundError:
            pass  # Checkpoint not present – skip without crashing.

    # Hierarchical neural dogs entry.
    hier_label = f"hierarchical_checkpoint_{hierarchical_checkpoint_episode or 0}"
    if hierarchical_model_path is not None:
        from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy

        hier_policy = ShepherdNeuralDogPolicy.load(
            hierarchical_model_path, config
        )
    else:
        from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy

        hier_policy = ShepherdNeuralDogPolicy.initialize(config)
    entries.append((hier_label, hier_policy))

    results, _, _, _ = harness.compare(
        entries,
        tuple(eval_seeds),
        output_stem="_herding_eval_tmp",
    )

    # Build custom report with proof-of-learning summary.
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "generated_at": timestamp,
        "seeds": list(eval_seeds),
        "policies": [r.to_dict() for r in results],
    }
    json_path = out / "herding_eval_latest.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown proof-of-learning table.
    md_lines = [
        "# Herding Evaluation Report",
        "",
        f"Generated: {timestamp}",
        f"Seeds: {', '.join(str(s) for s in eval_seeds)}",
        "",
        "## What is scripted vs learned",
        "",
        "| Component | Type |",
        "|-----------|------|",
        "| Sheep movement | Reactive (scripted) |",
        "| Shepherd commands | Scripted (Phase A) |",
        "| Dog movement – baseline policies | Scripted / hill-climbed |",
        "| Dog movement – hierarchical_checkpoint_* | **Learned via PPO** |",
        "",
        "Dogs in the `hierarchical_checkpoint_*` row learned *only* from reward signals.",
        "They were NOT given scripted movement targets.",
        "",
        "## Metrics",
        "",
        "| Policy | Success% | Avg Penned | Avg Reward | Timeout% | Dist to Pen |",
        "|--------|----------|------------|------------|----------|-------------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.label} "
            f"| {r.success_rate:.1%} "
            f"| {r.average_sheep_penned:.2f} "
            f"| {r.average_episode_reward:.2f} "
            f"| {r.timeout_rate:.1%} "
            f"| {r.average_distance_to_pen:.2f} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Learning is demonstrated when `hierarchical_checkpoint_N` outperforms",
            "`random_policy` and `instinct_only` on success rate and average reward,",
            "and ideally approaches or exceeds `heuristic_expert`.",
            "",
            "A checkpoint that equals `random_policy` has not yet learned.",
            "A checkpoint that matches `heuristic_expert` has learned well.",
            "",
        ]
    )
    md_path = out / "herding_eval_latest.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path