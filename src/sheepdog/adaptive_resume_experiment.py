"""Controlled comparison of restored and legacy-reset adaptive PPO state."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.atomic_io import atomic_write_json
from sheepdog.batch_size_experiment import (
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_EXPERIMENT_TIMESTEPS,
    DEFAULT_TRAIN_SEEDS,
    BatchSizeArmResult,
    _arm_config,
    _bootstrap_arm_state,
    _checkpoint_schedule,
    _exact_paired_p_value,
    _run_arm,
    _snapshot_baseline,
)
from sheepdog.config import LabConfig


@dataclass(frozen=True, slots=True)
class AdaptiveResumeArmResult:
    """Metrics for one training seed and adaptive-resume behavior."""

    mode: str
    train_seed: int
    initial_adaptive_stage: int
    initial_learning_rate: float
    initial_entropy_coef: float
    success_rate: float
    timeout_rate: float
    average_reward: float
    median_success_steps: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    policy_gradient_loss: float
    minimum_checkpoint_success_rate: float
    output_dir: str
    evaluation_json: str
    final_model_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this arm result."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveResumeExperimentResult:
    """Aggregate restored-versus-reset adaptive state comparison."""

    verdict: str
    reason: str
    timesteps_per_arm: int
    train_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    legacy_success_rate: float
    restored_success_rate: float
    success_rate_difference: float
    legacy_timeout_rate: float
    restored_timeout_rate: float
    timeout_rate_difference: float
    paired_restored_wins: int
    paired_legacy_wins: int
    paired_success_p_value: float
    restored_winning_train_seeds: int
    legacy_winning_train_seeds: int
    legacy_collapse_count: int
    restored_collapse_count: int
    per_seed_success_differences: tuple[float, ...]
    output_dir: str
    arm_results: tuple[AdaptiveResumeArmResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this experiment result."""

        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_legacy_comparison(
    payload: dict[str, Any],
    *,
    baseline_root: Path,
    config: LabConfig,
    total_timesteps: int,
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
) -> None:
    """Reject comparisons that do not share all non-treatment controls."""

    expected_baseline = (Path(str(payload["output_dir"])) / "baseline").resolve()
    checks = {
        "frozen baseline": (baseline_root.resolve(), expected_baseline),
        "batch size": (config.training.batch_size, int(payload["control_batch_size"])),
        "timesteps": (int(total_timesteps), int(payload["timesteps_per_arm"])),
        "training seeds": (train_seeds, tuple(payload["train_seeds"])),
        "evaluation seeds": (evaluation_seeds, tuple(payload["evaluation_seeds"])),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "Legacy comparison does not match the restored experiment controls: "
            + ", ".join(mismatches)
        )


def _legacy_arm_results(
    payload: dict[str, Any], train_seeds: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    """Select the legacy 1,024-batch result for every requested training seed."""

    control_batch_size = int(payload["control_batch_size"])
    selected = {
        int(result["train_seed"]): result
        for result in payload["arm_results"]
        if int(result["batch_size"]) == control_batch_size
        and int(result["train_seed"]) in train_seeds
    }
    if set(selected) != set(train_seeds):
        raise ValueError("Legacy comparison is missing a requested training seed")
    return selected


def _evaluation_outcomes(path: Path, seeds: tuple[int, ...]) -> tuple[bool, ...]:
    """Load success outcomes in the requested deterministic seed order."""

    records = _load_json(path).get("records")
    if not isinstance(records, list):
        raise ValueError(f"Evaluation has no records list: {path}")
    outcomes_by_seed = {int(record["seed"]): bool(record["success"]) for record in records}
    if set(outcomes_by_seed) != set(seeds):
        raise ValueError(f"Evaluation seed bank does not match the experiment: {path}")
    return tuple(outcomes_by_seed[seed] for seed in seeds)


def _legacy_result(result: dict[str, Any], config: LabConfig) -> AdaptiveResumeArmResult:
    """Convert a prior control result into the adaptive experiment schema."""

    return AdaptiveResumeArmResult(
        mode="legacy-reset",
        train_seed=int(result["train_seed"]),
        initial_adaptive_stage=1,
        initial_learning_rate=float(config.training.learning_rate),
        initial_entropy_coef=float(config.training.entropy_coef),
        success_rate=float(result["success_rate"]),
        timeout_rate=float(result["timeout_rate"]),
        average_reward=float(result["average_reward"]),
        median_success_steps=float(result["median_success_steps"]),
        approx_kl=float(result["approx_kl"]),
        clip_fraction=float(result["clip_fraction"]),
        explained_variance=float(result["explained_variance"]),
        policy_gradient_loss=float(result["policy_gradient_loss"]),
        minimum_checkpoint_success_rate=float(result["minimum_checkpoint_success_rate"]),
        output_dir=str(result["output_dir"]),
        evaluation_json=str(result["evaluation_json"]),
        final_model_path=str(result["final_model_path"]),
    )


def _restored_result(
    result: BatchSizeArmResult,
    *,
    initial_stage: int,
    initial_learning_rate: float,
    initial_entropy_coef: float,
) -> AdaptiveResumeArmResult:
    """Convert a newly trained restored result into the report schema."""

    return AdaptiveResumeArmResult(
        mode="restored",
        train_seed=result.train_seed,
        initial_adaptive_stage=initial_stage,
        initial_learning_rate=initial_learning_rate,
        initial_entropy_coef=initial_entropy_coef,
        success_rate=result.success_rate,
        timeout_rate=result.timeout_rate,
        average_reward=result.average_reward,
        median_success_steps=result.median_success_steps,
        approx_kl=result.approx_kl,
        clip_fraction=result.clip_fraction,
        explained_variance=result.explained_variance,
        policy_gradient_loss=result.policy_gradient_loss,
        minimum_checkpoint_success_rate=result.minimum_checkpoint_success_rate,
        output_dir=result.output_dir,
        evaluation_json=result.evaluation_json,
        final_model_path=result.final_model_path,
    )


def _decide(
    *,
    success_difference: float,
    timeout_difference: float,
    seed_differences: tuple[float, ...],
    legacy_collapses: int,
    restored_collapses: int,
) -> tuple[str, str]:
    """Classify performance separately from robustness and best-practice value."""

    restored_seed_wins = sum(difference > 0.0 for difference in seed_differences)
    legacy_seed_wins = sum(difference < 0.0 for difference in seed_differences)
    majority = len(seed_differences) // 2 + 1
    if (
        success_difference >= 0.02
        and timeout_difference <= 0.0
        and restored_seed_wins >= majority
        and restored_collapses <= legacy_collapses
    ):
        return "IMPROVED", "Restoration improved average outcomes across most training seeds."
    if (
        success_difference <= -0.02
        and timeout_difference >= 0.0
        and restored_collapses >= legacy_collapses
        and legacy_seed_wins >= majority
    ):
        return "WORSENED", "Restoration regressed outcomes across most training seeds."
    if (
        abs(success_difference) < 0.02
        and abs(timeout_difference) < 0.02
        and restored_collapses < legacy_collapses
    ):
        return "MORE_ROBUST", "Mean performance was similar, with fewer collapse events."
    if (
        abs(success_difference) < 0.02
        and abs(timeout_difference) < 0.02
        and restored_collapses == legacy_collapses
    ):
        return (
            "NO_MATERIAL_DIFFERENCE",
            "No meaningful performance or robustness change was observed.",
        )
    return "MIXED", "Training seeds disagreed or performance and robustness moved differently."


def _write_reports(
    result: AdaptiveResumeExperimentResult, output_root: Path
) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown experiment reports."""

    json_path = output_root / "comparison.json"
    csv_path = output_root / "arms.csv"
    markdown_path = output_root / "comparison.md"
    atomic_write_json(json_path, result.to_dict())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        rows = [arm.to_dict() for arm in result.arm_results]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        "\n".join(
            (
                "# Adaptive Resume A/B Experiment",
                "",
                f"**Verdict:** {result.verdict}",
                f"**Reason:** {result.reason}",
                "",
                "| Metric | Legacy reset | Restored | Difference |",
                "| --- | ---: | ---: | ---: |",
                f"| Success rate | {result.legacy_success_rate:.1%} | "
                f"{result.restored_success_rate:.1%} | {result.success_rate_difference:+.1%} |",
                f"| Timeout rate | {result.legacy_timeout_rate:.1%} | "
                f"{result.restored_timeout_rate:.1%} | {result.timeout_rate_difference:+.1%} |",
                f"| Collapse count | {result.legacy_collapse_count} | "
                f"{result.restored_collapse_count} | |",
                "",
                f"Restored-only paired successes: {result.paired_restored_wins}",
                f"Legacy-only paired successes: {result.paired_legacy_wins}",
                f"Exact paired p-value: {result.paired_success_p_value:.6f}",
                f"Training seeds won by restored: {result.restored_winning_train_seeds}",
                f"Training seeds won by legacy: {result.legacy_winning_train_seeds}",
                f"Timesteps per arm: {result.timesteps_per_arm:,}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return json_path, csv_path, markdown_path


def run_adaptive_resume_experiment(
    config: LabConfig,
    *,
    baseline_root: str | Path,
    legacy_comparison_path: str | Path,
    output_root: str | Path | None = None,
    total_timesteps: int = DEFAULT_EXPERIMENT_TIMESTEPS,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    evaluation_seeds: tuple[int, ...] = DEFAULT_EVALUATION_SEEDS,
    progress_callback: Any | None = None,
) -> AdaptiveResumeExperimentResult:
    """Run restored arms and compare them with matching legacy-reset results."""

    legacy_path = Path(legacy_comparison_path).resolve()
    legacy_payload = _load_json(legacy_path)
    source_baseline = Path(baseline_root).resolve()
    _validate_legacy_comparison(
        legacy_payload,
        baseline_root=source_baseline,
        config=config,
        total_timesteps=total_timesteps,
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    legacy_by_seed = _legacy_arm_results(legacy_payload, train_seeds)
    experiment_id = datetime.now(UTC).strftime("adaptive_resume_ab_%Y%m%d_%H%M%S")
    experiment_root = Path(
        output_root or source_baseline.parent / experiment_id
    ).resolve()
    experiment_root.mkdir(parents=True, exist_ok=False)
    baseline_state = _snapshot_baseline(source_baseline, experiment_root)
    adaptive_state = baseline_state.get("adaptive_step_state")
    if not isinstance(adaptive_state, dict) or int(adaptive_state.get("stage", 1)) <= 1:
        raise ValueError("Frozen baseline must contain adaptive state above stage 1")
    initial_stage = int(adaptive_state["stage"])
    curriculum_stage = int(adaptive_state["curriculum_stage"])
    if curriculum_stage != config.rewards.instincts.curriculum_stage:
        raise ValueError("Frozen adaptive state and config curriculum stages differ")
    initial_learning_rate = config.training.learning_rate * float(
        adaptive_state.get("multiplier", 0.8)
    )
    initial_entropy_coef = config.training.entropy_coef * float(
        adaptive_state.get("multiplier", 0.8)
    )
    checkpoint_targets = _checkpoint_schedule(total_timesteps, config.training.rollout_steps)

    legacy_results: list[AdaptiveResumeArmResult] = []
    restored_results: list[AdaptiveResumeArmResult] = []
    paired_restored_wins = 0
    paired_legacy_wins = 0
    for arm_index, train_seed in enumerate(train_seeds):
        legacy_raw = legacy_by_seed[train_seed]
        legacy_result = _legacy_result(legacy_raw, config)
        arm_root = experiment_root / f"seed-{train_seed}" / "restored"
        _bootstrap_arm_state(
            baseline_state,
            arm_root,
            experiment_id=experiment_id,
            train_seed=train_seed,
            batch_size=config.training.batch_size,
        )
        arm_config = _arm_config(
            config,
            arm_root=arm_root,
            batch_size=config.training.batch_size,
            train_seed=train_seed,
            checkpoint_targets=checkpoint_targets,
        )
        def report_arm_progress(
            payload: dict[str, Any],
            current_arm_index: int = arm_index,
            current_train_seed: int = train_seed,
        ) -> None:
            phase = payload.get("phase")
            arm_timesteps = 0
            if phase == "checkpoint":
                cumulative = int(payload.get("total_timesteps", 0))
                arm_timesteps = max(0, cumulative - int(baseline_state["total_timesteps"]))
            elif phase == "complete":
                arm_timesteps = checkpoint_targets[-1]
            overall_progress = (
                current_arm_index
                + min(1.0, arm_timesteps / checkpoint_targets[-1])
            ) / len(train_seeds)
            atomic_write_json(
                experiment_root / "progress.json",
                {
                    "phase": phase,
                    "current_train_seed": current_train_seed,
                    "completed_arms": current_arm_index,
                    "total_arms": len(train_seeds),
                    "arm_timesteps": arm_timesteps,
                    "arm_total_timesteps": checkpoint_targets[-1],
                    "overall_progress": overall_progress,
                },
            )
            if progress_callback is not None:
                progress_callback(payload)

        report_arm_progress({"phase": "experiment_arm_start", "train_seed": train_seed})
        trained, restored_outcomes = _run_arm(
            arm_config,
            arm_root=arm_root,
            evaluation_seeds=evaluation_seeds,
            progress_callback=report_arm_progress,
        )
        restored_result = _restored_result(
            trained,
            initial_stage=initial_stage,
            initial_learning_rate=initial_learning_rate,
            initial_entropy_coef=initial_entropy_coef,
        )
        legacy_outcomes = _evaluation_outcomes(
            Path(legacy_result.evaluation_json), evaluation_seeds
        )
        paired_restored_wins += sum(
            restored and not legacy
            for legacy, restored in zip(legacy_outcomes, restored_outcomes, strict=True)
        )
        paired_legacy_wins += sum(
            legacy and not restored
            for legacy, restored in zip(legacy_outcomes, restored_outcomes, strict=True)
        )
        legacy_results.append(legacy_result)
        restored_results.append(restored_result)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "experiment_arm_complete",
                    "train_seed": train_seed,
                    "success_rate": restored_result.success_rate,
                    "timeout_rate": restored_result.timeout_rate,
                }
            )

    legacy_success = fmean(result.success_rate for result in legacy_results)
    restored_success = fmean(result.success_rate for result in restored_results)
    legacy_timeout = fmean(result.timeout_rate for result in legacy_results)
    restored_timeout = fmean(result.timeout_rate for result in restored_results)
    seed_differences = tuple(
        restored.success_rate - legacy.success_rate
        for legacy, restored in zip(legacy_results, restored_results, strict=True)
    )
    legacy_collapses = sum(
        result.minimum_checkpoint_success_rate < 0.35 for result in legacy_results
    )
    restored_collapses = sum(
        result.minimum_checkpoint_success_rate < 0.35 for result in restored_results
    )
    verdict, reason = _decide(
        success_difference=restored_success - legacy_success,
        timeout_difference=restored_timeout - legacy_timeout,
        seed_differences=seed_differences,
        legacy_collapses=legacy_collapses,
        restored_collapses=restored_collapses,
    )
    result = AdaptiveResumeExperimentResult(
        verdict=verdict,
        reason=reason,
        timesteps_per_arm=checkpoint_targets[-1],
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        legacy_success_rate=legacy_success,
        restored_success_rate=restored_success,
        success_rate_difference=restored_success - legacy_success,
        legacy_timeout_rate=legacy_timeout,
        restored_timeout_rate=restored_timeout,
        timeout_rate_difference=restored_timeout - legacy_timeout,
        paired_restored_wins=paired_restored_wins,
        paired_legacy_wins=paired_legacy_wins,
        paired_success_p_value=_exact_paired_p_value(
            paired_restored_wins, paired_legacy_wins
        ),
        restored_winning_train_seeds=sum(difference > 0.0 for difference in seed_differences),
        legacy_winning_train_seeds=sum(difference < 0.0 for difference in seed_differences),
        legacy_collapse_count=legacy_collapses,
        restored_collapse_count=restored_collapses,
        per_seed_success_differences=seed_differences,
        output_dir=str(experiment_root),
        arm_results=tuple(legacy_results + restored_results),
    )
    _write_reports(result, experiment_root)
    return result


def main() -> None:
    """Run the adaptive-resume comparison from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare restored adaptive PPO state with prior legacy-reset results."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-comparison", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_EXPERIMENT_TIMESTEPS)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config = LabConfig.from_dict(_load_json(Path(args.config)))

    def report_progress(payload: dict[str, Any]) -> None:
        if payload.get("phase") == "experiment_arm_start":
            print(f"Starting restored seed={payload['train_seed']}...", flush=True)
        elif payload.get("phase") == "checkpoint":
            print(str(payload.get("message", "Checkpoint complete")), flush=True)
        elif payload.get("phase") == "experiment_arm_complete":
            print(
                f"Completed seed={payload['train_seed']}: "
                f"success={float(payload['success_rate']):.1%}, "
                f"timeout={float(payload['timeout_rate']):.1%}",
                flush=True,
            )

    result = run_adaptive_resume_experiment(
        config,
        baseline_root=args.baseline_dir,
        legacy_comparison_path=args.legacy_comparison,
        output_root=args.output_dir,
        total_timesteps=args.timesteps,
        train_seeds=tuple(args.train_seeds),
        evaluation_seeds=tuple(args.evaluation_seeds or DEFAULT_EVALUATION_SEEDS),
        progress_callback=report_progress,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
