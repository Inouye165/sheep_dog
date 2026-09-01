"""Controlled A/B experiments for MaskablePPO batch size changes."""

from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import EvaluationSummary, Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.factory import create_trainer

DEFAULT_EXPERIMENT_TIMESTEPS = 512_000
DEFAULT_TRAIN_SEEDS = (7, 17, 29)
DEFAULT_EVALUATION_SEEDS = tuple(1_000_003 + index * 7_919 for index in range(100))


@dataclass(frozen=True, slots=True)
class BatchSizeArmResult:
    """Metrics produced by one training seed and batch-size arm."""

    train_seed: int
    batch_size: int
    effective_batch_size: int
    success_rate: float
    timeout_rate: float
    average_reward: float
    median_success_steps: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    policy_gradient_loss: float
    minimum_checkpoint_success_rate: float
    candidate_only_successes: int
    control_only_successes: int
    output_dir: str
    evaluation_json: str
    final_model_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this arm result."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchSizeExperimentResult:
    """Aggregate report for a complete batch-size A/B experiment."""

    verdict: str
    reason: str
    control_batch_size: int
    candidate_batch_size: int
    timesteps_per_arm: int
    train_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    control_success_rate: float
    candidate_success_rate: float
    success_rate_difference: float
    control_timeout_rate: float
    candidate_timeout_rate: float
    timeout_rate_difference: float
    control_median_success_steps: float
    candidate_median_success_steps: float
    median_success_steps_difference: float
    paired_candidate_wins: int
    paired_control_wins: int
    paired_success_p_value: float
    output_dir: str
    arm_results: tuple[BatchSizeArmResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this experiment result."""

        return asdict(self)


def _resolve_model_path(raw_path: str, baseline_root: Path) -> Path:
    """Resolve a model path stored as either absolute or workspace-relative."""

    model_path = Path(raw_path).expanduser()
    candidates = (
        model_path,
        Path.cwd() / model_path,
        baseline_root.parent / model_path,
        baseline_root / model_path,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Baseline model does not exist: {raw_path}")


def _checkpoint_schedule(total_timesteps: int, rollout_steps: int) -> tuple[int, ...]:
    """Return rollout-aligned checkpoints near 20%, 50%, and 100%."""

    quantum = max(1, int(rollout_steps))
    aligned_total = max(quantum, (int(total_timesteps) // quantum) * quantum)
    targets = {
        max(quantum, (int(aligned_total * fraction) // quantum) * quantum)
        for fraction in (0.20, 0.50, 1.0)
    }
    targets.add(aligned_total)
    return tuple(sorted(target for target in targets if target <= aligned_total))


def _snapshot_baseline(baseline_root: Path, output_root: Path) -> dict[str, Any]:
    """Freeze baseline state and model so both arms start identically."""

    state_path = baseline_root / "training-state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Baseline training state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    raw_model_path = state.get("best_model_path") or state.get("policy_state_path")
    if not raw_model_path:
        raise ValueError("Baseline training state has no PPO model path")

    source_model = _resolve_model_path(str(raw_model_path), baseline_root)
    snapshot_root = output_root / "baseline"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    snapshot_model = snapshot_root / "start-model.zip"
    shutil.copy2(source_model, snapshot_model)
    if not zipfile.is_zipfile(snapshot_model):
        raise ValueError(f"Baseline model snapshot is not a valid zip file: {source_model}")

    state["policy_state_path"] = str(snapshot_model.resolve())
    state["best_model_path"] = str(snapshot_model.resolve())
    state["incomplete_batch"] = None
    atomic_write_json(snapshot_root / "training-state.json", state)
    return state


def _arm_config(
    config: LabConfig,
    *,
    arm_root: Path,
    batch_size: int,
    train_seed: int,
    checkpoint_targets: tuple[int, ...],
) -> LabConfig:
    """Build an isolated config that differs only in experiment controls."""

    training = replace(
        config.training,
        trainer_type="maskable_ppo",
        policy_type="neural",
        batch_size=int(batch_size),
        train_seed=int(train_seed),
        episodes=len(checkpoint_targets),
        checkpoint_episodes=tuple(range(len(checkpoint_targets))),
        checkpoint_timesteps=checkpoint_targets,
        total_timesteps=checkpoint_targets[-1],
        output_dir=str(arm_root),
        web_export_dir=str(arm_root / "web"),
        backup_enabled=False,
        backup_dir=str(arm_root / "backups"),
        wandb_enabled=False,
    )
    return replace(config, training=training)


def _bootstrap_arm_state(
    baseline_state: dict[str, Any],
    arm_root: Path,
    *,
    experiment_id: str,
    train_seed: int,
    batch_size: int,
) -> None:
    """Create an independent resumable state rooted at the frozen model."""

    arm_root.mkdir(parents=True, exist_ok=False)
    arm_state = dict(baseline_state)
    source_run_id = arm_state.get("run_id")
    source_checkpoint_id = arm_state.get("parent_checkpoint_id")
    arm_state.update(
        {
            "incomplete_batch": None,
            "run_id": f"{experiment_id}_seed{train_seed}_batch{batch_size}",
            "parent_run_id": source_run_id,
            "parent_checkpoint_id": source_checkpoint_id,
        }
    )
    atomic_write_json(arm_root / "training-state.json", arm_state)


def _median_success_steps(summary: EvaluationSummary) -> float:
    """Return median steps for successful episodes, or infinity if none succeed."""

    successful_steps = [record.steps for record in summary.records if record.success]
    return float(median(successful_steps)) if successful_steps else math.inf


def _last_training_metrics(checkpoints: list[dict[str, Any]]) -> dict[str, float]:
    """Extract PPO diagnostics from the final experiment checkpoint."""

    if not checkpoints:
        raise RuntimeError("Training produced no checkpoints")
    final = checkpoints[-1]
    success_rates = [float(checkpoint.get("success_rate", 0.0)) for checkpoint in checkpoints]
    return {
        "approx_kl": float(final.get("approx_kl", 0.0)),
        "clip_fraction": float(final.get("clip_fraction", 0.0)),
        "explained_variance": float(final.get("explained_variance", 0.0)),
        "policy_gradient_loss": float(final.get("policy_gradient_loss", 0.0)),
        "minimum_checkpoint_success_rate": min(success_rates),
    }


def _close_policy(policy: NeuralPolicy) -> None:
    """Close the policy's vector environment after evaluation."""

    environment = policy.model.get_env()
    if environment is not None:
        environment.close()


def _run_arm(
    config: LabConfig,
    *,
    arm_root: Path,
    evaluation_seeds: tuple[int, ...],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[BatchSizeArmResult, tuple[bool, ...]]:
    """Train and evaluate one isolated experiment arm."""

    trainer = create_trainer(config, arm_root)
    training_summary = trainer.train(progress_callback=progress_callback)
    metrics = _last_training_metrics(training_summary.checkpoints)
    policy = NeuralPolicy.load(
        training_summary.final_model_path,
        config,
        training_summary.policy_config,
    )
    try:
        effective_batch_size = int(policy.model.batch_size)
        if effective_batch_size != config.training.batch_size:
            raise RuntimeError(
                "PPO effective batch size does not match experiment configuration: "
                f"expected {config.training.batch_size}, got {effective_batch_size}"
            )
        evaluator = Evaluator(config, arm_root / "final-evaluation")
        state = json.loads((arm_root / "training-state.json").read_text(encoding="utf-8"))
        evaluation, evaluation_json, _evaluation_csv = evaluator.evaluate(
            policy,
            evaluation_seeds,
            checkpoint_episode=int(state.get("total_episodes_trained", 0)),
            deterministic=True,
            capture_replays=False,
            evaluation_mode="batch-size-ab",
            run_id=str(state.get("run_id")),
            policy_version=state.get("policy_version"),
            curriculum_stage=config.rewards.instincts.curriculum_stage,
        )
    finally:
        _close_policy(policy)

    result = BatchSizeArmResult(
        train_seed=config.training.train_seed,
        batch_size=config.training.batch_size,
        effective_batch_size=effective_batch_size,
        success_rate=evaluation.success_rate,
        timeout_rate=evaluation.timeout_rate,
        average_reward=evaluation.average_reward,
        median_success_steps=_median_success_steps(evaluation),
        approx_kl=metrics["approx_kl"],
        clip_fraction=metrics["clip_fraction"],
        explained_variance=metrics["explained_variance"],
        policy_gradient_loss=metrics["policy_gradient_loss"],
        minimum_checkpoint_success_rate=metrics["minimum_checkpoint_success_rate"],
        candidate_only_successes=0,
        control_only_successes=0,
        output_dir=str(arm_root),
        evaluation_json=str(evaluation_json),
        final_model_path=training_summary.final_model_path,
    )
    return result, tuple(record.success for record in evaluation.records)


def _exact_paired_p_value(candidate_wins: int, control_wins: int) -> float:
    """Return a two-sided exact McNemar p-value for paired success outcomes."""

    discordant = candidate_wins + control_wins
    if discordant == 0:
        return 1.0
    tail = min(candidate_wins, control_wins)
    probability = sum(
        math.comb(discordant, successes) * 0.5**discordant
        for successes in range(tail + 1)
    )
    return min(1.0, 2.0 * probability)


def _decide(
    *,
    success_difference: float,
    timeout_difference: float,
    candidate_wins: int,
    control_wins: int,
    p_value: float,
    candidate_minimum_success: float,
) -> tuple[str, str]:
    """Apply conservative guardrails to the batch-size decision."""

    if candidate_minimum_success < 0.35:
        return "KEEP_1024", "Candidate crossed the configured collapse threshold."
    if success_difference <= -0.02 or timeout_difference >= 0.02:
        return "KEEP_1024", "Candidate regressed success or timeout rate by at least 2 points."
    if (
        p_value < 0.05
        and candidate_wins > control_wins
        and success_difference >= 0.02
        and timeout_difference <= 0.0
    ):
        return "KEEP_512", "Candidate produced a significant paired success improvement."
    return "INCONCLUSIVE", "No statistically clear, guardrail-safe winner was observed."


def _write_reports(result: BatchSizeExperimentResult, output_root: Path) -> tuple[Path, Path, Path]:
    """Write machine-readable and human-readable experiment reports."""

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
                "# Batch Size A/B Experiment",
                "",
                f"**Verdict:** {result.verdict}",
                f"**Reason:** {result.reason}",
                "",
                "| Metric | Control | Candidate | Difference |",
                "| --- | ---: | ---: | ---: |",
                f"| Batch size | {result.control_batch_size} | "
                f"{result.candidate_batch_size} | |",
                f"| Success rate | {result.control_success_rate:.1%} | "
                f"{result.candidate_success_rate:.1%} | "
                f"{result.success_rate_difference:+.1%} |",
                f"| Timeout rate | {result.control_timeout_rate:.1%} | "
                f"{result.candidate_timeout_rate:.1%} | "
                f"{result.timeout_rate_difference:+.1%} |",
                f"| Median successful steps | {result.control_median_success_steps:.1f} | "
                f"{result.candidate_median_success_steps:.1f} | "
                f"{result.median_success_steps_difference:+.1f} |",
                "",
                f"Paired candidate wins: {result.paired_candidate_wins}",
                f"Paired control wins: {result.paired_control_wins}",
                f"Exact paired p-value: {result.paired_success_p_value:.6f}",
                f"Timesteps per arm: {result.timesteps_per_arm:,}",
                f"Training seeds: {', '.join(map(str, result.train_seeds))}",
                f"Evaluation seeds per arm: {len(result.evaluation_seeds)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return json_path, csv_path, markdown_path


def run_batch_size_experiment(
    config: LabConfig,
    *,
    baseline_root: str | Path = "artifacts",
    output_root: str | Path | None = None,
    total_timesteps: int = DEFAULT_EXPERIMENT_TIMESTEPS,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    evaluation_seeds: tuple[int, ...] = DEFAULT_EVALUATION_SEEDS,
    control_batch_size: int = 1024,
    candidate_batch_size: int = 512,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BatchSizeExperimentResult:
    """Train both batch sizes from one frozen checkpoint and compare paired outcomes."""

    if not train_seeds:
        raise ValueError("At least one training seed is required")
    if not evaluation_seeds:
        raise ValueError("At least one evaluation seed is required")
    if control_batch_size == candidate_batch_size:
        raise ValueError("Control and candidate batch sizes must differ")
    for batch_size in (control_batch_size, candidate_batch_size):
        if batch_size <= 0 or config.training.rollout_steps % batch_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must divide rollout_steps "
                f"({config.training.rollout_steps})"
            )
    experiment_id = datetime.now(UTC).strftime("batch_ab_%Y%m%d_%H%M%S")
    experiment_root = Path(
        output_root or Path(baseline_root) / "experiments" / experiment_id
    ).resolve()
    experiment_root.mkdir(parents=True, exist_ok=False)
    baseline_state = _snapshot_baseline(Path(baseline_root).resolve(), experiment_root)
    checkpoint_targets = _checkpoint_schedule(total_timesteps, config.training.rollout_steps)

    arm_results: list[BatchSizeArmResult] = []
    paired_candidate_wins = 0
    paired_control_wins = 0
    for train_seed in train_seeds:
        outcomes: dict[int, tuple[bool, ...]] = {}
        seed_results: dict[int, BatchSizeArmResult] = {}
        for batch_size in (control_batch_size, candidate_batch_size):
            arm_root = experiment_root / f"seed-{train_seed}" / f"batch-{batch_size}"
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "experiment_arm_start",
                        "train_seed": train_seed,
                        "batch_size": batch_size,
                        "output_dir": str(arm_root),
                    }
                )
            _bootstrap_arm_state(
                baseline_state,
                arm_root,
                experiment_id=experiment_id,
                train_seed=train_seed,
                batch_size=batch_size,
            )
            arm_config = _arm_config(
                config,
                arm_root=arm_root,
                batch_size=batch_size,
                train_seed=train_seed,
                checkpoint_targets=checkpoint_targets,
            )
            result, arm_outcomes = _run_arm(
                arm_config,
                arm_root=arm_root,
                evaluation_seeds=evaluation_seeds,
                progress_callback=progress_callback,
            )
            outcomes[batch_size] = arm_outcomes
            seed_results[batch_size] = result
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "experiment_arm_complete",
                        "train_seed": train_seed,
                        "batch_size": batch_size,
                        "effective_batch_size": result.effective_batch_size,
                        "success_rate": result.success_rate,
                        "timeout_rate": result.timeout_rate,
                        "approx_kl": result.approx_kl,
                    }
                )

        candidate_wins = sum(
            candidate and not control
            for control, candidate in zip(
                outcomes[control_batch_size], outcomes[candidate_batch_size], strict=True
            )
        )
        control_wins = sum(
            control and not candidate
            for control, candidate in zip(
                outcomes[control_batch_size], outcomes[candidate_batch_size], strict=True
            )
        )
        paired_candidate_wins += candidate_wins
        paired_control_wins += control_wins
        arm_results.extend(
            (
                replace(seed_results[control_batch_size], control_only_successes=control_wins),
                replace(
                    seed_results[candidate_batch_size],
                    candidate_only_successes=candidate_wins,
                ),
            )
        )

    control_results = [
        result for result in arm_results if result.batch_size == control_batch_size
    ]
    candidate_results = [
        result for result in arm_results if result.batch_size == candidate_batch_size
    ]
    control_success = fmean(result.success_rate for result in control_results)
    candidate_success = fmean(result.success_rate for result in candidate_results)
    control_timeout = fmean(result.timeout_rate for result in control_results)
    candidate_timeout = fmean(result.timeout_rate for result in candidate_results)
    control_steps = fmean(result.median_success_steps for result in control_results)
    candidate_steps = fmean(result.median_success_steps for result in candidate_results)
    p_value = _exact_paired_p_value(paired_candidate_wins, paired_control_wins)
    verdict, reason = _decide(
        success_difference=candidate_success - control_success,
        timeout_difference=candidate_timeout - control_timeout,
        candidate_wins=paired_candidate_wins,
        control_wins=paired_control_wins,
        p_value=p_value,
        candidate_minimum_success=min(
            result.minimum_checkpoint_success_rate for result in candidate_results
        ),
    )
    experiment_result = BatchSizeExperimentResult(
        verdict=verdict,
        reason=reason,
        control_batch_size=control_batch_size,
        candidate_batch_size=candidate_batch_size,
        timesteps_per_arm=checkpoint_targets[-1],
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        control_success_rate=control_success,
        candidate_success_rate=candidate_success,
        success_rate_difference=candidate_success - control_success,
        control_timeout_rate=control_timeout,
        candidate_timeout_rate=candidate_timeout,
        timeout_rate_difference=candidate_timeout - control_timeout,
        control_median_success_steps=control_steps,
        candidate_median_success_steps=candidate_steps,
        median_success_steps_difference=candidate_steps - control_steps,
        paired_candidate_wins=paired_candidate_wins,
        paired_control_wins=paired_control_wins,
        paired_success_p_value=p_value,
        output_dir=str(experiment_root),
        arm_results=tuple(arm_results),
    )
    _write_reports(experiment_result, experiment_root)
    return experiment_result
