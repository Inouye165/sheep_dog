"""Stage 1 policy benchmark.

Compares random_untrained, instinct_only, heuristic_expert, and the
latest trained checkpoint on Stage 1 (1 dog, 1 sheep, 60x45 grid) using
fixed seeds. Reports success_rate, average_sheep_penned, average_reward,
timeout_rate, and average_final_distance_to_pen.

Run with: python -m pytest tests/test_stage1_benchmark.py -v -s
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import pytest

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.factory import create_policy_from_name, load_playable_policy
from sheepdog.policies.heuristic import HeuristicExpertPolicy

STAGE1_SEEDS = (11, 23, 37, 41, 53)

POLICY_NAMES = [
    "random_untrained",
    "instinct_only",
    "heuristic_expert",
]


def _stage1_config(*, enable_instinct_rewards: bool = True) -> LabConfig:
    base = LabConfig()
    return apply_training_profile(
        base,
        enable_instinct_rewards=enable_instinct_rewards,
        curriculum_stage=1,
    )


def _run_policy_on_seeds(
    policy: object,
    config: LabConfig,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Run policy on all seeds and return aggregate metrics."""
    results = []
    for seed in seeds:
        result = SheepdogEnvironment(config).run_policy(policy, seed=seed, capture_replay=False)
        results.append(result)

    success_rate = fmean(1.0 if r.stats.success else 0.0 for r in results)
    timeout_rate = fmean(1.0 if r.stats.timeout else 0.0 for r in results)
    avg_sheep_penned = fmean(r.stats.sheep_penned for r in results)
    avg_reward = fmean(r.stats.reward_total for r in results)
    avg_distance = fmean(r.stats.final_avg_distance_to_pen for r in results)
    return {
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "avg_sheep_penned": avg_sheep_penned,
        "avg_reward": avg_reward,
        "avg_distance_to_pen": avg_distance,
    }


def _load_latest_trained_checkpoint(config: LabConfig) -> tuple[object | None, str]:
    """Return (policy, description) for the latest trained checkpoint, or (None, reason)."""
    output_root = Path(config.training.output_dir)
    summary_path = output_root / "training-summary.json"
    if not summary_path.exists():
        return None, "no training-summary.json found"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "training-summary.json could not be read"

    checkpoints = summary.get("checkpoints", [])
    if not checkpoints:
        return None, "training-summary.json has no checkpoints"

    latest = checkpoints[-1]
    total_trained = int(latest.get("total_training_episodes", 0))
    if total_trained == 0:
        return None, "latest checkpoint has total_training_episodes=0 (baseline only)"

    checkpoint_episode = int(latest.get("checkpoint_episode", 0))
    policy_name = str(latest.get("policy_name", "trained_policy"))
    trainer_type = str(latest.get("trainer_type", ""))
    policy_type = str(latest.get("policy_type", ""))
    curriculum_stage = (
        latest.get("environment_config", {}).get("curriculum_stage")
        or latest.get("reward_config", {}).get("instincts", {}).get("curriculum_stage")
        or 0
    )
    try:
        policy = load_playable_policy(
            config,
            checkpoint_episode=checkpoint_episode,
            policy_mode=policy_name,
        )
        desc = (
            f"checkpoint {checkpoint_episode} | {trainer_type} | {policy_type} "
            f"| trained_eps={total_trained} | trained_stage={curriculum_stage}"
        )
        return policy, desc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return None, f"could not load checkpoint {checkpoint_episode}: {exc}"


def _print_result(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"  {label:<40} "
        f"success={metrics['success_rate']:.0%}  "
        f"penned={metrics['avg_sheep_penned']:.2f}  "
        f"reward={metrics['avg_reward']:+.1f}  "
        f"timeout={metrics['timeout_rate']:.0%}  "
        f"dist={metrics['avg_distance_to_pen']:.1f}"
    )


def _diagnose(label: str, metrics: dict[str, Any]) -> list[str]:
    """Return a list of diagnostic notes for a policy result."""
    notes = []
    if metrics["success_rate"] == 0.0:
        notes.append(f"{label}: 0% success - cannot pen any sheep")
        if metrics["avg_distance_to_pen"] > 20:
            notes.append(
                f"  -> avg distance {metrics['avg_distance_to_pen']:.1f} is very high: "
                "sheep may not be moving toward pen at all"
            )
        if metrics["timeout_rate"] > 0.8:
            notes.append(
                f"  -> timeout rate {metrics['timeout_rate']:.0%}: "
                "episodes hit max_steps without progress"
            )
        if metrics["avg_reward"] < -50:
            notes.append(
                f"  -> avg reward {metrics['avg_reward']:+.1f}: "
                "heavy negative reward each episode - check penalty balance"
            )
    return notes


@pytest.mark.slow
def test_stage1_policy_benchmark() -> None:
    """Benchmark all policy types on Stage 1 and print a diagnostic table."""
    config = _stage1_config(enable_instinct_rewards=True)
    print(
        f"\n\nStage 1 benchmark  "
        f"(grid={config.environment.width}x{config.environment.height}, "
        f"dogs={config.environment.dogs}, sheep={config.environment.sheep}, "
        f"instincts={config.rewards.instincts.enable_instinct_rewards})"
    )
    print(f"  {'Policy':<40} success   penned  reward    timeout  dist-to-pen")
    print("  " + "-" * 80)

    all_diagnoses: list[str] = []
    heuristic_success = 0.0

    for policy_name in POLICY_NAMES:
        policy = create_policy_from_name(policy_name)
        metrics = _run_policy_on_seeds(policy, config, STAGE1_SEEDS)
        _print_result(policy_name, metrics)
        all_diagnoses.extend(_diagnose(policy_name, metrics))
        if policy_name == "heuristic_expert":
            heuristic_success = metrics["success_rate"]

    # Try latest trained checkpoint
    trained_policy, trained_desc = _load_latest_trained_checkpoint(config)
    if trained_policy is not None:
        metrics = _run_policy_on_seeds(trained_policy, config, STAGE1_SEEDS)
        _print_result(f"latest_trained ({trained_desc})", metrics)
        all_diagnoses.extend(_diagnose(f"latest_trained ({trained_desc})", metrics))
    else:
        print(f"  {'latest_trained_checkpoint':<40} SKIPPED: {trained_desc}")

    if all_diagnoses:
        print("\nDiagnosis:")
        for note in all_diagnoses:
            print(f"  {note}")

    if heuristic_success == 0.0:
        print(
            "\nSTOP: heuristic_expert cannot solve Stage 1. "
            "Diagnose environment/pen geometry/reward before adjusting PPO."
        )
        pytest.fail(
            "heuristic_expert achieved 0% success on Stage 1 — "
            "environment or pen geometry needs investigation before training"
        )
    else:
        print(
            f"\nheuristic_expert passes Stage 1 ({heuristic_success:.0%}) — "
            "environment is solvable; training may need more episodes or config tuning."
        )


@pytest.mark.slow
def test_stage1_reward_breakdown() -> None:
    """Verify instinct reward terms are active during a Stage 1 heuristic episode."""
    from dataclasses import replace as dc_replace

    config = _stage1_config(enable_instinct_rewards=True)
    debug_config = dc_replace(
        config,
        rewards=dc_replace(
            config.rewards,
            instincts=dc_replace(config.rewards.instincts, debug_reward_breakdown=True),
        ),
    )

    policy = HeuristicExpertPolicy()
    env = SheepdogEnvironment(debug_config)
    result = env.run_policy(policy, seed=11, capture_replay=True)

    instinct_terms = ["pressure_zone", "safe_pressure", "target_progress"]
    nonzero: dict[str, int] = {term: 0 for term in instinct_terms}

    for record in result.replay:
        for term in instinct_terms:
            val = getattr(record.reward, term, 0.0)
            if abs(val) > 1e-9:
                nonzero[term] += 1

    print("\nInstinct reward term non-zero step counts (Stage 1, heuristic_expert, seed=11):")
    for term, count in nonzero.items():
        print(f"  {term:<25}: {count}/{len(result.replay)} steps")

    all_zero = all(count == 0 for count in nonzero.values())
    if all_zero:
        pytest.fail(
            "All instinct reward terms are zero throughout the episode. "
            "enable_instinct_rewards may not be reaching the reward computer, "
            "or target_position/flock_centroid are not being set."
        )


@pytest.mark.slow
def test_positive_reward_when_dog_behind_sheep() -> None:
    """Confirm model receives positive reward when dog gets behind sheep toward pen."""
    config = _stage1_config(enable_instinct_rewards=True)
    policy = HeuristicExpertPolicy()
    env = SheepdogEnvironment(config)
    result = env.run_policy(policy, seed=11, capture_replay=True)

    positive_progress_steps = sum(1 for r in result.replay if r.reward.progress_to_pen > 0.0)
    positive_pressure_steps = sum(1 for r in result.replay if r.reward.pressure_zone > 0.0)
    total_steps = len(result.replay)

    print(f"\nReward signal check (Stage 1, heuristic_expert, seed=11, {total_steps} steps):")
    print(f"  progress_to_pen > 0:  {positive_progress_steps}/{total_steps} steps")
    print(f"  pressure_zone > 0:    {positive_pressure_steps}/{total_steps} steps")
    print(f"  final success: {result.stats.success}")
    print(f"  total reward:  {result.stats.reward_total:+.1f}")

    assert positive_progress_steps > 0, (
        "No steps with positive progress_to_pen reward — sheep are never moving toward the pen"
    )
