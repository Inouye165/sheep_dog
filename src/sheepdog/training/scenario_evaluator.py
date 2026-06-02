"""Evaluation helpers for scenario-based training."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.base import Policy
from sheepdog.training.training_scenarios import get_scenario_builder, list_scenario_types


@dataclass(frozen=True, slots=True)
class ScenarioTypeEvaluationResult:
    """Evaluation results for a single scenario type."""

    scenario_type: str
    seeds: tuple[int, ...]
    average_reward: float
    success_rate: float
    timeout_rate: float
    stopped_rate: float
    average_sheep_penned: float
    average_distance_to_pen: float
    average_flock_spread: float
    average_steps: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "scenario_type": self.scenario_type,
            "seeds": list(self.seeds),
            "average_reward": self.average_reward,
            "success_rate": self.success_rate,
            "timeout_rate": self.timeout_rate,
            "stopped_rate": self.stopped_rate,
            "average_sheep_penned": self.average_sheep_penned,
            "average_distance_to_pen": self.average_distance_to_pen,
            "average_flock_spread": self.average_flock_spread,
            "average_steps": self.average_steps,
        }


def evaluate_policy_on_scenario_types(
    policy: Policy,
    config: LabConfig,
    *,
    evaluation_seeds: tuple[int, ...] = (11, 23, 37, 41, 53),
    scenario_types: tuple[str, ...] | None = None,
) -> dict[str, ScenarioTypeEvaluationResult]:
    """Evaluate a policy against each scenario type separately.

    Args:
        policy: The policy to evaluate.
        config: The lab configuration.
        evaluation_seeds: Seeds to use for evaluation (reproducibility).
        scenario_types: Scenario types to evaluate. If None, evaluates all available types.

    Returns:
        A dictionary mapping scenario type names to evaluation results.
    """
    if scenario_types is None:
        scenario_types = list_scenario_types()

    results: dict[str, ScenarioTypeEvaluationResult] = {}

    for scenario_type in scenario_types:
        if scenario_type == "random":
            # For random, use normal env.reset() with evaluation seeds
            episode_results = [
                SheepdogEnvironment(config).run_policy(policy, seed=seed, capture_replay=False)
                for seed in evaluation_seeds
            ]
        else:
            # For predefined scenarios, build and run scenarios
            builder = get_scenario_builder(scenario_type)
            episode_results = []
            for seed in evaluation_seeds:
                scenario = builder(seed=seed, config=config.environment)
                episode_results.append(
                    SheepdogEnvironment(config).run_policy_on_scenario(
                        policy, scenario, capture_replay=False
                    )
                )

        results[scenario_type] = ScenarioTypeEvaluationResult(
            scenario_type=scenario_type,
            seeds=tuple(result.seed for result in episode_results),
            average_reward=fmean(result.stats.reward_total for result in episode_results),
            success_rate=fmean(1.0 if result.stats.success else 0.0 for result in episode_results),
            timeout_rate=fmean(1.0 if result.stats.timeout else 0.0 for result in episode_results),
            stopped_rate=fmean(1.0 if result.stats.stopped else 0.0 for result in episode_results),
            average_sheep_penned=fmean(result.stats.sheep_penned for result in episode_results),
            average_distance_to_pen=fmean(
                result.final_snapshot.average_distance_to_pen for result in episode_results
            ),
            average_flock_spread=fmean(
                result.final_snapshot.flock_spread for result in episode_results
            ),
            average_steps=fmean(result.stats.steps for result in episode_results),
        )

    return results


def format_scenario_type_comparison(
    results: dict[str, ScenarioTypeEvaluationResult],
) -> str:
    """Format scenario type evaluation results as a readable summary."""
    lines = ["Scenario Type Evaluation Summary", "=" * 50, ""]

    for scenario_type, result in results.items():
        lines.append(f"{scenario_type}:")
        lines.append(f"  Success Rate: {result.success_rate:.2%}")
        lines.append(f"  Average Reward: {result.average_reward:.2f}")
        lines.append(f"  Average Steps: {result.average_steps:.1f}")
        lines.append(f"  Average Sheep Penned: {result.average_sheep_penned:.1f}")
        lines.append(f"  Timeout Rate: {result.timeout_rate:.2%}")
        lines.append(f"  Stopped Rate: {result.stopped_rate:.2%}")
        lines.append("")

    return "\n".join(lines)
