"""Scenario sampler for mixing difficult training scenarios with random starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from sheepdog.config import EnvironmentConfig, TrainingConfig
from sheepdog.evaluation.scenarios import Scenario
from sheepdog.training.training_scenarios import (
    get_scenario_builder,
    list_scenario_types,
)


@dataclass(frozen=True, slots=True)
class ScenarioSelection:
    """Result of scenario sampling: either a seed for random reset or a Scenario object."""

    scenario_type: str
    seed: int
    scenario: Scenario | None = None


@dataclass(slots=True)
class ScenarioSampler:
    """Reproducible scenario sampler for training.

    Samples scenario types according to configured weights and tracks
    usage statistics for observability.
    """

    config: TrainingConfig
    env_config: EnvironmentConfig
    rng: Random = field(init=False)
    _episode_counter: int = field(init=False, default=0)
    _usage_counts: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the RNG with the training seed for reproducibility."""
        object.__setattr__(self, "rng", Random(self.config.train_seed))

    def sample(self, episode_index: int) -> ScenarioSelection:
        """Sample a scenario type for the given episode index.

        Uses the RNG to ensure reproducibility: the same seed produces
        the same sequence of scenario selections.
        """
        if not self.config.scenario_training_enabled:
            # Scenario training disabled: use normal random reset
            seed = self.config.train_seed + episode_index
            self._increment_usage("random")
            return ScenarioSelection(scenario_type="random", seed=seed, scenario=None)

        # Sample scenario type according to configured weights
        scenario_type = self._weighted_choice(self.config.scenario_mix)
        seed = self.config.train_seed + episode_index

        if scenario_type == "random":
            self._increment_usage("random")
            return ScenarioSelection(scenario_type="random", seed=seed, scenario=None)

        # Build the predefined scenario
        builder = get_scenario_builder(scenario_type)
        scenario = builder(seed=seed, config=self.env_config)
        self._increment_usage(scenario_type)
        return ScenarioSelection(
            scenario_type=scenario_type,
            seed=seed,
            scenario=scenario,
        )

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """Choose a scenario type according to weighted probabilities."""
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Scenario mix weights must sum to a positive value")

        normalized = {k: v / total for k, v in weights.items()}
        r = self.rng.random()
        cumulative = 0.0
        last_scenario_type = ""
        for scenario_type, weight in normalized.items():
            last_scenario_type = scenario_type
            cumulative += weight
            if r <= cumulative:
                return scenario_type

        # Fallback to last scenario type (handles floating-point precision)
        return last_scenario_type

    def _increment_usage(self, scenario_type: str) -> None:
        """Track scenario type usage for observability."""
        self._usage_counts[scenario_type] = self._usage_counts.get(scenario_type, 0) + 1

    def get_usage_counts(self) -> dict[str, int]:
        """Return the count of episodes started from each scenario type."""
        return dict(self._usage_counts)

    def get_usage_summary(self) -> dict[str, Any]:
        """Return a summary of scenario usage statistics."""
        total = sum(self._usage_counts.values())
        if total == 0:
            return {"total_episodes": 0, "scenario_counts": {}}

        return {
            "total_episodes": total,
            "scenario_counts": dict(self._usage_counts),
            "scenario_percentages": {
                k: (v / total) * 100 for k, v in self._usage_counts.items()
            },
        }

    def reset_counts(self) -> None:
        """Reset the usage counters (useful for per-checkpoint reporting)."""
        object.__setattr__(self, "_usage_counts", {})


def validate_scenario_mix(scenario_mix: dict[str, float]) -> None:
    """Validate that scenario mix weights are well-formed.

    Raises ValueError if validation fails.
    """
    if not isinstance(scenario_mix, dict):
        raise ValueError("scenario_mix must be a dictionary")

    valid_types = set(list_scenario_types()) | {"random"}
    for key in scenario_mix.keys():
        if key not in valid_types:
            raise ValueError(
                f"Invalid scenario type in mix: {key}. "
                f"Valid types: {sorted(valid_types)}"
            )

    for value in scenario_mix.values():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Scenario mix weight must be numeric, got {type(value)}")
        if value < 0:
            raise ValueError(f"Scenario mix weight must be non-negative, got {value}")

    total = sum(scenario_mix.values())
    if total <= 0:
        raise ValueError("Scenario mix weights must sum to a positive value")
