"""Scenario sampler for mixing difficult training scenarios with random starts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from sheepdog.config import EnvironmentConfig, TrainingConfig
from sheepdog.evaluation.scenarios import Scenario
from sheepdog.training.training_scenarios import (
    get_scenario_builder,
    list_scenario_types,
)

# Standard held-out evaluation seeds that must never be used directly in training
STANDARD_EVALUATION_SEEDS = frozenset({11, 23, 37, 41, 53, 59, 61, 67, 71, 73, 91, 92, 93, 94, 95})

# Map evaluation seed / failure archetypes to procedural hard scenario classes
SEED_TO_FAILURE_CLASS: dict[int, str] = {
    11: "subpen_flock",
    53: "gate_wall_flock",
    41: "isolated_stray",
    37: "isolated_stray",
    71: "isolated_stray",
}


@dataclass(frozen=True, slots=True)
class ScenarioSelection:
    """Result of scenario sampling: either a seed for random reset or a Scenario object."""

    scenario_type: str
    seed: int
    scenario: Scenario | None = None


@dataclass(slots=True)
class ScenarioSampler:
    """Reproducible scenario sampler for training.

    Supports both:
    1. Static scenario mix (``scenario_training_enabled``).
    2. Dynamic failure-directed training with anti-forgetting decay
       (``failure_directed_training_enabled``).
    """

    config: TrainingConfig
    env_config: EnvironmentConfig
    rng: Random = field(init=False)
    _episode_counter: int = field(init=False, default=0)
    _usage_counts: dict[str, int] = field(init=False, default_factory=dict)
    _failure_weights: dict[str, float] = field(init=False, default_factory=dict)
    _held_out_seeds: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the RNG with the training seed for reproducibility."""
        object.__setattr__(self, "rng", Random(self.config.train_seed))
        eval_seeds = set(STANDARD_EVALUATION_SEEDS)
        if hasattr(self.config, "evaluation_seeds") and self.config.evaluation_seeds:
            eval_seeds.update(self.config.evaluation_seeds)
        if hasattr(self.config, "candidate_evaluation_seeds") and self.config.candidate_evaluation_seeds:
            eval_seeds.update(self.config.candidate_evaluation_seeds)
        object.__setattr__(self, "_held_out_seeds", frozenset(eval_seeds))

    def _safe_training_seed(self, raw_seed: int) -> int:
        """Ensure training seeds never collide with held-out evaluation seeds."""
        seed = int(raw_seed)
        while seed in self._held_out_seeds:
            seed += 100_000
        return seed

    def sample(self, episode_index: int) -> ScenarioSelection:
        """Sample a scenario type for the given episode index.

        Uses the RNG to ensure reproducibility: the same seed produces
        the same sequence of scenario selections.
        """
        raw_seed = self.config.train_seed + episode_index
        seed = self._safe_training_seed(raw_seed)

        # 1. Failure-Directed Training (Dynamic closed-loop targeting)
        if getattr(self.config, "failure_directed_training_enabled", False):
            active_weights = {
                k: v for k, v in self._failure_weights.items()
                if v >= getattr(self.config, "failure_directed_min_weight", 0.05)
            }
            if active_weights:
                target_ratio = float(getattr(self.config, "failure_directed_target_ratio", 0.25))
                if self.rng.random() < target_ratio:
                    scenario_type = self._weighted_choice(active_weights)
                    builder = get_scenario_builder(scenario_type)
                    scenario = builder(seed=seed, config=self.env_config)
                    self._increment_usage(scenario_type)
                    return ScenarioSelection(
                        scenario_type=scenario_type,
                        seed=seed,
                        scenario=scenario,
                    )
            # Default normal training when not sampling a targeted hard scenario
            self._increment_usage("normal")
            return ScenarioSelection(scenario_type="normal", seed=seed, scenario=None)

        # 2. Static Scenario Training (Open-loop mix)
        if self.config.scenario_training_enabled:
            scenario_type = self._weighted_choice(self.config.scenario_mix)
            if scenario_type in ("random", "normal"):
                self._increment_usage("normal")
                return ScenarioSelection(scenario_type="normal", seed=seed, scenario=None)

            builder = get_scenario_builder(scenario_type)
            scenario = builder(seed=seed, config=self.env_config)
            self._increment_usage(scenario_type)
            return ScenarioSelection(
                scenario_type=scenario_type,
                seed=seed,
                scenario=scenario,
            )

        # 3. Standard Unmodified Training
        self._increment_usage("normal")
        return ScenarioSelection(scenario_type="normal", seed=seed, scenario=None)

    def update_from_evaluation(self, evaluation_records: Sequence[Any]) -> dict[str, float]:
        """Update failure class targeting weights based on evaluation results.

        Failed scenario classes are boosted to full weight (1.0).
        Previously failed classes that passed in this evaluation decay exponentially
        by ``failure_directed_decay_rate`` to prevent catastrophic forgetting.
        """
        if not evaluation_records:
            return dict(self._failure_weights)

        failed_classes: set[str] = set()

        for record in evaluation_records:
            def get_val(key: str, default: Any = None) -> Any:
                if isinstance(record, Mapping):
                    return record.get(key, default)
                return getattr(record, key, default)

            success = bool(get_val("success", False))
            if success:
                continue

            seed_val = get_val("seed")
            if seed_val is not None and int(seed_val) in SEED_TO_FAILURE_CLASS:
                failed_classes.add(SEED_TO_FAILURE_CLASS[int(seed_val)])
                continue

            # Fallback to diagnostic signature & spatial classification
            stop_reason = str(get_val("stop_reason") or "")
            final_zone = str(get_val("final_sheep_zone") or "")
            spawn_mode = str(get_val("spawn_mode") or "")
            gate_fail_steps = int(get_val("gate_corridor_failure_steps", 0) or 0)

            if gate_fail_steps >= 15 or final_zone == "top_wall":
                failed_classes.add("gate_wall_flock")
            elif stop_reason == "no-progress" or final_zone in ("bottom_right", "right_wall"):
                failed_classes.add("subpen_flock")
            elif spawn_mode in ("nearby_stray", "farther_stray", "two_strays"):
                failed_classes.add("isolated_stray")
            else:
                # Default to subpen flock for unclassified Stage 7 failures
                failed_classes.add("subpen_flock")

        decay_rate = float(getattr(self.config, "failure_directed_decay_rate", 0.60))
        min_weight = float(getattr(self.config, "failure_directed_min_weight", 0.05))

        # Update all registered / existing failure weights
        updated_weights: dict[str, float] = {}

        # 1. Boost active failures
        for failure_cls in failed_classes:
            updated_weights[failure_cls] = 1.0

        # 2. Decay previously active classes that passed
        for existing_cls, weight in self._failure_weights.items():
            if existing_cls not in failed_classes:
                decayed = weight * decay_rate
                if decayed >= min_weight:
                    updated_weights[existing_cls] = round(decayed, 4)

        self._failure_weights = updated_weights
        return dict(self._failure_weights)

    def set_failure_weights(self, weights: dict[str, float]) -> None:
        """Explicitly set active failure weights (e.g., synchronized across workers)."""
        self._failure_weights = {
            str(k): float(v) for k, v in weights.items() if float(v) > 0.0
        }

    def get_failure_weights(self) -> dict[str, float]:
        """Return the current active failure weights."""
        return dict(self._failure_weights)

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        """Choose a scenario type according to weighted probabilities."""
        total = sum(weights.values())
        if total <= 0:
            return "normal"

        normalized = {k: v / total for k, v in weights.items()}
        r = self.rng.random()
        cumulative = 0.0
        last_scenario_type = "normal"
        for scenario_type, weight in normalized.items():
            last_scenario_type = scenario_type
            cumulative += weight
            if r <= cumulative:
                return scenario_type

        return last_scenario_type

    def _increment_usage(self, scenario_type: str) -> None:
        """Track scenario type usage for observability."""
        self._usage_counts[scenario_type] = self._usage_counts.get(scenario_type, 0) + 1

    def get_usage_counts(self) -> dict[str, int]:
        """Return the count of episodes started from each scenario type."""
        return dict(self._usage_counts)

    def get_usage_summary(self) -> dict[str, Any]:
        """Return a summary of scenario usage statistics and failure-directed telemetry."""
        total = sum(self._usage_counts.values())
        normal_episodes = self._usage_counts.get("normal", 0) + self._usage_counts.get("random", 0)
        targeted_episodes = sum(
            v for k, v in self._usage_counts.items() if k not in ("normal", "random")
        )

        return {
            "total_episodes": total,
            "normal_episodes": normal_episodes,
            "targeted_episodes": targeted_episodes,
            "normal_percentage": (normal_episodes / total * 100.0) if total > 0 else 100.0,
            "targeted_percentage": (targeted_episodes / total * 100.0) if total > 0 else 0.0,
            "scenario_counts": dict(self._usage_counts),
            "scenario_percentages": {
                k: (v / total) * 100 for k, v in self._usage_counts.items()
            } if total > 0 else {},
            "active_failure_weights": dict(self._failure_weights),
            "failure_directed_enabled": bool(getattr(self.config, "failure_directed_training_enabled", False)),
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

    valid_types = set(list_scenario_types()) | {"random", "normal"}
    for key in scenario_mix:
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

