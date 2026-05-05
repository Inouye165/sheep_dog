"""Policy interfaces and action names."""

from __future__ import annotations

from typing import Literal, Protocol

Action = Literal[
    "up",
    "down",
    "left",
    "right",
    "sprint_up",
    "sprint_down",
    "sprint_left",
    "sprint_right",
    "wait",
]
PolicyMode = Literal[
    "random_untrained",
    "random_policy",
    "instinct_only",
    "heuristic_expert",
    "trained_policy",
    "neural_policy",
    # Hierarchical: scripted shepherd + neural dogs (Phase A+)
    "shepherd_neural_dogs",
]
PolicyType = Literal["linear", "neural"]
TrainerType = Literal["hill_climb", "maskable_ppo", "hierarchical_maskable_ppo"]


class Policy(Protocol):
    """Shared policy contract for all dog controllers."""

    name: str

    def select_actions(self, environment: object) -> list[Action]:
        """Return one action per dog."""
