"""Policy interfaces and action names."""

from __future__ import annotations

from typing import Literal, Protocol

Action = Literal["up", "down", "left", "right", "wait"]
PolicyMode = Literal[
    "random_untrained",
    "instinct_only",
    "heuristic_expert",
    "trained_policy",
]


class Policy(Protocol):
    """Shared policy contract for all dog controllers."""

    name: str

    def select_actions(self, environment: object) -> list[Action]:
        """Return one action per dog."""
