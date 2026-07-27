"""Deterministic heuristic policy for the shared dog team."""

from __future__ import annotations

from typing import Protocol

from sheepdog.policies.base import Action, PolicyMode


class _PolicyEnvironment(Protocol):
    """Minimal environment contract required by scored heuristic policies."""

    dog_count: int

    def prepare_policy_step(self) -> None:
        """Prepare any per-step role/action caches before policy queries."""
        raise NotImplementedError

    def ranked_actions_for_dog(
        self,
        dog_index: int,
        policy_mode: PolicyMode | None = None,
        weights: object | None = None,
        reserved_positions: set[object] | None = None,
    ) -> list[Action]:
        """Return legal actions for one dog sorted best-first."""
        raise NotImplementedError

    def project_dog_action(self, dog_index: int, action: Action) -> object:
        """Project the destination for one action without mutating state."""
        raise NotImplementedError


class _ScoredPolicy:
    """Select the highest-scoring legal action for each dog."""

    name: PolicyMode

    def __init__(self, policy_mode: PolicyMode) -> None:
        self.name = policy_mode

    def select_actions(
        self, environment: _PolicyEnvironment, deterministic: bool = True
    ) -> list[Action]:
        """Pick the highest-ranked legal action for each dog."""
        del deterministic
        actions: list[Action] = []
        reserved_positions: set[object] = set()
        environment.prepare_policy_step()
        for dog_index in range(environment.dog_count):
            ranked = environment.ranked_actions_for_dog(
                dog_index,
                policy_mode=self.name,
                reserved_positions=reserved_positions,
            )
            choice = ranked[0]
            actions.append(choice)
            reserved_positions.add(environment.project_dog_action(dog_index, choice))
        return actions


class InstinctOnlyPolicy(_ScoredPolicy):
    """Local flock-handling behavior with no knowledge of the pen target."""

    def __init__(self) -> None:
        super().__init__("instinct_only")


class HeuristicExpertPolicy(_ScoredPolicy):
    """Target-aware scripted expert that drives the flock toward the pen."""

    def __init__(self) -> None:
        super().__init__("heuristic_expert")
