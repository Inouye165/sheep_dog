"""Deterministic heuristic policy for the shared dog team."""

from __future__ import annotations

from sheepdog.policies.base import Action, PolicyMode


class _ScoredPolicy:
    """Select the highest-scoring legal action for each dog."""

    name: PolicyMode

    def __init__(self, policy_mode: PolicyMode) -> None:
        self.name = policy_mode

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        reserved_positions: set[object] = set()
        if hasattr(environment, "prepare_policy_step"):
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
