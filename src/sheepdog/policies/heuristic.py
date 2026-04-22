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
        for dog_index in range(environment.dog_count):
            mask = environment.action_mask_for_dog(dog_index, policy_mode=self.name)
            candidates = [action for action, allowed in mask.items() if allowed]
            ranked = sorted(
                candidates,
                key=lambda action: environment.score_action_for_dog(
                    dog_index,
                    action,
                    policy_mode=self.name,
                ),
                reverse=True,
            )
            actions.append(ranked[0])
        return actions


class InstinctOnlyPolicy(_ScoredPolicy):
    """Local flock-handling behavior with no knowledge of the pen target."""

    def __init__(self) -> None:
        super().__init__("instinct_only")


class HeuristicExpertPolicy(_ScoredPolicy):
    """Target-aware scripted expert that drives the flock toward the pen."""

    def __init__(self) -> None:
        super().__init__("heuristic_expert")
