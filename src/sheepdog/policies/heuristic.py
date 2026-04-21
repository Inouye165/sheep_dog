"""Deterministic heuristic policy for the shared dog team."""

from __future__ import annotations

from sheepdog.policies.base import Action


class HeuristicPolicy:
    """Select the highest-scoring legal action for each dog."""

    name = "heuristic"

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        for dog_index in range(environment.dog_count):
            mask = environment.action_mask_for_dog(dog_index)
            candidates = [action for action, allowed in mask.items() if allowed]
            ranked = sorted(
                candidates,
                key=lambda action: environment.score_action_for_dog(dog_index, action),
                reverse=True,
            )
            actions.append(ranked[0])
        return actions
