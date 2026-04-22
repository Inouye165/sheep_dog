"""Deterministic heuristic policy for the shared dog team."""

from __future__ import annotations

from sheepdog.policies.base import Action


class HeuristicPolicy:
    """Select the highest-scoring legal action for each dog."""

    name = "heuristic"

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        reserved_positions: set[object] = set()
        if hasattr(environment, "prepare_policy_step"):
            environment.prepare_policy_step()
        for dog_index in range(environment.dog_count):
            ranked = environment.ranked_actions_for_dog(
                dog_index,
                reserved_positions=reserved_positions,
            )
            choice = ranked[0]
            actions.append(choice)
            reserved_positions.add(environment.project_dog_action(dog_index, choice))
        return actions
