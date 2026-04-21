"""Random baseline policy."""

from __future__ import annotations

import random

from sheepdog.policies.base import Action


class RandomPolicy:
    """Choose uniformly among legal actions."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        for dog_index in range(environment.dog_count):
            mask = environment.action_mask_for_dog(dog_index)
            legal_actions = [action for action, allowed in mask.items() if allowed]
            actions.append(self._rng.choice(legal_actions))
        return actions
