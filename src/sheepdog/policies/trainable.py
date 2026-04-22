"""Simple trainable shared policy with tunable action scoring weights."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from sheepdog.policies.base import Action


@dataclass(frozen=True, slots=True)
class PolicyWeights:
    """Action scoring weights used by the linear baseline."""

    nearest_sheep: float = 1.4
    flock_center: float = 1.0
    pen_pressure: float = 0.9
    behind_flock: float = 1.1
    team_formation: float = 0.85
    dog_spacing: float = 0.6
    wall_margin: float = 0.35
    wait_bias: float = -1.5

    @classmethod
    def from_dict(cls, payload: dict[str, float] | None) -> PolicyWeights:
        """Build weights from persisted state while preserving new defaults."""

        defaults = cls()
        if not payload:
            return defaults
        return replace(
            defaults,
            nearest_sheep=float(payload.get("nearest_sheep", defaults.nearest_sheep)),
            flock_center=float(payload.get("flock_center", defaults.flock_center)),
            pen_pressure=float(payload.get("pen_pressure", defaults.pen_pressure)),
            behind_flock=float(payload.get("behind_flock", defaults.behind_flock)),
            team_formation=float(payload.get("team_formation", defaults.team_formation)),
            dog_spacing=float(payload.get("dog_spacing", defaults.dog_spacing)),
            wall_margin=float(payload.get("wall_margin", defaults.wall_margin)),
            wait_bias=float(payload.get("wait_bias", defaults.wait_bias)),
        )

    def mutated(self, rng: random.Random, scale: float) -> PolicyWeights:
        return replace(
            self,
            nearest_sheep=self.nearest_sheep + rng.uniform(-scale, scale),
            flock_center=self.flock_center + rng.uniform(-scale, scale),
            pen_pressure=self.pen_pressure + rng.uniform(-scale, scale),
            behind_flock=self.behind_flock + rng.uniform(-scale, scale),
            team_formation=self.team_formation + rng.uniform(-scale, scale),
            dog_spacing=self.dog_spacing + rng.uniform(-scale, scale),
            wall_margin=self.wall_margin + rng.uniform(-scale, scale),
            wait_bias=self.wait_bias + rng.uniform(-scale, scale),
        )


class TrainableLinearPolicy:
    """A transparent baseline that can be improved through hill climbing."""

    name = "trained_policy"

    def __init__(self, weights: PolicyWeights | None = None) -> None:
        self.weights = weights or PolicyWeights()

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        for dog_index in range(environment.dog_count):
            current_dog_index = dog_index
            mask = environment.action_mask_for_dog(
                dog_index,
                policy_mode=self.name,
                weights=self.weights,
            )
            candidates = [action for action, allowed in mask.items() if allowed]
            ranked = sorted(
                candidates,
                key=lambda action: environment.score_action_for_dog(
                    current_dog_index,
                    action,
                    policy_mode=self.name,
                    weights=self.weights,
                ),
                reverse=True,
            )
            actions.append(ranked[0])
        return actions
