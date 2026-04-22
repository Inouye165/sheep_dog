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
    rear_drive: float = 1.05
    flank_control: float = 0.95
    collector_focus: float = 1.15
    blocker_cover: float = 1.0
    anti_stack_penalty: float = 2.0
    oscillation_penalty: float = 0.8

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
            rear_drive=float(payload.get("rear_drive", defaults.rear_drive)),
            flank_control=float(payload.get("flank_control", defaults.flank_control)),
            collector_focus=float(payload.get("collector_focus", defaults.collector_focus)),
            blocker_cover=float(payload.get("blocker_cover", defaults.blocker_cover)),
            anti_stack_penalty=float(
                payload.get("anti_stack_penalty", defaults.anti_stack_penalty)
            ),
            oscillation_penalty=float(
                payload.get("oscillation_penalty", defaults.oscillation_penalty)
            ),
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
            rear_drive=self.rear_drive + rng.uniform(-scale, scale),
            flank_control=self.flank_control + rng.uniform(-scale, scale),
            collector_focus=self.collector_focus + rng.uniform(-scale, scale),
            blocker_cover=self.blocker_cover + rng.uniform(-scale, scale),
            anti_stack_penalty=self.anti_stack_penalty + rng.uniform(-scale, scale),
            oscillation_penalty=self.oscillation_penalty + rng.uniform(-scale, scale),
        )


class TrainableLinearPolicy:
    """A transparent baseline that can be improved through hill climbing."""

    name = "trained_policy"

    def __init__(self, weights: PolicyWeights | None = None) -> None:
        self.weights = weights or PolicyWeights()

    def select_actions(self, environment: object) -> list[Action]:
        actions: list[Action] = []
        reserved_positions: set[object] = set()
        if hasattr(environment, "prepare_policy_step"):
            environment.prepare_policy_step(weights=self.weights)
        for dog_index in range(environment.dog_count):
            ranked = environment.ranked_actions_for_dog(
                dog_index,
                policy_mode=self.name,
                weights=self.weights,
                reserved_positions=reserved_positions,
            )
            choice = ranked[0]
            actions.append(choice)
            reserved_positions.add(environment.project_dog_action(dog_index, choice))
        return actions
