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
    rear_behind_flock: float = 1.0
    rear_drive_to_pen: float = 1.0
    rear_avoid_overpressure: float = 0.9
    rear_spacing: float = 0.55
    flank_side_control: float = 1.1
    flank_handedness: float = 0.75
    flank_escape_blocking: float = 0.85
    flank_spacing: float = 0.65
    flank_wall_margin: float = 0.35
    collector_stray_focus: float = 1.15
    collector_return_to_flock: float = 0.9
    collector_avoid_scatter: float = 0.85
    collector_rejoin_angle: float = 0.75
    blocker_escape_route_cover: float = 1.0
    blocker_gate_control: float = 1.1
    blocker_funnel_lane: float = 0.9
    blocker_hold_position: float = 0.8
    blocker_spacing: float = 0.55
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
            rear_behind_flock=float(payload.get("rear_behind_flock", defaults.rear_behind_flock)),
            rear_drive_to_pen=float(payload.get("rear_drive_to_pen", defaults.rear_drive_to_pen)),
            rear_avoid_overpressure=float(
                payload.get("rear_avoid_overpressure", defaults.rear_avoid_overpressure)
            ),
            rear_spacing=float(payload.get("rear_spacing", defaults.rear_spacing)),
            flank_side_control=float(
                payload.get("flank_side_control", defaults.flank_side_control)
            ),
            flank_handedness=float(payload.get("flank_handedness", defaults.flank_handedness)),
            flank_escape_blocking=float(
                payload.get("flank_escape_blocking", defaults.flank_escape_blocking)
            ),
            flank_spacing=float(payload.get("flank_spacing", defaults.flank_spacing)),
            flank_wall_margin=float(payload.get("flank_wall_margin", defaults.flank_wall_margin)),
            collector_stray_focus=float(
                payload.get("collector_stray_focus", defaults.collector_stray_focus)
            ),
            collector_return_to_flock=float(
                payload.get("collector_return_to_flock", defaults.collector_return_to_flock)
            ),
            collector_avoid_scatter=float(
                payload.get("collector_avoid_scatter", defaults.collector_avoid_scatter)
            ),
            collector_rejoin_angle=float(
                payload.get("collector_rejoin_angle", defaults.collector_rejoin_angle)
            ),
            blocker_escape_route_cover=float(
                payload.get(
                    "blocker_escape_route_cover",
                    defaults.blocker_escape_route_cover,
                )
            ),
            blocker_gate_control=float(
                payload.get("blocker_gate_control", defaults.blocker_gate_control)
            ),
            blocker_funnel_lane=float(
                payload.get("blocker_funnel_lane", defaults.blocker_funnel_lane)
            ),
            blocker_hold_position=float(
                payload.get("blocker_hold_position", defaults.blocker_hold_position)
            ),
            blocker_spacing=float(payload.get("blocker_spacing", defaults.blocker_spacing)),
            anti_stack_penalty=float(
                payload.get("anti_stack_penalty", defaults.anti_stack_penalty)
            ),
            oscillation_penalty=float(
                payload.get("oscillation_penalty", defaults.oscillation_penalty)
            ),
        )

    def mutated(self, rng: random.Random, scale: float) -> PolicyWeights:
        """Return a copy of these weights with each field perturbed by a small random delta."""
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
            rear_behind_flock=self.rear_behind_flock + rng.uniform(-scale, scale),
            rear_drive_to_pen=self.rear_drive_to_pen + rng.uniform(-scale, scale),
            rear_avoid_overpressure=self.rear_avoid_overpressure + rng.uniform(-scale, scale),
            rear_spacing=self.rear_spacing + rng.uniform(-scale, scale),
            flank_side_control=self.flank_side_control + rng.uniform(-scale, scale),
            flank_handedness=self.flank_handedness + rng.uniform(-scale, scale),
            flank_escape_blocking=self.flank_escape_blocking + rng.uniform(-scale, scale),
            flank_spacing=self.flank_spacing + rng.uniform(-scale, scale),
            flank_wall_margin=self.flank_wall_margin + rng.uniform(-scale, scale),
            collector_stray_focus=self.collector_stray_focus + rng.uniform(-scale, scale),
            collector_return_to_flock=self.collector_return_to_flock + rng.uniform(-scale, scale),
            collector_avoid_scatter=self.collector_avoid_scatter + rng.uniform(-scale, scale),
            collector_rejoin_angle=self.collector_rejoin_angle + rng.uniform(-scale, scale),
            blocker_escape_route_cover=self.blocker_escape_route_cover + rng.uniform(-scale, scale),
            blocker_gate_control=self.blocker_gate_control + rng.uniform(-scale, scale),
            blocker_funnel_lane=self.blocker_funnel_lane + rng.uniform(-scale, scale),
            blocker_hold_position=self.blocker_hold_position + rng.uniform(-scale, scale),
            blocker_spacing=self.blocker_spacing + rng.uniform(-scale, scale),
            anti_stack_penalty=self.anti_stack_penalty + rng.uniform(-scale, scale),
            oscillation_penalty=self.oscillation_penalty + rng.uniform(-scale, scale),
        )


class TrainableLinearPolicy:
    """A transparent baseline that can be improved through hill climbing."""

    name = "trained_policy"

    def __init__(self, weights: PolicyWeights | None = None) -> None:
        self.weights = weights or PolicyWeights()

    def select_actions(self, environment: object) -> list[Action]:
        """Return the best-ranked action for every dog in the environment."""
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
