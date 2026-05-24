"""Dynamic team role assignment for cooperative sheepdog play."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from sheepdog.config import EnvironmentConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """Role and target for a dog on the current step."""

    dog_index: int
    role: DogRole
    target: Point
    target_sheep_index: int | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StrategySnapshot:
    """Derived flock state used for role assignment and metrics."""

    flock_center: Point | None
    flock_spread: float
    average_distance_to_pen: float
    stray_sheep_index: int | None
    near_pen: bool
    wall_pressure: bool


_LATERAL_DEAD_ZONE: int = 3


class TeamStrategy:
    """Assign dynamic cooperative roles from current flock state."""

    def __init__(self, width: int, height: int, config: EnvironmentConfig | None = None) -> None:
        self._width = width
        self._height = height
        self._config = config or EnvironmentConfig(width=width, height=height)
        # Sticky pen-approach direction: only updated when the flock is clearly
        # outside a dead zone around the pen axis.  This prevents LEFT/RIGHT
        # flanker targets from swapping every step when the flock drifts ±1 cell
        # across the pen centre line (the primary cause of dog oscillation).
        self._sticky_pen_dx: int = 1
        self._sticky_pen_dy: int = 0

    def assign_roles(
        self,
        dogs: list[DogState],
        sheep: list[SheepState],
        pen: Pen,
    ) -> tuple[dict[int, RoleAssignment], StrategySnapshot]:
        unpenned = [animal for animal in sheep if not animal.penned]
        if not dogs:
            return {}, StrategySnapshot(None, 0.0, 0.0, None, False, False)
        if not unpenned:
            return self._idle_assignments(dogs, pen)

        flock_center = self._flock_center(unpenned)
        flock_spread = self._flock_spread(unpenned, flock_center)
        average_distance_to_pen = fmean(
            animal.position.distance_to(pen.center) for animal in unpenned
        )
        stray = self._stray_sheep(unpenned, flock_center, flock_spread)
        # Blocker is only useful once the flock is close enough to the gate
        # that an escape guard provides real benefit.  Using the raw pen
        # diagonal (max(w,h)+4) was far too broad: for an 18×18 pen it fired
        # at ≤22 units, immediately stealing the only dog from rear-pressure
        # and stalling all herding progress.  gate_approach_distance (default
        # 10) matches the corridor logic and gives the dog time to herd first.
        near_pen = average_distance_to_pen <= self._config.gate_approach_distance
        wall_pressure = any(self._wall_margin(animal.position) <= 2 for animal in unpenned)
        snapshot = StrategySnapshot(
            flock_center=flock_center,
            flock_spread=flock_spread,
            average_distance_to_pen=average_distance_to_pen,
            stray_sheep_index=None if stray is None else stray.index,
            near_pen=near_pen,
            wall_pressure=wall_pressure,
        )

        pen_dx_raw = self._axis_sign(pen.center.x - flock_center.x)
        pen_dy_raw = self._axis_sign(pen.center.y - flock_center.y)
        # Only move the sticky direction when the flock is clearly outside the
        # dead zone.  Within ±_LATERAL_DEAD_ZONE cells of the pen centre line,
        # keep the last known direction to prevent the lateral axis from
        # flipping every step and bouncing LEFT/RIGHT flanker targets.
        if abs(pen.center.x - flock_center.x) >= _LATERAL_DEAD_ZONE:
            self._sticky_pen_dx = pen_dx_raw
        elif self._sticky_pen_dx == 0:
            self._sticky_pen_dx = pen_dx_raw or 1
        if abs(pen.center.y - flock_center.y) >= _LATERAL_DEAD_ZONE:
            self._sticky_pen_dy = pen_dy_raw
        # pen_dx drives the rear-target offset (how far behind the flock to stand).
        # Use the raw horizontal sign so we don't add a false horizontal offset when
        # the pen is directly above/below (pen_dx_raw == 0 → no horizontal shift).
        pen_dx = pen_dx_raw
        pen_dy = self._sticky_pen_dy
        # Lateral is perpendicular to the pen-approach direction.
        # lateral_x = -pen_dy  (always correct: rotates pen direction 90°)
        # lateral_y uses the RAW horizontal sign, not the sticky value.  The
        # sticky is only consulted when the pen genuinely has a horizontal
        # component; when the pen is directly above (pen_dx_raw == 0) the
        # sticky value of 1 would introduce a false diagonal and cluster all
        # three role targets below-and-to-the-side instead of spreading them
        # to the true sides of the flock.
        lateral_x = -pen_dy
        lateral_y = pen_dx_raw
        if lateral_x == 0 and lateral_y == 0:
            lateral_y = 1

        flank_offset = max(4, round(flock_spread) + 3)
        rear_target = Point(
            flock_center.x - pen_dx * 4,
            flock_center.y - pen_dy * 4,
        ).clamp(self._width, self._height)
        # Flankers sit BESIDE the flock (same pen-axis depth as flock_center),
        # not behind-and-to-the-side.  This puts them at the correct lateral
        # position to prevent sideways escapes and avoids clustering all three
        # role targets in the same rear quadrant.
        left_target = Point(
            flock_center.x + lateral_x * flank_offset,
            flock_center.y + lateral_y * flank_offset,
        ).clamp(self._width, self._height)
        right_target = Point(
            flock_center.x - lateral_x * flank_offset,
            flock_center.y - lateral_y * flank_offset,
        ).clamp(self._width, self._height)
        blocker_target = self._blocker_target(flock_center, pen, lateral_x, lateral_y)

        assignments: dict[int, RoleAssignment] = {}
        remaining = list(dogs)
        if stray is not None:
            collector_target = self._collector_target(stray.position, flock_center)
            collector = self._best_dog_for_role(
                remaining,
                collector_target,
                DogRole.COLLECTOR,
            )
            assignments[collector.index] = RoleAssignment(
                dog_index=collector.index,
                role=DogRole.COLLECTOR,
                target=collector_target,
                target_sheep_index=stray.index,
                reason="split_sheep_detected",
            )
            remaining = [dog for dog in remaining if dog.index != collector.index]

        # Only assign a blocker when enough dogs remain to still push the flock.
        # blocker_min_remaining_dogs (default 1) ensures at least one herder
        # keeps applying rear-pressure after the blocker is stationed.
        if near_pen and len(remaining) > self._config.blocker_min_remaining_dogs:
            blocker = self._best_dog_for_role(
                remaining,
                blocker_target,
                DogRole.BLOCKER,
            )
            assignments[blocker.index] = RoleAssignment(
                dog_index=blocker.index,
                role=DogRole.BLOCKER,
                target=blocker_target,
                reason="guard_gap",
            )
            remaining = [dog for dog in remaining if dog.index != blocker.index]

        for role, target, reason in (
            (DogRole.REAR_PRESSURE, rear_target, "push_from_rear"),
            (DogRole.LEFT_FLANKER, left_target, "cover_left_escape"),
            (DogRole.RIGHT_FLANKER, right_target, "cover_right_escape"),
        ):
            if not remaining:
                break
            dog = self._best_dog_for_role(remaining, target, role)
            assignments[dog.index] = RoleAssignment(
                dog_index=dog.index,
                role=role,
                target=target,
                reason=reason,
            )
            remaining = [animal for animal in remaining if animal.index != dog.index]

        for dog in remaining:
            assignments[dog.index] = RoleAssignment(
                dog_index=dog.index,
                role=DogRole.REAR_PRESSURE,
                target=rear_target,
                reason="extra_rear_support",
            )
        return assignments, snapshot

    def _idle_assignments(
        self,
        dogs: list[DogState],
        pen: Pen,
    ) -> tuple[dict[int, RoleAssignment], StrategySnapshot]:
        target = pen.center.clamp(self._width, self._height)
        return (
            {
                dog.index: RoleAssignment(
                    dog_index=dog.index,
                    role=DogRole.REAR_PRESSURE,
                    target=target,
                    reason="no_unpenned_sheep",
                )
                for dog in dogs
            },
            StrategySnapshot(target, 0.0, 0.0, None, True, False),
        )

    def _flock_center(self, sheep: list[SheepState]) -> Point:
        return Point(
            round(fmean(animal.position.x for animal in sheep)),
            round(fmean(animal.position.y for animal in sheep)),
        )

    def _flock_spread(self, sheep: list[SheepState], flock_center: Point) -> float:
        if len(sheep) <= 1:
            return 0.0
        return fmean(animal.position.distance_to(flock_center) for animal in sheep)

    def _stray_sheep(
        self,
        sheep: list[SheepState],
        flock_center: Point,
        flock_spread: float,
    ) -> SheepState | None:
        threshold = max(5.0, flock_spread + 2.0)
        furthest = max(sheep, key=lambda animal: animal.position.distance_to(flock_center))
        nearest_neighbor = min(
            (
                furthest.position.distance_to(other.position)
                for other in sheep
                if other.index != furthest.index
            ),
            default=0.0,
        )
        if (
            furthest.position.distance_to(flock_center) <= threshold
            or nearest_neighbor <= threshold
        ):
            return None
        return furthest

    def _collector_target(self, stray_position: Point, flock_center: Point) -> Point:
        dx = self._axis_sign(stray_position.x - flock_center.x)
        dy = self._axis_sign(stray_position.y - flock_center.y)
        if dx == 0 and dy == 0:
            dx = 1
        return Point(stray_position.x + dx * 2, stray_position.y + dy * 2).clamp(
            self._width,
            self._height,
        )

    def _blocker_target(
        self,
        flock_center: Point,
        pen: Pen,
        lateral_x: int,
        lateral_y: int,
    ) -> Point:
        if pen.opening == "left":
            gate = Point(pen.origin.x - 1, pen.center.y)
        elif pen.opening == "right":
            gate = Point(pen.origin.x + pen.width, pen.center.y)
        elif pen.opening == "top":
            gate = Point(pen.center.x, pen.origin.y - 1)
        else:
            gate = Point(pen.center.x, pen.origin.y + pen.height)
        approach_x = self._axis_sign(gate.x - flock_center.x)
        approach_y = self._axis_sign(gate.y - flock_center.y)
        if approach_x == 0 and approach_y == 0:
            approach_x = -lateral_y or 1
            approach_y = lateral_x
        escape_side = (
            self._axis_sign(
                (flock_center.x - gate.x) * lateral_x + (flock_center.y - gate.y) * lateral_y
            )
            or 1
        )
        return Point(
            gate.x - approach_x + lateral_x * escape_side * 2,
            gate.y - approach_y + lateral_y * escape_side * 2,
        ).clamp(self._width, self._height)

    def _best_dog_for_role(
        self,
        dogs: list[DogState],
        target: Point,
        role: DogRole,
    ) -> DogState:
        return min(
            dogs,
            key=lambda dog: (self._role_assignment_cost(dog, target, role), dog.index),
        )

    def _role_assignment_cost(self, dog: DogState, target: Point, role: DogRole) -> float:
        distance = dog.position.distance_to(target)
        if dog.current_role != role:
            return distance
        # Reward dogs that already hold this role: a base bonus plus a
        # hold-time ramp so role assignments remain stable across small flock
        # shifts. The graduated bonus prevents per-step oscillation between
        # near-equally-good candidates while still letting roles re-shuffle
        # when an event (stray, collector trigger) genuinely demands it.
        bonus = self._role_stickiness_bonus(role)
        hold_steps = max(0, getattr(dog, "steps_in_role", 0))
        # Saturate the hold ramp so it cannot dominate truly bad fits.
        ramp = min(hold_steps, self._config.role_minimum_hold_steps) * 0.5
        cap = self._config.role_stickiness_distance
        if distance > cap * 2.0:
            # Dog has drifted very far from its current target; let a fresh
            # assignment win so the team doesn't strand a role-holder.
            return distance
        return max(0.0, distance - bonus - ramp)

    def _role_stickiness_bonus(self, role: DogRole) -> float:
        if role in {DogRole.LEFT_FLANKER, DogRole.RIGHT_FLANKER}:
            return self._config.flank_role_stickiness_bonus
        if role == DogRole.BLOCKER:
            return self._config.blocker_role_stickiness_bonus
        return self._config.role_stickiness_bonus

    def _wall_margin(self, position: Point) -> int:
        return min(
            position.x,
            self._width - 1 - position.x,
            position.y,
            self._height - 1 - position.y,
        )

    def _axis_sign(self, value: float) -> int:
        return 0 if value == 0 else (1 if value > 0 else -1)
