"""Dynamic team role assignment for cooperative sheepdog play."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

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


class TeamStrategy:
    """Assign dynamic cooperative roles from current flock state."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

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
        near_pen = average_distance_to_pen <= max(pen.width, pen.height) + 4
        wall_pressure = any(self._wall_margin(animal.position) <= 2 for animal in unpenned)
        snapshot = StrategySnapshot(
            flock_center=flock_center,
            flock_spread=flock_spread,
            average_distance_to_pen=average_distance_to_pen,
            stray_sheep_index=None if stray is None else stray.index,
            near_pen=near_pen,
            wall_pressure=wall_pressure,
        )

        pen_dx = self._axis_sign(pen.center.x - flock_center.x)
        pen_dy = self._axis_sign(pen.center.y - flock_center.y)
        if pen_dx == 0 and pen_dy == 0:
            pen_dx = 1
        lateral_x = -pen_dy
        lateral_y = pen_dx
        if lateral_x == 0 and lateral_y == 0:
            lateral_y = 1

        flank_offset = max(3, round(flock_spread) + 2)
        rear_target = Point(
            flock_center.x - pen_dx * 4,
            flock_center.y - pen_dy * 4,
        ).clamp(self._width, self._height)
        left_target = Point(
            rear_target.x + lateral_x * flank_offset,
            rear_target.y + lateral_y * flank_offset,
        ).clamp(self._width, self._height)
        right_target = Point(
            rear_target.x - lateral_x * flank_offset,
            rear_target.y - lateral_y * flank_offset,
        ).clamp(self._width, self._height)
        blocker_target = self._blocker_target(flock_center, pen, lateral_x, lateral_y)

        assignments: dict[int, RoleAssignment] = {}
        remaining = list(dogs)
        if stray is not None:
            collector_target = self._collector_target(stray.position, flock_center)
            collector = self._closest_dog(remaining, collector_target)
            assignments[collector.index] = RoleAssignment(
                dog_index=collector.index,
                role=DogRole.COLLECTOR,
                target=collector_target,
                target_sheep_index=stray.index,
                reason="split_sheep_detected",
            )
            remaining = [dog for dog in remaining if dog.index != collector.index]

        if (near_pen or wall_pressure) and remaining:
            blocker = self._closest_dog(remaining, blocker_target)
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
            dog = self._closest_dog(remaining, target)
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
            base = Point(pen.origin.x - 2, pen.center.y)
        elif pen.opening == "right":
            base = Point(pen.origin.x + pen.width + 1, pen.center.y)
        elif pen.opening == "top":
            base = Point(pen.center.x, pen.origin.y - 2)
        else:
            base = Point(pen.center.x, pen.origin.y + pen.height + 1)
        direction = self._axis_sign(flock_center.y - pen.center.y) or 1
        return Point(
            base.x + lateral_x * direction * 2,
            base.y + lateral_y * direction * 2,
        ).clamp(self._width, self._height)

    def _closest_dog(self, dogs: list[DogState], target: Point) -> DogState:
        return min(dogs, key=lambda dog: (dog.position.distance_to(target), dog.index))

    def _wall_margin(self, position: Point) -> int:
        return min(
            position.x,
            self._width - 1 - position.x,
            position.y,
            self._height - 1 - position.y,
        )

    def _axis_sign(self, value: float) -> int:
        return 0 if value == 0 else (1 if value > 0 else -1)