"""Role-aware observation building for shared sheepdog policies."""

# pylint: disable=protected-access
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import TYPE_CHECKING, Any

from sheepdog.entities import DogRole, Point
from sheepdog.shepherd import COMMAND_INDEX, COMMAND_ORDER, ShepherdCommand
from sheepdog.team_strategy import RoleAssignment

if TYPE_CHECKING:
    from sheepdog.environment import SheepdogEnvironment


ROLE_ORDER: tuple[DogRole, ...] = (
    DogRole.REAR_PRESSURE,
    DogRole.LEFT_FLANKER,
    DogRole.RIGHT_FLANKER,
    DogRole.COLLECTOR,
    DogRole.BLOCKER,
)

# Fixed observation-vector capacity across all curriculum stages.
# Changing either constant is a breaking architecture change that requires
# clearing training state and retraining from scratch.
MAX_SHEEP_SLOTS: int = 6  # highest sheep count used in any curriculum stage
HERD_DOG_SLOTS: int = 3  # highest dog count used in any curriculum stage


@dataclass(frozen=True, slots=True)
class DogObservation:
    """Flat, inspectable role-aware observation for one dog."""

    dog_index: int
    role: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_feature_dict(self) -> dict[str, float]:
        """Return feature values keyed by stable names."""

        return dict(zip(self.feature_names, self.values, strict=True))


class RoleAwareObservationBuilder:
    """Build comparable dog observations for linear and neural policies."""

    def build(self, environment: SheepdogEnvironment, dog_index: int) -> DogObservation:
        """Return a flat observation vector for one dog."""

        environment.prepare_policy_step()
        dog = environment.dogs[dog_index]
        flock_center = environment._flock_center() or dog.position
        pen_center = environment.pen.center
        assignment = environment._role_assignments.get(
            dog_index,
            RoleAssignment(dog_index, dog.current_role, dog.position),
        )
        focus_sheep = self._focus_sheep(environment, dog_index, flock_center)
        stray_sheep = self._stray_sheep(environment)
        diagonal = max(
            1.0,
            hypot(environment.env_config.width - 1, environment.env_config.height - 1),
        )

        feature_names: list[str] = []
        values: list[float] = []

        self._append_point(feature_names, values, "own", dog.position, environment)
        self._append_point(feature_names, values, "pen", pen_center, environment)
        self._append_point(feature_names, values, "flock_center", flock_center, environment)
        self._append_point(feature_names, values, "target", assignment.target, environment)

        feature_names.extend(
            (
                "distance_to_pen",
                "distance_to_flock",
                "distance_to_target",
                "flock_spread",
                "average_distance_to_pen",
                "wall_left",
                "wall_right",
                "wall_top",
                "wall_bottom",
                "blocked_steps",
                "no_progress_steps",
                "revisits_recent_position",
                "two_position_loop",
                "stray_present",
            )
        )
        values.extend(
            (
                dog.position.distance_to(pen_center) / diagonal,
                dog.position.distance_to(flock_center) / diagonal,
                dog.position.distance_to(assignment.target) / diagonal,
                environment._flock_spread() / diagonal,
                environment._average_distance_to_pen() / diagonal,
                dog.position.x / max(1, environment.env_config.width - 1),
                (environment.env_config.width - 1 - dog.position.x)
                / max(1, environment.env_config.width - 1),
                dog.position.y / max(1, environment.env_config.height - 1),
                (environment.env_config.height - 1 - dog.position.y)
                / max(1, environment.env_config.height - 1),
                min(dog.blocked_steps, 10) / 10.0,
                min(environment._no_progress_steps, environment.env_config.no_progress_window)
                / max(1, environment.env_config.no_progress_window),
                1.0
                if environment._revisits_recent_position(dog.recent_positions, dog.position)
                else 0.0,
                1.0 if environment._in_two_position_loop(dog.recent_positions) else 0.0,
                1.0 if stray_sheep is not None else 0.0,
            )
        )

        for role in ROLE_ORDER:
            feature_names.append(f"role_{role.value}")
            values.append(1.0 if assignment.role == role else 0.0)

        self._append_relative_point(
            feature_names,
            values,
            "focus_sheep",
            dog.position,
            None if focus_sheep is None else focus_sheep.position,
            environment,
        )
        feature_names.append("focus_sheep_distance")
        values.append(
            0.0
            if focus_sheep is None
            else dog.position.distance_to(focus_sheep.position) / diagonal
        )
        self._append_relative_point(
            feature_names,
            values,
            "stray_sheep",
            dog.position,
            None if stray_sheep is None else stray_sheep.position,
            environment,
        )

        sorted_sheep = sorted(
            environment.sheep,
            key=lambda sheep: (sheep.penned, dog.position.distance_to(sheep.position), sheep.index),
        )
        for sheep_slot in range(MAX_SHEEP_SLOTS):
            sheep = sorted_sheep[sheep_slot] if sheep_slot < len(sorted_sheep) else None
            prefix = f"sheep_{sheep_slot}"
            self._append_relative_point(
                feature_names,
                values,
                prefix,
                dog.position,
                None if sheep is None else sheep.position,
                environment,
            )
            feature_names.append(f"{prefix}_penned")
            values.append(0.0 if sheep is None else (1.0 if sheep.penned else 0.0))

        other_dogs = [other for other in environment.dogs if other.index != dog_index]
        for teammate_slot in range(HERD_DOG_SLOTS - 1):
            other = other_dogs[teammate_slot] if teammate_slot < len(other_dogs) else None
            prefix = f"other_dog_{teammate_slot}"
            self._append_relative_point(
                feature_names,
                values,
                prefix,
                dog.position,
                None if other is None else other.position,
                environment,
            )

        return DogObservation(
            dog_index=dog_index,
            role=assignment.role.value,
            feature_names=tuple(feature_names),
            values=tuple(values),
            metadata={
                "assignment_reason": assignment.reason,
                "target_sheep_index": assignment.target_sheep_index,
                "stray_sheep_index": environment._strategy_snapshot.stray_sheep_index,
            },
        )

    def _focus_sheep(
        self,
        environment: SheepdogEnvironment,
        dog_index: int,
        _flock_center: Point,
    ) -> Any | None:
        assignment = environment._role_assignments.get(dog_index)
        if assignment is not None and assignment.target_sheep_index is not None:
            return next(
                (
                    sheep
                    for sheep in environment.sheep
                    if sheep.index == assignment.target_sheep_index and not sheep.penned
                ),
                None,
            )
        unpenned = [sheep for sheep in environment.sheep if not sheep.penned]
        if not unpenned:
            return None
        dog = environment.dogs[dog_index]
        return min(
            unpenned,
            key=lambda sheep: (dog.position.distance_to(sheep.position), sheep.index),
        )

    def _stray_sheep(self, environment: SheepdogEnvironment) -> Any | None:
        stray_index = environment._strategy_snapshot.stray_sheep_index
        if stray_index is None:
            return None
        return next((sheep for sheep in environment.sheep if sheep.index == stray_index), None)

    def _append_point(
        self,
        feature_names: list[str],
        values: list[float],
        prefix: str,
        point: Point,
        environment: SheepdogEnvironment,
    ) -> None:
        feature_names.extend((f"{prefix}_x", f"{prefix}_y"))
        values.extend(
            (
                point.x / max(1, environment.env_config.width - 1),
                point.y / max(1, environment.env_config.height - 1),
            )
        )

    def _append_relative_point(
        self,
        feature_names: list[str],
        values: list[float],
        prefix: str,
        origin: Point,
        point: Point | None,
        environment: SheepdogEnvironment,
    ) -> None:
        feature_names.extend((f"{prefix}_dx", f"{prefix}_dy"))
        if point is None:
            values.extend((0.0, 0.0))
            return
        values.extend(
            (
                (point.x - origin.x) / max(1, environment.env_config.width - 1),
                (point.y - origin.y) / max(1, environment.env_config.height - 1),
            )
        )


# Maximum dog count supported in the hierarchical one-hot identity encoding.
# Increasing this changes the observation vector size; retrain if changed.
MAX_DOG_SLOTS: int = 5


class HierarchicalObservationBuilder(RoleAwareObservationBuilder):
    """Extend the role-aware builder with shepherd command + dog identity.

    Additional features appended after the base role-aware vector:

    Shepherd command (one-hot, 8 values)
      shepherd_cmd_<name>  – 1.0 when this command is active

    Dog identity (scalar + one-hot, MAX_DOG_SLOTS + 2 values)
      dog_id_normalized    – dog_index / max(1, dog_count - 1)
      dog_count_normalized – dog_count / max(1, MAX_DOG_SLOTS)
      dog_id_slot_N        – one-hot slot for dog N (N = 0 … MAX_DOG_SLOTS-1)

    These features let a single shared policy develop role-differentiated
    behaviour without hard-coding which dog does what.
    """

    def build_hierarchical(
        self,
        environment: SheepdogEnvironment,
        dog_index: int,
        shepherd_command: ShepherdCommand | None = None,
    ) -> DogObservation:
        """Return a flat observation that includes command + identity appended
        after the standard role-aware features."""
        base = self.build(environment, dog_index)

        feature_names: list[str] = list(base.feature_names)
        values: list[float] = list(base.values)

        # --- Shepherd command (one-hot, 8 values) ---
        cmd = shepherd_command if shepherd_command is not None else ShepherdCommand.GATHER
        cmd_idx = COMMAND_INDEX.get(cmd, 0)
        for i, command in enumerate(COMMAND_ORDER):
            feature_names.append(f"shepherd_cmd_{command.value}")
            values.append(1.0 if i == cmd_idx else 0.0)

        # --- Dog identity ---
        dog_count = max(1, environment.env_config.dogs)
        feature_names.append("dog_id_normalized")
        values.append(dog_index / max(1, dog_count - 1) if dog_count > 1 else 0.0)

        feature_names.append("dog_count_normalized")
        values.append(dog_count / max(1, MAX_DOG_SLOTS))

        for slot in range(MAX_DOG_SLOTS):
            feature_names.append(f"dog_id_slot_{slot}")
            values.append(1.0 if slot == dog_index else 0.0)

        return DogObservation(
            dog_index=dog_index,
            role=base.role,
            feature_names=tuple(feature_names),
            values=tuple(values),
            metadata={
                **base.metadata,
                "shepherd_command": cmd.value,
            },
        )


class EmergentObservationBuilder:
    """Role-free observation builder for emergent PPO training.

    Omits all scripted role labels, assignment targets, and strategy-snapshot
    features so the policy must learn cooperative herding purely from raw
    observations and reward signals.

    Features included:
    - Dog position (own_x, own_y)
    - Dog identity one-hot (dog_id_slot_0 … dog_id_slot_{HERD_DOG_SLOTS-1})
    - Pen centre (pen_x, pen_y)
    - Flock centre (flock_center_x, flock_center_y)
    - Distances: distance_to_pen, distance_to_flock
    - Flock metrics: flock_spread, average_distance_to_pen
    - Wall distances: wall_left, wall_right, wall_top, wall_bottom
    - Anti-loop signals: blocked_steps, no_progress_steps,
      revisits_recent_position, two_position_loop
    - Stray indicator: stray_present
    - Nearest unpenned sheep relative position + distance
    - Farthest-from-pen unpenned sheep relative position + distance
    - Per-slot sheep positions + penned flag (MAX_SHEEP_SLOTS slots, sorted by
      distance from this dog)
    - Other dog positions (HERD_DOG_SLOTS - 1 slots)
    """

    def build(self, environment: SheepdogEnvironment, dog_index: int) -> DogObservation:
        """Return a flat emergent observation vector for one dog."""
        dog = environment.dogs[dog_index]
        flock_center = environment._flock_center() or dog.position  # pylint: disable=protected-access
        pen_center = environment.pen.center
        diagonal = max(
            1.0,
            hypot(environment.env_config.width - 1, environment.env_config.height - 1),
        )

        feature_names: list[str] = []
        values: list[float] = []

        # Own position
        self._append_point(feature_names, values, "own", dog.position, environment)

        # Dog identity one-hot (lets shared policy develop dog-differentiated behaviour)
        for slot in range(HERD_DOG_SLOTS):
            feature_names.append(f"dog_id_slot_{slot}")
            values.append(1.0 if slot == dog_index else 0.0)

        # Pen and flock centre
        self._append_point(feature_names, values, "pen", pen_center, environment)
        self._append_point(feature_names, values, "flock_center", flock_center, environment)

        # Scalar distances and flock metrics
        feature_names.extend(
            (
                "distance_to_pen",
                "distance_to_flock",
                "flock_spread",
                "average_distance_to_pen",
                "wall_left",
                "wall_right",
                "wall_top",
                "wall_bottom",
                "blocked_steps",
                "no_progress_steps",
                "revisits_recent_position",
                "two_position_loop",
                "stray_present",
            )
        )
        unpenned = [s for s in environment.sheep if not s.penned]
        stray_present = 1.0 if self._has_stray(environment, unpenned) else 0.0
        values.extend(
            (
                dog.position.distance_to(pen_center) / diagonal,
                dog.position.distance_to(flock_center) / diagonal,
                environment._flock_spread() / diagonal,  # pylint: disable=protected-access
                environment._average_distance_to_pen() / diagonal,  # pylint: disable=protected-access
                dog.position.x / max(1, environment.env_config.width - 1),
                (environment.env_config.width - 1 - dog.position.x)
                / max(1, environment.env_config.width - 1),
                dog.position.y / max(1, environment.env_config.height - 1),
                (environment.env_config.height - 1 - dog.position.y)
                / max(1, environment.env_config.height - 1),
                min(dog.blocked_steps, 10) / 10.0,
                min(environment._no_progress_steps, environment.env_config.no_progress_window)  # pylint: disable=protected-access
                / max(1, environment.env_config.no_progress_window),
                1.0
                if environment._revisits_recent_position(dog.recent_positions, dog.position)  # pylint: disable=protected-access
                else 0.0,
                1.0 if environment._in_two_position_loop(dog.recent_positions) else 0.0,  # pylint: disable=protected-access
                stray_present,
            )
        )

        # Nearest unpenned sheep (raw, no role assignment)
        nearest = self._nearest_unpenned(dog.position, unpenned)
        self._append_relative_point(
            feature_names, values, "nearest_unpenned", dog.position,
            None if nearest is None else nearest.position, environment,
        )
        feature_names.append("nearest_unpenned_distance")
        values.append(
            0.0 if nearest is None else dog.position.distance_to(nearest.position) / diagonal
        )

        # Farthest-from-pen unpenned sheep (the most critical stray)
        farthest = self._farthest_from_pen(pen_center, unpenned)
        self._append_relative_point(
            feature_names, values, "farthest_unpenned", dog.position,
            None if farthest is None else farthest.position, environment,
        )
        feature_names.append("farthest_unpenned_distance")
        values.append(
            0.0 if farthest is None else dog.position.distance_to(farthest.position) / diagonal
        )

        # Per-slot sheep positions + penned flag (unpenned sheep sorted first)
        sorted_sheep = sorted(
            environment.sheep,
            key=lambda sheep: (sheep.penned, dog.position.distance_to(sheep.position), sheep.index),
        )
        for sheep_slot in range(MAX_SHEEP_SLOTS):
            sheep = sorted_sheep[sheep_slot] if sheep_slot < len(sorted_sheep) else None
            prefix = f"sheep_{sheep_slot}"
            self._append_relative_point(
                feature_names, values, prefix, dog.position,
                None if sheep is None else sheep.position, environment,
            )
            feature_names.append(f"{prefix}_penned")
            values.append(0.0 if sheep is None else (1.0 if sheep.penned else 0.0))

        # Other dog positions
        other_dogs = [other for other in environment.dogs if other.index != dog_index]
        for teammate_slot in range(HERD_DOG_SLOTS - 1):
            other = other_dogs[teammate_slot] if teammate_slot < len(other_dogs) else None
            prefix = f"other_dog_{teammate_slot}"
            self._append_relative_point(
                feature_names, values, prefix, dog.position,
                None if other is None else other.position, environment,
            )

        return DogObservation(
            dog_index=dog_index,
            role="emergent",
            feature_names=tuple(feature_names),
            values=tuple(values),
            metadata={},
        )

    def _nearest_unpenned(self, origin: Any, unpenned: list[Any]) -> Any | None:
        """Return the unpenned sheep closest to *origin*."""
        if not unpenned:
            return None
        return min(unpenned, key=lambda s: (origin.distance_to(s.position), s.index))

    def _farthest_from_pen(self, pen_center: Any, unpenned: list[Any]) -> Any | None:
        """Return the unpenned sheep farthest from the pen centre."""
        if not unpenned:
            return None
        return max(unpenned, key=lambda s: (pen_center.distance_to(s.position), s.index))

    def _has_stray(self, environment: SheepdogEnvironment, unpenned: list[Any]) -> bool:
        """Return True if at least one sheep qualifies as a stray."""
        if len(unpenned) <= 1:
            return False
        flock_center = environment._flock_center()  # pylint: disable=protected-access
        if flock_center is None:
            return False
        spread = environment._flock_spread()  # pylint: disable=protected-access
        threshold = spread * 1.8
        return any(
            s.position.distance_to(flock_center) > threshold for s in unpenned
        )

    def _append_point(
        self,
        feature_names: list[str],
        values: list[float],
        prefix: str,
        point: Point,
        environment: SheepdogEnvironment,
    ) -> None:
        feature_names.extend((f"{prefix}_x", f"{prefix}_y"))
        values.extend(
            (
                point.x / max(1, environment.env_config.width - 1),
                point.y / max(1, environment.env_config.height - 1),
            )
        )

    def _append_relative_point(
        self,
        feature_names: list[str],
        values: list[float],
        prefix: str,
        origin: Point,
        point: Point | None,
        environment: SheepdogEnvironment,
    ) -> None:
        feature_names.extend((f"{prefix}_dx", f"{prefix}_dy"))
        if point is None:
            values.extend((0.0, 0.0))
            return
        values.extend(
            (
                (point.x - origin.x) / max(1, environment.env_config.width - 1),
                (point.y - origin.y) / max(1, environment.env_config.height - 1),
            )
        )
