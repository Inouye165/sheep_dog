"""Shepherd NPC: high-level herding commands and scripted command policy.

Architecture note
-----------------
The shepherd sits above the dog layer.  It does NOT move dogs tile-by-tile;
instead it reads flock state and issues one of the eight semantic commands
defined in ``ShepherdCommand``.  Dogs receive the current command as part
of their observation and must *learn* how to carry it out.

Phase A: scripted command policy (implemented here).
Phase B: learned command policy (architecture prepared – subclass ScriptedShepherd
         and override ``issue_command``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import hypot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sheepdog.environment import SheepdogEnvironment


class ShepherdCommand(StrEnum):
    """High-level herding commands issued by the shepherd NPC."""

    # Group dispersed sheep before attempting a drive.
    GATHER = "gather"
    # Move the compact flock toward the pen.
    DRIVE_TO_PEN = "drive_to_pen"
    # Plug the left escape lane relative to the pen approach.
    HOLD_LEFT = "hold_left"
    # Plug the right escape lane relative to the pen approach.
    HOLD_RIGHT = "hold_right"
    # Any escape is detected; block generically.
    BLOCK_ESCAPE = "block_escape"
    # Flock is near the pen – push from behind to funnel sheep in.
    APPLY_PRESSURE = "apply_pressure"
    # Dogs are too close; create space so sheep settle.
    BACK_OFF = "back_off"
    # All sheep penned; dogs may stop.
    STOP = "stop"


# Stable ordering used for one-hot encoding – do not reorder.
COMMAND_ORDER: tuple[ShepherdCommand, ...] = (
    ShepherdCommand.GATHER,
    ShepherdCommand.DRIVE_TO_PEN,
    ShepherdCommand.HOLD_LEFT,
    ShepherdCommand.HOLD_RIGHT,
    ShepherdCommand.BLOCK_ESCAPE,
    ShepherdCommand.APPLY_PRESSURE,
    ShepherdCommand.BACK_OFF,
    ShepherdCommand.STOP,
)

COMMAND_INDEX: dict[ShepherdCommand, int] = {cmd: i for i, cmd in enumerate(COMMAND_ORDER)}


@dataclass(frozen=True, slots=True)
class ShepherdContext:
    """Derived flock metrics used by the shepherd command policy."""

    flock_center_x: float
    flock_center_y: float
    flock_spread: float
    average_distance_to_pen: float
    sheep_penned: int
    total_sheep: int
    # Fraction (0-1) of the overpressure threshold: dogs too close to sheep.
    average_dog_sheep_distance: float
    # Whether any sheep are clearly drifting left or right of the pen approach.
    escape_left: bool
    escape_right: bool
    near_pen: bool


def _fmean(values: list[float]) -> float:
    """Return arithmetic mean; 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


class ScriptedShepherd:
    """Scripted Phase-A shepherd that issues high-level commands from flock state.

    This is intentionally simple and rule-based.  It serves as:
      - a stable baseline for training comparison
      - the ``heuristic_shepherd`` policy slot
      - the default shepherd during neural dog training (Phase A)

    To implement a learned shepherd (Phase B) subclass this and override
    ``issue_command(environment)``.
    """

    # Spread threshold below which the flock is considered compact.
    COMPACT_SPREAD: float = 5.0
    # Distance to pen center at which the flock is considered "near pen".
    NEAR_PEN_DISTANCE: float = 14.0
    # Average dog-to-sheep distance threshold for overpressure detection.
    OVERPRESSURE_DISTANCE: float = 1.8
    # Lateral offset (grid cells) required to signal a left/right escape.
    ESCAPE_LATERAL_THRESHOLD: float = 3.0

    def __init__(self) -> None:
        self._last_command: ShepherdCommand = ShepherdCommand.GATHER

    @property
    def last_command(self) -> ShepherdCommand:
        """The command issued on the most recent step."""
        return self._last_command

    def issue_command(self, environment: SheepdogEnvironment) -> ShepherdCommand:
        """Derive and return the current high-level herding command.

        The command is stored in ``last_command`` so policies can read it
        without calling this method a second time.
        """
        ctx = self._build_context(environment)
        command = self._decide(ctx, environment)
        self._last_command = command
        return command

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, environment: SheepdogEnvironment) -> ShepherdContext:
        """Compute the shepherd's situational context from the environment."""
        unpenned = [s for s in environment.sheep if not s.penned]
        penned_count = sum(1 for s in environment.sheep if s.penned)
        total = len(environment.sheep)
        pen_cx = environment.pen.center.x
        pen_cy = environment.pen.center.y

        if not unpenned:
            return ShepherdContext(
                flock_center_x=float(pen_cx),
                flock_center_y=float(pen_cy),
                flock_spread=0.0,
                average_distance_to_pen=0.0,
                sheep_penned=penned_count,
                total_sheep=total,
                average_dog_sheep_distance=99.0,
                escape_left=False,
                escape_right=False,
                near_pen=True,
            )

        flock_xs = [s.position.x for s in unpenned]
        flock_ys = [s.position.y for s in unpenned]
        fc_x = _fmean(flock_xs)
        fc_y = _fmean(flock_ys)

        spread = _fmean(
            [hypot(s.position.x - fc_x, s.position.y - fc_y) for s in unpenned]
        )
        avg_dist_pen = _fmean(
            [hypot(s.position.x - pen_cx, s.position.y - pen_cy) for s in unpenned]
        )
        near_pen = avg_dist_pen <= self.NEAR_PEN_DISTANCE

        # Determine pen approach axis (flock → pen vector)
        dpx = pen_cx - fc_x
        dpy = pen_cy - fc_y
        # Lateral axis is perpendicular to approach vector.
        # If approach is mostly horizontal (|dpx| >= |dpy|), lateral = y-axis.
        if abs(dpx) >= abs(dpy):
            lateral_offsets = [s.position.y - pen_cy for s in unpenned]
        else:
            lateral_offsets = [s.position.x - pen_cx for s in unpenned]

        escape_left = any(off < -self.ESCAPE_LATERAL_THRESHOLD for off in lateral_offsets)
        escape_right = any(off > self.ESCAPE_LATERAL_THRESHOLD for off in lateral_offsets)

        # Average min-distance from each dog to nearest sheep
        dog_sheep_distances: list[float] = []
        for dog in environment.dogs:
            dists = [
                hypot(dog.position.x - s.position.x, dog.position.y - s.position.y)
                for s in unpenned
            ]
            if dists:
                dog_sheep_distances.append(min(dists))
        avg_dog_sheep = _fmean(dog_sheep_distances) if dog_sheep_distances else 99.0

        return ShepherdContext(
            flock_center_x=fc_x,
            flock_center_y=fc_y,
            flock_spread=spread,
            average_distance_to_pen=avg_dist_pen,
            sheep_penned=penned_count,
            total_sheep=total,
            average_dog_sheep_distance=avg_dog_sheep,
            escape_left=escape_left,
            escape_right=escape_right,
            near_pen=near_pen,
        )

    def _decide(
        self,
        ctx: ShepherdContext,
        environment: SheepdogEnvironment,
    ) -> ShepherdCommand:
        """Map context to the highest-priority command."""
        # All sheep in pen → stop.
        if ctx.sheep_penned >= ctx.total_sheep:
            return ShepherdCommand.STOP

        # Flock badly scattered → gather first.
        if ctx.flock_spread > self.COMPACT_SPREAD:
            return ShepherdCommand.GATHER

        # Sheep near pen – handle escape lanes or funnel in.
        if ctx.near_pen:
            if ctx.escape_left and ctx.escape_right:
                return ShepherdCommand.BLOCK_ESCAPE
            if ctx.escape_left:
                return ShepherdCommand.HOLD_LEFT
            if ctx.escape_right:
                return ShepherdCommand.HOLD_RIGHT
            # Dogs too close – create space so sheep settle into pen.
            if ctx.average_dog_sheep_distance < self.OVERPRESSURE_DISTANCE:
                return ShepherdCommand.BACK_OFF
            return ShepherdCommand.APPLY_PRESSURE

        # Flock is compact but far from pen → drive it.
        return ShepherdCommand.DRIVE_TO_PEN
