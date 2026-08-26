"""Core simulation entities and shared geometry helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import hypot
from typing import Any


class DogRole(StrEnum):
    """Dynamic team roles for coordinated herding."""

    REAR_PRESSURE = "rear_pressure"
    LEFT_FLANKER = "left_flanker"
    RIGHT_FLANKER = "right_flanker"
    COLLECTOR = "collector"
    BLOCKER = "blocker"


@dataclass(frozen=True, slots=True)
class Point:
    """A grid position."""

    x: int
    y: int

    def clamp(self, width: int, height: int) -> Point:
        return Point(max(0, min(self.x, width - 1)), max(0, min(self.y, height - 1)))

    def distance_to(self, other: Point) -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def manhattan_distance_to(self, other: Point) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def step_toward(self, target: Point) -> Point:
        dx = 0 if self.x == target.x else (1 if target.x > self.x else -1)
        dy = 0 if self.y == target.y else (1 if target.y > self.y else -1)
        if abs(target.x - self.x) >= abs(target.y - self.y):
            return Point(self.x + dx, self.y)
        return Point(self.x, self.y + dy)


@dataclass(slots=True)
class DogState:
    """State for a single dog agent."""

    index: int
    position: Point
    current_role: DogRole = DogRole.REAR_PRESSURE
    last_action: str = "wait"
    blocked_steps: int = 0
    movement_budget: float = 0.0
    recent_positions: list[Point] = field(default_factory=list)
    steps_in_role: int = 0


SHEEP_PERSONALITIES: tuple[str, ...] = (
    "obedient",
    "pen_fearful",
    "pen_shy",
    "escapist",
    "bold",
)
"""Available sheep personality archetypes.

* ``obedient``    - neutral baseline; no extra bias.
* ``pen_fearful`` - proximity-scaled push away from the pen; strongest near the
  pen opening, fading with distance. Makes the dog work harder to drive sheep in.
* ``pen_shy``     - constant mild push away from the pen center.
* ``escapist``   - when panicked, breaks away from the flock instead of cohering.
* ``bold``       - reduced flee response to dogs beyond close range; the dog
  must get closer (or bark) to drive this sheep effectively.
"""


@dataclass(slots=True)
class SheepState:
    """State for a single sheep agent."""

    index: int
    position: Point
    penned: bool = False
    panic_steps: int = 0
    blocked_steps: int = 0
    movement_budget: float = 0.0
    recent_positions: list[Point] = field(default_factory=list)
    personality: str = "obedient"


@dataclass(frozen=True, slots=True)
class Pen:
    """Goal area for penned sheep, fenced on three sides with one open gate."""

    origin: Point
    width: int
    height: int
    opening: str = "left"

    def contains(self, position: Point) -> bool:
        return (
            self.origin.x <= position.x < self.origin.x + self.width
            and self.origin.y <= position.y < self.origin.y + self.height
        )

    @property
    def center(self) -> Point:
        return Point(self.origin.x + self.width // 2, self.origin.y + self.height // 2)

    def fence_segments(self) -> tuple[tuple[Point, Point], ...]:
        """Return the (start, end) endpoints of the three closed fence sides.

        Coordinates are at cell-corner space: a segment from (x, y) to (x2, y2)
        runs along the boundary between cells. The open side is omitted.
        """
        ox, oy = self.origin.x, self.origin.y
        right = ox + self.width
        bottom = oy + self.height
        sides = {
            "left": (Point(ox, oy), Point(ox, bottom)),
            "right": (Point(right, oy), Point(right, bottom)),
            "top": (Point(ox, oy), Point(right, oy)),
            "bottom": (Point(ox, bottom), Point(right, bottom)),
        }
        return tuple(segment for side, segment in sides.items() if side != self.opening)

    def fence_cells(self) -> frozenset[Point]:
        """Return the cells that act as solid fence walls (cannot be entered)."""
        ox, oy = self.origin.x, self.origin.y
        right = ox + self.width
        bottom = oy + self.height
        cells: set[Point] = set()
        if self.opening != "top":
            for x in range(ox - 1, right + 1):
                cells.add(Point(x, oy - 1))
        if self.opening != "bottom":
            for x in range(ox - 1, right + 1):
                cells.add(Point(x, bottom))
        if self.opening != "left":
            for y in range(oy - 1, bottom + 1):
                cells.add(Point(ox - 1, y))
        if self.opening != "right":
            for y in range(oy - 1, bottom + 1):
                cells.add(Point(right, y))
        # Carve out the corner cells along the open edge so the gate is fully clear.
        if self.opening == "left":
            cells = {cell for cell in cells if cell.x != ox - 1}
        elif self.opening == "right":
            cells = {cell for cell in cells if cell.x != right}
        elif self.opening == "top":
            cells = {cell for cell in cells if cell.y != oy - 1}
        elif self.opening == "bottom":
            cells = {cell for cell in cells if cell.y != bottom}
        return frozenset(cells)


@dataclass(slots=True)
class EpisodeStats:
    """High-level metrics collected during a run."""

    steps: int = 0
    simulated_seconds: float = 0.0
    sheep_penned: int = 0
    timeout: bool = False
    terminated: bool = False
    success: bool = False
    stopped: bool = False
    stop_reason: str = ""
    reward_total: float = 0.0
    no_progress_steps: int = 0
    final_avg_distance_to_pen: float = 0.0
    final_flock_spread: float = 0.0
    final_farthest_distance_to_pen: float = 0.0
    final_farthest_distance_to_flock_center: float = 0.0
    spawn_mode: str = ""
    role_distribution: dict[str, int] = field(default_factory=dict)
    dog_role_occupancy: dict[str, dict[str, int]] = field(default_factory=dict)
    role_switches: int = 0
    collector_activations: int = 0
    blocker_activations: int = 0
    sheep_split_events: int = 0
    cumulative_gate_progress: float = 0.0
    controlled_stall_steps: int = 0
    left_flank_occupancy_steps: int = 0
    right_flank_occupancy_steps: int = 0
    gate_corridor_occupancy_peak: float = 0.0
    gate_corridor_failure_steps: int = 0
    final_reward_breakdown: dict[str, float] = field(default_factory=dict)
    initial_sheep_distance_to_pen: float = 0.0
    min_sheep_distance_to_pen: float = 9999.0
    final_dog_to_sheep_distance: float = 0.0
    final_dog_positions: list[tuple[float, float]] = field(default_factory=list)
    final_sheep_positions: list[tuple[float, float]] = field(default_factory=list)
    pen_position: tuple[float, float] = (0.0, 0.0)
    num_waits: int = 0
    num_sprints: int = 0
    num_invalid_actions: int = 0
    most_frequent_action: str = ""
    oscillation_detected: bool = False
    pen_zone: str = ""
    initial_sheep_zone: str = ""
    final_sheep_zone: str = ""
    corner_steps_total: int = 0
    corner_time_pct: float = 0.0
    wall_steps_total: int = 0
    wall_time_pct: float = 0.0
    corner_stuck_at_end: bool = False
    corner_entered: bool = False
    corner_extracted: bool = False
    spatial_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

