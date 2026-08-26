"""Spatial analytics and stage bottleneck diagnostic engine for herding dynamics.

Discretizes the continuous/grid arena into semantic spatial zones (4 corners,
4 wall boundaries, and center) to monitor corner entrapment, extraction success,
pen placement difficulty, and localized failure modes across stage lifecycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from sheepdog.entities import Pen, Point


ZONE_TOP_LEFT = "top_left"
ZONE_TOP_RIGHT = "top_right"
ZONE_BOTTOM_LEFT = "bottom_left"
ZONE_BOTTOM_RIGHT = "bottom_right"
ZONE_TOP_WALL = "top_wall"
ZONE_BOTTOM_WALL = "bottom_wall"
ZONE_LEFT_WALL = "left_wall"
ZONE_RIGHT_WALL = "right_wall"
ZONE_CENTER = "center"

ALL_ZONES = (
    ZONE_TOP_LEFT,
    ZONE_TOP_RIGHT,
    ZONE_BOTTOM_LEFT,
    ZONE_BOTTOM_RIGHT,
    ZONE_TOP_WALL,
    ZONE_BOTTOM_WALL,
    ZONE_LEFT_WALL,
    ZONE_RIGHT_WALL,
    ZONE_CENTER,
)

CORNER_ZONES = frozenset({
    ZONE_TOP_LEFT,
    ZONE_TOP_RIGHT,
    ZONE_BOTTOM_LEFT,
    ZONE_BOTTOM_RIGHT,
})

WALL_ZONES = frozenset({
    ZONE_TOP_WALL,
    ZONE_BOTTOM_WALL,
    ZONE_LEFT_WALL,
    ZONE_RIGHT_WALL,
})


def classify_field_zone(
    x: float,
    y: float,
    width: float,
    height: float,
    corner_ratio: float = 0.25,
) -> str:
    """Classify a 2D position into one of 9 spatial zones.

    Corner zones span the outer ``corner_ratio`` (default 25%) of both X and Y dimensions.
    Wall zones span the outer ``corner_ratio`` along one axis while remaining in the interior of the other.
    The center zone spans the remaining interior area.
    """
    if width <= 0 or height <= 0:
        return ZONE_CENTER

    left_bound = width * corner_ratio
    right_bound = width * (1.0 - corner_ratio)
    top_bound = height * corner_ratio
    bottom_bound = height * (1.0 - corner_ratio)

    is_left = x < left_bound
    is_right = x >= right_bound
    is_top = y < top_bound
    is_bottom = y >= bottom_bound

    if is_left and is_top:
        return ZONE_TOP_LEFT
    if is_right and is_top:
        return ZONE_TOP_RIGHT
    if is_left and is_bottom:
        return ZONE_BOTTOM_LEFT
    if is_right and is_bottom:
        return ZONE_BOTTOM_RIGHT

    if is_top:
        return ZONE_TOP_WALL
    if is_bottom:
        return ZONE_BOTTOM_WALL
    if is_left:
        return ZONE_LEFT_WALL
    if is_right:
        return ZONE_RIGHT_WALL

    return ZONE_CENTER


def classify_flock_zone(
    positions: Sequence[Point | tuple[float, float]],
    width: float,
    height: float,
    corner_ratio: float = 0.25,
) -> str:
    """Classify the centroid position of a flock of sheep into a spatial zone."""
    if not positions:
        return ZONE_CENTER
    
    total_x = 0.0
    total_y = 0.0
    for pos in positions:
        if isinstance(pos, Point):
            total_x += pos.x
            total_y += pos.y
        else:
            total_x += float(pos[0])
            total_y += float(pos[1])
            
    avg_x = total_x / len(positions)
    avg_y = total_y / len(positions)
    return classify_field_zone(avg_x, avg_y, width, height, corner_ratio)


def classify_pen_zone(pen: Pen | None, width: float, height: float) -> str:
    """Determine which field zone the pen resides in."""
    if pen is None:
        return "unknown"
    center = pen.center
    return classify_field_zone(center.x, center.y, width, height, corner_ratio=0.33)


@dataclass(slots=True)
class SpatialEpisodeTracker:
    """Tracks sheep spatial residency, corner entrapment, and dislodgement dynamics."""

    field_width: float
    field_height: float
    corner_ratio: float = 0.25
    initial_sheep_zone: str = ZONE_CENTER
    initial_dog_zone: str = ZONE_CENTER
    pen_zone: str = "unknown"
    spawn_mode: str = "default"

    _total_steps: int = 0
    _corner_steps: int = 0
    _wall_steps: int = 0
    _zone_counts: dict[str, int] = field(default_factory=lambda: {z: 0 for z in ALL_ZONES})
    _entered_corner: bool = False
    _extracted_from_corner: bool = False
    _last_flock_zone: str = ZONE_CENTER
    _was_in_corner_prev: bool = False

    def initialize(
        self,
        sheep_positions: Sequence[Point | tuple[float, float]],
        dog_positions: Sequence[Point | tuple[float, float]],
        pen: Pen | None,
        spawn_mode: str = "default",
    ) -> None:
        """Initialize tracker state at the start of an episode."""
        self.spawn_mode = spawn_mode
        self.pen_zone = classify_pen_zone(pen, self.field_width, self.field_height)
        self.initial_sheep_zone = classify_flock_zone(sheep_positions, self.field_width, self.field_height, self.corner_ratio)
        self.initial_dog_zone = classify_flock_zone(dog_positions, self.field_width, self.field_height, self.corner_ratio) if dog_positions else ZONE_CENTER
        self._last_flock_zone = self.initial_sheep_zone
        self._was_in_corner_prev = self.initial_sheep_zone in CORNER_ZONES
        if self._was_in_corner_prev:
            self._entered_corner = True
        self._zone_counts = {z: 0 for z in ALL_ZONES}
        self._total_steps = 0
        self._corner_steps = 0
        self._wall_steps = 0
        self._extracted_from_corner = False

    def record_step(self, sheep_positions: Sequence[Point | tuple[float, float]]) -> None:
        """Record spatial metrics for a single simulation step."""
        if not sheep_positions:
            return

        self._total_steps += 1
        current_zone = classify_flock_zone(sheep_positions, self.field_width, self.field_height, self.corner_ratio)
        self._zone_counts[current_zone] = self._zone_counts.get(current_zone, 0) + 1

        is_corner = current_zone in CORNER_ZONES
        is_wall = current_zone in WALL_ZONES

        if is_corner:
            self._corner_steps += 1
            self._entered_corner = True
        elif is_wall:
            self._wall_steps += 1

        # Check for corner extraction: was previously in a corner, now in open field / center or wall
        if self._was_in_corner_prev and not is_corner:
            self._extracted_from_corner = True

        self._was_in_corner_prev = is_corner
        self._last_flock_zone = current_zone

    def get_summary(self, success: bool, timeout: bool, stopped: bool) -> dict[str, Any]:
        """Produce a complete spatial metrics summary for the episode."""
        total = max(1, self._total_steps)
        corner_pct = round(self._corner_steps / total, 4)
        wall_pct = round(self._wall_steps / total, 4)
        is_stuck_at_end = (not success) and (self._last_flock_zone in CORNER_ZONES or (timeout and self._last_flock_zone in WALL_ZONES))

        return {
            "pen_zone": self.pen_zone,
            "spawn_mode": self.spawn_mode,
            "initial_sheep_zone": self.initial_sheep_zone,
            "initial_dog_zone": self.initial_dog_zone,
            "final_sheep_zone": self._last_flock_zone,
            "corner_steps_total": self._corner_steps,
            "corner_time_pct": corner_pct,
            "wall_steps_total": self._wall_steps,
            "wall_time_pct": wall_pct,
            "corner_stuck_at_end": bool(is_stuck_at_end),
            "corner_entered": bool(self._entered_corner),
            "corner_extracted": bool(self._extracted_from_corner),
            "zone_step_counts": dict(self._zone_counts),
        }


def diagnose_stage_bottlenecks(
    stage: int,
    total_episodes: int,
    zone_stats: dict[str, dict[str, Any]],
    pen_stats: dict[str, dict[str, Any]],
    setup_stats: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Synthesize aggregated historical statistics into actionable bottleneck insights."""
    insights: list[dict[str, Any]] = []

    if total_episodes < 5:
        insights.append({
            "severity": "info",
            "type": "sample_size",
            "title": "Insufficient Sample Size",
            "message": f"Stage {stage} has only {total_episodes} recorded episodes. Run more episodes to generate statistically confident spatial diagnostics.",
        })
        return insights

    # 1. Compare Corner win rates vs Open Field / Center
    center_stat = zone_stats.get(ZONE_CENTER, {})
    center_total = center_stat.get("total", 0)
    center_wins = center_stat.get("wins", 0)
    center_rate = (center_wins / center_total) if center_total > 0 else 0.0

    corner_totals = 0
    corner_wins = 0
    corner_trapped_fails = 0
    corner_rates: dict[str, float] = {}

    for cz in CORNER_ZONES:
        stat = zone_stats.get(cz, {})
        c_tot = stat.get("total", 0)
        c_win = stat.get("wins", 0)
        corner_totals += c_tot
        corner_wins += c_win
        corner_trapped_fails += stat.get("trapped_at_end", 0)
        if c_tot >= 3:
            corner_rates[cz] = c_win / c_tot

    avg_corner_rate = (corner_wins / corner_totals) if corner_totals > 0 else 0.0

    if corner_totals >= 5:
        disparity = center_rate - avg_corner_rate
        if (disparity >= 0.20 and avg_corner_rate < 0.65) or corner_trapped_fails >= 5:
            insights.append({
                "severity": "critical",
                "type": "corner_entrapment",
                "title": "General Corner Entrapment Bottleneck",
                "message": (
                    f"Episodes starting with sheep in corners succeed only {avg_corner_rate:.1%} of the time, "
                    f"compared to {center_rate:.1%} in the center field. "
                    f"{corner_trapped_fails} episodes timed out or stalled with sheep pinned in a corner."
                ),
                "metric": f"{avg_corner_rate:.1%} vs {center_rate:.1%}",
            })

    # 2. Check for Specific Corner Biases
    if corner_rates:
        worst_corner = min(corner_rates.items(), key=lambda kv: kv[1])
        best_corner = max(corner_rates.items(), key=lambda kv: kv[1])
        if (best_corner[1] - worst_corner[1]) >= 0.30:
            worst_name = worst_corner[0].replace("_", " ").title()
            best_name = best_corner[0].replace("_", " ").title()
            insights.append({
                "severity": "warning",
                "type": "corner_bias",
                "title": f"Severe Asymmetry in {worst_name} Corner",
                "message": (
                    f"Sheep in the {worst_name} corner have a win rate of only {worst_corner[1]:.1%}, "
                    f"whereas the {best_name} corner achieves {best_corner[1]:.1%}. "
                    f"The policy has a directional blindspot when dislodging sheep from {worst_name}."
                ),
                "metric": f"{worst_name}: {worst_corner[1]:.1%}",
            })

    # 3. Check Left vs Right and Top vs Bottom Axis Disparity
    left_totals = sum(zone_stats.get(z, {}).get("total", 0) for z in (ZONE_TOP_LEFT, ZONE_BOTTOM_LEFT, ZONE_LEFT_WALL))
    left_wins = sum(zone_stats.get(z, {}).get("wins", 0) for z in (ZONE_TOP_LEFT, ZONE_BOTTOM_LEFT, ZONE_LEFT_WALL))
    left_rate = (left_wins / left_totals) if left_totals >= 5 else None

    right_totals = sum(zone_stats.get(z, {}).get("total", 0) for z in (ZONE_TOP_RIGHT, ZONE_BOTTOM_RIGHT, ZONE_RIGHT_WALL))
    right_wins = sum(zone_stats.get(z, {}).get("wins", 0) for z in (ZONE_TOP_RIGHT, ZONE_BOTTOM_RIGHT, ZONE_RIGHT_WALL))
    right_rate = (right_wins / right_totals) if right_totals >= 5 else None

    if left_rate is not None and right_rate is not None and abs(left_rate - right_rate) >= 0.30:
        weaker_side = "Left" if left_rate < right_rate else "Right"
        stronger_side = "Right" if weaker_side == "Left" else "Left"
        w_rate = left_rate if weaker_side == "Left" else right_rate
        s_rate = right_rate if weaker_side == "Left" else left_rate
        insights.append({
            "severity": "warning",
            "type": "axis_bias",
            "title": f"{weaker_side}-Side Field Bias",
            "message": f"Herding from the {weaker_side} side of the field ({w_rate:.1%} success) is significantly harder than the {stronger_side} side ({s_rate:.1%} success).",
            "metric": f"{weaker_side}: {w_rate:.1%} vs {stronger_side}: {s_rate:.1%}",
        })

    # 4. Check Pen Placement Dependencies
    if len(pen_stats) > 1:
        valid_pens = [(k, v["wins"] / v["total"]) for k, v in pen_stats.items() if v.get("total", 0) >= 4]
        if valid_pens:
            worst_pen = min(valid_pens, key=lambda kv: kv[1])
            best_pen = max(valid_pens, key=lambda kv: kv[1])
            if (best_pen[1] - worst_pen[1]) >= 0.30:
                pen_name = worst_pen[0].replace("_", " ").title()
                insights.append({
                    "severity": "warning",
                    "type": "pen_placement",
                    "title": f"Pen Placement Sensitivity ({pen_name})",
                    "message": f"When the pen is located in {pen_name}, the success rate drops to {worst_pen[1]:.1%} (compared to {best_pen[1]:.1%} for other placements).",
                    "metric": f"{pen_name}: {worst_pen[1]:.1%}",
                })

    # 5. Check Setup / Spawn Mode Difficulties
    if len(setup_stats) > 1:
        valid_setups = [(k, v["wins"] / v["total"]) for k, v in setup_stats.items() if v.get("total", 0) >= 4]
        if valid_setups:
            worst_setup = min(valid_setups, key=lambda kv: kv[1])
            if worst_setup[1] < 0.30:
                setup_name = worst_setup[0].replace("_", " ").title()
                insights.append({
                    "severity": "critical",
                    "type": "setup_failure",
                    "title": f"High Failure Rate in Setup '{setup_name}'",
                    "message": f"Setup '{setup_name}' has a {worst_setup[1]:.1%} success rate and is the primary drag on stage promotion readiness.",
                    "metric": f"Setup {setup_name}: {worst_setup[1]:.1%}",
                })

    if not insights:
        insights.append({
            "severity": "success",
            "type": "balanced",
            "title": "Even Learning Distribution",
            "message": f"No dominant spatial or configuration bottlenecks detected for Stage {stage}. Failure modes are distributed evenly across corners and setups.",
        })

    return insights
