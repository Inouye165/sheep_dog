"""Deterministic semantic failure signature classification with uncertainty support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CORNER_ZONES = frozenset({"top_left", "top_right", "bottom_left", "bottom_right"})
WALL_ZONES = frozenset({"top_wall", "bottom_wall", "left_wall", "right_wall"})


def extract_failure_candidate_causes(record: Any) -> list[str]:
    """Identify all plausible candidate causes for an episode failure."""
    def get_val(key: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(key, default)
        return getattr(record, key, default)

    success = bool(get_val("success", False))
    if success:
        return []

    candidates: list[str] = []

    # 1. Corner Entrapment
    corner_stuck = bool(get_val("corner_stuck_at_end", False))
    final_zone = str(get_val("final_sheep_zone") or "")
    if corner_stuck or (final_zone in CORNER_ZONES and bool(get_val("timeout", False))):
        zone_name = final_zone if final_zone in CORNER_ZONES else "corner"
        candidates.append(f"corner_entrapment_{zone_name}")

    # 2. Gate Corridor / Pen Mouth Obstruction
    gate_fail_steps = int(get_val("gate_corridor_failure_steps", 0) or 0)
    if gate_fail_steps >= 20:
        candidates.append("pen_mouth_obstruction")

    # 3. Wall Stall
    wall_pct = float(get_val("wall_time_pct", 0.0) or 0.0)
    if wall_pct >= 0.40 and final_zone in WALL_ZONES:
        candidates.append(f"wall_stall_{final_zone}")

    # 4. Action Oscillation
    if bool(get_val("oscillation_detected", False)):
        candidates.append("action_oscillation")

    # 5. Flock Splitting vs Stray Abandonment
    flock_spread = float(get_val("final_flock_spread", 0.0) or 0.0)
    role_switches = int(get_val("role_switches", 0) or 0)
    if flock_spread >= 12.0 or role_switches >= 25:
        candidates.append("flock_split_disorganization")

    farthest_flock_dist = float(get_val("final_farthest_distance_to_flock_center", 0.0) or 0.0)
    if farthest_flock_dist >= 10.0:
        candidates.append("stray_recovery_failure")

    # 6. Stopped reason
    if bool(get_val("stopped", False)):
        stop_reason = str(get_val("stop_reason") or "unspecified")
        candidates.append(f"stopped_{stop_reason}")

    return candidates


def classify_failure_signature(record: Any) -> str:
    """Classify a single failed evaluation episode into a deterministic signature.

    Permits explicit uncertainty:
    - 'none': Successful episode.
    - 'insufficient_telemetry': Episode record lacks basic telemetry metrics.
    - 'multiple_candidate_causes': More than one distinct behavioral trigger fired.
    - 'unknown': Episode failed without matching any defined behavioral failure mode.
    - Specific canonical signature if exactly one distinct mode fired.
    """
    def get_val(key: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(key, default)
        return getattr(record, key, default)

    success = bool(get_val("success", False))
    if success:
        return "none"

    # Check for empty / insufficient telemetry
    if isinstance(record, Mapping) and len(record) <= 2:
        return "insufficient_telemetry"

    # Evaluate all candidate behavioral causes
    candidates = extract_failure_candidate_causes(record)

    if len(candidates) > 1:
        # Check if they are related (e.g. corner entrapment + near pen timeout)
        return "multiple_candidate_causes"

    if len(candidates) == 1:
        return candidates[0]

    # If no specific behavioral trigger fired, check for generic timeout proximity
    if bool(get_val("timeout", False)):
        final_dist = float(get_val("final_sheep_distance_to_pen", 999.0) or 999.0)
        min_dist = float(get_val("min_sheep_distance_to_pen", 999.0) or 999.0)
        penned = int(get_val("sheep_penned", 0) or 0)
        if final_dist <= 8.0 or min_dist <= 4.0 or (penned > 0 and final_dist <= 15.0):
            return "near_pen_timeout"
        if final_dist < 900.0 or min_dist < 900.0:
            return "open_field_timeout"

    return "unknown"
