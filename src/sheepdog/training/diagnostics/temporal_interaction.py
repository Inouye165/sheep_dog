"""Offline temporal interaction analysis and opt-in step instrumentation for recorded episodes.

This module explains repeated, unproductive dog-sheep interactions over time across
any number of unpenned sheep (1..6) and distinguishes productive retreat/repositioning
from stationary holds, oscillation loops, and counterproductive pressure, with explicit
epistemic tiers separating recorded observations, derived measurements, uncertain
reconstructions, hypothesized mechanisms, and causes supported by controlled experiments.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from sheepdog.config import EnvironmentConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.environment import ACTION_ORDER
from sheepdog.pen_navigation import PenNavigation, pen_navigation
from sheepdog.team_strategy import RoleAssignment, StrategySnapshot, TeamStrategy

OPPOSITE_BASE_ACTIONS: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "up_left": "down_right",
    "down_right": "up_left",
    "up_right": "down_left",
    "down_left": "up_right",
}

REQUIRED_RECONSTRUCTION_CONFIG_FIELDS: tuple[str, ...] = (
    "width",
    "height",
    "dogs",
    "sheep",
    "gate_approach_distance",
    "blocker_min_remaining_dogs",
    "role_minimum_hold_steps",
    "role_stickiness_distance",
    "flank_role_stickiness_bonus",
    "blocker_role_stickiness_bonus",
    "role_stickiness_bonus",
)

INTERVAL_READABLE_LABELS: dict[str, str] = {
    "oscillation_loop": "Repeated movement",
    "stationary_hold": "Stationary hold",
    "productive_repositioning": "Productive repositioning",
    "counterproductive_pressure": "Counterproductive pressure candidate",
    "unresolved_ambiguity": "Unresolved pattern",
}

ROLE_AWARE_OBSERVATION_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("own_x", "Own X (normalized)", "Positions & Targets"),
    ("own_y", "Own Y (normalized)", "Positions & Targets"),
    ("pen_x", "Pen Center X (normalized)", "Positions & Targets"),
    ("pen_y", "Pen Center Y (normalized)", "Positions & Targets"),
    ("flock_center_x", "Flock Center X (normalized)", "Positions & Targets"),
    ("flock_center_y", "Flock Center Y (normalized)", "Positions & Targets"),
    ("target_x", "Role Target X (normalized)", "Positions & Targets"),
    ("target_y", "Role Target Y (normalized)", "Positions & Targets"),
    ("distance_to_pen", "Dog Distance to Pen", "Distances & Spread"),
    ("distance_to_flock", "Dog Distance to Flock", "Distances & Spread"),
    ("distance_to_target", "Dog Distance to Role Target", "Distances & Spread"),
    ("flock_spread", "Flock Spread", "Distances & Spread"),
    ("average_distance_to_pen", "Avg Sheep Distance to Pen", "Distances & Spread"),
    ("wall_left", "Left Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_right", "Right Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_top", "Top Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_bottom", "Bottom Wall Margin", "Boundaries & Loop Indicators"),
    ("blocked_steps", "Blocked Steps (scaled)", "Boundaries & Loop Indicators"),
    ("no_progress_steps", "No-Progress Steps (scaled)", "Boundaries & Loop Indicators"),
    ("revisits_recent_position", "Revisits Recent Position", "Boundaries & Loop Indicators"),
    ("two_position_loop", "Two-Position Loop Flag", "Boundaries & Loop Indicators"),
    ("stray_present", "Stray Sheep Present", "Boundaries & Loop Indicators"),
    ("role_rear_pressure", "Role: Rear Pressure", "Assigned Scripted Role"),
    ("role_left_flanker", "Role: Left Flanker", "Assigned Scripted Role"),
    ("role_right_flanker", "Role: Right Flanker", "Assigned Scripted Role"),
    ("role_collector", "Role: Collector", "Assigned Scripted Role"),
    ("role_blocker", "Role: Blocker", "Assigned Scripted Role"),
    ("focus_sheep_dx", "Focus Sheep dX", "Focus & Stray Sheep"),
    ("focus_sheep_dy", "Focus Sheep dY", "Focus & Stray Sheep"),
    ("focus_sheep_distance", "Focus Sheep Distance", "Focus & Stray Sheep"),
    ("stray_sheep_dx", "Stray Sheep dX", "Focus & Stray Sheep"),
    ("stray_sheep_dy", "Stray Sheep dY", "Focus & Stray Sheep"),
    ("sheep_0_dx", "Sheep Slot 0 dX", "Sorted Sheep Slots"),
    ("sheep_0_dy", "Sheep Slot 0 dY", "Sorted Sheep Slots"),
    ("sheep_0_penned", "Sheep Slot 0 Penned", "Sorted Sheep Slots"),
    ("sheep_1_dx", "Sheep Slot 1 dX", "Sorted Sheep Slots"),
    ("sheep_1_dy", "Sheep Slot 1 dY", "Sorted Sheep Slots"),
    ("sheep_1_penned", "Sheep Slot 1 Penned", "Sorted Sheep Slots"),
    ("sheep_2_dx", "Sheep Slot 2 dX", "Sorted Sheep Slots"),
    ("sheep_2_dy", "Sheep Slot 2 dY", "Sorted Sheep Slots"),
    ("sheep_2_penned", "Sheep Slot 2 Penned", "Sorted Sheep Slots"),
    ("sheep_3_dx", "Sheep Slot 3 dX", "Sorted Sheep Slots"),
    ("sheep_3_dy", "Sheep Slot 3 dY", "Sorted Sheep Slots"),
    ("sheep_3_penned", "Sheep Slot 3 Penned", "Sorted Sheep Slots"),
    ("sheep_4_dx", "Sheep Slot 4 dX", "Sorted Sheep Slots"),
    ("sheep_4_dy", "Sheep Slot 4 dY", "Sorted Sheep Slots"),
    ("sheep_4_penned", "Sheep Slot 4 Penned", "Sorted Sheep Slots"),
    ("sheep_5_dx", "Sheep Slot 5 dX", "Sorted Sheep Slots"),
    ("sheep_5_dy", "Sheep Slot 5 dY", "Sorted Sheep Slots"),
    ("sheep_5_penned", "Sheep Slot 5 Penned", "Sorted Sheep Slots"),
    ("other_dog_0_dx", "Teammate Dog 0 dX", "Teammates"),
    ("other_dog_0_dy", "Teammate Dog 0 dY", "Teammates"),
    ("other_dog_1_dx", "Teammate Dog 1 dX", "Teammates"),
    ("other_dog_1_dy", "Teammate Dog 1 dY", "Teammates"),
)

EMERGENT_OBSERVATION_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("own_x", "Own X (normalized)", "Positions & Identity"),
    ("own_y", "Own Y (normalized)", "Positions & Identity"),
    ("dog_id_slot_0", "Dog Identity Slot 0", "Positions & Identity"),
    ("dog_id_slot_1", "Dog Identity Slot 1", "Positions & Identity"),
    ("dog_id_slot_2", "Dog Identity Slot 2", "Positions & Identity"),
    ("dog_id_slot_3", "Dog Identity Slot 3", "Positions & Identity"),
    ("dog_id_slot_4", "Dog Identity Slot 4", "Positions & Identity"),
    ("pen_x", "Pen Center X (normalized)", "Positions & Identity"),
    ("pen_y", "Pen Center Y (normalized)", "Positions & Identity"),
    ("flock_center_x", "Flock Center X (normalized)", "Positions & Identity"),
    ("flock_center_y", "Flock Center Y (normalized)", "Positions & Identity"),
    ("distance_to_pen", "Dog Distance to Pen", "Distances & Spread"),
    ("distance_to_flock", "Dog Distance to Flock", "Distances & Spread"),
    ("flock_spread", "Flock Spread", "Distances & Spread"),
    ("average_distance_to_pen", "Avg Sheep Distance to Pen", "Distances & Spread"),
    ("wall_left", "Left Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_right", "Right Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_top", "Top Wall Margin", "Boundaries & Loop Indicators"),
    ("wall_bottom", "Bottom Wall Margin", "Boundaries & Loop Indicators"),
    ("blocked_steps", "Blocked Steps (scaled)", "Boundaries & Loop Indicators"),
    ("no_progress_steps", "No-Progress Steps (scaled)", "Boundaries & Loop Indicators"),
    ("revisits_recent_position", "Revisits Recent Position", "Boundaries & Loop Indicators"),
    ("two_position_loop", "Two-Position Loop Flag", "Boundaries & Loop Indicators"),
    ("stray_present", "Stray Sheep Present", "Boundaries & Loop Indicators"),
    ("nearest_unpenned_dx", "Nearest Unpenned Sheep dX", "Nearest & Farthest Sheep"),
    ("nearest_unpenned_dy", "Nearest Unpenned Sheep dY", "Nearest & Farthest Sheep"),
    ("nearest_unpenned_distance", "Nearest Unpenned Distance", "Nearest & Farthest Sheep"),
    ("farthest_unpenned_dx", "Farthest Unpenned Sheep dX", "Nearest & Farthest Sheep"),
    ("farthest_unpenned_dy", "Farthest Unpenned Sheep dY", "Nearest & Farthest Sheep"),
    ("farthest_unpenned_distance", "Farthest Unpenned Distance", "Nearest & Farthest Sheep"),
    ("sheep_0_dx", "Sheep Slot 0 dX", "Sorted Sheep Slots"),
    ("sheep_0_dy", "Sheep Slot 0 dY", "Sorted Sheep Slots"),
    ("sheep_0_penned", "Sheep Slot 0 Penned", "Sorted Sheep Slots"),
    ("sheep_1_dx", "Sheep Slot 1 dX", "Sorted Sheep Slots"),
    ("sheep_1_dy", "Sheep Slot 1 dY", "Sorted Sheep Slots"),
    ("sheep_1_penned", "Sheep Slot 1 Penned", "Sorted Sheep Slots"),
    ("sheep_2_dx", "Sheep Slot 2 dX", "Sorted Sheep Slots"),
    ("sheep_2_dy", "Sheep Slot 2 dY", "Sorted Sheep Slots"),
    ("sheep_2_penned", "Sheep Slot 2 Penned", "Sorted Sheep Slots"),
    ("sheep_3_dx", "Sheep Slot 3 dX", "Sorted Sheep Slots"),
    ("sheep_3_dy", "Sheep Slot 3 dY", "Sorted Sheep Slots"),
    ("sheep_3_penned", "Sheep Slot 3 Penned", "Sorted Sheep Slots"),
    ("sheep_4_dx", "Sheep Slot 4 dX", "Sorted Sheep Slots"),
    ("sheep_4_dy", "Sheep Slot 4 dY", "Sorted Sheep Slots"),
    ("sheep_4_penned", "Sheep Slot 4 Penned", "Sorted Sheep Slots"),
    ("sheep_5_dx", "Sheep Slot 5 dX", "Sorted Sheep Slots"),
    ("sheep_5_dy", "Sheep Slot 5 dY", "Sorted Sheep Slots"),
    ("sheep_5_penned", "Sheep Slot 5 Penned", "Sorted Sheep Slots"),
    ("other_dog_0_dx", "Teammate Dog 0 dX", "Teammates"),
    ("other_dog_0_dy", "Teammate Dog 0 dY", "Teammates"),
    ("other_dog_1_dx", "Teammate Dog 1 dX", "Teammates"),
    ("other_dog_1_dy", "Teammate Dog 1 dY", "Teammates"),
)


HIERARCHICAL_OBSERVATION_SCHEMA: tuple[tuple[str, str, str], ...] = ROLE_AWARE_OBSERVATION_SCHEMA + (
    ("shepherd_cmd_gather", "Shepherd Command: Gather (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_drive_to_pen", "Shepherd Command: Drive to Pen (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_hold_left", "Shepherd Command: Hold Left (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_hold_right", "Shepherd Command: Hold Right (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_block_escape", "Shepherd Command: Block Escape (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_apply_pressure", "Shepherd Command: Apply Pressure (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_back_off", "Shepherd Command: Back Off (one-hot)", "Shepherd & Identity"),
    ("shepherd_cmd_stop", "Shepherd Command: Stop (one-hot)", "Shepherd & Identity"),
    ("dog_id_normalized", "Dog ID Normalized", "Shepherd & Identity"),
    ("dog_count_normalized", "Dog Count Normalized", "Shepherd & Identity"),
    ("dog_id_slot_0", "Dog Identity Slot 0 (one-hot)", "Shepherd & Identity"),
    ("dog_id_slot_1", "Dog Identity Slot 1 (one-hot)", "Shepherd & Identity"),
    ("dog_id_slot_2", "Dog Identity Slot 2 (one-hot)", "Shepherd & Identity"),
    ("dog_id_slot_3", "Dog Identity Slot 3 (one-hot)", "Shepherd & Identity"),
    ("dog_id_slot_4", "Dog Identity Slot 4 (one-hot)", "Shepherd & Identity"),
)


def label_dog_observation_fields(
    observation: tuple[float, ...] | list[float] | None,
    *,
    observation_mode: str | None = None,
    telemetry_truncated: bool = False,
    telemetry_omitted: bool = False,
) -> dict[str, Any]:
    """Return labeled observation fields when a verified schema matches, or explicit limitation."""
    if observation is None:
        if telemetry_truncated:
            reason = "Not recorded (telemetry capture budget exhausted before this step)."
            schema_status = "truncated"
        elif telemetry_omitted:
            reason = "Not recorded (intentionally omitted middle window in head/tail telemetry mode)."
            schema_status = "omitted"
        else:
            reason = "Not recorded (raw policy observation vectors were not captured in this replay)."
            schema_status = "not_recorded"
        return {
            "available": False,
            "schema_verified": False,
            "schema_status": schema_status,
            "schema_name": None,
            "vector_length": 0,
            "reason": reason,
            "fields": [],
        }

    values = [float(x) for x in observation]
    n = len(values)
    if n == len(ROLE_AWARE_OBSERVATION_SCHEMA) and observation_mode != "emergent":
        return {
            "available": True,
            "schema_verified": True,
            "schema_status": "verified",
            "schema_name": "RoleAwareObservationBuilder (54 features, guided)",
            "vector_length": n,
            "reason": "Verified 54-feature role-aware observation schema.",
            "fields": [
                {
                    "index": idx,
                    "key": key,
                    "label": label,
                    "group": group,
                    "value": round(val, 4),
                }
                for idx, ((key, label, group), val) in enumerate(
                    zip(ROLE_AWARE_OBSERVATION_SCHEMA, values, strict=True)
                )
            ],
        }
    if n == len(EMERGENT_OBSERVATION_SCHEMA) and observation_mode in {None, "emergent"}:
        return {
            "available": True,
            "schema_verified": True,
            "schema_status": "verified",
            "schema_name": "EmergentObservationBuilder (52 features, role-free)",
            "vector_length": n,
            "reason": "Verified 52-feature emergent observation schema.",
            "fields": [
                {
                    "index": idx,
                    "key": key,
                    "label": label,
                    "group": group,
                    "value": round(val, 4),
                }
                for idx, ((key, label, group), val) in enumerate(
                    zip(EMERGENT_OBSERVATION_SCHEMA, values, strict=True)
                )
            ],
        }
    if n == len(HIERARCHICAL_OBSERVATION_SCHEMA) and observation_mode in {None, "hierarchical"}:
        return {
            "available": True,
            "schema_verified": True,
            "schema_status": "verified",
            "schema_name": "HierarchicalObservationBuilder (69 features)",
            "vector_length": n,
            "reason": "Verified 69-feature hierarchical observation schema.",
            "fields": [
                {
                    "index": idx,
                    "key": key,
                    "label": label,
                    "group": group,
                    "value": round(val, 4),
                }
                for idx, ((key, label, group), val) in enumerate(
                    zip(HIERARCHICAL_OBSERVATION_SCHEMA, values, strict=True)
                )
            ],
        }
    return {
        "available": False,
        "schema_verified": False,
        "schema_status": "unverified_schema",
        "schema_name": None,
        "vector_length": n,
        "reason": (
            f"Observation vector length ({n}) does not match a verified standalone schema "
            "(54 role-aware, 52 emergent, or 69 hierarchical); labels omitted because no verified schema exists."
        ),
        "fields": [],
    }


def _base_action(action: str) -> str:
    if action.startswith("sprint_"):
        return action.removeprefix("sprint_")
    return action


def _is_reverse_action(prev_action: str, curr_action: str) -> bool:
    prev_base = _base_action(prev_action)
    curr_base = _base_action(curr_action)
    return prev_base in OPPOSITE_BASE_ACTIONS and OPPOSITE_BASE_ACTIONS[prev_base] == curr_base


def _wall_margin(pt: Point, width: int, height: int) -> int:
    return min(pt.x, width - 1 - pt.x, pt.y, height - 1 - pt.y)


def _is_corner(pt: Point, width: int, height: int, margin: int = 3) -> bool:
    near_x = pt.x <= margin or pt.x >= width - 1 - margin
    near_y = pt.y <= margin or pt.y >= height - 1 - margin
    return near_x and near_y


def _alignment_behind_point(dog_pt: Point, anchor_pt: Point, target_pt: Point) -> float:
    """Cosine alignment of dog relative to the behind-anchor herding side toward target_pt.

    Returns +1.0 when dog is directly behind anchor_pt pushing toward target_pt,
    and -1.0 when dog is directly between anchor_pt and target_pt (blocking approach).
    """
    vx = float(target_pt.x - anchor_pt.x)
    vy = float(target_pt.y - anchor_pt.y)
    dx = float(dog_pt.x - anchor_pt.x)
    dy = float(dog_pt.y - anchor_pt.y)
    v_norm = math.hypot(vx, vy)
    d_norm = math.hypot(dx, dy)
    if v_norm < 1e-6 or d_norm < 1e-6:
        return 0.0
    return -(dx * vx + dy * vy) / (v_norm * d_norm)


@dataclass(frozen=True, slots=True)
class DogDecisionRecord:
    """Captured decision evidence at the dog's policy decision boundary."""

    dog_index: int
    world_step: int
    observation: tuple[float, ...] | None
    decision_action_mask: dict[str, bool]
    recomputed_physical_mask: dict[str, bool]
    selected_action: str
    executed_action: str
    replacement_reason: str | None
    action_probabilities: dict[str, float] | None
    logits: dict[str, float] | None
    value_estimate: float | None
    decision_mode: str
    policy_version: int | None
    checkpoint_id: str | None
    mask_type: str
    action_probabilities_source: str | None = None
    value_estimate_source: str | None = None
    capture_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "dog_index": self.dog_index,
            "world_step": self.world_step,
            "observation": list(self.observation) if self.observation is not None else None,
            "decision_action_mask": dict(self.decision_action_mask),
            "recomputed_physical_mask": dict(self.recomputed_physical_mask),
            "selected_action": self.selected_action,
            "executed_action": self.executed_action,
            "replacement_reason": self.replacement_reason,
            "action_probabilities": (
                dict(self.action_probabilities)
                if self.action_probabilities is not None
                else None
            ),
            "logits": dict(self.logits) if self.logits is not None else None,
            "value_estimate": (
                round(self.value_estimate, 4) if self.value_estimate is not None else None
            ),
            "decision_mode": self.decision_mode,
            "policy_version": self.policy_version,
            "checkpoint_id": self.checkpoint_id,
            "mask_type": self.mask_type,
            "action_probabilities_source": self.action_probabilities_source,
            "value_estimate_source": self.value_estimate_source,
            "capture_error": self.capture_error,
        }


@dataclass(frozen=True, slots=True)
class RewardCycleRecord:
    """Rigorous accounting for a candidate reward cycle."""

    cycle_id: str
    start_step: int
    end_step: int
    duration_steps: int
    is_complete_cycle: bool
    recurrence_method: str  # "candidate_joint_position_recurrence", "none_incomplete"
    matched_state: str  # "joint_grid_positions"
    unmatched_or_unavailable_state: tuple[str, ...]
    incomplete_reason: str | None
    positive_reward_by_component: dict[str, float]
    negative_reward_by_component: dict[str, float]
    net_reward_by_component: dict[str, float]
    total_reward: float
    discount_factor: float | None
    discount_timebase: str | None
    discounted_return: float | None
    discounting_status: str
    sheep_penned_delta: int
    net_route_progress: float
    has_lasting_state_change: bool
    cycle_classification: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration_steps": self.duration_steps,
            "is_complete_cycle": self.is_complete_cycle,
            "recurrence_method": self.recurrence_method,
            "matched_state": self.matched_state,
            "unmatched_or_unavailable_state": list(self.unmatched_or_unavailable_state),
            "incomplete_reason": self.incomplete_reason,
            "positive_reward_by_component": {
                k: round(v, 4) for k, v in self.positive_reward_by_component.items()
            },
            "negative_reward_by_component": {
                k: round(v, 4) for k, v in self.negative_reward_by_component.items()
            },
            "net_reward_by_component": {
                k: round(v, 4) for k, v in self.net_reward_by_component.items()
            },
            "total_reward": round(self.total_reward, 4),
            "discount_factor": self.discount_factor,
            "discount_timebase": self.discount_timebase,
            "discounted_return": (
                round(self.discounted_return, 4)
                if self.discounted_return is not None
                else None
            ),
            "discounting_status": self.discounting_status,
            "sheep_penned_delta": self.sheep_penned_delta,
            "net_route_progress": round(self.net_route_progress, 3),
            "has_lasting_state_change": self.has_lasting_state_change,
            "cycle_classification": self.cycle_classification,
            "description": self.description,
        }


def build_step_instrumentation_payload(
    environment: Any,
    *,
    decisions: list[dict[str, Any]] | None = None,
    raw_actions: tuple[str, ...] | list[str] | None = None,
    action_probabilities: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Capture a read-only pre-step diagnostic telemetry dictionary without mutating state.

    Safe to call immediately before ``environment.step(...)`` when opt-in step
    instrumentation is enabled. Does not call RNG or alter environment state.
    """
    dogs: list[DogState] = list(getattr(environment, "_dogs", ()))
    sheep: list[SheepState] = list(getattr(environment, "_sheep", ()))
    role_assignments: dict[int, RoleAssignment] = dict(
        getattr(environment, "_role_assignments", {})
    )
    strategy_snapshot: StrategySnapshot | None = getattr(
        environment, "_strategy_snapshot", None
    )

    pre_step_roles: list[dict[str, Any]] = []
    dog_kinematics: list[dict[str, Any]] = []
    legal_action_masks: list[dict[str, bool]] = []

    blocked_by_sheep = {s.position for s in sheep if not s.penned}
    for dog in dogs:
        assignment = role_assignments.get(dog.index)
        pre_step_roles.append(
            {
                "dog_index": dog.index,
                "current_role": dog.current_role.value,
                "assigned_role": (
                    assignment.role.value if assignment is not None else dog.current_role.value
                ),
                "target": (
                    {"x": int(assignment.target.x), "y": int(assignment.target.y)}
                    if assignment is not None
                    else {"x": int(dog.position.x), "y": int(dog.position.y)}
                ),
                "target_sheep_index": (
                    assignment.target_sheep_index if assignment is not None else None
                ),
                "reason": assignment.reason if assignment is not None else "unknown",
                "steps_in_role": int(dog.steps_in_role),
            }
        )
        dog_kinematics.append(
            {
                "dog_index": dog.index,
                "x": int(dog.position.x),
                "y": int(dog.position.y),
                "movement_budget": round(float(dog.movement_budget), 4),
                "blocked_steps": int(dog.blocked_steps),
            }
        )
        # Compute physical movement mask without calling state-mutating preparation
        other_dogs = {other.position for other in dogs if other.index != dog.index}
        blocked = blocked_by_sheep | other_dogs
        mask_dict: dict[str, bool] = {}
        for act in ACTION_ORDER:
            if act == "wait":
                mask_dict[act] = True
            else:
                speed = float(environment._dog_action_speed(act))
                steps, _ = environment._movement_steps(
                    speed, dog.movement_budget, allow_accumulation=True
                )
                dest = environment._advance_position(
                    dog.position, act, steps, blocked_positions=blocked
                )
                mask_dict[act] = dest != dog.position
        legal_action_masks.append(mask_dict)

    sheep_kinematics = [
        {
            "sheep_index": s.index,
            "x": int(s.position.x),
            "y": int(s.position.y),
            "penned": bool(s.penned),
            "movement_budget": round(float(s.movement_budget), 4),
            "blocked_steps": int(s.blocked_steps),
            "panic_steps": int(s.panic_steps),
            "is_packed": bool(s.is_packed),
        }
        for s in sheep
    ]

    strategy_dict: dict[str, Any] | None = None
    if strategy_snapshot is not None:
        fc = strategy_snapshot.flock_center
        strategy_dict = {
            "flock_center": {"x": int(fc.x), "y": int(fc.y)} if fc is not None else None,
            "flock_spread": round(float(strategy_snapshot.flock_spread), 4),
            "average_distance_to_pen": round(float(strategy_snapshot.average_distance_to_pen), 4),
            "stray_sheep_index": strategy_snapshot.stray_sheep_index,
            "near_pen": bool(strategy_snapshot.near_pen),
            "wall_pressure": bool(strategy_snapshot.wall_pressure),
            "collector_active": any(
                a.role == DogRole.COLLECTOR for a in role_assignments.values()
            ),
            "blocker_active": any(
                a.role == DogRole.BLOCKER for a in role_assignments.values()
            ),
            "sticky_pen_dx": getattr(getattr(environment, "_team_strategy", None), "_sticky_pen_dx", None),
            "sticky_pen_dy": getattr(getattr(environment, "_team_strategy", None), "_sticky_pen_dy", None),
            "role_hold_steps": {dog.index: int(dog.steps_in_role) for dog in dogs},
            "role_minimum_hold_steps": getattr(getattr(getattr(environment, "_team_strategy", None), "_config", None), "role_minimum_hold_steps", None),
        }

    decisions_list: list[dict[str, Any]] = []
    if decisions is not None:
        decisions_list = list(decisions)
    else:
        for dog_i, dog in enumerate(dogs):
            act = (
                str(raw_actions[dog_i])
                if raw_actions and dog_i < len(raw_actions)
                else "wait"
            )
            mask_d = (
                legal_action_masks[dog_i] if dog_i < len(legal_action_masks) else {}
            )
            probs = (
                action_probabilities[dog_i]
                if action_probabilities and dog_i < len(action_probabilities)
                else None
            )
            decisions_list.append(
                {
                    "dog_index": dog.index,
                    "world_step": getattr(environment, "step_count", 0),
                    "observation": None,
                    "decision_action_mask": mask_d,
                    "recomputed_physical_mask": mask_d,
                    "selected_action": act,
                    "executed_action": act,
                    "replacement_reason": None,
                    "action_probabilities": probs,
                    "logits": None,
                    "value_estimate": None,
                    "decision_mode": "unavailable",
                    "policy_version": None,
                    "checkpoint_id": None,
                    "mask_type": "recomputed_physical",
                }
            )

    selected_acts = [d["selected_action"] for d in decisions_list]
    executed_acts = [d["executed_action"] for d in decisions_list]

    return {
        "pre_step_roles": pre_step_roles,
        "strategy": strategy_dict,
        "dog_kinematics": dog_kinematics,
        "sheep_kinematics": sheep_kinematics,
        "legal_action_masks": legal_action_masks,
        "decisions": decisions_list,
        "raw_actions": list(raw_actions) if raw_actions is not None else None,
        "selected_actions": selected_acts,
        "executed_actions": executed_acts,
        "action_probabilities": action_probabilities,
    }


@dataclass(frozen=True, slots=True)
class ReplayEvidenceAudit:
    """Audit of what is directly recorded, deterministically reconstructed, or missing."""

    has_initial_step_zero: bool
    has_contiguous_frames: bool
    has_recorded_debug_targets: bool
    has_recorded_step_telemetry: bool
    has_recorded_action_probabilities: bool
    has_full_environment_config: bool
    has_checkpoint_provenance: bool
    guidance_certainty: str
    discount_factor_available: bool
    discount_timebase: str | None
    discount_timebase_status: str
    strategy_version: str | None
    strategy_version_status: str
    missing_required_config: tuple[str, ...]
    recorded_fields: tuple[str, ...]
    reconstructed_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    trainer_type: str | None = None
    checkpoint_gamma: float | None = None
    evaluation_gamma: float | None = None
    gamma_mismatch: bool = False
    has_initial_strategy_state: bool = False
    telemetry_truncated: bool = False
    telemetry_limit_steps: int | None = None
    telemetry_truncation_step: int | None = None
    telemetry_omitted: bool = False
    telemetry_mode: str | None = None
    telemetry_head_steps: int | None = None
    telemetry_tail_steps: int | None = None
    telemetry_coverage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DogIntervalSummary:
    """Per-dog behavioral, kinematic, guidance, and action-mask summary over an interval."""

    dog_index: int
    dominant_role: str
    role_distribution: dict[str, int]
    role_switches: int
    start_position: tuple[int, int]
    end_position: tuple[int, int]
    net_displacement: float
    total_path_length: float
    tortuosity_ratio: float
    unique_positions_count: int
    bounding_box_width: int
    bounding_box_height: int
    bounding_box_span: float
    action_reversal_rate: float
    wait_rate: float
    sprint_rate: float
    top_actions: tuple[tuple[str, int], ...]
    mean_distance_to_nearest_sheep: float
    within_sheep_vision_rate: float
    on_gate_or_waypoint_side_rate: float
    behind_flock_alignment_mean: float
    behind_flock_alignment_delta: float
    guidance_source: str
    guidance_certainty: str
    evidence_limitations: tuple[str, ...]
    start_target: tuple[int, int]
    end_target: tuple[int, int]
    mean_distance_to_target: float
    legal_action_mask_source: str
    mask_type: str
    mean_legal_action_count: float
    action_probabilities_available: bool
    selected_actions_available: bool = False
    action_replacements_count: int = 0
    value_estimates_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "dog_index": self.dog_index,
            "dominant_role": self.dominant_role,
            "role_distribution": dict(self.role_distribution),
            "role_switches": self.role_switches,
            "start_position": list(self.start_position),
            "end_position": list(self.end_position),
            "net_displacement": round(self.net_displacement, 3),
            "total_path_length": round(self.total_path_length, 3),
            "tortuosity_ratio": round(self.tortuosity_ratio, 3),
            "unique_positions_count": self.unique_positions_count,
            "bounding_box_width": self.bounding_box_width,
            "bounding_box_height": self.bounding_box_height,
            "bounding_box_span": round(self.bounding_box_span, 3),
            "action_reversal_rate": round(self.action_reversal_rate, 3),
            "wait_rate": round(self.wait_rate, 3),
            "sprint_rate": round(self.sprint_rate, 3),
            "top_actions": [[act, cnt] for act, cnt in self.top_actions],
            "mean_distance_to_nearest_sheep": round(self.mean_distance_to_nearest_sheep, 3),
            "within_sheep_vision_rate": round(self.within_sheep_vision_rate, 3),
            "on_gate_or_waypoint_side_rate": round(self.on_gate_or_waypoint_side_rate, 3),
            "behind_flock_alignment_mean": round(self.behind_flock_alignment_mean, 3),
            "behind_flock_alignment_delta": round(self.behind_flock_alignment_delta, 3),
            "guidance_source": self.guidance_source,
            "guidance_certainty": self.guidance_certainty,
            "evidence_limitations": list(self.evidence_limitations),
            "start_target": list(self.start_target),
            "end_target": list(self.end_target),
            "mean_distance_to_target": round(self.mean_distance_to_target, 3),
            "legal_action_mask_source": self.legal_action_mask_source,
            "mask_type": self.mask_type,
            "mean_legal_action_count": round(self.mean_legal_action_count, 3),
            "action_probabilities_available": self.action_probabilities_available,
            "selected_actions_available": self.selected_actions_available,
            "action_replacements_count": self.action_replacements_count,
            "value_estimates_available": self.value_estimates_available,
        }


@dataclass(frozen=True, slots=True)
class SheepIntervalSummary:
    """Per-sheep trajectory, route-distance, and obstacle interaction summary over an interval."""

    sheep_index: int
    personality: str
    penned_at_start: bool
    penned_at_end: bool
    start_position: tuple[int, int]
    end_position: tuple[int, int]
    start_waypoint: tuple[int, int]
    end_waypoint: tuple[int, int]
    start_route_distance: float
    end_route_distance: float
    min_route_distance: float
    max_route_distance: float
    net_route_progress: float
    unique_positions_count: int
    bounding_box_span: float
    total_path_length: float
    position_revisit_rate: float
    closed_fence_adjacent_rate: float
    outer_wall_or_corner_rate: float
    packed_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "sheep_index": self.sheep_index,
            "personality": self.personality,
            "penned_at_start": self.penned_at_start,
            "penned_at_end": self.penned_at_end,
            "start_position": list(self.start_position),
            "end_position": list(self.end_position),
            "start_waypoint": list(self.start_waypoint),
            "end_waypoint": list(self.end_waypoint),
            "start_route_distance": round(self.start_route_distance, 3),
            "end_route_distance": round(self.end_route_distance, 3),
            "min_route_distance": round(self.min_route_distance, 3),
            "max_route_distance": round(self.max_route_distance, 3),
            "net_route_progress": round(self.net_route_progress, 3),
            "unique_positions_count": self.unique_positions_count,
            "bounding_box_span": round(self.bounding_box_span, 3),
            "total_path_length": round(self.total_path_length, 3),
            "position_revisit_rate": round(self.position_revisit_rate, 3),
            "closed_fence_adjacent_rate": round(self.closed_fence_adjacent_rate, 3),
            "outer_wall_or_corner_rate": round(self.outer_wall_or_corner_rate, 3),
            "packed_rate": round(self.packed_rate, 3),
        }


@dataclass(frozen=True, slots=True)
class RewardIntervalSummary:
    """Aggregated reward signals, complete cycles, and potential reward-farming indicators."""

    total_reward: float
    mean_step_reward: float
    component_totals: dict[str, float]
    positive_progress_reward_steps: int
    negative_progress_reward_steps: int
    zero_net_progress_with_positive_reward_steps: bool
    cycles: tuple[RewardCycleRecord, ...] = field(default_factory=tuple)
    has_candidate_positive_return_cycle: bool = False
    has_net_negative_cycle_with_positive_steps: bool = False
    has_incomplete_cycle: bool = False
    has_exploitable_positive_cycle: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "total_reward": round(self.total_reward, 4),
            "mean_step_reward": round(self.mean_step_reward, 4),
            "component_totals": {k: round(v, 4) for k, v in self.component_totals.items()},
            "positive_progress_reward_steps": self.positive_progress_reward_steps,
            "negative_progress_reward_steps": self.negative_progress_reward_steps,
            "zero_net_progress_with_positive_reward_steps": (
                self.zero_net_progress_with_positive_reward_steps
            ),
            "cycles": [c.to_dict() for c in self.cycles],
            "has_candidate_positive_return_cycle": self.has_candidate_positive_return_cycle,
            "has_exploitable_positive_cycle": self.has_candidate_positive_return_cycle or self.has_exploitable_positive_cycle,
            "has_net_negative_cycle_with_positive_steps": (
                self.has_net_negative_cycle_with_positive_steps
            ),
            "has_incomplete_cycle": self.has_incomplete_cycle,
        }


@dataclass(frozen=True, slots=True)
class IntervalContextWindowSummary:
    """Summary of the lead-in or aftermath window surrounding a candidate interval."""

    window_type: str  # "lead_in" or "aftermath"
    start_step: int
    end_step: int
    duration_steps: int
    unpenned_sheep_start: int
    unpenned_sheep_end: int
    sheep_penned_delta: int
    avg_route_distance_start: float
    avg_route_distance_end: float
    net_route_progress: float
    flock_spread_start: float
    flock_spread_end: float
    dog_role_switches: int
    behind_flock_alignment_start: float
    behind_flock_alignment_end: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "window_type": self.window_type,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration_steps": self.duration_steps,
            "unpenned_sheep_start": self.unpenned_sheep_start,
            "unpenned_sheep_end": self.unpenned_sheep_end,
            "sheep_penned_delta": self.sheep_penned_delta,
            "avg_route_distance_start": round(self.avg_route_distance_start, 3),
            "avg_route_distance_end": round(self.avg_route_distance_end, 3),
            "net_route_progress": round(self.net_route_progress, 3),
            "flock_spread_start": round(self.flock_spread_start, 3),
            "flock_spread_end": round(self.flock_spread_end, 3),
            "dog_role_switches": self.dog_role_switches,
            "behind_flock_alignment_start": round(self.behind_flock_alignment_start, 3),
            "behind_flock_alignment_end": round(self.behind_flock_alignment_end, 3),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class TemporalInteractionInterval:
    """A contiguous world-step interval of repeated behavior, stall, pressure, or repositioning."""

    interval_id: str
    replay_id: str
    classification: str
    start_step: int
    end_step: int
    duration_steps: int
    unpenned_sheep_count: int
    affected_dog_indices: tuple[int, ...]
    affected_sheep_indices: tuple[int, ...]
    avg_route_distance_start: float
    avg_route_distance_end: float
    net_route_progress: float
    flock_spread_start: float
    flock_spread_end: float
    no_progress_steps_start: int
    no_progress_steps_end: int
    no_progress_resets_without_best_route_gain: int
    collector_blind_spot_steps: int
    corner_target_inversion_steps: int
    dogs_outside_sheep_vision_steps: int
    lead_in: IntervalContextWindowSummary
    aftermath: IntervalContextWindowSummary
    dog_summaries: tuple[DogIntervalSummary, ...]
    sheep_summaries: tuple[SheepIntervalSummary, ...]
    reward_summary: RewardIntervalSummary
    candidate_mechanisms: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    unavailable_signals: tuple[str, ...]
    guidance_certainty: str
    evidence_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "interval_id": self.interval_id,
            "replay_id": self.replay_id,
            "classification": self.classification,
            "readable_label": INTERVAL_READABLE_LABELS.get(self.classification, self.classification),
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration_steps": self.duration_steps,
            "unpenned_sheep_count": self.unpenned_sheep_count,
            "affected_dog_indices": list(self.affected_dog_indices),
            "affected_sheep_indices": list(self.affected_sheep_indices),
            "avg_route_distance_start": round(self.avg_route_distance_start, 3),
            "avg_route_distance_end": round(self.avg_route_distance_end, 3),
            "net_route_progress": round(self.net_route_progress, 3),
            "flock_spread_start": round(self.flock_spread_start, 3),
            "flock_spread_end": round(self.flock_spread_end, 3),
            "no_progress_steps_start": self.no_progress_steps_start,
            "no_progress_steps_end": self.no_progress_steps_end,
            "no_progress_resets_without_best_route_gain": (
                self.no_progress_resets_without_best_route_gain
            ),
            "collector_blind_spot_steps": self.collector_blind_spot_steps,
            "corner_target_inversion_steps": self.corner_target_inversion_steps,
            "dogs_outside_sheep_vision_steps": self.dogs_outside_sheep_vision_steps,
            "lead_in": self.lead_in.to_dict(),
            "aftermath": self.aftermath.to_dict(),
            "dog_summaries": [d.to_dict() for d in self.dog_summaries],
            "sheep_summaries": [s.to_dict() for s in self.sheep_summaries],
            "reward_summary": self.reward_summary.to_dict(),
            "candidate_mechanisms": list(self.candidate_mechanisms),
            "supporting_evidence": list(self.supporting_evidence),
            "alternative_explanations": list(self.alternative_explanations),
            "unavailable_signals": list(self.unavailable_signals),
            "guidance_certainty": self.guidance_certainty,
            "evidence_limitations": list(self.evidence_limitations),
            "plain_language": build_interval_plain_language_explanation(self),
        }


@dataclass(frozen=True, slots=True)
class EpisodeTemporalAnalysis:
    """Complete offline temporal interaction analysis for a single recorded episode."""

    replay_id: str
    replay_path: str | None
    seed: int | None
    policy_name: str
    outcome_status: str  # "success", "timeout", "stopped", "running"
    stop_reason: str
    total_steps: int
    total_sheep: int
    final_penned_count: int
    final_unpenned_count: int
    pen_geometry: dict[str, Any]
    penning_steps: tuple[tuple[int, int], ...]
    evidence_audit: ReplayEvidenceAudit
    intervals: tuple[TemporalInteractionInterval, ...]
    unpenned_count_coverage: tuple[int, ...]
    summary_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "replay_id": self.replay_id,
            "replay_path": self.replay_path,
            "seed": self.seed,
            "policy_name": self.policy_name,
            "outcome_status": self.outcome_status,
            "stop_reason": self.stop_reason,
            "total_steps": self.total_steps,
            "total_sheep": self.total_sheep,
            "final_penned_count": self.final_penned_count,
            "final_unpenned_count": self.final_unpenned_count,
            "pen_geometry": dict(self.pen_geometry),
            "penning_steps": [[step, s_idx] for step, s_idx in self.penning_steps],
            "evidence_audit": self.evidence_audit.to_dict(),
            "intervals": [iv.to_dict() for iv in self.intervals],
            "unpenned_count_coverage": list(self.unpenned_count_coverage),
            "summary_notes": list(self.summary_notes),
        }


@dataclass(frozen=True, slots=True)
class FailureRecoveryComparison:
    """Structured comparison between a failed episode and a successful recovery episode."""

    failure_replay_id: str
    recovery_replay_id: str
    shared_seed: int | None
    failure_unpenned_at_end: int
    recovery_completion_step: int
    failure_dominant_classifications: tuple[str, ...]
    recovery_interval_classifications: tuple[str, ...]
    key_divergences: tuple[str, ...]
    shared_bottlenecks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return asdict(self)


@dataclass
class _ReconstructedStep:
    """Internal per-step aligned representation for temporal analysis."""

    step: int
    actions: tuple[str, ...]
    dogs: list[DogState]
    sheep: list[SheepState]
    penned_count: int
    unpenned_count: int
    average_distance_to_pen: float
    flock_spread: float
    no_progress_steps: int
    reward_dict: dict[str, float]
    role_assignments: dict[int, RoleAssignment]
    strategy_snapshot: StrategySnapshot
    guidance_source: str
    guidance_certainty: str
    legal_action_masks: dict[int, dict[str, bool]]
    legal_action_mask_source: str
    mask_type: str
    action_probabilities_available: bool
    decisions: list[DogDecisionRecord]
    sheep_route_distances: dict[int, float]
    sheep_waypoints: dict[int, Point]
    sheep_fence_adjacent: dict[int, bool]
    sheep_wall_or_corner: dict[int, bool]
    dog_behind_alignment: dict[int, float]
    dog_on_gate_or_waypoint_side: dict[int, bool]
    dog_nearest_sheep_dist: dict[int, float]
    best_route_distance_so_far: float
    no_progress_reset_without_best_gain: bool
    collector_blind_spot: bool
    corner_target_inverted: bool
    any_dog_outside_sheep_vision_oscillating: bool
    has_recorded_step_telemetry: bool = False
    has_recorded_debug_targets: bool = False
    step_telemetry_dict: dict[str, Any] | None = None
    pre_step_dogs: list[DogState] = field(default_factory=list)
    pre_step_sheep: list[SheepState] = field(default_factory=list)
    has_prior_contiguous_step: bool = False


def load_replay_bundle(source: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str, str | None]:
    """Load a replay dictionary from a path (`.json` or `.json.gz`) or in-memory dict."""
    if isinstance(source, dict):
        replay_id = str(source.get("replay_id") or f"seed-{source.get('seed', 'unknown')}")
        return source, replay_id, None

    path = Path(source)
    if not path.exists():
        gz_candidate = Path(f"{path}.gz")
        if gz_candidate.exists():
            path = gz_candidate
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

    if isinstance(data, list):
        wrapped: dict[str, Any] = {
            "seed": None,
            "policy_name": "exported_replay",
            "frames": data,
            "final_snapshot": data[-1]["snapshot"] if data else {},
            "stats": {},
        }
        return wrapped, path.stem, str(path)

    raw_frames = list(data.get("frames") or [])
    if (
        raw_frames
        and int(raw_frames[0].get("step", 0)) != 0
        and isinstance(data.get("initial_snapshot"), dict)
    ):
        raw_frames.insert(
            0,
            {
                "step": 0,
                "actions": [],
                "snapshot": data["initial_snapshot"],
                "reward": {},
            },
        )
        data["frames"] = raw_frames

    name = path.name
    if name.endswith(".json.gz"):
        replay_id = name[: -len(".json.gz")]
    elif name.endswith(".json"):
        replay_id = name[: -len(".json")]
    else:
        replay_id = path.stem
    return data, replay_id, str(path)


def audit_replay_evidence(
    bundle: dict[str, Any],
    *,
    has_initial_step_zero: bool,
    has_contiguous_frames: bool,
    has_recorded_debug_targets: bool,
    has_recorded_step_telemetry: bool,
    has_recorded_action_probabilities: bool,
    has_initial_strategy_state: bool = False,
    telemetry_truncated: bool = False,
    telemetry_limit_steps: int | None = None,
    telemetry_truncation_step: int | None = None,
    telemetry_omitted: bool = False,
    telemetry_mode: str | None = None,
    telemetry_head_steps: int | None = None,
    telemetry_tail_steps: int | None = None,
    telemetry_coverage: dict[str, Any] | None = None,
    discount_factor: float | None = None,
    discount_timebase: str | None = None,
    strategy_version: str | None = None,
    trainer_type: str | None = None,
    checkpoint_gamma: float | None = None,
    evaluation_gamma: float | None = None,
    gamma_mismatch: bool = False,
) -> ReplayEvidenceAudit:
    """Construct an explicit audit of recorded, reconstructed, and unavailable fields."""
    env_cfg = bundle.get("environment") or {}
    missing_config = tuple(
        k for k in REQUIRED_RECONSTRUCTION_CONFIG_FIELDS if k not in env_cfg
    )
    has_full_env = len(missing_config) == 0
    has_provenance = bool(bundle.get("checkpoint_id") or bundle.get("run_id"))

    if strategy_version == TeamStrategy.VERSION:
        strategy_version_status = "compatible_strategy_version"
    elif strategy_version is None:
        strategy_version_status = "missing_strategy_version"
    else:
        strategy_version_status = "incompatible_strategy_version"

    if gamma_mismatch:
        discount_timebase_status = "configuration_checkpoint_gamma_mismatch"
    elif trainer_type in {"hill_climb", "baseline", "heuristic", "random"}:
        discount_timebase_status = "unsupported_trainer_discount_unavailable"
    elif discount_timebase is not None:
        if discount_timebase in {"sequential_shared_dog", "joint_world_step"}:
            discount_timebase_status = "explicit_timebase_recorded"
        else:
            discount_timebase_status = f"unsupported_timebase_{discount_timebase}"
    elif discount_factor is not None:
        discount_timebase_status = "unknown_timebase_historical_alignment_unavailable"
    else:
        discount_timebase_status = "discount_factor_unavailable"

    if has_recorded_step_telemetry:
        if telemetry_truncated or telemetry_omitted:
            guidance_certainty = "partial_runtime_telemetry"
        else:
            guidance_certainty = "recorded_runtime_telemetry"
    elif has_recorded_debug_targets:
        guidance_certainty = "recorded_runtime_debug"
    elif (
        has_initial_step_zero
        and has_contiguous_frames
        and has_full_env
        and strategy_version == TeamStrategy.VERSION
    ):
        guidance_certainty = "estimated_reconstruction_unrecorded_initial_strategy_state"
    elif not has_initial_step_zero:
        guidance_certainty = "uncertain_reconstruction_missing_step0"
    elif not has_full_env:
        guidance_certainty = "uncertain_reconstruction_missing_config"
    elif not has_contiguous_frames:
        guidance_certainty = "uncertain_reconstruction_discontinuous_steps"
    elif strategy_version is None:
        guidance_certainty = "estimated_reconstruction_missing_strategy_version"
    elif strategy_version != TeamStrategy.VERSION:
        guidance_certainty = "estimated_reconstruction_strategy_version_mismatch"
    else:
        guidance_certainty = "estimated_reconstruction"

    recorded: list[str] = [
        "snapshot.dogs[i].(index, x, y, role, last_action)",
        "snapshot.sheep[j].(index, x, y, penned, personality, packed)",
        "frame.actions (validated dog actions per step)",
        "snapshot.(pen, fence_cells, penned_count, average_distance_to_pen, flock_spread, no_progress_steps)",
        "frame.reward (28 RewardBreakdown components per step)",
        "snapshot.(terminated, timeout, stopped, success, status)",
    ]
    reconstructed: list[str] = [
        "per-step discrete displacements (dx, dy) for dogs and sheep",
        "BFS legal entry route distance and immediate navigation waypoint per sheep (via pen_navigation)",
        "closed-fence adjacency, outer-wall margin, and behind-sheep/waypoint cosine alignment",
    ]
    unavailable: list[str] = []

    if has_initial_step_zero:
        recorded.append("step 0 initial state snapshot")
    else:
        unavailable.append(
            "step 0 initial state snapshot (evaluation replay starts at step 1; step 1 displacement is unavailable)"
        )

    if has_recorded_step_telemetry:
        recorded.append("frame.step_telemetry (pre-step roles, decisions, movement_budget, actual_decision_masks)")
    elif has_recorded_debug_targets:
        recorded.append("snapshot.debug.dogs (desired_pressure_target, alignment, focus_mode)")
        reconstructed.append("physical legal-action masks (historical approximate without fractional movement_budget)")
        unavailable.append("sub-step fractional movement_budget, blocked_steps, and sheep panic_steps")
    else:
        reconstructed.append(
            "TeamStrategy role assignments and targets (estimated reconstruction; snapshot.debug was unrecorded)"
        )
        reconstructed.append(
            "physical legal-action masks (historical approximate from pre-step positions without fractional budget)"
        )
        unavailable.append("snapshot.debug per-dog target payload (debug_reward_breakdown was False during capture)")
        unavailable.append("sub-step fractional movement_budget, blocked_steps, and sheep panic_steps")

    if not (has_recorded_step_telemetry or has_recorded_debug_targets):
        unavailable.append(
            "initial strategy internal state (sticky pen approach axis and dog role hold steps at step 0; targets estimated)"
        )
    if telemetry_truncated:
        step_info = f" at step {telemetry_truncation_step}" if telemetry_truncation_step is not None else ""
        limit_info = f" (budget limit: {telemetry_limit_steps} steps)" if telemetry_limit_steps is not None else ""
        unavailable.append(
            f"complete episode runtime telemetry (truncated{step_info} due to per_episode_telemetry_budget_exhausted{limit_info})"
        )
    elif telemetry_omitted:
        omitted_desc = "head/tail windowing"
        if telemetry_coverage and telemetry_coverage.get("omitted_range"):
            rng = telemetry_coverage["omitted_range"]
            omitted_desc = f"steps {rng[0]}..{rng[1]}"
        elif telemetry_head_steps is not None and telemetry_tail_steps is not None:
            omitted_desc = f"head={telemetry_head_steps}, tail={telemetry_tail_steps}"
        unavailable.append(
            f"complete episode runtime telemetry (middle window omitted: {omitted_desc})"
        )

    if has_recorded_action_probabilities:
        recorded.append("policy action probabilities / logits per dog")
    else:
        unavailable.append("policy action probabilities, logits, and value estimates")
        unavailable.append("raw policy observation vectors per dog")

    if has_full_env:
        recorded.append("required environment configuration fields present")
    else:
        unavailable.append(f"required configuration fields missing: {', '.join(missing_config)}")

    if strategy_version == TeamStrategy.VERSION:
        recorded.append(f"strategy_version='{strategy_version}' matching current TeamStrategy.VERSION")
    else:
        unavailable.append(
            f"compatible strategy_version (recorded: '{strategy_version}', current: '{TeamStrategy.VERSION}')"
        )

    if has_recorded_action_probabilities:
        recorded.append("policy action probabilities / logits per dog")
    else:
        unavailable.append("policy action probabilities, logits, and value estimates")
        unavailable.append("raw policy observation vectors per dog")

    if has_full_env:
        recorded.append("required environment configuration fields present")
    else:
        unavailable.append(f"required configuration fields missing: {', '.join(missing_config)}")

    if strategy_version == TeamStrategy.VERSION:
        recorded.append(f"strategy_version='{strategy_version}' matching current TeamStrategy.VERSION")
    else:
        unavailable.append(
            f"compatible strategy_version (recorded: '{strategy_version}', current: '{TeamStrategy.VERSION}')"
        )

    if has_provenance:
        recorded.append("checkpoint_id and run_id provenance in replay header")
    else:
        unavailable.append("checkpoint_id, run_id, and policy_version inside replay file header")

    if discount_factor is not None:
        recorded.append(f"discount_factor gamma={discount_factor} in configuration")
    else:
        unavailable.append("discount factor gamma in configuration (discounted return unavailable)")

    if discount_timebase is not None:
        recorded.append(f"discount_timebase='{discount_timebase}'")
    else:
        unavailable.append("discount_timebase metadata (transition alignment unavailable)")

    return ReplayEvidenceAudit(
        has_initial_step_zero=has_initial_step_zero,
        has_contiguous_frames=has_contiguous_frames,
        has_recorded_debug_targets=has_recorded_debug_targets,
        has_recorded_step_telemetry=has_recorded_step_telemetry,
        has_recorded_action_probabilities=has_recorded_action_probabilities,
        has_full_environment_config=has_full_env,
        has_checkpoint_provenance=has_provenance,
        guidance_certainty=guidance_certainty,
        discount_factor_available=discount_factor is not None,
        discount_timebase=discount_timebase,
        discount_timebase_status=discount_timebase_status,
        strategy_version=strategy_version,
        strategy_version_status=strategy_version_status,
        missing_required_config=missing_config,
        recorded_fields=tuple(recorded),
        reconstructed_fields=tuple(reconstructed),
        unavailable_fields=tuple(unavailable),
        trainer_type=trainer_type,
        checkpoint_gamma=checkpoint_gamma,
        evaluation_gamma=evaluation_gamma,
        gamma_mismatch=gamma_mismatch,
        telemetry_truncated=telemetry_truncated,
        telemetry_limit_steps=telemetry_limit_steps,
        telemetry_truncation_step=telemetry_truncation_step,
        telemetry_omitted=telemetry_omitted,
        telemetry_mode=telemetry_mode,
        telemetry_head_steps=telemetry_head_steps,
        telemetry_tail_steps=telemetry_tail_steps,
        telemetry_coverage=telemetry_coverage,
    )


def _compute_discounted_return(
    cycle_steps: list[_ReconstructedStep],
    discount_factor: float | None,
    discount_timebase: str | None,
    dog_count: int,
    cycle_is_contiguous: bool = True,
    *,
    trainer_type: str | None = None,
    gamma_mismatch: bool = False,
) -> tuple[float | None, str]:
    """Compute discounted return with explicit transition-timebase awareness."""
    if gamma_mismatch:
        return None, "configuration_checkpoint_gamma_mismatch"
    if trainer_type in {"hill_climb", "baseline", "heuristic", "random"}:
        return None, "unsupported_trainer_discount_unavailable"
    if discount_factor is None:
        return None, "discount_factor_unavailable"
    if not cycle_is_contiguous:
        return None, "broken_transition_alignment_frame_gaps"
    if discount_timebase == "joint_world_step":
        ret = 0.0
        for k_idx, st in enumerate(cycle_steps):
            step_tot = st.reward_dict.get("total", sum(st.reward_dict.values()))
            ret += (discount_factor**k_idx) * step_tot
        return ret, "aligned_and_computed_joint_world_step"
    elif discount_timebase == "sequential_shared_dog":
        ret = 0.0
        k_dogs = max(1, dog_count)
        for k_idx, st in enumerate(cycle_steps):
            step_tot = st.reward_dict.get("total", sum(st.reward_dict.values()))
            delta_tau = k_idx * k_dogs + (k_dogs - 1)
            ret += (discount_factor**delta_tau) * step_tot
        return ret, "aligned_and_computed_sequential_shared_dog"
    else:
        return None, "discount_timebase_unknown_historical_alignment_unavailable"


def _reconstruct_episode_steps(
    bundle: dict[str, Any],
) -> tuple[list[_ReconstructedStep], Pen, int, int, ReplayEvidenceAudit, float | None]:
    """Reconstruct aligned per-step kinematics, geometry, guidance, and masks from a replay bundle."""
    raw_frames: list[dict[str, Any]] = list(bundle.get("frames") or [])
    if not raw_frames:
        raise ValueError("Replay bundle contains no frames.")

    frames: list[dict[str, Any]] = []
    first_step = int(raw_frames[0].get("step", 0))
    has_initial_step_zero = first_step == 0
    if not has_initial_step_zero and isinstance(bundle.get("initial_snapshot"), dict):
        frames.append(
            {
                "step": 0,
                "actions": [],
                "snapshot": bundle["initial_snapshot"],
                "reward": {},
            }
        )
        has_initial_step_zero = True
    frames.extend(raw_frames)

    has_contiguous = True
    for i in range(1, len(frames)):
        if int(frames[i].get("step", 0)) != int(frames[i - 1].get("step", 0)) + 1:
            has_contiguous = False
            break

    last_snap = frames[-1]["snapshot"]
    width = int(last_snap.get("grid_width") or bundle.get("environment", {}).get("width", 35))
    height = int(last_snap.get("grid_height") or bundle.get("environment", {}).get("height", 21))
    pen_dict = last_snap["pen"]
    pen = Pen(
        origin=Point(int(pen_dict["origin"]["x"]), int(pen_dict["origin"]["y"])),
        width=int(pen_dict["width"]),
        height=int(pen_dict["height"]),
        opening=str(pen_dict.get("opening", "bottom")),
    )
    fence_cells = pen.fence_cells()
    nav: PenNavigation = pen_navigation(pen, width, height)

    env_meta = bundle.get("environment") or {}
    valid_env_keys = set(EnvironmentConfig.__dataclass_fields__.keys())
    filtered_env_kwargs = {k: v for k, v in env_meta.items() if k in valid_env_keys}
    filtered_env_kwargs["width"] = width
    filtered_env_kwargs["height"] = height
    env_config = EnvironmentConfig(**filtered_env_kwargs)
    team_strategy = TeamStrategy(width, height, env_config)

    telemetry_truncated = bool(bundle.get("telemetry_truncated", False))
    telemetry_limit_steps = bundle.get("telemetry_limit_steps")
    telemetry_truncation_step = None
    telemetry_omitted = bool(bundle.get("telemetry_omitted", False))
    telemetry_mode = bundle.get("telemetry_mode")
    telemetry_head_steps = bundle.get("telemetry_head_steps")
    telemetry_tail_steps = bundle.get("telemetry_tail_steps")
    telemetry_coverage = bundle.get("telemetry_coverage")

    for frame in frames:
        st_tele = frame.get("step_telemetry")
        if isinstance(st_tele, dict):
            if st_tele.get("telemetry_truncated"):
                telemetry_truncated = True
                telemetry_truncation_step = int(frame.get("step", 0))
                if telemetry_limit_steps is None:
                    telemetry_limit_steps = st_tele.get("telemetry_limit_steps")
            if st_tele.get("telemetry_omitted"):
                telemetry_omitted = True
                if telemetry_mode is None:
                    telemetry_mode = st_tele.get("telemetry_mode", "head_tail")
                if telemetry_head_steps is None:
                    telemetry_head_steps = st_tele.get("head_steps")
                if telemetry_tail_steps is None:
                    telemetry_tail_steps = st_tele.get("tail_steps")

    trainer_type: str | None = (
        bundle.get("trainer_type")
        or bundle.get("trainer")
        or bundle.get("training", {}).get("trainer_type")
        or (
            bundle.get("config", {}).get("training", {}).get("trainer_type")
            if isinstance(bundle.get("config"), dict)
            else None
        )
    )

    checkpoint_gamma: float | None = None
    if isinstance(bundle.get("checkpoint_gamma"), int | float):
        checkpoint_gamma = float(bundle["checkpoint_gamma"])
    else:
        chk_meta = bundle.get("checkpoint_metadata") or bundle.get("checkpoint") or {}
        if isinstance(chk_meta, dict):
            if isinstance(chk_meta.get("gamma"), int | float):
                checkpoint_gamma = float(chk_meta["gamma"])
            elif isinstance(chk_meta.get("training", {}).get("gamma"), int | float):
                checkpoint_gamma = float(chk_meta["training"]["gamma"])

    evaluation_gamma: float | None = None
    if isinstance(bundle.get("evaluation_gamma"), int | float):
        evaluation_gamma = float(bundle["evaluation_gamma"])
    elif isinstance(bundle.get("discount_factor"), int | float):
        evaluation_gamma = float(bundle["discount_factor"])
    elif isinstance(env_meta.get("gamma"), int | float):
        evaluation_gamma = float(env_meta["gamma"])
    elif isinstance(bundle.get("config", {}).get("training", {}).get("gamma"), int | float):
        evaluation_gamma = float(bundle["config"]["training"]["gamma"])
    elif isinstance(bundle.get("training", {}).get("gamma"), int | float):
        evaluation_gamma = float(bundle["training"]["gamma"])
    elif isinstance(bundle.get("gamma"), int | float):
        evaluation_gamma = float(bundle["gamma"])

    gamma_mismatch: bool = bool(
        bundle.get("gamma_mismatch")
        or (
            checkpoint_gamma is not None
            and evaluation_gamma is not None
            and abs(checkpoint_gamma - evaluation_gamma) > 1e-6
        )
    )
    discount_factor: float | None = (
        checkpoint_gamma if checkpoint_gamma is not None else evaluation_gamma
    )

    discount_timebase: str | None = (
        bundle.get("discount_timebase")
        or env_meta.get("discount_timebase")
        or bundle.get("training", {}).get("discount_timebase")
    )
    if trainer_type in {"hill_climb", "baseline", "heuristic", "random"}:
        discount_timebase = None

    strategy_version: str | None = (
        bundle.get("strategy_version")
        or env_meta.get("strategy_version")
        or bundle.get("strategy", {}).get("version")
    )
    dog_count = int(
        env_meta.get("dogs")
        or bundle.get("dog_count")
        or (
            len(bundle.get("frames", [{}])[0].get("snapshot", {}).get("dogs", ()))
            if bundle.get("frames")
            else 3
        )
    )
    missing_config = tuple(
        k for k in REQUIRED_RECONSTRUCTION_CONFIG_FIELDS if k not in env_meta
    )

    has_recorded_debug_targets = False
    has_recorded_step_telemetry = False
    has_recorded_action_probs = False

    reconstructed_steps: list[_ReconstructedStep] = []
    prev_dogs_by_idx: dict[int, DogState] | None = None
    prev_sheep_by_idx: dict[int, SheepState] | None = None
    prev_step_num: int | None = None
    best_route_dist = float("inf")
    prev_no_progress = 0

    for frame in frames:
        step_num = int(frame.get("step", 0))
        has_prior_contiguous_step = (
            prev_step_num is not None and step_num == prev_step_num + 1
        )
        actions = tuple(str(a) for a in (frame.get("actions") or ()))
        snap = frame["snapshot"]
        reward_raw = frame.get("reward") or {}
        reward_dict = {
            str(k): float(v) for k, v in reward_raw.items() if isinstance(v, int | float)
        }
        step_telemetry = frame.get("step_telemetry")
        debug_payload = snap.get("debug") or {}

        curr_dogs = [
            DogState(
                index=int(d["index"]),
                position=Point(int(d["x"]), int(d["y"])),
                current_role=(
                    DogRole(str(d["role"]))
                    if d.get("role") in {r.value for r in DogRole}
                    else DogRole.REAR_PRESSURE
                ),
                last_action=str(d.get("last_action", "wait")),
            )
            for d in snap.get("dogs", ())
        ]
        curr_sheep = [
            SheepState(
                index=int(s["index"]),
                position=Point(int(s["x"]), int(s["y"])),
                penned=bool(s.get("penned", False)),
                personality=str(s.get("personality") or "obedient"),
                is_packed=bool(s.get("packed", False)),
            )
            for s in snap.get("sheep", ())
        ]

        decision_dogs = (
            list(prev_dogs_by_idx.values()) if prev_dogs_by_idx is not None else curr_dogs
        )
        decision_sheep = (
            list(prev_sheep_by_idx.values()) if prev_sheep_by_idx is not None else curr_sheep
        )
        pre_step_dogs_copy = [
            DogState(
                index=d.index,
                position=d.position,
                current_role=d.current_role,
                last_action=d.last_action,
                steps_in_role=d.steps_in_role,
            )
            for d in decision_dogs
        ]
        pre_step_sheep_copy = [
            SheepState(
                index=s.index,
                position=s.position,
                penned=s.penned,
                personality=s.personality,
                is_packed=s.is_packed,
            )
            for s in decision_sheep
        ]

        for d in decision_dogs:
            snap_dog = next((cd for cd in curr_dogs if cd.index == d.index), d)
            if d.current_role == snap_dog.current_role:
                d.steps_in_role += 1
            else:
                d.steps_in_role = 0
                d.current_role = snap_dog.current_role

        reconstructed_assignments, reconstructed_strategy = team_strategy.assign_roles(
            decision_dogs, decision_sheep, pen
        )

        guidance_source = "reconstructed_team_strategy"
        guidance_cert = (
            "estimated_reconstruction_unrecorded_initial_strategy_state"
            if (
                has_initial_step_zero
                and has_contiguous
                and len(missing_config) == 0
                and strategy_version == TeamStrategy.VERSION
            )
            else "uncertain_reconstruction_missing_step0"
            if not has_initial_step_zero
            else "uncertain_reconstruction_missing_config"
            if len(missing_config) > 0
            else "uncertain_reconstruction_discontinuous_steps"
            if not has_contiguous
            else "estimated_reconstruction_missing_strategy_version"
            if strategy_version is None
            else "estimated_reconstruction_strategy_version_mismatch"
            if strategy_version != TeamStrategy.VERSION
            else "estimated_reconstruction"
        )
        role_assignments = dict(reconstructed_assignments)

        if isinstance(step_telemetry, dict) and step_telemetry.get("pre_step_roles"):
            has_recorded_step_telemetry = True
            guidance_source = "recorded_step_telemetry"
            guidance_cert = "recorded_runtime_telemetry"
            pre_step_targets = step_telemetry.get("pre_step_targets")
            for idx_entry, entry in enumerate(step_telemetry["pre_step_roles"]):
                if isinstance(entry, dict):
                    d_idx = int(entry.get("dog_index", idx_entry))
                    role_str = str(
                        entry.get("assigned_role")
                        or entry.get("current_role")
                        or "rear_pressure"
                    )
                    tgt = entry.get("target") or {"x": 0, "y": 0}
                    tgt_sheep_idx = entry.get("target_sheep_index")
                    reason_str = str(entry.get("reason", "recorded"))
                else:
                    d_idx = idx_entry
                    role_str = str(entry or "rear_pressure")
                    tgt = (
                        pre_step_targets[idx_entry]
                        if isinstance(pre_step_targets, list)
                        and idx_entry < len(pre_step_targets)
                        and isinstance(pre_step_targets[idx_entry], dict)
                        else {"x": 0, "y": 0}
                    )
                    tgt_sheep_idx = None
                    reason_str = "recorded"
                role_enum = (
                    DogRole(role_str)
                    if role_str in {r.value for r in DogRole}
                    else DogRole.REAR_PRESSURE
                )
                role_assignments[d_idx] = RoleAssignment(
                    dog_index=d_idx,
                    role=role_enum,
                    target=Point(int(tgt["x"]), int(tgt["y"])),
                    target_sheep_index=tgt_sheep_idx,
                    reason=reason_str,
                )
            if step_telemetry.get("action_probabilities"):
                has_recorded_action_probs = True
        elif isinstance(debug_payload, dict) and debug_payload.get("dogs"):
            has_recorded_debug_targets = True
            guidance_source = "recorded_debug"
            guidance_cert = "recorded_runtime_debug"
            for d_dbg in debug_payload["dogs"]:
                d_idx = int(d_dbg["index"])
                tgt = d_dbg.get("desired_pressure_target")
                if isinstance(tgt, dict) and "x" in tgt and "y" in tgt:
                    base_assign = role_assignments.get(d_idx)
                    role_enum = (
                        base_assign.role
                        if base_assign is not None
                        else next(
                            (cd.current_role for cd in curr_dogs if cd.index == d_idx),
                            DogRole.REAR_PRESSURE,
                        )
                    )
                    role_assignments[d_idx] = RoleAssignment(
                        dog_index=d_idx,
                        role=role_enum,
                        target=Point(int(tgt["x"]), int(tgt["y"])),
                        target_sheep_index=(
                            base_assign.target_sheep_index if base_assign else None
                        ),
                        reason=str(d_dbg.get("focus_mode", "recorded_debug")),
                    )

        # Build DogDecisionRecords
        legal_masks: dict[int, dict[str, bool]] = {}
        mask_source = "historical_approximate_no_carry"
        mask_type = "historical_approximate"
        decisions: list[DogDecisionRecord] = []

        raw_telemetry_decisions = (
            step_telemetry.get("decisions")
            if isinstance(step_telemetry, dict) and "decisions" in step_telemetry
            else None
        )

        if raw_telemetry_decisions is not None:
            has_recorded_step_telemetry = True
            mask_source = "recorded_step_telemetry"
            mask_type = "actual_decision_time"
            for dec_d in raw_telemetry_decisions:
                d_idx = int(dec_d["dog_index"])
                dec_mask = {str(k): bool(v) for k, v in dec_d.get("decision_action_mask", {}).items()}
                phys_mask = {
                    str(k): bool(v) for k, v in dec_d.get("recomputed_physical_mask", {}).items()
                }
                legal_masks[d_idx] = dec_mask
                if dec_d.get("action_probabilities"):
                    has_recorded_action_probs = True
                decisions.append(
                    DogDecisionRecord(
                        dog_index=d_idx,
                        world_step=step_num,
                        observation=(
                            tuple(float(x) for x in dec_d["observation"])
                            if dec_d.get("observation") is not None
                            else None
                        ),
                        decision_action_mask=dec_mask,
                        recomputed_physical_mask=phys_mask,
                        selected_action=str(dec_d.get("selected_action", "wait")),
                        executed_action=str(dec_d.get("executed_action", "wait")),
                        replacement_reason=dec_d.get("replacement_reason"),
                        action_probabilities=dec_d.get("action_probabilities"),
                        logits=dec_d.get("logits"),
                        value_estimate=dec_d.get("value_estimate"),
                        decision_mode=str(dec_d.get("decision_mode", "unavailable")),
                        policy_version=dec_d.get("policy_version"),
                        checkpoint_id=dec_d.get("checkpoint_id"),
                        mask_type=str(dec_d.get("mask_type", "actual_decision_time")),
                        action_probabilities_source=str(
                            dec_d.get("action_probabilities_source")
                            or (
                                "recorded_runtime_telemetry"
                                if dec_d.get("action_probabilities") is not None
                                else "unavailable"
                            )
                        ),
                        value_estimate_source=str(
                            dec_d.get("value_estimate_source")
                            or (
                                "recorded_runtime_telemetry"
                                if dec_d.get("value_estimate") is not None
                                else "unavailable"
                            )
                        ),
                        capture_error=dec_d.get("capture_error"),
                    )
                )
        else:
            blocked_sheep_pts = {s.position for s in decision_sheep if not s.penned}
            for d_i, d in enumerate(decision_dogs):
                other_dog_pts = {od.position for od in decision_dogs if od.index != d.index}
                blocked_pts = blocked_sheep_pts | other_dog_pts
                d_mask: dict[str, bool] = {"wait": True}
                for act in ACTION_ORDER:
                    if act == "wait":
                        continue
                    base = _base_action(act)
                    dx, dy = 0, 0
                    if "left" in base:
                        dx = -1
                    elif "right" in base:
                        dx = 1
                    if "up" in base:
                        dy = -1
                    elif "down" in base:
                        dy = 1
                    cand = Point(d.position.x + dx, d.position.y + dy).clamp(width, height)
                    if cand in fence_cells or pen.contains(cand) or cand in blocked_pts:
                        d_mask[act] = False
                    else:
                        d_mask[act] = cand != d.position
                legal_masks[d.index] = d_mask

                act_recorded = (
                    actions[d_i] if d_i < len(actions) else d.last_action
                )
                decisions.append(
                    DogDecisionRecord(
                        dog_index=d.index,
                        world_step=step_num,
                        observation=None,
                        decision_action_mask=d_mask,
                        recomputed_physical_mask=d_mask,
                        selected_action=act_recorded,
                        executed_action=act_recorded,
                        replacement_reason=None,
                        action_probabilities=None,
                        logits=None,
                        value_estimate=None,
                        decision_mode="unavailable",
                        policy_version=None,
                        checkpoint_id=None,
                        mask_type=mask_type,
                    )
                )

        # Route distance and geometry
        unpenned_sheep = [s for s in curr_sheep if not s.penned]
        sheep_route_distances: dict[int, float] = {}
        sheep_waypoints: dict[int, Point] = {}
        sheep_fence_adjacent: dict[int, bool] = {}
        sheep_wall_or_corner: dict[int, bool] = {}

        for s in curr_sheep:
            if s.penned:
                sheep_route_distances[s.index] = 0.0
                sheep_waypoints[s.index] = pen.center
                sheep_fence_adjacent[s.index] = False
                sheep_wall_or_corner[s.index] = False
            else:
                r_dist = float(nav.distance(s.position))
                wp = nav.target(s.position)
                sheep_route_distances[s.index] = r_dist
                sheep_waypoints[s.index] = wp
                adj_fence = any(
                    max(abs(s.position.x - fc.x), abs(s.position.y - fc.y)) <= 1
                    for fc in fence_cells
                )
                sheep_fence_adjacent[s.index] = adj_fence
                sheep_wall_or_corner[s.index] = (
                    _wall_margin(s.position, width, height) <= 2
                    or _is_corner(s.position, width, height, margin=3)
                )

        avg_route_dist = (
            fmean(sheep_route_distances[s.index] for s in unpenned_sheep)
            if unpenned_sheep
            else 0.0
        )
        entry_dist = (
            sum(sheep_route_distances[s.index] for s in unpenned_sheep) / max(1, len(curr_sheep))
        )
        best_gain = entry_dist < best_route_dist - 1e-6
        if best_gain:
            best_route_dist = entry_dist

        curr_no_progress = int(snap.get("no_progress_steps", 0))
        no_progress_reset_without_best_gain = (
            step_num > 1
            and curr_no_progress < prev_no_progress
            and not best_gain
            and len(unpenned_sheep) == sum(1 for s in decision_sheep if not s.penned)
        )
        prev_no_progress = curr_no_progress

        if unpenned_sheep:
            fc_x = round(fmean(s.position.x for s in unpenned_sheep))
            fc_y = round(fmean(s.position.y for s in unpenned_sheep))
            flock_anchor = Point(fc_x, fc_y)
            if flock_anchor in fence_cells or pen.contains(flock_anchor):
                flock_anchor = min(
                    (s.position for s in unpenned_sheep),
                    key=lambda p: p.distance_to(flock_anchor),
                )
            anchor_waypoint = nav.target(flock_anchor)
        else:
            flock_anchor = pen.center
            anchor_waypoint = pen.center

        dog_behind_alignment: dict[int, float] = {}
        dog_on_gate_or_waypoint_side: dict[int, bool] = {}
        dog_nearest_sheep_dist: dict[int, float] = {}

        for d in curr_dogs:
            if not unpenned_sheep:
                dog_behind_alignment[d.index] = 0.0
                dog_on_gate_or_waypoint_side[d.index] = False
                dog_nearest_sheep_dist[d.index] = 0.0
                continue
            nearest_s = min(unpenned_sheep, key=lambda s: d.position.distance_to(s.position))
            nearest_dist = float(d.position.distance_to(nearest_s.position))
            dog_nearest_sheep_dist[d.index] = nearest_dist

            s_wp = sheep_waypoints[nearest_s.index]
            local_align = _alignment_behind_point(d.position, nearest_s.position, s_wp)
            flock_align = _alignment_behind_point(d.position, flock_anchor, anchor_waypoint)
            dog_behind_alignment[d.index] = flock_align
            dog_on_gate_or_waypoint_side[d.index] = (
                nearest_dist <= float(env_config.sheep_vision) and local_align < -0.15
            )

        max_pairwise_sheep_dist = 0.0
        if len(unpenned_sheep) >= 2:
            for i_s, s_a in enumerate(unpenned_sheep):
                for s_b in unpenned_sheep[i_s + 1 :]:
                    max_pairwise_sheep_dist = max(
                        max_pairwise_sheep_dist, s_a.position.distance_to(s_b.position)
                    )
        collector_active = (
            any(a.role == DogRole.COLLECTOR for a in role_assignments.values())
            if has_recorded_step_telemetry
            else any(cd.current_role == DogRole.COLLECTOR for cd in curr_dogs)
        )
        collector_blind_spot = (
            len(unpenned_sheep) >= 2
            and max_pairwise_sheep_dist >= 8.0
            and not collector_active
        )

        corner_target_inverted = False
        for s in unpenned_sheep:
            s_margin = _wall_margin(s.position, width, height)
            if s_margin <= 1:
                for assign in role_assignments.values():
                    if assign.target.distance_to(s.position) <= 5.0:
                        t_margin = _wall_margin(assign.target, width, height)
                        if t_margin > s_margin and assign.role in {
                            DogRole.COLLECTOR,
                            DogRole.REAR_PRESSURE,
                        }:
                            corner_target_inverted = True
                            break

        any_dog_outside_vision = any(
            dog_nearest_sheep_dist.get(d.index, 0.0) > float(env_config.sheep_vision)
            and role_assignments.get(
                d.index, RoleAssignment(d.index, DogRole.REAR_PRESSURE, d.position)
            ).role
            in {DogRole.COLLECTOR, DogRole.REAR_PRESSURE}
            for d in curr_dogs
        )

        reconstructed_steps.append(
            _ReconstructedStep(
                step=step_num,
                actions=actions,
                dogs=curr_dogs,
                sheep=curr_sheep,
                penned_count=int(snap.get("penned_count", len(curr_sheep) - len(unpenned_sheep))),
                unpenned_count=len(unpenned_sheep),
                average_distance_to_pen=float(snap.get("average_distance_to_pen", avg_route_dist)),
                flock_spread=float(snap.get("flock_spread", 0.0)),
                no_progress_steps=curr_no_progress,
                reward_dict=reward_dict,
                role_assignments=role_assignments,
                strategy_snapshot=reconstructed_strategy,
                guidance_source=guidance_source,
                guidance_certainty=guidance_cert,
                legal_action_masks=legal_masks,
                legal_action_mask_source=mask_source,
                mask_type=mask_type,
                action_probabilities_available=has_recorded_action_probs,
                decisions=decisions,
                sheep_route_distances=sheep_route_distances,
                sheep_waypoints=sheep_waypoints,
                sheep_fence_adjacent=sheep_fence_adjacent,
                sheep_wall_or_corner=sheep_wall_or_corner,
                dog_behind_alignment=dog_behind_alignment,
                dog_on_gate_or_waypoint_side=dog_on_gate_or_waypoint_side,
                dog_nearest_sheep_dist=dog_nearest_sheep_dist,
                best_route_distance_so_far=best_route_dist,
                no_progress_reset_without_best_gain=no_progress_reset_without_best_gain,
                collector_blind_spot=collector_blind_spot,
                corner_target_inverted=corner_target_inverted,
                any_dog_outside_sheep_vision_oscillating=any_dog_outside_vision,
                has_recorded_step_telemetry=(
                    isinstance(step_telemetry, dict)
                    and bool(step_telemetry.get("pre_step_roles") or step_telemetry.get("decisions"))
                    and not bool(step_telemetry.get("telemetry_omitted"))
                    and not bool(step_telemetry.get("telemetry_truncated"))
                ),
                has_recorded_debug_targets=(
                    isinstance(debug_payload, dict) and bool(debug_payload.get("dogs"))
                ),
                step_telemetry_dict=step_telemetry if isinstance(step_telemetry, dict) else None,
                pre_step_dogs=pre_step_dogs_copy,
                pre_step_sheep=pre_step_sheep_copy,
                has_prior_contiguous_step=has_prior_contiguous_step,
            )
        )

        prev_dogs_by_idx = {d.index: d for d in curr_dogs}
        prev_sheep_by_idx = {s.index: s for s in curr_sheep}
        prev_step_num = step_num

    audit = audit_replay_evidence(
        bundle,
        has_initial_step_zero=has_initial_step_zero,
        has_contiguous_frames=has_contiguous,
        has_recorded_debug_targets=has_recorded_debug_targets,
        has_recorded_step_telemetry=has_recorded_step_telemetry,
        has_recorded_action_probabilities=has_recorded_action_probs,
        discount_factor=discount_factor,
        discount_timebase=discount_timebase,
        strategy_version=strategy_version,
        trainer_type=trainer_type,
        checkpoint_gamma=checkpoint_gamma,
        evaluation_gamma=evaluation_gamma,
        gamma_mismatch=gamma_mismatch,
        telemetry_truncated=telemetry_truncated,
        telemetry_limit_steps=telemetry_limit_steps,
        telemetry_truncation_step=telemetry_truncation_step,
        telemetry_omitted=telemetry_omitted,
        telemetry_mode=telemetry_mode,
        telemetry_head_steps=telemetry_head_steps,
        telemetry_tail_steps=telemetry_tail_steps,
        telemetry_coverage=telemetry_coverage,
    )
    return (
        reconstructed_steps,
        pen,
        width,
        height,
        audit,
        discount_factor,
        discount_timebase,
        dog_count,
    )


def _detect_reward_cycles(
    sub_steps: list[_ReconstructedStep],
    discount_factor: float | None,
    discount_timebase: str | None = None,
    dog_count: int = 3,
    *,
    trainer_type: str | None = None,
    gamma_mismatch: bool = False,
) -> tuple[RewardCycleRecord, ...]:
    """Identify candidate complete and incomplete cycles, evaluating lasting state changes and discounting."""
    if len(sub_steps) < 2:
        return ()

    joint_states = [
        (
            tuple((d.position.x, d.position.y) for d in st.dogs),
            tuple((s.position.x, s.position.y) for s in st.sheep if not s.penned),
        )
        for st in sub_steps
    ]

    cycles: list[RewardCycleRecord] = []
    visited_intervals: set[tuple[int, int]] = set()

    i = 0
    while i < len(sub_steps) - 1:
        match_j = None
        for j in range(i + 1, len(sub_steps)):
            if joint_states[j] == joint_states[i]:
                match_j = j
                break
        if match_j is not None and (sub_steps[i].step, sub_steps[match_j].step) not in visited_intervals:
            s_start = sub_steps[i]
            s_end = sub_steps[match_j]
            cycle_steps = sub_steps[i + 1 : match_j + 1]
            pos_comps: dict[str, float] = {}
            neg_comps: dict[str, float] = {}
            net_comps: dict[str, float] = {}
            total_r = 0.0

            cycle_is_contiguous = (
                all(
                    cycle_steps[k].step == cycle_steps[k - 1].step + 1
                    for k in range(1, len(cycle_steps))
                )
                and (cycle_steps[0].step == s_start.step + 1)
            )
            discounted_ret, disc_status = _compute_discounted_return(
                cycle_steps,
                discount_factor,
                discount_timebase,
                dog_count,
                cycle_is_contiguous=cycle_is_contiguous,
                trainer_type=trainer_type,
                gamma_mismatch=gamma_mismatch,
            )

            for st in cycle_steps:
                step_tot = st.reward_dict.get("total", sum(st.reward_dict.values()))
                total_r += step_tot
                for k_name, val in st.reward_dict.items():
                    if val > 0:
                        pos_comps[k_name] = pos_comps.get(k_name, 0.0) + val
                    elif val < 0:
                        neg_comps[k_name] = neg_comps.get(k_name, 0.0) + val
                    net_comps[k_name] = net_comps.get(k_name, 0.0) + val

            penned_delta = s_end.penned_count - s_start.penned_count
            route_prog = s_start.average_distance_to_pen - s_end.average_distance_to_pen
            has_lasting = bool(penned_delta > 0 or route_prog >= 1.0)
            has_pos_steps = any(
                pos_comps.get(c, 0.0) > 1e-4 for c in ("progress_to_pen", "farthest_sheep_progress")
            )

            if has_lasting:
                c_class = "productive_cycle"
                desc = (
                    f"Complete cycle steps {s_start.step}..{s_end.step} with lasting progress "
                    f"(+{route_prog:.1f} route progress, +{penned_delta} penned)."
                )
            elif total_r > 1e-4 or (discounted_ret is not None and discounted_ret > 1e-4):
                c_class = "candidate_position_recurrence_positive_return"
                desc = (
                    f"Candidate position recurrence steps {s_start.step}..{s_end.step} with net positive return "
                    f"(+{total_r:.3f}) and zero lasting progress. "
                    f"(Joint grid positions matched; hidden state such as no_progress_steps or movement_budget may differ)."
                )
            elif has_pos_steps:
                c_class = "net_negative_cycle_with_positive_steps"
                desc = (
                    f"Complete cycle steps {s_start.step}..{s_end.step} with positive progress steps "
                    f"but net negative return ({total_r:.3f}). Penalty dominates; not reward farming."
                )
            else:
                c_class = "net_negative_cycle"
                desc = (
                    f"Complete cycle steps {s_start.step}..{s_end.step} with net negative return "
                    f"({total_r:.3f}) and zero positive progress."
                )

            cycles.append(
                RewardCycleRecord(
                    cycle_id=f"cycle_{s_start.step}_{s_end.step}",
                    start_step=s_start.step,
                    end_step=s_end.step,
                    duration_steps=s_end.step - s_start.step,
                    is_complete_cycle=True,
                    recurrence_method="candidate_joint_position_recurrence",
                    matched_state="joint_grid_positions",
                    unmatched_or_unavailable_state=(
                        "sub_step_movement_budget",
                        "role_duration_history",
                        "sheep_panic_state",
                        "no_progress_watermark",
                        "rng_state",
                    ),
                    incomplete_reason=None,
                    positive_reward_by_component=pos_comps,
                    negative_reward_by_component=neg_comps,
                    net_reward_by_component=net_comps,
                    total_reward=total_r,
                    discount_factor=discount_factor,
                    discount_timebase=discount_timebase,
                    discounted_return=discounted_ret,
                    discounting_status=disc_status,
                    sheep_penned_delta=penned_delta,
                    net_route_progress=route_prog,
                    has_lasting_state_change=has_lasting,
                    cycle_classification=c_class,
                    description=desc,
                )
            )
            visited_intervals.add((s_start.step, s_end.step))
            i = match_j
        else:
            i += 1

    if not cycles:
        s_start = sub_steps[0]
        s_end = sub_steps[-1]
        cycle_steps = sub_steps[1:]
        pos_comps = {}
        neg_comps = {}
        net_comps = {}
        total_r = 0.0

        cycle_is_contiguous = (
            all(
                cycle_steps[k].step == cycle_steps[k - 1].step + 1
                for k in range(1, len(cycle_steps))
            )
            and (cycle_steps[0].step == s_start.step + 1)
        )
        discounted_ret, disc_status = _compute_discounted_return(
            cycle_steps,
            discount_factor,
            discount_timebase,
            dog_count,
            cycle_is_contiguous=cycle_is_contiguous,
            trainer_type=trainer_type,
            gamma_mismatch=gamma_mismatch,
        )

        for st in cycle_steps:
            step_tot = st.reward_dict.get("total", sum(st.reward_dict.values()))
            total_r += step_tot
            for k_name, val in st.reward_dict.items():
                if val > 0:
                    pos_comps[k_name] = pos_comps.get(k_name, 0.0) + val
                elif val < 0:
                    neg_comps[k_name] = neg_comps.get(k_name, 0.0) + val
                net_comps[k_name] = net_comps.get(k_name, 0.0) + val

        penned_delta = s_end.penned_count - s_start.penned_count
        route_prog = s_start.average_distance_to_pen - s_end.average_distance_to_pen
        has_lasting = bool(penned_delta > 0 or route_prog >= 1.0)
        cycles.append(
            RewardCycleRecord(
                cycle_id=f"incomplete_{s_start.step}_{s_end.step}",
                start_step=s_start.step,
                end_step=s_end.step,
                duration_steps=s_end.step - s_start.step,
                is_complete_cycle=False,
                recurrence_method="none_incomplete",
                matched_state="none",
                unmatched_or_unavailable_state=(
                    "sub_step_movement_budget",
                    "role_duration_history",
                    "sheep_panic_state",
                    "no_progress_watermark",
                    "rng_state",
                ),
                incomplete_reason="trajectory ended before joint state recurrence",
                positive_reward_by_component=pos_comps,
                negative_reward_by_component=neg_comps,
                net_reward_by_component=net_comps,
                total_reward=total_r,
                discount_factor=discount_factor,
                discount_timebase=discount_timebase,
                discounted_return=discounted_ret,
                discounting_status=disc_status,
                sheep_penned_delta=penned_delta,
                net_route_progress=route_prog,
                has_lasting_state_change=has_lasting,
                cycle_classification="incomplete_cycle",
                description=(
                    f"Incomplete cycle steps {s_start.step}..{s_end.step}: "
                    f"trajectory ended before recurrence."
                ),
            )
        )

    return tuple(cycles)


def _summarize_context_window(
    steps: list[_ReconstructedStep],
    start_idx: int,
    end_idx: int,
    window_type: str,
) -> IntervalContextWindowSummary:
    """Summarize the lead-in or aftermath step slice around a candidate interval."""
    sub = steps[start_idx : end_idx + 1]
    if not sub:
        return IntervalContextWindowSummary(
            window_type=window_type,
            start_step=0,
            end_step=0,
            duration_steps=0,
            unpenned_sheep_start=0,
            unpenned_sheep_end=0,
            sheep_penned_delta=0,
            avg_route_distance_start=0.0,
            avg_route_distance_end=0.0,
            net_route_progress=0.0,
            flock_spread_start=0.0,
            flock_spread_end=0.0,
            dog_role_switches=0,
            behind_flock_alignment_start=0.0,
            behind_flock_alignment_end=0.0,
            summary="No steps in context window.",
        )

    first = sub[0]
    last = sub[-1]
    penned_delta = last.penned_count - first.penned_count
    route_progress = first.average_distance_to_pen - last.average_distance_to_pen

    role_switches = 0
    for i in range(1, len(sub)):
        for d_prev, d_curr in zip(sub[i - 1].dogs, sub[i].dogs, strict=False):
            if d_prev.current_role != d_curr.current_role:
                role_switches += 1

    align_start = (
        fmean(first.dog_behind_alignment.values()) if first.dog_behind_alignment else 0.0
    )
    align_end = fmean(last.dog_behind_alignment.values()) if last.dog_behind_alignment else 0.0

    if penned_delta > 0:
        desc = (
            f"{penned_delta} sheep penned (unpenned {first.unpenned_count}->{last.unpenned_count}); "
            f"avg route distance {first.average_distance_to_pen:.1f}->{last.average_distance_to_pen:.1f}."
        )
    elif route_progress >= 2.0:
        desc = (
            f"Flock advanced toward pen by {route_progress:+.1f} route cells "
            f"({first.average_distance_to_pen:.1f}->{last.average_distance_to_pen:.1f}); "
            f"behind-flock alignment {align_start:+.2f}->{align_end:+.2f}."
        )
    elif route_progress <= -2.0:
        desc = (
            f"Flock route distance increased by {-route_progress:.1f} cells "
            f"({first.average_distance_to_pen:.1f}->{last.average_distance_to_pen:.1f}); "
            f"behind-flock alignment {align_start:+.2f}->{align_end:+.2f}."
        )
    else:
        desc = (
            f"Route distance remained near {last.average_distance_to_pen:.1f} "
            f"(net progress {route_progress:+.1f}); {role_switches} role switches."
        )

    return IntervalContextWindowSummary(
        window_type=window_type,
        start_step=first.step,
        end_step=last.step,
        duration_steps=max(1, last.step - first.step + 1),
        unpenned_sheep_start=first.unpenned_count,
        unpenned_sheep_end=last.unpenned_count,
        sheep_penned_delta=penned_delta,
        avg_route_distance_start=first.average_distance_to_pen,
        avg_route_distance_end=last.average_distance_to_pen,
        net_route_progress=route_progress,
        flock_spread_start=first.flock_spread,
        flock_spread_end=last.flock_spread,
        dog_role_switches=role_switches,
        behind_flock_alignment_start=align_start,
        behind_flock_alignment_end=align_end,
        summary=desc,
    )


def _build_interval_analysis(
    steps: list[_ReconstructedStep],
    start_idx: int,
    end_idx: int,
    *,
    interval_index: int,
    replay_id: str,
    sheep_vision: float,
    audit: ReplayEvidenceAudit,
    discount_factor: float | None,
    discount_timebase: str | None = None,
    dog_count: int = 3,
    context_steps: int = 20,
) -> TemporalInteractionInterval:
    """Compute detailed per-dog, per-sheep, geometry, reward, and classification metrics for a slice."""
    sub = steps[start_idx : end_idx + 1]
    first = sub[0]
    last = sub[-1]
    duration = max(1, len(sub))

    lead_start = max(0, start_idx - context_steps)
    lead_end = max(0, start_idx - 1) if start_idx > 0 else 0
    after_start = min(len(steps) - 1, end_idx + 1) if end_idx + 1 < len(steps) else end_idx
    after_end = min(len(steps) - 1, end_idx + context_steps)

    lead_in = _summarize_context_window(steps, lead_start, lead_end, "lead_in")
    aftermath = _summarize_context_window(steps, after_start, after_end, "aftermath")

    has_all_telemetry = all(st.has_recorded_step_telemetry for st in sub)
    has_some_telemetry = any(st.has_recorded_step_telemetry for st in sub)
    has_all_debug = all(st.has_recorded_debug_targets for st in sub)
    has_some_debug = any(st.has_recorded_debug_targets for st in sub)

    if has_all_telemetry:
        interval_guidance_cert = "recorded_runtime_telemetry"
    elif has_some_telemetry:
        interval_guidance_cert = "partial_runtime_telemetry"
    elif has_all_debug:
        interval_guidance_cert = "recorded_runtime_debug"
    elif has_some_debug:
        interval_guidance_cert = "partial_runtime_debug"
    else:
        interval_guidance_cert = audit.guidance_certainty

    dog_count = len(first.dogs)
    dog_summaries: list[DogIntervalSummary] = []

    for d_i in range(dog_count):
        d_idx = first.dogs[d_i].index
        positions = [(st.dogs[d_i].position.x, st.dogs[d_i].position.y) for st in sub]
        roles = [st.dogs[d_i].current_role.value for st in sub]
        actions = [
            st.actions[d_i] if d_i < len(st.actions) else st.dogs[d_i].last_action for st in sub
        ]
        role_counts = Counter(roles)
        dominant_role = role_counts.most_common(1)[0][0] if role_counts else "rear_pressure"
        role_switches = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i - 1])

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        bbox_span = math.hypot(bbox_w, bbox_h)
        net_disp = math.hypot(positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1])
        path_len = sum(
            math.hypot(positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
            for i in range(1, len(positions))
        )
        tortuosity = net_disp / max(1.0, path_len)
        unique_pos = len(set(positions))

        reversals = sum(
            1
            for i in range(1, len(actions))
            if actions[i] != "wait"
            and actions[i - 1] != "wait"
            and _is_reverse_action(actions[i - 1], actions[i])
        )
        reversal_rate = reversals / max(1, len(actions) - 1)
        wait_rate = sum(1 for a in actions if a == "wait") / duration
        sprint_rate = sum(1 for a in actions if a.startswith("sprint_")) / duration
        top_actions = tuple(Counter(actions).most_common(3))

        nearest_dists = [st.dog_nearest_sheep_dist.get(d_idx, 0.0) for st in sub]
        mean_nearest = fmean(nearest_dists) if nearest_dists else 0.0
        within_vision_rate = (
            sum(1 for dist in nearest_dists if dist <= sheep_vision) / duration
        )
        gate_side_rate = (
            sum(1 for st in sub if st.dog_on_gate_or_waypoint_side.get(d_idx, False)) / duration
        )
        aligns = [st.dog_behind_alignment.get(d_idx, 0.0) for st in sub]
        align_mean = fmean(aligns) if aligns else 0.0
        align_delta = (aligns[-1] - aligns[0]) if len(aligns) >= 2 else 0.0

        targets = [
            (
                st.role_assignments[d_idx].target.x,
                st.role_assignments[d_idx].target.y,
            )
            if d_idx in st.role_assignments
            else positions[idx_s]
            for idx_s, st in enumerate(sub)
        ]
        target_dists = [
            math.hypot(positions[idx_s][0] - targets[idx_s][0], positions[idx_s][1] - targets[idx_s][1])
            for idx_s in range(len(sub))
        ]
        legal_counts = [
            sum(1 for allowed in st.legal_action_masks.get(d_idx, {}).values() if allowed)
            for st in sub
        ]

        # Audit decision records for this dog in this interval
        dog_decisions = [
            next((d for d in st.decisions if d.dog_index == d_idx), None) for st in sub
        ]
        has_sel_actions = any(
            d is not None and d.selected_action is not None and d.mask_type == "actual_decision_time"
            for d in dog_decisions
        )
        replacements_cnt = sum(
            1
            for d in dog_decisions
            if d is not None and d.replacement_reason is not None
        )
        has_val_est = any(
            d is not None and d.value_estimate is not None for d in dog_decisions
        )

        limitations: list[str] = []
        if interval_guidance_cert in {"mixed_recorded_and_reconstructed_telemetry", "partial_runtime_telemetry"}:
            limitations.append(
                "Guidance targets have mixed_recorded_and_reconstructed_telemetry (partial_runtime_telemetry): runtime telemetry is only partially recorded across this interval; some steps use reconstructed targets."
            )
        elif interval_guidance_cert == "estimated_reconstruction_unrecorded_initial_strategy_state":
            limitations.append(
                "Guidance targets are reconstructed estimates (estimated_reconstruction_unrecorded_initial_strategy_state): initial strategy state (sticky pen approach axis and dog role hold steps at step 0) was not recorded, so lateral flanker and sticky targets cannot be guaranteed exact."
            )
        elif interval_guidance_cert.startswith("uncertain_") or interval_guidance_cert.startswith("estimated_"):
            limitations.append(
                f"Guidance targets are uncertain estimates ({interval_guidance_cert}); historical replay lacked full config, step 0, or verified strategy version."
            )
        if first.mask_type == "historical_approximate":
            limitations.append(
                "Action masks are historical approximations computed without sub-step fractional movement budget."
            )
        if not first.action_probabilities_available:
            limitations.append(
                "Policy action probabilities and logits were not recorded; action selection confidence is unavailable."
            )

        dog_summaries.append(
            DogIntervalSummary(
                dog_index=d_idx,
                dominant_role=dominant_role,
                role_distribution=dict(role_counts),
                role_switches=role_switches,
                start_position=positions[0],
                end_position=positions[-1],
                net_displacement=net_disp,
                total_path_length=path_len,
                tortuosity_ratio=tortuosity,
                unique_positions_count=unique_pos,
                bounding_box_width=bbox_w,
                bounding_box_height=bbox_h,
                bounding_box_span=bbox_span,
                action_reversal_rate=reversal_rate,
                wait_rate=wait_rate,
                sprint_rate=sprint_rate,
                top_actions=top_actions,
                mean_distance_to_nearest_sheep=mean_nearest,
                within_sheep_vision_rate=within_vision_rate,
                on_gate_or_waypoint_side_rate=gate_side_rate,
                behind_flock_alignment_mean=align_mean,
                behind_flock_alignment_delta=align_delta,
                guidance_source=first.guidance_source,
                guidance_certainty=interval_guidance_cert,
                evidence_limitations=tuple(limitations),
                start_target=targets[0],
                end_target=targets[-1],
                mean_distance_to_target=fmean(target_dists) if target_dists else 0.0,
                legal_action_mask_source=first.legal_action_mask_source,
                mask_type=first.mask_type,
                mean_legal_action_count=fmean(legal_counts) if legal_counts else 0.0,
                action_probabilities_available=first.action_probabilities_available,
                selected_actions_available=has_sel_actions,
                action_replacements_count=replacements_cnt,
                value_estimates_available=has_val_est,
            )
        )

    sheep_count = len(first.sheep)
    sheep_summaries: list[SheepIntervalSummary] = []
    affected_sheep_indices: list[int] = []

    for s_i in range(sheep_count):
        s_idx = first.sheep[s_i].index
        s_first = first.sheep[s_i]
        s_last = last.sheep[s_i]
        if s_first.penned and s_last.penned:
            continue
        affected_sheep_indices.append(s_idx)
        s_positions = [(st.sheep[s_i].position.x, st.sheep[s_i].position.y) for st in sub]
        r_dists = [st.sheep_route_distances.get(s_idx, 0.0) for st in sub]
        s_xs = [p[0] for p in s_positions]
        s_ys = [p[1] for p in s_positions]
        s_span = math.hypot(max(s_xs) - min(s_xs), max(s_ys) - min(s_ys))
        s_path = sum(
            math.hypot(
                s_positions[i][0] - s_positions[i - 1][0],
                s_positions[i][1] - s_positions[i - 1][1],
            )
            for i in range(1, len(s_positions))
        )
        revisits = sum(
            1
            for i in range(2, len(s_positions))
            if s_positions[i] in s_positions[max(0, i - 6) : i - 1]
            and s_positions[i] != s_positions[i - 1]
        )
        revisit_rate = revisits / max(1, len(s_positions) - 2)
        fence_rate = (
            sum(1 for st in sub if st.sheep_fence_adjacent.get(s_idx, False)) / duration
        )
        wall_rate = (
            sum(1 for st in sub if st.sheep_wall_or_corner.get(s_idx, False)) / duration
        )
        packed_rate = sum(1 for st in sub if st.sheep[s_i].is_packed) / duration
        wp_start = first.sheep_waypoints.get(s_idx, s_first.position)
        wp_end = last.sheep_waypoints.get(s_idx, s_last.position)

        sheep_summaries.append(
            SheepIntervalSummary(
                sheep_index=s_idx,
                personality=s_first.personality or "obedient",
                penned_at_start=s_first.penned,
                penned_at_end=s_last.penned,
                start_position=s_positions[0],
                end_position=s_positions[-1],
                start_waypoint=(wp_start.x, wp_start.y),
                end_waypoint=(wp_end.x, wp_end.y),
                start_route_distance=r_dists[0],
                end_route_distance=r_dists[-1],
                min_route_distance=min(r_dists),
                max_route_distance=max(r_dists),
                net_route_progress=r_dists[0] - r_dists[-1],
                unique_positions_count=len(set(s_positions)),
                bounding_box_span=s_span,
                total_path_length=s_path,
                position_revisit_rate=revisit_rate,
                closed_fence_adjacent_rate=fence_rate,
                outer_wall_or_corner_rate=wall_rate,
                packed_rate=packed_rate,
            )
        )

    # Reward & Cycle Accounting
    comp_totals: dict[str, float] = {}
    pos_prog_steps = 0
    neg_prog_steps = 0
    for st in sub:
        for k, v in st.reward_dict.items():
            comp_totals[k] = comp_totals.get(k, 0.0) + v
        prog_r = st.reward_dict.get("progress_to_pen", 0.0) + st.reward_dict.get(
            "farthest_sheep_progress", 0.0
        )
        if prog_r > 1e-6:
            pos_prog_steps += 1
        elif prog_r < -1e-6:
            neg_prog_steps += 1

    total_rew = comp_totals.get("total", sum(comp_totals.values()))
    net_route_prog = first.average_distance_to_pen - last.average_distance_to_pen
    newly_penned_in_interval = last.penned_count - first.penned_count
    zero_net_with_pos_reward = (
        newly_penned_in_interval == 0
        and net_route_prog <= 1.0
        and pos_prog_steps >= max(3, duration // 5)
    )

    detected_cycles = _detect_reward_cycles(
        sub,
        discount_factor,
        discount_timebase,
        dog_count,
        trainer_type=audit.trainer_type,
        gamma_mismatch=audit.gamma_mismatch,
    )
    has_pos_cycle = any(
        c.cycle_classification in {"candidate_position_recurrence_positive_return", "exploitable_positive_cycle"}
        for c in detected_cycles
    )
    has_neg_pos_cycle = any(
        c.cycle_classification == "net_negative_cycle_with_positive_steps"
        for c in detected_cycles
    )
    has_incomp_cycle = any(
        c.cycle_classification == "incomplete_cycle" for c in detected_cycles
    )

    reward_summary = RewardIntervalSummary(
        total_reward=total_rew,
        mean_step_reward=total_rew / duration,
        component_totals=comp_totals,
        positive_progress_reward_steps=pos_prog_steps,
        negative_progress_reward_steps=neg_prog_steps,
        zero_net_progress_with_positive_reward_steps=zero_net_with_pos_reward,
        cycles=detected_cycles,
        has_candidate_positive_return_cycle=has_pos_cycle,
        has_exploitable_positive_cycle=has_pos_cycle,
        has_net_negative_cycle_with_positive_steps=has_neg_pos_cycle,
        has_incomplete_cycle=has_incomp_cycle,
    )

    # Diagnostic counters
    np_resets = sum(1 for st in sub if st.no_progress_reset_without_best_gain)
    collector_blind_steps = sum(1 for st in sub if st.collector_blind_spot)
    corner_inv_steps = sum(1 for st in sub if st.corner_target_inverted)
    outside_vision_steps = sum(1 for st in sub if st.any_dog_outside_sheep_vision_oscillating)

    mean_wait_rate = fmean(d.wait_rate for d in dog_summaries) if dog_summaries else 0.0
    max_dog_net_disp = max((d.net_displacement for d in dog_summaries), default=0.0)
    max_dog_span = max((d.bounding_box_span for d in dog_summaries), default=0.0)
    mean_dog_unique_ratio = (
        fmean(d.unique_positions_count / duration for d in dog_summaries)
        if dog_summaries
        else 1.0
    )
    max_gate_side_rate = max(
        (d.on_gate_or_waypoint_side_rate for d in dog_summaries), default=0.0
    )
    mean_align_delta = (
        fmean(d.behind_flock_alignment_delta for d in dog_summaries) if dog_summaries else 0.0
    )
    mean_align_end = (
        fmean(
            d.behind_flock_alignment_mean + 0.5 * d.behind_flock_alignment_delta
            for d in dog_summaries
        )
        if dog_summaries
        else 0.0
    )
    total_role_switches = sum(d.role_switches for d in dog_summaries)
    max_fence_rate = max((s.closed_fence_adjacent_rate for s in sheep_summaries), default=0.0)
    max_wall_rate = max((s.outer_wall_or_corner_rate for s in sheep_summaries), default=0.0)

    # Classification logic
    combined_route_prog = net_route_prog + aftermath.net_route_progress
    is_productive_repositioning = (
        net_route_prog >= -6.0
        and (
            newly_penned_in_interval > 0
            or aftermath.sheep_penned_delta > 0
            or net_route_prog >= 3.0
            or (aftermath.net_route_progress >= 3.0 and combined_route_prog >= 0.0)
            or (
                mean_align_delta >= 0.25
                and mean_align_end >= 0.15
                and max_dog_net_disp >= 5.0
                and net_route_prog >= -4.0
            )
        )
        and (max_dog_net_disp >= 5.0 or max_dog_span >= 7.0 or mean_align_delta >= 0.20)
        and mean_dog_unique_ratio >= 0.30
    )

    is_stationary_hold = (
        not is_productive_repositioning
        and (
            mean_wait_rate >= 0.65
            or all(d.unique_positions_count <= 2 and d.total_path_length <= 3.0 for d in dog_summaries)
        )
        and abs(net_route_prog) <= 2.0
    )

    has_bounded_dog_loop = any(
        d.total_path_length >= max(4.0, duration * 0.35)
        and (
            d.unique_positions_count <= max(4, int(duration * 0.30))
            or d.bounding_box_span <= 8.5
            or d.action_reversal_rate >= 0.22
            or d.tortuosity_ratio <= 0.22
        )
        for d in dog_summaries
    )
    has_sheep_revisit_loop = any(
        s.total_path_length >= 3.0
        and (s.position_revisit_rate >= 0.18 or s.unique_positions_count <= max(5, int(duration * 0.25)))
        for s in sheep_summaries
    )
    is_oscillation_loop = (
        not is_productive_repositioning
        and not is_stationary_hold
        and (has_bounded_dog_loop or has_sheep_revisit_loop)
        and net_route_prog < 3.0
        and newly_penned_in_interval == 0
    )

    is_counterproductive_pressure = (
        not is_productive_repositioning
        and not is_stationary_hold
        and (
            net_route_prog <= -2.5
            or (
                max_gate_side_rate >= 0.30
                and any(d.within_sheep_vision_rate >= 0.35 for d in dog_summaries)
            )
            or (
                (max_fence_rate >= 0.50 or max_wall_rate >= 0.50)
                and any(d.within_sheep_vision_rate >= 0.35 for d in dog_summaries)
                and net_route_prog <= 1.0
            )
        )
        and not is_oscillation_loop
    )

    if is_productive_repositioning:
        classification = "productive_repositioning"
    elif is_stationary_hold:
        classification = "stationary_hold"
    elif is_oscillation_loop:
        classification = "oscillation_loop"
    elif is_counterproductive_pressure:
        classification = "counterproductive_pressure"
    else:
        classification = "unresolved_ambiguity"

    mechanisms: list[str] = []
    evidence: list[str] = [
        f"[Recorded Observation] Steps {first.step}..{last.step} ({duration} steps) with {first.unpenned_count} unpenned sheep: "
        f"BFS legal route distance {first.average_distance_to_pen:.1f} -> {last.average_distance_to_pen:.1f} "
        f"[Derived Measurement: net progress {net_route_prog:+.1f} cells]."
    ]

    if max_fence_rate >= 0.40 and max_gate_side_rate >= 0.25:
        mechanisms.append("wrong_side_fence_overshoot")
        evidence.append(
            f"[Hypothesized Mechanism: wrong_side_fence_overshoot] Sheep spent {max_fence_rate * 100:.0f}% of interval adjacent to closed pen fence while "
            f"a dog was on the waypoint/gate side {max_gate_side_rate * 100:.0f}% of steps. (Proximity/alignment alone does not prove physical blockage; controlled ablation needed)."
        )
    if collector_blind_steps >= max(3, duration // 4):
        mechanisms.append("two_sheep_collector_blind_spot")
        evidence.append(
            f"[Hypothesized Mechanism: two_sheep_collector_blind_spot] Unpenned sheep were separated across the field/pen for {collector_blind_steps}/{duration} steps "
            f"while TeamStrategy.collector_active remained False."
        )
    if corner_inv_steps >= max(3, duration // 5):
        mechanisms.append("corner_target_clamping_inversion")
        evidence.append(
            f"[Hypothesized Mechanism: corner_target_clamping_inversion] Wall/corner sheep had an inward-clamped dog target for {corner_inv_steps}/{duration} steps, "
            f"placing the dog on the open-field side of the trapped sheep."
        )
    if outside_vision_steps >= max(5, duration // 3):
        mechanisms.append("dog_outside_sheep_vision_loop")
        evidence.append(
            f"[Hypothesized Mechanism: dog_outside_sheep_vision_loop] At least one pressure/collector dog remained outside sheep_vision ({sheep_vision:.0f} cells) "
            f"for {outside_vision_steps}/{duration} steps, exerting zero repulsion on target sheep."
        )
    if total_role_switches >= max(6, duration // 4):
        mechanisms.append("role_churn_target_jumping")
        evidence.append(
            f"[Hypothesized Mechanism: role_churn_target_jumping] Dogs switched TeamStrategy roles {total_role_switches} times over {duration} steps."
        )
    if np_resets >= 1 and net_route_prog <= 1.0:
        mechanisms.append("no_progress_spread_jitter_reset")
        evidence.append(
            f"[Hypothesized Mechanism: no_progress_spread_jitter_reset] no_progress_steps reset {np_resets} time(s) without improving best route distance "
            f"(start={first.no_progress_steps}, end={last.no_progress_steps})."
        )
    if has_pos_cycle:
        mechanisms.append("candidate_position_recurrence_positive_return")
        evidence.append(
            f"[Derived Measurement: candidate_position_recurrence_positive_return] Candidate position recurrence achieved positive return (+{reward_summary.total_reward:.2f}) "
            f"without penning sheep or making net route progress. (Joint positions matched; hidden state and transitions may differ; not proof of an exploitable policy loop)."
        )
    elif has_neg_pos_cycle:
        evidence.append(
            f"[Derived Measurement: net_negative_cycle_with_positive_steps] Complete cycle earned positive progress steps but total net return was negative "
            f"({reward_summary.total_reward:.2f}). Penalty terms dominate; not reward farming."
        )
    if classification == "productive_repositioning":
        mechanisms.append("flanking_or_collection_repositioning")
        evidence.append(
            f"[Derived Measurement: flanking_or_collection_repositioning] Dogs displaced up to {max_dog_net_disp:.1f} cells (span {max_dog_span:.1f}), "
            f"behind-flock alignment delta {mean_align_delta:+.2f}, followed by aftermath: {aftermath.summary} "
            f"(Note: improvements in scripted guidance positions must remain distinct from learned autonomous recovery)."
        )

    for d_sum in dog_summaries:
        evidence.append(
            f"  - `D{d_sum.dog_index}` (`{d_sum.dominant_role}`): pos {d_sum.start_position}->{d_sum.end_position}, "
            f"unique={d_sum.unique_positions_count}/{duration}, span={d_sum.bounding_box_span:.1f}, "
            f"reversals={d_sum.action_reversal_rate * 100:.0f}%, target={d_sum.start_target}->{d_sum.end_target} "
            f"[{d_sum.guidance_certainty}], legal_actions={d_sum.mean_legal_action_count:.1f} [{d_sum.mask_type}]."
        )

    alternatives: list[str] = []
    if not audit.has_recorded_action_probabilities:
        alternatives.append(
            "[Alternative Hypothesis: Action Distribution Ambiguity] Because policy action probabilities/logits and raw observations were not recorded in this replay, "
            "we cannot distinguish from replay data alone whether action cycling was driven by near-tied policy logits (memoryless observation aliasing) "
            "or strong policy preference toward current features."
        )
    if first.guidance_certainty.startswith("uncertain_") or first.guidance_certainty.startswith("estimated_"):
        alternatives.append(
            f"[Alternative Hypothesis: Guidance Uncertainty] Role targets are reconstructed estimates ({first.guidance_certainty}); "
            f"if historical weights, unrecorded initial strategy state, or non-default options were active, actual target coordinates may differ."
        )
    if not alternatives:
        alternatives.append(
            "[Controlled Experiment Required] Controlled ablation is required to separate policy representation limits from reward shaping incentives."
        )

    interval_limitations = list(dict.fromkeys(lim for d in dog_summaries for lim in d.evidence_limitations))

    return TemporalInteractionInterval(
        interval_id=f"{replay_id}:steps-{first.step}-{last.step}",
        replay_id=replay_id,
        classification=classification,
        start_step=first.step,
        end_step=last.step,
        duration_steps=duration,
        unpenned_sheep_count=first.unpenned_count,
        affected_dog_indices=tuple(d.dog_index for d in dog_summaries),
        affected_sheep_indices=tuple(affected_sheep_indices),
        avg_route_distance_start=first.average_distance_to_pen,
        avg_route_distance_end=last.average_distance_to_pen,
        net_route_progress=net_route_prog,
        flock_spread_start=first.flock_spread,
        flock_spread_end=last.flock_spread,
        no_progress_steps_start=first.no_progress_steps,
        no_progress_steps_end=last.no_progress_steps,
        no_progress_resets_without_best_route_gain=np_resets,
        collector_blind_spot_steps=collector_blind_steps,
        corner_target_inversion_steps=corner_inv_steps,
        dogs_outside_sheep_vision_steps=outside_vision_steps,
        lead_in=lead_in,
        aftermath=aftermath,
        dog_summaries=tuple(dog_summaries),
        sheep_summaries=tuple(sheep_summaries),
        reward_summary=reward_summary,
        candidate_mechanisms=tuple(mechanisms),
        supporting_evidence=tuple(evidence),
        alternative_explanations=tuple(alternatives),
        unavailable_signals=audit.unavailable_fields,
        guidance_certainty=interval_guidance_cert,
        evidence_limitations=tuple(interval_limitations),
    )


def build_interval_plain_language_explanation(
    interval: TemporalInteractionInterval,
) -> dict[str, Any]:
    """Build a structured, plain-language explanation for a detected interval."""
    readable_label = INTERVAL_READABLE_LABELS.get(
        interval.classification, interval.classification
    )
    what_was_observed = (
        f"During world steps {interval.start_step}..{interval.end_step} "
        f"({interval.duration_steps} steps) with {interval.unpenned_sheep_count} unpenned sheep, "
        f"the flock's legal-route distance to the pen changed from "
        f"{interval.avg_route_distance_start:.1f} to {interval.avg_route_distance_end:.1f} cells "
        f"(net progress {interval.net_route_progress:+.1f} cells). "
        f"Lead-in (steps {interval.lead_in.start_step}..{interval.lead_in.end_step}): {interval.lead_in.summary} "
        f"Aftermath (steps {interval.aftermath.start_step}..{interval.aftermath.end_step}): {interval.aftermath.summary}"
    )

    mech_descriptions: list[str] = []
    for mech in interval.candidate_mechanisms:
        if mech == "wrong_side_fence_overshoot":
            mech_descriptions.append(
                "One or more unpenned sheep remained adjacent to a closed pen fence wall while a dog occupied the pen-gate/waypoint side within sheep vision."
            )
        elif mech == "two_sheep_collector_blind_spot":
            mech_descriptions.append(
                "Two or more unpenned sheep were separated across the field while the scripted TeamStrategy did not activate a dedicated Collector role."
            )
        elif mech == "corner_target_clamping_inversion":
            mech_descriptions.append(
                "A sheep near the outer boundary/corner had a scripted dog target clamped inward onto the field, placing the target on the open-field side of the sheep."
            )
        elif mech == "dog_outside_sheep_vision_loop":
            mech_descriptions.append(
                "At least one pressure/collector dog cycled outside the sheep vision radius, exerting zero repulsive force on the unpenned sheep."
            )
        elif mech == "role_churn_target_jumping":
            mech_descriptions.append(
                "Dogs repeatedly switched scripted TeamStrategy roles, shifting target coordinates across consecutive steps."
            )
        elif mech == "no_progress_spread_jitter_reset":
            mech_descriptions.append(
                "The environment's no_progress_steps stall counter reset due to flock-spread or center jitter without achieving a new best route distance to the pen."
            )
        elif mech == "candidate_position_recurrence_positive_return":
            mech_descriptions.append(
                "Dogs and sheep revisited an earlier joint grid configuration while accumulating positive step reward without penning a sheep."
            )
        elif mech == "flanking_or_collection_repositioning":
            mech_descriptions.append(
                "Dogs displaced around the flock to improve behind-flock alignment or gather separated sheep, enabling subsequent route progress."
            )
        else:
            mech_descriptions.append(f"Candidate pattern '{mech}' was flagged by kinematic thresholds.")

    if interval.classification == "productive_repositioning":
        candidate_explanation = (
            "Candidate explanation — Productive repositioning: Although legal-route distance to the pen did not "
            "decrease monotonically during every step of this window, the dogs repositioned around the sheep "
            "and improved alignment or flock cohesion, followed by measurable route progress or penning in the aftermath. "
            + (" ".join(mech_descriptions) if mech_descriptions else "")
        ).strip()
    elif interval.classification == "stationary_hold":
        candidate_explanation = (
            "Candidate explanation — Stationary hold: Dogs executed predominantly wait or minimal-displacement actions "
            "while unpenned sheep remained nearly stationary. This is an observational classification, not proof of policy lockup. "
            + (" ".join(mech_descriptions) if mech_descriptions else "")
        ).strip()
    elif interval.classification == "oscillation_loop":
        candidate_explanation = (
            "Candidate explanation — Repeated movement (oscillation candidate): Dogs or sheep cycled through a bounded "
            "set of grid cells or reversed actions repeatedly without net route progress toward the pen gate. "
            + (" ".join(mech_descriptions) if mech_descriptions else "")
        ).strip()
    elif interval.classification == "counterproductive_pressure":
        candidate_explanation = (
            "Candidate explanation — Counterproductive pressure candidate: Derived geometry indicates dogs were positioned "
            "on the pen-waypoint side of unpenned sheep or applying pressure while sheep moved away from the legal pen route. "
            "Geometric alignment and proximity are descriptive evidence, not proven physical blockage. "
            + (" ".join(mech_descriptions) if mech_descriptions else "")
        ).strip()
    else:
        candidate_explanation = (
            "Candidate explanation — Unresolved pattern: Route progress toward the pen stalled during this window, "
            "but kinematic signals do not cleanly isolate a single stationary hold, bounded loop, or productive flank. "
            + (" ".join(mech_descriptions) if mech_descriptions else "")
        ).strip()

    missing_and_alternatives: list[str] = list(interval.alternative_explanations)
    for lim in interval.evidence_limitations:
        if lim not in missing_and_alternatives:
            missing_and_alternatives.append(f"[Reconstruction Limitation] {lim}")
    if interval.unavailable_signals:
        missing_and_alternatives.append(
            "Unavailable signals in this recording: "
            + ", ".join(interval.unavailable_signals)
        )

    controlled_experiment_required = (
        "Observational replay analysis identifies when and where a pattern occurred, but cannot prove causality on its own. "
        "To test whether a candidate mechanism caused the stall, run a controlled experiment comparing an unchanged baseline "
        "against a single isolated change (under identical training/evaluation seeds, episode budgets, and assistance disclosure)."
    )
    moving_away_note = (
        "Why moving away from the pen can be productive: Herding is non-monotonic. Dogs often need to back off to release "
        "counterproductive pressure, circle wide around a closed pen fence or wall corner, or travel away from the gate to "
        "collect a straggler before driving the flock into the pen."
    )

    return {
        "readable_label": readable_label,
        "what_was_observed": what_was_observed,
        "candidate_explanation": candidate_explanation,
        "supporting_evidence": list(interval.supporting_evidence),
        "missing_evidence_and_alternatives": missing_and_alternatives,
        "controlled_experiment_required": controlled_experiment_required,
        "moving_away_note": moving_away_note,
    }


def _analyze_reconstructed_episode_steps(
    bundle: dict[str, Any],
    replay_id: str,
    replay_path: str | None,
    steps: list[_ReconstructedStep],
    pen: Pen,
    width: int,
    height: int,
    audit: ReplayEvidenceAudit,
    discount_factor: float | None,
    discount_timebase: str | None,
    dog_count: int,
    *,
    min_interval_steps: int = 12,
    window_size: int = 16,
    context_steps: int = 20,
    sheep_vision: float = 10.0,
) -> EpisodeTemporalAnalysis:
    """Build EpisodeTemporalAnalysis from already-reconstructed steps."""
    penning_steps: list[tuple[int, int]] = []
    for i in range(1, len(steps)):
        prev_s = steps[i - 1].sheep
        curr_s = steps[i].sheep
        for s_p, s_c in zip(prev_s, curr_s, strict=False):
            if not s_p.penned and s_c.penned:
                penning_steps.append((steps[i].step, s_c.index))

    phases: list[tuple[int, int]] = []
    phase_start = 0
    for i in range(1, len(steps)):
        if steps[i].unpenned_count != steps[phase_start].unpenned_count:
            if steps[phase_start].unpenned_count > 0:
                phases.append((phase_start, i - 1))
            phase_start = i
    if steps and steps[phase_start].unpenned_count > 0:
        phases.append((phase_start, len(steps) - 1))

    candidate_slices: list[tuple[int, int]] = []
    eff_win = max(6, min(window_size, min_interval_steps))

    for p_start, p_end in phases:
        phase_len = p_end - p_start + 1
        if phase_len < min_interval_steps:
            continue

        flagged = [False] * phase_len
        for offset in range(0, phase_len - eff_win + 1):
            w_start = p_start + offset
            w_end = w_start + eff_win - 1
            s0 = steps[w_start]
            s1 = steps[w_end]
            route_gain = s0.average_distance_to_pen - s1.average_distance_to_pen
            if route_gain < 1.5:
                for k in range(offset, offset + eff_win):
                    flagged[k] = True

        idx = 0
        raw_intervals: list[tuple[int, int]] = []
        while idx < phase_len:
            if not flagged[idx]:
                idx += 1
                continue
            seg_start = idx
            while idx < phase_len and flagged[idx]:
                idx += 1
            seg_end = idx - 1
            if raw_intervals and (p_start + seg_start) - raw_intervals[-1][1] <= 5:
                raw_intervals[-1] = (raw_intervals[-1][0], p_start + seg_end)
            else:
                raw_intervals.append((p_start + seg_start, p_start + seg_end))

        for iv_s, iv_e in raw_intervals:
            if iv_e - iv_s + 1 >= min_interval_steps:
                candidate_slices.append((iv_s, iv_e))

    intervals: list[TemporalInteractionInterval] = []
    for idx_iv, (iv_s, iv_e) in enumerate(candidate_slices, start=1):
        intervals.append(
            _build_interval_analysis(
                steps,
                iv_s,
                iv_e,
                interval_index=idx_iv,
                replay_id=replay_id,
                sheep_vision=sheep_vision,
                audit=audit,
                discount_factor=discount_factor,
                discount_timebase=discount_timebase,
                dog_count=dog_count,
                context_steps=context_steps,
            )
        )

    final_snap = bundle.get("final_snapshot") or (
        bundle["frames"][-1]["snapshot"] if bundle.get("frames") else {}
    )
    stats = bundle.get("stats") or {}
    outcome_status = str(final_snap.get("status") or "unknown")
    stop_reason = str(stats.get("stop_reason") or outcome_status)
    total_sheep = len(steps[-1].sheep) if steps else 0
    final_penned = steps[-1].penned_count if steps else 0
    unpenned_coverage = tuple(
        sorted({iv.unpenned_sheep_count for iv in intervals}, reverse=True)
    )

    notes: list[str] = []
    osc_flag = stats.get("oscillation_detected")
    has_osc_interval = any(iv.classification == "oscillation_loop" for iv in intervals)
    if osc_flag is False and has_osc_interval:
        notes.append(
            "stats.oscillation_detected was False (only checks a 2-position loop in the final 4 steps), "
            "whereas temporal window analysis detected sustained multi-step oscillation."
        )
    total_np_resets = sum(iv.no_progress_resets_without_best_route_gain for iv in intervals)
    if total_np_resets > 0:
        notes.append(
            f"no_progress_steps reset {total_np_resets} time(s) during non-progress intervals without "
            f"new best entry distance (triggered by flock_spread or farthest_to_flock_center deltas)."
        )

    return EpisodeTemporalAnalysis(
        replay_id=replay_id,
        replay_path=replay_path,
        seed=bundle.get("seed"),
        policy_name=str(bundle.get("policy_name", "unknown")),
        outcome_status=outcome_status,
        stop_reason=stop_reason,
        total_steps=steps[-1].step if steps else 0,
        total_sheep=total_sheep,
        final_penned_count=final_penned,
        final_unpenned_count=max(0, total_sheep - final_penned),
        pen_geometry={
            "origin": {"x": pen.origin.x, "y": pen.origin.y},
            "width": pen.width,
            "height": pen.height,
            "opening": pen.opening,
            "grid_width": width,
            "grid_height": height,
        },
        penning_steps=tuple(penning_steps),
        evidence_audit=audit,
        intervals=tuple(intervals),
        unpenned_count_coverage=unpenned_coverage,
        summary_notes=tuple(notes),
    )


def analyze_episode_temporal_interactions(
    source: str | Path | dict[str, Any],
    *,
    min_interval_steps: int = 12,
    window_size: int = 16,
    context_steps: int = 20,
    sheep_vision: float = 10.0,
) -> EpisodeTemporalAnalysis:
    """Perform offline temporal interaction analysis on a recorded episode replay.

    Segments the episode by constant unpenned-sheep count (so every phase 6..1 unpenned
    sheep is analyzed cleanly without mixing across penning transitions) and detects
    candidate intervals of repeated behavior, non-progress, wrong-side pressure, or
    productive repositioning.
    """
    bundle, replay_id, replay_path = load_replay_bundle(source)
    (
        steps,
        pen,
        width,
        height,
        audit,
        discount_factor,
        discount_timebase,
        dog_count,
    ) = _reconstruct_episode_steps(bundle)
    return _analyze_reconstructed_episode_steps(
        bundle,
        replay_id,
        replay_path,
        steps,
        pen,
        width,
        height,
        audit,
        discount_factor,
        discount_timebase,
        dog_count,
        min_interval_steps=min_interval_steps,
        window_size=window_size,
        context_steps=context_steps,
        sheep_vision=sheep_vision,
    )


def build_temporal_diagnostic_payload(
    source: str | Path | dict[str, Any],
    *,
    min_interval_steps: int = 12,
    window_size: int = 16,
    context_steps: int = 20,
    sheep_vision: float = 10.0,
    include_replay: bool = False,
) -> dict[str, Any]:
    """Build a complete, synchronized JSON-serializable payload for the web viewer.

    Reuses the Python temporal analyzer (`_reconstruct_episode_steps` +
    `_analyze_reconstructed_episode_steps`) so the frontend never duplicates
    simulation, navigation, or diagnostic logic.
    """
    bundle, replay_id, replay_path = load_replay_bundle(source)
    (
        steps,
        pen,
        width,
        height,
        audit,
        discount_factor,
        discount_timebase,
        dog_count,
    ) = _reconstruct_episode_steps(bundle)
    analysis = _analyze_reconstructed_episode_steps(
        bundle,
        replay_id,
        replay_path,
        steps,
        pen,
        width,
        height,
        audit,
        discount_factor,
        discount_timebase,
        dog_count,
        min_interval_steps=min_interval_steps,
        window_size=window_size,
        context_steps=context_steps,
        sheep_vision=sheep_vision,
    )

    env_meta = bundle.get("environment") or {}
    eff_sheep_vision = float(env_meta.get("sheep_vision", sheep_vision))
    observation_mode = (
        bundle.get("observation_mode")
        or env_meta.get("observation_mode")
        or bundle.get("training", {}).get("observation_mode")
    )

    reward_keys_set: set[str] = set()
    step_evidence: list[dict[str, Any]] = []

    for st in steps:
        for rk in st.reward_dict:
            reward_keys_set.add(rk)

        st_tele = st.step_telemetry_dict
        step_truncated = bool(
            (isinstance(st_tele, dict) and st_tele.get("telemetry_truncated"))
            or (
                audit.telemetry_truncated
                and audit.telemetry_truncation_step is not None
                and st.step >= audit.telemetry_truncation_step
            )
        )
        step_omitted = bool(
            (isinstance(st_tele, dict) and st_tele.get("telemetry_omitted"))
            or (st_tele is None and not st.has_recorded_step_telemetry and audit.telemetry_omitted)
        )
        omitted_range_str: str | None = None
        if step_omitted:
            if isinstance(st_tele, dict) and st_tele.get("omitted_range"):
                omitted_range_str = str(st_tele["omitted_range"])
            elif audit.telemetry_coverage and audit.telemetry_coverage.get("omitted_range"):
                cov_rng = audit.telemetry_coverage["omitted_range"]
                if isinstance(cov_rng, (list, tuple)) and len(cov_rng) == 2:
                    omitted_range_str = f"steps_{cov_rng[0]}_to_{cov_rng[1]}"

        pre_dogs_by_idx = {d.index: d for d in st.pre_step_dogs}
        pre_sheep_by_idx = {s.index: s for s in st.pre_step_sheep}
        decisions_by_dog = {dec.dog_index: dec for dec in st.decisions}

        dog_records: list[dict[str, Any]] = []
        for d in st.dogs:
            pre_d = pre_dogs_by_idx.get(d.index)
            has_pre_pos = bool(st.has_prior_contiguous_step)
            pre_pos_dict = (
                {"x": pre_d.position.x, "y": pre_d.position.y}
                if (pre_d is not None and has_pre_pos)
                else None
            )
            post_pos_dict = {"x": d.position.x, "y": d.position.y}
            disp_dict = (
                {
                    "dx": d.position.x - pre_d.position.x,
                    "dy": d.position.y - pre_d.position.y,
                    "distance": round(d.position.distance_to(pre_d.position), 3),
                }
                if (pre_d is not None and st.has_prior_contiguous_step)
                else None
            )
            assign = st.role_assignments.get(
                d.index,
                RoleAssignment(
                    dog_index=d.index,
                    role=d.current_role,
                    target=d.position,
                    target_sheep_index=None,
                    reason="fallback",
                ),
            )
            dec = decisions_by_dog.get(d.index)
            dec_dict = dec.to_dict() if dec is not None else None
            if dec_dict is not None:
                dec_dict["observation_labeled"] = label_dog_observation_fields(
                    dec.observation if dec is not None else None,
                    observation_mode=observation_mode,
                    telemetry_truncated=step_truncated,
                    telemetry_omitted=step_omitted,
                )

            nearest_dist = st.dog_nearest_sheep_dist.get(d.index)
            dog_records.append(
                {
                    "dog_index": d.index,
                    "pre_step_position": pre_pos_dict,
                    "post_step_position": post_pos_dict,
                    "displacement": disp_dict,
                    "recorded_role": d.current_role.value,
                    "assigned_role": assign.role.value,
                    "target": {"x": assign.target.x, "y": assign.target.y},
                    "target_sheep_index": assign.target_sheep_index,
                    "target_reason": assign.reason,
                    "guidance_source": st.guidance_source,
                    "guidance_certainty": st.guidance_certainty,
                    "distance_to_target": round(d.position.distance_to(assign.target), 3),
                    "nearest_unpenned_sheep_distance": (
                        round(nearest_dist, 3)
                        if (nearest_dist is not None and st.unpenned_count > 0)
                        else None
                    ),
                    "within_sheep_vision": bool(
                        st.unpenned_count > 0
                        and nearest_dist is not None
                        and nearest_dist <= eff_sheep_vision
                    ),
                    "behind_flock_alignment": round(
                        st.dog_behind_alignment.get(d.index, 0.0), 3
                    ),
                    "on_gate_or_waypoint_side": bool(
                        st.dog_on_gate_or_waypoint_side.get(d.index, False)
                    ),
                    "decision": dec_dict,
                }
            )

        sheep_records: list[dict[str, Any]] = []
        for s in st.sheep:
            pre_s = pre_sheep_by_idx.get(s.index)
            pre_s_pos = (
                {"x": pre_s.position.x, "y": pre_s.position.y}
                if (pre_s is not None and st.has_prior_contiguous_step)
                else None
            )
            post_s_pos = {"x": s.position.x, "y": s.position.y}
            s_disp = (
                {
                    "dx": s.position.x - pre_s.position.x,
                    "dy": s.position.y - pre_s.position.y,
                    "distance": round(s.position.distance_to(pre_s.position), 3),
                }
                if (pre_s is not None and st.has_prior_contiguous_step)
                else None
            )
            wp = st.sheep_waypoints.get(s.index, pen.center)
            nearby_dogs: list[dict[str, Any]] = []
            for d in st.dogs:
                d_dist = float(d.position.distance_to(s.position))
                d_assign = st.role_assignments.get(d.index)
                d_align = (
                    0.0
                    if s.penned
                    else _alignment_behind_point(d.position, s.position, wp)
                )
                nearby_dogs.append(
                    {
                        "dog_index": d.index,
                        "role": (
                            d_assign.role.value
                            if d_assign is not None
                            else d.current_role.value
                        ),
                        "distance": round(d_dist, 3),
                        "within_sheep_vision": bool(d_dist <= eff_sheep_vision),
                        "dx_from_sheep": d.position.x - s.position.x,
                        "dy_from_sheep": d.position.y - s.position.y,
                        "alignment_to_waypoint": round(d_align, 3),
                        "on_waypoint_side": bool(
                            not s.penned
                            and d_dist <= eff_sheep_vision
                            and d_align < -0.15
                        ),
                    }
                )
            nearby_dogs.sort(key=lambda item: item["distance"])

            sheep_records.append(
                {
                    "sheep_index": s.index,
                    "pre_step_position": pre_s_pos,
                    "post_step_position": post_s_pos,
                    "displacement": s_disp,
                    "penned": s.penned,
                    "penned_at_pre_step": (
                        pre_s.penned
                        if (pre_s is not None and st.has_prior_contiguous_step)
                        else None
                    ),
                    "personality": s.personality or "obedient",
                    "is_packed": s.is_packed,
                    "route_distance": round(st.sheep_route_distances.get(s.index, 0.0), 3),
                    "waypoint": {"x": wp.x, "y": wp.y},
                    "closed_fence_adjacent": bool(
                        st.sheep_fence_adjacent.get(s.index, False)
                    ),
                    "outer_wall_or_corner": bool(
                        st.sheep_wall_or_corner.get(s.index, False)
                    ),
                    "wall_margin": _wall_margin(s.position, width, height),
                    "nearby_dogs": nearby_dogs,
                }
            )

        fc_pt = st.strategy_snapshot.flock_center
        step_evidence.append(
            {
                "step": st.step,
                "has_prior_contiguous_step": st.has_prior_contiguous_step,
                "actions": list(st.actions),
                "unpenned_count": st.unpenned_count,
                "penned_count": st.penned_count,
                "average_distance_to_pen": round(st.average_distance_to_pen, 3),
                "best_route_distance_so_far": (
                    round(st.best_route_distance_so_far, 3)
                    if math.isfinite(st.best_route_distance_so_far)
                    else None
                ),
                "flock_spread": round(st.flock_spread, 3),
                "no_progress_steps": st.no_progress_steps,
                "no_progress_reset_without_best_gain": st.no_progress_reset_without_best_gain,
                "collector_blind_spot": st.collector_blind_spot,
                "corner_target_inverted": st.corner_target_inverted,
                "any_dog_outside_sheep_vision_oscillating": (
                    st.any_dog_outside_sheep_vision_oscillating
                ),
                "guidance_source": st.guidance_source,
                "guidance_certainty": st.guidance_certainty,
                "legal_action_mask_source": st.legal_action_mask_source,
                "mask_type": st.mask_type,
                "has_recorded_step_telemetry": st.has_recorded_step_telemetry,
                "has_recorded_debug_targets": st.has_recorded_debug_targets,
                "telemetry_truncated": step_truncated,
                "telemetry_omitted": step_omitted,
                "telemetry_omitted_range": omitted_range_str if step_omitted else None,
                "reward": {k: round(v, 4) for k, v in st.reward_dict.items()},
                "strategy": {
                    "flock_center": (
                        {"x": fc_pt.x, "y": fc_pt.y} if fc_pt is not None else None
                    ),
                    "stray_sheep_index": st.strategy_snapshot.stray_sheep_index,
                    "collector_active": any(
                        a.role == DogRole.COLLECTOR
                        for a in st.role_assignments.values()
                    ),
                },
                "dogs": dog_records,
                "sheep": sheep_records,
            }
        )

    payload: dict[str, Any] = {
        "replay_id": replay_id,
        "sheep_vision": eff_sheep_vision,
        "action_order": list(ACTION_ORDER),
        "available_reward_components": sorted(reward_keys_set),
        "analysis": analysis.to_dict(),
        "step_evidence": step_evidence,
    }

    if include_replay:
        raw_frames = list(bundle.get("frames") or [])
        replay_frames = list(raw_frames)
        if (
            replay_frames
            and int(replay_frames[0].get("step", 0)) != 0
            and isinstance(bundle.get("initial_snapshot"), dict)
        ):
            replay_frames.insert(
                0,
                {
                    "step": 0,
                    "actions": [],
                    "snapshot": bundle["initial_snapshot"],
                    "reward": {},
                },
            )
        payload["replay"] = {
            "replay_id": replay_id,
            "seed": bundle.get("seed"),
            "policy_name": str(bundle.get("policy_name", "unknown")),
            "frames": replay_frames,
            "final_snapshot": bundle.get("final_snapshot")
            or (replay_frames[-1]["snapshot"] if replay_frames else {}),
            "stats": bundle.get("stats") or {},
        }

    return payload


def compare_failure_and_recovery_episodes(
    failure: EpisodeTemporalAnalysis,
    recovery: EpisodeTemporalAnalysis,
) -> FailureRecoveryComparison:
    """Compare a failed episode's intervals with a successful recovery episode's intervals."""
    shared_seed = (
        failure.seed if (failure.seed is not None and failure.seed == recovery.seed) else None
    )
    fail_classes = tuple(dict.fromkeys(iv.classification for iv in failure.intervals))
    rec_classes = tuple(dict.fromkeys(iv.classification for iv in recovery.intervals))

    divergences: list[str] = []
    shared_bottlenecks: list[str] = []

    fail_mechs = {m for iv in failure.intervals for m in iv.candidate_mechanisms}
    rec_mechs = {m for iv in recovery.intervals for m in iv.candidate_mechanisms}
    for m in sorted(fail_mechs & rec_mechs):
        shared_bottlenecks.append(f"Both episodes encountered candidate mechanism '{m}'.")

    rec_reposition = [
        iv for iv in recovery.intervals if iv.classification == "productive_repositioning"
    ]
    fail_terminal = failure.intervals[-1] if failure.intervals else None

    if rec_reposition:
        rep_iv = rec_reposition[0]
        divergences.append(
            f"Recovery episode ({recovery.replay_id}) executed productive repositioning in steps "
            f"{rep_iv.start_step}..{rep_iv.end_step} ({rep_iv.unpenned_sheep_count} unpenned sheep), "
            f"followed by: {rep_iv.aftermath.summary}"
        )
    if fail_terminal is not None:
        divergences.append(
            f"Failure episode ({failure.replay_id}) remained in '{fail_terminal.classification}' "
            f"during steps {fail_terminal.start_step}..{fail_terminal.end_step} "
            f"({fail_terminal.duration_steps} steps, {fail_terminal.unpenned_sheep_count} unpenned sheep, "
            f"net route progress {fail_terminal.net_route_progress:+.1f})."
        )
    if failure.total_steps != recovery.total_steps:
        divergences.append(
            f"Recovery completed all {recovery.total_sheep} sheep at step {recovery.total_steps}, "
            f"whereas failure ended at step {failure.total_steps} ({failure.stop_reason}) with "
            f"{failure.final_unpenned_count} unpenned sheep."
        )

    return FailureRecoveryComparison(
        failure_replay_id=failure.replay_id,
        recovery_replay_id=recovery.replay_id,
        shared_seed=shared_seed,
        failure_unpenned_at_end=failure.final_unpenned_count,
        recovery_completion_step=recovery.total_steps,
        failure_dominant_classifications=fail_classes,
        recovery_interval_classifications=rec_classes,
        key_divergences=tuple(divergences),
        shared_bottlenecks=tuple(shared_bottlenecks),
    )


def format_temporal_diagnostic_report(
    analyses: list[EpisodeTemporalAnalysis] | tuple[EpisodeTemporalAnalysis, ...],
    comparisons: list[FailureRecoveryComparison] | tuple[FailureRecoveryComparison, ...] = (),
    *,
    unavailable_sources: list[str] | tuple[str, ...] = (),
) -> str:
    """Format a readable Markdown report from one or more EpisodeTemporalAnalysis objects."""
    lines: list[str] = [
        "# Temporal Dog–Sheep Interaction Diagnostic Report",
        "",
        "> **Epistemic Standards & Key**:",
        "> - `[Recorded Observation]`: Directly serialized fact from replay frames.",
        "> - `[Derived Measurement]`: Computed mathematically from recorded facts (e.g. BFS legal-route distance, reversal rates).",
        "> - `[Uncertain Reconstruction]`: Reconstructed target/mask with explicit stated limitations.",
        "> - `[Hypothesized Mechanism]`: Candidate causal explanation fitting the spatial pattern; requires controlled ablation to prove.",
        "> - `[Cause Supported by Controlled Experiment]`: Reserved exclusively for mechanisms proven via controlled ablation against an unchanged control.",
        "",
        "## Diagnostic Capability & Provenance Summary",
        "",
        "### What the Tool Can Now Establish",
        "- **Passive Decision Telemetry**: Captures decision masks, selected vs executed actions, and recomputed policy distributions without altering environment transitions, re-sampling actions, or mutating RNG state.",
        "- **Transition Discount Timebase Semantics**: Distinguishes sequential shared-dog transitions (weight $\\gamma^{k \\cdot K + K - 1}$ for $K$ dogs) from joint world-step transitions (weight $\\gamma^k$). Refuses to invent aligned returns when timebase is unknown, trainer is unsupported, or checkpoint/configuration gamma mismatches.",
        "- **Strict Reconstruction Provenance**: Verifies all 11 required TeamStrategy configuration fields, strategy implementation version (1.0.0), initial step 0 state, and frame continuity before labeling reconstruction exact; otherwise marks results estimated or uncertain with explicit missing fields.",
        "- **Honest Candidate Recurrence Accounting**: Distinguishes candidate position recurrences (`candidate_position_recurrence_positive_return`) from proven policy exploitation, tracking unobserved hidden state (`no_progress_steps`, panic counters, sub-step movement budgets) that frequently diverges despite matching grid coordinates.",
        "",
        "### What Historical Recordings Cannot Establish",
        "- **Unrecorded Policy Outputs**: Historical replays captured before this sprint lack decision-time policy logits, action probabilities, raw observation vectors, value estimates, and sub-step fractional movement budgets.",
        "- **Uncertain Guidance Targets**: Historical replays lacking step 0 or omitting strategy configuration parameters cannot support exact reconstruction claims and are explicitly tagged with stated limitations.",
        "- **Unverified Discount Alignment**: Historical replays lacking explicit transition timebase metadata cannot establish discounted return alignment; only undiscounted cumulative sums are reported.",
        "",
        "### Prior Conclusions Corrected",
        "- **Outcome & Stop Reason Alignment**: Episodes previously labeled as timeouts (such as seeds 61 and 41) were verified from serialized termination flags to be controlled stalls stopped by the `no-progress` termination rule.",
        "- **Recurrence Classification**: Prior claims of 'exploitable positive cycles' were corrected to candidate position recurrences, recognizing that identical positions frequently conceal divergent internal state.",
        "- **Descriptive Comparisons**: Checkpoint comparisons across different training runs are explicitly labeled descriptive comparisons rather than controlled experiments, preserving experimental rigor.",
        "",
        "### Behavioral Hypotheses Remaining Untested",
        "- **Flanker Target Clamping**: The hypothesis that open-field target clamping at boundaries traps sheep along walls and corners requires controlled ablation under equal compute budgets against an unchanged control.",
        "- **Stray Sheep Threshold (2-Sheep Blind Spot)**: The hypothesis that `TeamStrategy._stray_sheep` fails to activate when exactly 2 sheep remain unpenned requires controlled ablation against a modified threshold.",
        "- **Stall Counter Spread-Jitter Reset**: The hypothesis that flock-spread deltas reset `no_progress_steps` without true route-to-pen gains requires controlled ablation against best-route gating.",
        "",
    ]

    for analysis in analyses:
        pen = analysis.pen_geometry
        lines.append(
            f"## Episode `{analysis.replay_id}` (seed={analysis.seed}, outcome=`{analysis.outcome_status}`, "
            f"stop_reason=`{analysis.stop_reason}`, penned={analysis.final_penned_count}/{analysis.total_sheep}, "
            f"steps={analysis.total_steps})"
        )
        lines.append(
            f"- **[Recorded Observation] Pen Geometry**: origin=({pen['origin']['x']},{pen['origin']['y']}), "
            f"size={pen['width']}x{pen['height']}, opening=`{pen['opening']}` on "
            f"{pen['grid_width']}x{pen['grid_height']} field"
        )
        if analysis.penning_steps:
            pen_str = ", ".join(f"S{s_idx}@step {st}" for st, s_idx in analysis.penning_steps)
            lines.append(f"- **[Recorded Observation] Sheep Penning Events**: {pen_str}")
        else:
            lines.append("- **[Recorded Observation] Sheep Penning Events**: None (0 sheep penned)")

        lines.append(
            f"- **[Audit] Guidance Certainty**: `{analysis.evidence_audit.guidance_certainty}` "
            f"(step0_present={analysis.evidence_audit.has_initial_step_zero}, "
            f"contiguous_frames={analysis.evidence_audit.has_contiguous_frames}, "
            f"full_env_config={analysis.evidence_audit.has_full_environment_config}, "
            f"gamma_available={analysis.evidence_audit.discount_factor_available})"
        )

        if analysis.summary_notes:
            for note in analysis.summary_notes:
                lines.append(f"- **[Derived Measurement] Diagnostic Note**: {note}")
        lines.append("")

        for iv in analysis.intervals:
            lines.append(
                f"### Interval `{iv.interval_id}` — `{iv.classification}` "
                f"(steps {iv.start_step}..{iv.end_step}, duration={iv.duration_steps}, "
                f"unpenned_sheep={iv.unpenned_sheep_count}, guidance=`{iv.guidance_certainty}`)"
            )
            lines.append(
                f"- **[Derived Measurement] Lead-in (steps {iv.lead_in.start_step}..{iv.lead_in.end_step})**: {iv.lead_in.summary}"
            )
            lines.append(
                f"- **[Derived Measurement] Route Progress**: {iv.avg_route_distance_start:.1f} -> "
                f"{iv.avg_route_distance_end:.1f} (net {iv.net_route_progress:+.1f} cells), "
                f"flock spread {iv.flock_spread_start:.1f} -> {iv.flock_spread_end:.1f}, "
                f"`no_progress_steps` {iv.no_progress_steps_start} -> {iv.no_progress_steps_end} "
                f"(resets without best route gain: {iv.no_progress_resets_without_best_route_gain})"
            )
            lines.append(
                f"- **[Derived Measurement] Aftermath (steps {iv.aftermath.start_step}..{iv.aftermath.end_step})**: {iv.aftermath.summary}"
            )
            if iv.candidate_mechanisms:
                lines.append(
                    f"- **[Hypothesized Mechanism] Candidates**: {', '.join(f'`{m}`' for m in iv.candidate_mechanisms)}"
                )
            lines.append("- **Per-Dog Behavior & Decision Evidence**:")
            for d in iv.dog_summaries:
                lines.append(
                    f"  - `D{d.dog_index}` (`{d.dominant_role}`, switches={d.role_switches}): "
                    f"pos {d.start_position}->{d.end_position}, unique={d.unique_positions_count}, "
                    f"span={d.bounding_box_span:.1f}, reversals={d.action_reversal_rate * 100:.0f}%, "
                    f"waypoint_side={d.on_gate_or_waypoint_side_rate * 100:.0f}%, "
                    f"target={d.start_target}->{d.end_target} [`{d.guidance_certainty}`], "
                    f"mask={d.mean_legal_action_count:.1f} [`{d.mask_type}`], "
                    f"action_replacements={d.action_replacements_count}"
                )
            lines.append("- **Unpenned Sheep Responses**:")
            for s in iv.sheep_summaries:
                lines.append(
                    f"  - `S{s.sheep_index}` (`{s.personality}`): pos {s.start_position}->{s.end_position}, "
                    f"route_dist {s.start_route_distance:.1f}->{s.end_route_distance:.1f} "
                    f"(net {s.net_route_progress:+.1f}), fence_adj={s.closed_fence_adjacent_rate * 100:.0f}%, "
                    f"wall_corner={s.outer_wall_or_corner_rate * 100:.0f}%"
                )
            lines.append(
                f"- **[Derived Measurement] Rewards & Cycles**: total={iv.reward_summary.total_reward:+.2f} "
                f"(mean={iv.reward_summary.mean_step_reward:+.3f}/step, "
                f"pos_progress_steps={iv.reward_summary.positive_progress_reward_steps}, "
                f"neg_progress_steps={iv.reward_summary.negative_progress_reward_steps})"
            )
            for c in iv.reward_summary.cycles:
                lines.append(
                    f"  - `Cycle {c.cycle_id}` (`{c.cycle_classification}`): {c.description} "
                    f"[total={c.total_reward:+.3f}, discounted={c.discounted_return}, "
                    f"timebase={c.discount_timebase}, status={c.discounting_status}]"
                )
            if iv.evidence_limitations:
                lines.append("- **[Uncertain Reconstruction] Evidence Limitations**:")
                for lim in iv.evidence_limitations:
                    lines.append(f"  - {lim}")
            lines.append("- **[Alternative Hypothesis] Alternative Explanations**:")
            for alt in iv.alternative_explanations:
                lines.append(f"  - {alt}")
            lines.append("")

    if comparisons:
        lines.append("## Descriptive Checkpoint Comparisons (Not Controlled Experiments)")
        lines.append("")
        for comp in comparisons:
            lines.append(
                f"### `{comp.failure_replay_id}` (fail) vs. `{comp.recovery_replay_id}` (recovery, seed={comp.shared_seed})"
            )
            for div in comp.key_divergences:
                lines.append(f"- {div}")
            for sb in comp.shared_bottlenecks:
                lines.append(f"- {sb}")
            lines.append("")

    if unavailable_sources:
        lines.append("## Unavailable Source Recordings (Historical Files Pruned or Missing)")
        lines.append("")
        for src_id in unavailable_sources:
            lines.append(
                f"- **`{src_id}`**: Source replay recording is not present on disk in `artifacts/replays/` "
                f"or `artifacts/evaluations/replays/`. Under project evidence standards (AGENTS.md), "
                f"missing recordings cannot be fabricated or presented as newly regenerated evidence. "
                f"Prior observational notes for this episode are preserved as historical references only."
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    target_paths = sys.argv[1:]
    if not target_paths:
        print("Usage: python -m sheepdog.training.diagnostics.temporal_interaction <replay.json.gz ...>")
        sys.exit(1)

    analysis_list = [analyze_episode_temporal_interactions(p) for p in target_paths]
    print(format_temporal_diagnostic_report(analysis_list))
