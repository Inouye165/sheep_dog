"""Deterministic 2D sheep herding environment."""

# pylint: disable=too-many-lines
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import inf
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any, cast

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import (
    SHEEP_PERSONALITIES,
    DogRole,
    DogState,
    EpisodeStats,
    Pen,
    Point,
    SheepState,
)
from sheepdog.evaluation.scenarios import SavedScenario

# from sheepdog.observations import DogObservation, RoleAwareObservationBuilder
from sheepdog.observations import (
    DogObservation,
    EmergentObservationBuilder,
    RoleAwareObservationBuilder,
)
from sheepdog.policies.base import Action, Policy, PolicyMode
from sheepdog.rewards import RewardBreakdown, RewardComputer, RewardInputs
from sheepdog.team_strategy import RoleAssignment, StrategySnapshot, TeamStrategy

ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "sprint_up": (0, -1),
    "sprint_down": (0, 1),
    "sprint_left": (-1, 0),
    "sprint_right": (1, 0),
    "wait": (0, 0),
}

ACTION_ORDER: tuple[Action, ...] = (
    "up",
    "down",
    "left",
    "right",
    "sprint_up",
    "sprint_down",
    "sprint_left",
    "sprint_right",
    "wait",
)
OPPOSITE_DIRECTIONS: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "wait": "wait",
}


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Immutable view of one agent."""

    index: int
    x: int
    y: int
    role: str | None = None
    penned: bool = False
    last_action: str = "wait"
    personality: str | None = None
    color: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Immutable environment snapshot for UI playback and replay export."""

    step: int
    simulated_seconds: float
    grid_width: int
    grid_height: int
    field_width: int
    field_height: int
    dogs: tuple[AgentSnapshot, ...]
    sheep: tuple[AgentSnapshot, ...]
    pen: Pen
    fence_cells: tuple[tuple[int, int], ...]
    penned_count: int
    average_distance_to_pen: float
    flock_spread: float
    no_progress_steps: int
    terminated: bool
    timeout: bool
    stopped: bool
    success: bool
    status: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A single frame of simulation history."""

    step: int
    actions: tuple[str, ...]
    snapshot: EnvironmentSnapshot
    reward: RewardBreakdown

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Result of a completed policy run."""

    seed: int
    policy_name: str
    final_snapshot: EnvironmentSnapshot
    stats: EpisodeStats
    replay: tuple[StepRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


class SheepdogEnvironment:
    """Grid-based herding environment with deterministic transitions."""

    def __init__(self, config: LabConfig | None = None) -> None:
        self.config = config or LabConfig()
        self.env_config: EnvironmentConfig = self.config.environment
        self.reward_computer = RewardComputer(self.config.rewards)
        self._rng = Random()
        self._seed = 0
        self._step_count = 0
        self._simulated_seconds = 0.0
        self._dogs: list[DogState] = []
        self._sheep: list[SheepState] = []
        self._pen = Pen(Point(0, 0), 0, 0)
        self._fence_cells: frozenset[Point] = frozenset()
        self._previous_average_distance = 0.0
        self._previous_flock_spread = 0.0
        self._previous_farthest_distance = 0.0
        self._no_progress_steps = 0
        self._reward_total = 0.0
        self._terminated = False
        self._timeout = False
        self._stopped = False
        self._success = False
        self._stop_reason = ""
        self._history: list[StepRecord] = []
        self._stats = EpisodeStats()
        self._team_strategy = TeamStrategy(
            self.env_config.width,
            self.env_config.height,
            self.env_config,
        )
        if config is None:
            obs_mode = "guided"
        else:
            training_config = getattr(config, "training", None)
            obs_mode = getattr(training_config, "observation_mode", "guided")
        self._observation_builder: RoleAwareObservationBuilder | EmergentObservationBuilder

        if obs_mode == "emergent":
            self._observation_builder = EmergentObservationBuilder()
        else:
            self._observation_builder = RoleAwareObservationBuilder()
        self._role_assignments: dict[int, RoleAssignment] = {}
        self._strategy_snapshot = StrategySnapshot(None, 0.0, 0.0, None, False, False)
        self._roles_prepared_step: int | None = None
        self._role_distribution: dict[str, int] = {role.value: 0 for role in DogRole}
        self._dog_role_occupancy: dict[str, dict[str, int]] = {}
        self._role_switches = 0
        self._collector_activations = 0
        self._blocker_activations = 0
        self._sheep_split_events = 0
        self._cumulative_gate_progress = 0.0
        self._controlled_stall_steps = 0
        self._controlled_stall_streak = 0
        self._left_flank_occupancy_steps = 0
        self._right_flank_occupancy_steps = 0
        self._gate_corridor_occupancy_peak = 0.0
        self._gate_corridor_failure_steps = 0

    @property
    def dog_count(self) -> int:
        """Return the number of dogs in this episode."""
        return self.env_config.dogs

    @property
    def sheep_count(self) -> int:
        """Return the number of sheep in this episode."""
        return self.env_config.sheep

    @property
    def pen(self) -> Pen:
        """Return the pen object."""
        return self._pen

    @property
    def dogs(self) -> tuple[DogState, ...]:
        """Return an immutable snapshot of all dog states."""
        return tuple(self._dogs)

    @property
    def sheep(self) -> tuple[SheepState, ...]:
        """Return an immutable snapshot of all sheep states."""
        return tuple(self._sheep)

    @property
    def history(self) -> tuple[StepRecord, ...]:
        """Return the completed step records in order."""
        return tuple(self._history)

    def invalidate_role_assignments(self) -> None:
        """Force role reassignment on the next call to prepare_policy_step."""
        self._roles_prepared_step = None

    def reset(self, seed: int | None = None) -> EnvironmentSnapshot:
        """Reset to a new episode and return the initial snapshot."""
        self._seed = 0 if seed is None else seed
        self._rng = Random(self._seed + self.env_config.seed_offset)
        self._step_count = 0
        self._simulated_seconds = 0.0
        self._no_progress_steps = 0
        self._reward_total = 0.0
        self._terminated = False
        self._timeout = False
        self._stopped = False
        self._success = False
        self._stop_reason = ""
        self._history = []
        self._stats = EpisodeStats()
        self._role_assignments = {}
        self._strategy_snapshot = StrategySnapshot(None, 0.0, 0.0, None, False, False)
        self._roles_prepared_step = None
        self._role_distribution = {role.value: 0 for role in DogRole}
        self._dog_role_occupancy = {
            str(index): {role.value: 0 for role in DogRole} for index in range(self.env_config.dogs)
        }
        self._role_switches = 0
        self._collector_activations = 0
        self._blocker_activations = 0
        self._sheep_split_events = 0
        self._cumulative_gate_progress = 0.0
        self._controlled_stall_steps = 0
        self._controlled_stall_streak = 0
        self._left_flank_occupancy_steps = 0
        self._right_flank_occupancy_steps = 0
        self._gate_corridor_occupancy_peak = 0.0
        self._gate_corridor_failure_steps = 0
        pen_origin = Point(self.env_config.width - self.env_config.pen_width, 1)
        self._pen = Pen(
            pen_origin,
            self.env_config.pen_width,
            self.env_config.pen_height,
            opening=self.env_config.pen_opening,
        )
        self._fence_cells = self._pen.fence_cells()
        self._dogs = self._initial_dogs()
        self._sheep = self._initial_sheep()
        for dog in self._dogs:
            self._record_position_history(dog.recent_positions, dog.position)
        for sheep in self._sheep:
            self._record_position_history(sheep.recent_positions, sheep.position)
        self._previous_average_distance = self._average_distance_to_pen()
        self._previous_flock_spread = self._flock_spread()
        self._previous_farthest_distance = self._farthest_distance_to_pen()
        return self.get_state_snapshot()

    def reset_from_scenario(self, scenario: object) -> EnvironmentSnapshot:
        """Reset using a fixed layout from a :class:`SavedScenario`."""
        if not isinstance(scenario, SavedScenario):
            raise TypeError("scenario must be a SavedScenario")
        self._seed = scenario.seed
        self._rng = Random(self._seed + scenario.seed_offset)
        self._step_count = 0
        self._simulated_seconds = 0.0
        self._no_progress_steps = 0
        self._reward_total = 0.0
        self._terminated = False
        self._timeout = False
        self._stopped = False
        self._success = False
        self._stop_reason = ""
        self._history = []
        self._stats = EpisodeStats()
        self._role_assignments = {}
        self._strategy_snapshot = StrategySnapshot(None, 0.0, 0.0, None, False, False)
        self._roles_prepared_step = None
        self._role_distribution = {role.value: 0 for role in DogRole}
        self._dog_role_occupancy = {
            str(index): {role.value: 0 for role in DogRole}
            for index in range(len(scenario.dogs))
        }
        self._role_switches = 0
        self._collector_activations = 0
        self._blocker_activations = 0
        self._sheep_split_events = 0
        self._cumulative_gate_progress = 0.0
        self._controlled_stall_steps = 0
        self._controlled_stall_streak = 0
        self._left_flank_occupancy_steps = 0
        self._right_flank_occupancy_steps = 0
        self._gate_corridor_occupancy_peak = 0.0
        self._gate_corridor_failure_steps = 0
        self._pen = Pen(
            Point(scenario.pen.origin_x, scenario.pen.origin_y),
            scenario.pen.width,
            scenario.pen.height,
            opening=scenario.pen.opening,
        )
        self._fence_cells = self._pen.fence_cells()
        self._dogs = [
            DogState(index=layout.index, position=Point(layout.x, layout.y).clamp(
                scenario.width, scenario.height
            ))
            for layout in sorted(scenario.dogs, key=lambda item: item.index)
        ]
        self._sheep = []
        for layout in sorted(scenario.sheep, key=lambda item: item.index):
            personality = layout.personality or "obedient"
            if personality not in SHEEP_PERSONALITIES:
                personality = "obedient"
            self._sheep.append(
                SheepState(
                    index=layout.index,
                    position=Point(layout.x, layout.y).clamp(scenario.width, scenario.height),
                    personality=personality,
                )
            )
        for dog in self._dogs:
            self._record_position_history(dog.recent_positions, dog.position)
        for sheep in self._sheep:
            self._record_position_history(sheep.recent_positions, sheep.position)
        self._previous_average_distance = self._average_distance_to_pen()
        self._previous_flock_spread = self._flock_spread()
        self._previous_farthest_distance = self._farthest_distance_to_pen()
        return self.get_state_snapshot()

    def _advance_position(
        self,
        position: Point,
        action: Action,
        steps: int,
        blocked_positions: set[Point] | None = None,
    ) -> Point:
        current = position
        blocked = blocked_positions or set()
        for _ in range(max(0, steps)):
            candidate = self._target_position(current, action)
            if candidate == current or candidate in blocked:
                break
            current = candidate
        return current

    def _movement_steps(
        self,
        speed: float,
        carry: float,
        *,
        allow_accumulation: bool = True,
    ) -> tuple[int, float]:
        if not allow_accumulation:
            return 0, 0.0
        budget = max(0.0, carry) + max(0.0, speed)
        steps = int(budget)
        return steps, budget - steps

    def _is_sprint_action(self, action: Action | str) -> bool:
        return str(action).startswith("sprint_")

    def _base_action(self, action: Action | str) -> str:
        text = str(action)
        if text.startswith("sprint_"):
            return text.removeprefix("sprint_")
        return text

    def _dog_action_speed(self, action: Action) -> float:
        if self._is_sprint_action(action):
            return self.env_config.dog_speed * self.env_config.dog_sprint_multiplier
        return self.env_config.dog_speed

    def _sheep_display_color(self, personality: str | None) -> str:
        palette = self.env_config.sheep_personality_colors
        if not palette:
            return "#f8fafc"
        if personality and personality in palette:
            return palette[personality]
        return palette.get("obedient", next(iter(palette.values())))

    def get_state_snapshot(self) -> EnvironmentSnapshot:
        """Return a frozen snapshot of the current environment state."""
        debug_payload = self._snapshot_debug_payload()
        return EnvironmentSnapshot(
            step=self._step_count,
            simulated_seconds=self._simulated_seconds,
            grid_width=self.env_config.width,
            grid_height=self.env_config.height,
            field_width=self.env_config.width,
            field_height=self.env_config.height,
            dogs=tuple(
                AgentSnapshot(
                    index=dog.index,
                    x=dog.position.x,
                    y=dog.position.y,
                    role=dog.current_role.value,
                    last_action=dog.last_action,
                )
                for dog in self._dogs
            ),
            sheep=tuple(
                AgentSnapshot(
                    index=sheep.index,
                    x=sheep.position.x,
                    y=sheep.position.y,
                    penned=sheep.penned,
                    personality=sheep.personality,
                    color=self._sheep_display_color(sheep.personality),
                )
                for sheep in self._sheep
            ),
            pen=self._pen,
            fence_cells=tuple(
                (cell.x, cell.y) for cell in sorted(self._fence_cells, key=lambda p: (p.y, p.x))
            ),
            penned_count=sum(1 for sheep in self._sheep if sheep.penned),
            average_distance_to_pen=self._average_distance_to_pen(),
            flock_spread=self._flock_spread(),
            no_progress_steps=self._no_progress_steps,
            terminated=self._terminated,
            timeout=self._timeout,
            stopped=self._stopped,
            success=self._success,
            status=self._status(),
            debug=debug_payload,
        )

    def action_mask_for_dog(
        self,
        dog_index: int,
        policy_mode: PolicyMode | None = None,
        weights: Any | None = None,
        reserved_positions: set[Point] | None = None,
    ) -> dict[Action, bool]:
        """Return a legal-action mask for the specified dog."""
        self.prepare_policy_step(weights=weights)
        dog = self._dogs[dog_index]
        current_score = self._action_score(
            dog_index,
            "wait",
            policy_mode=policy_mode,
            weights=weights,
        )
        move_scores = {
            action: self._action_score(
                dog_index,
                action,
                policy_mode=policy_mode,
                weights=weights,
                reserved_positions=reserved_positions,
            )
            for action in ACTION_ORDER
            if action != "wait"
        }
        best_move_score = max(move_scores.values()) if move_scores else -inf
        wait_allowed = current_score >= best_move_score - 0.05 or self._tactically_valid_wait(
            dog,
            policy_mode=policy_mode,
            weights=weights,
            reserved_positions=reserved_positions,
        )
        mask: dict[Action, bool] = {
            action: self.project_dog_action(dog_index, action) != dog.position
            for action in ACTION_ORDER
            if action != "wait"
        }
        # Neural / RL training modes must never have wait masked out: the
        # heuristic scoring threshold creates an artificial exploration ceiling
        # that prevents MaskablePPO from discovering hold-position strategies.
        # Heuristic, linear, and hill-climbing modes keep the existing logic.
        if policy_mode in {None, "neural_policy", "shepherd_neural_dogs"}:
            mask["wait"] = True
        else:
            mask["wait"] = wait_allowed
        return mask

    def ranked_actions_for_dog(
        self,
        dog_index: int,
        policy_mode: PolicyMode | None = None,
        weights: Any | None = None,
        reserved_positions: set[Point] | None = None,
    ) -> list[Action]:
        """Return all legal actions sorted best-first for the specified dog."""
        mask = self.action_mask_for_dog(
            dog_index,
            policy_mode=policy_mode,
            weights=weights,
            reserved_positions=reserved_positions,
        )
        candidates: list[Action] = [action for action, allowed in mask.items() if allowed]
        return sorted(
            candidates,
            key=lambda action: self._action_score(
                dog_index,
                cast(Action, action),
                policy_mode=policy_mode,
                weights=weights,
                reserved_positions=reserved_positions,
            ),
            reverse=True,
        )

    def project_dog_action(self, dog_index: int, action: Action) -> Point:
        """Return the destination point a dog would reach by taking the given action."""
        dog = self._dogs[dog_index]
        blocked_positions = {sheep.position for sheep in self._sheep if not sheep.penned}
        blocked_positions.update(other.position for other in self._dogs if other.index != dog_index)
        steps, _ = self._movement_steps(
            self._dog_action_speed(action),
            dog.movement_budget,
            allow_accumulation=action != "wait",
        )
        return self._advance_position(
            dog.position,
            action,
            steps,
            blocked_positions=blocked_positions,
        )

    def prepare_policy_step(self, weights: Any | None = None) -> None:
        """Compute role assignments for this step if they are not yet cached."""
        del weights
        if self._roles_prepared_step == self._step_count:
            return
        assignments, snapshot = self._team_strategy.assign_roles(self._dogs, self._sheep, self._pen)
        self._role_assignments = assignments
        self._strategy_snapshot = snapshot
        for dog in self._dogs:
            assignment = assignments.get(
                dog.index,
                RoleAssignment(dog.index, DogRole.REAR_PRESSURE, dog.position),
            )
            if dog.current_role != assignment.role:
                self._role_switches += 1
                dog.steps_in_role = 0
            else:
                dog.steps_in_role += 1
            dog.current_role = assignment.role
            self._role_distribution[assignment.role.value] = (
                self._role_distribution.get(assignment.role.value, 0) + 1
            )
            self._dog_role_occupancy.setdefault(
                str(dog.index),
                {role.value: 0 for role in DogRole},
            )[assignment.role.value] += 1
            if assignment.role == DogRole.LEFT_FLANKER:
                self._left_flank_occupancy_steps += 1
            if assignment.role == DogRole.RIGHT_FLANKER:
                self._right_flank_occupancy_steps += 1
        if any(assignment.role == DogRole.COLLECTOR for assignment in assignments.values()):
            self._collector_activations += 1
        if any(assignment.role == DogRole.BLOCKER for assignment in assignments.values()):
            self._blocker_activations += 1
        if snapshot.stray_sheep_index is not None:
            self._sheep_split_events += 1
        self._roles_prepared_step = self._step_count

    def current_role_assignments(self) -> dict[int, str]:
        """Return the role string keyed by dog index for the current step."""
        self.prepare_policy_step()
        return {
            dog_index: assignment.role.value
            for dog_index, assignment in self._role_assignments.items()
        }

    def build_observation_for_dog(self, dog_index: int) -> DogObservation:
        """Build the shared role-aware observation for one dog."""

        return self._observation_builder.build(self, dog_index)

    def score_action_for_dog(
        self,
        dog_index: int,
        action: Action,
        policy_mode: PolicyMode | None = None,
        weights: Any | None = None,
    ) -> float:
        """Return the heuristic score for a single dog-action combination."""
        return self._action_score(
            dog_index,
            action,
            policy_mode=policy_mode,
            weights=weights,
        )

    def run_policy(self, policy: Policy, seed: int, capture_replay: bool = False) -> EpisodeResult:
        """Run *policy* from *seed* until termination and return the episode result."""
        self.reset(seed)
        return self._run_policy_loop(policy, seed=seed, capture_replay=capture_replay)

    def run_policy_on_scenario(
        self, policy: Policy, scenario: object, capture_replay: bool = False
    ) -> EpisodeResult:
        """Run *policy* on a fixed :class:`SavedScenario` layout."""
        self.reset_from_scenario(scenario)
        assert isinstance(scenario, SavedScenario)
        return self._run_policy_loop(policy, seed=scenario.seed, capture_replay=capture_replay)

    def _run_policy_loop(
        self, policy: Policy, *, seed: int, capture_replay: bool
    ) -> EpisodeResult:
        while not self._terminated:
            actions = policy.select_actions(self)
            self.step(actions, capture_replay=capture_replay)
        final_snapshot = self.get_state_snapshot()
        return EpisodeResult(
            seed=seed,
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            final_snapshot=final_snapshot,
            stats=self._stats,
            replay=tuple(self._history),
        )

    def step(
        self, actions: Sequence[str], capture_replay: bool = False
    ) -> tuple[EnvironmentSnapshot, RewardBreakdown]:
        """Advance the simulation one team step and return the new snapshot and reward."""
        if self._terminated:
            raise RuntimeError("Cannot step a terminated episode.")
        if len(actions) != len(self._dogs):
            raise ValueError("Action count does not match dog count.")
        validated_actions: list[Action] = [self._validate_action(action) for action in actions]
        self.prepare_policy_step()

        previous_snapshot = self.get_state_snapshot()
        previous_flock_centroid = self._flock_center()
        previous_gate_distance = self._flock_gate_distance(previous_flock_centroid)
        previous_gate_corridor_distance = self._average_gate_corridor_distance()
        previous_gate_corridor_occupancy = self._gate_corridor_occupancy()
        self._apply_dog_actions(validated_actions)
        self._move_sheep()
        self._step_count += 1
        self._simulated_seconds += self.env_config.seconds_per_step
        self._roles_prepared_step = None

        current_snapshot = self.get_state_snapshot()
        newly_penned = current_snapshot.penned_count - previous_snapshot.penned_count
        progress_delta = (
            previous_snapshot.average_distance_to_pen - current_snapshot.average_distance_to_pen
        )
        progress_made = (
            newly_penned > 0 or progress_delta >= self.env_config.no_progress_distance_delta
        )
        if progress_made:
            self._no_progress_steps = max(0, self._no_progress_steps - 1)
        else:
            self._no_progress_steps += 1

        self._success = current_snapshot.penned_count == len(self._sheep)
        self._timeout = self._step_count >= self.env_config.max_steps and not self._success
        self._stopped = (
            not self._success
            and not self._timeout
            and self._no_progress_steps >= self.env_config.no_progress_window
        )
        self._terminated = self._success or self._timeout or self._stopped
        self._stop_reason = (
            "success"
            if self._success
            else "timeout"
            if self._timeout
            else "no-progress"
            if self._stopped
            else ""
        )

        current_flock_centroid = self._flock_center()
        current_gate_distance = self._flock_gate_distance(current_flock_centroid)
        current_gate_corridor_distance = self._average_gate_corridor_distance()
        current_gate_corridor_occupancy = self._gate_corridor_occupancy()
        gate_progress_delta = previous_gate_distance - current_gate_distance
        gate_corridor_delta = previous_gate_corridor_distance - current_gate_corridor_distance
        progressing_to_gate = (
            newly_penned > 0
            or gate_progress_delta >= self.env_config.gate_progress_epsilon
            or gate_corridor_delta >= self.env_config.gate_progress_epsilon / 2.0
            or current_gate_corridor_occupancy > previous_gate_corridor_occupancy + 1e-6
        )
        tactically_valid_hold = bool(
            current_gate_distance <= self.env_config.gate_hold_safe_distance
            and current_gate_corridor_occupancy > 0.0
        )
        wrong_hold_active = False
        if (
            self._is_controlled_state(current_flock_centroid)
            and newly_penned == 0
            and not progressing_to_gate
            and not tactically_valid_hold
        ):
            self._controlled_stall_steps += 1
            self._controlled_stall_streak += 1
            wall_pinned = self._wall_pinned_sheep_ratio() >= 0.34
            if (
                self._controlled_stall_streak >= self.env_config.stalled_control_activation_steps
            ) and (
                wall_pinned
                or current_gate_distance > self.env_config.gate_hold_safe_distance
                or current_gate_corridor_occupancy == 0.0
            ):
                wrong_hold_active = True
        else:
            self._controlled_stall_streak = 0
        self._cumulative_gate_progress += gate_progress_delta
        self._gate_corridor_occupancy_peak = max(
            self._gate_corridor_occupancy_peak,
            current_gate_corridor_occupancy,
        )
        if current_gate_corridor_occupancy > 0.0 and newly_penned == 0 and not progressing_to_gate:
            self._gate_corridor_failure_steps += 1
        unpenned_sheep_positions = tuple(
            (float(sheep.position.x), float(sheep.position.y))
            for sheep in self._sheep
            if not sheep.penned
        )
        breakdown = self.reward_computer.compute(
            RewardInputs(
                previous_average_distance=previous_snapshot.average_distance_to_pen,
                current_average_distance=current_snapshot.average_distance_to_pen,
                previous_flock_spread=previous_snapshot.flock_spread,
                current_flock_spread=current_snapshot.flock_spread,
                newly_penned=newly_penned,
                no_progress_step=not progress_made,
                touched_wall=self._touched_wall_this_step(validated_actions),
                waited_without_reason=self._waited_without_reason(validated_actions),
                sprint_count=sum(
                    1 for action in validated_actions if self._is_sprint_action(action)
                ),
                previous_gate_distance=previous_gate_distance,
                current_gate_distance=current_gate_distance,
                previous_gate_corridor_distance=previous_gate_corridor_distance,
                current_gate_corridor_distance=current_gate_corridor_distance,
                previous_gate_corridor_occupancy=previous_gate_corridor_occupancy,
                current_gate_corridor_occupancy=current_gate_corridor_occupancy,
                controlled_stall_steps=self._controlled_stall_streak,
                wrong_hold_active=wrong_hold_active,
                tactically_valid_hold=tactically_valid_hold,
                terminated=self._terminated,
                timeout=self._timeout or self._stopped,
                success=self._success,
                dog_positions=tuple(
                    (float(dog.position.x), float(dog.position.y)) for dog in self._dogs
                ),
                sheep_positions=unpenned_sheep_positions,
                flock_centroid=(
                    (float(current_flock_centroid.x), float(current_flock_centroid.y))
                    if current_flock_centroid is not None
                    else None
                ),
                previous_flock_centroid=(
                    (float(previous_flock_centroid.x), float(previous_flock_centroid.y))
                    if previous_flock_centroid is not None
                    else None
                ),
                target_position=(float(self._pen.center.x), float(self._pen.center.y)),
                previous_farthest_distance=self._previous_farthest_distance,
                current_farthest_distance=self._farthest_distance_to_pen(),
            )
        )
        self._reward_total += breakdown.total
        self._previous_average_distance = current_snapshot.average_distance_to_pen
        self._previous_flock_spread = current_snapshot.flock_spread
        self._previous_farthest_distance = self._farthest_distance_to_pen()

        final_snapshot = self.get_state_snapshot()

        self._stats = EpisodeStats(
            steps=self._step_count,
            simulated_seconds=self._simulated_seconds,
            sheep_penned=final_snapshot.penned_count,
            timeout=self._timeout,
            terminated=self._terminated,
            success=self._success,
            stopped=self._stopped,
            stop_reason=self._stop_reason,
            reward_total=self._reward_total,
            no_progress_steps=self._no_progress_steps,
            final_avg_distance_to_pen=final_snapshot.average_distance_to_pen,
            final_flock_spread=final_snapshot.flock_spread,
            role_distribution=dict(self._role_distribution),
            dog_role_occupancy={
                dog_index: dict(occupancy)
                for dog_index, occupancy in self._dog_role_occupancy.items()
            },
            role_switches=self._role_switches,
            collector_activations=self._collector_activations,
            blocker_activations=self._blocker_activations,
            sheep_split_events=self._sheep_split_events,
            controlled_stall_steps=self._controlled_stall_steps,
            left_flank_occupancy_steps=self._left_flank_occupancy_steps,
            right_flank_occupancy_steps=self._right_flank_occupancy_steps,
            gate_corridor_occupancy_peak=self._gate_corridor_occupancy_peak,
            gate_corridor_failure_steps=self._gate_corridor_failure_steps,
            final_reward_breakdown=breakdown.to_dict(),
        )

        if capture_replay:
            self._history.append(
                StepRecord(
                    step=self._step_count,
                    actions=tuple(validated_actions),
                    snapshot=final_snapshot,
                    reward=breakdown,
                )
            )

        return final_snapshot, breakdown

    def export_replay(self, path: str | Path) -> Path:
        """Write the episode replay to *path* as a JSON file and return it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self._history], handle, indent=2)
        return target

    def _initial_dogs(self) -> list[DogState]:
        dogs: list[DogState] = []
        base_y = self.env_config.height // 2 + 2
        occupied: set[Point] = set()
        for index in range(self.env_config.dogs):
            position = self._sample_unique_position(
                preferred_x=2 + index * 2,
                preferred_y=base_y,
                occupied=occupied,
                jitter_x=0,
                jitter_y=1,
            )
            occupied.add(position)
            dogs.append(
                DogState(
                    index=index,
                    position=position,
                )
            )
        return dogs

    def _random_sheep_spawn_center(
        self,
        occupied: set[Point],
        avoid_points: list[Point] | None = None,
        min_distance: float = 15.0,
    ) -> Point:
        current_min_distance = min_distance
        for _ in range(3):
            for _ in range(50):
                x = self._rng.randint(4, self.env_config.width - 5)
                y = self._rng.randint(4, self.env_config.height - 5)
                pt = Point(x, y)
                if self._pen.contains(pt) or pt in self._fence_cells:
                    continue
                if (
                    x >= self.env_config.width - self.env_config.pen_width - 2
                    and y <= self.env_config.pen_height + 2
                ):
                    continue
                if y >= self.env_config.height - 4:
                    continue
                if avoid_points:
                    if any(pt.distance_to(ap) < current_min_distance for ap in avoid_points):
                        continue
                return pt
            current_min_distance *= 0.5

        for _ in range(100):
            x = self._rng.randint(4, self.env_config.width - 5)
            y = self._rng.randint(4, self.env_config.height - 5)
            pt = Point(x, y)
            if not self._pen.contains(pt) and pt not in self._fence_cells:
                return pt
        return Point(self.env_config.width // 2, self.env_config.height // 2)

    def _initial_sheep(self) -> list[SheepState]:
        sheep: list[SheepState] = []
        occupied = {dog.position for dog in self._dogs}
        assign_personalities = self.env_config.sheep_personality_strength > 0.0
        personality_rng = Random(
            self._seed
            + self.env_config.seed_offset
            + self.env_config.sheep_personality_seed_offset
            + 9973
        )

        stage = self.env_config.curriculum_stage

        if stage == 6:
            # One group of 5 and 1 alone randomly placed
            center_5 = self._random_sheep_spawn_center(occupied)
            center_1 = self._random_sheep_spawn_center(occupied, avoid_points=[center_5], min_distance=15.0)

            for index in range(5):
                position = self._sample_unique_position(
                    preferred_x=center_5.x,
                    preferred_y=center_5.y,
                    occupied=occupied,
                    jitter_x=2,
                    jitter_y=2,
                )
                occupied.add(position)
                personality = (
                    personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
                )
                sheep.append(
                    SheepState(
                        index=index,
                        position=position,
                        personality=personality,
                    )
                )

            position = self._sample_unique_position(
                preferred_x=center_1.x,
                preferred_y=center_1.y,
                occupied=occupied,
                jitter_x=0,
                jitter_y=0,
            )
            occupied.add(position)
            personality = (
                personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
            )
            sheep.append(
                SheepState(
                    index=5,
                    position=position,
                    personality=personality,
                )
            )

        elif stage == 7:
            # One group of 3 and 3 randomly placed alone
            center_3 = self._random_sheep_spawn_center(occupied)
            avoid_list = [center_3]
            other_centers = []
            for _ in range(3):
                c = self._random_sheep_spawn_center(occupied, avoid_points=avoid_list, min_distance=15.0)
                other_centers.append(c)
                avoid_list.append(c)

            for index in range(3):
                position = self._sample_unique_position(
                    preferred_x=center_3.x,
                    preferred_y=center_3.y,
                    occupied=occupied,
                    jitter_x=2,
                    jitter_y=2,
                )
                occupied.add(position)
                personality = (
                    personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
                )
                sheep.append(
                    SheepState(
                        index=index,
                        position=position,
                        personality=personality,
                    )
                )

            for i, center in enumerate(other_centers):
                position = self._sample_unique_position(
                    preferred_x=center.x,
                    preferred_y=center.y,
                    occupied=occupied,
                    jitter_x=0,
                    jitter_y=0,
                )
                occupied.add(position)
                personality = (
                    personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
                )
                sheep.append(
                    SheepState(
                        index=3 + i,
                        position=position,
                        personality=personality,
                    )
                )

        elif stage == 8:
            # All sheep randomly placed
            centers = []
            for index in range(self.env_config.sheep):
                c = self._random_sheep_spawn_center(occupied, avoid_points=centers, min_distance=10.0)
                centers.append(c)
                position = self._sample_unique_position(
                    preferred_x=c.x,
                    preferred_y=c.y,
                    occupied=occupied,
                    jitter_x=0,
                    jitter_y=0,
                )
                occupied.add(position)
                personality = (
                    personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
                )
                sheep.append(
                    SheepState(
                        index=index,
                        position=position,
                        personality=personality,
                    )
                )

        else:
            # Stage 0 to 5 default herding layout
            base_x = self.env_config.width // 3
            base_y = self.env_config.height // 2
            for index in range(self.env_config.sheep):
                position = self._sample_unique_position(
                    preferred_x=base_x,
                    preferred_y=base_y,
                    occupied=occupied,
                    jitter_x=2,
                    jitter_y=2,
                )
                occupied.add(position)
                personality = (
                    personality_rng.choice(SHEEP_PERSONALITIES) if assign_personalities else "obedient"
                )
                sheep.append(
                    SheepState(
                        index=index,
                        position=position,
                        personality=personality,
                    )
                )
        return sheep

    def _sample_unique_position(
        self,
        *,
        preferred_x: int,
        preferred_y: int,
        occupied: set[Point],
        jitter_x: int,
        jitter_y: int,
    ) -> Point:
        for _ in range(64):
            candidate = Point(
                preferred_x + self._rng.randint(-jitter_x, jitter_x),
                preferred_y + self._rng.randint(-jitter_y, jitter_y),
            ).clamp(self.env_config.width, self.env_config.height)
            if candidate not in occupied and candidate not in self._fence_cells:
                return candidate

        for y in range(self.env_config.height):
            for x in range(self.env_config.width):
                candidate = Point(x, y)
                if candidate in occupied or candidate in self._fence_cells:
                    continue
                if self._pen.contains(candidate):
                    continue
                return candidate
        raise RuntimeError("No free spawn position available.")

    def _sample_open_position(
        self,
        center: Point,
        radius_x: int,
        radius_y: int,
        occupied: set[Point],
    ) -> Point:
        return self._sample_unique_position(
            preferred_x=center.x,
            preferred_y=center.y,
            occupied=occupied,
            jitter_x=radius_x,
            jitter_y=radius_y,
        )

    def _target_position(self, position: Point, action: Action) -> Point:
        dx, dy = ACTION_DELTAS[action]
        candidate = Point(position.x + dx, position.y + dy).clamp(
            self.env_config.width, self.env_config.height
        )
        if self._pen.contains(candidate) and not self._pen.contains(position):
            return position
        if candidate in self._fence_cells and not self._pen.contains(position):
            return position
        return candidate

    def _apply_dog_actions(self, actions: list[Action]) -> None:
        sheep_positions = {sheep.position for sheep in self._sheep if not sheep.penned}
        occupied_positions = {dog.position for dog in self._dogs}
        claimed_positions: set[Point] = set()
        next_positions: list[Point] = []
        next_budgets: list[float] = []
        for dog, action in zip(self._dogs, actions, strict=True):
            blocked_positions = (
                sheep_positions
                | claimed_positions
                | {position for position in occupied_positions if position != dog.position}
            )
            steps, remaining_budget = self._movement_steps(
                self._dog_action_speed(action),
                dog.movement_budget,
                allow_accumulation=action != "wait",
            )
            candidate = self._advance_position(
                dog.position,
                action,
                steps,
                blocked_positions=blocked_positions,
            )
            # When a dog is stopped because a sheep occupies the immediate
            # target cell, forfeit the accumulated fractional budget so it
            # cannot build up sustained pressure against a cornered sheep.
            if (
                candidate == dog.position
                and self._target_position(dog.position, action) in sheep_positions
            ):
                remaining_budget = 0.0
            claimed_positions.add(candidate)
            next_positions.append(candidate)
            next_budgets.append(remaining_budget)

        for dog, action, candidate, remaining_budget in zip(
            self._dogs,
            actions,
            next_positions,
            next_budgets,
            strict=True,
        ):
            moved = candidate != dog.position
            dog.position = candidate
            dog.last_action = action
            dog.movement_budget = remaining_budget
            dog.blocked_steps = 0 if moved or action == "wait" else dog.blocked_steps + 1
            self._record_position_history(dog.recent_positions, candidate)

    def _move_sheep(self) -> None:
        dog_positions = [dog.position for dog in self._dogs]
        occupied_sheep_positions = {sheep.position for sheep in self._sheep if not sheep.penned}
        flock_center = self._flock_center()
        claimed_positions: set[Point] = set()
        next_positions: list[Point] = []
        next_budgets: list[float] = []
        for sheep in self._sheep:
            if sheep.penned or self._pen.contains(sheep.position):
                sheep.penned = True
                sheep.panic_steps = 0
                next_positions.append(sheep.position)
                next_budgets.append(0.0)
                claimed_positions.add(sheep.position)
                continue
            move = sheep.position
            blocked_positions = claimed_positions | {
                position for position in occupied_sheep_positions if position != sheep.position
            }
            steps, remaining_budget = self._movement_steps(
                self.env_config.sheep_speed,
                sheep.movement_budget,
            )
            for _ in range(steps):
                next_move = self._sheep_step(
                    move,
                    dog_positions,
                    flock_center,
                    sheep.panic_steps,
                    blocked_positions=blocked_positions,
                    sheep=sheep,
                )
                if sheep.panic_steps > 0:
                    sheep.panic_steps = max(0, sheep.panic_steps - 1)
                if next_move == move:
                    break
                move = next_move
            if self._nearest_dog_distance(move, dog_positions) <= self.env_config.sheep_vision:
                sheep.panic_steps = max(sheep.panic_steps, 2)
            next_positions.append(move)
            next_budgets.append(remaining_budget)
            claimed_positions.add(move)

        for sheep, position, remaining_budget in zip(
            self._sheep,
            next_positions,
            next_budgets,
            strict=True,
        ):
            moved = position != sheep.position
            sheep.position = position
            sheep.penned = self._pen.contains(position)
            sheep.movement_budget = 0.0 if sheep.penned else remaining_budget
            sheep.blocked_steps = 0 if moved or sheep.penned else sheep.blocked_steps + 1
            self._record_position_history(sheep.recent_positions, position)

    def _validate_action(self, action: str) -> Action:
        if action not in ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action}")
        return cast(Action, action)

    def _sheep_step(
        self,
        position: Point,
        dog_positions: list[Point],
        flock_center: Point | None,
        panic_steps: int,
        blocked_positions: set[Point] | None = None,
        sheep: SheepState | None = None,
    ) -> Point:
        personality = sheep.personality if sheep is not None else "obedient"
        strength = max(0.0, float(self.env_config.sheep_personality_strength))
        
        # Check if sheep should idle (no dog pressure, not panicked, near flock, away from wall)
        nearest_dog_distance = self._nearest_dog_distance(position, dog_positions)
        should_idle = (
            panic_steps <= 0
            and nearest_dog_distance > self.env_config.sheep_vision
            and flock_center is not None
            and position.distance_to(flock_center) <= self.env_config.flock_radius
            and self._wall_margin(position) > 1.0
        )
        if should_idle and self._rng.random() < 0.7:
            # 70% chance to idle when conditions are met
            return position
        
        vector_x = 0.0
        vector_y = 0.0
        for dog_position in dog_positions:
            distance = max(1.0, position.distance_to(dog_position))
            nearest_dog_distance = min(nearest_dog_distance, distance)
            if distance <= self.env_config.sheep_vision:
                weight = (self.env_config.sheep_vision - distance + 1.0) / distance
                # Bold sheep ignore distant dogs more readily, so the dog must
                # close the gap (or bark) to apply meaningful pressure.
                if personality == "bold" and strength > 0.0 and distance > 3.0:
                    weight *= max(0.0, 1.0 - 0.4 * strength)
                vector_x += (position.x - dog_position.x) * weight
                vector_y += (position.y - dog_position.y) * weight
        if panic_steps <= 0 and flock_center is not None:
            # Flock cohesion: only apply when dogs are nearby OR sheep is far from flock
            # This prevents constant self-driving when no dogs are present
            dogs_nearby = nearest_dog_distance <= self.env_config.sheep_vision
            far_from_flock = position.distance_to(flock_center) > self.env_config.flock_radius
            if dogs_nearby or far_from_flock:
                vector_x += (flock_center.x - position.x) * 0.35
                vector_y += (flock_center.y - position.y) * 0.35
        elif personality == "escapist" and strength > 0.0 and flock_center is not None:
            # When scared, an escapist sheep bolts away from the flock instead
            # of cohering with it.
            vector_x += (position.x - flock_center.x) * 0.35 * strength
            vector_y += (position.y - flock_center.y) * 0.35 * strength
        if nearest_dog_distance <= self.env_config.sheep_vision:
            vector_x *= 1.5
            vector_y *= 1.5
        vector_x += self._wall_avoidance(position, axis="x")
        vector_y += self._wall_avoidance(position, axis="y")
        if strength > 0.0 and personality == "pen_shy":
            pen_center = self._pen.center
            dx = pen_center.x - position.x
            dy = pen_center.y - position.y
            norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
            vector_x -= (dx / norm) * strength
            vector_y -= (dy / norm) * strength
        if strength > 0.0 and personality == "pen_fearful":
            pen_center = self._pen.center
            dx = pen_center.x - position.x
            dy = pen_center.y - position.y
            dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
            # Scale repulsion up to 4× at the pen entrance (dist ≈ 1) and fade
            # smoothly to 1× at pen_width distance away.
            proximity_scale = max(1.0, self.env_config.pen_width / dist * 4.0)
            magnitude = strength * proximity_scale
            vector_x -= (dx / dist) * magnitude
            vector_y -= (dy / dist) * magnitude
        blocked = blocked_positions or set()
        candidate_actions = ("up", "down", "left", "right")
        primary_action: Action | None = None
        if vector_x != 0 or vector_y != 0:
            if abs(vector_x) >= abs(vector_y):
                primary_action = "right" if vector_x > 0 else "left"
            else:
                primary_action = "down" if vector_y > 0 else "up"
        primary_blocked = False
        if primary_action is not None:
            primary_candidate = Point(
                position.x + ACTION_DELTAS[primary_action][0],
                position.y + ACTION_DELTAS[primary_action][1],
            ).clamp(self.env_config.width, self.env_config.height)
            primary_blocked = self._sheep_move_blocked(
                position,
                primary_candidate,
                dog_positions,
                blocked,
            )
        edge_escape_actions = self._edge_escape_actions(position)

        best_score = -inf
        best_moves: list[Point] = [position]
        for action in candidate_actions:
            candidate = Point(
                position.x + ACTION_DELTAS[action][0],
                position.y + ACTION_DELTAS[action][1],
            ).clamp(self.env_config.width, self.env_config.height)
            if self._sheep_move_blocked(position, candidate, dog_positions, blocked):
                continue
            score = self._score_sheep_candidate(
                position,
                candidate,
                dog_positions,
                vector_x,
                vector_y,
            )
            if primary_blocked and action in edge_escape_actions:
                # When the natural flee direction is blocked, sheep should
                # aggressively try lateral openings rather than stand still.
                score += 1.6
            if action in edge_escape_actions:
                score += 0.4
            if primary_action is not None and action == primary_action:
                score += 0.2
            if score > best_score + 1e-9:
                best_score = score
                best_moves = [candidate]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(candidate)

        if best_score == -inf:
            return position
        return self._rng.choice(best_moves)

    def _sheep_move_blocked(
        self,
        position: Point,
        candidate: Point,
        dog_positions: list[Point],
        blocked_positions: set[Point],
    ) -> bool:
        if candidate == position:
            return True
        if candidate in self._fence_cells and not self._pen.contains(position):
            return True
        return candidate in dog_positions or candidate in blocked_positions

    def _score_sheep_candidate(
        self,
        position: Point,
        candidate: Point,
        dog_positions: list[Point],
        vector_x: float,
        vector_y: float,
    ) -> float:
        step_x = candidate.x - position.x
        step_y = candidate.y - position.y
        dog_clearance = self._nearest_dog_distance(candidate, dog_positions)
        return (
            step_x * vector_x
            + step_y * vector_y
            + min(dog_clearance, self.env_config.sheep_vision * 2) * 0.35
            + self._wall_margin(candidate) * 0.1
            + self._rng.random() * 1e-6
        )

    def _edge_escape_actions(self, position: Point) -> tuple[Action, ...]:
        actions: list[Action] = []
        # A sheep cornered against (or one cell from) a wall should consider
        # lateral escape along the wall as a real option, not just when
        # exactly on the boundary.
        max_x = self.env_config.width - 1
        max_y = self.env_config.height - 1
        if position.x <= 1 or position.x >= max_x - 1:
            actions.extend(("up", "down"))
        if position.y <= 1 or position.y >= max_y - 1:
            actions.extend(("left", "right"))
        return tuple(dict.fromkeys(actions))

    def _nearest_dog_distance(self, position: Point, dog_positions: list[Point]) -> float:
        if not dog_positions:
            return inf
        return min(position.distance_to(dog_position) for dog_position in dog_positions)

    def _action_score(
        self,
        dog_index: int,
        action: Action,
        policy_mode: PolicyMode | None = None,
        weights: Any | None = None,
        reserved_positions: set[Point] | None = None,
    ) -> float:
        self.prepare_policy_step(weights=weights)
        dog = self._dogs[dog_index]
        active_policy_mode = self._resolve_policy_mode(policy_mode, weights)
        candidate = self.project_dog_action(dog_index, action)
        candidate_debug = self._action_context_debug(active_policy_mode, dog_index, candidate)
        current_debug = self._action_context_debug(active_policy_mode, dog_index, dog.position)
        flock_center = self._flock_center() or dog.position
        pen_center = self._pen.center
        assignment = self._role_assignments.get(
            dog_index,
            RoleAssignment(dog.index, DogRole.REAR_PRESSURE, dog.position),
        )
        teammate_spacing = self._teammate_spacing(candidate, dog_index)
        distance_to_sheep = candidate_debug["distance_to_focus_sheep"]
        preferred_sheep_distance = 2.5 if candidate_debug["focus_mode"] == "collect" else 3.5
        sheep_term = 0.0
        if distance_to_sheep is not None and (
            candidate_debug["pressure_side_alignment"] >= 0.25
            or candidate_debug["focus_mode"] == "collect"
        ):
            sheep_term = -abs(distance_to_sheep - preferred_sheep_distance)
        inside_distance = max(
            0.0,
            candidate_debug["flock_buffer_radius"] + 1.0 - candidate_debug["distance_to_flock"],
        )
        inside_flock_penalty = -inside_distance
        pen_side_penalty = -1.0 if candidate_debug["between_flock_and_pen"] else 0.0
        formation_hold_bonus = 0.75 if candidate_debug["holding_pressure_position"] else 0.0
        wall_margin = self._wall_margin(candidate)
        if weights is None:
            weights = self._default_action_weights(active_policy_mode)
        assert weights is not None
        wait_bias = 0.0
        if action == "wait":
            if candidate_debug["holding_pressure_position"]:
                wait_bias = 0.5
            elif (
                candidate_debug["pressure_side_alignment"] >= 0.8
                and not candidate_debug["between_flock_and_pen"]
                and not candidate_debug["inside_or_too_close_to_flock"]
            ):
                wait_bias = -0.1
            else:
                wait_bias = weights.wait_bias
        role_term = self._role_score(
            assignment,
            candidate,
            flock_center,
            pen_center,
            weights,
            active_policy_mode,
            candidate_debug,
            teammate_spacing,
        )
        stack_penalty = 0.0
        if reserved_positions and action != "wait" and candidate in reserved_positions:
            stack_penalty = weights.anti_stack_penalty
        oscillation_penalty = self._oscillation_penalty(dog, candidate, action, weights)
        score = (
            weights.nearest_sheep * sheep_term
            + weights.flock_center * inside_flock_penalty
            + weights.pen_pressure * pen_side_penalty
            + weights.behind_flock
            * (candidate_debug["pressure_side_alignment"] + formation_hold_bonus)
            + weights.team_formation * (-candidate_debug["distance_to_pressure_target"])
            + weights.dog_spacing * (-teammate_spacing)
            + weights.wall_margin * wall_margin
            + role_term
            + wait_bias
            - stack_penalty
            - oscillation_penalty
        )
        score += self._deadlock_action_adjustment(
            dog_index,
            dog.position,
            candidate,
            action,
            active_policy_mode,
            candidate_debug,
        )
        score += self._loop_penalty(dog, candidate)
        score -= self._sprint_action_penalty(dog, action, candidate, assignment, candidate_debug)
        if self._is_reverse_action(dog.last_action, action) and not self._reverse_is_clearly_better(
            current_debug,
            candidate_debug,
        ):
            score -= 1.2
        return score

    def _role_score(
        self,
        assignment: RoleAssignment,
        candidate: Point,
        flock_center: Point,
        pen_center: Point,
        weights: Any,
        policy_mode: PolicyMode,
        candidate_debug: dict[str, Any],
        teammate_spacing: float,
    ) -> float:
        if policy_mode == "instinct_only" and not self._instinct_target_awareness_enabled(
            policy_mode
        ):
            return 0.0

        def value(name: str) -> float:
            return float(getattr(weights, name, 0.0))

        target_distance = candidate.distance_to(assignment.target)
        if assignment.role == DogRole.REAR_PRESSURE:
            behind_term = candidate_debug["pressure_side_alignment"]
            overpressure_term = -1.0 if candidate_debug["inside_or_too_close_to_flock"] else 0.0
            spacing_term = -teammate_spacing
            return (
                value("rear_drive") * (-target_distance)
                + value("rear_behind_flock") * behind_term
                + value("rear_drive_to_pen")
                * self._alignment_score(candidate, flock_center, pen_center)
                + value("rear_avoid_overpressure") * overpressure_term
                + value("rear_spacing") * spacing_term
            )
        if assignment.role in {DogRole.LEFT_FLANKER, DogRole.RIGHT_FLANKER}:
            return (
                value("flank_control") * (-target_distance)
                + value("flank_side_control")
                * self._flank_score(candidate, flock_center, pen_center)
                + value("flank_handedness")
                * self._flank_handedness_score(candidate, flock_center, pen_center, assignment.role)
                + value("flank_escape_blocking")
                * self._alignment_score(candidate, assignment.target, pen_center)
                + value("flank_spacing") * (-teammate_spacing)
                + value("flank_wall_margin") * self._wall_margin(candidate)
            )
        if assignment.role == DogRole.COLLECTOR:
            focus = 0.0
            return_to_flock = -candidate.distance_to(flock_center)
            avoid_scatter = max(
                -1.0,
                1.0
                - max(
                    0.0,
                    candidate_debug["distance_to_flock"] - candidate_debug["flock_buffer_radius"],
                ),
            )
            rejoin_angle = 0.0
            if assignment.target_sheep_index is not None:
                stray = next(
                    (
                        sheep
                        for sheep in self._sheep
                        if sheep.index == assignment.target_sheep_index and not sheep.penned
                    ),
                    None,
                )
                if stray is not None:
                    focus = -candidate.distance_to(stray.position)
                    rejoin_angle = self._alignment_score(candidate, stray.position, flock_center)
            return (
                value("collector_focus") * (-target_distance)
                + value("collector_stray_focus") * focus
                + value("collector_return_to_flock") * return_to_flock
                + value("collector_avoid_scatter") * avoid_scatter
                + value("collector_rejoin_angle") * rejoin_angle
            )
        guard = self._alignment_score(candidate, assignment.target, pen_center)
        gate_position = self._gate_position().clamp(self.env_config.width, self.env_config.height)
        hold_term = -target_distance if candidate_debug["holding_pressure_position"] else 0.0
        return (
            value("blocker_cover") * (-target_distance)
            + value("blocker_escape_route_cover") * guard
            + value("blocker_gate_control") * (-candidate.distance_to(gate_position))
            + value("blocker_funnel_lane") * self._blocker_funnel_score(candidate, gate_position)
            + value("blocker_hold_position") * hold_term
            + value("blocker_spacing") * (-teammate_spacing)
        )

    def _gate_position(self) -> Point:
        if self._pen.opening == "left":
            return Point(self._pen.origin.x - 1, self._pen.center.y)
        if self._pen.opening == "right":
            return Point(self._pen.origin.x + self._pen.width, self._pen.center.y)
        if self._pen.opening == "top":
            return Point(self._pen.center.x, self._pen.origin.y - 1)
        return Point(self._pen.center.x, self._pen.origin.y + self._pen.height)

    def _formation_target(
        self,
        dog_index: int,
        flock_center: Point,
        pen_center: Point,
    ) -> Point:
        to_pen_x = self._sign(pen_center.x - flock_center.x)
        to_pen_y = self._sign(pen_center.y - flock_center.y)
        behind_distance = max(3, min(6, round(self._flock_spread()) + 3))
        lateral_spacing = 3
        base_x = flock_center.x - to_pen_x * behind_distance
        base_y = flock_center.y - to_pen_y * behind_distance
        lateral_x = -to_pen_y
        lateral_y = to_pen_x
        offset = self._pressure_role_offset(dog_index) * lateral_spacing
        return Point(base_x + lateral_x * offset, base_y + lateral_y * offset).clamp(
            self.env_config.width,
            self.env_config.height,
        )

    def _instinct_role_direction(self, dog_index: int) -> tuple[int, int]:
        directions = ((-1, 0), (0, -1), (0, 1), (1, 0))
        return directions[dog_index % len(directions)]

    def _instinct_target(
        self,
        dog_index: int,
        flock_center: Point,
        position: Point,
        focus_sheep: SheepState | None,
        focus_mode: str,
    ) -> Point:
        if focus_mode == "collect" and focus_sheep is not None:
            recovery_distance = max(2, min(5, round(self._flock_spread()) + 2))
            away_x = focus_sheep.position.x - flock_center.x
            away_y = focus_sheep.position.y - flock_center.y
            if away_x == 0 and away_y == 0:
                away_x = position.x - flock_center.x
                away_y = position.y - flock_center.y
            return Point(
                focus_sheep.position.x + self._sign(away_x) * recovery_distance,
                focus_sheep.position.y + self._sign(away_y) * recovery_distance,
            ).clamp(self.env_config.width, self.env_config.height)

        radial_x = position.x - flock_center.x
        radial_y = position.y - flock_center.y
        if radial_x == 0 and radial_y == 0:
            direction_x, direction_y = self._instinct_role_direction(dog_index)
        else:
            direction_x = self._sign(radial_x)
            direction_y = self._sign(radial_y)
        orbit_distance = max(3, round(self._flock_buffer_radius()) + 1)
        tangent_x = -direction_y
        tangent_y = direction_x
        offset = self._pressure_role_offset(dog_index) * 2
        return Point(
            flock_center.x + direction_x * orbit_distance + tangent_x * offset,
            flock_center.y + direction_y * orbit_distance + tangent_y * offset,
        ).clamp(self.env_config.width, self.env_config.height)

    def _pressure_role_offset(self, dog_index: int) -> int:
        if self.dog_count <= 1:
            return 0
        if self.dog_count == 2:
            return -1 if dog_index == 0 else 1
        return (-1, 0, 1)[dog_index % 3]

    def _teammate_spacing(self, candidate: Point, dog_index: int) -> float:
        teammate_positions = [dog.position for dog in self._dogs if dog.index != dog_index]
        if not teammate_positions:
            return 0.0
        nearest_teammate_distance = min(
            candidate.distance_to(position) for position in teammate_positions
        )
        target_spacing = 4.0 if len(teammate_positions) > 1 else 3.0
        return abs(nearest_teammate_distance - target_spacing)

    def _tactically_valid_wait(
        self,
        dog: DogState,
        policy_mode: PolicyMode | None = None,
        weights: Any | None = None,
        reserved_positions: set[Point] | None = None,
    ) -> bool:
        if not self._sheep:
            return False
        current_score = self._action_score(
            dog.index,
            "wait",
            policy_mode=policy_mode,
            weights=weights,
            reserved_positions=reserved_positions,
        )
        best_move_score = max(
            self._action_score(
                dog.index,
                action,
                policy_mode=policy_mode,
                weights=weights,
                reserved_positions=reserved_positions,
            )
            for action in ACTION_ORDER
            if action != "wait"
        )
        return current_score >= best_move_score - 0.05

    def _waited_without_reason(self, actions: list[Action]) -> bool:
        return any(
            action == "wait" and not self._tactically_valid_wait(self._dogs[index])
            for index, action in enumerate(actions)
        )

    def _touched_wall_this_step(self, actions: list[Action]) -> bool:
        for dog, action in zip(self._dogs, actions, strict=True):
            candidate = self.project_dog_action(dog.index, action)
            if candidate == dog.position and action != "wait":
                return True
        return False

    def _flock_center(self) -> Point | None:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return None
        x = round(fmean(position.x for position in unpenned))
        y = round(fmean(position.y for position in unpenned))
        return Point(x, y)

    def _flock_spread(self) -> float:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if len(unpenned) <= 1:
            return 0.0
        centroid = self._flock_center()
        assert centroid is not None
        return fmean(position.distance_to(centroid) for position in unpenned)

    def _average_distance_to_pen(self) -> float:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return 0.0
        pen_center = self._pen.center
        return fmean(position.distance_to(pen_center) for position in unpenned)

    def _farthest_distance_to_pen(self) -> float:
        """Return the max distance any unpenned sheep is from the pen centre."""
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return 0.0
        pen_center = self._pen.center
        return max(position.distance_to(pen_center) for position in unpenned)

    def _nearest_unpenned_sheep(self, position: Point) -> SheepState | None:
        unpenned = [sheep for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return None
        return min(unpenned, key=lambda sheep: sheep.position.distance_to(position))

    def _alignment_score(self, candidate: Point, flock_center: Point, pen_center: Point) -> float:
        vector_to_pen_x = pen_center.x - flock_center.x
        vector_to_pen_y = pen_center.y - flock_center.y
        vector_candidate_x = candidate.x - flock_center.x
        vector_candidate_y = candidate.y - flock_center.y
        numerator = vector_candidate_x * vector_to_pen_x + vector_candidate_y * vector_to_pen_y
        pen_norm = (vector_to_pen_x**2 + vector_to_pen_y**2) ** 0.5
        candidate_norm = (vector_candidate_x**2 + vector_candidate_y**2) ** 0.5
        denominator = max(1.0, pen_norm * candidate_norm)
        return -numerator / denominator

    def _radius_alignment(self, distance_to_flock: float, desired_radius: float) -> float:
        if desired_radius <= 0:
            return 0.0
        return max(-1.0, 1.0 - abs(distance_to_flock - desired_radius) / desired_radius)

    def _resolve_policy_mode(
        self,
        policy_mode: PolicyMode | None,
        weights: Any | None,
    ) -> PolicyMode:
        if policy_mode is not None:
            return policy_mode
        if weights is not None:
            return "trained_policy"
        return self.config.policy.policy_mode

    def _instinct_target_awareness_enabled(self, policy_mode: PolicyMode) -> bool:
        if policy_mode in {"heuristic_expert", "trained_policy"}:
            return True
        if policy_mode != "instinct_only":
            return False
        return (
            self.config.policy.allow_instinct_target_awareness
            or self.config.policy.handler_target_enabled
        )

    def _default_action_weights(self, policy_mode: PolicyMode) -> Any:
        if self._instinct_target_awareness_enabled(policy_mode):
            return type(
                "Weights",
                (),
                {
                    "nearest_sheep": 0.45,
                    "flock_center": 1.8,
                    "pen_pressure": 2.2,
                    "behind_flock": 3.2,
                    "team_formation": 2.8,
                    "dog_spacing": 0.6,
                    "wall_margin": 0.2,
                    "wait_bias": 0.1,
                    "rear_drive": 1.05,
                    "flank_control": 0.95,
                    "collector_focus": 1.15,
                    "blocker_cover": 1.0,
                    "rear_behind_flock": 1.0,
                    "rear_drive_to_pen": 1.0,
                    "rear_avoid_overpressure": 0.9,
                    "rear_spacing": 0.55,
                    "flank_side_control": 1.1,
                    "flank_handedness": 0.75,
                    "flank_escape_blocking": 0.85,
                    "flank_spacing": 0.65,
                    "flank_wall_margin": 0.35,
                    "collector_stray_focus": 1.15,
                    "collector_return_to_flock": 0.9,
                    "collector_avoid_scatter": 0.85,
                    "collector_rejoin_angle": 0.75,
                    "blocker_escape_route_cover": 1.0,
                    "blocker_gate_control": 1.1,
                    "blocker_funnel_lane": 0.9,
                    "blocker_hold_position": 0.8,
                    "blocker_spacing": 0.55,
                    "anti_stack_penalty": 2.0,
                    "oscillation_penalty": 0.8,
                },
            )()
        return type(
            "Weights",
            (),
            {
                "nearest_sheep": 0.75,
                "flock_center": 2.1,
                "pen_pressure": 0.0,
                "behind_flock": 1.1,
                "team_formation": 2.2,
                "dog_spacing": 0.7,
                "wall_margin": 0.25,
                "wait_bias": -1.2,
                "rear_drive": 0.0,
                "flank_control": 0.0,
                "collector_focus": 0.0,
                "blocker_cover": 0.0,
                "rear_behind_flock": 0.0,
                "rear_drive_to_pen": 0.0,
                "rear_avoid_overpressure": 0.0,
                "rear_spacing": 0.0,
                "flank_side_control": 0.0,
                "flank_handedness": 0.0,
                "flank_escape_blocking": 0.0,
                "flank_spacing": 0.0,
                "flank_wall_margin": 0.0,
                "collector_stray_focus": 0.0,
                "collector_return_to_flock": 0.0,
                "collector_avoid_scatter": 0.0,
                "collector_rejoin_angle": 0.0,
                "blocker_escape_route_cover": 0.0,
                "blocker_gate_control": 0.0,
                "blocker_funnel_lane": 0.0,
                "blocker_hold_position": 0.0,
                "blocker_spacing": 0.0,
                "anti_stack_penalty": 2.0,
                "oscillation_penalty": 0.8,
            },
        )()

    def _action_context_debug(
        self,
        policy_mode: PolicyMode,
        dog_index: int,
        position: Point,
    ) -> dict[str, Any]:
        if self._instinct_target_awareness_enabled(policy_mode):
            return self._pressure_position_debug(dog_index, position)
        return self._instinct_position_debug(dog_index, position)

    def _instinct_position_debug(self, dog_index: int, position: Point) -> dict[str, Any]:
        flock_center = self._flock_center()
        if flock_center is None:
            return {
                "desired_pressure_target": {"x": position.x, "y": position.y},
                "distance_to_pressure_target": 0.0,
                "pressure_side_alignment": 0.0,
                "between_flock_and_pen": False,
                "inside_or_too_close_to_flock": False,
                "distance_to_flock": 0.0,
                "flock_buffer_radius": 0.0,
                "focus_mode": "formation",
                "distance_to_focus_sheep": None,
                "holding_pressure_position": False,
                "role_slot": self._pressure_role_offset(dog_index),
            }

        focus_sheep, focus_mode = self._focus_sheep_for_dog(dog_index, flock_center, position)
        target = self._instinct_target(dog_index, flock_center, position, focus_sheep, focus_mode)
        distance_to_flock = position.distance_to(flock_center)
        flock_buffer_radius = self._flock_buffer_radius()
        desired_radius = max(3.0, flock_buffer_radius + 1.0)
        alignment = self._radius_alignment(distance_to_flock, desired_radius)
        distance_to_target = position.distance_to(target)
        return {
            "desired_pressure_target": {"x": target.x, "y": target.y},
            "distance_to_pressure_target": distance_to_target,
            "pressure_side_alignment": alignment,
            "between_flock_and_pen": False,
            "inside_or_too_close_to_flock": distance_to_flock < flock_buffer_radius,
            "distance_to_flock": distance_to_flock,
            "flock_buffer_radius": flock_buffer_radius,
            "focus_mode": focus_mode,
            "distance_to_focus_sheep": (
                position.distance_to(focus_sheep.position) if focus_sheep is not None else None
            ),
            "holding_pressure_position": distance_to_target <= 1.5 and alignment >= 0.25,
            "role_slot": self._pressure_role_offset(dog_index),
        }

    def _pressure_position_debug(self, dog_index: int, position: Point) -> dict[str, Any]:
        flock_center = self._flock_center()
        pen_center = self._pen.center
        if flock_center is None:
            return {
                "desired_pressure_target": {"x": position.x, "y": position.y},
                "distance_to_pressure_target": 0.0,
                "pressure_side_alignment": 0.0,
                "between_flock_and_pen": False,
                "inside_or_too_close_to_flock": False,
                "distance_to_flock": 0.0,
                "flock_buffer_radius": 0.0,
                "focus_mode": "formation",
                "distance_to_focus_sheep": None,
                "holding_pressure_position": False,
                "role_slot": self._pressure_role_offset(dog_index),
            }

        role_assignment = self._role_assignments.get(dog_index)
        target = (
            role_assignment.target
            if role_assignment is not None
            else self._formation_target(dog_index, flock_center, pen_center)
        )
        alignment = self._alignment_score(position, flock_center, pen_center)
        focus_sheep, focus_mode = self._focus_sheep_for_dog(dog_index, flock_center, position)
        distance_to_flock = position.distance_to(flock_center)
        flock_buffer_radius = self._flock_buffer_radius()
        distance_to_target = position.distance_to(target)
        return {
            "desired_pressure_target": {"x": target.x, "y": target.y},
            "distance_to_pressure_target": distance_to_target,
            "pressure_side_alignment": alignment,
            "between_flock_and_pen": alignment < -0.1,
            "inside_or_too_close_to_flock": distance_to_flock < flock_buffer_radius,
            "distance_to_flock": distance_to_flock,
            "flock_buffer_radius": flock_buffer_radius,
            "focus_mode": focus_mode,
            "distance_to_focus_sheep": (
                position.distance_to(focus_sheep.position) if focus_sheep is not None else None
            ),
            "holding_pressure_position": distance_to_target <= 1.5 and alignment >= 0.35,
            "role_slot": self._pressure_role_offset(dog_index),
        }

    def _focus_sheep_for_dog(
        self,
        dog_index: int,
        flock_center: Point,
        position: Point,
    ) -> tuple[SheepState | None, str]:
        unpenned = [sheep for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return None, "formation"
        straggler = self._straggler_sheep(flock_center)
        if straggler is not None and self._assigned_straggler_dog(straggler) == dog_index:
            return straggler, "collect"
        nearest = min(unpenned, key=lambda sheep: sheep.position.distance_to(position))
        return nearest, "formation"

    def _straggler_sheep(self, flock_center: Point) -> SheepState | None:
        unpenned = [sheep for sheep in self._sheep if not sheep.penned]
        if len(unpenned) <= 1:
            return None
        straggler = max(unpenned, key=lambda sheep: sheep.position.distance_to(flock_center))
        spread = max(1.5, self._flock_spread())
        if straggler.position.distance_to(flock_center) <= max(3.5, spread * 1.8):
            return None
        return straggler

    def _assigned_straggler_dog(self, straggler: SheepState) -> int:
        return min(
            self._dogs,
            key=lambda dog: (dog.position.distance_to(straggler.position), dog.index),
        ).index

    def _flock_buffer_radius(self) -> float:
        return max(2.0, self._flock_spread() + 1.0)

    def _wall_avoidance(self, position: Point, axis: str) -> float:
        margin = 3
        if axis == "x":
            if position.x < margin:
                return margin - position.x
            if position.x > self.env_config.width - 1 - margin:
                return -(position.x - (self.env_config.width - 1 - margin))
            return 0.0
        if position.y < margin:
            return margin - position.y
        if position.y > self.env_config.height - 1 - margin:
            return -(position.y - (self.env_config.height - 1 - margin))
        return 0.0

    def _wall_margin(self, position: Point) -> float:
        left = position.x
        right = self.env_config.width - 1 - position.x
        top = position.y
        bottom = self.env_config.height - 1 - position.y
        return min(left, right, top, bottom)

    def _record_position_history(
        self, history: list[Point], position: Point, limit: int = 6
    ) -> None:
        history.append(position)
        if len(history) > limit:
            del history[:-limit]

    def _revisits_recent_position(self, history: list[Point], candidate: Point) -> bool:
        return len(history) >= 2 and candidate == history[-2]

    def _in_two_position_loop(self, history: list[Point]) -> bool:
        if len(history) < 4:
            return False
        tail = history[-4:]
        return tail[-1] == tail[-3] and tail[-2] == tail[-4] and len({*tail}) <= 2

    def _oscillation_penalty(
        self,
        dog: DogState,
        candidate: Point,
        action: Action,
        weights: Any,
    ) -> float:
        if action == "wait":
            return 0.0
        penalty = 0.0
        if candidate in dog.recent_positions[-3:]:
            penalty += weights.oscillation_penalty * 0.5
        if len(dog.recent_positions) >= 4 and dog.recent_positions[-1] == dog.recent_positions[-3]:
            penalty += weights.oscillation_penalty
        if dog.blocked_steps > 1:
            penalty += dog.blocked_steps * 0.1
        return penalty

    def _sprint_action_penalty(
        self,
        dog: DogState,
        action: Action,
        candidate: Point,
        assignment: RoleAssignment,
        candidate_debug: dict[str, Any],
    ) -> float:
        if not self._is_sprint_action(action):
            return 0.0
        penalty = 0.35
        target_gain = dog.position.distance_to(assignment.target) - candidate.distance_to(
            assignment.target
        )
        if target_gain < 1.5:
            penalty += 0.45
        focus_distance = candidate_debug.get("distance_to_focus_sheep")
        if isinstance(focus_distance, (int, float)) and focus_distance <= 2.5:
            penalty += 0.9
        if candidate_debug["holding_pressure_position"]:
            penalty += 0.8
        if candidate_debug["inside_or_too_close_to_flock"]:
            penalty += 0.75
        return penalty

    def _is_reverse_action(self, previous: str, action: Action) -> bool:
        previous_direction = self._base_action(previous)
        action_direction = self._base_action(action)
        return (
            previous_direction in OPPOSITE_DIRECTIONS
            and OPPOSITE_DIRECTIONS[previous_direction] == action_direction
        )

    def _reverse_is_clearly_better(
        self,
        current_debug: dict[str, Any],
        candidate_debug: dict[str, Any],
    ) -> bool:
        return bool(
            current_debug["between_flock_and_pen"]
            and not candidate_debug["between_flock_and_pen"]
            or current_debug["inside_or_too_close_to_flock"]
            and not candidate_debug["inside_or_too_close_to_flock"]
            or candidate_debug["distance_to_pressure_target"]
            <= current_debug["distance_to_pressure_target"] - 1.0
            or candidate_debug["pressure_side_alignment"]
            >= current_debug["pressure_side_alignment"] + 0.45
        )

    def _snapshot_debug_payload(self) -> dict[str, Any]:
        instincts = self.config.rewards.instincts
        if not instincts.debug_reward_breakdown:
            return {}
        flock_center = self._flock_center()
        return {
            "curriculum_stage": instincts.curriculum_stage,
            "enable_instinct_rewards": instincts.enable_instinct_rewards,
            "policy_mode": self.config.policy.policy_mode,
            "allow_instinct_target_awareness": self.config.policy.allow_instinct_target_awareness,
            "handler_target_enabled": self.config.policy.handler_target_enabled,
            "cumulative_gate_progress": self._cumulative_gate_progress,
            "controlled_stall_steps": self._controlled_stall_steps,
            "gate_corridor_occupancy": self._gate_corridor_occupancy(),
            "flock_center": (
                {"x": flock_center.x, "y": flock_center.y} if flock_center is not None else None
            ),
            "dogs": [
                {
                    "index": dog.index,
                    **self._action_context_debug(
                        self.config.policy.policy_mode,
                        dog.index,
                        dog.position,
                    ),
                }
                for dog in self._dogs
            ],
        }

    def _deadlock_state(self) -> dict[str, Any]:
        oscillating_dogs = [
            dog.index for dog in self._dogs if self._in_two_position_loop(dog.recent_positions)
        ]
        active = self._no_progress_steps >= 4 and bool(oscillating_dogs)
        return {
            "active": active,
            "wall_pinned_sheep": tuple(),
            "oscillating_dogs": tuple(oscillating_dogs),
        }

    def _flank_score(self, candidate: Point, flock_center: Point, target: Point) -> float:
        target_dx = target.x - flock_center.x
        target_dy = target.y - flock_center.y
        candidate_dx = candidate.x - flock_center.x
        candidate_dy = candidate.y - flock_center.y
        target_norm = max(1.0, (target_dx**2 + target_dy**2) ** 0.5)
        candidate_norm = max(1.0, (candidate_dx**2 + candidate_dy**2) ** 0.5)
        cross = abs(target_dx * candidate_dy - target_dy * candidate_dx)
        return cross / (target_norm * candidate_norm)

    def _flank_handedness_score(
        self,
        candidate: Point,
        flock_center: Point,
        target: Point,
        role: DogRole,
    ) -> float:
        target_dx = target.x - flock_center.x
        target_dy = target.y - flock_center.y
        candidate_dx = candidate.x - flock_center.x
        candidate_dy = candidate.y - flock_center.y
        target_norm = max(1.0, (target_dx**2 + target_dy**2) ** 0.5)
        candidate_norm = max(1.0, (candidate_dx**2 + candidate_dy**2) ** 0.5)
        signed_cross = (target_dx * candidate_dy - target_dy * candidate_dx) / (
            target_norm * candidate_norm
        )
        preferred_sign = 1.0 if role == DogRole.LEFT_FLANKER else -1.0
        return preferred_sign * signed_cross

    def _blocker_funnel_score(self, candidate: Point, gate_position: Point) -> float:
        del gate_position
        along_distance, lateral_distance = self._gate_axis_distances(candidate)
        lane_fit = 1.0 - min(1.0, abs(lateral_distance - 2.0) / 2.0)
        approach_fit = 1.0 - min(1.0, abs(along_distance - 1.0) / 3.0)
        return max(-1.0, lane_fit + approach_fit - 1.0)

    def _is_controlled_state(self, flock_center: Point | None) -> bool:
        if flock_center is None:
            return False
        if self._flock_spread() > self.env_config.controlled_flock_spread_threshold:
            return False
        return any(
            dog.position.distance_to(flock_center) <= self._flock_buffer_radius() + 3.0
            for dog in self._dogs
        )

    def _flock_gate_distance(self, flock_center: Point | None) -> float:
        if flock_center is None:
            return 0.0
        return flock_center.distance_to(self._gate_position())

    def _average_gate_corridor_distance(self) -> float:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return 0.0
        return fmean(self._gate_axis_distances(position)[1] for position in unpenned)

    def _gate_corridor_occupancy(self) -> float:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return 0.0
        occupied = sum(1 for position in unpenned if self._is_in_gate_corridor(position))
        return occupied / len(unpenned)

    def _is_in_gate_corridor(self, position: Point) -> bool:
        along_distance, lateral_distance = self._gate_axis_distances(position)
        return (
            lateral_distance <= self.env_config.gate_corridor_half_width
            and along_distance <= self.env_config.gate_approach_distance
        )

    def _gate_axis_distances(self, position: Point) -> tuple[float, float]:
        gate = self._gate_position()
        if self._pen.opening in {"left", "right"}:
            return abs(gate.x - position.x), abs(gate.y - position.y)
        return abs(gate.y - position.y), abs(gate.x - position.x)

    def _wall_pinned_sheep_ratio(self) -> float:
        unpenned = [sheep.position for sheep in self._sheep if not sheep.penned]
        if not unpenned:
            return 0.0
        pinned = sum(1 for position in unpenned if self._wall_margin(position) <= 1.0)
        return pinned / len(unpenned)

    def _deadlock_action_adjustment(
        self,
        dog_index: int,
        current: Point,
        candidate: Point,
        action: Action,
        policy_mode: PolicyMode,
        candidate_debug: dict[str, Any],
    ) -> float:
        state = self._deadlock_state()
        if not state["active"]:
            return 0.0
        flock_center = self._flock_center()
        if flock_center is None:
            return 0.0
        if self._instinct_target_awareness_enabled(policy_mode):
            target = self._pen.center
        else:
            desired = candidate_debug["desired_pressure_target"]
            target = Point(int(desired["x"]), int(desired["y"]))
        flank_bonus = self._flank_score(candidate, flock_center, target) * 1.6
        straight_pressure_penalty = -0.7 if candidate_debug["holding_pressure_position"] else 0.0
        bounce_penalty = 0.0
        if (
            dog_index in state["oscillating_dogs"]
            and action in {"up", "down"}
            and candidate.x == current.x
        ):
            bounce_penalty -= 0.9
        return flank_bonus + straight_pressure_penalty + bounce_penalty

    def _loop_penalty(self, dog: DogState, candidate: Point) -> float:
        penalty = 0.0
        if self._revisits_recent_position(dog.recent_positions, candidate):
            penalty -= 0.35
        if self._in_two_position_loop(dog.recent_positions):
            penalty -= 0.55
        return penalty

    def _sign(self, value: float) -> int:
        return 0 if value == 0 else (1 if value > 0 else -1)

    def _status(self) -> str:
        if self._success:
            return "success"
        if self._timeout:
            return "timeout"
        if self._stopped:
            return "stopped"
        if self._terminated:
            return "terminated"
        return "running"
