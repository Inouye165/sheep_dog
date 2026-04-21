"""Deterministic 2D sheep herding environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import inf
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import DogState, EpisodeStats, Pen, Point, SheepState
from sheepdog.policies.base import Action
from sheepdog.rewards import RewardBreakdown, RewardComputer, RewardInputs

ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "wait": (0, 0),
}

ACTION_ORDER: tuple[Action, ...] = ("up", "down", "left", "right", "wait")


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Immutable view of one agent."""

    index: int
    x: int
    y: int
    penned: bool = False
    last_action: str = "wait"


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Immutable environment snapshot for UI playback and replay export."""

    step: int
    simulated_seconds: float
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A single frame of simulation history."""

    step: int
    actions: tuple[str, ...]
    snapshot: EnvironmentSnapshot
    reward: RewardBreakdown

    def to_dict(self) -> dict[str, Any]:
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
        self._no_progress_steps = 0
        self._reward_total = 0.0
        self._terminated = False
        self._timeout = False
        self._stopped = False
        self._success = False
        self._stop_reason = ""
        self._history: list[StepRecord] = []
        self._stats = EpisodeStats()

    @property
    def dog_count(self) -> int:
        return self.env_config.dogs

    @property
    def sheep_count(self) -> int:
        return self.env_config.sheep

    @property
    def pen(self) -> Pen:
        return self._pen

    @property
    def history(self) -> tuple[StepRecord, ...]:
        return tuple(self._history)

    def reset(self, seed: int | None = None) -> EnvironmentSnapshot:
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
        pen_origin = Point(self.env_config.width - self.env_config.pen_width - 2, 1)
        self._pen = Pen(
            pen_origin,
            self.env_config.pen_width,
            self.env_config.pen_height,
            opening=self.env_config.pen_opening,
        )
        self._fence_cells = self._pen.fence_cells()
        self._dogs = self._initial_dogs()
        self._sheep = self._initial_sheep()
        self._previous_average_distance = self._average_distance_to_pen()
        self._previous_flock_spread = self._flock_spread()
        return self.get_state_snapshot()

    def _advance_position(self, position: Point, action: Action, speed: int) -> Point:
        current = position
        for _ in range(max(1, speed)):
            candidate = self._target_position(current, action)
            if candidate == current:
                break
            current = candidate
        return current

    def get_state_snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            step=self._step_count,
            simulated_seconds=self._simulated_seconds,
            dogs=tuple(
                AgentSnapshot(
                    index=dog.index,
                    x=dog.position.x,
                    y=dog.position.y,
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
                )
                for sheep in self._sheep
            ),
            pen=self._pen,
            fence_cells=tuple(
                (cell.x, cell.y)
                for cell in sorted(self._fence_cells, key=lambda p: (p.y, p.x))
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
        )

    def action_mask_for_dog(self, dog_index: int) -> dict[Action, bool]:
        dog = self._dogs[dog_index]
        current_score = self._action_score(dog_index, "wait")
        move_scores = {
            action: self._action_score(dog_index, action)
            for action in ACTION_ORDER
            if action != "wait"
        }
        best_move_score = max(move_scores.values()) if move_scores else -inf
        wait_allowed = current_score >= best_move_score - 0.05 or self._tactically_valid_wait(dog)
        return {
            "up": self._target_position(dog.position, "up") != dog.position,
            "down": self._target_position(dog.position, "down") != dog.position,
            "left": self._target_position(dog.position, "left") != dog.position,
            "right": self._target_position(dog.position, "right") != dog.position,
            "wait": wait_allowed,
        }

    def score_action_for_dog(
        self,
        dog_index: int,
        action: Action,
        weights: Any | None = None,
    ) -> float:
        return self._action_score(dog_index, action, weights=weights)

    def run_policy(self, policy: object, seed: int, capture_replay: bool = False) -> EpisodeResult:
        self.reset(seed)
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
        self, actions: list[Action], capture_replay: bool = False
    ) -> tuple[EnvironmentSnapshot, RewardBreakdown]:
        if self._terminated:
            raise RuntimeError("Cannot step a terminated episode.")
        if len(actions) != len(self._dogs):
            raise ValueError("Action count does not match dog count.")

        previous_snapshot = self.get_state_snapshot()
        self._apply_dog_actions(actions)
        self._move_sheep()
        self._step_count += 1
        self._simulated_seconds += self.env_config.seconds_per_step

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

        breakdown = self.reward_computer.compute(
            RewardInputs(
                previous_average_distance=previous_snapshot.average_distance_to_pen,
                current_average_distance=current_snapshot.average_distance_to_pen,
                previous_flock_spread=previous_snapshot.flock_spread,
                current_flock_spread=current_snapshot.flock_spread,
                newly_penned=newly_penned,
                no_progress_step=not progress_made,
                touched_wall=self._touched_wall_this_step(actions),
                waited_without_reason=self._waited_without_reason(actions),
                terminated=self._terminated,
                timeout=self._timeout,
                success=self._success,
            )
        )
        self._reward_total += breakdown.total
        self._previous_average_distance = current_snapshot.average_distance_to_pen
        self._previous_flock_spread = current_snapshot.flock_spread

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
            final_reward_breakdown=breakdown.to_dict(),
        )

        if capture_replay:
            self._history.append(
                StepRecord(
                    step=self._step_count,
                    actions=tuple(actions),
                    snapshot=final_snapshot,
                    reward=breakdown,
                )
            )

        return final_snapshot, breakdown

    def export_replay(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            import json

            json.dump([record.to_dict() for record in self._history], handle, indent=2)
        return target

    def _initial_dogs(self) -> list[DogState]:
        dogs: list[DogState] = []
        base_y = self.env_config.height // 2 + 2
        for index in range(self.env_config.dogs):
            dogs.append(
                DogState(
                    index=index,
                    position=Point(2 + index * 2, base_y + self._rng.randint(-1, 1)),
                )
            )
        return dogs

    def _initial_sheep(self) -> list[SheepState]:
        sheep: list[SheepState] = []
        base_x = self.env_config.width // 3
        base_y = self.env_config.height // 2
        for index in range(self.env_config.sheep):
            sheep.append(
                SheepState(
                    index=index,
                    position=Point(
                        base_x + self._rng.randint(-2, 2),
                        base_y + self._rng.randint(-2, 2),
                    ).clamp(self.env_config.width, self.env_config.height),
                )
            )
        return sheep

    def _target_position(self, position: Point, action: Action) -> Point:
        dx, dy = ACTION_DELTAS[action]
        candidate = Point(position.x + dx, position.y + dy).clamp(
            self.env_config.width, self.env_config.height
        )
        if candidate in self._fence_cells and not self._pen.contains(position):
            return position
        return candidate

    def _apply_dog_actions(self, actions: list[Action]) -> None:
        for dog, action in zip(self._dogs, actions, strict=True):
            candidate = self._advance_position(
                dog.position,
                action,
                self.env_config.dog_speed,
            )
            dog.position = candidate
            dog.last_action = action

    def _move_sheep(self) -> None:
        dog_positions = [dog.position for dog in self._dogs]
        next_positions: list[Point] = []
        for sheep in self._sheep:
            if sheep.penned or self._pen.contains(sheep.position):
                sheep.penned = True
                sheep.panic_steps = 0
                next_positions.append(sheep.position)
                continue
            move = sheep.position
            flock_center = self._flock_center()
            for _ in range(max(1, self.env_config.sheep_speed)):
                move = self._sheep_step(move, dog_positions, flock_center, sheep.panic_steps)
                if sheep.panic_steps > 0:
                    sheep.panic_steps = max(0, sheep.panic_steps - 1)
                if move == sheep.position:
                    break
            if self._nearest_dog_distance(move, dog_positions) <= self.env_config.dog_vision:
                sheep.panic_steps = max(sheep.panic_steps, 2)
            next_positions.append(move)

        for sheep, position in zip(self._sheep, next_positions, strict=True):
            sheep.position = position
            sheep.penned = self._pen.contains(position)

    def _sheep_step(
        self,
        position: Point,
        dog_positions: list[Point],
        flock_center: Point | None,
        panic_steps: int,
    ) -> Point:
        vector_x = 0.0
        vector_y = 0.0
        nearest_dog_distance = inf
        for dog_position in dog_positions:
            distance = max(1.0, position.distance_to(dog_position))
            nearest_dog_distance = min(nearest_dog_distance, distance)
            if distance <= self.env_config.dog_vision:
                weight = (self.env_config.dog_vision - distance + 1.0) / distance
                vector_x += (position.x - dog_position.x) * weight
                vector_y += (position.y - dog_position.y) * weight
        if panic_steps <= 0 and flock_center is not None:
            vector_x += (flock_center.x - position.x) * 0.35
            vector_y += (flock_center.y - position.y) * 0.35
        if nearest_dog_distance <= self.env_config.dog_vision:
            vector_x *= 1.5
            vector_y *= 1.5
        vector_x += self._wall_avoidance(position, axis="x")
        vector_y += self._wall_avoidance(position, axis="y")
        if vector_x == 0 and vector_y == 0:
            return position
        if abs(vector_x) >= abs(vector_y):
            move = Point(position.x + self._sign(vector_x), position.y)
        else:
            move = Point(position.x, position.y + self._sign(vector_y))
        move = move.clamp(self.env_config.width, self.env_config.height)
        if move in self._fence_cells and not self._pen.contains(position):
            return position
        if move in dog_positions:
            return position
        return move

    def _nearest_dog_distance(self, position: Point, dog_positions: list[Point]) -> float:
        if not dog_positions:
            return inf
        return min(position.distance_to(dog_position) for dog_position in dog_positions)

    def _action_score(
        self,
        dog_index: int,
        action: Action,
        weights: Any | None = None,
    ) -> float:
        dog = self._dogs[dog_index]
        candidate = self._target_position(dog.position, action)
        flock_center = self._flock_center() or dog.position
        pen_center = self._pen.center
        formation_target = self._formation_target(dog_index, flock_center, pen_center)
        teammate_spacing = self._teammate_spacing(candidate, dog_index)
        driving_sheep = self._nearest_unpenned_sheep(candidate)
        distance_to_sheep = (
            candidate.distance_to(driving_sheep.position) if driving_sheep is not None else 0.0
        )
        drive_distance = 3.0
        drive_distance_error = abs(distance_to_sheep - drive_distance)
        distance_to_flock = candidate.distance_to(flock_center)
        distance_to_pen = candidate.distance_to(pen_center)
        distance_to_formation = candidate.distance_to(formation_target)
        alignment = self._alignment_score(candidate, flock_center, pen_center)
        wall_margin = self._wall_margin(candidate)
        if weights is None:
            weights = type(
                "Weights",
                (),
                {
                    "nearest_sheep": 1.4,
                    "flock_center": 1.0,
                    "pen_pressure": 0.9,
                    "behind_flock": 1.1,
                    "team_formation": 0.85,
                    "dog_spacing": 0.6,
                    "wall_margin": 0.35,
                    "wait_bias": -1.5,
                },
            )()
        wait_bias = weights.wait_bias if action == "wait" else 0.0
        return (
            weights.nearest_sheep * (-drive_distance_error)
            + weights.flock_center * (-distance_to_flock)
            + weights.pen_pressure * (-distance_to_pen)
            + weights.behind_flock * alignment
            + weights.team_formation * (-distance_to_formation)
            + weights.dog_spacing * (-teammate_spacing)
            + weights.wall_margin * wall_margin
            + wait_bias
        )

    def _formation_target(
        self,
        dog_index: int,
        flock_center: Point,
        pen_center: Point,
    ) -> Point:
        to_pen_x = self._sign(pen_center.x - flock_center.x)
        to_pen_y = self._sign(pen_center.y - flock_center.y)
        behind_distance = 3
        lateral_spacing = 3
        base_x = flock_center.x - to_pen_x * behind_distance
        base_y = flock_center.y - to_pen_y * behind_distance
        lateral_x = -to_pen_y
        lateral_y = to_pen_x
        lateral_offset = dog_index - (self.dog_count - 1) / 2
        offset = round(lateral_offset * lateral_spacing)
        return Point(base_x + lateral_x * offset, base_y + lateral_y * offset).clamp(
            self.env_config.width,
            self.env_config.height,
        )

    def _teammate_spacing(self, candidate: Point, dog_index: int) -> float:
        teammate_positions = [dog.position for dog in self._dogs if dog.index != dog_index]
        if not teammate_positions:
            return 0.0
        nearest_teammate_distance = min(
            candidate.distance_to(position) for position in teammate_positions
        )
        target_spacing = 4.0 if len(teammate_positions) > 1 else 3.0
        return abs(nearest_teammate_distance - target_spacing)

    def _tactically_valid_wait(self, dog: DogState) -> bool:
        if not self._sheep:
            return False
        current_score = self._action_score(dog.index, "wait")
        best_move_score = max(
            self._action_score(dog.index, action) for action in ACTION_ORDER if action != "wait"
        )
        return current_score >= best_move_score - 0.05

    def _waited_without_reason(self, actions: list[Action]) -> bool:
        return any(
            action == "wait" and not self._tactically_valid_wait(self._dogs[index])
            for index, action in enumerate(actions)
        )

    def _touched_wall_this_step(self, actions: list[Action]) -> bool:
        for dog, action in zip(self._dogs, actions, strict=True):
            candidate = self._target_position(dog.position, action)
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
        denominator = max(1.0, (vector_to_pen_x**2 + vector_to_pen_y**2) ** 0.5)
        return -numerator / denominator

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
