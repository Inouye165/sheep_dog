"""Reward calculation for herding progress."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot

from sheepdog.config import InstinctRewardConfig, RewardConfig

Position = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Named reward components for a single environment step."""

    progress_to_pen: float = 0.0
    sheep_penned: float = 0.0
    flock_cohesion: float = 0.0
    scatter_penalty: float = 0.0
    time_penalty: float = 0.0
    no_progress_penalty: float = 0.0
    wall_pressure_penalty: float = 0.0
    wait_penalty: float = 0.0
    gate_progress: float = 0.0
    gate_corridor_progress: float = 0.0
    gate_alignment: float = 0.0
    stalled_control_penalty: float = 0.0
    wrong_hold_penalty: float = 0.0
    terminal_success: float = 0.0
    terminal_failure: float = 0.0
    pressure_zone: float = 0.0
    safe_pressure: float = 0.0
    grouping: float = 0.0
    target_progress: float = 0.0
    chaos_penalty: float = 0.0
    overpressure_penalty: float = 0.0
    split_flock_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardInputs:
    """Minimal set of values used to score a step.

    The trailing positional fields are optional and feed the instinct
    reward terms; when omitted (or when instincts are disabled in the
    reward config) those terms remain zero and legacy reward behavior
    is unchanged.
    """

    previous_average_distance: float
    current_average_distance: float
    previous_flock_spread: float
    current_flock_spread: float
    newly_penned: int
    no_progress_step: bool
    touched_wall: bool
    waited_without_reason: bool
    terminated: bool
    timeout: bool
    success: bool
    previous_gate_distance: float = 0.0
    current_gate_distance: float = 0.0
    previous_gate_corridor_distance: float = 0.0
    current_gate_corridor_distance: float = 0.0
    previous_gate_corridor_occupancy: float = 0.0
    current_gate_corridor_occupancy: float = 0.0
    controlled_stall_steps: int = 0
    wrong_hold_active: bool = False
    tactically_valid_hold: bool = False
    dog_positions: tuple[Position, ...] = field(default_factory=tuple)
    sheep_positions: tuple[Position, ...] = field(default_factory=tuple)
    flock_centroid: Position | None = None
    previous_flock_centroid: Position | None = None
    target_position: Position | None = None


def _distance(a: Position, b: Position) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _safe_pressure_score(distance: float, config: InstinctRewardConfig) -> float:
    """Annulus reward: positive in the band, negative when too close."""

    if distance < config.safe_pressure_min_distance:
        return -(config.safe_pressure_min_distance - distance)
    if distance > config.safe_pressure_max_distance:
        return 0.0
    band = max(1e-6, config.safe_pressure_max_distance - config.safe_pressure_min_distance)
    midpoint = (config.safe_pressure_min_distance + config.safe_pressure_max_distance) / 2.0
    return 1.0 - abs(distance - midpoint) / (band / 2.0)


def _pressure_zone_alignment(dog: Position, flock: Position, target: Position) -> float:
    """Return a value in [-1, 1]; 1 when the dog is opposite the target across the flock."""

    target_dx = target[0] - flock[0]
    target_dy = target[1] - flock[1]
    dog_dx = dog[0] - flock[0]
    dog_dy = dog[1] - flock[1]
    target_norm = hypot(target_dx, target_dy)
    dog_norm = hypot(dog_dx, dog_dy)
    if target_norm < 1e-6 or dog_norm < 1e-6:
        return 0.0
    cosine = (target_dx * dog_dx + target_dy * dog_dy) / (target_norm * dog_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return -cosine


def _max_sheep_offset(centroid: Position, sheep_positions: tuple[Position, ...]) -> float:
    if not sheep_positions:
        return 0.0
    return max(_distance(centroid, sheep) for sheep in sheep_positions)


class RewardComputer:
    """Compute a structured reward breakdown."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config

    def compute(self, inputs: RewardInputs) -> RewardBreakdown:
        progress_delta = inputs.previous_average_distance - inputs.current_average_distance
        progress_to_pen = progress_delta * self._config.progress_scale
        sheep_penned = inputs.newly_penned * self._config.sheep_penned_reward
        cohesion_delta = inputs.previous_flock_spread - inputs.current_flock_spread
        flock_cohesion = cohesion_delta * self._config.flock_cohesion_scale
        scatter_penalty = max(0.0, -cohesion_delta) * self._config.scatter_penalty_scale
        time_penalty = -self._config.time_penalty
        no_progress_penalty = -self._config.no_progress_penalty if inputs.no_progress_step else 0.0
        wall_pressure_penalty = -self._config.wall_pressure_penalty if inputs.touched_wall else 0.0
        wait_penalty = -self._config.wait_penalty if inputs.waited_without_reason else 0.0
        gate_progress = (
            inputs.previous_gate_distance - inputs.current_gate_distance
        ) * self._config.gate_progress_scale
        gate_corridor_progress = (
            (inputs.previous_gate_corridor_distance - inputs.current_gate_corridor_distance)
            * self._config.gate_corridor_progress_scale
        )
        gate_alignment = (
            inputs.current_gate_corridor_occupancy - inputs.previous_gate_corridor_occupancy
        ) * self._config.gate_alignment_scale
        stalled_control_penalty = 0.0
        if inputs.controlled_stall_steps > 0 and not inputs.tactically_valid_hold:
            stalled_control_penalty = -self._config.stalled_control_penalty * min(
                2.0,
                inputs.controlled_stall_steps / 4.0,
            )
        wrong_hold_penalty = (
            -self._config.wrong_hold_penalty
            if inputs.wrong_hold_active and not inputs.tactically_valid_hold
            else 0.0
        )
        terminal_success = self._config.terminal_success_reward if inputs.success else 0.0
        terminal_failure = -self._config.terminal_failure_penalty if inputs.timeout else 0.0
        instincts = self._compute_instincts(inputs)
        total = (
            progress_to_pen
            + sheep_penned
            + flock_cohesion
            - scatter_penalty
            + time_penalty
            + no_progress_penalty
            + wall_pressure_penalty
            + wait_penalty
            + gate_progress
            + gate_corridor_progress
            + gate_alignment
            + stalled_control_penalty
            + wrong_hold_penalty
            + terminal_success
            + terminal_failure
            + instincts["pressure_zone"]
            + instincts["safe_pressure"]
            + instincts["grouping"]
            + instincts["target_progress"]
            + instincts["chaos_penalty"]
            + instincts["overpressure_penalty"]
            + instincts["split_flock_penalty"]
        )
        return RewardBreakdown(
            progress_to_pen=progress_to_pen,
            sheep_penned=sheep_penned,
            flock_cohesion=flock_cohesion,
            scatter_penalty=scatter_penalty,
            time_penalty=time_penalty,
            no_progress_penalty=no_progress_penalty,
            wall_pressure_penalty=wall_pressure_penalty,
            wait_penalty=wait_penalty,
            gate_progress=gate_progress,
            gate_corridor_progress=gate_corridor_progress,
            gate_alignment=gate_alignment,
            stalled_control_penalty=stalled_control_penalty,
            wrong_hold_penalty=wrong_hold_penalty,
            terminal_success=terminal_success,
            terminal_failure=terminal_failure,
            pressure_zone=instincts["pressure_zone"],
            safe_pressure=instincts["safe_pressure"],
            grouping=instincts["grouping"],
            target_progress=instincts["target_progress"],
            chaos_penalty=instincts["chaos_penalty"],
            overpressure_penalty=instincts["overpressure_penalty"],
            split_flock_penalty=instincts["split_flock_penalty"],
            total=total,
        )

    def _compute_instincts(self, inputs: RewardInputs) -> dict[str, float]:
        zero = {
            "pressure_zone": 0.0,
            "safe_pressure": 0.0,
            "grouping": 0.0,
            "target_progress": 0.0,
            "chaos_penalty": 0.0,
            "overpressure_penalty": 0.0,
            "split_flock_penalty": 0.0,
        }
        instinct_config = self._config.instincts
        if not instinct_config.enable_instinct_rewards:
            return zero
        flock = inputs.flock_centroid
        target = inputs.target_position
        if flock is None or target is None or not inputs.dog_positions:
            return zero

        pressure_zone = 0.0
        safe_pressure = 0.0
        chaos_penalty = 0.0
        overpressure_penalty = 0.0
        for dog in inputs.dog_positions:
            pressure_zone += _pressure_zone_alignment(dog, flock, target)
            safe_pressure += _safe_pressure_score(_distance(dog, flock), instinct_config)
            if _distance(dog, flock) <= instinct_config.chaos_inside_flock_distance:
                chaos_penalty -= 1.0
            for sheep in inputs.sheep_positions:
                if _distance(dog, sheep) <= instinct_config.overpressure_distance:
                    overpressure_penalty -= 1.0
        dog_count = max(1, len(inputs.dog_positions))
        pressure_zone *= instinct_config.pressure_zone_weight / dog_count
        safe_pressure *= instinct_config.safe_pressure_weight / dog_count

        grouping_delta = inputs.previous_flock_spread - inputs.current_flock_spread
        grouping = grouping_delta * instinct_config.grouping_weight

        target_progress = 0.0
        if inputs.previous_flock_centroid is not None:
            previous_distance = _distance(inputs.previous_flock_centroid, target)
            current_distance = _distance(flock, target)
            target_progress = (
                previous_distance - current_distance
            ) * instinct_config.target_progress_weight

        scatter_delta = inputs.current_flock_spread - inputs.previous_flock_spread
        if scatter_delta > instinct_config.chaos_scatter_delta:
            chaos_penalty -= scatter_delta - instinct_config.chaos_scatter_delta
        chaos_penalty *= instinct_config.chaos_penalty_weight
        overpressure_penalty *= instinct_config.overpressure_penalty_weight

        max_offset = _max_sheep_offset(flock, inputs.sheep_positions)
        split_flock_penalty = 0.0
        if (
            inputs.current_flock_spread > 0
            and max_offset > instinct_config.split_flock_ratio * inputs.current_flock_spread
        ):
            overflow = max_offset - instinct_config.split_flock_ratio * inputs.current_flock_spread
            split_flock_penalty = -overflow * instinct_config.split_flock_penalty_weight

        return {
            "pressure_zone": pressure_zone,
            "safe_pressure": safe_pressure,
            "grouping": grouping,
            "target_progress": target_progress,
            "chaos_penalty": chaos_penalty,
            "overpressure_penalty": overpressure_penalty,
            "split_flock_penalty": split_flock_penalty,
        }
