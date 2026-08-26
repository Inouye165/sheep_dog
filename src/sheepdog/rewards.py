"""Reward calculation for herding progress."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot, inf

from sheepdog.config import InstinctRewardConfig, RewardConfig

Position = tuple[float, float]

REWARD_SCHEMA_VERSION = "1.0"


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
    sprint_cost: float = 0.0
    gate_progress: float = 0.0
    gate_corridor_progress: float = 0.0
    gate_alignment: float = 0.0
    lane_crowding_penalty: float = 0.0
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
    farthest_sheep_progress: float = 0.0
    stray_ignore_penalty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to a plain dict."""
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
    sprint_count: int = 0
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
    previous_farthest_distance: float = 0.0
    current_farthest_distance: float = 0.0
    previous_dog_positions: tuple[Position, ...] = field(default_factory=tuple)
    previous_sheep_positions: tuple[Position, ...] = field(default_factory=tuple)


def _distance(a: Position, b: Position) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _safe_pressure_score(distance: float, config: InstinctRewardConfig) -> float:
    """Annulus reward: positive in the band, negative when too close."""

    if distance < config.safe_pressure_min_distance:
        return -(config.safe_pressure_min_distance - distance)
    if distance > config.safe_pressure_max_distance:
        far_overflow = distance - config.safe_pressure_max_distance
        return -min(1.0, far_overflow / max(1e-6, config.safe_pressure_max_distance))
    band = max(1e-6, config.safe_pressure_max_distance - config.safe_pressure_min_distance)
    midpoint = (config.safe_pressure_min_distance + config.safe_pressure_max_distance) / 2.0
    return 1.0 - abs(distance - midpoint) / (band / 2.0)


def _engagement_scale(distance: float, config: InstinctRewardConfig) -> float:
    """Decay instinct alignment rewards when a dog is too far to influence the flock."""

    if distance <= config.safe_pressure_max_distance:
        return 1.0
    fade_limit = max(
        config.safe_pressure_max_distance + 1e-6,
        config.safe_pressure_max_distance * 2.0,
    )
    if distance >= fade_limit:
        return 0.0
    return 1.0 - (
        (distance - config.safe_pressure_max_distance)
        / (fade_limit - config.safe_pressure_max_distance)
    )


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


def _lane_projection(
    sheep: Position, target: Position, dog: Position
) -> tuple[float, float, float]:
    lane_dx = target[0] - sheep[0]
    lane_dy = target[1] - sheep[1]
    lane_length = hypot(lane_dx, lane_dy)
    if lane_length < 1e-6:
        return 0.0, inf, 0.0
    unit_x = lane_dx / lane_length
    unit_y = lane_dy / lane_length
    dog_dx = dog[0] - sheep[0]
    dog_dy = dog[1] - sheep[1]
    forward_distance = dog_dx * unit_x + dog_dy * unit_y
    lateral_distance = abs(dog_dx * unit_y - dog_dy * unit_x)
    return forward_distance, lateral_distance, lane_length


class RewardComputer:
    """Compute a structured reward breakdown."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config

    def compute(self, inputs: RewardInputs) -> RewardBreakdown:
        """Compute a full reward breakdown for one team step."""
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
        sprint_cost = -inputs.sprint_count * self._config.sprint_cost_scale
        gate_progress = (
            inputs.previous_gate_distance - inputs.current_gate_distance
        ) * self._config.gate_progress_scale
        gate_corridor_progress = (
            inputs.previous_gate_corridor_distance - inputs.current_gate_corridor_distance
        ) * self._config.gate_corridor_progress_scale
        gate_alignment = (
            inputs.current_gate_corridor_occupancy - inputs.previous_gate_corridor_occupancy
        ) * self._config.gate_alignment_scale
        lane_crowding_penalty = self._lane_crowding_penalty(inputs)
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
        farthest_sheep_progress, stray_ignore_penalty = self._compute_stray_terms(inputs)
        total = (
            progress_to_pen
            + sheep_penned
            + flock_cohesion
            - scatter_penalty
            + time_penalty
            + no_progress_penalty
            + wall_pressure_penalty
            + wait_penalty
            + sprint_cost
            + gate_progress
            + gate_corridor_progress
            + gate_alignment
            + lane_crowding_penalty
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
        total += farthest_sheep_progress + stray_ignore_penalty
        total = max(-15.0, total)
        return RewardBreakdown(
            progress_to_pen=progress_to_pen,
            sheep_penned=sheep_penned,
            flock_cohesion=flock_cohesion,
            scatter_penalty=scatter_penalty,
            time_penalty=time_penalty,
            no_progress_penalty=no_progress_penalty,
            wall_pressure_penalty=wall_pressure_penalty,
            wait_penalty=wait_penalty,
            sprint_cost=sprint_cost,
            gate_progress=gate_progress,
            gate_corridor_progress=gate_corridor_progress,
            gate_alignment=gate_alignment,
            lane_crowding_penalty=lane_crowding_penalty,
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
            farthest_sheep_progress=farthest_sheep_progress,
            stray_ignore_penalty=stray_ignore_penalty,
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
            distance_to_flock = _distance(dog, flock)
            alignment = _pressure_zone_alignment(dog, flock, target)
            engagement = _engagement_scale(distance_to_flock, instinct_config)
            # When the dog has overshot the flock toward the pen (alignment < 0),
            # preserve a minimum penalty signal so distance decay cannot silence it.
            if alignment < 0.0:
                engagement = max(engagement, instinct_config.dog_overshoot_penalty_hold)
            pressure_zone += alignment * engagement
            safe_pressure += _safe_pressure_score(distance_to_flock, instinct_config)
            if distance_to_flock <= instinct_config.chaos_inside_flock_distance:
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
            # Cap the contribution so a single large flock movement cannot
            # override safe-pressure and overpressure signals.
            target_progress = max(-1.5, min(1.5, target_progress))

        scatter_delta = inputs.current_flock_spread - inputs.previous_flock_spread
        if scatter_delta > instinct_config.chaos_scatter_delta:
            chaos_penalty -= scatter_delta - instinct_config.chaos_scatter_delta
        chaos_penalty *= instinct_config.chaos_penalty_weight
        chaos_penalty = max(-5.0 * instinct_config.chaos_penalty_weight, chaos_penalty)
        overpressure_penalty *= instinct_config.overpressure_penalty_weight
        overpressure_penalty = max(-10.0 * instinct_config.overpressure_penalty_weight, overpressure_penalty)

        max_offset = _max_sheep_offset(flock, inputs.sheep_positions)
        split_flock_penalty = 0.0
        if (
            inputs.current_flock_spread > 0
            and max_offset > instinct_config.split_flock_ratio * inputs.current_flock_spread
        ):
            overflow = max_offset - instinct_config.split_flock_ratio * inputs.current_flock_spread
            split_flock_penalty = -overflow * instinct_config.split_flock_penalty_weight
            split_flock_penalty = max(-5.0 * instinct_config.split_flock_penalty_weight, split_flock_penalty)

        return {
            "pressure_zone": pressure_zone,
            "safe_pressure": safe_pressure,
            "grouping": grouping,
            "target_progress": target_progress,
            "chaos_penalty": chaos_penalty,
            "overpressure_penalty": overpressure_penalty,
            "split_flock_penalty": split_flock_penalty,
        }

    def _compute_stray_terms(self, inputs: RewardInputs) -> tuple[float, float]:
        """Compute farthest-sheep progress and stray-ignore penalty.

        Returns ``(farthest_sheep_progress, stray_ignore_penalty)``.  Both are
        0.0 when the corresponding scale config is 0.0 (default), so existing
        guided training is completely unaffected.
        """
        fp_scale = self._config.farthest_sheep_progress_scale
        si_scale = self._config.stray_ignore_penalty_scale
        if fp_scale == 0.0 and si_scale == 0.0:
            return 0.0, 0.0

        farthest_progress = 0.0
        if (
            fp_scale != 0.0
            and inputs.previous_farthest_distance != inputs.current_farthest_distance
        ):
            delta = inputs.previous_farthest_distance - inputs.current_farthest_distance
            farthest_progress = delta * fp_scale

        stray_penalty = 0.0
        if si_scale != 0.0 and inputs.current_farthest_distance > 0.0:
            # Penalise each step that the farthest unpenned sheep remains far.
            # Scale by the normalised distance so nearer strays cost less.
            effective_farthest_dist = min(30.0, inputs.current_farthest_distance)
            stray_penalty = -effective_farthest_dist * si_scale

            # Add dense dog approach penalty and progress reward for isolated strays
            if inputs.flock_centroid is not None and inputs.dog_positions and inputs.sheep_positions:
                farthest_sheep = None
                max_dist = -1.0
                for s in inputs.sheep_positions:
                    dist = _distance(s, inputs.target_position)
                    if dist > max_dist:
                        max_dist = dist
                        farthest_sheep = s
                
                if farthest_sheep is not None:
                    dist_to_centroid = _distance(farthest_sheep, inputs.flock_centroid)
                    # A sheep/cluster requires approach incentive if it's separated from the flock centroid (>8.0),
                    # or if unpenned sheep remain far from the pen target (>10.0)
                    is_isolated = (dist_to_centroid > 8.0) or (max_dist > 10.0)
                    if is_isolated:
                        min_dog_dist = min(_distance(dog, farthest_sheep) for dog in inputs.dog_positions)
                        # Dense approach penalty: 10x the stray ignore scale to provide a clear gradient
                        effective_min_dog_dist = min(12.0, min_dog_dist)
                        stray_penalty -= effective_min_dog_dist * (si_scale * 10.0)

                        # Dense approach progress reward: localized distance-based multiplier
                        # when moving toward isolated farther_stray sheep or distant unpenned cluster.
                        if inputs.previous_dog_positions:
                            prev_min_dog_dist = min(_distance(dog, farthest_sheep) for dog in inputs.previous_dog_positions)
                            progress = prev_min_dog_dist - min_dog_dist
                            # Localized distance-based multiplier: increases as the sheep is more isolated or distant
                            isolation_multiplier = max(1.0, dist_to_centroid / 8.0) if dist_to_centroid > 8.0 else max(1.0, max_dist / 10.0)
                            # Progress reward is scaled by the stray ignore penalty scale
                            stray_approach_reward = progress * (si_scale * 20.0) * isolation_multiplier
                            stray_penalty += stray_approach_reward
            
            # Cap the overall stray penalty to avoid catastrophic policy updates
            stray_penalty = max(-15.0, stray_penalty)

        return farthest_progress, stray_penalty

    def _lane_crowding_penalty(self, inputs: RewardInputs) -> float:
        target = inputs.target_position
        if target is None or not inputs.dog_positions or not inputs.sheep_positions:
            return 0.0
        active_sheep = [
            sheep
            for sheep in inputs.sheep_positions
            if _distance(sheep, target) <= self._config.lane_crowding_activation_distance
        ]
        if not active_sheep:
            return 0.0
        blocking_score = 0.0
        for sheep in active_sheep:
            for dog in inputs.dog_positions:
                forward_distance, lateral_distance, lane_length = _lane_projection(
                    sheep,
                    target,
                    dog,
                )
                if forward_distance <= 0.0 or forward_distance >= lane_length:
                    continue
                if lateral_distance >= self._config.lane_crowding_lateral_tolerance:
                    continue
                lateral_factor = 1.0 - min(
                    1.0,
                    lateral_distance / max(0.1, self._config.lane_crowding_lateral_tolerance),
                )
                forward_factor = 1.0 - min(
                    1.0,
                    forward_distance / max(0.1, lane_length),
                )
                blocking_score += 0.35 + 0.65 * lateral_factor * forward_factor
        if blocking_score <= 0.0:
            return 0.0
        return (
            -(blocking_score / max(1, len(active_sheep))) * self._config.lane_crowding_penalty_scale
        )


# ---------------------------------------------------------------------------
# Hierarchical reward system for autonomous neural-dog training
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HierarchicalRewardConfig:
    """Configurable reward terms for the shepherd + neural dog training path.

    Goals
    -----
    Reward *outcomes* and good herding *principles*, not exact positions.
    Weights are exposed so they can be tuned in config or curriculum without
    touching code.  Disabling a term sets its weight to 0.

    Positive signals
    ----------------
    sheep_closer_to_pen_scale   – reward when average distance to pen decreases
    sheep_penned_reward         – flat reward per newly penned sheep
    flock_grouped_scale         – reward when flock spread decreases (cohesion)
    pressure_from_behind_scale  – dogs are behind flock relative to pen → reward
    dog_spread_scale            – dogs are spread out (not stacked) → reward
    dog_blocking_escape_scale   – dogs between sheep and escape direction
    task_completion_reward      – bonus for penning all sheep
    speed_bonus_scale           – per-step bonus ∝ (1 - t/max_t) when succeeding

    Negative signals (penalties)
    ----------------------------
    scatter_penalty_scale       – flock spread increases
    overpressure_penalty_scale  – any dog is too close (< overpressure_distance) to a sheep
    gate_blocking_penalty_scale – any dog is directly in front of the pen entrance
    dog_stack_penalty_scale     – dogs are bunched together (< stack_distance apart)
    sheep_away_from_pen_scale   – average flock distance to pen increases
    wandering_penalty           – time penalty (encourages task completion)
    timeout_penalty             – terminal penalty on timeout
    """

    # Positive
    sheep_closer_to_pen_scale: float = 2.5
    sheep_penned_reward: float = 8.0
    flock_grouped_scale: float = 0.4
    pressure_from_behind_scale: float = 0.8
    dog_spread_scale: float = 0.3
    dog_blocking_escape_scale: float = 0.2
    task_completion_reward: float = 25.0
    speed_bonus_scale: float = 0.0  # disabled by default; enable for speed curriculum

    # Negative
    scatter_penalty_scale: float = 0.6
    overpressure_penalty_scale: float = 1.2
    gate_blocking_penalty_scale: float = 1.0
    dog_stack_penalty_scale: float = 0.5
    sheep_away_from_pen_scale: float = 0.5
    wandering_penalty: float = 0.04
    timeout_penalty: float = 15.0

    # Thresholds
    overpressure_distance: float = 1.5
    dog_stack_distance: float = 2.0
    gate_block_distance: float = 3.0
    pressure_from_behind_min_cosine: float = 0.3


@dataclass(frozen=True, slots=True)
class HierarchicalRewardBreakdown:
    """Named reward components for a single hierarchical environment step."""

    sheep_closer_to_pen: float = 0.0
    sheep_penned: float = 0.0
    flock_grouped: float = 0.0
    pressure_from_behind: float = 0.0
    dog_spread: float = 0.0
    dog_blocking_escape: float = 0.0
    task_completion: float = 0.0
    speed_bonus: float = 0.0
    scatter_penalty: float = 0.0
    overpressure_penalty: float = 0.0
    gate_blocking_penalty: float = 0.0
    dog_stack_penalty: float = 0.0
    sheep_away_from_pen: float = 0.0
    wandering_penalty: float = 0.0
    timeout_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HierarchicalRewardInputs:
    """Inputs consumed by HierarchicalRewardComputer.

    All positions are (x, y) float tuples.  Distances are grid units.
    """

    previous_average_distance_to_pen: float
    current_average_distance_to_pen: float
    previous_flock_spread: float
    current_flock_spread: float
    newly_penned: int
    success: bool
    timeout: bool
    # (x, y) positions of all dogs this step
    dog_positions: tuple[Position, ...]
    # (x, y) positions of unpenned sheep this step
    sheep_positions: tuple[Position, ...]
    # pen gate center – used for gate-blocking check
    pen_gate_x: float
    pen_gate_y: float
    # flock centroid this step
    flock_centroid: Position
    # vector from flock centroid to pen center
    flock_to_pen_dx: float
    flock_to_pen_dy: float
    # current step / max steps, for speed bonus
    step_fraction: float = 0.0
    # Positions of dogs explicitly assigned the BLOCKER role.  These dogs are
    # near the gate by design; exempting them avoids self-defeating penalties.
    blocker_positions: tuple[Position, ...] = field(default_factory=tuple)


def _cos_similarity(ax: float, ay: float, bx: float, by: float) -> float:
    """Cosine similarity between vectors (ax, ay) and (bx, by)."""
    na = hypot(ax, ay)
    nb = hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))


class HierarchicalRewardComputer:
    """Compute a structured reward for the autonomous neural dog training path.

    Design goals
    ------------
    - Reward herding *outcomes* (sheep moving toward pen, penned) heavily.
    - Reward good herding *principles* (dogs behind flock, dogs spread) lightly.
    - Penalise bad practices (overpressure, gate blocking, stacking) moderately.
    - Do NOT tell dogs exactly where to stand.  The neural policy must learn positions.
    """

    def __init__(self, config: HierarchicalRewardConfig) -> None:
        self._cfg = config

    def compute(self, inputs: HierarchicalRewardInputs) -> HierarchicalRewardBreakdown:
        """Compute the hierarchical reward breakdown for one team step."""
        cfg = self._cfg

        # --- Progress: flock moving toward pen ---
        dist_delta = (
            inputs.previous_average_distance_to_pen - inputs.current_average_distance_to_pen
        )
        if dist_delta > 0:
            sheep_closer = dist_delta * cfg.sheep_closer_to_pen_scale
            sheep_away = 0.0
        else:
            sheep_closer = 0.0
            sheep_away = max(-10.0, -abs(dist_delta) * cfg.sheep_away_from_pen_scale)

        # --- Sheep penned ---
        sheep_penned = inputs.newly_penned * cfg.sheep_penned_reward

        # --- Flock cohesion ---
        spread_delta = inputs.previous_flock_spread - inputs.current_flock_spread
        if spread_delta > 0:
            flock_grouped = spread_delta * cfg.flock_grouped_scale
            scatter = 0.0
        else:
            flock_grouped = 0.0
            scatter = max(-10.0, -abs(spread_delta) * cfg.scatter_penalty_scale)

        # --- Dogs behind flock relative to pen ---
        # A dog is "behind" when its vector from flock centroid is roughly
        # *opposite* to the flock→pen vector; cosine < -threshold = behind.
        pressure_from_behind = 0.0
        if inputs.dog_positions and cfg.pressure_from_behind_scale > 0.0:
            fc_x, fc_y = inputs.flock_centroid
            for dog in inputs.dog_positions:
                dog_dx = dog[0] - fc_x
                dog_dy = dog[1] - fc_y
                # Behind means dog is on the opposite side from pen.
                cosine = _cos_similarity(
                    dog_dx, dog_dy, inputs.flock_to_pen_dx, inputs.flock_to_pen_dy
                )
                if cosine < -cfg.pressure_from_behind_min_cosine:
                    pressure_from_behind += abs(cosine) - cfg.pressure_from_behind_min_cosine
            n_dogs = max(1, len(inputs.dog_positions))
            pressure_from_behind = pressure_from_behind / n_dogs * cfg.pressure_from_behind_scale

        # --- Dog spread: avoid stacking ---
        dog_spread_score = 0.0
        dog_stack_penalty = 0.0
        if len(inputs.dog_positions) > 1:
            pairs = [
                _distance(inputs.dog_positions[i], inputs.dog_positions[j])
                for i in range(len(inputs.dog_positions))
                for j in range(i + 1, len(inputs.dog_positions))
            ]
            avg_pair_dist = sum(pairs) / len(pairs)
            # Reward spread (capped at 20 grid units for normalisation)
            dog_spread_score = min(20.0, avg_pair_dist) / 20.0 * cfg.dog_spread_scale
            # Penalise stacking
            stacked_pairs = sum(1 for d in pairs if d < cfg.dog_stack_distance)
            dog_stack_penalty = -stacked_pairs * cfg.dog_stack_penalty_scale
            dog_stack_penalty = max(-5.0 * cfg.dog_stack_penalty_scale, dog_stack_penalty)

        # --- Dog blocking escape (light positive) ---
        # A dog earns a small bonus when it is positioned between the flock
        # and the edge opposite the pen (i.e., plugging an escape lane).
        dog_blocking_escape = 0.0
        if cfg.dog_blocking_escape_scale > 0.0 and inputs.dog_positions:
            fc_x, fc_y = inputs.flock_centroid
            # Escape direction is opposite pen direction.
            esc_dx = -inputs.flock_to_pen_dx
            esc_dy = -inputs.flock_to_pen_dy
            for dog in inputs.dog_positions:
                dog_dx = dog[0] - fc_x
                dog_dy = dog[1] - fc_y
                cosine = _cos_similarity(dog_dx, dog_dy, esc_dx, esc_dy)
                if cosine > 0.5:
                    dog_blocking_escape += cosine - 0.5
            n_dogs = max(1, len(inputs.dog_positions))
            dog_blocking_escape = dog_blocking_escape / n_dogs * cfg.dog_blocking_escape_scale

        # --- Overpressure: dogs too close to sheep ---
        overpressure = 0.0
        for dog in inputs.dog_positions:
            for sheep in inputs.sheep_positions:
                if _distance(dog, sheep) < cfg.overpressure_distance:
                    overpressure -= cfg.overpressure_penalty_scale
        overpressure = max(-10.0 * cfg.overpressure_penalty_scale, overpressure)

        # --- Gate blocking: dog in front of pen opening ---
        # Exempt any dog that is explicitly assigned the BLOCKER role: those
        # dogs are near the gate by design and should not be penalised for it.
        gate_blocking = 0.0
        blocker_set = set(inputs.blocker_positions)
        for dog in inputs.dog_positions:
            if dog in blocker_set:
                continue
            d = hypot(dog[0] - inputs.pen_gate_x, dog[1] - inputs.pen_gate_y)
            if d < cfg.gate_block_distance:
                # Weight by proximity: closer = worse
                gate_blocking -= (
                    1.0 - d / cfg.gate_block_distance
                ) * cfg.gate_blocking_penalty_scale
        gate_blocking = max(-5.0 * cfg.gate_blocking_penalty_scale, gate_blocking)

        # --- Task completion + speed bonus ---
        task_completion = cfg.task_completion_reward if inputs.success else 0.0
        speed_bonus = 0.0
        if inputs.success and cfg.speed_bonus_scale > 0.0:
            speed_bonus = (1.0 - inputs.step_fraction) * cfg.speed_bonus_scale

        # --- Time penalty + timeout ---
        wandering = -cfg.wandering_penalty
        timeout = -cfg.timeout_penalty if inputs.timeout else 0.0

        total = (
            sheep_closer
            + sheep_penned
            + flock_grouped
            + pressure_from_behind
            + dog_spread_score
            + dog_blocking_escape
            + task_completion
            + speed_bonus
            + scatter
            + overpressure
            + gate_blocking
            + dog_stack_penalty
            + sheep_away
            + wandering
            + timeout
        )
        total = max(-15.0, total)
        return HierarchicalRewardBreakdown(
            sheep_closer_to_pen=sheep_closer,
            sheep_penned=sheep_penned,
            flock_grouped=flock_grouped,
            pressure_from_behind=pressure_from_behind,
            dog_spread=dog_spread_score,
            dog_blocking_escape=dog_blocking_escape,
            task_completion=task_completion,
            speed_bonus=speed_bonus,
            scatter_penalty=scatter,
            overpressure_penalty=overpressure,
            gate_blocking_penalty=gate_blocking,
            dog_stack_penalty=dog_stack_penalty,
            sheep_away_from_pen=sheep_away,
            wandering_penalty=wandering,
            timeout_penalty=timeout,
            total=total,
        )
