"""Reward calculation for herding progress."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sheepdog.config import RewardConfig


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
    terminal_success: float = 0.0
    terminal_failure: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardInputs:
    """Minimal set of values used to score a step."""

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
        terminal_success = self._config.terminal_success_reward if inputs.success else 0.0
        terminal_failure = -self._config.terminal_failure_penalty if inputs.timeout else 0.0
        total = (
            progress_to_pen
            + sheep_penned
            + flock_cohesion
            - scatter_penalty
            + time_penalty
            + no_progress_penalty
            + wall_pressure_penalty
            + wait_penalty
            + terminal_success
            + terminal_failure
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
            terminal_success=terminal_success,
            terminal_failure=terminal_failure,
            total=total,
        )
