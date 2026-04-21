"""Regression tests for reward calculation."""

from __future__ import annotations

from sheepdog.config import RewardConfig
from sheepdog.rewards import RewardComputer, RewardInputs


def test_reward_favors_real_progress() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=12.0,
            current_average_distance=10.5,
            previous_flock_spread=4.0,
            current_flock_spread=3.6,
            newly_penned=1,
            no_progress_step=False,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
        )
    )

    assert breakdown.progress_to_pen > 0
    assert breakdown.sheep_penned > 0
    assert breakdown.total == sum(
        [
            breakdown.progress_to_pen,
            breakdown.sheep_penned,
            breakdown.flock_cohesion,
            -breakdown.scatter_penalty,
            breakdown.time_penalty,
            breakdown.no_progress_penalty,
            breakdown.wall_pressure_penalty,
            breakdown.wait_penalty,
            breakdown.terminal_success,
            breakdown.terminal_failure,
        ]
    )


def test_reward_penalizes_passive_no_progress_waiting() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=10.0,
            current_average_distance=10.0,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=True,
            touched_wall=False,
            waited_without_reason=True,
            terminated=False,
            timeout=False,
            success=False,
        )
    )

    assert breakdown.total < 0
    assert breakdown.wait_penalty < 0
    assert breakdown.no_progress_penalty < 0


def test_reward_penalizes_scattering() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=10.0,
            current_average_distance=10.0,
            previous_flock_spread=2.0,
            current_flock_spread=3.5,
            newly_penned=0,
            no_progress_step=False,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
        )
    )

    assert breakdown.scatter_penalty > 0
