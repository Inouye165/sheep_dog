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
            breakdown.sprint_cost,
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


def test_gate_progress_reward_is_positive_when_flock_moves_toward_opening() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=10.0,
            current_average_distance=9.5,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=False,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
            previous_gate_distance=8.0,
            current_gate_distance=5.5,
            previous_gate_corridor_distance=3.0,
            current_gate_corridor_distance=1.5,
            previous_gate_corridor_occupancy=0.0,
            current_gate_corridor_occupancy=0.5,
        )
    )

    assert breakdown.gate_progress > 0
    assert breakdown.gate_corridor_progress > 0
    assert breakdown.gate_alignment > 0


def test_sprint_cost_penalizes_high_speed_moves() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=10.0,
            current_average_distance=9.5,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=False,
            touched_wall=False,
            waited_without_reason=False,
            sprint_count=2,
            terminated=False,
            timeout=False,
            success=False,
        )
    )

    assert breakdown.sprint_cost < 0


def test_lane_crowding_penalty_triggers_when_dog_blocks_sheep_path_to_pen() -> None:
    breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=8.0,
            current_average_distance=8.0,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=True,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
            dog_positions=((17.0, 10.0),),
            sheep_positions=((14.0, 10.0),),
            flock_centroid=(14.0, 10.0),
            previous_flock_centroid=(14.0, 10.0),
            target_position=(20.0, 10.0),
        )
    )

    assert breakdown.lane_crowding_penalty < 0


def test_wrong_hold_penalty_triggers_only_for_stalled_control() -> None:
    penalty_breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=8.0,
            current_average_distance=8.0,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=True,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
            controlled_stall_steps=7,
            wrong_hold_active=True,
        )
    )
    valid_hold_breakdown = RewardComputer(RewardConfig()).compute(
        RewardInputs(
            previous_average_distance=4.0,
            current_average_distance=4.0,
            previous_flock_spread=2.0,
            current_flock_spread=2.0,
            newly_penned=0,
            no_progress_step=True,
            touched_wall=False,
            waited_without_reason=False,
            terminated=False,
            timeout=False,
            success=False,
            controlled_stall_steps=3,
            wrong_hold_active=True,
            tactically_valid_hold=True,
            current_gate_corridor_occupancy=0.5,
        )
    )

    assert penalty_breakdown.stalled_control_penalty < 0
    assert penalty_breakdown.wrong_hold_penalty < 0
    assert valid_hold_breakdown.stalled_control_penalty == 0.0
    assert valid_hold_breakdown.wrong_hold_penalty == 0.0
