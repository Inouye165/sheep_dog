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


def test_stray_approach_reward_for_isolated_stray() -> None:
    config = RewardConfig(stray_ignore_penalty_scale=0.01)
    computer = RewardComputer(config)

    # 1. Moving closer to isolated stray -> positive approach reward should reduce the penalty
    closer_inputs = RewardInputs(
        previous_average_distance=10.0,
        current_average_distance=10.0,
        previous_flock_spread=2.0,
        current_flock_spread=2.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        # Farthest sheep is at (30.0, 10.0), target/pen is at (0.0, 10.0) -> current_farthest_distance = 30.0
        current_farthest_distance=30.0,
        previous_farthest_distance=30.0,
        # Flock centroid is at (10.0, 10.0) -> farthest sheep is isolated (dist = 20.0 > 8.0)
        flock_centroid=(10.0, 10.0),
        target_position=(0.0, 10.0),
        sheep_positions=((30.0, 10.0), (10.0, 10.0)),
        # Dog moved from (25.0, 10.0) to (27.0, 10.0) -> closer by 2.0 units
        previous_dog_positions=((25.0, 10.0),),
        dog_positions=((27.0, 10.0),),
    )

    # 2. Moving away from isolated stray -> negative progress should increase the penalty
    further_inputs = RewardInputs(
        previous_average_distance=10.0,
        current_average_distance=10.0,
        previous_flock_spread=2.0,
        current_flock_spread=2.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        current_farthest_distance=30.0,
        previous_farthest_distance=30.0,
        flock_centroid=(10.0, 10.0),
        target_position=(0.0, 10.0),
        sheep_positions=((30.0, 10.0), (10.0, 10.0)),
        # Dog moved from (27.0, 10.0) to (25.0, 10.0) -> further by 2.0 units
        previous_dog_positions=((27.0, 10.0),),
        dog_positions=((25.0, 10.0),),
    )

    breakdown_closer = computer.compute(closer_inputs)
    breakdown_further = computer.compute(further_inputs)

    assert breakdown_closer.stray_ignore_penalty > breakdown_further.stray_ignore_penalty


def test_stray_approach_reward_for_single_remaining_unpenned_sheep() -> None:
    config = RewardConfig(stray_ignore_penalty_scale=0.01)
    computer = RewardComputer(config)

    # When 3 sheep are penned and only 1 remains far from pen and dogs
    lone_sheep = (40.0, 10.0)
    target = (0.0, 10.0)

    # Dog moves closer to the lone sheep
    closer_inputs = RewardInputs(
        previous_average_distance=40.0,
        current_average_distance=40.0,
        previous_flock_spread=0.0,
        current_flock_spread=0.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        current_farthest_distance=40.0,
        previous_farthest_distance=40.0,
        flock_centroid=lone_sheep,
        target_position=target,
        sheep_positions=(lone_sheep,),
        previous_dog_positions=((10.0, 10.0),),
        dog_positions=((15.0, 10.0),),
    )

    # Dog moves away from the lone sheep
    further_inputs = RewardInputs(
        previous_average_distance=40.0,
        current_average_distance=40.0,
        previous_flock_spread=0.0,
        current_flock_spread=0.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        current_farthest_distance=40.0,
        previous_farthest_distance=40.0,
        flock_centroid=lone_sheep,
        target_position=target,
        sheep_positions=(lone_sheep,),
        previous_dog_positions=((15.0, 10.0),),
        dog_positions=((10.0, 10.0),),
    )

    breakdown_closer = computer.compute(closer_inputs)
    breakdown_further = computer.compute(further_inputs)

    assert breakdown_closer.stray_ignore_penalty > breakdown_further.stray_ignore_penalty


def test_stray_approach_reward_for_multi_sheep_unpenned_cluster() -> None:
    config = RewardConfig(stray_ignore_penalty_scale=0.01)
    computer = RewardComputer(config)

    # When 2 sheep are unpenned and together across the arena (e.g. at (52, 35) and (58, 35)),
    # target/pen is at (0, 0), and dogs are near the pen at (5, 5).
    target = (0.0, 0.0)
    sheep1 = (52.0, 35.0)
    sheep2 = (58.0, 35.0)
    flock_centroid = (55.0, 35.0)  # dist_to_centroid is 3.0 (<8.0, not a single outlier), but max_dist is ~67.8 (>10.0)

    # Dog moves from (5.0, 5.0) to (10.0, 10.0) towards the sheep
    closer_inputs = RewardInputs(
        previous_average_distance=60.0,
        current_average_distance=60.0,
        previous_flock_spread=3.0,
        current_flock_spread=3.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        current_farthest_distance=67.8,
        previous_farthest_distance=67.8,
        flock_centroid=flock_centroid,
        target_position=target,
        sheep_positions=(sheep1, sheep2),
        previous_dog_positions=((5.0, 5.0),),
        dog_positions=((10.0, 10.0),),
    )

    # Dog moves from (10.0, 10.0) to (5.0, 5.0) away from the sheep (e.g. retreating to pen)
    further_inputs = RewardInputs(
        previous_average_distance=60.0,
        current_average_distance=60.0,
        previous_flock_spread=3.0,
        current_flock_spread=3.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
        current_farthest_distance=67.8,
        previous_farthest_distance=67.8,
        flock_centroid=flock_centroid,
        target_position=target,
        sheep_positions=(sheep1, sheep2),
        previous_dog_positions=((10.0, 10.0),),
        dog_positions=((5.0, 5.0),),
    )

    breakdown_closer = computer.compute(closer_inputs)
    breakdown_further = computer.compute(further_inputs)

    assert breakdown_closer.stray_ignore_penalty > breakdown_further.stray_ignore_penalty


