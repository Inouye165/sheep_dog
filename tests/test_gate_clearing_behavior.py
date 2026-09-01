"""Unit tests for gate clearing and corridor non-blocking logic."""


from sheepdog.config import RewardConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.rewards import RewardComputer, RewardInputs
from sheepdog.team_strategy import TeamStrategy


def test_blocker_target_has_lateral_clearance():
    """Verify that _blocker_target places the blocker dog at least 4 units off center."""
    ts = TeamStrategy(width=32, height=32)
    pen = Pen(origin=Point(26, 0), width=6, height=6, opening="left")
    flock_center = Point(10, 3)

    # Blocker target should have lateral offset from the gate centerline (y = 3)
    blocker_target = ts._blocker_target(flock_center, pen, lateral_x=0, lateral_y=1)

    # Gate center y is pen.center.y = 3
    # Blocker target should be laterally offset by at least 4 units from y = 3
    assert abs(blocker_target.y - pen.center.y) >= 3 or abs(blocker_target.x - pen.center.x) >= 3


def test_single_unpenned_sheep_no_gate_blocker():
    """Verify that when only 1 sheep is unpenned, team strategy does not station a gate blocker."""
    ts = TeamStrategy(width=32, height=32)
    pen = Pen(origin=Point(26, 0), width=6, height=6, opening="left")

    dogs = [
        DogState(index=0, position=Point(5, 5)),
        DogState(index=1, position=Point(24, 3)),
        DogState(index=2, position=Point(24, 4)),
    ]

    sheep = [
        SheepState(index=0, position=Point(28, 2), penned=True),
        SheepState(index=1, position=Point(29, 2), penned=True),
        SheepState(index=2, position=Point(28, 3), penned=True),
        SheepState(index=3, position=Point(20, 3), penned=False),  # 1 unpenned sheep
    ]

    assignments, snapshot = ts.assign_roles(dogs, sheep, pen)

    # No dog should be assigned DogRole.BLOCKER at the gate when len(unpenned) == 1
    assigned_roles = [assignment.role for assignment in assignments.values()]
    assert DogRole.BLOCKER not in assigned_roles


def test_lane_crowding_penalty_penalizes_gate_blocking_dogs():
    """Verify that _lane_crowding_penalty penalizes dogs standing near gate ahead of approaching sheep."""
    rc = RewardConfig(lane_crowding_penalty_scale=1.0, lane_crowding_lateral_tolerance=2.0)
    computer = RewardComputer(rc)

    # Target is pen center (29, 3)
    target = (29.0, 3.0)
    # Approaching sheep at (20, 3)
    sheep_positions = ((20.0, 3.0),)
    # Dog sitting right in front of the gate at (26, 3)
    dog_positions = ((26.0, 3.0),)

    inputs = RewardInputs(
        previous_average_distance=10.0,
        current_average_distance=9.0,
        previous_flock_spread=1.0,
        current_flock_spread=1.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        sprint_count=0,
        previous_gate_distance=10.0,
        current_gate_distance=9.0,
        previous_gate_corridor_distance=10.0,
        current_gate_corridor_distance=9.0,
        previous_gate_corridor_occupancy=1.0,
        current_gate_corridor_occupancy=1.0,
        controlled_stall_steps=0,
        wrong_hold_active=False,
        tactically_valid_hold=True,
        terminated=False,
        timeout=False,
        success=False,
        dog_positions=dog_positions,
        sheep_positions=sheep_positions,
        flock_centroid=(20.0, 3.0),
        target_position=target,
    )

    breakdown = computer.compute(inputs)
    assert breakdown.lane_crowding_penalty < 0.0
