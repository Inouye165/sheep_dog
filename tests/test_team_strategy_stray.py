"""Unit tests for TeamStrategy 2-sheep blind spot resolution and instrumentation."""

from __future__ import annotations

from sheepdog.config import LabConfig
from sheepdog.entities import DogRole, DogState, Point, SheepState
from sheepdog.environment import Pen, SheepdogEnvironment
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.diagnostics.temporal_interaction import (
    build_step_instrumentation_payload,
)


def test_stray_sheep_two_sheep_close_returns_none() -> None:
    """When 2 unpenned sheep are clustered within clustering distance (<= 5.0), no stray is designated."""
    strategy = TeamStrategy(width=35, height=21)
    s0 = SheepState(index=0, position=Point(10, 10))
    s1 = SheepState(index=1, position=Point(13, 10))  # Distance = 3.0 (< 5.0 and < 8.0)
    sheep = [s0, s1]
    flock_center = strategy._flock_center(sheep)
    flock_spread = strategy._flock_spread(sheep, flock_center)
    pen = Pen(origin=Point(2, 2), width=6, height=6, opening="bottom")

    stray = strategy._stray_sheep(sheep, flock_center, flock_spread, pen=pen)
    assert stray is None


def test_stray_sheep_two_sheep_separated_designates_further_from_pen() -> None:
    """When 2 unpenned sheep are separated (> 8.0 units), the one further from the pen is designated as stray."""
    strategy = TeamStrategy(width=35, height=21)
    pen = Pen(origin=Point(2, 2), width=6, height=6, opening="bottom")  # pen center is around (5, 5)

    # s0 is near the pen, s1 is far away on the right
    s0 = SheepState(index=0, position=Point(8, 8))
    s1 = SheepState(index=1, position=Point(25, 15))  # Separated from s0 (>8 units) and further from pen
    sheep = [s0, s1]
    flock_center = strategy._flock_center(sheep)
    flock_spread = strategy._flock_spread(sheep, flock_center)

    stray = strategy._stray_sheep(sheep, flock_center, flock_spread, pen=pen)
    assert stray is not None
    assert stray.index == 1  # s1 is further from pen

    # Swap positions so s0 is further from pen
    s0_far = SheepState(index=0, position=Point(28, 18))
    s1_near = SheepState(index=1, position=Point(8, 8))
    sheep2 = [s0_far, s1_near]
    flock_center2 = strategy._flock_center(sheep2)
    flock_spread2 = strategy._flock_spread(sheep2, flock_center2)

    stray2 = strategy._stray_sheep(sheep2, flock_center2, flock_spread2, pen=pen)
    assert stray2 is not None
    assert stray2.index == 0  # s0_far is further from pen


def test_assign_roles_with_two_separated_sheep_activates_collector() -> None:
    """In assign_roles with 2 separated sheep, the stray triggers a COLLECTOR role."""
    strategy = TeamStrategy(width=35, height=21)
    pen = Pen(origin=Point(2, 2), width=6, height=6, opening="bottom")
    dogs = [
        DogState(index=0, position=Point(10, 10)),
        DogState(index=1, position=Point(11, 10)),
        DogState(index=2, position=Point(12, 10)),
    ]
    s0 = SheepState(index=0, position=Point(8, 8))
    s1 = SheepState(index=1, position=Point(25, 15))  # Separated stray
    assignments, snapshot = strategy.assign_roles(dogs, [s0, s1], pen)

    assert snapshot.stray_sheep_index == 1
    roles = [a.role for a in assignments.values()]
    assert DogRole.COLLECTOR in roles


def test_build_step_instrumentation_payload_contains_role_hold_steps() -> None:
    """build_step_instrumentation_payload includes role_hold_steps and role_minimum_hold_steps in strategy_dict."""
    config = LabConfig()
    env = SheepdogEnvironment(config, capture_step_telemetry=True)
    env.reset(seed=42)

    payload = build_step_instrumentation_payload(env)
    assert "strategy" in payload
    strategy = payload["strategy"]
    assert strategy is not None
    assert "role_hold_steps" in strategy
    assert isinstance(strategy["role_hold_steps"], dict)
    assert len(strategy["role_hold_steps"]) == env.dog_count

    for dog_idx in range(env.dog_count):
        assert dog_idx in strategy["role_hold_steps"]
        assert isinstance(strategy["role_hold_steps"][dog_idx], int)

    assert "role_minimum_hold_steps" in strategy


def test_role_stagnation_decay_reduces_assignment_stickiness() -> None:
    """When steps_in_role exceeds 150, stagnation penalty decays the assignment stickiness bonus."""
    strategy = TeamStrategy(width=50, height=50)
    target = Point(20, 20)

    # Dog at distance 15 holding REAR_PRESSURE
    dog_fresh = DogState(index=0, position=Point(5, 20), current_role=DogRole.REAR_PRESSURE, steps_in_role=10)
    cost_fresh = strategy._role_assignment_cost(dog_fresh, target, DogRole.REAR_PRESSURE)

    # Dog holding the role for 150 steps (still full stickiness bonus)
    dog_150 = DogState(index=0, position=Point(5, 20), current_role=DogRole.REAR_PRESSURE, steps_in_role=150)
    cost_150 = strategy._role_assignment_cost(dog_150, target, DogRole.REAR_PRESSURE)
    assert cost_150 == cost_fresh

    # Same dog holding the role for 180 steps (> 150, penalty begins)
    dog_stagnant_180 = DogState(index=0, position=Point(5, 20), current_role=DogRole.REAR_PRESSURE, steps_in_role=180)
    cost_stagnant_180 = strategy._role_assignment_cost(dog_stagnant_180, target, DogRole.REAR_PRESSURE)

    # Same dog holding the role for 210 steps (max 15.0 stagnation penalty)
    dog_stagnant_210 = DogState(index=0, position=Point(5, 20), current_role=DogRole.REAR_PRESSURE, steps_in_role=210)
    cost_stagnant_210 = strategy._role_assignment_cost(dog_stagnant_210, target, DogRole.REAR_PRESSURE)

    # Stagnant cost must be strictly higher than fresh/150 cost (less sticky bonus)
    assert cost_stagnant_180 > cost_fresh
    assert cost_stagnant_210 > cost_stagnant_180


def test_collector_target_boundary_backoff() -> None:
    """When a stray sheep is near a boundary (wall_margin <= 3), collector target backs off into the open field."""
    strategy = TeamStrategy(width=100, height=100)
    flock_center = Point(50, 50)

    # Stray near left wall (x=2 <= 3)
    stray_left = Point(2, 50)
    assert strategy._wall_margin(stray_left) == 2 <= 3
    target_left = strategy._collector_target(stray_left, flock_center)
    # Must back off to the right (away from x=0 boundary)
    assert target_left.x >= 5
    assert target_left.x > stray_left.x

    # Stray in top-left corner (x=1, y=2)
    stray_corner = Point(1, 2)
    assert strategy._wall_margin(stray_corner) == 1 <= 3
    target_corner = strategy._collector_target(stray_corner, flock_center)
    # Must back off away from both left and top boundaries
    assert target_corner.x >= 4
    assert target_corner.y >= 5
    assert target_corner.x > stray_corner.x
    assert target_corner.y > stray_corner.y
