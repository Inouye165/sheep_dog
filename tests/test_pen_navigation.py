"""Entrance-route regressions, including the Stage 33 seed-67 fence trap."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from sheepdog.config import EnvironmentConfig, InstinctRewardConfig, LabConfig, RewardConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.scenarios import scenario_from_snapshot
from sheepdog.pen_navigation import pen_navigation
from sheepdog.rewards import RewardComputer, RewardInputs
from sheepdog.team_strategy import TeamStrategy


def fence_environment() -> SheepdogEnvironment:
    """Build a portable fixture from the failure geometry, not generated artifacts."""
    config = LabConfig(
        environment=EnvironmentConfig(
            width=144, height=104, dogs=3, sheep=6, no_progress_window=760,
        ),
        rewards=RewardConfig(instincts=InstinctRewardConfig(enable_instinct_rewards=True)),
    )
    env = SheepdogEnvironment(config)
    env.reset(seed=1234)
    env._pen = Pen(Point(128, 60), 14, 14, "left")
    env._fence_cells = env.pen.fence_cells()
    env._sheep = [
        SheepState(0, Point(132, 58), personality="pen_shy"),
        *(SheepState(i, Point(128, 60 + i), penned=True) for i in range(1, 6)),
    ]
    env._dogs = [
        DogState(0, Point(130, 52)),
        DogState(1, Point(133, 50)),
        DogState(2, Point(127, 53)),
    ]
    env.invalidate_role_assignments()
    env._previous_farthest_distance = env._farthest_distance_to_pen()
    return env


@pytest.mark.parametrize(
    ("opening", "start", "away"),
    [
        ("left", Point(16, 8), Point(17, 8)),
        ("right", Point(14, 8), Point(13, 8)),
        ("top", Point(8, 16), Point(8, 17)),
        ("bottom", Point(8, 14), Point(8, 13)),
    ],
)
def test_routes_go_around_closed_sides(opening: str, start: Point, away: Point) -> None:
    pen = Pen(Point(10, 10), 10, 10, opening)
    route = pen_navigation(pen, 32, 32)
    assert route.distance(away) > route.distance(start)
    target = route.target(start)
    assert target != pen.center
    assert route.clear_segment(start, target)
    assert route.distance(target) < route.distance(start)


def test_seed67_wrong_way_progress_and_correct_detour() -> None:
    env = fence_environment()
    route = pen_navigation(env.pen, 144, 104)
    assert route.distance(Point(132, 58)) == 8
    assert route.distance(Point(133, 58)) == 9
    assert route.distance(Point(131, 58)) == 7
    assert route.target(Point(132, 58)) == Point(127, 58)
    assert route.target(Point(127, 58)) == Point(127, 60)
    assert route.target(Point(127, 60)) == env.pen.center
    assert route.distance(Point(128, 60)) == 0


def test_route_rejects_fence_and_unreachable_cells() -> None:
    route = pen_navigation(Pen(Point(10, 10), 5, 5, "left"), 24, 24)
    with pytest.raises(ValueError, match="No route"):
        route.distance(Point(12, 9))
    sealed = pen_navigation(Pen(Point(0, 5), 5, 5, "left"), 24, 24)
    with pytest.raises(ValueError, match="No route"):
        sealed.distance(Point(12, 5))


@pytest.mark.parametrize(
    ("opening", "start", "direction"),
    [
        ("left", Point(59, 40), Point(1, 0)),
        ("right", Point(62, 40), Point(-1, 0)),
        ("top", Point(60, 39), Point(0, 1)),
        ("bottom", Point(60, 42), Point(0, -1)),
    ],
)
@pytest.mark.parametrize("surging", [False, True])
def test_multi_move_sheep_stop_at_first_pen_entry(
    opening: str, start: Point, direction: Point, surging: bool,
) -> None:
    env = fence_environment()
    env.env_config = replace(
        env.env_config, sheep_speed=3.0, sheep_surge_multiplier=2.0,
    )
    env._pen = Pen(Point(60, 40), 2, 2, opening)
    env._fence_cells = env.pen.fence_cells()
    env._sheep = [SheepState(0, start)]
    if surging:
        env._sheep.append(SheepState(1, env.pen.center, penned=True))
    for dog in env.dogs:
        dog.position = Point(dog.index, 0)
    env.invalidate_role_assignments()

    def score(
        _position: Point, candidate: Point, _dogs: list[Point], _vx: float, _vy: float,
    ) -> float:
        return float(candidate.x * direction.x + candidate.y * direction.y)

    with (
        patch.object(env, "_score_sheep_candidate", side_effect=score),
        patch.object(env._rng, "random", return_value=0.99),
    ):
        snapshot, _ = env.step(["wait"] * 3)
    sheep = env.sheep[0]
    assert sheep.position == Point(start.x + direction.x, start.y + direction.y)
    assert sheep.penned
    assert sheep.position not in env._fence_cells
    assert sheep.movement_budget == 0
    assert snapshot.success


def test_sheep_cannot_step_from_pen_interior_onto_fence() -> None:
    env = fence_environment()
    interior = Point(141, 67)
    fence = Point(142, 67)
    assert env.pen.contains(interior)
    assert fence in env._fence_cells
    assert env._sheep_move_blocked(interior, fence, [], set())


def test_invalid_route_error_has_layout_context_and_target_also_rejects_it() -> None:
    route = pen_navigation(Pen(Point(10, 10), 5, 5, "left"), 24, 24)
    with pytest.raises(ValueError, match=r"field=24x24, on_fence=True"):
        route.distance(Point(12, 9))
    with pytest.raises(ValueError, match="No route"):
        route.target(Point(-1, 12))


def test_direct_approach_and_cache_preserve_open_field_targets() -> None:
    pen = Pen(Point(20, 10), 8, 8, "left")
    route = pen_navigation(pen, 32, 32)
    assert route.target(Point(5, 14)) == pen.center
    assert route is pen_navigation(pen, 32, 32)
    assert route is not pen_navigation(replace(pen, opening="right"), 32, 32)


def test_role_targets_push_left_around_fence_not_down_into_it() -> None:
    env = fence_environment()
    strategy = TeamStrategy(144, 104, env.env_config)
    assignments, _ = strategy.assign_roles(env.dogs, env.sheep, env.pen)
    for role in (DogRole.REAR_PRESSURE, DogRole.COLLECTOR):
        target = next(a.target for a in assignments.values() if a.role == role)
        assert target.x > 132
        assert target.y == 58
        assert target not in env._fence_cells


def test_controlled_state_covers_actual_sheep_influence_range() -> None:
    env = fence_environment()
    assert env._is_controlled_state(Point(132, 58))
    for dog in env.dogs:
        dog.position = Point(dog.index, 0)
    assert not env._is_controlled_state(Point(132, 58))


@pytest.mark.parametrize(("next_x", "sign"), [(133, -1), (131, 1)])
def test_environment_rewards_legal_route_progress(next_x: int, sign: int) -> None:
    env = fence_environment()

    def move() -> None:
        env.sheep[0].position = Point(next_x, 58)

    with patch.object(env, "_move_sheep", side_effect=move):
        _, reward = env.step(["wait"] * 3)
    assert reward.progress_to_pen * sign > 0
    assert reward.gate_progress * sign > 0
    assert env._cumulative_gate_progress * sign > 0
    assert reward.target_progress * sign > 0
    assert reward.farthest_sheep_progress * sign >= 0


def test_penning_near_sheep_does_not_fake_regression_for_remaining_sheep() -> None:
    env = fence_environment()
    env.sheep[1].penned = False
    env.sheep[1].position = Point(127, 60)

    def enter() -> None:
        env.sheep[1].position = Point(128, 60)
        env.sheep[1].penned = True

    with patch.object(env, "_move_sheep", side_effect=enter):
        _, reward = env.step(["wait"] * 3)
    assert reward.sheep_penned > 0
    assert reward.progress_to_pen > 0
    assert reward.target_progress > 0


def test_fence_loop_loses_position_bonuses_and_activates_wrong_hold() -> None:
    env = fence_environment()
    threshold = env.env_config.stalled_control_activation_steps
    progress_total = 0.0
    for _step in range(threshold + 12):
        with patch.object(env, "_move_sheep"):
            _, reward = env.step(["wait"] * 3)
        progress_total += reward.progress_to_pen
    assert progress_total == 0
    assert env._entry_stall_steps > threshold
    assert reward.pressure_zone <= 0
    assert reward.safe_pressure <= 0
    assert reward.wrong_hold_penalty < 0
    assert reward.stalled_control_penalty < 0

    with patch.object(env, "_move_sheep"):
        env.reset(seed=1234)
    assert env._entry_stall_steps == 0
    assert env._best_entry_distance is None


def test_repeated_reversals_cannot_reset_best_entry_progress() -> None:
    env = fence_environment()
    threshold = env.env_config.stalled_control_activation_steps
    for step in range(threshold + 20):
        next_position = Point(133 if step % 2 == 0 else 132, 58)

        def move(position: Point = next_position) -> None:
            env.sheep[0].position = position

        with patch.object(env, "_move_sheep", side_effect=move):
            _, reward = env.step(["wait"] * 3)
    assert env._entry_stall_steps == threshold + 20
    assert env._no_progress_steps == threshold + 20
    assert reward.pressure_zone <= 0
    assert reward.wrong_hold_penalty < 0


def test_new_best_entry_progress_clears_sustained_stall() -> None:
    env = fence_environment()
    for _ in range(env.env_config.stalled_control_activation_steps + 2):
        with patch.object(env, "_move_sheep"):
            env.step(["wait"] * 3)
    assert env._entry_stall_steps > env.env_config.stalled_control_activation_steps
    previous_best = env._best_entry_distance
    assert previous_best is not None
    previous_no_progress = env._no_progress_steps

    def move() -> None:
        env.sheep[0].position = Point(131, 58)

    with patch.object(env, "_move_sheep", side_effect=move):
        _, reward = env.step(["wait"] * 3)
    assert env._entry_stall_steps == 0
    assert env._best_entry_distance is not None
    assert env._best_entry_distance < previous_best
    assert reward.progress_to_pen > 0
    assert env._no_progress_steps == previous_no_progress - 1


def test_scenario_reset_clears_entry_stall_history() -> None:
    env = fence_environment()
    scenario = scenario_from_snapshot(
        name="fence regression", seed=67, snapshot=env.get_state_snapshot().to_dict()
    )
    with patch.object(env, "_move_sheep"):
        env.step(["wait"] * 3)
    assert env._best_entry_distance is not None
    assert env._entry_stall_steps > 0
    env.reset_from_scenario(scenario)
    assert env._best_entry_distance is None
    assert env._entry_stall_steps == 0


def test_corridor_does_not_include_wrong_side_of_closed_fence() -> None:
    env = fence_environment()
    assert env._is_in_gate_corridor(Point(126, 67))
    assert not env._is_in_gate_corridor(Point(140, 67))


def test_closed_route_loop_has_zero_signed_progress_reward() -> None:
    env = fence_environment()
    progress_total = 0.0
    gate_total = 0.0
    for position in (Point(133, 58), Point(134, 58), Point(133, 58), Point(132, 58)):
        def move(next_position: Point = position) -> None:
            env.sheep[0].position = next_position

        with patch.object(env, "_move_sheep", side_effect=move):
            _, reward = env.step(["wait"] * 3)
        progress_total += reward.progress_to_pen
        gate_total += reward.gate_progress
    assert progress_total == pytest.approx(0)
    assert gate_total == pytest.approx(0)


def test_flock_centroid_on_fence_does_not_break_role_assignment() -> None:
    env = fence_environment()
    env.sheep[1].penned = False
    env.sheep[1].position = Point(132, 60)
    assert env._flock_center() in env._fence_cells
    assignments, _ = TeamStrategy(144, 104, env.env_config).assign_roles(
        env.dogs, env.sheep, env.pen
    )
    assert len(assignments) == 3
    assert env._navigation_anchor() not in env._fence_cells


def test_reward_suppresses_only_positive_position_bonuses_during_stalls() -> None:
    computer = RewardComputer(
        RewardConfig(instincts=InstinctRewardConfig(enable_instinct_rewards=True))
    )
    inputs = RewardInputs(
        previous_average_distance=8, current_average_distance=8,
        previous_flock_spread=0, current_flock_spread=0, newly_penned=0,
        no_progress_step=True, touched_wall=False, waited_without_reason=False,
        terminated=False, timeout=False, success=False,
        dog_positions=((14.0, 10.0),), flock_centroid=(10.0, 10.0),
        target_position=(0.0, 10.0), entry_stall_steps=20,
    )
    stalled = computer.compute(inputs)
    assert stalled.pressure_zone == 0
    assert stalled.safe_pressure == 0
    wrong_side = computer.compute(replace(inputs, target_position=(20.0, 10.0)))
    assert wrong_side.pressure_zone < 0
    assert computer.compute(replace(inputs, entry_stall_steps=0)).pressure_zone > 0


def test_observation_shape_and_bounds_remain_checkpoint_compatible() -> None:
    env = fence_environment()
    obs = env.build_observation_for_dog(0)
    assert len(obs.values) == 54
    assert all(-1 <= value <= 1 for value in obs.values)
