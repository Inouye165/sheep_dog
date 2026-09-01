"""Focused diagnostic unit tests for wall/corner sheep recovery behavior.

Validates:
1. Raw vs clamped role targets near walls.
2. Role targets in corners.
3. Reconstruction of sheep wall distance from observation vector.
4. Availability of retreat and lateral actions in action mask.
5. Complete legal recovery route transition physics.
6. Fully blocked or impossible corner layout detection.
7. Reward totals for scripted recovery vs pinning.
8. Initial spawn distance distribution bounds.
"""

from __future__ import annotations

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import RoleAwareObservationBuilder
from sheepdog.team_strategy import TeamStrategy


def test_raw_vs_clamped_role_targets_near_walls() -> None:
    """Verify that role targets near walls are kept in open field on the open-field side of sheep."""
    w, h = 108, 78
    config = EnvironmentConfig(width=w, height=h)
    strategy = TeamStrategy(w, h, config)

    # Sheep at top wall (50, 0), pen at (88, 58) -> pen direction (dx=+1, dy=+1)
    sheep = [SheepState(0, Point(50, 0))]
    dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]
    pen = Pen(Point(88, 58), 14, 14, "bottom")

    assignments, snapshot = strategy.assign_roles(dogs, sheep, pen)

    rear_assign = next(a for a in assignments.values() if a.role == DogRole.REAR_PRESSURE)
    # Target must be safely in open field (y >= 2) on open-field side of sheep
    assert rear_assign.target.y >= 2
    assert rear_assign.target.x == 46


def test_role_targets_in_corners() -> None:
    """Verify role targets when sheep are in the top-left corner (0, 0) stay in open field."""
    w, h = 108, 78
    config = EnvironmentConfig(width=w, height=h)
    strategy = TeamStrategy(w, h, config)

    sheep = [SheepState(0, Point(0, 0))]
    dogs = [DogState(0, Point(0, 1)), DogState(1, Point(1, 0)), DogState(2, Point(2, 2))]
    pen = Pen(Point(88, 58), 14, 14, "bottom")

    assignments, _ = strategy.assign_roles(dogs, sheep, pen)

    # Targets must stay in open field (x >= 2 and y >= 2)
    for assign in assignments.values():
        t = assign.target
        assert t.x >= 2 and t.y >= 2


def test_reconstruction_of_sheep_wall_distance_from_observation() -> None:
    """Verify exact reconstruction of sheep coordinates and wall distances from obs vector."""
    lab_config = LabConfig()
    env = SheepdogEnvironment(lab_config)
    env.reset(seed=42)
    w, h = env.env_config.width, env.env_config.height

    sheep_pos = Point(50, 0)
    dog_pos = Point(50, 1)
    env._sheep = [SheepState(0, sheep_pos)]
    env._dogs = [DogState(0, dog_pos), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    obs_builder = RoleAwareObservationBuilder()
    obs = obs_builder.build(env, 0)
    feats = obs.as_feature_dict()

    rec_sheep_x = (feats["own_x"] + feats["focus_sheep_dx"]) * (w - 1)
    rec_sheep_y = (feats["own_y"] + feats["focus_sheep_dy"]) * (h - 1)

    assert abs(rec_sheep_x - sheep_pos.x) < 1e-5
    assert abs(rec_sheep_y - sheep_pos.y) < 1e-5

    rec_wall_top = rec_sheep_y
    assert abs(rec_wall_top - sheep_pos.y) < 1e-5


def test_availability_of_retreat_and_lateral_actions() -> None:
    """Verify that action masking permits down, left, right, and wait when dog is at (50, 1)."""
    lab_config = LabConfig()
    env = SheepdogEnvironment(lab_config)
    env.reset(seed=42)

    env._sheep = [SheepState(0, Point(50, 0))]
    env._dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    mask = env.action_mask_for_dog(0, policy_mode="neural_policy")

    assert mask["wait"] is True
    assert mask["down"] is True
    assert mask["left"] is True
    assert mask["right"] is True
    assert mask["sprint_down"] is True
    assert mask["up"] is False  # Blocked by sheep at (50, 0)


def test_complete_legal_recovery_route() -> None:
    """Demonstrate step-by-step physical recovery route moving sheep off wall/corner position."""
    lab_config = LabConfig()
    env = SheepdogEnvironment(lab_config)
    env.reset(seed=42)

    initial_sheep_pos = Point(50, 0)
    env._sheep = [SheepState(0, initial_sheep_pos)]
    env._dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    route = [
        ["sprint_down", "wait", "wait"],  # (50,1) -> (50,3)
        ["sprint_left", "wait", "wait"],  # (50,3) -> (48,3)
        ["sprint_left", "wait", "wait"],  # (48,3) -> (46,3)
        ["up", "wait", "wait"],           # (46,3) -> (46,2)
        ["up", "wait", "wait"],           # (46,2) -> (46,1)
        ["right", "wait", "wait"],        # (46,1) -> (47,1)
        ["right", "wait", "wait"],        # (47,1) -> (48,1)
    ]

    for step_actions in route:
        env.step(step_actions)

    final_sheep = env._sheep[0].position
    # Sheep moved away from initial (50, 0) position via lateral dog maneuvers
    assert final_sheep != initial_sheep_pos


def test_fully_blocked_impossible_layout() -> None:
    """Verify that a corner sheep blocked on all non-wall neighbors cannot move."""
    lab_config = LabConfig()
    env = SheepdogEnvironment(lab_config)
    env.reset(seed=42)

    sheep_pos = Point(0, 0)
    # Dogs block both open adjacent cells (0, 1) and (1, 0)
    env._sheep = [SheepState(0, sheep_pos)]
    env._dogs = [DogState(0, Point(0, 1)), DogState(1, Point(1, 0)), DogState(2, Point(1, 1))]

    env.step(["wait", "wait", "wait"])
    assert env._sheep[0].position == Point(0, 0)


def test_reward_comparison_scripted_recovery_vs_pinning() -> None:
    """Verify reward outputs for recovery trajectory vs pinning trajectory."""
    lab_config = LabConfig()

    env_rec = SheepdogEnvironment(lab_config)
    env_rec.reset(seed=42)
    env_rec._sheep = [SheepState(0, Point(50, 0))]
    env_rec._dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    env_pin = SheepdogEnvironment(lab_config)
    env_pin.reset(seed=42)
    env_pin._sheep = [SheepState(0, Point(50, 0))]
    env_pin._dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    # Step 1: Pinning incurs wall_pressure_penalty (-0.4) because dog presses into wall/sheep
    _, rew_pin_obj = env_pin.step(["sprint_up", "wait", "wait"])
    rew_pin = rew_pin_obj.total

    _, rew_rec_obj = env_rec.step(["sprint_down", "wait", "wait"])
    rew_rec = rew_rec_obj.total

    assert rew_pin < rew_rec  # Pinning gets heavily penalized for wall pressure!


def test_initial_spawn_distance_distribution() -> None:
    """Verify that random spawn center keeps sheep at least 4 cells away from walls."""
    lab_config = LabConfig()
    env = SheepdogEnvironment(lab_config)
    w, h = env.env_config.width, env.env_config.height

    for seed in range(50):
        env.reset(seed=seed)
        for sheep in env._sheep:
            sx, sy = sheep.position.x, sheep.position.y
            assert sx >= 4 and sx <= w - 5
            assert sy >= 4 and sy <= h - 5
