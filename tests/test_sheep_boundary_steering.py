"""Tests for sheep boundary steering, role target positioning, and wall stall metrics."""

from __future__ import annotations

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.entities import DogState, Pen, Point, SheepState
from sheepdog.environment import SheepdogEnvironment
from sheepdog.team_strategy import TeamStrategy


def test_sheep_boundary_steering_inward_recovery() -> None:
    """Verify that sheep on top wall (50, 0) chooses inward recovery to (50, 1) when dog is at (50, 3)."""
    config = LabConfig()
    env = SheepdogEnvironment(config)
    env.reset(seed=42)

    sheep = SheepState(0, Point(50, 0))
    dogs = [DogState(0, Point(50, 3)), DogState(1, Point(40, 20)), DogState(2, Point(60, 20))]
    env._sheep = [sheep]
    env._dogs = dogs

    next_pos = env._sheep_step(sheep.position, [d.position for d in dogs], None, 0, sheep=sheep)

    # Sheep must step INWARD away from top wall
    assert next_pos == Point(50, 1)


def test_sheep_boundary_steering_all_four_walls() -> None:
    """Verify true inward recovery across top, bottom, left, and right walls."""
    config = LabConfig()
    env = SheepdogEnvironment(config)
    w, h = env.env_config.width, env.env_config.height

    # Top Wall
    sheep_top = SheepState(0, Point(50, 0))
    next_top = env._sheep_step(Point(50, 0), [Point(50, 4)], None, 0, sheep=sheep_top)
    assert next_top.y > 0

    # Bottom Wall
    sheep_bot = SheepState(0, Point(50, h - 1))
    next_bot = env._sheep_step(Point(50, h - 1), [Point(50, h - 5)], None, 0, sheep=sheep_bot)
    assert next_bot.y < h - 1

    # Left Wall
    sheep_left = SheepState(0, Point(0, 39))
    next_left = env._sheep_step(Point(0, 39), [Point(4, 39)], None, 0, sheep=sheep_left)
    assert next_left.x > 0

    # Right Wall
    sheep_right = SheepState(0, Point(w - 1, 39))
    next_right = env._sheep_step(Point(w - 1, 39), [Point(w - 5, 39)], None, 0, sheep=sheep_right)
    assert next_right.x < w - 1


def test_sheep_steering_fallback_when_inward_blocked() -> None:
    """Verify that sheep falls back to lateral movement when inward cell is occupied by a dog."""
    config = LabConfig()
    env = SheepdogEnvironment(config)
    env.reset(seed=42)

    sheep = SheepState(0, Point(50, 0))
    # Dog 0 occupies inward cell (50, 1), Dog 1 at (50, 2)
    dogs = [DogState(0, Point(50, 1)), DogState(1, Point(50, 2)), DogState(2, Point(60, 20))]
    env._sheep = [sheep]
    env._dogs = dogs

    next_pos = env._sheep_step(sheep.position, [d.position for d in dogs], None, 0, sheep=sheep)

    # Inward cell (50, 1) is blocked by dog; sheep must fall back to lateral step (49, 0) or (51, 0)
    assert next_pos.y == 0
    assert next_pos.x in (49, 51)


def test_role_targets_open_field_safety_near_walls() -> None:
    """Verify that role targets near walls are kept in the open field and do not clamp to boundary."""
    w, h = 108, 78
    config = EnvironmentConfig(width=w, height=h)
    strategy = TeamStrategy(w, h, config)

    sheep = [SheepState(0, Point(50, 0))]
    dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]
    pen = Pen(Point(88, 58), 14, 14, "bottom")

    assignments, _ = strategy.assign_roles(dogs, sheep, pen)

    for assign in assignments.values():
        target = assign.target
        # Targets must be at least 2 units away from boundary
        assert target.x >= 2 and target.x <= w - 3
        assert target.y >= 2 and target.y <= h - 3
        # Target must be on open-field side of sheep (y >= 2 for sheep at y=0)
        assert target.y >= sheep[0].position.y + 2


def test_wall_stall_diagnostic_metrics() -> None:
    """Verify wall-stall step tracking and inward recovery metrics."""
    config = LabConfig()
    env = SheepdogEnvironment(config)
    env.reset(seed=42)

    # Place sheep at top wall (50, 0)
    env._sheep = [SheepState(0, Point(50, 0))]
    env._dogs = [DogState(0, Point(50, 1)), DogState(1, Point(45, 5)), DogState(2, Point(55, 5))]

    env.step(["wait", "wait", "wait"])
    assert env._wall_stall_steps == 1
    assert env._max_wall_stall_steps >= 1
