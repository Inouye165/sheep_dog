"""Unit tests for Change 3: wall_pressure_penalty uses pre-move blocked-action detection."""

from __future__ import annotations

import pytest

from sheepdog.config import EnvironmentConfig, LabConfig
from sheepdog.environment import Action, Point, SheepdogEnvironment

_DIR_TO_ACTION: dict[tuple[int, int], Action] = {
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up",
}


def _make_env() -> SheepdogEnvironment:
    cfg = LabConfig(
        environment=EnvironmentConfig(
            dogs=2,
            sheep=1,
            width=20,
            height=20,
            dog_speed=1.0,
            max_steps=40,
        )
    )
    env = SheepdogEnvironment(cfg)
    env.reset(seed=7)
    for dog in env._dogs:
        dog.movement_budget = 0.0
    return env


def _find_open_point(env: SheepdogEnvironment, preferred: Point) -> Point:
    if (
        0 <= preferred.x < env.env_config.width
        and 0 <= preferred.y < env.env_config.height
        and preferred not in env._fence_cells
        and not env._pen.contains(preferred)
    ):
        return preferred
    for y in range(2, env.env_config.height - 2):
        for x in range(2, env.env_config.width - 2):
            cand = Point(x, y)
            if cand not in env._fence_cells and not env._pen.contains(cand):
                return cand
    raise RuntimeError("No open point found in test environment.")


def _find_fence_approach(env: SheepdogEnvironment) -> tuple[Point, Point, Action]:
    """Return (start_2_away, adjacent_to_fence, action_toward_fence)."""
    for fc in sorted(env._fence_cells, key=lambda p: (p.y, p.x)):
        for (dx, dy), action in _DIR_TO_ACTION.items():
            adj = Point(fc.x - dx, fc.y - dy)
            start = Point(fc.x - 2 * dx, fc.y - 2 * dy)
            if not (
                0 <= adj.x < env.env_config.width
                and 0 <= adj.y < env.env_config.height
                and 0 <= start.x < env.env_config.width
                and 0 <= start.y < env.env_config.height
            ):
                continue
            if (
                adj not in env._fence_cells
                and start not in env._fence_cells
                and not env._pen.contains(adj)
                and not env._pen.contains(start)
            ):
                return start, adj, action
    raise RuntimeError("No unobstructed fence approach found in test environment.")


@pytest.mark.parametrize("obstacle", ["wall", "fence", "sheep", "teammate"])
def test_legal_move_ending_adjacent_to_obstacle_has_zero_wall_penalty(obstacle: str) -> None:
    env = _make_env()
    # Park sheep and teammate away from test area by default.
    env._sheep[0].position = _find_open_point(env, Point(15, 15))
    env._sheep[0].penned = False
    env._dogs[1].position = _find_open_point(env, Point(15, 12))

    if obstacle == "wall":
        env._dogs[0].position = Point(1, 5)
        action: Action = "left"
    elif obstacle == "fence":
        start, _adj, action = _find_fence_approach(env)
        env._dogs[0].position = start
    elif obstacle == "sheep":
        sheep_pos = _find_open_point(env, Point(10, 10))
        env._sheep[0].position = sheep_pos
        env._dogs[0].position = Point(sheep_pos.x - 2, sheep_pos.y)
        action = "right"
    else:
        mate_pos = _find_open_point(env, Point(10, 10))
        env._dogs[1].position = mate_pos
        env._dogs[0].position = Point(mate_pos.x - 2, mate_pos.y)
        action = "right"

    assert env.project_dog_action(0, action) != env._dogs[0].position
    _, breakdown = env.step([action, "wait"])
    assert breakdown.wall_pressure_penalty == 0.0


@pytest.mark.parametrize("obstacle", ["wall", "fence", "sheep", "teammate"])
def test_blocked_move_into_obstacle_applies_wall_penalty(obstacle: str) -> None:
    env = _make_env()
    env._sheep[0].position = _find_open_point(env, Point(15, 15))
    env._sheep[0].penned = False
    env._dogs[1].position = _find_open_point(env, Point(15, 12))

    if obstacle == "wall":
        env._dogs[0].position = Point(0, 5)
        action: Action = "left"
    elif obstacle == "fence":
        _start, adj, action = _find_fence_approach(env)
        env._dogs[0].position = adj
    elif obstacle == "sheep":
        sheep_pos = _find_open_point(env, Point(10, 10))
        env._sheep[0].position = sheep_pos
        env._dogs[0].position = Point(sheep_pos.x - 1, sheep_pos.y)
        action = "right"
    else:
        mate_pos = _find_open_point(env, Point(10, 10))
        env._dogs[1].position = mate_pos
        env._dogs[0].position = Point(mate_pos.x - 1, mate_pos.y)
        action = "right"

    assert env.project_dog_action(0, action) == env._dogs[0].position
    _, breakdown = env.step([action, "wait"])
    assert breakdown.wall_pressure_penalty == -env.config.rewards.wall_pressure_penalty
