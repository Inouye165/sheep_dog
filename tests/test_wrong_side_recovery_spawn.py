"""Tests for the wrong_side_recovery spawn mode and spawn determinism."""

from __future__ import annotations

import math

from sheepdog.config import SPAWN_MODES, EnvironmentConfig, LabConfig
from sheepdog.entities import Point
from sheepdog.environment import SheepdogEnvironment


def _make_env(spawn_mix: dict[str, float]) -> SheepdogEnvironment:
    env_cfg = EnvironmentConfig(
        width=108,
        height=78,
        dogs=3,
        sheep=4,
        pen_width=14,
        pen_height=14,
        pen_opening="bottom",
        max_steps=1040,
        curriculum_stage=8,
        spawn_mix=spawn_mix,
    )
    return SheepdogEnvironment(LabConfig(environment=env_cfg))


def test_wrong_side_recovery_in_spawn_modes() -> None:
    """wrong_side_recovery is registered in SPAWN_MODES."""
    assert "wrong_side_recovery" in SPAWN_MODES


def test_wrong_side_recovery_valid_diverse_non_overlapping_across_50_seeds() -> None:
    """Across 50 seeds, dogs spawn on the pen side of the flock and all entities are valid."""
    env = _make_env({"wrong_side_recovery": 1.0})

    flock_centers: set[tuple[int, int]] = set()
    dog_position_tuples: set[tuple[tuple[int, int], ...]] = set()

    for seed in range(1, 51):
        snapshot = env.reset(seed=seed)
        pen_center = env.pen.center
        fence_cells = env.pen.fence_cells()
        assert env._spawn_mode == "wrong_side_recovery"
        assert len(snapshot.dogs) == 3
        assert len(snapshot.sheep) == 4

        all_coords = [(pt.x, pt.y) for pt in (*snapshot.dogs, *snapshot.sheep)]
        # Mutually non-overlapping
        assert len(set(all_coords)) == len(all_coords)

        # Inside field bounds, outside pen and fences
        for x, y in all_coords:
            pt = Point(x, y)
            assert 0 <= x < env.env_config.width
            assert 0 <= y < env.env_config.height
            assert not env.pen.contains(pt)
            assert pt not in fence_cells

        flock_x = sum(pt.x for pt in snapshot.sheep) / len(snapshot.sheep)
        flock_y = sum(pt.y for pt in snapshot.sheep) / len(snapshot.sheep)
        flock_dist_to_pen = math.hypot(flock_x - pen_center.x, flock_y - pen_center.y)
        assert flock_dist_to_pen >= 28.0  # center >= 30 with +-2 jitter

        for dog_pt in snapshot.dogs:
            dog_dist_to_pen = math.hypot(dog_pt.x - pen_center.x, dog_pt.y - pen_center.y)
            assert dog_dist_to_pen < flock_dist_to_pen

        flock_centers.add((round(flock_x), round(flock_y)))
        dog_position_tuples.add(tuple((pt.x, pt.y) for pt in snapshot.dogs))

    # Diverse across 50 seeds
    assert len(flock_centers) >= 25
    assert len(dog_position_tuples) >= 25


def test_existing_spawn_modes_deterministic_and_unchanged() -> None:
    """All existing SPAWN_MODES remain deterministic across repeated resets."""
    existing_modes = [m for m in SPAWN_MODES if m != "wrong_side_recovery"]
    for mode in existing_modes:
        env_a = _make_env({mode: 1.0})
        env_b = _make_env({mode: 1.0})
        snap_a1 = env_a.reset(seed=42)
        snap_a2 = env_a.reset(seed=42)
        snap_b = env_b.reset(seed=42)

        assert env_a._spawn_mode == mode
        assert snap_a1.dogs == snap_a2.dogs == snap_b.dogs
        assert snap_a1.sheep == snap_a2.sheep == snap_b.sheep

        all_positions = [*snap_a1.dogs, *snap_a1.sheep]
        assert len(set(all_positions)) == len(all_positions)
