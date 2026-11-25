"""Tests for coherent_wall_recovery and wrong_side_mild progressive recovery spawn modes."""

from __future__ import annotations

import math

from sheepdog.config import SPAWN_MODES, EnvironmentConfig, LabConfig
from sheepdog.entities import Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.heuristic import HeuristicExpertPolicy


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


def test_progressive_recovery_modes_registered() -> None:
    """Both progressive recovery modes are registered in SPAWN_MODES."""
    assert "coherent_wall_recovery" in SPAWN_MODES
    assert "wrong_side_mild" in SPAWN_MODES


def test_coherent_wall_recovery_across_50_seeds() -> None:
    """Across 50 seeds, coherent_wall_recovery spawns a compact flock near one wall/corner."""
    env = _make_env({"coherent_wall_recovery": 1.0})
    flock_centers: set[tuple[int, int]] = set()

    for seed in range(1, 51):
        snapshot = env.reset(seed=seed)
        fence_cells = env.pen.fence_cells()
        assert env._spawn_mode == "coherent_wall_recovery"
        assert len(snapshot.dogs) == 3
        assert len(snapshot.sheep) == 4

        all_coords = [(pt.x, pt.y) for pt in (*snapshot.dogs, *snapshot.sheep)]
        assert len(set(all_coords)) == len(all_coords)

        for x, y in all_coords:
            pt = Point(x, y)
            assert 0 <= x < env.env_config.width
            assert 0 <= y < env.env_config.height
            assert not env.pen.contains(pt)
            assert pt not in fence_cells

        for pt in snapshot.sheep:
            assert not (94 <= pt.x <= 107 and 1 <= pt.y <= 14)

        flock_x = sum(pt.x for pt in snapshot.sheep) / len(snapshot.sheep)
        flock_y = sum(pt.y for pt in snapshot.sheep) / len(snapshot.sheep)
        spread = max(math.hypot(pt.x - flock_x, pt.y - flock_y) for pt in snapshot.sheep)
        assert spread <= 6.0

        wall_margin = min(
            min(
                pt.x,
                env.env_config.width - 1 - pt.x,
                pt.y,
                env.env_config.height - 1 - pt.y,
            )
            for pt in snapshot.sheep
        )
        assert wall_margin <= 5

        flock_centers.add((round(flock_x), round(flock_y)))

    assert len(flock_centers) >= 25


def test_wrong_side_mild_across_50_seeds() -> None:
    """Across 50 seeds, wrong_side_mild places dogs mildly ahead and flanked without overlap."""
    env = _make_env({"wrong_side_mild": 1.0})
    flock_centers: set[tuple[int, int]] = set()
    dog_tuples: set[tuple[tuple[int, int], ...]] = set()

    for seed in range(1, 51):
        snapshot = env.reset(seed=seed)
        pen_center = env.pen.center
        fence_cells = env.pen.fence_cells()
        assert env._spawn_mode == "wrong_side_mild"
        assert len(snapshot.dogs) == 3
        assert len(snapshot.sheep) == 4

        all_coords = [(pt.x, pt.y) for pt in (*snapshot.dogs, *snapshot.sheep)]
        assert len(set(all_coords)) == len(all_coords)

        for x, y in all_coords:
            pt = Point(x, y)
            assert 0 <= x < env.env_config.width
            assert 0 <= y < env.env_config.height
            assert not env.pen.contains(pt)
            assert pt not in fence_cells

        flock_x = sum(pt.x for pt in snapshot.sheep) / len(snapshot.sheep)
        flock_y = sum(pt.y for pt in snapshot.sheep) / len(snapshot.sheep)
        flock_dist = math.hypot(flock_x - pen_center.x, flock_y - pen_center.y)
        assert flock_dist >= 23.0  # center >= 25 with +-2 jitter

        dx = pen_center.x - flock_x
        dy = pen_center.y - flock_y
        norm = math.hypot(dx, dy)
        ux, uy = (dx / norm, dy / norm) if norm > 1e-6 else (1.0, 0.0)
        perp_x, perp_y = -uy, ux

        for dog_pt in snapshot.dogs:
            vx = dog_pt.x - flock_x
            vy = dog_pt.y - flock_y
            forward_proj = vx * ux + vy * uy
            lateral_proj = abs(vx * perp_x + vy * perp_y)
            assert 1.5 <= forward_proj <= 13.5
            assert 1.5 <= lateral_proj <= 11.5

        flock_centers.add((round(flock_x), round(flock_y)))
        dog_tuples.add(tuple((pt.x, pt.y) for pt in snapshot.dogs))

    assert len(flock_centers) >= 25
    assert len(dog_tuples) >= 25


def test_heuristic_expert_solvability_on_progressive_modes() -> None:
    """Verify progressive modes are genuinely solvable by HeuristicExpertPolicy without deadlocks."""
    policy = HeuristicExpertPolicy()

    wall_env = _make_env({"coherent_wall_recovery": 1.0})
    wall_successes = sum(
        1 for seed in range(2016, 2021) if wall_env.run_policy(policy, seed=seed).stats.success
    )
    assert wall_successes >= 1, f"coherent_wall_recovery solved {wall_successes}/5"

    mild_env = _make_env({"wrong_side_mild": 1.0})
    mild_successes = sum(
        1 for seed in range(2021, 2026) if mild_env.run_policy(policy, seed=seed).stats.success
    )
    assert mild_successes >= 3, f"wrong_side_mild solved {mild_successes}/5"


def test_existing_spawn_modes_remain_deterministic() -> None:
    """All pre-existing SPAWN_MODES remain deterministic across resets."""
    existing = [
        m for m in SPAWN_MODES if m not in ("coherent_wall_recovery", "wrong_side_mild")
    ]
    for mode in existing:
        env_a = _make_env({mode: 1.0})
        env_b = _make_env({mode: 1.0})
        snap_a1 = env_a.reset(seed=42)
        snap_a2 = env_a.reset(seed=42)
        snap_b = env_b.reset(seed=42)

        assert env_a._spawn_mode == mode
        assert snap_a1.dogs == snap_a2.dogs == snap_b.dogs
        assert snap_a1.sheep == snap_a2.sheep == snap_b.sheep
