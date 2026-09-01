"""Unit and integration tests for spatial analytics and stage bottleneck diagnostics."""

from __future__ import annotations

from pathlib import Path

from sheepdog.entities import Pen, Point
from sheepdog.training.episode_store import EpisodeStore
from sheepdog.training.spatial_analytics import (
    ZONE_BOTTOM_LEFT,
    ZONE_BOTTOM_RIGHT,
    ZONE_BOTTOM_WALL,
    ZONE_CENTER,
    ZONE_LEFT_WALL,
    ZONE_RIGHT_WALL,
    ZONE_TOP_LEFT,
    ZONE_TOP_RIGHT,
    ZONE_TOP_WALL,
    SpatialEpisodeTracker,
    classify_field_zone,
    classify_flock_zone,
    classify_pen_zone,
    diagnose_stage_bottlenecks,
)


def test_classify_field_zone_corners() -> None:
    width, height = 60.0, 40.0
    # Corner threshold is 25% (x < 15, x >= 45, y < 10, y >= 30)
    assert classify_field_zone(5.0, 5.0, width, height) == ZONE_TOP_LEFT
    assert classify_field_zone(50.0, 5.0, width, height) == ZONE_TOP_RIGHT
    assert classify_field_zone(5.0, 35.0, width, height) == ZONE_BOTTOM_LEFT
    assert classify_field_zone(50.0, 35.0, width, height) == ZONE_BOTTOM_RIGHT


def test_classify_field_zone_walls_and_center() -> None:
    width, height = 60.0, 40.0
    assert classify_field_zone(30.0, 5.0, width, height) == ZONE_TOP_WALL
    assert classify_field_zone(30.0, 35.0, width, height) == ZONE_BOTTOM_WALL
    assert classify_field_zone(5.0, 20.0, width, height) == ZONE_LEFT_WALL
    assert classify_field_zone(50.0, 20.0, width, height) == ZONE_RIGHT_WALL
    assert classify_field_zone(30.0, 20.0, width, height) == ZONE_CENTER


def test_classify_flock_zone_centroid() -> None:
    width, height = 60.0, 40.0
    sheep_points = [Point(2, 2), Point(4, 4), Point(3, 3)]
    assert classify_flock_zone(sheep_points, width, height) == ZONE_TOP_LEFT

    scattered_points = [Point(2, 2), Point(58, 38)]
    assert classify_flock_zone(scattered_points, width, height) == ZONE_CENTER


def test_classify_pen_zone() -> None:
    width, height = 60.0, 45.0
    pen_tl = Pen(Point(0, 0), 12, 12, opening="bottom")
    assert classify_pen_zone(pen_tl, width, height) == ZONE_TOP_LEFT

    pen_tr = Pen(Point(48, 0), 12, 12, opening="bottom")
    assert classify_pen_zone(pen_tr, width, height) == ZONE_TOP_RIGHT


def test_spatial_episode_tracker_lifecycle() -> None:
    tracker = SpatialEpisodeTracker(field_width=60.0, field_height=40.0)
    pen = Pen(Point(48, 0), 12, 12, opening="bottom")

    # Start with sheep in Top-Left corner
    tracker.initialize(
        sheep_positions=[Point(5, 5)],
        dog_positions=[Point(30, 20)],
        pen=pen,
        spawn_mode="fixed_easy",
    )
    assert tracker.initial_sheep_zone == ZONE_TOP_LEFT
    assert tracker.pen_zone == ZONE_TOP_RIGHT

    # Sheep stays in top left for 3 steps
    for _ in range(3):
        tracker.record_step([Point(5, 5)])

    # Sheep extracted to center field for 5 steps
    for _ in range(5):
        tracker.record_step([Point(30, 20)])

    summary = tracker.get_summary(success=True, timeout=False, stopped=False)
    assert summary["initial_sheep_zone"] == ZONE_TOP_LEFT
    assert summary["final_sheep_zone"] == ZONE_CENTER
    assert summary["corner_entered"] is True
    assert summary["corner_extracted"] is True
    assert summary["corner_stuck_at_end"] is False
    assert summary["corner_steps_total"] == 3
    assert summary["corner_time_pct"] == round(3 / 8, 4)


def test_spatial_episode_tracker_stuck_in_corner() -> None:
    tracker = SpatialEpisodeTracker(field_width=60.0, field_height=40.0)
    pen = Pen(Point(48, 0), 12, 12, opening="bottom")

    tracker.initialize(
        sheep_positions=[Point(5, 35)],
        dog_positions=[Point(30, 20)],
        pen=pen,
        spawn_mode="corner_cluster",
    )
    for _ in range(10):
        tracker.record_step([Point(5, 35)])

    summary = tracker.get_summary(success=False, timeout=True, stopped=False)
    assert summary["initial_sheep_zone"] == ZONE_BOTTOM_LEFT
    assert summary["final_sheep_zone"] == ZONE_BOTTOM_LEFT
    assert summary["corner_stuck_at_end"] is True
    assert summary["corner_entered"] is True
    assert summary["corner_extracted"] is False


def test_diagnose_stage_bottlenecks_corner_and_axis_bias() -> None:
    zone_stats = {
        ZONE_CENTER: {"total": 20, "wins": 18, "trapped_at_end": 0},
        ZONE_TOP_LEFT: {"total": 15, "wins": 2, "trapped_at_end": 12},
        ZONE_BOTTOM_LEFT: {"total": 10, "wins": 1, "trapped_at_end": 8},
        ZONE_TOP_RIGHT: {"total": 15, "wins": 14, "trapped_at_end": 0},
        ZONE_BOTTOM_RIGHT: {"total": 15, "wins": 13, "trapped_at_end": 1},
        ZONE_TOP_WALL: {"total": 5, "wins": 4, "trapped_at_end": 0},
        ZONE_BOTTOM_WALL: {"total": 5, "wins": 4, "trapped_at_end": 0},
        ZONE_LEFT_WALL: {"total": 8, "wins": 1, "trapped_at_end": 6},
        ZONE_RIGHT_WALL: {"total": 8, "wins": 7, "trapped_at_end": 0},
    }
    pen_stats = {
        ZONE_TOP_RIGHT: {"total": 50, "wins": 45},
        ZONE_TOP_LEFT: {"total": 50, "wins": 15},
    }
    setup_stats = {
        "fixed_easy": {"total": 60, "wins": 50},
        "corner_cluster": {"total": 40, "wins": 8},
    }

    insights = diagnose_stage_bottlenecks(
        stage=3,
        total_episodes=100,
        zone_stats=zone_stats,
        pen_stats=pen_stats,
        setup_stats=setup_stats,
    )

    insight_types = {i["type"] for i in insights}
    assert "corner_entrapment" in insight_types
    assert "corner_bias" in insight_types
    assert "axis_bias" in insight_types
    assert "pen_placement" in insight_types
    assert "setup_failure" in insight_types


def test_episode_store_stage_diagnostics_integration(tmp_path: Path) -> None:
    db_file = tmp_path / "test-telemetry.sqlite"
    store = EpisodeStore(db_path=db_file)

    # Record 10 episodes in Stage 2
    # 5 episodes with sheep in top_left (all timeouts/failures)
    for i in range(5):
        store.add_episode({
            "global_environment_episode": i + 1,
            "curriculum_stage": 2,
            "reward": 10.0,
            "penned": 0,
            "total_sheep": 1,
            "success": False,
            "status": "TIMEOUT",
            "seed": 100 + i,
            "pen_zone": ZONE_TOP_RIGHT,
            "spawn_mode": "corner_cluster",
            "initial_sheep_zone": ZONE_TOP_LEFT,
            "final_sheep_zone": ZONE_TOP_LEFT,
            "corner_time_pct": 0.85,
            "wall_time_pct": 0.15,
            "corner_stuck_at_end": 1,
        })

    # 5 episodes with sheep in center (all successes)
    for i in range(5):
        store.add_episode({
            "global_environment_episode": i + 6,
            "curriculum_stage": 2,
            "reward": 150.0,
            "penned": 1,
            "total_sheep": 1,
            "success": True,
            "status": "SUCCESS",
            "seed": 200 + i,
            "pen_zone": ZONE_TOP_RIGHT,
            "spawn_mode": "fixed_easy",
            "initial_sheep_zone": ZONE_CENTER,
            "final_sheep_zone": ZONE_TOP_RIGHT,
            "corner_time_pct": 0.05,
            "wall_time_pct": 0.10,
            "corner_stuck_at_end": 0,
        })

    store.flush()

    diagnostics = store.get_stage_diagnostics(stage=2)
    assert diagnostics["curriculum_stage"] == 2
    assert diagnostics["total_episodes"] == 10
    assert diagnostics["success_count"] == 5
    assert diagnostics["overall_success_rate"] == 0.5
    assert diagnostics["corner_stuck_count"] == 5

    # Check zone stats
    assert diagnostics["zone_stats"][ZONE_TOP_LEFT]["total"] == 5
    assert diagnostics["zone_stats"][ZONE_TOP_LEFT]["win_rate"] == 0.0
    assert diagnostics["zone_stats"][ZONE_TOP_LEFT]["trapped_at_end"] == 5

    assert diagnostics["zone_stats"][ZONE_CENTER]["total"] == 5
    assert diagnostics["zone_stats"][ZONE_CENTER]["win_rate"] == 1.0

    # Check setup stats
    assert diagnostics["setup_stats"]["corner_cluster"]["win_rate"] == 0.0
    assert diagnostics["setup_stats"]["fixed_easy"]["win_rate"] == 1.0

    # Verify insights were generated
    assert len(diagnostics["insights"]) > 0
    assert any(i["type"] == "corner_entrapment" for i in diagnostics["insights"])
