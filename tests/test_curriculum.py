"""Tests for the curriculum staging helpers."""

# pylint: disable=missing-function-docstring
from __future__ import annotations

from sheepdog.config import LabConfig
from sheepdog.curriculum import (
    CURRICULUM_STAGES,
    apply_curriculum_stage,
    available_stages,
    stage_summary,
)


def test_available_stages_are_sorted_and_non_empty() -> None:
    stages = available_stages()
    assert stages
    assert list(stages) == sorted(stages)


def test_stage_one_is_simpler_than_stage_three() -> None:
    base = LabConfig()

    stage_one = apply_curriculum_stage(base, 1).environment
    stage_three = apply_curriculum_stage(base, 3).environment

    assert stage_one.dogs <= stage_three.dogs
    assert stage_one.sheep <= stage_three.sheep
    assert stage_one.width <= stage_three.width
    assert stage_one.height <= stage_three.height
    assert stage_one.max_steps <= stage_three.max_steps


def test_stage_one_uses_single_dog_single_sheep() -> None:
    stage_one = apply_curriculum_stage(LabConfig(), 1).environment

    assert stage_one.dogs == 1
    assert stage_one.sheep == 1
    assert stage_one.dog_speed == 2
    assert stage_one.sheep_speed == 1
    assert stage_one.width >= 60
    assert stage_one.height >= 45


def test_stage_two_dog_speed_advantage() -> None:
    stage_two = apply_curriculum_stage(LabConfig(), 2).environment

    assert stage_two.dogs == 1
    assert stage_two.sheep == 3
    assert stage_two.dog_speed == 2


def test_unknown_stage_returns_original_config() -> None:
    base = LabConfig()

    result = apply_curriculum_stage(base, 0)

    assert result.environment == base.environment


def test_each_stage_has_a_summary() -> None:
    for stage in CURRICULUM_STAGES:
        assert stage_summary(stage)


def test_stages_6_7_8_sheep_placements() -> None:
    from sheepdog.environment import SheepdogEnvironment

    # --- Stage 6 ---
    config = apply_curriculum_stage(LabConfig(), 6)
    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    sheep_positions = [s.position for s in env.sheep]
    assert len(sheep_positions) == 6
    stray_found = False
    for i, p1 in enumerate(sheep_positions):
        others = [p2 for j, p2 in enumerate(sheep_positions) if j != i]
        closest_dist = min(p1.distance_to(p2) for p2 in others)
        max_group_dist = max(o1.distance_to(o2) for o1 in others for o2 in others)
        if closest_dist >= 12.0 and max_group_dist <= 8.5:
            stray_found = True
            break
    assert stray_found, "Stage 6 should have 1 group of 5 sheep and 1 stray far away"

    # --- Stage 7 ---
    config = apply_curriculum_stage(LabConfig(), 7)
    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    sheep_positions = [s.position for s in env.sheep]
    assert len(sheep_positions) == 6
    strays_count = 0
    for i, p1 in enumerate(sheep_positions):
        others = [p2 for j, p2 in enumerate(sheep_positions) if j != i]
        closest_dist = min(p1.distance_to(p2) for p2 in others)
        if closest_dist >= 8.0:
            strays_count += 1
    assert strays_count >= 3, f"Stage 7 should have 3 isolated strays, got {strays_count}"

    # --- Stage 8 ---
    config = apply_curriculum_stage(LabConfig(), 8)
    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    sheep_positions = [s.position for s in env.sheep]
    assert len(sheep_positions) == 6
    for i, p1 in enumerate(sheep_positions):
        others = [p2 for j, p2 in enumerate(sheep_positions) if j != i]
        closest_dist = min(p1.distance_to(p2) for p2 in others)
        assert closest_dist >= 8.0, f"Stage 8 sheep should be spaced out, but closest is {closest_dist}"
