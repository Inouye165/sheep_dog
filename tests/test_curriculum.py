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
