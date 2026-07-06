"""Tests for the curriculum staging helpers."""

# pylint: disable=missing-function-docstring
from __future__ import annotations

from dataclasses import replace

from sheepdog.config import LabConfig
from sheepdog.curriculum import (
    CURRICULUM_STAGES,
    apply_curriculum_stage,
    available_stages,
    stage_summary,
)
from sheepdog.environment import SheepdogEnvironment


def test_available_stages_are_sorted_and_non_empty() -> None:
    stages = available_stages()
    assert stages
    assert list(stages) == sorted(stages)
    assert stages[0] == 1
    assert stages[-1] == 38
    assert len(stages) == 38


def test_stage_one_is_simpler_than_stage_eight() -> None:
    base = LabConfig()

    stage_one = apply_curriculum_stage(base, 1).environment
    stage_eight = apply_curriculum_stage(base, 8).environment

    assert stage_one.dogs <= stage_eight.dogs
    assert stage_one.sheep <= stage_eight.sheep
    assert stage_one.width <= stage_eight.width
    assert stage_one.height <= stage_eight.height
    assert stage_one.max_steps <= stage_eight.max_steps


def test_stage_one_uses_single_dog_single_sheep() -> None:
    stage_one = apply_curriculum_stage(LabConfig(), 1).environment

    assert stage_one.dogs == 1
    assert stage_one.sheep == 1
    assert stage_one.dog_speed == 1.75
    assert stage_one.sheep_speed == 1
    assert stage_one.width >= 60
    assert stage_one.height >= 45
    assert stage_one.spawn_mix == {"fixed_easy": 1.0}


def test_stage_two_dog_speed_advantage() -> None:
    stage_two = apply_curriculum_stage(LabConfig(), 2).environment

    assert stage_two.dogs == 1
    assert stage_two.sheep == 1
    assert stage_two.dog_speed > stage_two.sheep_speed
    assert stage_two.spawn_mix == {"fixed_easy": 0.7, "randomized_flock": 0.3}


def test_unknown_stage_returns_original_config() -> None:
    base = LabConfig()

    result = apply_curriculum_stage(base, 0)

    assert result.environment == base.environment


def test_each_stage_has_a_summary() -> None:
    for stage in CURRICULUM_STAGES:
        assert stage_summary(stage)


def test_curriculum_includes_stage_21() -> None:
    assert 21 in CURRICULUM_STAGES


def test_curriculum_includes_stage_25() -> None:
    assert 25 in CURRICULUM_STAGES


def test_curriculum_includes_stage_26() -> None:
    assert 26 in CURRICULUM_STAGES


def test_curriculum_includes_stage_27() -> None:
    assert 27 in CURRICULUM_STAGES


def test_curriculum_includes_stage_28() -> None:
    assert 28 in CURRICULUM_STAGES


def test_curriculum_includes_stage_29() -> None:
    assert 29 in CURRICULUM_STAGES


def test_curriculum_includes_stage_30() -> None:
    assert 30 in CURRICULUM_STAGES


def test_curriculum_includes_stage_31() -> None:
    assert 31 in CURRICULUM_STAGES


def test_curriculum_includes_stage_32() -> None:
    assert 32 in CURRICULUM_STAGES


def test_curriculum_includes_stage_33() -> None:
    assert 33 in CURRICULUM_STAGES


def test_curriculum_includes_stage_38() -> None:
    assert 38 in CURRICULUM_STAGES


def test_stage_9_enables_collection_progress_signals() -> None:
    stage_eight = apply_curriculum_stage(LabConfig(), 8)
    stage_nine = apply_curriculum_stage(LabConfig(), 9)

    assert stage_nine.environment.count_collection_progress is True
    assert stage_nine.environment.no_progress_window > stage_eight.environment.no_progress_window


def test_stage_9_enables_collection_reward_shaping() -> None:
    stage_eight = apply_curriculum_stage(LabConfig(), 8)
    stage_nine = apply_curriculum_stage(LabConfig(), 9)

    assert (
        stage_nine.rewards.farthest_sheep_progress_scale
        > stage_eight.rewards.farthest_sheep_progress_scale
    )
    assert (
        stage_nine.rewards.stray_ignore_penalty_scale
        > stage_eight.rewards.stray_ignore_penalty_scale
    )
    assert stage_nine.rewards.flock_cohesion_scale > stage_eight.rewards.flock_cohesion_scale
    assert stage_nine.rewards.scatter_penalty_scale > stage_eight.rewards.scatter_penalty_scale


def test_stage_15_spawn_mix_matches_expected_blend() -> None:
    config = apply_curriculum_stage(LabConfig(), 15)
    mix = config.environment.spawn_mix
    assert mix == {
        "randomized_flock": 0.35,
        "nearby_stray": 0.3,
        "farther_stray": 0.25,
        "split_flock": 0.1,
    }


def test_stage_27_uses_random_pen_placement() -> None:
    config = apply_curriculum_stage(LabConfig(), 27)
    assert config.environment.pen_placement == "random"


def test_stage_30_uses_all_corners_spawn_mode() -> None:
    config = apply_curriculum_stage(LabConfig(), 30)
    assert config.environment.spawn_mix == {"all_corners": 1.0}


def test_stage_31_keeps_personality_bias_disabled() -> None:
    config = apply_curriculum_stage(LabConfig(), 31)
    assert config.environment.sheep_personality_override == ""
    assert config.environment.sheep_personality_strength == 0.0


def test_stage_30_spawns_sheep_near_field_corners() -> None:
    config = apply_curriculum_stage(LabConfig(), 30)
    env = SheepdogEnvironment(config)
    env.reset(seed=123)

    width = config.environment.width
    height = config.environment.height
    margin = max(2, min(width, height) // 12)
    corner_candidates = (
        (margin, margin),
        (width - 1 - margin, margin),
        (margin, height - 1 - margin),
        (width - 1 - margin, height - 1 - margin),
    )

    for sheep in env.sheep:
        nearest_corner_dist = min(
            ((sheep.position.x - cx) ** 2 + (sheep.position.y - cy) ** 2) ** 0.5
            for cx, cy in corner_candidates
        )
        assert nearest_corner_dist <= 3.0


def test_stage_32_spawned_sheep_are_not_all_pen_fearful() -> None:
    config = apply_curriculum_stage(LabConfig(), 32)
    env = SheepdogEnvironment(config)
    env.reset(seed=321)

    assert any(sheep.personality != "pen_fearful" for sheep in env.sheep)


def test_stage_32_to_38_phase_personality_and_grouping_changes() -> None:
    stage_32 = apply_curriculum_stage(LabConfig(), 32)
    stage_33 = apply_curriculum_stage(LabConfig(), 33)
    stage_34 = apply_curriculum_stage(LabConfig(), 34)
    stage_35 = apply_curriculum_stage(LabConfig(), 35)
    stage_36 = apply_curriculum_stage(LabConfig(), 36)
    stage_37 = apply_curriculum_stage(LabConfig(), 37)
    stage_38 = apply_curriculum_stage(LabConfig(), 38)

    assert stage_32.environment.sheep_personality_strength == 0.0
    assert (
        stage_33.environment.sheep_personality_strength
        > stage_32.environment.sheep_personality_strength
    )
    assert (
        stage_34.environment.sheep_personality_strength
        > stage_33.environment.sheep_personality_strength
    )
    assert (
        stage_35.environment.sheep_personality_strength
        == stage_34.environment.sheep_personality_strength
    )
    assert (
        stage_36.environment.sheep_personality_strength
        == stage_35.environment.sheep_personality_strength
    )
    assert (
        stage_37.environment.sheep_personality_strength
        > stage_36.environment.sheep_personality_strength
    )
    assert (
        stage_38.environment.sheep_personality_strength
        > stage_37.environment.sheep_personality_strength
    )

    assert stage_32.environment.sheep_cohere_without_dog_pressure is True
    assert stage_33.environment.sheep_cohere_without_dog_pressure is True
    assert stage_34.environment.sheep_cohere_without_dog_pressure is True
    assert stage_35.environment.sheep_cohere_without_dog_pressure is True
    assert stage_36.environment.sheep_cohere_without_dog_pressure is False
    assert stage_37.environment.sheep_cohere_without_dog_pressure is False
    assert stage_38.environment.sheep_cohere_without_dog_pressure is False

    assert (
        stage_35.environment.sheep_flock_cohesion_weight
        == stage_34.environment.sheep_flock_cohesion_weight
    )
    assert (
        stage_36.environment.sheep_flock_cohesion_weight
        < stage_35.environment.sheep_flock_cohesion_weight
    )
    assert (
        stage_37.environment.sheep_flock_cohesion_weight
        < stage_36.environment.sheep_flock_cohesion_weight
    )
    assert (
        stage_38.environment.sheep_flock_cohesion_weight
        < stage_37.environment.sheep_flock_cohesion_weight
    )


def test_spawn_mode_nearby_stray_places_separated_sheep() -> None:
    config = apply_curriculum_stage(LabConfig(), 9)
    config = replace(
        config,
        environment=replace(config.environment, spawn_mix={"nearby_stray": 1.0}),
    )
    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    sheep_positions = [s.position for s in env.sheep]
    assert len(sheep_positions) == 4
    stray_found = False
    for i, p1 in enumerate(sheep_positions):
        others = [p2 for j, p2 in enumerate(sheep_positions) if j != i]
        closest_dist = min(p1.distance_to(p2) for p2 in others)
        if closest_dist >= config.environment.stray_near_min:
            stray_found = True
            break
    assert stray_found, "Stage 9 should include at least one nearby stray layout"
