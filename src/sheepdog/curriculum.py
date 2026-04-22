"""Curriculum stages for incremental herding training.

Stages start simple (one dog, one sheep, nearby pen) and become harder so
training does not need to discover everything from scratch. Stages tweak
``EnvironmentConfig`` only; they do not change the reward formula. Combine
with ``InstinctRewardConfig.enable_instinct_rewards`` for shaped training.

Extension point: this module deliberately exposes ``CURRICULUM_STAGES`` and
``apply_curriculum_stage`` so a future handler/command system can add stages
or per-stage policy hints without rewriting the trainer. Do not implement a
human-handler control system here yet.
"""

from __future__ import annotations

from dataclasses import replace

from sheepdog.config import LabConfig

# Each stage entry overrides a small set of EnvironmentConfig fields.
# Keep the values conservative; the trainer should still be able to learn.
CURRICULUM_STAGES: dict[int, dict[str, object]] = {
    1: {
        "dogs": 1,
        "sheep": 1,
        "width": 60,
        "height": 45,
        "pen_width": 18,
        "pen_height": 18,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 600,
        "no_progress_window": 100,
    },
    2: {
        "dogs": 1,
        "sheep": 3,
        "width": 72,
        "height": 54,
        "pen_width": 18,
        "pen_height": 18,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 720,
        "no_progress_window": 110,
    },
    3: {
        "dogs": 1,
        "sheep": 3,
        "width": 120,
        "height": 84,
        "pen_width": 15,
        "pen_height": 15,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 840,
        "no_progress_window": 120,
    },
    4: {
        "dogs": 2,
        "sheep": 5,
        "width": 132,
        "height": 90,
        "pen_width": 15,
        "pen_height": 15,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 960,
        "no_progress_window": 135,
    },
    5: {
        "dogs": 3,
        "sheep": 6,
        "width": 144,
        "height": 96,
        "pen_width": 15,
        "pen_height": 15,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 1020,
        "no_progress_window": 150,
    },
}


def available_stages() -> tuple[int, ...]:
    """Return the sorted tuple of available curriculum stage numbers."""

    return tuple(sorted(CURRICULUM_STAGES))


def validate_curriculum_stage(stage: int) -> int:
    """Return a normalized curriculum stage or raise for unknown non-zero values."""

    normalized = max(0, int(stage))
    if normalized == 0:
        return 0
    if normalized not in CURRICULUM_STAGES:
        choices = ", ".join(str(value) for value in available_stages())
        raise ValueError(f"Unknown curriculum stage {stage}. Available stages: {choices}")
    return normalized


def apply_training_profile(
    config: LabConfig,
    *,
    enable_instinct_rewards: bool | None = None,
    curriculum_stage: int | None = None,
    debug_reward_breakdown: bool | None = None,
) -> LabConfig:
    """Apply instinct toggles and curriculum overrides to a config copy."""

    base_instincts = config.rewards.instincts
    stage = validate_curriculum_stage(
        base_instincts.curriculum_stage if curriculum_stage is None else curriculum_stage
    )
    updated_instincts = replace(
        base_instincts,
        enable_instinct_rewards=(
            base_instincts.enable_instinct_rewards
            if enable_instinct_rewards is None
            else enable_instinct_rewards
        ),
        debug_reward_breakdown=(
            base_instincts.debug_reward_breakdown
            if debug_reward_breakdown is None
            else debug_reward_breakdown
        ),
        curriculum_stage=stage,
    )
    updated_config = replace(
        config,
        rewards=replace(config.rewards, instincts=updated_instincts),
    )
    return apply_curriculum_stage(updated_config, stage)


def apply_curriculum_stage(config: LabConfig, stage: int) -> LabConfig:
    """Return a copy of ``config`` with the stage's environment overrides applied.

    Stage ``0`` (or any value not in ``CURRICULUM_STAGES``) returns the
    original config unchanged so curriculum is opt-in.
    """

    overrides = CURRICULUM_STAGES.get(stage)
    if not overrides:
        return config
    new_environment = replace(config.environment, **overrides)
    return replace(config, environment=new_environment)


def stage_summary(stage: int) -> str:
    """Return a short human-readable description of a curriculum stage."""

    descriptions = {
        1: "One dog, one sheep, dense grid, one-cell movement, forgiving pen.",
        2: "One dog, three sheep, dense grid grouping with one-cell movement.",
        3: "One dog, three sheep, larger dense field for longer drive/fetch paths.",
        4: "Two dogs, medium flock, dense-grid pressure control and role spacing.",
        5: "Three dogs, larger flock, dense-grid multi-dog cooperation.",
    }
    return descriptions.get(stage, "Custom stage.")
