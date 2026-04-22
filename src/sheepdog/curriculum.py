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

from sheepdog.config import EnvironmentConfig, LabConfig

# Each stage entry overrides a small set of EnvironmentConfig fields.
# Keep the values conservative; the trainer should still be able to learn.
CURRICULUM_STAGES: dict[int, dict[str, object]] = {
    1: {
        "dogs": 1,
        "sheep": 1,
        "width": 20,
        "height": 15,
        "pen_width": 6,
        "pen_height": 6,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 200,
        "no_progress_window": 40,
    },
    2: {
        "dogs": 1,
        "sheep": 3,
        "width": 24,
        "height": 18,
        "pen_width": 6,
        "pen_height": 6,
        "dog_speed": 1,
        "sheep_speed": 1,
        "max_steps": 240,
        "no_progress_window": 40,
    },
    3: {
        "dogs": 1,
        "sheep": 3,
        "width": 40,
        "height": 28,
        "pen_width": 5,
        "pen_height": 5,
        "dog_speed": 2,
        "sheep_speed": 1,
        "max_steps": 280,
        "no_progress_window": 40,
    },
    4: {
        "dogs": 2,
        "sheep": 5,
        "width": 44,
        "height": 30,
        "pen_width": 5,
        "pen_height": 5,
        "dog_speed": 2,
        "sheep_speed": 1,
        "max_steps": 320,
        "no_progress_window": 45,
    },
    5: {
        "dogs": 3,
        "sheep": 6,
        "width": 48,
        "height": 32,
        "pen_width": 5,
        "pen_height": 5,
        "dog_speed": 3,
        "sheep_speed": 1,
        "max_steps": 340,
        "no_progress_window": 50,
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
        1: "One dog, one sheep, slow movement, open field, forgiving pen.",
        2: "One dog, three sheep, slow movement, open field grouping.",
        3: "One dog, three sheep, larger field for longer drive/fetch paths.",
        4: "Two dogs, medium flock, harder pressure control and role spacing.",
        5: "Three dogs, larger flock, multi-dog cooperation under full speed.",
    }
    return descriptions.get(stage, "Custom stage.")
