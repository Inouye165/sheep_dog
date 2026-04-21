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
        "pen_width": 4,
        "pen_height": 4,
        "max_steps": 200,
        "no_progress_window": 30,
    },
    2: {
        "dogs": 1,
        "sheep": 3,
        "width": 24,
        "height": 18,
        "pen_width": 4,
        "pen_height": 4,
        "max_steps": 220,
        "no_progress_window": 35,
    },
    3: {
        "dogs": 1,
        "sheep": 3,
        "width": 40,
        "height": 28,
        "pen_width": 5,
        "pen_height": 5,
        "max_steps": 280,
        "no_progress_window": 40,
    },
    4: {
        "dogs": 1,
        "sheep": 4,
        "width": 40,
        "height": 28,
        "pen_width": 5,
        "pen_height": 5,
        "max_steps": 320,
        "no_progress_window": 45,
    },
    5: {
        "dogs": 2,
        "sheep": 5,
        "width": 44,
        "height": 30,
        "pen_width": 5,
        "pen_height": 5,
        "max_steps": 340,
        "no_progress_window": 50,
    },
}


def available_stages() -> tuple[int, ...]:
    """Return the sorted tuple of available curriculum stage numbers."""

    return tuple(sorted(CURRICULUM_STAGES))


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
        1: "One dog, one sheep, small open field, nearby pen.",
        2: "One dog, small flock, learn grouping in an open field.",
        3: "One dog, small flock, larger field for sustained drive/fetch.",
        4: "One dog, slightly bigger flock, longer episodes for guarded pens.",
        5: "Two dogs, small flock, basic cooperation without interference.",
    }
    return descriptions.get(stage, "Custom stage.")
