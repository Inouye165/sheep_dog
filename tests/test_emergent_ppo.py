"""Tests for emergent PPO observation mode, stray rewards, and new scenario."""

# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

import math

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import EmergentObservationBuilder, RoleAwareObservationBuilder
from sheepdog.rewards import RewardComputer, RewardInputs
from sheepdog.training.training_scenarios import (
    create_partial_pen_with_stray_scenario,
    list_scenario_types,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(observation_mode: str = "guided", **env_overrides) -> LabConfig:
    """Return a minimal LabConfig for tests."""
    return LabConfig(
        environment=EnvironmentConfig(**env_overrides),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir="artifacts",
            web_export_dir="web/public/generated",
            observation_mode=observation_mode,
        ),
    )


def _env_config() -> EnvironmentConfig:
    return EnvironmentConfig()


def _base_reward_inputs(**overrides) -> RewardInputs:
    """Return a minimal valid RewardInputs with optional field overrides."""
    defaults: dict = dict(
        previous_average_distance=10.0,
        current_average_distance=10.0,
        previous_flock_spread=4.0,
        current_flock_spread=4.0,
        newly_penned=0,
        no_progress_step=False,
        touched_wall=False,
        waited_without_reason=False,
        terminated=False,
        timeout=False,
        success=False,
    )
    defaults.update(overrides)
    return RewardInputs(**defaults)


# ---------------------------------------------------------------------------
# Observation builder -- emergent mode
# ---------------------------------------------------------------------------


def test_emergent_observation_has_no_role_labels() -> None:
    """EmergentObservationBuilder must not include any role_* feature names."""
    builder = EmergentObservationBuilder()
    env = SheepdogEnvironment(_make_config("emergent"))
    env.reset()
    obs = builder.build(env, 0)
    assert not any(name.startswith("role_") for name in obs.feature_names), (
        "Emergent observations must not contain role label features"
    )


def test_emergent_observation_has_no_scripted_target() -> None:
    """Emergent mode must not expose scripted target_x/target_y features."""
    builder = EmergentObservationBuilder()
    env = SheepdogEnvironment(_make_config("emergent"))
    env.reset()
    obs = builder.build(env, 0)
    scripted_target_names = {"target_x", "target_y", "distance_to_target"}
    assert not any(name in scripted_target_names for name in obs.feature_names), (
        "Emergent observations must not expose scripted-role target coordinates"
    )


def test_emergent_observation_has_farthest_unpenned_features() -> None:
    """Emergent mode must expose farthest-unpenned-sheep distance features."""
    builder = EmergentObservationBuilder()
    env = SheepdogEnvironment(_make_config("emergent"))
    env.reset()
    obs = builder.build(env, 0)
    required = {"farthest_unpenned_dx", "farthest_unpenned_dy", "farthest_unpenned_distance"}
    assert required.issubset(set(obs.feature_names)), (
        f"Missing farthest-unpenned features. Got: {obs.feature_names}"
    )


def test_emergent_adapter_obs_size_differs_from_guided() -> None:
    """Guided and emergent modes must produce observation vectors of different length."""
    guided_builder = RoleAwareObservationBuilder()
    emergent_builder = EmergentObservationBuilder()

    env = SheepdogEnvironment(_make_config("guided"))
    env.reset()

    guided_obs = guided_builder.build(env, 0)
    emergent_obs = emergent_builder.build(env, 0)

    guided_len = len(guided_obs.values)
    emergent_len = len(emergent_obs.values)
    assert guided_len != emergent_len, (
        "Guided and emergent observation vectors should differ in length; "
        f"got guided={guided_len}, emergent={emergent_len}"
    )


# ---------------------------------------------------------------------------
# Scenario registry -- partial_pen_with_stray
# ---------------------------------------------------------------------------


def test_scenario_mix_includes_partial_pen_with_stray() -> None:
    """list_scenario_types must include the new partial_pen_with_stray scenario."""
    assert "partial_pen_with_stray" in list_scenario_types()


def test_partial_pen_with_stray_creates_stray() -> None:
    """The new scenario must place at least one sheep far from the pen."""
    config = _env_config()
    scenario = create_partial_pen_with_stray_scenario(seed=42, config=config)

    assert len(scenario.sheep) == config.sheep

    # At least one sheep should be far from pen centre (far corner placement).
    # Use a threshold of 30% of the field diagonal.
    diagonal = math.hypot(config.width, config.height)
    threshold = diagonal * 0.30

    pen_cx = scenario.pen.origin_x + scenario.pen.width / 2.0
    pen_cy = scenario.pen.origin_y + scenario.pen.height / 2.0

    distances = [math.hypot(s.x - pen_cx, s.y - pen_cy) for s in scenario.sheep]
    assert max(distances) >= threshold, (
        f"Expected a stray far from pen (>{threshold:.1f}), got distances={distances}"
    )


# ---------------------------------------------------------------------------
# Stray-ignore reward term
# ---------------------------------------------------------------------------


def test_farthest_sheep_progress_reward_penalizes_stray_getting_further() -> None:
    """When farthest sheep moves away from pen, farthest_sheep_progress should be negative."""
    config = RewardConfig(farthest_sheep_progress_scale=1.0)
    computer = RewardComputer(config)
    breakdown = computer.compute(
        _base_reward_inputs(previous_farthest_distance=10.0, current_farthest_distance=12.0)
    )
    assert breakdown.farthest_sheep_progress < 0.0, (
        "farthest_sheep_progress should be negative when stray moves further from pen"
    )


def test_farthest_sheep_progress_reward_is_positive_when_stray_approaches() -> None:
    """When farthest sheep moves closer to pen, farthest_sheep_progress should be positive."""
    config = RewardConfig(farthest_sheep_progress_scale=1.0)
    computer = RewardComputer(config)
    breakdown = computer.compute(
        _base_reward_inputs(previous_farthest_distance=12.0, current_farthest_distance=10.0)
    )
    assert breakdown.farthest_sheep_progress > 0.0, (
        "farthest_sheep_progress should be positive when stray approaches pen"
    )


def test_stray_terms_zero_by_default() -> None:
    """Default RewardConfig (scales=0.0) must produce zero stray reward terms."""
    computer = RewardComputer(RewardConfig())
    breakdown = computer.compute(
        _base_reward_inputs(previous_farthest_distance=5.0, current_farthest_distance=8.0)
    )
    assert breakdown.farthest_sheep_progress == 0.0
    assert breakdown.stray_ignore_penalty == 0.0
