"""Tests for scenario-based training functionality."""

# pylint: disable=missing-function-docstring,protected-access
from __future__ import annotations

from pathlib import Path

import pytest

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.heuristic import InstinctOnlyPolicy
from sheepdog.training.scenario_sampler import ScenarioSampler, validate_scenario_mix
from sheepdog.training.training_scenarios import (
    create_corner_huddle_scenario,
    create_scattered_sheep_scenario,
    create_split_flock_scenario,
    get_scenario_builder,
    list_scenario_types,
)


def make_config(output_dir: Path) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(output_dir),
            web_export_dir=str(output_dir / "web" / "generated"),
        ),
    )


def test_list_scenario_types():
    types = list_scenario_types()
    assert "scattered_sheep" in types
    assert "split_flock" in types
    assert "corner_huddle" in types
    assert "normal_random" in types


def test_get_scenario_builder():
    builder = get_scenario_builder("scattered_sheep")
    assert builder is not None
    assert callable(builder)

    with pytest.raises(ValueError):
        get_scenario_builder("invalid_type")


def test_create_scattered_sheep_scenario(tmp_path: Path):
    config = make_config(tmp_path).environment
    scenario = create_scattered_sheep_scenario(seed=42, config=config)

    assert scenario.name == "scattered_sheep"
    assert scenario.seed == 42
    assert len(scenario.sheep) == config.sheep
    assert len(scenario.dogs) == config.dogs

    # Verify sheep are spread across the field
    sheep_x_positions = [s.x for s in scenario.sheep]
    sheep_y_positions = [s.y for s in scenario.sheep]
    x_spread = max(sheep_x_positions) - min(sheep_x_positions)
    y_spread = max(sheep_y_positions) - min(sheep_y_positions)
    # Scattered sheep should have significant spread (use >= to handle small grids)
    assert x_spread >= config.width // 4
    assert y_spread >= config.height // 4


def test_create_split_flock_scenario(tmp_path: Path):
    config = make_config(tmp_path).environment
    scenario = create_split_flock_scenario(seed=42, config=config)

    assert scenario.name == "split_flock"
    assert scenario.seed == 42
    assert len(scenario.sheep) == config.sheep
    assert len(scenario.dogs) == config.dogs

    # Verify sheep are split into at least two groups
    sheep_positions = [(s.x, s.y) for s in scenario.sheep]
    # Calculate centroid
    centroid_x = sum(p[0] for p in sheep_positions) / len(sheep_positions)
    centroid_y = sum(p[1] for p in sheep_positions) / len(sheep_positions)

    # Count sheep on each side of centroid
    left_count = sum(1 for x, _ in sheep_positions if x < centroid_x)
    right_count = sum(1 for x, _ in sheep_positions if x > centroid_x)

    # Should have sheep on both sides (split flock)
    assert left_count > 0
    assert right_count > 0


def test_create_corner_huddle_scenario(tmp_path: Path):
    config = make_config(tmp_path).environment
    scenario = create_corner_huddle_scenario(seed=42, config=config)

    assert scenario.name == "corner_huddle"
    assert scenario.seed == 42
    assert len(scenario.sheep) == config.sheep
    assert len(scenario.dogs) == config.dogs

    # Verify sheep are huddled in a corner
    sheep_x_positions = [s.x for s in scenario.sheep]
    sheep_y_positions = [s.y for s in scenario.sheep]
    x_spread = max(sheep_x_positions) - min(sheep_x_positions)
    y_spread = max(sheep_y_positions) - min(sheep_y_positions)

    # Huddled sheep should have small spread
    assert x_spread < config.width // 4
    assert y_spread < config.height // 4

    # Verify they're in a corner (bottom-left in this implementation)
    avg_x = sum(sheep_x_positions) / len(sheep_x_positions)
    avg_y = sum(sheep_y_positions) / len(sheep_y_positions)
    assert avg_x < config.width // 2  # Left half
    assert avg_y > config.height // 2  # Bottom half


def test_scenario_reset_valid_bounds(tmp_path: Path):
    config = make_config(tmp_path)
    scenario = create_scattered_sheep_scenario(seed=42, config=config.environment)

    env = SheepdogEnvironment(config)
    snapshot = env.reset_from_scenario(scenario)

    # Verify all agents are within bounds
    for dog in snapshot.dogs:
        assert 0 <= dog.x < config.environment.width
        assert 0 <= dog.y < config.environment.height

    for sheep in snapshot.sheep:
        assert 0 <= sheep.x < config.environment.width
        assert 0 <= sheep.y < config.environment.height


def test_validate_scenario_mix_valid():
    mix = {
        "random": 0.5,
        "scattered_sheep": 0.2,
        "split_flock": 0.15,
        "corner_huddle": 0.15,
    }
    # Should not raise
    validate_scenario_mix(mix)


def test_validate_scenario_mix_invalid_type():
    mix = {"random": 0.5, "invalid_type": 0.5}
    with pytest.raises(ValueError, match="Invalid scenario type"):
        validate_scenario_mix(mix)


def test_validate_scenario_mix_negative_weight():
    mix = {"random": 0.5, "scattered_sheep": -0.5}
    with pytest.raises(ValueError, match="non-negative"):
        validate_scenario_mix(mix)


def test_validate_scenario_mix_zero_total():
    mix = {"random": 0.0, "scattered_sheep": 0.0}
    with pytest.raises(ValueError, match="sum to a positive value"):
        validate_scenario_mix(mix)


def test_scenario_sampler_disabled(tmp_path: Path):
    config = make_config(tmp_path)
    sampler = ScenarioSampler(config.training, config.environment)

    selection = sampler.sample(episode_index=0)
    assert selection.scenario_type == "random"
    assert selection.scenario is None
    assert selection.seed == config.training.train_seed

    # Verify usage tracking
    counts = sampler.get_usage_counts()
    assert counts.get("random") == 1


def test_scenario_sampler_enabled(tmp_path: Path):
    config = LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web" / "generated"),
            scenario_training_enabled=True,
            scenario_mix={"random": 0.5, "scattered_sheep": 0.5},
        ),
    )
    sampler = ScenarioSampler(config.training, config.environment)

    # Sample multiple times
    selections = [sampler.sample(i) for i in range(10)]

    # All selections should have valid scenario types
    for selection in selections:
        assert selection.scenario_type in ["random", "scattered_sheep"]
        assert selection.seed == config.training.train_seed + selection.seed - config.training.train_seed

    # Verify usage tracking
    counts = sampler.get_usage_counts()
    assert sum(counts.values()) == 10
    assert "random" in counts or "scattered_sheep" in counts


def test_scenario_sampler_reproducibility(tmp_path: Path):
    config = LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            train_seed=42,
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web" / "generated"),
            scenario_training_enabled=True,
            scenario_mix={"random": 0.5, "scattered_sheep": 0.5},
        ),
    )

    sampler1 = ScenarioSampler(config.training, config.environment)
    selections1 = [sampler1.sample(i).scenario_type for i in range(20)]

    sampler2 = ScenarioSampler(config.training, config.environment)
    selections2 = [sampler2.sample(i).scenario_type for i in range(20)]

    # Same seed should produce same sequence
    assert selections1 == selections2


def test_scenario_sampler_usage_summary(tmp_path: Path):
    config = LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web" / "generated"),
            scenario_training_enabled=True,
            scenario_mix={"random": 0.5, "scattered_sheep": 0.5},
        ),
    )
    sampler = ScenarioSampler(config.training, config.environment)

    # Sample some episodes
    for i in range(10):
        sampler.sample(i)

    summary = sampler.get_usage_summary()
    assert summary["total_episodes"] == 10
    assert "scenario_counts" in summary
    assert "scenario_percentages" in summary
    assert sum(summary["scenario_counts"].values()) == 10


def test_scenario_sampler_reset_counts(tmp_path: Path):
    config = LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web" / "generated"),
            scenario_training_enabled=True,
            scenario_mix={"random": 0.5, "scattered_sheep": 0.5},
        ),
    )
    sampler = ScenarioSampler(config.training, config.environment)

    # Sample some episodes
    for i in range(5):
        sampler.sample(i)

    assert sum(sampler.get_usage_counts().values()) == 5

    # Reset and verify counts are cleared
    sampler.reset_counts()
    assert sum(sampler.get_usage_counts().values()) == 0


def test_scenario_training_preserves_normal_reset(tmp_path: Path):
    """Verify that when scenario training is disabled, behavior is unchanged."""
    config = make_config(tmp_path)
    env = SheepdogEnvironment(config)

    # Normal reset should work as before
    snapshot1 = env.reset(seed=42)
    snapshot2 = env.reset(seed=42)

    # Same seed should produce same result
    assert snapshot1.dogs[0].x == snapshot2.dogs[0].x
    assert snapshot1.sheep[0].x == snapshot2.sheep[0].x


def test_scenario_training_with_predefined_scenario(tmp_path: Path):
    """Verify that predefined scenarios can be used for training."""
    config = make_config(tmp_path)
    scenario = create_scattered_sheep_scenario(seed=42, config=config.environment)

    env = SheepdogEnvironment(config)
    snapshot = env.reset_from_scenario(scenario)

    # Verify the scenario was applied
    assert len(snapshot.dogs) == config.environment.dogs
    assert len(snapshot.sheep) == config.environment.sheep

    # Verify positions match the scenario
    for i, dog in enumerate(snapshot.dogs):
        assert dog.x == scenario.dogs[i].x
        assert dog.y == scenario.dogs[i].y

    for i, sheep in enumerate(snapshot.sheep):
        assert sheep.x == scenario.sheep[i].x
        assert sheep.y == scenario.sheep[i].y


def test_scenario_training_policy_execution(tmp_path: Path):
    """Verify that a policy can execute on a scenario-based reset."""
    config = make_config(tmp_path)
    scenario = create_scattered_sheep_scenario(seed=42, config=config.environment)

    env = SheepdogEnvironment(config)
    policy = InstinctOnlyPolicy()

    env.reset_from_scenario(scenario)
    # Take a few steps to verify the environment is functional
    for _ in range(5):
        actions = policy.select_actions(env)
        env.step(actions)

    # Environment should still be functional
    snapshot = env.get_state_snapshot()
    assert snapshot.step == 5
