"""Tests for failure-directed training, anti-forgetting decay, and benchmark protection."""

import pytest
from sheepdog.config import EnvironmentConfig, LabConfig, TrainingConfig
from sheepdog.training.scenario_sampler import (
    STANDARD_EVALUATION_SEEDS,
    ScenarioSampler,
    validate_scenario_mix,
)
from sheepdog.training.training_scenarios import (
    create_gate_wall_flock_scenario,
    create_isolated_stray_scenario,
    create_subpen_flock_scenario,
    get_scenario_builder,
    list_scenario_types,
)


def test_scenario_types_registered() -> None:
    """Verify new hard scenario types are registered and buildable."""
    types = list_scenario_types()
    assert "subpen_flock" in types
    assert "gate_wall_flock" in types
    assert "isolated_stray" in types

    env_config = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)

    s11 = create_subpen_flock_scenario(seed=42, config=env_config)
    assert s11.name == "subpen_flock"
    assert len(s11.sheep) == 4
    assert len(s11.dogs) == 2

    s53 = create_gate_wall_flock_scenario(seed=42, config=env_config)
    assert s53.name == "gate_wall_flock"
    assert len(s53.sheep) == 4

    s41 = create_isolated_stray_scenario(seed=42, config=env_config)
    assert s41.name == "isolated_stray"
    assert len(s41.sheep) == 4


def test_feature_disabled_preserves_normal_training() -> None:
    """Requirement 1: Feature disabled -> existing training selection behavior is unchanged."""
    train_cfg = TrainingConfig(
        train_seed=100,
        failure_directed_training_enabled=False,
        scenario_training_enabled=False,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)

    # Sample 100 episodes; all must be 'normal' with seed = train_seed + index
    for ep in range(100):
        selection = sampler.sample(ep)
        assert selection.scenario_type == "normal"
        assert selection.scenario is None
        assert selection.seed == 100 + ep

    summary = sampler.get_usage_summary()
    assert summary["normal_percentage"] == 100.0
    assert summary["targeted_percentage"] == 0.0
    assert summary["targeted_episodes"] == 0


def test_evaluation_failures_activate_targeted_classes() -> None:
    """Requirement 2: Evaluation failure information increases sampling of matching hard scenario class."""
    train_cfg = TrainingConfig(
        train_seed=200,
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.25,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)

    # Initially, no failures recorded -> 100% normal
    for ep in range(20):
        sel = sampler.sample(ep)
        assert sel.scenario_type == "normal"

    # Evaluation: Seed 11 fails (subpen_flock) and Seed 53 fails (gate_wall_flock)
    eval_records = [
        {"seed": 11, "success": False, "sheep_penned": 3, "stop_reason": "no-progress"},
        {"seed": 23, "success": True, "sheep_penned": 4},
        {"seed": 53, "success": False, "sheep_penned": 0, "stop_reason": "timeout"},
        {"seed": 59, "success": True, "sheep_penned": 4},
    ]
    weights = sampler.update_from_evaluation(eval_records)
    assert "subpen_flock" in weights
    assert "gate_wall_flock" in weights
    assert weights["subpen_flock"] == 1.0
    assert weights["gate_wall_flock"] == 1.0

    # Sample 500 episodes: targeted episodes should appear for subpen_flock and gate_wall_flock
    counts = {"normal": 0, "subpen_flock": 0, "gate_wall_flock": 0}
    for ep in range(500):
        sel = sampler.sample(ep)
        counts[sel.scenario_type] = counts.get(sel.scenario_type, 0) + 1

    assert counts["subpen_flock"] > 0
    assert counts["gate_wall_flock"] > 0
    assert counts["normal"] > counts["subpen_flock"] + counts["gate_wall_flock"]


def test_normal_training_remains_majority() -> None:
    """Requirement 3: Normal/random training remains the majority of training."""
    train_cfg = TrainingConfig(
        train_seed=300,
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.25,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)

    # Force all failure classes active
    sampler.set_failure_weights({
        "subpen_flock": 1.0,
        "gate_wall_flock": 1.0,
        "isolated_stray": 1.0,
    })

    for ep in range(1000):
        sampler.sample(ep)

    summary = sampler.get_usage_summary()
    assert summary["normal_percentage"] > 70.0  # Approx 75%
    assert summary["normal_percentage"] < 80.0
    assert summary["targeted_percentage"] > 20.0
    assert summary["targeted_percentage"] < 30.0


def test_multiple_failure_classes_receive_balanced_exposure() -> None:
    """Requirement 4: Multiple failure classes can receive targeted exposure."""
    train_cfg = TrainingConfig(
        train_seed=400,
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.30,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)

    # Both Seed 11 and Seed 53 fail
    eval_records = [
        {"seed": 11, "success": False, "sheep_penned": 3},
        {"seed": 53, "success": False, "sheep_penned": 1},
    ]
    sampler.update_from_evaluation(eval_records)

    for ep in range(1000):
        sampler.sample(ep)

    summary = sampler.get_usage_summary()
    counts = summary["scenario_counts"]
    assert counts.get("subpen_flock", 0) > 100
    assert counts.get("gate_wall_flock", 0) > 100
    # Ratio between the two equal-weight classes should be reasonably balanced
    ratio = counts["subpen_flock"] / counts["gate_wall_flock"]
    assert 0.7 < ratio < 1.4


def test_anti_forgetting_decay_gradual_retention() -> None:
    """Requirement 5: A recently resolved failure class decays gradually rather than disappearing immediately."""
    train_cfg = TrainingConfig(
        train_seed=500,
        failure_directed_training_enabled=True,
        failure_directed_decay_rate=0.60,
        failure_directed_min_weight=0.05,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)

    # Eval 1: Seed 11 and Seed 53 fail
    w1 = sampler.update_from_evaluation([
        {"seed": 11, "success": False},
        {"seed": 53, "success": False},
    ])
    assert w1["subpen_flock"] == 1.0
    assert w1["gate_wall_flock"] == 1.0

    # Eval 2: Seed 11 passes, Seed 53 fails again
    w2 = sampler.update_from_evaluation([
        {"seed": 11, "success": True},
        {"seed": 53, "success": False},
    ])
    assert w2["gate_wall_flock"] == 1.0
    assert w2["subpen_flock"] == pytest.approx(0.60, rel=1e-2)  # Decayed to 0.60, NOT removed

    # Eval 3: Both pass
    w3 = sampler.update_from_evaluation([
        {"seed": 11, "success": True},
        {"seed": 53, "success": True},
    ])
    assert w3["gate_wall_flock"] == pytest.approx(0.60, rel=1e-2)
    assert w3["subpen_flock"] == pytest.approx(0.36, rel=1e-2)  # Decayed to 0.36

    # Eval 4: Both pass
    w4 = sampler.update_from_evaluation([
        {"seed": 11, "success": True},
        {"seed": 53, "success": True},
    ])
    assert w4["gate_wall_flock"] == pytest.approx(0.36, rel=1e-2)
    assert w4["subpen_flock"] == pytest.approx(0.216, rel=1e-2)


def test_held_out_evaluation_seeds_protected() -> None:
    """Requirement 6: Standardized evaluation seeds are not accidentally inserted into the training pool."""
    train_cfg = TrainingConfig(
        train_seed=11,  # Set train_seed directly to an evaluation seed to test collision prevention
        evaluation_seeds=(11, 23, 37, 41, 53, 59, 61, 67, 71, 73),
        candidate_evaluation_seeds=(91, 92, 93, 94, 95),
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.50,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)
    sampler = ScenarioSampler(train_cfg, env_cfg)
    sampler.set_failure_weights({"subpen_flock": 1.0, "gate_wall_flock": 1.0})

    held_out = STANDARD_EVALUATION_SEEDS | set(train_cfg.evaluation_seeds) | set(train_cfg.candidate_evaluation_seeds)

    for ep in range(200):
        sel = sampler.sample(ep)
        assert sel.seed not in held_out, f"Sampled seed {sel.seed} collided with held-out evaluation seeds!"


def test_deterministic_reproducibility() -> None:
    """Requirement 7: Sampling remains deterministic given the same seed/configuration."""
    train_cfg1 = TrainingConfig(
        train_seed=777,
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.25,
    )
    train_cfg2 = TrainingConfig(
        train_seed=777,
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=0.25,
    )
    env_cfg = EnvironmentConfig(width=96, height=72, dogs=2, sheep=4)

    s1 = ScenarioSampler(train_cfg1, env_cfg)
    s2 = ScenarioSampler(train_cfg2, env_cfg)

    s1.set_failure_weights({"subpen_flock": 1.0, "gate_wall_flock": 0.6})
    s2.set_failure_weights({"subpen_flock": 1.0, "gate_wall_flock": 0.6})

    seq1 = [s1.sample(i) for i in range(100)]
    seq2 = [s2.sample(i) for i in range(100)]

    for item1, item2 in zip(seq1, seq2):
        assert item1.scenario_type == item2.scenario_type
        assert item1.seed == item2.seed
        if item1.scenario is not None:
            assert item2.scenario is not None
            assert item1.scenario.name == item2.scenario.name
