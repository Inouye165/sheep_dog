"""Tests for HierarchicalObservationBuilder (dog identity + shepherd command features)."""

from __future__ import annotations

import numpy as np
import pytest

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import (
    MAX_DOG_SLOTS,
    HierarchicalObservationBuilder,
    RoleAwareObservationBuilder,
)
from sheepdog.shepherd import COMMAND_ORDER, ShepherdCommand

NUM_SHEPHERD_FEATURES: int = len(COMMAND_ORDER)  # 8
NUM_IDENTITY_FEATURES: int = 2 + MAX_DOG_SLOTS  # dog_id_normalized, dog_count_normalized, 5 one-hot


def _make_env(seed: int = 0) -> SheepdogEnvironment:
    env = SheepdogEnvironment(LabConfig())
    env.reset(seed=seed)
    return env


def test_hierarchical_builder_returns_dog_observation():
    from sheepdog.observations import DogObservation  # noqa: PLC0415

    env = _make_env()
    builder = HierarchicalObservationBuilder()
    obs = builder.build_hierarchical(env, dog_index=0)
    assert isinstance(obs, DogObservation)


def test_hierarchical_obs_vector_length_matches_base_plus_extras():
    env = _make_env()
    base_builder = RoleAwareObservationBuilder()
    base_obs = base_builder.build(env, 0)
    base_len = len(base_obs.values)

    hier_builder = HierarchicalObservationBuilder()
    hier_obs = hier_builder.build_hierarchical(env, dog_index=0)
    expected_extra = NUM_SHEPHERD_FEATURES + NUM_IDENTITY_FEATURES
    assert len(hier_obs.values) == base_len + expected_extra


def test_shepherd_cmd_features_sum_to_one():
    """Shepherd command is one-hot encoded; exactly one value should be 1.0."""
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    obs = builder.build_hierarchical(
        env, dog_index=0, shepherd_command=ShepherdCommand.DRIVE_TO_PEN
    )
    names = obs.feature_names
    cmd_vals = [obs.values[names.index(f"shepherd_cmd_{cmd.value}")] for cmd in COMMAND_ORDER]
    assert abs(sum(cmd_vals) - 1.0) < 1e-6
    hot = [v for v in cmd_vals if v > 0.5]
    assert len(hot) == 1


def test_correct_shepherd_cmd_is_hot():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    for cmd in COMMAND_ORDER:
        obs = builder.build_hierarchical(env, dog_index=0, shepherd_command=cmd)
        idx = obs.feature_names.index(f"shepherd_cmd_{cmd.value}")
        assert obs.values[idx] == pytest.approx(1.0)


def test_shepherd_cmd_none_defaults_to_gather():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    obs_none = builder.build_hierarchical(env, dog_index=0, shepherd_command=None)
    gather_idx = obs_none.feature_names.index("shepherd_cmd_gather")
    assert obs_none.values[gather_idx] == pytest.approx(1.0)


def test_dog_id_normalized_in_unit_range():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    for i in range(len(env.dogs)):
        obs = builder.build_hierarchical(env, dog_index=i)
        idx = obs.feature_names.index("dog_id_normalized")
        assert 0.0 <= obs.values[idx] <= 1.0


def test_dog_id_slot_is_one_hot():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    obs = builder.build_hierarchical(env, dog_index=0)
    names = obs.feature_names
    slot_vals = [obs.values[names.index(f"dog_id_slot_{n}")] for n in range(MAX_DOG_SLOTS)]
    assert abs(sum(slot_vals) - 1.0) < 1e-6


def test_different_dogs_have_different_identity_features():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    if len(env.dogs) < 2:
        pytest.skip("Need at least 2 dogs")
    obs0 = builder.build_hierarchical(env, dog_index=0)
    obs1 = builder.build_hierarchical(env, dog_index=1)
    assert not np.array_equal(obs0.values, obs1.values)


def test_obs_is_deterministic_for_same_inputs():
    env = _make_env(seed=7)
    builder = HierarchicalObservationBuilder()
    obs_a = builder.build_hierarchical(env, dog_index=0, shepherd_command=ShepherdCommand.GATHER)
    obs_b = builder.build_hierarchical(env, dog_index=0, shepherd_command=ShepherdCommand.GATHER)
    np.testing.assert_array_equal(obs_a.values, obs_b.values)


def test_metadata_contains_shepherd_command():
    env = _make_env()
    builder = HierarchicalObservationBuilder()
    obs = builder.build_hierarchical(env, dog_index=0, shepherd_command=ShepherdCommand.STOP)
    assert "shepherd_command" in obs.metadata
    assert obs.metadata["shepherd_command"] == ShepherdCommand.STOP
