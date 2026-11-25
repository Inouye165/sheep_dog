"""Regression tests for restoring observation config from checkpoint policy_config."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.server import _config_from_checkpoint_payload, _resolve_replay_selection


def test_resolve_replay_selection_restores_emergent_gate_obs_config(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    checkpoint_episode = 10
    checkpoint_payload = {
        "checkpoint_episode": checkpoint_episode,
        "total_training_episodes": checkpoint_episode,
        "policy_name": "neural_policy",
        "trainer_type": "ppo",
        "policy_type": "neural",
        "environment_config": {
            "width": 60,
            "height": 45,
            "dogs": 3,
            "sheep": 6,
        },
        "policy_config": {
            "observation_mode": "emergent_gate",
            "observation_history_strides": [0, 4, 8, 16],
            "include_loop_flags_in_observation": False,
            "hidden_sizes": [256, 256, 128],
        },
    }
    (checkpoints_dir / f"checkpoint-{checkpoint_episode:06d}.json").write_text(
        json.dumps(checkpoint_payload),
        encoding="utf-8",
    )

    base_config = LabConfig(
        environment=EnvironmentConfig(dogs=3, sheep=6),
        training=TrainingConfig(output_dir=str(tmp_path)),
    )

    restored = _config_from_checkpoint_payload(base_config, checkpoint_payload)
    assert restored.training.observation_mode == "emergent_gate"
    assert restored.training.observation_history_strides == (0, 4, 8, 16)
    assert restored.training.include_loop_flags_in_observation is False
    assert restored.training.neural_hidden_sizes == (256, 256, 128)

    selection = _resolve_replay_selection(
        base_config,
        checkpoint_episode=checkpoint_episode,
    )
    assert selection.config.training.observation_mode == "emergent_gate"
    assert selection.config.training.observation_history_strides == (0, 4, 8, 16)
    assert selection.config.training.include_loop_flags_in_observation is False
    assert selection.config.training.neural_hidden_sizes == (256, 256, 128)

    env = SheepdogEnvironment(selection.config)
    env.reset(seed=0)
    obs = env.build_observation_for_dog(0)
    assert len(obs.values) == 208
    assert len(obs.feature_names) == 208


def test_config_from_checkpoint_payload_preserves_training_without_policy_config() -> None:
    base_config = LabConfig(
        training=replace(
            TrainingConfig(),
            observation_mode="emergent_gate",
            observation_history_strides=(0, 2, 6),
            include_loop_flags_in_observation=False,
            neural_hidden_sizes=(128, 128),
        )
    )

    without_key = _config_from_checkpoint_payload(base_config, {})
    assert without_key.training == base_config.training

    with_none = _config_from_checkpoint_payload(base_config, {"policy_config": None})
    assert with_none.training == base_config.training
