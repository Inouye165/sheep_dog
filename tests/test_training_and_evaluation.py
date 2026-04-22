"""Regression tests for checkpoint and evaluation export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicExpertPolicy
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy
from sheepdog.server import _build_training_job_config, _load_playable_policy, TrainingManager
from sheepdog.training.trainer import Trainer


def make_config(output_dir: Path) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(max_steps=30, dogs=2, sheep=3),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11, 13),
            train_seed=7,
            evaluation_seed=9,
            mutation_scale=0.05,
            output_dir=str(output_dir),
            web_export_dir=str(output_dir / "web" / "generated"),
        ),
    )


def test_checkpoint_metadata_is_written(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    trainer = Trainer(config, tmp_path)

    summary = trainer.train()

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint-000000.json"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_episode"] == 0
    assert payload["environment_config"]["dogs"] == 2
    assert payload["reward_config"]["progress_scale"] > 0
    assert summary.checkpoints


def test_evaluation_writes_json_and_csv(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    summary, json_path, csv_path = evaluator.evaluate(
        HeuristicExpertPolicy(), (11, 13), checkpoint_episode=0
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert len(summary.records) == 2
    assert {record.seed for record in summary.records} == {11, 13}


def test_evaluation_summary_includes_success_timeout_and_completion_metrics(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    summary, _, _ = evaluator.evaluate(HeuristicExpertPolicy(), (11, 13), checkpoint_episode=0)

    assert 0.0 <= summary.success_rate <= 1.0
    assert 0.0 <= summary.timeout_rate <= 1.0
    assert summary.average_completion_steps >= 0
    assert summary.average_completion_seconds >= 0


def test_training_state_persists_across_runs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    Trainer(config, tmp_path).train()
    state_path = tmp_path / Trainer.STATE_FILENAME
    assert state_path.exists()
    first_total = json.loads(state_path.read_text(encoding="utf-8"))["total_episodes_trained"]

    second = Trainer(config, tmp_path)
    assert second.total_episodes_trained == first_total
    second.train()
    second_total = json.loads(state_path.read_text(encoding="utf-8"))["total_episodes_trained"]
    assert second_total >= first_total


def test_policy_weights_load_legacy_state_payload() -> None:
    payload = {
        "nearest_sheep": 1.0,
        "flock_center": 2.0,
        "pen_pressure": 3.0,
        "behind_flock": 4.0,
        "wall_margin": 5.0,
        "wait_bias": -6.0,
    }

    weights = PolicyWeights.from_dict(payload)

    assert weights.nearest_sheep == 1.0
    assert weights.team_formation == PolicyWeights().team_formation
    assert weights.collector_focus == PolicyWeights().collector_focus


def test_hill_climber_training_saves_role_aware_weights(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    summary = Trainer(config, tmp_path).train()
    state_path = tmp_path / Trainer.STATE_FILENAME
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert state_path.exists()
    assert "rear_drive" in payload["weights"]
    assert "collector_focus" in payload["weights"]
    assert summary.final_weights.collector_focus == payload["weights"]["collector_focus"]


def test_build_training_job_config_applies_fast_mode_and_curriculum() -> None:
    config = _build_training_job_config(
        4,
        True,
        enable_instinct_rewards=True,
        curriculum_stage=2,
        debug_reward_breakdown=True,
    )

    assert config.training.episodes == 3
    assert config.training.evaluation_seeds == (11,)
    assert config.rewards.instincts.enable_instinct_rewards is True
    assert config.rewards.instincts.curriculum_stage == 2
    assert config.environment.dogs == 1


def test_load_playable_policy_reads_checkpoint_weights(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), training=replace(make_config(tmp_path).training))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "checkpoint-000003.json").write_text(
        json.dumps(
            {
                "checkpoint_episode": 3,
                "policy_weights": {"rear_drive": 2.5, "collector_focus": 1.9},
            }
        ),
        encoding="utf-8",
    )

    policy = _load_playable_policy(config, checkpoint_episode=3, policy_mode="trained_policy")

    assert isinstance(policy, TrainableLinearPolicy)
    assert policy.weights.rear_drive == 2.5
    assert policy.weights.collector_focus == 1.9


def test_training_manager_live_replay_writes_latest_replay(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "generated"
    config = replace(
        make_config(artifacts),
        training=replace(make_config(artifacts).training, output_dir=str(artifacts), web_export_dir=str(generated)),
    )
    manager = TrainingManager()

    import sheepdog.server as server_module

    original_config = server_module.LabConfig

    class TestConfig:
        def __new__(cls):
            return config

    server_module.LabConfig = TestConfig
    try:
        replay = manager.run_live_replay(11)
    finally:
        server_module.LabConfig = original_config

    assert replay["seed"] == 11
    assert (generated / "latest-replay.json").exists()
