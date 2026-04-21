"""Regression tests for checkpoint and evaluation export."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicPolicy
from sheepdog.policies.trainable import PolicyWeights
from sheepdog.server import TrainingManager, _load_playable_policy, _run_live_replay
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
        HeuristicPolicy(), (11, 13), checkpoint_episode=0
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert len(summary.records) == 2
    assert {record.seed for record in summary.records} == {11, 13}


def test_evaluation_summary_includes_success_timeout_and_completion_metrics(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    summary, _, _ = evaluator.evaluate(HeuristicPolicy(), (11, 13), checkpoint_episode=0)

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


def test_training_manager_clear_removes_artifacts_and_resets_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    Trainer(config, tmp_path).train()

    generated_root = Path(config.training.web_export_dir)
    assert (tmp_path / Trainer.STATE_FILENAME).exists()
    assert (tmp_path / "checkpoints").exists()
    assert (tmp_path / "evaluations").exists()
    assert (generated_root / "checkpoint-index.json").exists()
    assert (generated_root / "replays").exists()

    monkeypatch.setattr("sheepdog.server.LabConfig", lambda: config)
    manager = TrainingManager()

    status = manager.clear()

    assert status["running"] is False
    assert status["total_episodes_trained"] == 0
    assert status["message"] == "Training history cleared"
    assert not (tmp_path / Trainer.STATE_FILENAME).exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "evaluations").exists()
    assert not (generated_root / "checkpoint-index.json").exists()
    assert not (generated_root / "replays").exists()


def test_live_replay_uses_instinct_only_when_no_training_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr("sheepdog.server.LabConfig", lambda: config)

    policy, policy_name = _load_playable_policy(config)
    payload = _run_live_replay(seed=11)

    assert policy.__class__.__name__ == "HeuristicPolicy"
    assert policy_name == "instinct-only"
    assert payload["policy_name"] == "instinct-only"
    assert payload["seed"] == 11
    assert payload["frames"]
    assert (Path(config.training.web_export_dir) / "latest-replay.json").exists()


def test_live_replay_uses_trained_policy_when_training_state_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_config = make_config(tmp_path)
    config = LabConfig(
        environment=base_config.environment,
        rewards=base_config.rewards,
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0, 1),
            evaluation_seeds=base_config.training.evaluation_seeds,
            train_seed=base_config.training.train_seed,
            evaluation_seed=base_config.training.evaluation_seed,
            mutation_scale=base_config.training.mutation_scale,
            output_dir=base_config.training.output_dir,
            web_export_dir=base_config.training.web_export_dir,
        ),
    )
    Trainer(config, tmp_path).train()
    monkeypatch.setattr("sheepdog.server.LabConfig", lambda: config)

    policy, policy_name = _load_playable_policy(config)
    payload = _run_live_replay(seed=13)

    assert policy.__class__.__name__ == "TrainableLinearPolicy"
    assert policy_name == "trained-checkpoint"
    assert payload["policy_name"] == "trained-checkpoint"
    assert payload["seed"] == 13
    assert payload["frames"]
