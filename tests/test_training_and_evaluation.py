"""Regression tests for checkpoint and evaluation export."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicPolicy
from sheepdog.policies.trainable import PolicyWeights
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
