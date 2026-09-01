"""Tests for standardized 10-seed evaluation configuration and checkpoint metadata consistency."""

from __future__ import annotations

from pathlib import Path

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.server import SheepdogEnvironment


def test_training_config_defaults() -> None:
    config = TrainingConfig()
    expected_seeds = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    assert config.evaluation_seeds == expected_seeds
    assert config.quick_evaluation_seed_count == 10


def test_eval_seed_geometry_fallback_and_environment() -> None:
    config = LabConfig(
        training=TrainingConfig(
            evaluation_seeds=(11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
        )
    )
    env = SheepdogEnvironment(config)
    for seed in config.training.evaluation_seeds:
        snapshot = env.reset(seed=seed)
        assert snapshot is not None


def test_promotion_gate_with_10_seeds(tmp_path: Path) -> None:
    import json

    from sheepdog.server import compute_promotion_gate_snapshot

    output_root = tmp_path / "artifacts"
    eval_dir = output_root / "evaluations"
    cp_dir = output_root / "checkpoints"
    eval_dir.mkdir(parents=True)
    cp_dir.mkdir(parents=True)

    seeds = [11, 23, 37, 41, 53, 59, 61, 67, 71, 73]
    cp_payload = {
        "curriculum_stage": 1,
        "policy_version": 1,
        "checkpoint_id": "chk_10_seeds",
        "evaluation_seed_set_id": "seed_set_10",
        "evaluation_seeds": seeds,
        "evaluation_seed_count": 10,
    }
    cp_file = cp_dir / "checkpoint-000100.json"
    cp_file.write_text(json.dumps(cp_payload), encoding="utf-8")

    eval_payload = {
        "checkpoint_episode": 100,
        "checkpoint_id": "chk_10_seeds",
        "policy_version": 1,
        "curriculum_stage": 1,
        "evaluation_seed_set_id": "seed_set_10",
        "records": [{"seed": s, "success": True, "timeout": False} for s in seeds],
    }
    eval_file = eval_dir / "evaluation-checkpoint-000100.json"
    eval_file.write_text(json.dumps(eval_payload), encoding="utf-8")

    snapshot = compute_promotion_gate_snapshot(
        output_root=output_root,
        target_ep=100,
        checkpoint_payload=cp_payload,
    )
    assert snapshot["seed_count"] == 10
    assert snapshot["success_count"] == 10

