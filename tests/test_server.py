"""Tests for the interactive training server."""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.server import TrainingManager


def test_clear_training_restores_untrained_baseline(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    generated.mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    (artifacts / "training-state.json").write_text("{}", encoding="utf-8")
    (artifacts / "training-summary.json").write_text("{}", encoding="utf-8")
    (generated / "latest-replay.json").write_text("{}", encoding="utf-8")

    manager = TrainingManager()
    manager._remove_path = manager._remove_path  # keep attribute access explicit for lint stability

    import sheepdog.server as server_module

    original_config = server_module.LabConfig

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    server_module.LabConfig = TestConfig
    try:
        payload, status = manager.clear()
    finally:
        server_module.LabConfig = original_config

    assert status == 200
    assert payload["message"] == "Training cleared. Baseline replay restored"
    assert (artifacts / "training-state.json").exists()
    assert (artifacts / "training-summary.json").exists()
    assert (artifacts / "checkpoints" / "checkpoint-000000.json").exists()
    assert (artifacts / "evaluations" / "evaluation-checkpoint-000000.json").exists()
    index_payload = json.loads((generated / "checkpoint-index.json").read_text(encoding="utf-8"))
    assert index_payload["checkpoints"][0]["checkpoint_episode"] == 0
    assert index_payload["latest"]["checkpoint_episode"] == 0
    assert generated.joinpath("latest-replay.json").exists()

    replay_payload = json.loads((generated / "latest-replay.json").read_text(encoding="utf-8"))
    final_snapshot = replay_payload["final_snapshot"]
    assert final_snapshot["grid_width"] == config.environment.width
    assert final_snapshot["grid_height"] == config.environment.height