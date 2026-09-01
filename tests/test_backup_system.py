"""Tests for the multi-tier stage milestone and hourly snapshot backup system."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from sheepdog.config import LabConfig
from sheepdog.server import TrainingManager
from sheepdog.training.backup import TrainingBackupManager


def test_backup_completed_stage(tmp_path: Path) -> None:
    backup_root = tmp_path / "backups"
    manager = TrainingBackupManager(backup_root=backup_root)

    # Create dummy model and sidecar
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = models_dir / "best-model.zip"
    dummy_model.write_text("PK\x03\x04dummy_stage2_model", encoding="utf-8")
    dummy_sidecar = models_dir / "best-model.zip.sidecar.json"
    dummy_sidecar.write_text(json.dumps({"observation_schema_hash": "abc123hash"}), encoding="utf-8")

    chk_payload = {
        "checkpoint_id": "chk_run_1_stage_2",
        "checkpoint_episode": 100,
        "policy_version": 25,
        "success_rate": 0.95,
    }
    eval_payload = {
        "evaluation_id": "eval_1",
        "success_rate": 0.95,
        "average_reward": 120.5,
    }

    manifest = manager.backup_completed_stage(
        stage=2,
        model_path=dummy_model,
        checkpoint_payload=chk_payload,
        evaluation_payload=eval_payload,
        config_dict={"curriculum_stage": 2},
        metrics={"success_rate": 0.95, "best_reward": 120.5},
    )

    assert manifest["stage"] == 2
    assert manifest["has_model"] is True
    assert manifest["success_rate"] == 0.95

    stage_2_dir = backup_root / "stages" / "stage_2"
    assert (stage_2_dir / "stage_2_best_model.zip").exists()
    assert (stage_2_dir / "stage_2_best_model.zip.sidecar.json").exists()
    assert (stage_2_dir / "stage_2_checkpoint.json").exists()
    assert (stage_2_dir / "stage_2_evaluation.json").exists()
    assert (stage_2_dir / "stage_2_manifest.json").exists()

    # Verify listing
    backups_list = manager.list_backups()
    assert len(backups_list["completed_stages"]) == 1
    assert backups_list["completed_stages"][0]["stage"] == 2


def test_backup_hourly_snapshot_and_pruning(tmp_path: Path) -> None:
    backup_root = tmp_path / "backups"
    manager = TrainingBackupManager(backup_root=backup_root)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = models_dir / "active-model.zip"
    dummy_model.write_text("PK\x03\x04active_snapshot_model", encoding="utf-8")

    # Force first snapshot
    snap1 = manager.backup_hourly_snapshot(
        stage=3,
        model_path=dummy_model,
        training_state={"total_timesteps": 50000},
        checkpoint_payload={"success_rate": 0.82},
        interval_seconds=3600.0,
        max_snapshots_per_stage=3,
        force=True,
    )
    assert snap1 is not None
    assert snap1.exists()

    # Second snapshot forced
    time.sleep(0.01)
    snap2 = manager.backup_hourly_snapshot(
        stage=3,
        model_path=dummy_model,
        training_state={"total_timesteps": 100000},
        checkpoint_payload={"success_rate": 0.88},
        interval_seconds=3600.0,
        max_snapshots_per_stage=3,
        force=True,
    )
    assert snap2 is not None

    # Third snapshot forced
    time.sleep(0.01)
    snap3 = manager.backup_hourly_snapshot(
        stage=3,
        model_path=dummy_model,
        training_state={"total_timesteps": 150000},
        checkpoint_payload={"success_rate": 0.91},
        interval_seconds=3600.0,
        max_snapshots_per_stage=3,
        force=True,
    )
    assert snap3 is not None

    # Fourth snapshot should trigger pruning with max_snapshots_per_stage=3
    time.sleep(0.01)
    snap4 = manager.backup_hourly_snapshot(
        stage=3,
        model_path=dummy_model,
        training_state={"total_timesteps": 200000},
        checkpoint_payload={"success_rate": 0.94},
        interval_seconds=3600.0,
        max_snapshots_per_stage=3,
        force=True,
    )
    assert snap4 is not None

    listing = manager.list_backups()
    assert len(listing["hourly_snapshots"]) == 3
    # The oldest (snap1) was pruned, snap4 exists
    assert not snap1.exists()
    assert snap4.exists()


def test_restore_stage_milestone(tmp_path: Path) -> None:
    backup_root = tmp_path / "backups"
    manager = TrainingBackupManager(backup_root=backup_root)

    models_dir = tmp_path / "source_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.zip"
    model_path.write_text("PK\x03\x04stage3_milestone_bytes", encoding="utf-8")

    manager.backup_completed_stage(
        stage=3,
        model_path=model_path,
        checkpoint_payload={"checkpoint_id": "chk_stg3", "policy_version": 42},
    )

    # Now restore into a fresh workspace output dir
    workspace_out = tmp_path / "workspace_artifacts"
    res = manager.restore_stage_milestone(stage=3, output_root=workspace_out)

    assert res["status"] == "success"
    restored_model = workspace_out / "models" / "best-model.zip"
    assert restored_model.exists()
    assert restored_model.read_text(encoding="utf-8") == "PK\x03\x04stage3_milestone_bytes"


def test_backup_protection_against_accidental_deletion(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    backups_dir = artifacts_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = backups_dir / "stages" / "stage_1"
    stage_dir.mkdir(parents=True, exist_ok=True)
    secret_file = stage_dir / "protected.zip"
    secret_file.write_text("PK\x03\x04protected", encoding="utf-8")

    class TestConfig:
        def __new__(cls):
            cfg = LabConfig()
            training_cfg = cfg.training.__class__(
                output_dir=str(artifacts_dir),
                backup_dir=str(backups_dir),
            )
            return cfg.__class__(training=training_cfg)

    with patch("sheepdog.server.LabConfig", TestConfig):
        server_mgr = TrainingManager()
        # Call _remove_path on the backup dir or file
        server_mgr._remove_path(backups_dir)
        assert backups_dir.exists()
        assert secret_file.exists()

        server_mgr._remove_path(secret_file)
        assert secret_file.exists()
