"""Focused regression tests for backend persistence, resume, SQLite, and simulation state fixes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sheepdog.training.episode_store as episode_store_mod
import sheepdog.training.replay_writer as replay_writer_mod
from sheepdog.atomic_io import atomic_replace
from sheepdog.checkpoints.sidecar import get_sidecar_path_for_model
from sheepdog.config import EnvironmentConfig, InstinctRewardConfig, LabConfig, TrainingConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.scenarios import AgentLayout, PenLayout, Scenario
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.backup import TrainingBackupManager
from sheepdog.training.early_warning import EarlyWarningMonitor
from sheepdog.training.episode_store import EpisodeStore, get_episode_store
from sheepdog.training.hierarchical_trainer import HierarchicalMaskablePPOTrainer
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.replay_writer import get_replay_writer


def test_atomic_replace_preserves_dest_on_permission_error(tmp_path: Path) -> None:
    """Verify atomic_replace never truncates dest when os.replace raises PermissionError."""
    dest = tmp_path / "dest.json"
    tmp = tmp_path / "dest.tmp"
    dest.write_text('{"preserved": true}', encoding="utf-8")
    tmp.write_text('{"new": true}', encoding="utf-8")

    with (
        patch("sheepdog.atomic_io.os.replace", side_effect=PermissionError("locked")) as mock_rep,
        patch("sheepdog.atomic_io.time.sleep"),
        pytest.raises(PermissionError),
    ):
        atomic_replace(tmp, dest)

    assert mock_rep.call_count == 30
    assert dest.read_text(encoding="utf-8") == '{"preserved": true}'
    assert not tmp.exists()


def test_config_checkpoint_timesteps_normalized_and_curriculum_stage_synced() -> None:
    """Verify LabConfig.from_dict normalizes checkpoint_timesteps and apply_curriculum_stage syncs stage."""
    cfg = LabConfig.from_dict(
        {
            "training": {
                "checkpoint_episodes": [10, 20],
                "checkpoint_timesteps": [1000, 2000],
                "neural_hidden_sizes": [256, 128],
            }
        }
    )
    assert cfg.training.checkpoint_timesteps == (1000, 2000)
    assert isinstance(cfg.training.checkpoint_timesteps, tuple)

    base = LabConfig()
    staged = apply_curriculum_stage(base, stage=5)
    assert staged.environment.curriculum_stage == 5
    assert staged.rewards.instincts.curriculum_stage == 5


def test_environment_and_team_strategy_reset_clear_per_episode_state() -> None:
    """Verify reset() and reset_from_scenario() clear sticky pen direction, wall stalls, and sync geometry."""
    env_cfg = EnvironmentConfig(width=40, height=30, dogs=1, sheep=1)
    strategy = TeamStrategy(40, 30, env_cfg)
    strategy._sticky_pen_dx = -1
    strategy._sticky_pen_dy = 1
    strategy.reset()
    assert (strategy._sticky_pen_dx, strategy._sticky_pen_dy) == (1, 0)

    env = SheepdogEnvironment(LabConfig(environment=env_cfg))
    env.reset(seed=7)
    env._team_strategy._sticky_pen_dx = -1
    env._team_strategy._sticky_pen_dy = -1
    env._wall_stall_steps = 9
    env._max_wall_stall_steps = 12
    env._sheep_wall_stalls = {0: 5}
    env._wall_lateral_movements = 4
    env._wall_inward_recoveries = 3
    env._corner_occupancy_steps = 7

    env.reset(seed=8)
    assert (env._team_strategy._sticky_pen_dx, env._team_strategy._sticky_pen_dy) == (1, 0)
    assert env._wall_stall_steps == 0
    assert env._max_wall_stall_steps == 0
    assert env._sheep_wall_stalls == {}
    assert env._wall_lateral_movements == 0
    assert env._wall_inward_recoveries == 0
    assert env._corner_occupancy_steps == 0

    # Dirty again and reset from a Scenario with different dimensions and agent counts
    env._team_strategy._sticky_pen_dx = 0
    env._team_strategy._sticky_pen_dy = -1
    env._wall_stall_steps = 5
    env._corner_occupancy_steps = 4
    scenario = Scenario(
        id="scen_test",
        name="Scenario Resize",
        created_at="2026-11-12T00:00:00Z",
        seed=42,
        width=60,
        height=45,
        dogs=(AgentLayout(index=0, x=5, y=5), AgentLayout(index=1, x=6, y=5)),
        sheep=(
            AgentLayout(index=0, x=20, y=20),
            AgentLayout(index=1, x=22, y=20),
            AgentLayout(index=2, x=24, y=20),
        ),
        pen=PenLayout(origin_x=48, origin_y=18, width=8, height=8, opening="left"),
    )
    env.reset_from_scenario(scenario)
    assert env.env_config.width == 60
    assert env.env_config.height == 45
    assert env.env_config.dogs == 2
    assert env.env_config.sheep == 3
    assert env._team_strategy._width == 60
    assert env._team_strategy._height == 45
    assert env._team_strategy._config.dogs == 2
    assert env._team_strategy._config.sheep == 3
    assert (env._team_strategy._sticky_pen_dx, env._team_strategy._sticky_pen_dy) == (1, 0)
    assert env._wall_stall_steps == 0
    assert env._corner_occupancy_steps == 0


def test_maskable_ppo_stage9_no_fallback_and_hidden_size_validation(tmp_path: Path) -> None:
    """Verify Stage 9+ avoids models/best-model.zip fallback, checks hidden_sizes, and flags corrupt state."""
    artifacts = tmp_path / "artifacts"
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True)
    stale_best = models_dir / "best-model.zip"
    stale_best.write_bytes(b"PK\x03\x04stale")

    # 1. Stage 9+ with missing stage checkpoint must NOT fall back to models/best-model.zip
    (artifacts / "training-state.json").write_text(
        json.dumps(
            {
                "total_episodes_trained": 100,
                "policy_state_path": str(artifacts / "checkpoints" / "stage12" / "missing.zip"),
                "policy_config": {"hidden_sizes": [256, 256, 128], "observation_size": 54, "action_size": 9},
            }
        ),
        encoding="utf-8",
    )
    cfg_s12 = replace(
        LabConfig(),
        rewards=replace(LabConfig().rewards, instincts=InstinctRewardConfig(curriculum_stage=12)),
        training=replace(LabConfig().training, output_dir=str(artifacts), web_export_dir=str(tmp_path / "web")),
    )
    trainer_s12 = MaskablePPOTrainer(cfg_s12, artifacts)
    assert trainer_s12.has_compatible_policy_state() is False

    # 2. Existing checkpoint with mismatched hidden_sizes must return False
    stage12_dir = artifacts / "checkpoints" / "stage12"
    stage12_dir.mkdir(parents=True)
    valid_zip = stage12_dir / "stage12-best-model.zip"
    valid_zip.write_bytes(b"PK\x03\x04stage12")
    (artifacts / "training-state.json").write_text(
        json.dumps(
            {
                "total_episodes_trained": 100,
                "policy_state_path": str(valid_zip),
                "policy_config": {"hidden_sizes": [64, 64], "observation_size": 54, "action_size": 9},
            }
        ),
        encoding="utf-8",
    )
    assert MaskablePPOTrainer(cfg_s12, artifacts).has_compatible_policy_state() is False

    # 3. Corrupt training-state.json records _state_load_error
    (artifacts / "training-state.json").write_text("{corrupt json", encoding="utf-8")
    corrupt_trainer = MaskablePPOTrainer(cfg_s12, artifacts)
    assert "_state_load_error" in corrupt_trainer._loaded_state


def test_maskable_ppo_resume_prefers_policy_state_on_incomplete_batch(tmp_path: Path) -> None:
    """Verify train() loads policy_state_path rather than best_model_path when incomplete_batch is True."""
    artifacts = tmp_path / "artifacts"
    stage_dir = artifacts / "checkpoints" / "stage10"
    stage_dir.mkdir(parents=True)
    inprog_zip = stage_dir / "inprogress-policy.zip"
    best_zip = stage_dir / "stage10-best-model.zip"
    inprog_zip.write_bytes(b"PK\x03\x04inprog")
    best_zip.write_bytes(b"PK\x03\x04best")

    state_payload = {
        "total_episodes_trained": 50,
        "incomplete_batch": True,
        "policy_state_path": str(inprog_zip),
        "best_model_path": str(best_zip),
        "best_model_curriculum_stage": 10,
        "policy_config": {"hidden_sizes": [256, 256, 128], "observation_size": 54, "action_size": 9},
    }
    (artifacts / "training-state.json").write_text(json.dumps(state_payload), encoding="utf-8")

    cfg = replace(
        LabConfig(),
        rewards=replace(LabConfig().rewards, instincts=InstinctRewardConfig(curriculum_stage=10)),
        training=TrainingConfig(
            checkpoint_episodes=(1,),
            total_timesteps=64,
            output_dir=str(artifacts),
            web_export_dir=str(tmp_path / "web"),
        ),
    )
    trainer = MaskablePPOTrainer(cfg, artifacts)
    mock_policy = MagicMock()
    mock_policy.config.to_dict.return_value = {"hidden_sizes": [256, 256, 128]}
    with patch.object(MaskablePPOTrainer, "POLICY_CLASS") as mock_cls:
        mock_cls.load.return_value = mock_policy
        trainer.train(should_stop=lambda: True)
        loaded_arg = Path(mock_cls.load.call_args[0][0]).resolve()
        assert loaded_arg == inprog_zip.resolve()


def test_hierarchical_trainer_checks_file_existence_and_hidden_sizes(tmp_path: Path) -> None:
    """Verify HierarchicalMaskablePPOTrainer checks policy_state_path existence and hidden_sizes."""
    artifacts = tmp_path / "hier"
    artifacts.mkdir(parents=True)
    cfg = LabConfig()
    probe = HierarchicalMaskablePPOTrainer(cfg, artifacts)
    sig = probe._training_signature()

    model_file = artifacts / "shepherd.zip"
    state = {
        "total_episodes_trained": 5,
        "total_timesteps": 100,
        "policy_state_path": str(model_file),
        "policy_config": {"hidden_sizes": list(cfg.training.neural_hidden_sizes)},
        "training_signature": sig,
    }
    state_file = artifacts / HierarchicalMaskablePPOTrainer.STATE_FILENAME
    state_file.write_text(json.dumps(state), encoding="utf-8")
    # Missing model file on disk -> incompatible
    assert HierarchicalMaskablePPOTrainer(cfg, artifacts).has_compatible_policy_state() is False

    # Present model file + matching hidden_sizes -> compatible
    model_file.write_bytes(b"PK\x03\x04hier")
    assert HierarchicalMaskablePPOTrainer(cfg, artifacts).has_compatible_policy_state() is True

    # Mismatched hidden_sizes -> incompatible
    state["policy_config"] = {"hidden_sizes": [64, 64]}
    state_file.write_text(json.dumps(state), encoding="utf-8")
    assert HierarchicalMaskablePPOTrainer(cfg, artifacts).has_compatible_policy_state() is False


def test_backup_manager_canonical_sidecar_and_stage9_restore(tmp_path: Path) -> None:
    """Verify TrainingBackupManager preserves canonical sidecars and restores Stage 9+ checkpoints & state."""
    out_root = tmp_path / "artifacts"
    backup_root = out_root / "backups"
    src_dir = tmp_path / "src_models"
    src_dir.mkdir(parents=True)
    src_zip = src_dir / "stage11-best-model.zip"
    src_zip.write_bytes(b"PK\x03\x04stage11")
    src_sidecar = get_sidecar_path_for_model(src_zip)
    src_sidecar.write_text('{"stage": 11}', encoding="utf-8")

    mgr = TrainingBackupManager(backup_root=backup_root, output_root=out_root)
    mgr.backup_completed_stage(
        stage=11,
        model_path=src_zip,
        checkpoint_payload={"training_state": {"total_timesteps": 5000}},
    )
    res_stage = mgr.restore_stage_milestone(stage=11)
    expected_stage_zip = out_root / "checkpoints" / "stage11" / "stage11-best-model.zip"
    assert Path(res_stage["restored_model_path"]) == expected_stage_zip
    assert expected_stage_zip.exists()
    assert get_sidecar_path_for_model(expected_stage_zip).exists()
    assert (out_root / "training-state.json").exists()

    snap_meta = mgr.backup_hourly_snapshot(
        stage=12,
        model_path=src_zip,
        training_state={"total_timesteps": 9999, "policy_state_path": "checkpoints/stage12/active.zip"},
        force=True,
    )
    assert snap_meta is not None
    res_snap = mgr.restore_hourly_snapshot(snap_meta.stem)
    expected_s12_zip = out_root / "checkpoints" / "stage12" / "stage12-best-model.zip"
    expected_active = out_root / "checkpoints" / "stage12" / "active.zip"
    assert Path(res_snap["restored_model_path"]) == expected_s12_zip
    assert expected_s12_zip.exists()
    assert expected_active.exists()
    assert get_sidecar_path_for_model(expected_active).exists()
    restored_state = json.loads((out_root / "training-state.json").read_text(encoding="utf-8"))
    assert restored_state["total_timesteps"] == 9999


def test_replay_writer_wires_episode_store_on_existing_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify get_replay_writer attaches episode_store when existing writer had None."""
    monkeypatch.setattr(replay_writer_mod, "_global_replay_writer", None)
    writer1 = get_replay_writer(output_dir=tmp_path / "replays", episode_store=None)
    try:
        assert writer1.episode_store is None
        sentinel_store = object()
        writer2 = get_replay_writer(output_dir=tmp_path / "replays", episode_store=sentinel_store)
        assert writer2 is writer1
        assert writer2.episode_store is sentinel_store
    finally:
        writer1.close()


def test_episode_store_read_only_connection_close_and_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify EpisodeStore closes connections, honors read_only=True, and preserves shutdown sentinel."""
    missing_db = tmp_path / "nonexistent_dir" / "missing.sqlite"
    ro_missing = EpisodeStore(db_path=missing_db, read_only=True)
    assert not missing_db.parent.exists()
    assert ro_missing._worker_thread is None
    assert ro_missing.get_max_episode() == 0
    assert ro_missing.get_episodes()["episodes"] == []
    assert ro_missing.get_telemetry_summary()["window_count"] == 0
    assert ro_missing.get_stage_diagnostics(stage=3)["total_episodes"] == 0
    assert ro_missing.get_recent_failed_episodes_with_replays() == []
    assert ro_missing.get_episode_by_id_or_replay_id("ep_1") is None

    db_path = tmp_path / "telemetry.sqlite"
    monkeypatch.setattr(episode_store_mod, "_global_episode_store", None)
    monkeypatch.setattr(episode_store_mod, "_global_ro_episode_stores", {})

    rw_store = get_episode_store(db_path=db_path, read_only=False)
    try:
        # Context manager closes SQLite connection upon exiting block
        with rw_store._get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

        rw_store.add_episode(
            {
                "event_key": "ev_ro_1",
                "run_id": "run_1",
                "global_environment_episode": 12,
                "curriculum_stage": 4,
                "reward": -5.0,
                "result": "TIMEOUT",
                "success": False,
                "timeout": True,
                "replay_available": 1,
                "replay_id": "missing_replay_file",
                "replay_path": str(tmp_path / "does_not_exist.json"),
                "capture_status": "queued",
            }
        )
        rw_store.flush()

        # Read-only singleton does not overwrite write singleton or mutate DB rows
        ro_store = get_episode_store(db_path=db_path, read_only=True)
        assert ro_store.read_only is True
        assert get_episode_store(db_path=db_path, read_only=False) is rw_store
        assert ro_store.get_max_episode() == 12
        summary = ro_store.get_telemetry_summary(stage=4)
        assert summary["window_count"] == 1
        assert summary["current_stage_environment_episode"] == 12

        # Querying failed replays in read_only mode must NOT mark missing file as 'pruned' in DB
        assert ro_store.get_recent_failed_episodes_with_replays() == []
        rec = ro_store.get_episode_by_id_or_replay_id("ev_ro_1")
        assert rec is not None
        assert rec["capture_status"] == "queued"

        # Verify flush() preserves None sentinel when worker is stopped
        rw_store._stop_event.set()
        if rw_store._worker_thread is not None:
            rw_store._queue.put_nowait(None)
            rw_store._worker_thread.join(timeout=2.0)
        rw_store.add_episode(
            {
                "event_key": "ev_ro_2",
                "run_id": "run_1",
                "global_environment_episode": 13,
                "curriculum_stage": 4,
                "reward": 10.0,
                "result": "SUCCESS",
                "success": True,
            }
        )
        rw_store._queue.put_nowait(None)
        rw_store.flush()
        assert rw_store._queue.get_nowait() is None
        assert rw_store.get_max_episode() == 13
    finally:
        rw_store.close()


def test_early_warning_acknowledge_invalidates_eval_cache(tmp_path: Path) -> None:
    """Verify acknowledge_warning clears _eval_cache so next poll reflects acknowledged status."""
    eval_dir = tmp_path / "evaluations"
    diag_dir = eval_dir / "diagnostics"
    diag_dir.mkdir(parents=True)

    eval_id = "eval_chk_100_quick_abc"
    eval_payload = {
        "evaluation_id": eval_id,
        "checkpoint_id": "chk_100",
        "checkpoint_episode": 100,
        "policy_version": 10,
        "curriculum_stage": 5,
        "success_rate": 0.4,
        "run_id": "run_1",
        "evaluation_seed_set_id": "seeds_a",
        "evaluation_seed_count": 2,
        "policy_mode": "guided",
        "policy_type": "neural",
        "environment_config_hash": "hash_1",
        "episode_budget": 400,
        "max_steps": 400,
        "evaluation_mode": "quick",
        "is_complete": True,
        "evaluation_timestamp": "2026-11-12T10:00:00Z",
        "records": [
            {"seed": 11, "success": False, "timeout": True, "replay_path": None},
            {"seed": 23, "success": True, "timeout": False, "replay_path": None},
        ],
    }
    diag_payload = {
        "metrics_version": "2.0",
        "evaluation_id": eval_id,
        "checkpoint_episode": 100,
        "curriculum_stage": 5,
        "total_world_steps": 800,
        "repeated_movement_steps": 500,
        "repeated_movement_fraction": 0.625,
        "stalls_count": 6,
        "recovered_stalls_count": 1,
        "recovery_rate": 0.1667,
        "representative_seed": 11,
        "representative_replay_path": None,
        "representative_step": 10,
        "representative_interval_id": "iv_0",
    }
    (eval_dir / f"{eval_id}.json").write_text(json.dumps(eval_payload), encoding="utf-8")
    (diag_dir / f"{eval_id}.diagnostics.json").write_text(json.dumps(diag_payload), encoding="utf-8")

    monitor = EarlyWarningMonitor(tmp_path)
    first = monitor.get_warnings_summary(current_stage=5)
    active_warnings = first["active_warnings"]
    assert active_warnings
    warning_id = active_warnings[0]["id"]

    assert monitor.acknowledge_warning(warning_id, note="reviewed") is True
    second = monitor.get_warnings_summary(current_stage=5)
    assert all(w["id"] != warning_id for w in second["active_warnings"])
    matching = [w for w in second["acknowledged_warnings"] if w["id"] == warning_id]
    assert matching
    assert matching[0]["status"] == "acknowledged"
