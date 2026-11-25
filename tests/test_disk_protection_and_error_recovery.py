"""Focused tests for disk-space protection, temp cleanup, pruning, backup hardening, and error recovery."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import namedtuple
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheepdog import server as server_module
from sheepdog.atomic_io import (
    DEFAULT_MIN_FREE_DISK_BYTES,
    DEFAULT_WARN_FREE_DISK_BYTES,
    DiskSpaceError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    cleanup_orphaned_tmp_files,
    ensure_free_disk_space,
    get_free_disk_bytes,
    get_min_free_disk_bytes,
)
from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import EvaluationSummary
from sheepdog.server import ThreadingHTTPServer, TrainingManager, TrainingRequestHandler
from sheepdog.training.backup import TrainingBackupManager
from sheepdog.training.factory import create_trainer
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.runtime import TrainingRuntimeTracker

_FakeDiskUsage = namedtuple("_FakeDiskUsage", ["total", "used", "free"])


def _shutdown_manager(mgr: TrainingManager) -> None:
    """Stop background threads owned by a TrainingManager instance in tests."""
    watchdog_stop = getattr(mgr, "_watchdog_stop_event", None)
    if watchdog_stop is not None:
        with contextlib.suppress(Exception):
            watchdog_stop.set()
    thread = getattr(mgr, "_thread", None)
    if thread is not None and thread.is_alive():
        with contextlib.suppress(Exception):
            mgr._control_request = "stopped"
            thread.join(timeout=1.0)
    runtime_tracker = getattr(mgr, "runtime_tracker", None)
    if runtime_tracker is not None:
        with contextlib.suppress(Exception):
            runtime_tracker.end_session("test_cleanup")
        rt_stop = getattr(runtime_tracker, "_stop_event", None)
        if rt_stop is not None:
            with contextlib.suppress(Exception):
                rt_stop.set()


@pytest.fixture(autouse=True)
def _reset_handler_and_startup_state() -> Iterator[None]:
    """Keep TrainingRequestHandler singleton and startup gate isolated per test."""
    TrainingRequestHandler._manager_instance = None
    server_module._MANAGER_STARTUP.finish()
    server_module._MANAGER_STARTUP.error = None
    yield
    existing = getattr(TrainingRequestHandler, "_manager_instance", None)
    if isinstance(existing, TrainingManager):
        _shutdown_manager(existing)
    TrainingRequestHandler._manager_instance = None
    server_module._MANAGER_STARTUP.finish()
    server_module._MANAGER_STARTUP.error = None


def _make_lab_config(tmp_path: Path, **training_overrides: Any) -> LabConfig:
    training_kwargs: dict[str, Any] = {
        "episodes": 1,
        "output_dir": str(tmp_path / "artifacts"),
        "web_export_dir": str(tmp_path / "web" / "generated"),
        "backup_dir": str(tmp_path / "artifacts" / "backups"),
        "runtime_heartbeat_seconds": 60.0,
    }
    training_kwargs.update(training_overrides)
    return LabConfig(
        environment=EnvironmentConfig(max_steps=10, dogs=2, sheep=2),
        rewards=RewardConfig(),
        training=TrainingConfig(**training_kwargs),
    )


def test_atomic_writes_cleanup_tmp_on_enospc_write_and_replace(tmp_path: Path) -> None:
    """Verify atomic JSON/text/bytes writes remove .tmp files and preserve targets on ENOSPC."""
    json_target = tmp_path / "state.json"
    text_target = tmp_path / "notes.txt"
    bytes_target = tmp_path / "replay.bin"
    json_target.write_text('{"preserved": true}', encoding="utf-8")
    text_target.write_text("original-text", encoding="utf-8")
    bytes_target.write_bytes(b"original-bytes")

    enospc = OSError(errno.ENOSPC, "No space left on device")

    # 1. Failure during write phase
    with (
        patch("sheepdog.atomic_io.json.dump", side_effect=enospc),
        pytest.raises(OSError) as exc_info,
    ):
        atomic_write_json(json_target, {"new": True})
    assert exc_info.value.errno == errno.ENOSPC
    assert json_target.read_text(encoding="utf-8") == '{"preserved": true}'
    assert list(tmp_path.glob("*.tmp")) == []

    with (
        patch.object(Path, "write_text", side_effect=enospc),
        pytest.raises(OSError) as exc_info,
    ):
        atomic_write_text(text_target, "replacement-text")
    assert exc_info.value.errno == errno.ENOSPC
    assert text_target.read_text(encoding="utf-8") == "original-text"
    assert list(tmp_path.glob("*.tmp")) == []

    with (
        patch.object(Path, "write_bytes", side_effect=enospc),
        pytest.raises(OSError) as exc_info,
    ):
        atomic_write_bytes(bytes_target, b"replacement-bytes")
    assert exc_info.value.errno == errno.ENOSPC
    assert bytes_target.read_bytes() == b"original-bytes"
    assert list(tmp_path.glob("*.tmp")) == []

    # 2. Failure during os.replace phase
    with patch("sheepdog.atomic_io.os.replace", side_effect=enospc):
        with pytest.raises(OSError) as exc_info:
            atomic_write_json(json_target, {"new": True})
        assert exc_info.value.errno == errno.ENOSPC
        assert json_target.read_text(encoding="utf-8") == '{"preserved": true}'
        assert list(tmp_path.glob("*.tmp")) == []

        with pytest.raises(OSError) as exc_info:
            atomic_write_text(text_target, "replacement-text")
        assert exc_info.value.errno == errno.ENOSPC
        assert text_target.read_text(encoding="utf-8") == "original-text"
        assert list(tmp_path.glob("*.tmp")) == []

        with pytest.raises(OSError) as exc_info:
            atomic_write_bytes(bytes_target, b"replacement-bytes")
        assert exc_info.value.errno == errno.ENOSPC
        assert bytes_target.read_bytes() == b"original-bytes"
        assert list(tmp_path.glob("*.tmp")) == []


def test_cleanup_orphaned_tmp_files_respects_age_and_extension(tmp_path: Path) -> None:
    """Verify cleanup_orphaned_tmp_files deletes stale *.tmp files and keeps recent or non-.tmp files."""
    nested = tmp_path / "nested"
    nested.mkdir()
    stale_root_tmp = tmp_path / "orphan-1.tmp"
    stale_nested_tmp = nested / "orphan-2.tmp"
    recent_tmp = tmp_path / "active.tmp"
    old_json = tmp_path / "state.json"

    for path in (stale_root_tmp, stale_nested_tmp, recent_tmp, old_json):
        path.write_text("data", encoding="utf-8")

    now = time.time()
    os.utime(stale_root_tmp, (now - 600.0, now - 600.0))
    os.utime(stale_nested_tmp, (now - 301.0, now - 301.0))
    os.utime(old_json, (now - 600.0, now - 600.0))
    os.utime(recent_tmp, (now - 10.0, now - 10.0))

    missing_root = tmp_path / "does-not-exist"
    removed = cleanup_orphaned_tmp_files([tmp_path, missing_root], min_age_seconds=300.0)

    assert removed == 2
    assert not stale_root_tmp.exists()
    assert not stale_nested_tmp.exists()
    assert recent_tmp.exists()
    assert old_json.exists()


def test_export_web_assets_and_reconcile_prune_unreferenced_replays(tmp_path: Path) -> None:
    """Verify _export_web_assets and reconcile_web_exports prune stale web replays safely."""
    config = _make_lab_config(
        tmp_path,
        trainer_type="hill_climb",
        policy_type="linear",
        checkpoint_episodes=(0,),
        evaluation_seeds=(11,),
    )
    trainer = create_trainer(config, config.training.output_dir)
    artifacts_dir = Path(config.training.output_dir)
    web_dir = Path(config.training.web_export_dir)
    replays_src_dir = artifacts_dir / "replays"
    replays_src_dir.mkdir(parents=True, exist_ok=True)
    cp_path = artifacts_dir / "checkpoints" / "checkpoint-000011.json"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text("{}", encoding="utf-8")

    checkpoint_payloads: list[dict[str, Any]] = []
    for idx in range(12):
        src_replay = replays_src_dir / f"ep_{idx:04d}_seed_11.json"
        src_replay.write_text(json.dumps({"episode": idx}), encoding="utf-8")
        checkpoint_payloads.append(
            {
                "checkpoint_episode": idx,
                "checkpoint": f"checkpoint-{idx:06d}.json",
                "records": [{"seed": 11, "replay_path": str(src_replay)}],
            }
        )

    latest_eval_replay = replays_src_dir / "latest_eval_only.json"
    latest_eval_replay.write_text('{"episode": "latest_eval"}', encoding="utf-8")
    summary = EvaluationSummary(
        checkpoint_episode=11,
        policy_name="heuristic",
        records=({"seed": 99, "replay_path": str(latest_eval_replay)},),  # type: ignore[arg-type]
        success_rate=0.0,
        timeout_rate=1.0,
        average_completion_steps=10.0,
        average_completion_seconds=5.0,
        average_sheep_penned=0.0,
        average_reward=-1.0,
    )

    web_replays_dir = web_dir / "replays"
    web_replays_dir.mkdir(parents=True, exist_ok=True)
    orphan_replay = web_replays_dir / "stale_orphan.json"
    orphan_replay.write_text("{}", encoding="utf-8")
    # Pre-seed an old checkpoint replay that is now outside the kept latest 10
    old_exported = web_replays_dir / "ep_0000_seed_11.json"
    old_exported.write_text("{}", encoding="utf-8")

    trainer._export_web_assets(web_dir, checkpoint_payloads, summary, latest_eval_replay, cp_path)

    assert not orphan_replay.exists()
    assert not (web_replays_dir / "ep_0000_seed_11.json").exists()
    assert not (web_replays_dir / "ep_0001_seed_11.json").exists()
    assert (web_replays_dir / "latest_eval_only.json").exists()
    for kept_idx in range(2, 12):
        assert (web_replays_dir / f"ep_{kept_idx:04d}_seed_11.json").exists()

    # Now verify reconcile_web_exports also prunes unreferenced files when catching up
    extra_orphan = web_replays_dir / "orphan_during_reconcile.json"
    extra_orphan.write_text("{}", encoding="utf-8")
    summary_file = artifacts_dir / "training-summary.json"
    summary_file.write_text(json.dumps({"checkpoints": checkpoint_payloads}), encoding="utf-8")
    # Make checkpoint-index lag behind so reconcile_web_exports executes
    (web_dir / "checkpoint-index.json").write_text(
        json.dumps({"latest": {"checkpoint_episode": 1}, "checkpoints": []}),
        encoding="utf-8",
    )

    reconciled_ep = trainer.reconcile_web_exports(web_dir)
    assert reconciled_ep == 11
    assert not extra_orphan.exists()
    assert not (web_replays_dir / "ep_0000_seed_11.json").exists()
    for kept_idx in range(2, 12):
        assert (web_replays_dir / f"ep_{kept_idx:04d}_seed_11.json").exists()


def test_backup_hourly_snapshot_cleans_partial_files_on_enospc(tmp_path: Path) -> None:
    """Verify backup_hourly_snapshot and backup_completed_stage leave no partial zip/tmp on ENOSPC."""
    backup_root = tmp_path / "backups"
    mgr = TrainingBackupManager(backup_root=backup_root)

    src_dir = tmp_path / "models"
    src_dir.mkdir(parents=True)
    src_model = src_dir / "model.zip"
    src_model.write_bytes(b"PK\x03\x04model-bytes")
    src_sidecar = src_dir / "model.sidecar.json"
    src_sidecar.write_text('{"schema": 1}', encoding="utf-8")

    # 1. Direct copy failure on model zip
    with patch(
        "sheepdog.training.backup.shutil.copy2",
        side_effect=OSError(errno.ENOSPC, "No space left on device"),
    ):
        meta_path = mgr.backup_hourly_snapshot(
            stage=8,
            model_path=src_model,
            force=True,
        )
    assert meta_path is not None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model_saved"] is False
    assert list(mgr.hourly_dir.glob("*.zip")) == []
    assert list(mgr.hourly_dir.glob("*.tmp")) == []

    # 2. Model copy succeeds, sidecar copy fails -> partial destination .zip must be removed
    orig_copy2 = shutil.copy2
    call_count = 0

    def _fail_on_sidecar(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return orig_copy2(src, dst, *args, **kwargs)

    with patch("sheepdog.training.backup.shutil.copy2", side_effect=_fail_on_sidecar):
        meta_path_2 = mgr.backup_hourly_snapshot(
            stage=8,
            model_path=src_model,
            force=True,
        )
    assert meta_path_2 is not None
    meta_2 = json.loads(meta_path_2.read_text(encoding="utf-8"))
    assert meta_2["model_saved"] is False
    assert list(mgr.hourly_dir.glob("*.zip")) == []
    assert list(mgr.hourly_dir.glob("*.tmp")) == []

    # 3. Stage milestone backup also cleans up partial files on sidecar ENOSPC
    call_count = 0
    with patch("sheepdog.training.backup.shutil.copy2", side_effect=_fail_on_sidecar):
        manifest = mgr.backup_completed_stage(stage=8, model_path=src_model)
    assert manifest["has_model"] is False
    assert manifest["model_path"] is None
    stage8_dir = mgr.stages_dir / "stage_8"
    assert list(stage8_dir.glob("*.zip")) == []
    assert list(stage8_dir.glob("*.tmp")) == []


def test_backup_manager_seeds_hourly_watermark_across_instances(tmp_path: Path) -> None:
    """Verify fresh TrainingBackupManager instances respect interval_seconds via disk watermark."""
    backup_root = tmp_path / "backups"
    src_model = tmp_path / "best-model.zip"
    src_model.write_bytes(b"PK\x03\x04valid")

    mgr1 = TrainingBackupManager(backup_root=backup_root)
    snap1 = mgr1.backup_hourly_snapshot(
        stage=8,
        model_path=src_model,
        active_runtime_seconds=1200.0,
        interval_seconds=3600.0,
        force=True,
    )
    assert snap1 is not None and snap1.exists()

    # Corrupt a newer file to verify fallback to the latest valid snapshot
    corrupt_snap = mgr1.hourly_dir / "snapshot_stage8_99999999_999999_000000.json"
    corrupt_snap.write_text("{corrupt json", encoding="utf-8")
    now = time.time()
    os.utime(snap1, (now - 10.0, now - 10.0))
    os.utime(corrupt_snap, (now - 1.0, now - 1.0))

    mgr2 = TrainingBackupManager(backup_root=backup_root)
    assert mgr2._last_hourly_active_seconds == 1200.0
    assert abs(mgr2._last_hourly_backup_time - (now - 10.0)) < 1.0

    # Within both wall-clock and active-runtime intervals -> skipped
    skipped = mgr2.backup_hourly_snapshot(
        stage=8,
        model_path=src_model,
        active_runtime_seconds=1260.0,
        interval_seconds=3600.0,
        force=False,
    )
    assert skipped is None

    # Advancing active runtime past interval_seconds -> snapshot created
    created = mgr2.backup_hourly_snapshot(
        stage=8,
        model_path=src_model,
        active_runtime_seconds=4850.0,
        interval_seconds=3600.0,
        force=False,
    )
    assert created is not None and created.exists()


def test_runtime_tracker_end_session_completes_when_persist_hits_enospc(tmp_path: Path) -> None:
    """Verify TrainingRuntimeTracker.end_session clears session and stops heartbeat on ENOSPC."""
    enospc = OSError(errno.ENOSPC, "No space left on device")

    # 1. ENOSPC inside atomic_write_json
    tracker1 = TrainingRuntimeTracker(
        tmp_path / "runtime1.json",
        start_heartbeat_thread=False,
    )
    tracker1.start_session("run-1", session_id="sess-1")
    with patch("sheepdog.training.runtime.atomic_write_json", side_effect=enospc):
        tracker1.heartbeat()
        tracker1.end_session("error")

    assert tracker1._session is None
    assert tracker1._phase_started_monotonic is None
    assert tracker1._stop_event.is_set()
    assert tracker1._heartbeat_thread is None
    assert tracker1._payload["sessions"][-1]["status"] == "ended"
    assert tracker1._payload["sessions"][-1]["end_reason"] == "error"

    # 2. Direct _persist() mock raising ENOSPC
    tracker2 = TrainingRuntimeTracker(
        tmp_path / "runtime2.json",
        start_heartbeat_thread=False,
    )
    tracker2.start_session("run-2", session_id="sess-2")
    with patch.object(tracker2, "_persist", side_effect=enospc):
        tracker2.end_session("error")

    assert tracker2._session is None
    assert tracker2._phase_started_monotonic is None
    assert tracker2._stop_event.is_set()
    assert tracker2._heartbeat_thread is None


def test_training_manager_run_training_handles_enospc_without_double_fault(tmp_path: Path) -> None:
    """Verify _run_training recovers cleanly when trainer.train and session persistence hit ENOSPC."""
    cfg = _make_lab_config(tmp_path)

    class TestConfig:
        def __new__(cls) -> LabConfig:
            return cfg

    class EnospcTrainer:
        total_episodes_trained = 0

        def train(self, progress_callback: Any = None, should_stop: Any = None) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=EnospcTrainer()),
    ):
        manager = TrainingManager()
        try:
            enospc = OSError(errno.ENOSPC, "No space left on device")
            clear_calls = 0
            orig_clear = manager._clear_training_session

            def _spy_clear() -> None:
                nonlocal clear_calls
                clear_calls += 1
                orig_clear()

            with (
                patch.object(manager, "_persist_training_session", side_effect=enospc),
                patch.object(manager, "_clear_training_session", side_effect=_spy_clear),
                patch.object(
                    manager.runtime_tracker,
                    "end_session",
                    wraps=manager.runtime_tracker.end_session,
                ) as spy_end_session,
            ):
                manager._run_training(1, True)

            snap = manager.snapshot()
            assert snap["running"] is False
            assert snap["phase"] == "error"
            assert snap["error_type"] == "OSError"
            assert clear_calls == 1
            spy_end_session.assert_called_once_with("error")
        finally:
            _shutdown_manager(manager)


def test_training_manager_start_rejects_low_disk_space(tmp_path: Path) -> None:
    """Verify TrainingManager.start rejects training with ValueError and spawns no thread when disk is full."""
    cfg = _make_lab_config(tmp_path)

    class TestConfig:
        def __new__(cls) -> LabConfig:
            return cfg

    low_usage = _FakeDiskUsage(total=100 * 1024**3, used=99 * 1024**3, free=10 * 1024 * 1024)
    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.atomic_io.shutil.disk_usage", return_value=low_usage),
    ):
        manager = TrainingManager()
        try:
            with pytest.raises(ValueError, match="Insufficient free disk space"):
                manager.start(requested_episodes=1)
            assert manager._thread is None
            assert manager.snapshot()["running"] is False

            # Also verify HTTP POST /api/training/start maps this to 409 TRAINING_START_REJECTED
            with patch.object(TrainingRequestHandler, "manager", manager):
                httpd = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
                port = httpd.server_port
                srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                srv_thread.start()
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/training/start",
                        data=json.dumps({"episodes": 1}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with pytest.raises(urllib.error.HTTPError) as http_exc:
                        urllib.request.urlopen(req, timeout=5)
                    assert http_exc.value.code == 409
                    payload = json.loads(http_exc.value.read().decode("utf-8"))
                    assert payload["code"] == "TRAINING_START_REJECTED"
                    assert "Insufficient free disk space" in payload["error"]
                finally:
                    httpd.shutdown()
                    httpd.server_close()
        finally:
            _shutdown_manager(manager)


def test_disk_helpers_pre_segment_guard_and_health_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify disk helper functions, MaskablePPO pre-segment guard, and /api/health telemetry."""
    # 1. Env threshold parsing
    monkeypatch.delenv("SHEEPDOG_MIN_FREE_DISK_MB", raising=False)
    assert get_min_free_disk_bytes() == DEFAULT_MIN_FREE_DISK_BYTES
    monkeypatch.setenv("SHEEPDOG_MIN_FREE_DISK_MB", "128")
    assert get_min_free_disk_bytes() == 128 * 1024 * 1024
    monkeypatch.setenv("SHEEPDOG_MIN_FREE_DISK_MB", "0.0")
    assert get_min_free_disk_bytes() == 0
    monkeypatch.setenv("SHEEPDOG_MIN_FREE_DISK_MB", "invalid")
    assert get_min_free_disk_bytes() == DEFAULT_MIN_FREE_DISK_BYTES
    monkeypatch.delenv("SHEEPDOG_MIN_FREE_DISK_MB", raising=False)

    # 2. get_free_disk_bytes resolves non-existent child paths and handles OSError
    missing_child = tmp_path / "future" / "nested" / "file.json"
    ample_usage = _FakeDiskUsage(total=100 * 1024**3, used=50 * 1024**3, free=5 * 1024**3)
    with patch("sheepdog.atomic_io.shutil.disk_usage", return_value=ample_usage) as mock_du:
        assert get_free_disk_bytes(missing_child) == 5 * 1024**3
        assert mock_du.call_args[0][0] == tmp_path.resolve()

    with patch("sheepdog.atomic_io.shutil.disk_usage", side_effect=OSError("statvfs failed")):
        assert get_free_disk_bytes(tmp_path) is None

    # 3. ensure_free_disk_space raises DiskSpaceError (subclass of OSError / ENOSPC)
    low_usage = _FakeDiskUsage(total=100 * 1024**3, used=99 * 1024**3, free=64 * 1024 * 1024)
    with patch("sheepdog.atomic_io.shutil.disk_usage", return_value=low_usage):
        with pytest.raises(DiskSpaceError) as exc_info:
            ensure_free_disk_space(tmp_path, context="unit test")
        assert isinstance(exc_info.value, OSError)
        assert exc_info.value.errno == errno.ENOSPC
        assert "unit test" in str(exc_info.value)

    # 4. MaskablePPOTrainer pre-segment disk guard raises DiskSpaceError before policy.model.learn
    cfg = _make_lab_config(
        tmp_path,
        trainer_type="maskable_ppo",
        policy_type="neural",
        checkpoint_episodes=(1,),
        evaluation_seeds=(11,),
        candidate_evaluation_seeds=(91,),
        rollout_steps=8,
        batch_size=4,
        total_timesteps=32,
        ppo_env_workers=1,
    )
    trainer = MaskablePPOTrainer(cfg, cfg.training.output_dir)
    mock_policy = MagicMock()
    mock_policy.config.to_dict.return_value = {"hidden_sizes": list(cfg.training.neural_hidden_sizes)}
    mock_policy.model.get_env.return_value = None
    with (
        patch.object(MaskablePPOTrainer, "POLICY_CLASS") as mock_cls,
        patch("sheepdog.atomic_io.shutil.disk_usage", return_value=low_usage),
    ):
        mock_cls.initialize.return_value = mock_policy
        with pytest.raises(DiskSpaceError):
            trainer.train()
        mock_policy.model.learn.assert_not_called()

    # 5. /api/health and /api/ready expose disk_free_bytes and disk_low accurately
    class TestConfig:
        def __new__(cls) -> LabConfig:
            return cfg

    warn_usage = _FakeDiskUsage(
        total=100 * 1024**3,
        used=99 * 1024**3,
        free=DEFAULT_WARN_FREE_DISK_BYTES - 1024,
    )
    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch.object(BaseHTTPRequestHandler, "log_message"),
    ):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = httpd.server_port
        srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        srv_thread.start()
        try:
            with patch("sheepdog.atomic_io.shutil.disk_usage", return_value=ample_usage):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
                    health_ok = json.loads(resp.read().decode("utf-8"))
                assert health_ok["disk_free_bytes"] == 5 * 1024**3
                assert health_ok["disk_low"] is False

            with patch("sheepdog.atomic_io.shutil.disk_usage", return_value=warn_usage):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ready", timeout=5) as resp:
                    ready_low = json.loads(resp.read().decode("utf-8"))
                assert ready_low["disk_free_bytes"] == DEFAULT_WARN_FREE_DISK_BYTES - 1024
                assert ready_low["disk_low"] is True
        finally:
            httpd.shutdown()
            httpd.server_close()
