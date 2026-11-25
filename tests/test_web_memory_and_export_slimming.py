"""Tests for web export slimming, HTTP ETag/gzip caching, and scandir eval caching."""

from __future__ import annotations

import contextlib
import gzip
import http.client
import io
import json
import os
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.evaluation.evaluator import EvaluationSummary
from sheepdog.evaluation.retention import EvaluationReplayRetentionManager
from sheepdog.server import (
    _GZIP_FILE_CACHE,
    TrainingManager,
    TrainingRequestHandler,
)
from sheepdog.training.factory import create_trainer
from sheepdog.training.trainer import slim_web_checkpoint_entry


def _make_lab_config(tmp_path: Path) -> LabConfig:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    backups = tmp_path / "backups"
    return LabConfig(
        training=TrainingConfig(
            trainer_type="hill_climb",
            policy_type="linear",
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
            backup_enabled=False,
            backup_dir=str(backups),
        )
    )


def _shutdown_manager(manager: TrainingManager) -> None:
    watchdog_stop = getattr(manager, "_watchdog_stop_event", None)
    if watchdog_stop is not None:
        with contextlib.suppress(Exception):
            watchdog_stop.set()
    runtime_tracker = getattr(manager, "runtime_tracker", None)
    if runtime_tracker is not None:
        with contextlib.suppress(Exception):
            runtime_tracker.end_session("test_cleanup")
        rt_stop = getattr(runtime_tracker, "_stop_event", None)
        if rt_stop is not None:
            with contextlib.suppress(Exception):
                rt_stop.set()
    for attr_name in ("telemetry_manager", "backup_manager", "_early_warning_monitor"):
        component = getattr(manager, attr_name, None)
        close_fn = getattr(component, "close", None)
        if callable(close_fn):
            with contextlib.suppress(Exception):
                close_fn()


def _bloated_checkpoint(episode: int, stage: int = 8, replay_path: str = "") -> dict[str, Any]:
    return {
        "checkpoint_episode": episode,
        "checkpoint": f"checkpoint-{episode:06d}.json",
        "curriculum_stage": stage,
        "success_rate": 0.75,
        "average_reward": 12.5,
        "average_completion_steps": 88.0,
        "average_sheep_penned": 5.5,
        "total_training_episodes": episode * 10,
        "global_timestep": episode * 1000,
        "recorded_at": "2026-11-22T00:00:00Z",
        "policy_mode": "trained_policy",
        "replay_mode": "trained_linear",
        "policy_weights": {"weights": [1.0] * 50},
        "training_scenario_coverage": {"scenarios": ["a", "b"]},
        "failure_directed_telemetry": {"buckets": {"corner": [1, 2, 3]}},
        "failure_directed_weights": {"corner": 2.5},
        "policy_config": {"learning_rate": 3e-4, "hidden": [256, 256]},
        "environment_config": {
            "dogs": 3,
            "sheep": 6,
            "width": 30,
            "height": 30,
            "curriculum_stage": stage,
            "max_steps": 240,
            "extra_heavy_field": "drop_me",
        },
        "reward_config": {
            "pen_bonus": 10.0,
            "step_penalty": -0.01,
            "instincts": {
                "curriculum_stage": stage,
                "enable_instinct_rewards": True,
                "approach_weight": 0.5,
            },
        },
        "records": [
            {
                "seed": 11,
                "success": True,
                "steps": 80,
                "replay_path": replay_path,
                "failed_trajectory_summary": {"stuck_steps": 4},
                "observation_diagnostics": {"nan_count": 0},
            }
        ],
    }


def test_atomic_write_json_compact_vs_indented(tmp_path: Path) -> None:
    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    payload = {"a": 1, "b": [2, 3]}

    atomic_write_json(compact_path, payload, indent=None)
    atomic_write_json(pretty_path, payload)

    compact_raw = compact_path.read_text(encoding="utf-8")
    pretty_raw = pretty_path.read_text(encoding="utf-8")

    assert compact_raw == '{"a":1,"b":[2,3]}'
    assert "\n" not in compact_raw
    assert "\n" in pretty_raw
    assert json.loads(compact_raw) == payload
    assert json.loads(pretty_raw) == payload


def test_slim_web_checkpoint_entry_strips_bloat_and_preserves_contract() -> None:
    payload = _bloated_checkpoint(42, stage=8, replay_path="/generated/replays/r1.json")
    slimmed = slim_web_checkpoint_entry(payload)

    # Dropped heavyweight keys
    for dropped in (
        "policy_weights",
        "training_scenario_coverage",
        "failure_directed_telemetry",
        "failure_directed_weights",
        "policy_config",
    ):
        assert dropped not in slimmed
        assert dropped in payload  # original payload not mutated

    # Trimmed environment_config
    assert slimmed["environment_config"] == {
        "dogs": 3,
        "sheep": 6,
        "width": 30,
        "height": 30,
        "curriculum_stage": 8,
    }
    assert "max_steps" in payload["environment_config"]

    # Trimmed reward_config preserves instincts subset for scenario_evaluator and UI
    assert slimmed["reward_config"] == {
        "instincts": {
            "curriculum_stage": 8,
            "enable_instinct_rewards": True,
        }
    }
    assert "pen_bonus" in payload["reward_config"]

    # Records strip failed_trajectory_summary & observation_diagnostics without mutating input
    assert len(slimmed["records"]) == 1
    assert "failed_trajectory_summary" not in slimmed["records"][0]
    assert "observation_diagnostics" not in slimmed["records"][0]
    assert slimmed["records"][0]["replay_path"] == "/generated/replays/r1.json"
    assert slimmed["records"][0]["seed"] == 11
    assert "failed_trajectory_summary" in payload["records"][0]

    # Preserved UI & scenario_evaluator fields
    for key in (
        "checkpoint_episode",
        "checkpoint",
        "curriculum_stage",
        "success_rate",
        "average_reward",
        "average_completion_steps",
        "average_sheep_penned",
        "total_training_episodes",
        "global_timestep",
        "recorded_at",
        "policy_mode",
        "replay_mode",
    ):
        assert slimmed[key] == payload[key]

    # reward_config without instincts dict is removed
    no_instincts = {"checkpoint_episode": 1, "reward_config": {"pen_bonus": 5.0}}
    assert "reward_config" not in slim_web_checkpoint_entry(no_instincts)


def test_export_web_assets_and_reconcile_slim_index_only(tmp_path: Path) -> None:
    config = _make_lab_config(tmp_path)
    trainer = create_trainer(config, config.training.output_dir)
    artifacts_dir = Path(config.training.output_dir)
    web_dir = Path(config.training.web_export_dir)
    replays_src_dir = artifacts_dir / "replays"
    replays_src_dir.mkdir(parents=True, exist_ok=True)
    cp_path = artifacts_dir / "checkpoints" / "checkpoint-000005.json"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text("{}", encoding="utf-8")

    # Archived journey with bloated entry
    archive_dir = artifacts_dir / "archive" / "journey-old"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_cp = _bloated_checkpoint(1, stage=4)
    (archive_dir / "checkpoint-index.json").write_text(
        json.dumps({"checkpoints": [archived_cp]}), encoding="utf-8"
    )

    replay_file = replays_src_dir / "ep_0005_seed_11.json"
    replay_file.write_text('{"frames": []}', encoding="utf-8")

    cp_payload = _bloated_checkpoint(5, stage=8, replay_path=str(replay_file))
    summary = EvaluationSummary(
        checkpoint_episode=5,
        policy_name="trained_policy",
        records=(
            {  # type: ignore[arg-type]
                "seed": 11,
                "success": True,
                "steps": 80,
                "replay_path": str(replay_file),
                "failed_trajectory_summary": {"stuck": 1},
                "observation_diagnostics": {"obs": 1},
            },
            {  # type: ignore[arg-type]
                "seed": 12,
                "success": False,
                "steps": 120,
                "replay_path": "",
                "failed_trajectory_summary": {"stuck": 5},
                "observation_diagnostics": {"obs": 2},
            },
        ),
        success_rate=1.0,
        timeout_rate=0.0,
        average_completion_steps=80.0,
        average_completion_seconds=4.0,
        average_sheep_penned=6.0,
        average_reward=25.0,
        curriculum_stage=8,
    )

    trainer._export_web_assets(web_dir, [cp_payload], summary, replay_file, cp_path)

    index_raw = (web_dir / "checkpoint-index.json").read_text(encoding="utf-8")
    assert "\n" not in index_raw
    assert ": " not in index_raw
    index_data = json.loads(index_raw)

    assert len(index_data["checkpoints"]) == 2
    for entry in index_data["checkpoints"]:
        assert "failure_directed_telemetry" not in entry
        assert "failure_directed_weights" not in entry
        assert "policy_config" not in entry
        assert "policy_weights" not in entry
        assert "training_scenario_coverage" not in entry
        assert set(entry["environment_config"]) <= {
            "dogs",
            "sheep",
            "width",
            "height",
            "curriculum_stage",
        }
        assert list(entry["reward_config"].keys()) == ["instincts"]
    assert "records" not in index_data["checkpoints"][0]  # archived entry stripped records
    for rec in index_data["latest"]["records"]:
        assert "failed_trajectory_summary" not in rec
        assert "observation_diagnostics" not in rec

    # latest-evaluation.json remains full/unstripped (summary_payload untouched by slimming)
    latest_eval = json.loads((web_dir / "latest-evaluation.json").read_text(encoding="utf-8"))
    assert latest_eval["checkpoint_episode"] == 5
    assert latest_eval["records"][1]["failed_trajectory_summary"] == {"stuck": 5}
    assert latest_eval["records"][1]["observation_diagnostics"] == {"obs": 2}

    # training-summary.json remains full/unstripped and bloated checkpoint-index.json triggers reconcile
    summary_file = artifacts_dir / "training-summary.json"
    summary_file.write_text(
        json.dumps({"checkpoints": [cp_payload]}, indent=2), encoding="utf-8"
    )
    (web_dir / "checkpoint-index.json").write_text(
        json.dumps(
            {
                "checkpoints": [cp_payload],
                "latest": cp_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reconciled = trainer.reconcile_web_exports(web_dir)
    assert reconciled == 5

    reconciled_raw = (web_dir / "checkpoint-index.json").read_text(encoding="utf-8")
    assert "\n" not in reconciled_raw
    reconciled_index = json.loads(reconciled_raw)
    assert "failure_directed_telemetry" not in reconciled_index["latest"]
    assert "failure_directed_telemetry" not in reconciled_index["checkpoints"][-1]
    assert "policy_config" not in reconciled_index["latest"]
    assert "max_steps" not in reconciled_index["latest"]["environment_config"]

    # latest-evaluation.json and training-summary.json stay full after reconcile
    reconciled_latest_eval = json.loads(
        (web_dir / "latest-evaluation.json").read_text(encoding="utf-8")
    )
    assert "failure_directed_telemetry" in reconciled_latest_eval
    assert "pen_bonus" in reconciled_latest_eval["reward_config"]
    full_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "failure_directed_telemetry" in full_summary["checkpoints"][0]
    assert "policy_weights" in full_summary["checkpoints"][0]

    # Second reconcile call is a no-op once slimmed
    assert trainer.reconcile_web_exports(web_dir) is None


def test_server_rewrite_web_exports_slims_and_compacts_index(tmp_path: Path) -> None:
    config = _make_lab_config(tmp_path)
    artifacts_dir = Path(config.training.output_dir)
    replay_file = artifacts_dir / "replays" / "ep_0007_seed_11.json"
    replay_file.parent.mkdir(parents=True, exist_ok=True)
    replay_file.write_text('{"frames": []}', encoding="utf-8")

    with patch("sheepdog.server.LabConfig", return_value=config):
        manager = TrainingManager()
        try:
            cp = _bloated_checkpoint(7, stage=8, replay_path=str(replay_file))
            manager._rewrite_web_exports_from_checkpoints(config, [cp])

            web_dir = Path(config.training.web_export_dir)
            index_raw = (web_dir / "checkpoint-index.json").read_text(encoding="utf-8")
            assert "\n" not in index_raw
            index_data = json.loads(index_raw)
            assert "failure_directed_telemetry" not in index_data["latest"]
            assert "failure_directed_telemetry" not in index_data["checkpoints"][0]
            assert "policy_config" not in index_data["latest"]

            latest_eval = json.loads(
                (web_dir / "latest-evaluation.json").read_text(encoding="utf-8")
            )
            assert "failure_directed_telemetry" in latest_eval
            assert "pen_bonus" in latest_eval["reward_config"]
        finally:
            _shutdown_manager(manager)


def test_file_response_etag_304_and_gzip(tmp_path: Path) -> None:
    config = _make_lab_config(tmp_path)
    generated = Path(config.training.web_export_dir)
    generated.mkdir(parents=True, exist_ok=True)

    payload_obj = {"checkpoints": [{"id": i, "text": "x" * 64} for i in range(40)]}
    raw_json_bytes = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    assert len(raw_json_bytes) >= 1024
    target_json = generated / "checkpoint-index.json"
    target_json.write_bytes(raw_json_bytes)

    small_json = generated / "small.json"
    small_json.write_bytes(b'{"ok":true}')

    _GZIP_FILE_CACHE.clear()

    class TestConfig:
        def __new__(cls) -> LabConfig:
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        server = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. Plain urllib client without Accept-Encoding: gzip receives plain JSON + ETag
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/generated/checkpoint-index.json"
            ) as resp:
                assert resp.status == 200
                assert resp.headers.get("Content-Encoding") is None
                assert resp.headers.get("Cache-Control") == "no-cache"
                assert "Accept-Encoding" in (resp.headers.get("Vary") or "")
                etag1 = resp.headers.get("ETag")
                assert etag1 and etag1.startswith('W/"')
                body = resp.read()
                assert body == raw_json_bytes

            # 2. Client with Accept-Encoding: gzip receives valid gzip-compressed JSON
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request(
                    "GET",
                    "/generated/checkpoint-index.json",
                    headers={"Accept-Encoding": "gzip, deflate"},
                )
                resp_gz = conn.getresponse()
                gz_bytes = resp_gz.read()
                assert resp_gz.status == 200
                assert resp_gz.getheader("Content-Encoding") == "gzip"
                assert resp_gz.getheader("ETag") == etag1
                assert int(resp_gz.getheader("Content-Length") or "0") == len(gz_bytes)
                assert gzip.decompress(gz_bytes) == raw_json_bytes

                # 2b. Second gzip request hits _GZIP_FILE_CACHE without re-reading or re-compressing
                with patch.object(
                    Path, "read_bytes", side_effect=AssertionError("cache hit must not read file")
                ):
                    conn.request(
                        "GET",
                        "/generated/checkpoint-index.json",
                        headers={"Accept-Encoding": "gzip"},
                    )
                    resp_gz_cached = conn.getresponse()
                    assert resp_gz_cached.status == 200
                    assert resp_gz_cached.getheader("Content-Encoding") == "gzip"
                    assert resp_gz_cached.read() == gz_bytes

                # 3. If-None-Match matching ETag returns 304 with empty body and skips disk read
                with patch.object(
                    Path, "read_bytes", side_effect=AssertionError("304 must not read file")
                ):
                    conn.request(
                        "GET",
                        "/generated/checkpoint-index.json",
                        headers={
                            "If-None-Match": f'W/"stale-1", {etag1}',
                            "Accept-Encoding": "gzip",
                        },
                    )
                    resp_304 = conn.getresponse()
                    body_304 = resp_304.read()
                    assert resp_304.status == 304
                    assert body_304 == b""
                    assert resp_304.getheader("ETag") == etag1
                    assert resp_304.getheader("Cache-Control") == "no-cache"

                # 4. Small JSON (< 1024 bytes) and non-JSON files are NOT gzipped
                conn.request(
                    "GET",
                    "/generated/small.json",
                    headers={"Accept-Encoding": "gzip"},
                )
                resp_small = conn.getresponse()
                small_body = resp_small.read()
                assert resp_small.status == 200
                assert resp_small.getheader("Content-Encoding") is None
                assert small_body == b'{"ok":true}'

                raw_bin = b"Z" * 2048
                (generated / "blob.bin").write_bytes(raw_bin)
                conn.request(
                    "GET",
                    "/generated/blob.bin",
                    headers={"Accept-Encoding": "gzip"},
                )
                resp_bin = conn.getresponse()
                bin_body = resp_bin.read()
                assert resp_bin.status == 200
                assert resp_bin.getheader("Content-Encoding") is None
                assert resp_bin.getheader("Content-Type") == "application/octet-stream"
                assert bin_body == raw_bin

                # 4b. _GZIP_FILE_CACHE stays bounded to at most 4 entries
                for idx in range(5):
                    extra_file = generated / f"extra_{idx}.json"
                    extra_file.write_bytes(
                        json.dumps({"idx": idx, "pad": "a" * 1100}).encode("utf-8")
                    )
                    conn.request(
                        "GET",
                        f"/generated/extra_{idx}.json",
                        headers={"Accept-Encoding": "gzip"},
                    )
                    extra_resp = conn.getresponse()
                    assert extra_resp.status == 200
                    extra_resp.read()
                assert len(_GZIP_FILE_CACHE) <= 4

                # 5. Modifying file invalidates ETag and returns 200 with updated JSON
                updated_bytes = json.dumps(
                    {"checkpoints": [{"id": 999, "text": "y" * 1200}]},
                    separators=(",", ":"),
                ).encode("utf-8")
                target_json.write_bytes(updated_bytes)
                stat_after = target_json.stat()
                os.utime(target_json, ns=(stat_after.st_atime_ns, stat_after.st_mtime_ns + 1_000_000))

                conn.request(
                    "GET",
                    "/generated/checkpoint-index.json",
                    headers={"If-None-Match": etag1, "Accept-Encoding": "gzip"},
                )
                resp_updated = conn.getresponse()
                updated_gz = resp_updated.read()
                assert resp_updated.status == 200
                etag2 = resp_updated.getheader("ETag")
                assert etag2 and etag2 != etag1
                assert gzip.decompress(updated_gz) == updated_bytes
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            _GZIP_FILE_CACHE.clear()


def test_get_recent_evaluations_scandir_and_mtime_cache(tmp_path: Path) -> None:
    config = _make_lab_config(tmp_path)
    artifacts = Path(config.training.output_dir)
    evals_dir = artifacts / "evaluations"
    evals_dir.mkdir(parents=True, exist_ok=True)

    def _write_eval(filename: str, eval_id: str, idx: int, stage: int, mtime: float) -> Path:
        fpath = evals_dir / filename
        fpath.write_text(
            json.dumps(
                {
                    "evaluation_id": eval_id,
                    "evaluation_index": idx,
                    "curriculum_stage": stage,
                    "records": [{"seed": 11, "success": True}],
                }
            ),
            encoding="utf-8",
        )
        os.utime(fpath, (mtime, mtime))
        return fpath

    base_t = time.time() - 100.0
    _write_eval("eval_001.json", "eval_first", 1, 8, base_t + 10.0)
    _write_eval("eval_002.json", "eval_dup", 10, 7, base_t + 20.0)
    _write_eval("eval_003.json", "eval_dup", 10, 7, base_t + 30.0)  # newer duplicate
    _write_eval("eval_004.json", "eval_milestone", 25, 8, base_t + 40.0)
    # Non-matching files that must be ignored
    (evals_dir / "eval_retention_index.json").write_text("{}", encoding="utf-8")
    (evals_dir / "other.json").write_text("{}", encoding="utf-8")

    ret_mgr = EvaluationReplayRetentionManager(evals_dir)
    assert ret_mgr.pin_evaluation("eval_milestone", pinned=True)

    with patch("sheepdog.server.LabConfig", return_value=config):
        manager = TrainingManager()
        try:
            with patch("sheepdog.server.os.scandir", wraps=os.scandir) as scandir_spy:
                all_evals = manager.get_recent_evaluations(limit=10)
                assert scandir_spy.call_count == 1
                assert [e["evaluation_id"] for e in all_evals] == [
                    "eval_milestone",
                    "eval_dup",
                    "eval_first",
                ]
                assert all_evals[0]["pinned"] is True
                assert all_evals[0]["is_milestone"] is True
                assert all_evals[0]["is_first"] is False
                assert all_evals[2]["pinned"] is False
                assert all_evals[2]["is_first"] is True
                assert all_evals[2]["is_milestone"] is False

                # Cache hit when directory mtime_ns is unchanged (stage filter uses cached file list)
                stage8_evals = manager.get_recent_evaluations(limit=10, stage=8)
                assert scandir_spy.call_count == 1
                assert [e["evaluation_id"] for e in stage8_evals] == [
                    "eval_milestone",
                    "eval_first",
                ]
                assert len(manager._recent_eval_files_cache) == 1

                # Adding a new evaluation file updates directory mtime_ns and triggers rescan
                prev_dir_mtime_ns = evals_dir.stat().st_mtime_ns
                _write_eval("eval_005.json", "eval_newest", 26, 8, base_t + 50.0)
                if evals_dir.stat().st_mtime_ns == prev_dir_mtime_ns:
                    os.utime(evals_dir, ns=(prev_dir_mtime_ns, prev_dir_mtime_ns + 1_000_000))

                refreshed = manager.get_recent_evaluations(limit=10, stage=8)
                assert scandir_spy.call_count == 2
                assert [e["evaluation_id"] for e in refreshed] == [
                    "eval_newest",
                    "eval_milestone",
                    "eval_first",
                ]
                assert len(manager._recent_eval_files_cache) == 1
        finally:
            _shutdown_manager(manager)


def test_get_recent_evaluations_slims_heavy_diagnostic_blobs_without_touching_disk(
    tmp_path: Path,
) -> None:
    config = _make_lab_config(tmp_path)
    artifacts = Path(config.training.output_dir)
    evals_dir = artifacts / "evaluations"
    evals_dir.mkdir(parents=True, exist_ok=True)

    original_eval = {
        "evaluation_id": "eval_heavy",
        "evaluation_index": 25,
        "checkpoint_episode": 250,
        "curriculum_stage": 8,
        "failure_directed_telemetry": {"buckets": {"corner": [1, 2, 3]}},
        "failure_directed_weights": {"corner": 2.5},
        "policy_weights": {"weights": [0.1] * 100},
        "training_scenario_coverage": {"scenarios": ["s1", "s2"]},
        "records": [
            {
                "seed": 11,
                "replay_path": "/generated/replays/r11.json",
                "reward_breakdown": {"pen_bonus": 10.0},
                "spatial_metrics": {"corner_pct": 0.1},
                "dog_role_occupancy": {"0": {"collector": 5}},
                "success": True,
                "timeout": False,
                "stopped": False,
                "steps": 45,
                "penned_count": 6,
                "failed_trajectory_summary": [{"step": 1, "note": "drop_me"}],
                "observation_diagnostics": {"nan_count": 0, "heavy": [1, 2, 3]},
            }
        ],
    }
    eval_path = evals_dir / "eval_heavy.json"
    eval_path.write_text(json.dumps(original_eval), encoding="utf-8")

    ret_mgr = EvaluationReplayRetentionManager(evals_dir)
    assert ret_mgr.pin_evaluation("eval_heavy", pinned=True)

    with patch("sheepdog.server.LabConfig", return_value=config):
        manager = TrainingManager()
        try:
            evals = manager.get_recent_evaluations(limit=5)
            assert len(evals) == 1
            item = evals[0]

            for dropped in (
                "failure_directed_telemetry",
                "failure_directed_weights",
                "policy_weights",
                "training_scenario_coverage",
            ):
                assert dropped not in item

            assert item["evaluation_id"] == "eval_heavy"
            assert item["evaluation_index"] == 25
            assert item["checkpoint_episode"] == 250
            assert item["curriculum_stage"] == 8
            assert item["pinned"] is True
            assert item["is_milestone"] is True
            assert item["is_first"] is False

            rec = item["records"][0]
            assert "failed_trajectory_summary" not in rec
            assert "observation_diagnostics" not in rec
            assert rec["seed"] == 11
            assert rec["replay_path"] == "/generated/replays/r11.json"
            assert rec["reward_breakdown"] == {"pen_bonus": 10.0}
            assert rec["spatial_metrics"] == {"corner_pct": 0.1}
            assert rec["dog_role_occupancy"] == {"0": {"collector": 5}}
            assert rec["success"] is True
            assert rec["timeout"] is False
            assert rec["stopped"] is False
            assert rec["steps"] == 45
            assert rec["penned_count"] == 6

            # Original file on disk must remain completely intact
            on_disk = json.loads(eval_path.read_text(encoding="utf-8"))
            assert on_disk == original_eval
        finally:
            _shutdown_manager(manager)


def test_json_response_compact_and_optional_gzip(tmp_path: Path) -> None:
    def _invoke_handler(
        payload: Any,
        *,
        compact: bool = False,
        req_headers: dict[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
        include_headers_attr: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        handler = object.__new__(TrainingRequestHandler)
        if include_headers_attr:
            handler.headers = req_headers  # type: ignore[assignment]
        sent_status: list[int] = []
        sent_headers: dict[str, str] = {}
        buf = io.BytesIO()
        handler.wfile = buf  # type: ignore[assignment]
        handler.send_response = lambda code, message=None: sent_status.append(int(code))  # type: ignore[method-assign]
        handler.send_header = lambda k, v: sent_headers.__setitem__(k, v)  # type: ignore[method-assign]
        handler.end_headers = lambda: None  # type: ignore[method-assign]
        handler._json_response(payload, headers=extra_headers, compact=compact)
        return sent_status[0], sent_headers, buf.getvalue()

    large_payload = {"items": [{"id": i, "text": "x" * 64} for i in range(30)]}

    # 1. Handler without headers attr -> safe, compact omits whitespace, no gzip
    status, hdrs, raw_compact = _invoke_handler(
        large_payload, compact=True, include_headers_attr=False
    )
    assert status == 200
    assert "Content-Encoding" not in hdrs
    compact_text = raw_compact.decode("utf-8")
    assert "\n" not in compact_text
    assert ": " not in compact_text
    assert json.loads(compact_text) == large_payload

    # 2. Default compact=False keeps indent=2 formatting
    _, hdrs_pretty, raw_pretty = _invoke_handler(
        large_payload, compact=False, req_headers={}
    )
    assert "Content-Encoding" not in hdrs_pretty
    assert "\n" in raw_pretty.decode("utf-8")

    # 3. Accept-Encoding: gzip on >= 1024-byte compact JSON compresses and sets headers
    _, hdrs_gz, body_gz = _invoke_handler(
        large_payload, compact=True, req_headers={"Accept-Encoding": "gzip, deflate"}
    )
    assert hdrs_gz.get("Content-Encoding") == "gzip"
    assert hdrs_gz.get("Vary") == "Accept-Encoding"
    assert int(hdrs_gz["Content-Length"]) == len(body_gz)
    assert gzip.decompress(body_gz) == raw_compact

    # 4. Small JSON (< 1024 bytes) and pre-set Content-Encoding skip gzip
    _, hdrs_small, body_small = _invoke_handler(
        {"ok": True}, compact=True, req_headers={"Accept-Encoding": "gzip"}
    )
    assert "Content-Encoding" not in hdrs_small
    assert body_small == b'{"ok":true}'

    _, hdrs_preset, body_preset = _invoke_handler(
        large_payload,
        compact=True,
        req_headers={"Accept-Encoding": "gzip"},
        extra_headers={"content-encoding": "identity"},
    )
    assert hdrs_preset["content-encoding"] == "identity"
    assert body_preset == raw_compact

    # 5. End-to-end HTTP check on /api/insights/evaluations
    config = _make_lab_config(tmp_path)
    evals_dir = Path(config.training.output_dir) / "evaluations"
    evals_dir.mkdir(parents=True, exist_ok=True)
    (evals_dir / "eval_001.json").write_text(
        json.dumps(
            {
                "evaluation_id": "eval_e2e",
                "evaluation_index": 1,
                "curriculum_stage": 8,
                "failure_directed_telemetry": {"drop": True},
                "records": [
                    {
                        "seed": i,
                        "replay_path": f"/generated/replays/r{i}.json",
                        "padding": "p" * 80,
                        "failed_trajectory_summary": [{"step": 0}],
                        "observation_diagnostics": {"obs": [1, 2]},
                    }
                    for i in range(25)
                ],
            }
        ),
        encoding="utf-8",
    )

    class TestConfig:
        def __new__(cls) -> LabConfig:
            return config

    existing_mgr = getattr(TrainingRequestHandler, "_manager_instance", None)
    if existing_mgr is not None:
        _shutdown_manager(existing_mgr)
        TrainingRequestHandler._manager_instance = None

    with patch("sheepdog.server.LabConfig", TestConfig):
        server = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/insights/evaluations?limit=5"
            ) as resp:
                assert resp.status == 200
                assert resp.headers.get("Content-Encoding") is None
                plain_bytes = resp.read()
                plain_text = plain_bytes.decode("utf-8")
                assert "\n" not in plain_text
                parsed = json.loads(plain_text)
                assert len(parsed["evaluations"]) == 1
                assert "failure_directed_telemetry" not in parsed["evaluations"][0]
                assert "failed_trajectory_summary" not in parsed["evaluations"][0]["records"][0]

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request(
                    "GET",
                    "/api/insights/evaluations?limit=5",
                    headers={"Accept-Encoding": "gzip"},
                )
                resp_gz = conn.getresponse()
                gz_payload = resp_gz.read()
                assert resp_gz.status == 200
                assert resp_gz.getheader("Content-Encoding") == "gzip"
                assert "Accept-Encoding" in (resp_gz.getheader("Vary") or "")
                assert gzip.decompress(gz_payload) == plain_bytes
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            created_mgr = getattr(TrainingRequestHandler, "_manager_instance", None)
            if created_mgr is not None:
                _shutdown_manager(created_mgr)
                TrainingRequestHandler._manager_instance = None
