"""Tests for durable SQLite model-lineage, stage-spec parsing, backfill, and runtime totals."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts.backfill_model_history import main as backfill_cli_main

from sheepdog.tracking import model_history
from sheepdog.tracking.model_history import (
    MODEL_HISTORY_DB_FILENAME,
    SCHEMA_VERSION,
    ModelHistoryStore,
    classify_pass_type,
    compute_file_sha256,
    format_stage_spec,
    parse_stage_spec,
)
from sheepdog.training.runtime import TrainingRuntimeTracker


class FakeClock:
    """Controllable monotonic and UTC clock pair."""

    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.utc_value = datetime(2026, 11, 25, 10, 0, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_value

    def utc_now(self) -> datetime:
        return self.utc_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.utc_value += timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def _stub_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_history, "git_info", lambda: {"revision": "deadbeef1234", "dirty": False}
    )


def test_parse_and_format_stage_spec_and_classify_pass_type() -> None:
    assert parse_stage_spec("1-38") == list(range(1, 39))
    assert classify_pass_type(parse_stage_spec("1-38")) == "full_curriculum"
    assert format_stage_spec(list(range(1, 39))) == "1-38"

    targeted = "10-12, 15-20, 25, 33-38"
    parsed = parse_stage_spec(targeted)
    assert parsed == [10, 11, 12, 15, 16, 17, 18, 19, 20, 25, 33, 34, 35, 36, 37, 38]
    assert format_stage_spec(parsed) == "10-12, 15-20, 25, 33-38"
    assert classify_pass_type(parsed) == "targeted_stages"

    # En-dash, semicolon, spaces around dash, trailing comma
    assert parse_stage_spec("10\u201312; 15 - 17, 25,") == [10, 11, 12, 15, 16, 17, 25]

    # Intentional repeats preserve exact order
    repeats = parse_stage_spec("25, 36, 25")
    assert repeats == [25, 36, 25]
    assert format_stage_spec(repeats) == "25, 36, 25"
    assert classify_pass_type(repeats) == "targeted_stages"

    # Single stage
    assert parse_stage_spec("25") == [25]
    assert format_stage_spec([25]) == "25"
    assert classify_pass_type([25]) == "single_stage"
    assert format_stage_spec([]) == ""

    # Sequence of ints input
    assert parse_stage_spec([5, 6, 6, 7]) == [5, 6, 6, 7]
    assert format_stage_spec([5, 6, 6, 7]) == "5-6, 6-7"

    # Invalid inputs
    for bad in ("", "   ", ",", ";", "1,,2", "12-10", "0", "39", "abc", "1-abc", "1 - -2"):
        with pytest.raises(ValueError):
            parse_stage_spec(bad)

    for bad_seq in ([], [True], [1, False], [0], [39]):
        with pytest.raises(ValueError):
            parse_stage_spec(bad_seq)  # type: ignore[arg-type]


def test_compute_file_sha256(tmp_path: Path) -> None:
    file_path = tmp_path / "model.zip"
    payload = b"sheepdog-checkpoint-bytes" * 5000
    file_path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert compute_file_sha256(file_path) == expected
    assert compute_file_sha256(None) is None
    assert compute_file_sha256(tmp_path / "missing.zip") is None
    assert compute_file_sha256(tmp_path) is None


def test_four_pass_lineage_and_branch_pass(tmp_path: Path) -> None:
    db_path = tmp_path / MODEL_HISTORY_DB_FILENAME
    store = ModelHistoryStore(db_path)

    ckpt1 = tmp_path / "p1.zip"
    ckpt1.write_bytes(b"pass1-weights")
    sha1 = compute_file_sha256(ckpt1)

    ckpt2 = tmp_path / "p2.zip"
    ckpt2.write_bytes(b"pass2-weights")
    sha2 = compute_file_sha256(ckpt2)

    ckpt3 = tmp_path / "p3.zip"
    ckpt3.write_bytes(b"pass3-weights")
    sha3 = compute_file_sha256(ckpt3)

    ckpt4 = tmp_path / "p4.zip"
    ckpt4.write_bytes(b"pass4-weights")
    sha4 = compute_file_sha256(ckpt4)

    # Pass 1: initial full curriculum
    p1 = store.start_pass(
        pass_id="pass_1",
        label="Pass 1 Full",
        run_id="run_1",
        stage_spec="1-38",
        started_at="2026-11-25T01:00:00+00:00",
        observation_mode="guided",
    )
    sr1_25 = store.start_stage_run(
        p1,
        run_id="run_1",
        stage=25,
        sequence_order=0,
        started_at="2026-11-25T01:10:00+00:00",
        start_policy_timesteps=1000,
    )
    store.finish_stage_run(
        sr1_25,
        status="promoted",
        ended_at="2026-11-25T01:12:00+00:00",
        runtime_totals={
            "training_seconds": 100.0,
            "evaluation_seconds": 10.0,
            "checkpoint_save_seconds": 5.0,
            "replay_serialization_seconds": 5.0,
        },
        current_policy_timesteps=2000,
        success_rate=1.0,
        avg_steps=320.0,
        checkpoint_id="chk_p1_25",
    )
    sr1_36 = store.start_stage_run(
        p1,
        run_id="run_1",
        stage=36,
        sequence_order=1,
        started_at="2026-11-25T01:12:00+00:00",
        start_policy_timesteps=2000,
    )
    store.finish_stage_run(
        sr1_36,
        status="promoted",
        ended_at="2026-11-25T01:14:00+00:00",
        runtime_totals={
            "training_seconds": 80.0,
            "evaluation_seconds": 10.0,
            "checkpoint_save_seconds": 5.0,
            "replay_serialization_seconds": 5.0,
        },
        current_policy_timesteps=2800,
        success_rate=1.0,
        checkpoint_id="chk_p1_36",
    )
    store.finish_pass(
        p1,
        status="completed",
        ended_at="2026-11-25T01:20:00+00:00",
        resulting_checkpoint_id="chk_p1_final",
        resulting_checkpoint_path=ckpt1,
    )

    # Pass 2: targeted stages auto-resolving parent from sha1
    p2 = store.start_pass(
        pass_id="pass_2",
        label="Pass 2 Targeted",
        run_id="run_2",
        parent_checkpoint_sha256=sha1,
        stage_spec="10-12, 15-20, 25, 33-38",
        started_at="2026-11-25T02:00:00+00:00",
        observation_mode="guided",
    )
    p2_data = store.get_pass(p2)
    assert p2_data is not None
    assert p2_data["parent_pass_id"] == p1
    assert p2_data["pass_type"] == "targeted_stages"
    assert p2_data["assistance_mode"] == "guided"

    sr2_25 = store.start_stage_run(
        p2,
        run_id="run_2",
        stage=25,
        sequence_order=0,
        started_at="2026-11-25T02:01:00+00:00",
        start_policy_timesteps=2800,
    )
    store.finish_stage_run(
        sr2_25,
        status="promoted",
        ended_at="2026-11-25T02:02:00+00:00",
        runtime_totals={"training_seconds": 50.0, "evaluation_seconds": 10.0},
        current_policy_timesteps=3400,
        success_rate=1.0,
    )
    store.finish_pass(
        p2,
        status="completed",
        ended_at="2026-11-25T02:10:00+00:00",
        resulting_checkpoint_id="chk_p2_final",
        resulting_checkpoint_sha256=sha2,
    )

    # Pass 3: targeted "25, 36" off Pass 2
    p3 = store.start_pass(
        pass_id="pass_3",
        label="Pass 3 Polish",
        run_id="run_3",
        parent_checkpoint_sha256=sha2,
        stage_spec="25, 36",
        started_at="2026-11-25T03:00:00+00:00",
        observation_mode="emergent",
    )
    sr3_25 = store.start_stage_run(
        p3,
        run_id="run_3",
        stage=25,
        sequence_order=0,
        started_at="2026-11-25T03:01:00+00:00",
        start_policy_timesteps=3400,
    )
    store.finish_stage_run(
        sr3_25,
        status="promoted",
        ended_at="2026-11-25T03:02:00+00:00",
        runtime_totals={"training_seconds": 40.0, "evaluation_seconds": 5.0},
        current_policy_timesteps=3900,
        success_rate=1.0,
    )
    store.finish_pass(
        p3,
        status="completed",
        ended_at="2026-11-25T03:05:00+00:00",
        resulting_checkpoint_id="chk_p3_final",
        resulting_checkpoint_sha256=sha3,
    )

    # Pass 4: single stage "25" off Pass 3
    p4 = store.start_pass(
        pass_id="pass_4",
        label="Pass 4 Focus 25",
        run_id="run_4",
        parent_checkpoint_sha256=sha3,
        stage_spec="25",
        started_at="2026-11-25T04:00:00+00:00",
        observation_mode="guided",
    )
    assert store.get_pass(p4)["pass_type"] == "single_stage"  # type: ignore[index]
    sr4_25 = store.start_stage_run(
        p4,
        run_id="run_4",
        stage=25,
        sequence_order=0,
        started_at="2026-11-25T04:01:00+00:00",
        start_policy_timesteps=3900,
    )
    store.finish_stage_run(
        sr4_25,
        status="mastered",
        ended_at="2026-11-25T04:01:30+00:00",
        runtime_totals={"training_seconds": 25.0, "evaluation_seconds": 5.0},
        current_policy_timesteps=4200,
        success_rate=1.0,
    )
    store.finish_pass(
        p4,
        status="completed",
        ended_at="2026-11-25T04:02:00+00:00",
        resulting_checkpoint_id="chk_p4_final",
        resulting_checkpoint_sha256=sha4,
    )

    # Branch pass off Pass 1 (must NOT appear in Pass 4's lineage)
    p_branch = store.start_pass(
        pass_id="pass_branch",
        label="Branch off Pass 1",
        run_id="run_branch",
        parent_checkpoint_sha256=sha1,
        stage_spec="33-38",
        started_at="2026-11-25T05:00:00+00:00",
    )
    assert store.get_pass(p_branch)["parent_pass_id"] == p1  # type: ignore[index]

    lineage = store.get_lineage(p4)
    assert lineage is not None
    assert lineage["target_pass_id"] == p4
    assert [p["pass_id"] for p in lineage["lineage"]] == [p1, p2, p3, p4]
    assert lineage["cumulative"]["pass_count"] == 4
    assert lineage["cumulative"]["stage_runs_count"] == 5
    assert lineage["cumulative"]["completed_stage_runs_count"] == 5
    # Pass1(120+100) + Pass2(60) + Pass3(45) + Pass4(30) = 355
    assert lineage["cumulative"]["total_active_seconds"] == pytest.approx(355.0)
    # 1000 + 800 + 600 + 500 + 300 = 3200
    assert lineage["cumulative"]["total_policy_timesteps"] == 3200

    comp_by_stage = {entry["stage"]: entry for entry in lineage["stage_comparison"]}
    assert sorted(comp_by_stage) == [25, 36]
    s25 = comp_by_stage[25]
    assert len(s25["runs"]) == 4
    assert [r["pass_index"] for r in s25["runs"]] == [0, 1, 2, 3]
    # Run 0: active=120, train=100, pts=1000; Run 2: active=45, train=40, pts=500; Run 3: active=30, train=25, pts=300
    assert s25["delta_vs_first"]["active_seconds_pct"] == pytest.approx(-75.0)
    assert s25["delta_vs_first"]["training_seconds_pct"] == pytest.approx(-75.0)
    assert s25["delta_vs_first"]["policy_timesteps_pct"] == pytest.approx(-70.0)
    assert s25["delta_vs_previous"]["training_seconds_pct"] == pytest.approx(-37.5)
    assert s25["delta_vs_previous"]["policy_timesteps_pct"] == pytest.approx(-40.0)

    # Single-run stage 36 has None deltas
    s36 = comp_by_stage[36]
    assert len(s36["runs"]) == 1
    assert s36["delta_vs_first"]["active_seconds_pct"] is None
    assert s36["delta_vs_previous"]["training_seconds_pct"] is None

    # Branch lineage only includes Pass 1 -> Branch
    branch_lineage = store.get_lineage(p_branch)
    assert branch_lineage is not None
    assert [p["pass_id"] for p in branch_lineage["lineage"]] == [p1, p_branch]

    # Missing lineage returns None
    assert store.get_lineage("nonexistent_pass") is None

    # API payload and stage filter
    payload = store.build_api_payload(pass_id=p4, stage=25)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert len(payload["passes"]) == 5
    assert payload["passes"][0]["pass_id"] == p_branch  # newest first
    assert payload["selected_lineage"]["target_pass_id"] == p4
    assert [entry["stage"] for entry in payload["stage_matrix"]] == [25]
    assert payload["stage_matrix"][0]["run_count"] == 4


def test_update_stage_run_baseline_subtraction_and_initial_sr_pinning(tmp_path: Path) -> None:
    store = ModelHistoryStore(tmp_path / MODEL_HISTORY_DB_FILENAME)
    pass_id = store.start_pass(pass_id="p_revisit", stage_spec="25, 25", observation_mode="guided")

    # First visit completed stage 25 with 100s training
    sr1 = store.start_stage_run(
        pass_id,
        run_id="r1",
        stage=25,
        sequence_order=0,
        start_policy_timesteps=0,
    )
    store.finish_stage_run(
        sr1,
        status="promoted",
        runtime_totals={"training_seconds": 100.0, "evaluation_seconds": 10.0},
        current_policy_timesteps=1000,
        success_rate=1.0,
    )

    # Revisit same stage 25 within same run_id: baseline is previous cumulative totals
    sr2 = store.start_stage_run(
        pass_id,
        run_id="r1",
        stage=25,
        sequence_order=1,
        start_policy_timesteps=1000,
        baseline_runtime={
            "training_seconds": 100.0,
            "evaluation_seconds": 10.0,
            "checkpoint_save_seconds": 5.0,
            "replay_serialization_seconds": 2.0,
        },
    )

    # First observation sets initial_success_rate and subtracts baseline
    store.update_stage_run(
        sr2,
        runtime_totals={
            "training_seconds": 150.0,
            "evaluation_seconds": 15.0,
            "checkpoint_save_seconds": 6.0,
            "replay_serialization_seconds": 1.0,  # below baseline -> clamped to 0.0
        },
        current_policy_timesteps=1500,
        episodes=12,
        success_rate=0.4,
        avg_steps=500.0,
        avg_reward=120.5,
        eval_seed_count=10,
    )
    rec = store.get_stage_run(sr2)
    assert rec is not None
    assert "baseline_runtime_json" not in rec
    assert rec["training_seconds"] == pytest.approx(50.0)
    assert rec["evaluation_seconds"] == pytest.approx(5.0)
    assert rec["checkpoint_save_seconds"] == pytest.approx(1.0)
    assert rec["replay_serialization_seconds"] == pytest.approx(0.0)
    assert rec["active_seconds"] == pytest.approx(56.0)
    assert rec["policy_timesteps"] == 500
    assert rec["steps_per_second"] == pytest.approx(10.0)
    assert rec["initial_success_rate"] == pytest.approx(0.4)
    assert rec["final_success_rate"] == pytest.approx(0.4)
    assert rec["best_success_rate"] == pytest.approx(0.4)

    # Second observation updates best/final without overwriting initial_success_rate
    store.update_stage_run(sr2, success_rate=0.9, avg_steps=380.0)
    store.update_stage_run(sr2, success_rate=0.7, avg_steps=410.0)
    rec2 = store.get_stage_run(sr2)
    assert rec2 is not None
    assert rec2["initial_success_rate"] == pytest.approx(0.4)
    assert rec2["best_success_rate"] == pytest.approx(0.9)
    assert rec2["final_success_rate"] == pytest.approx(0.7)
    assert rec2["final_avg_steps"] == pytest.approx(410.0)

    # Parent rollups reflect 1 completed stage (sr1 promoted, sr2 still running)
    pass_rec = store.get_pass(pass_id)
    assert pass_rec is not None
    assert pass_rec["completed_stages_count"] == 1
    assert pass_rec["total_training_seconds"] == pytest.approx(150.0)
    assert pass_rec["total_policy_timesteps"] == 1500


def test_read_only_missing_db_creates_zero_files_and_existing_ro_works(tmp_path: Path) -> None:
    missing_dir = tmp_path / "never_created" / "nested"
    missing_db = missing_dir / MODEL_HISTORY_DB_FILENAME

    ro_store = ModelHistoryStore(missing_db, read_only=True)
    assert ro_store.list_passes() == []
    assert ro_store.get_pass("p1") is None
    assert ro_store.get_stage_run("sr1") is None
    assert ro_store.get_lineage("p1") is None
    assert ro_store.find_parent_pass_id("abc") is None
    payload = ro_store.build_api_payload()
    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "passes": [],
        "active_pass_id": None,
        "selected_lineage": None,
        "stage_matrix": [],
    }
    assert not missing_dir.exists()
    assert not missing_db.exists()
    assert list(tmp_path.iterdir()) == []

    # Verify read_only=True works with percent-encoded URI on path containing spaces
    spaced_dir = tmp_path / "dir with spaces"
    spaced_db = spaced_dir / MODEL_HISTORY_DB_FILENAME
    rw_store = ModelHistoryStore(spaced_db)
    pid = rw_store.start_pass(pass_id="p_spaced", stage_spec="25", observation_mode="guided")
    rw_store.finish_pass(pid, status="completed")

    ro_existing = ModelHistoryStore(spaced_db, read_only=True)
    assert len(ro_existing.list_passes()) == 1
    assert ro_existing.get_pass("p_spaced")["stage_spec"] == "25"  # type: ignore[index]
    with pytest.raises(PermissionError):
        ro_existing.start_pass(stage_spec="1")

    # Verify SQLite user_version pragma
    with sqlite3.connect(str(spaced_db)) as conn:
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_backfill_from_artifacts_stage38_and_idempotency(tmp_path: Path) -> None:
    art_dir = tmp_path / "artifacts"
    ckpt_dir = art_dir / "checkpoints" / "stage38"
    ckpt_dir.mkdir(parents=True)
    model_file = ckpt_dir / "stage38-best-model.zip"
    model_bytes = b"stage38-best-checkpoint-data"
    model_file.write_bytes(model_bytes)
    expected_sha = hashlib.sha256(model_bytes).hexdigest()

    run_id = "run_20261124_093542_33c5"

    # Write stage-timing.jsonl for stages 1..37
    records = []
    start_t = datetime(2026, 11, 25, 0, 0, 0, tzinfo=UTC)
    cum_ts = 0
    for stage in range(1, 38):
        s_at = (start_t + timedelta(minutes=(stage - 1) * 5)).isoformat()
        e_at = (start_t + timedelta(minutes=stage * 5)).isoformat()
        cum_ts += 100_000
        records.append(
            {
                "schema_version": 1,
                "promotion_event_id": f"evt_{stage}",
                "run_id": run_id,
                "stage": stage,
                "to_stage": stage + 1,
                "started_at": s_at,
                "ended_at": e_at,
                "wall_seconds": 300.0,
                "policy_timesteps": 100_000,
                "trigger_timesteps": cum_ts,
                "trigger_checkpoint_id": f"chk_{run_id}_pv_{stage}_ts_{cum_ts}",
                "episodes": 50 + stage,
                "success_rate": 1.0,
                "evaluation_seed_count": 10,
                "source": "recorded" if stage >= 24 else "derived",
                "assistance_mode": "guided" if stage >= 24 else None,
                "config_hash": "cfg123" if stage >= 24 else None,
            }
        )
    (art_dir / "stage-timing.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    stage38_start = (start_t + timedelta(minutes=37 * 5)).isoformat()
    stage38_end = (start_t + timedelta(minutes=37 * 5, seconds=400)).isoformat()

    (art_dir / "stage-timing-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": 38,
                "started_at": stage38_start,
                "start_timesteps": cum_ts,
            }
        ),
        encoding="utf-8",
    )

    stage_seconds = {
        str(st): {
            "training_seconds": 200.0 + st,
            "evaluation_seconds": 20.0,
            "checkpoint_save_seconds": 10.0,
            "replay_serialization_seconds": 5.0,
        }
        for st in range(1, 39)
    }
    (art_dir / "training-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": [
                    {
                        "session_id": "sess_1",
                        "run_id": run_id,
                        "curriculum_stage": 38,
                        "started_at": start_t.isoformat(),
                        "last_heartbeat_at": stage38_end,
                        "ended_at": stage38_end,
                        "end_reason": "completed",
                        "status": "ended",
                        "stage_seconds": stage_seconds,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (art_dir / "training-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "total_timesteps": cum_ts + 500_000,
                "best_model_path": "artifacts\\checkpoints\\stage38\\stage38-best-model.zip",
                "best_model_curriculum_stage": 38,
                "best_success_rate": 1.0,
                "best_average_reward": 195.25,
                "best_completion_steps": 284.5,
                "policy_config": {"observation_mode": "guided"},
            }
        ),
        encoding="utf-8",
    )

    (art_dir / "run-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "active_curriculum_stage": 38,
                "active_checkpoint_id": f"chk_{run_id}_pv_99_ts_{cum_ts + 500_000}",
                "active_model_path": "artifacts\\checkpoints\\stage38\\stage38-best-model.zip",
            }
        ),
        encoding="utf-8",
    )

    (art_dir / "effective-training-config.json").write_text(
        json.dumps({"training": {"observation_mode": "guided", "ppo_env_workers": 16}}),
        encoding="utf-8",
    )
    (art_dir / "stage-history.json").write_text(json.dumps({"38": 7807}), encoding="utf-8")

    store = ModelHistoryStore(art_dir / MODEL_HISTORY_DB_FILENAME)
    first = store.backfill_from_artifacts(art_dir)
    assert first["imported_passes"] == 1
    assert first["skipped_passes"] == 0
    assert first["imported_stage_runs"] == 38
    assert first["pass_ids"] == [f"pass_backfill_{run_id}"]

    # Second run is strictly idempotent
    second = store.backfill_from_artifacts(art_dir)
    assert second["imported_passes"] == 0
    assert second["skipped_passes"] == 1
    assert second["imported_stage_runs"] == 0

    pass_rec = store.get_pass(f"pass_backfill_{run_id}")
    assert pass_rec is not None
    assert pass_rec["pass_type"] == "full_curriculum"
    assert pass_rec["stage_spec"] == "1-38"
    assert pass_rec["label"] == f"Backfilled 1-38 ({run_id})"
    assert pass_rec["source"] == "backfilled"
    assert pass_rec["status"] == "completed"
    assert pass_rec["completed_stages_count"] == 38
    assert pass_rec["assistance_mode"] == "guided"
    assert pass_rec["observation_mode"] == "guided"
    assert pass_rec["resulting_checkpoint_sha256"] == expected_sha

    stage38 = pass_rec["stage_runs"][-1]
    assert stage38["stage"] == 38
    assert stage38["status"] == "mastered"
    assert stage38["policy_timesteps"] == 500_000
    assert stage38["wall_seconds"] == pytest.approx(400.0)
    assert stage38["training_seconds"] == pytest.approx(238.0)
    assert stage38["active_seconds"] == pytest.approx(273.0)
    assert stage38["episodes"] == 7807
    assert stage38["end_checkpoint_sha256"] == expected_sha
    assert stage38["best_success_rate"] == pytest.approx(1.0)
    assert stage38["final_avg_steps"] == pytest.approx(284.5)

    # CLI test with --output-dir and --report
    exit_code = backfill_cli_main(["--output-dir", str(art_dir), "--report"])
    assert exit_code == 0


def test_backfill_promotion_history_fallback_and_stopped_final_stage(tmp_path: Path) -> None:
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True)
    run_id = "run_20261124_120000_abcd"

    (art_dir / "promotion-history.json").write_text(
        json.dumps(
            [
                {
                    "event_type": "promotion",
                    "promotion_event_id": "evt_p1",
                    "run_id": run_id,
                    "from_stage": 1,
                    "to_stage": 2,
                    "promoted_at": "2026-11-24T12:05:00+00:00",
                    "trigger_checkpoint_id": f"chk_{run_id}_pv_5_ts_50000",
                    "success_rate": 1.0,
                    "evaluation_seed_count": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    (art_dir / "stage-timing-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": 2,
                "started_at": "2026-11-24T12:05:00+00:00",
                "start_timesteps": 50000,
            }
        ),
        encoding="utf-8",
    )
    (art_dir / "training-runtime.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "run_id": run_id,
                        "status": "ended",
                        "end_reason": "manually_stopped",
                        "started_at": "2026-11-24T12:00:00+00:00",
                        "ended_at": "2026-11-24T12:10:00+00:00",
                        "stage_seconds": {
                            "1": {"training_seconds": 250.0},
                            "2": {"training_seconds": 200.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (art_dir / "effective-training-config.json").write_text(
        json.dumps({"training": {"observation_mode": "emergent"}}),
        encoding="utf-8",
    )

    store = ModelHistoryStore(art_dir / MODEL_HISTORY_DB_FILENAME)
    res = store.backfill_from_artifacts(art_dir)
    assert res["imported_passes"] == 1
    assert res["imported_stage_runs"] == 2
    pass_rec = store.get_pass(f"pass_backfill_{run_id}")
    assert pass_rec is not None
    assert pass_rec["assistance_mode"] == "emergent"
    assert pass_rec["status"] == "stopped"
    assert pass_rec["stage_runs"][0]["status"] == "promoted"
    assert pass_rec["stage_runs"][1]["status"] == "stopped"


def test_runtime_tracker_get_stage_phase_totals_excludes_paused(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = TrainingRuntimeTracker(
        tmp_path / "training-runtime.json",
        monotonic=clock.monotonic,
        utc_now=clock.utc_now,
        start_heartbeat_thread=False,
    )

    # Before any session
    empty = tracker.get_stage_phase_totals("run-1", 25)
    assert empty == {
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "checkpoint_save_seconds": 0.0,
        "replay_serialization_seconds": 0.0,
        "active_seconds": 0.0,
    }

    tracker.start_session("run-1", stage=25)
    clock.advance(10.0)
    tracker.transition("evaluation")
    clock.advance(4.0)
    tracker.transition("checkpoint_save")
    clock.advance(2.0)
    tracker.transition("replay_serialization")
    clock.advance(1.5)
    tracker.transition("paused")
    clock.advance(30.0)  # paused time must NOT count toward stage active/phase totals
    tracker.transition("training")
    clock.advance(5.0)
    tracker.end_session("completed")

    # Second session on same run and stage plus another stage/run
    tracker.start_session("run-1", stage=25)
    clock.advance(3.0)
    # Live call before ending session accumulates up to clock.monotonic()
    live_totals = tracker.get_stage_phase_totals("run-1", 25)
    assert live_totals["training_seconds"] == pytest.approx(18.0)
    assert live_totals["evaluation_seconds"] == pytest.approx(4.0)
    assert live_totals["checkpoint_save_seconds"] == pytest.approx(2.0)
    assert live_totals["replay_serialization_seconds"] == pytest.approx(1.5)
    assert live_totals["active_seconds"] == pytest.approx(25.5)

    tracker.start_session("run-1", stage=26)
    clock.advance(7.0)
    tracker.end_session("completed")

    # Test run_id=None matching
    tracker.start_session(None, stage=25)
    clock.advance(6.0)
    tracker.end_session("completed")

    none_totals = tracker.get_stage_phase_totals(None, 25)
    assert none_totals["training_seconds"] == pytest.approx(6.0)
    assert none_totals["active_seconds"] == pytest.approx(6.0)
    assert tracker.get_stage_phase_totals("run-1", 26)["training_seconds"] == pytest.approx(7.0)
