"""Tests for per-stage training-time records, backfill, comparison and endpoint."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from sheepdog import server as server_module
from sheepdog.training import stage_timing


def _event(event_id: str, run: str, frm: int, at: str, ts: int, **extra) -> dict:
    return {
        "event_type": "promotion",
        "promotion_event_id": event_id,
        "run_id": run,
        "from_stage": frm,
        "to_stage": frm + 1,
        "promoted_at": at,
        "timestamp": at,
        "trigger_checkpoint_id": f"chk_{run}_pv_1_ts_{ts}",
        "success_rate": 1.0,
        "success_count": 6,
        **extra,
    }


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr(stage_timing, "git_info", lambda: {"revision": "abc1234", "dirty": False})
    monkeypatch.delenv(stage_timing.RUN_LABEL_ENV_VAR, raising=False)


def test_record_promotion_writes_recorded_record_with_ledger_seconds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(stage_timing.RUN_LABEL_ENV_VAR, "baseline")
    (tmp_path / "training-runtime.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {"run_id": "run_x", "stage_seconds": {"1": {"training_seconds": 60.0, "evaluation_seconds": 5.0}}},
                    {"run_id": "run_x", "stage_seconds": {"1": {"training_seconds": 40.0}}},
                ]
            }
        )
    )
    (tmp_path / "effective-training-config.json").write_text(
        json.dumps({"training": {"observation_mode": "guided", "ppo_env_workers": 16}, "environment": {"dogs": 3}})
    )
    (tmp_path / "state").mkdir()
    stage_timing.note_stage_start(tmp_path, "run_x", 1, "2026-11-24T10:00:00+00:00", 0)

    rec = stage_timing.record_promotion(
        tmp_path, _event("e1", "run_x", 1, "2026-11-24T10:10:00+00:00", 1000)
    )

    assert rec is not None
    assert rec["source"] == "recorded"
    assert rec["wall_seconds"] == 600.0
    assert rec["training_seconds"] == 100.0
    assert rec["evaluation_seconds"] == 5.0
    assert rec["policy_timesteps"] == 1000
    assert rec["steps_per_second"] == 10.0
    assert rec["steps_per_second_basis"] == "training_seconds"
    assert rec["assistance_mode"] == "guided"
    assert rec["config_hash"]
    assert rec["run_label"] == "baseline"
    assert rec["git_revision"] == "abc1234" and rec["git_dirty"] is False
    assert rec["software"]["python"]
    assert rec["hardware"]["cpu_count"]
    stored = stage_timing.read_records(tmp_path)
    assert len(stored) == 1
    # Idempotent by event id; next stage start was persisted.
    assert stage_timing.record_promotion(tmp_path, _event("e1", "run_x", 1, "2026-11-24T10:10:00+00:00", 1000)) is None
    state = json.loads((tmp_path / stage_timing.STAGE_TIMING_STATE_FILENAME).read_text())
    assert state["stage"] == 2 and state["started_at"] == "2026-11-24T10:10:00+00:00"


def test_read_records_tolerates_torn_line_and_append_recovers(tmp_path: Path) -> None:
    path = tmp_path / stage_timing.STAGE_TIMING_FILENAME
    path.write_text('{"run_id":"a","stage":1}\n{"run_id":"b","sta')
    assert len(stage_timing.read_records(tmp_path)) == 1
    stage_timing._append_record(tmp_path, {"run_id": "c", "stage": 2})
    assert [r["run_id"] for r in stage_timing.read_records(tmp_path)] == ["a", "c"]


def test_backfill_is_idempotent_and_chains_stage_starts(tmp_path: Path) -> None:
    events = [
        _event("e1", "run_20261124_093542_33c5", 1, "2026-11-24T17:47:00+00:00", 1000),
        _event("e2", "run_20261124_093542_33c5", 2, "2026-11-24T17:52:00+00:00", 1500),
        {"event_type": "manual_change", "promotion_event_id": "m1", "run_id": "r", "from_stage": 3, "to_stage": 1},
    ]
    assert stage_timing.backfill_from_promotion_history(tmp_path, events) == 2
    assert stage_timing.backfill_from_promotion_history(tmp_path, events) == 0
    records = stage_timing.read_records(tmp_path)
    assert len(records) == 2
    assert all(r["source"] == "derived" and r["training_seconds"] is None for r in records)
    assert records[1]["wall_seconds"] == 300.0
    assert records[1]["policy_timesteps"] == 500
    assert records[0]["timing_confidence"] == "low"  # start from run id
    assert records[1]["timing_confidence"] == "medium"
    # Recorded write for an already-derived event does not duplicate.
    assert stage_timing.record_promotion(tmp_path, events[0]) is None


def test_compare_stages_math_and_like_with_like_flags() -> None:
    def r(run: str, ended: str, wall: float, steps: int, sps: float, mode: str, cfg: str) -> dict:
        return {
            "run_id": run, "stage": 1, "ended_at": ended, "wall_seconds": wall,
            "policy_timesteps": steps, "steps_per_second": sps, "assistance_mode": mode,
            "config_hash": cfg, "source": "recorded",
        }

    records = [
        r("a", "2026-01-01T00:00:00+00:00", 100.0, 1000, 10.0, "guided", "h1"),
        r("b", "2026-01-02T00:00:00+00:00", 50.0, 1000, 20.0, "guided", "h1"),
        r("c", "2026-01-03T00:00:00+00:00", 75.0, 1500, 20.0, "autonomous", "h2"),
    ]
    entry = stage_timing.compare_stages(records)[0]
    assert entry["run_count"] == 3 and entry["latest_run_id"] == "c"
    prev = entry["vs_previous"]
    assert prev["run_id"] == "b"
    assert prev["deltas"]["wall_seconds"]["pct_delta"] == 50.0
    assert prev["deltas"]["policy_timesteps"]["pct_delta"] == 50.0
    assert prev["deltas"]["steps_per_second"]["pct_delta"] == 0.0
    assert prev["assistance_differs"] and prev["config_hash_differs"] and not prev["comparable"]
    assert entry["best_run_id"] == "b"
    assert entry["vs_best"]["run_id"] == "b"
    single = stage_timing.compare_stages(records[:1])[0]
    assert single["vs_previous"] is None and single["vs_best"] is None


def test_stage_timings_endpoint_backfills_and_returns_payload(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "promotion-history.json").write_text(
        json.dumps(
            [
                _event("e1", "run_20261124_093542_33c5", 1, "2026-11-24T17:47:00+00:00", 1000),
                _event("e2", "run_20261124_093542_33c5", 2, "2026-11-24T17:52:00+00:00", 1500),
            ]
        )
    )
    monkeypatch.setattr(
        server_module, "LabConfig", lambda: SimpleNamespace(training=SimpleNamespace(output_dir=str(tmp_path)))
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.TrainingRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/api/stage-timings"
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(url) as response:
            again = json.loads(response.read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert len(payload["records"]) == 2
    assert len(again["records"]) == 2  # idempotent
    assert payload["comparison"][0]["run_count"] == 1
    assert payload["records"][0]["source"] == "derived"


def test_append_promotion_history_hook_writes_timing_record(tmp_path: Path) -> None:
    server_module._append_promotion_history(tmp_path, _event("e1", "run_x", 1, "2026-11-24T10:10:00+00:00", 500))
    records = stage_timing.read_records(tmp_path)
    assert [r["promotion_event_id"] for r in records] == ["e1"]
    assert records[0]["source"] == "recorded"


def test_concurrent_backfill_writes_no_duplicates(tmp_path: Path) -> None:
    events = [
        _event(
            f"e{stage}",
            "run_20261124_093542_33c5",
            stage,
            f"2026-11-24T17:{stage:02d}:00+00:00",
            stage * 1000,
        )
        for stage in range(1, 21)
    ]
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    counts: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait()
            added = stage_timing.backfill_from_promotion_history(tmp_path, events)
            with lock:
                counts.append(added)
        except BaseException as exc:  # pragma: no cover - asserted empty below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert sum(counts) == len(events)
    raw_lines = [
        line for line in (tmp_path / stage_timing.STAGE_TIMING_FILENAME).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(raw_lines) == len(events)
    raw_ids = [json.loads(line)["promotion_event_id"] for line in raw_lines]
    assert raw_ids == [e["promotion_event_id"] for e in events]
    records = stage_timing.read_records(tmp_path)
    assert [r["promotion_event_id"] for r in records] == [e["promotion_event_id"] for e in events]


def test_read_records_deduplicates_preferring_recorded_over_derived(tmp_path: Path) -> None:
    path = tmp_path / stage_timing.STAGE_TIMING_FILENAME
    rows = [
        {"promotion_event_id": "e1", "source": "derived", "stage": 1, "wall_seconds": 100.0},
        {"promotion_event_id": "e1", "source": "derived", "stage": 1, "wall_seconds": 101.0},
        {"promotion_event_id": "e2", "source": "recorded", "stage": 2, "wall_seconds": 200.0},
        {"promotion_event_id": "e1", "source": "recorded", "stage": 1, "wall_seconds": 99.0},
        {"promotion_event_id": "e2", "source": "derived", "stage": 2, "wall_seconds": 201.0},
        {"promotion_event_id": "e2", "source": "recorded", "stage": 2, "wall_seconds": 202.0},
        {"promotion_event_id": "e3", "source": "derived", "stage": 3, "wall_seconds": 300.0},
        {"promotion_event_id": "e3", "source": "derived", "stage": 3, "wall_seconds": 301.0},
        {"run_id": "legacy_a", "stage": 1},
        {"run_id": "legacy_b", "stage": 2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    records = stage_timing.read_records(tmp_path)
    assert len(records) == 5
    assert [r.get("promotion_event_id") for r in records] == ["e1", "e2", "e3", None, None]
    assert records[0]["source"] == "recorded" and records[0]["wall_seconds"] == 99.0
    assert records[1]["source"] == "recorded" and records[1]["wall_seconds"] == 200.0
    assert records[2]["source"] == "derived" and records[2]["wall_seconds"] == 300.0
    assert [records[3]["run_id"], records[4]["run_id"]] == ["legacy_a", "legacy_b"]
