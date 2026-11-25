"""Prove stage training_seconds excludes pauses, stops, offline gaps and non-training phases."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sheepdog import server as server_module
from sheepdog.training import stage_timing
from sheepdog.training.runtime import TrainingRuntimeTracker

RUN = "run_pause_resume"


class FakeClock:
    def __init__(self) -> None:
        self.mono = 100.0
        self.utc = datetime(2026, 7, 28, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.mono

    def utc_now(self) -> datetime:
        return self.utc

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.utc += timedelta(seconds=seconds)


def _tracker(path: Path, clock: FakeClock) -> TrainingRuntimeTracker:
    return TrainingRuntimeTracker(
        path, monotonic=clock.monotonic, utc_now=clock.utc_now, start_heartbeat_thread=False
    )


def _stage(root: Path, stage: int = 1) -> dict[str, float]:
    seconds = stage_timing._runtime_stage_seconds(root, RUN, stage)
    assert seconds is not None
    return seconds


def _ledger(root: Path) -> dict:
    return json.loads((root / "training-runtime.json").read_text(encoding="utf-8"))


def test_pause_is_excluded_from_stage_training_seconds(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = _tracker(tmp_path / "training-runtime.json", clock)
    tracker.start_session(RUN, stage=1)
    clock.advance(100)
    tracker.transition("paused")
    clock.advance(500)
    tracker.transition("training")
    clock.advance(50)
    tracker.end_session("completed")

    assert _stage(tmp_path)["training_seconds"] == 150
    session = _ledger(tmp_path)["sessions"][0]
    assert session["paused_seconds"] == 500
    assert session["training_seconds"] == 150
    assert "paused_seconds" not in session["stage_seconds"]["1"]


def test_stop_and_offline_gap_excluded_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "training-runtime.json"
    clock = FakeClock()
    first = _tracker(path, clock)
    first.start_session(RUN, stage=1)
    clock.advance(100)
    first.end_session("manually_stopped")

    clock.advance(8 * 3600)  # offline gap

    second = _tracker(path, clock)
    second.start_session(RUN, stage=1)
    clock.advance(60)
    second.end_session("completed")

    assert _stage(tmp_path)["training_seconds"] == 160
    sessions = _ledger(tmp_path)["sessions"]
    assert [s["end_reason"] for s in sessions] == ["manually_stopped", "completed"]
    assert second.snapshot()["offline_or_unknown_seconds"] >= 8 * 3600 - 1


def test_crashed_session_closed_at_last_heartbeat_and_next_session_adds(tmp_path: Path) -> None:
    path = tmp_path / "training-runtime.json"
    clock = FakeClock()
    crashed = _tracker(path, clock)
    crashed.start_session(RUN, stage=1)
    clock.advance(40)
    crashed.heartbeat()
    last_heartbeat = clock.utc_now().isoformat()
    clock.advance(6 * 3600)  # process died silently; no end_session

    payload = _ledger(tmp_path)
    payload["sessions"][0]["process_id"] = 999_999_999  # dead PID
    path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = _tracker(path, clock)  # closes stale session on load
    stale = _ledger(tmp_path)["sessions"][0]
    assert stale["end_reason"] == "crashed_or_stale"
    assert stale["ended_at"] == last_heartbeat
    recovered.start_session(RUN, stage=1)
    clock.advance(30)
    recovered.end_session("completed")

    assert _stage(tmp_path)["training_seconds"] == 70  # 40 + 30, 6h gap excluded


def test_checkpoint_save_and_evaluation_excluded_but_recorded(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = _tracker(tmp_path / "training-runtime.json", clock)
    tracker.start_session(RUN, stage=1)
    clock.advance(100)
    with tracker.phase("checkpoint_save"):
        clock.advance(7)
    clock.advance(20)
    with tracker.phase("evaluation"):
        clock.advance(13)
    clock.advance(5)
    tracker.end_session("completed")

    seconds = _stage(tmp_path)
    assert seconds["training_seconds"] == 125
    assert seconds["evaluation_seconds"] == 13
    block = _ledger(tmp_path)["sessions"][0]["stage_seconds"]["1"]
    assert block["checkpoint_save_seconds"] == 7
    assert block["training_seconds"] == 125


def test_record_promotion_uses_pause_free_training_seconds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stage_timing, "git_info", lambda: {"revision": None, "dirty": None})
    clock = FakeClock()
    tracker = _tracker(tmp_path / "training-runtime.json", clock)
    tracker.start_session(RUN, stage=1)
    clock.advance(100)
    tracker.transition("paused")
    clock.advance(500)
    tracker.transition("training")
    clock.advance(50)
    tracker.heartbeat()
    event = {
        "event_type": "promotion",
        "promotion_event_id": "e1",
        "run_id": RUN,
        "from_stage": 1,
        "to_stage": 2,
        "promoted_at": clock.utc_now().isoformat(),
        "trigger_checkpoint_id": f"chk_{RUN}_pv_1_ts_1500",
    }
    stage_timing.note_stage_start(tmp_path, RUN, 1, "2026-07-28T00:00:00+00:00", 0)
    record = stage_timing.record_promotion(tmp_path, event)
    assert record is not None
    assert record["wall_seconds"] == 650
    assert record["training_seconds"] == 150
    assert record["steps_per_second"] == 10.0


def test_server_pause_and_stop_branches_call_tracker() -> None:
    """Structural check only: the pause/stop branch lives inside the training thread loop.

    Driving it needs a live training run, which is out of scope, so this asserts the
    source wiring (pause -> transition('paused'); stop -> end_session('manually_stopped')).
    """

    import inspect

    source = inspect.getsource(server_module)
    pattern = re.compile(
        r'if control_request == "paused":\s*'
        r'self\.runtime_tracker\.transition\("paused"\)\s*'
        r"self\.runtime_tracker\.heartbeat\(\)\s*"
        r"else:\s*"
        r'self\.runtime_tracker\.end_session\("manually_stopped"\)'
    )
    assert pattern.search(source)
