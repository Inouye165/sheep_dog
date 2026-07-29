"""Deterministic tests for durable training runtime measurement."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from sheepdog.training.runtime import TrainingRuntimeTracker


class FakeClock:
    """Controllable monotonic and UTC clock pair."""

    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.utc_value = datetime(2026, 7, 28, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.monotonic_value

    def utc_now(self) -> datetime:
        return self.utc_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.utc_value += timedelta(seconds=seconds)


def make_tracker(tmp_path, clock: FakeClock) -> TrainingRuntimeTracker:
    return TrainingRuntimeTracker(
        tmp_path / "training-runtime.json",
        monotonic=clock.monotonic,
        utc_now=clock.utc_now,
        start_heartbeat_thread=False,
    )


def test_phases_accumulate_without_overlap_and_flush_on_end(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1", session_id="session-1")
    clock.advance(5)
    tracker.transition("evaluation")
    clock.advance(3)
    tracker.end_session("completed")

    summary = tracker.snapshot()
    assert summary["training_seconds"] == 5
    assert summary["evaluation_seconds"] == 3
    assert summary["active_seconds_total"] == 8


def test_phase_context_records_exception_and_reraises(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1")

    with pytest.raises(RuntimeError, match="original"), tracker.phase("checkpoint_save"):
        clock.advance(4)
        raise RuntimeError("original")

    assert tracker.snapshot()["checkpoint_save_seconds"] == 4


def test_heartbeat_updates_live_session(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1")
    clock.advance(12)
    tracker.heartbeat()

    payload = json.loads((tmp_path / "training-runtime.json").read_text(encoding="utf-8"))
    assert payload["sessions"][0]["last_heartbeat_at"] == clock.utc_now().isoformat()
    assert payload["sessions"][0]["training_seconds"] == 12


def test_stale_session_closes_at_last_heartbeat_without_counting_offline_gap(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1")
    clock.advance(15)
    tracker.heartbeat()
    clock.advance(6 * 60 * 60)

    # Mutate process_id to a dead PID (e.g. 999999) to simulate process crash
    runtime_path = tmp_path / "training-runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["sessions"][0]["process_id"] = 999999
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = make_tracker(tmp_path, clock)
    session = recovered.snapshot()["sessions"][0]
    assert session["ended_at"] == session["last_heartbeat_at"]
    assert session["end_reason"] == "crashed_or_stale"
    assert session["training_seconds"] == 15


def test_intentional_pause_is_measured_while_process_is_live(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1")
    clock.advance(2)
    tracker.transition("paused")
    clock.advance(7)
    tracker.heartbeat()

    summary = tracker.snapshot()
    assert summary["training_seconds"] == 2
    assert summary["paused_seconds"] == 7
    assert summary["offline_or_unknown_seconds"] == 0


def test_process_is_alive_invalid_pids() -> None:
    assert not TrainingRuntimeTracker._process_is_alive(None)
    assert not TrainingRuntimeTracker._process_is_alive("invalid_pid")
    assert not TrainingRuntimeTracker._process_is_alive(0)
    assert not TrainingRuntimeTracker._process_is_alive(-10)


def test_process_is_alive_current_and_dead_pids() -> None:
    import os

    assert TrainingRuntimeTracker._process_is_alive(os.getpid())
    assert not TrainingRuntimeTracker._process_is_alive(999999)


def test_close_stale_session_handles_exceptions_gracefully(tmp_path) -> None:
    clock = FakeClock()
    tracker = make_tracker(tmp_path, clock)
    tracker.start_session("run-1")

    # Put a non-integer or bad PID in the session file
    runtime_path = tmp_path / "training-runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["sessions"][0]["process_id"] = "not_an_int"
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    # Creating a new tracker should not throw any exception
    recovered = make_tracker(tmp_path, clock)
    session = recovered.snapshot()["sessions"][0]
    assert session["status"] == "ended"
    assert session["end_reason"] == "crashed_or_stale"