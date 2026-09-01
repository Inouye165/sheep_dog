"""Durable active-runtime and phase timing for training sessions."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json

PHASE_FIELDS = {
    "training": "training_seconds",
    "evaluation": "evaluation_seconds",
    "replay_capture": "replay_capture_seconds",
    "replay_serialization": "replay_serialization_seconds",
    "checkpoint_save": "checkpoint_save_seconds",
    "paused": "paused_seconds",
}


class TrainingRuntimeTracker:
    """Measure verified process activity without counting offline gaps."""

    def __init__(
        self,
        path: str | Path,
        *,
        heartbeat_seconds: float = 15.0,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] | None = None,
        start_heartbeat_thread: bool = True,
    ) -> None:
        self.path = Path(path)
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self._monotonic = monotonic
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._session: dict[str, Any] | None = None
        self._phase_started_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._start_heartbeat_thread = start_heartbeat_thread
        self._payload = self._load()
        self._close_stale_session()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                import json

                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
                    return payload
            except (OSError, ValueError):
                pass
        return {"schema_version": 1, "sessions": []}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self._payload)

    def _close_stale_session(self) -> None:
        sessions = self._payload["sessions"]
        if not sessions or sessions[-1].get("status") not in {"running", "paused"}:
            return
        stale = sessions[-1]
        process_id = stale.get("process_id")
        try:
            if self._process_is_alive(process_id):
                return
        except Exception:
            pass
        stale["ended_at"] = stale.get("last_heartbeat_at") or stale.get("started_at")
        stale["end_reason"] = "crashed_or_stale"
        stale["status"] = "ended"
        stale["current_phase"] = None
        self._persist()

    @staticmethod
    def _process_is_alive(process_id: Any) -> bool:
        """Cross-platform check if a process ID is currently running."""
        if not isinstance(process_id, int) or process_id <= 0:
            return False
        try:
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                ERROR_ACCESS_DENIED = 5
                STILL_ACTIVE = 259

                handle = kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, process_id
                )
                if not handle:
                    return kernel32.GetLastError() == ERROR_ACCESS_DENIED

                exit_code = ctypes.c_ulong()
                try:
                    if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return exit_code.value == STILL_ACTIVE
                    return False
                finally:
                    kernel32.CloseHandle(handle)
            else:
                os.kill(process_id, 0)
                return True
        except PermissionError:
            return True
        except (OSError, OverflowError, ValueError, Exception):
            return False

    def start_session(self, run_id: str | None, *, session_id: str | None = None) -> str:
        """Start a measured process session and its heartbeat."""
        with self._lock:
            if self._session is not None:
                return str(self._session["session_id"])
            now = self._utc_now().isoformat()
            session = {
                "session_id": session_id or f"session_{uuid.uuid4().hex}",
                "run_id": run_id,
                "process_id": os.getpid(),
                "started_at": now,
                "last_heartbeat_at": now,
                "ended_at": None,
                "end_reason": None,
                "status": "running",
                "current_phase": "training",
                **{field: 0.0 for field in PHASE_FIELDS.values()},
                "active_seconds_total": 0.0,
            }
            self._payload["sessions"].append(session)
            self._session = session
            self._phase_started_monotonic = self._monotonic()
            self._persist()
            self._ensure_heartbeat_thread()
            return str(session["session_id"])

    def _ensure_heartbeat_thread(self) -> None:
        if not self._start_heartbeat_thread or self._heartbeat_thread is not None:
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="training-runtime-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            try:
                self.heartbeat()
            except OSError:
                continue

    def _accumulate(self, now_monotonic: float) -> None:
        if self._session is None or self._phase_started_monotonic is None:
            return
        phase = self._session.get("current_phase")
        elapsed = max(0.0, now_monotonic - self._phase_started_monotonic)
        field = PHASE_FIELDS.get(str(phase))
        if field is not None:
            self._session[field] = float(self._session.get(field, 0.0)) + elapsed
        self._session["active_seconds_total"] = sum(
            float(self._session.get(name, 0.0)) for name in PHASE_FIELDS.values()
        )
        self._phase_started_monotonic = now_monotonic

    def transition(self, phase: str) -> None:
        """Move to one non-overlapping measured phase."""
        if phase not in PHASE_FIELDS:
            raise ValueError(f"Unknown runtime phase: {phase}")
        with self._lock:
            if self._session is None:
                return
            self._accumulate(self._monotonic())
            self._session["current_phase"] = phase
            self._session["status"] = "paused" if phase == "paused" else "running"

    @contextlib.contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        """Measure an operation and restore its previous phase on exit."""
        with self._lock:
            previous = self._session.get("current_phase") if self._session else None
        self.transition(phase)
        try:
            yield
        finally:
            if previous in PHASE_FIELDS:
                self.transition(str(previous))

    def heartbeat(self) -> None:
        """Persist current totals and the latest confirmed UTC activity."""
        with self._lock:
            if self._session is None:
                return
            self._accumulate(self._monotonic())
            self._session["last_heartbeat_at"] = self._utc_now().isoformat()
            self._persist()

    def end_session(self, reason: str) -> None:
        """Flush and close the current measured session."""
        with self._lock:
            if self._session is None:
                return
            self._accumulate(self._monotonic())
            now = self._utc_now().isoformat()
            self._session.update(
                {
                    "last_heartbeat_at": now,
                    "ended_at": now,
                    "end_reason": reason,
                    "status": "ended",
                    "current_phase": None,
                }
            )
            self._persist()
            self._session = None
            self._phase_started_monotonic = None
            self._stop_event.set()
            self._heartbeat_thread = None

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative measured and wall-clock timing."""
        with self._lock:
            if self._session is not None:
                self._accumulate(self._monotonic())
            sessions = self._payload["sessions"]
            totals = {
                field: sum(float(session.get(field, 0.0)) for session in sessions)
                for field in PHASE_FIELDS.values()
            }
            active_total = sum(totals.values())
            first_started = sessions[0].get("started_at") if sessions else None
            latest = None
            if sessions:
                latest = sessions[-1].get("ended_at") or sessions[-1].get("last_heartbeat_at")
            wall_clock = 0.0
            if first_started and latest:
                wall_clock = max(
                    0.0,
                    (
                        datetime.fromisoformat(latest) - datetime.fromisoformat(first_started)
                    ).total_seconds(),
                )
            current = self._session
            return {
                **totals,
                "active_seconds_total": active_total,
                "wall_clock_seconds": wall_clock,
                "offline_or_unknown_seconds": max(0.0, wall_clock - active_total),
                "session_id": current.get("session_id") if current else None,
                "run_id": current.get("run_id") if current else None,
                "current_phase": current.get("current_phase") if current else None,
                "session_count": len(sessions),
                "sessions": [dict(session) for session in sessions],
            }

    def episode_snapshot(self) -> dict[str, Any]:
        """Return additive cumulative fields for checkpoint history records."""
        summary = self.snapshot()
        return {
            "active_runtime_seconds_total": summary["active_seconds_total"],
            "training_seconds_total": summary["training_seconds"],
            "evaluation_seconds_total": summary["evaluation_seconds"],
            "wall_clock_elapsed_seconds": summary["wall_clock_seconds"],
            "session_id": summary["session_id"],
        }

    @property
    def active_seconds_total(self) -> float:
        """Return cumulative active runtime across all sessions."""
        return float(self.snapshot().get("active_seconds_total", 0.0))
