"""Asynchronous, non-blocking writer for selective authentic training replays."""

from __future__ import annotations

import atexit
import contextlib
import gzip
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CapturePolicy:
    """Configurable capture policy for training episode replays."""

    mode: str = "selective"  # "off", "failures", "selective", "next_n", "all"
    next_n_counter: int = 0
    success_sample_rate: float = 0.05
    target_stage: int | None = None
    target_outcome: str = "all"  # "all", "failures"
    max_replays_per_stage: int = 50
    max_total_replays: int = 200
    max_disk_mb: float = 500.0

    def should_capture(
        self,
        *,
        stage: int,
        success: bool,
        status: str,
        reward: float,
    ) -> tuple[bool, str]:
        """Determine if an episode should be captured and return (should_capture, capture_reason)."""
        if self.mode == "off":
            return False, "not_requested"

        # Check next_n override first
        if self.next_n_counter > 0:
            if self.target_stage is not None and stage != self.target_stage:
                pass  # Skip if stage doesn't match filter
            elif self.target_outcome == "failures" and success:
                pass  # Skip if filtering for failures only
            else:
                self.next_n_counter -= 1
                return True, "next_n"

        if self.mode == "all":
            return True, "development_all"

        # Unsuccessful episodes (TIMEOUT, STOPPED, or non-success)
        is_failure = not success or status in {"TIMEOUT", "STOPPED"}
        if is_failure and self.mode in {"failures", "selective"}:
            reason = (
                "timeout"
                if status == "TIMEOUT"
                else "stopped"
                if status == "STOPPED"
                else "unsuccessful_terminal"
            )
            return True, reason

        # Success sampling in selective mode
        if self.mode == "selective" and success:
            import random

            if self.success_sample_rate > 0 and random.random() < self.success_sample_rate:
                return True, "sampled_success"

        return False, "not_requested"


_global_capture_policy: CapturePolicy = CapturePolicy()


def get_global_capture_policy() -> CapturePolicy:
    global _global_capture_policy
    return _global_capture_policy


def set_global_capture_policy(policy: CapturePolicy) -> None:
    global _global_capture_policy
    _global_capture_policy = policy


@dataclass
class ReplayWriteJob:
    replay_id: str
    event_key: str
    payload: dict[str, Any]
    output_path: Path
    capture_reason: str
    replay_source: str = "training-diagnostic"
    use_gzip: bool = True


class AsyncReplayWriter:
    """Thread-safe background worker for serializing episode replays to disk."""

    def __init__(
        self,
        output_dir: str | Path = "artifacts/replays",
        max_queue_size: int = 100,
        episode_store: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue_size = max_queue_size
        self.episode_store = episode_store

        self._queue: queue.Queue[ReplayWriteJob | None] = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._busy: bool = False

        self.queued_count: int = 0
        self.written_count: int = 0
        self.dropped_count: int = 0
        self.failure_count: int = 0
        self.last_error: str | None = None

        self.start()
        atexit.register(self.close)

    def start(self) -> None:
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name="AsyncReplayWriter"
                )
                self._worker_thread.start()

    def enqueue(self, job: ReplayWriteJob) -> bool:
        """Enqueue a replay write job without blocking rollout collection."""
        try:
            self._queue.put_nowait(job)
            self.queued_count += 1
            if self.episode_store:
                self.episode_store.update_replay_info(
                    event_key=job.event_key,
                    replay_available=0,
                    replay_id=job.replay_id,
                    replay_path=str(job.output_path),
                    replay_source=job.replay_source,
                    capture_reason=job.capture_reason,
                    capture_status="queued",
                )
            return True
        except queue.Full:
            self.dropped_count += 1
            logger.warning(
                "AsyncReplayWriter queue is full (%d). Dropped replay %s",
                self.max_queue_size,
                job.replay_id,
            )
            if self.episode_store:
                self.episode_store.update_replay_info(
                    event_key=job.event_key,
                    replay_available=0,
                    replay_id=job.replay_id,
                    replay_path=None,
                    replay_source=job.replay_source,
                    capture_reason=job.capture_reason,
                    capture_status="failed",
                )
            return False

    def _write_job(self, job: ReplayWriteJob) -> None:
        tmp_path: Path | None = None
        try:
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = job.output_path.with_suffix(job.output_path.suffix + ".tmp")

            if job.use_gzip:
                with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
                    json.dump(job.payload, handle, indent=None)
            else:
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(job.payload, handle, indent=2)

            os.replace(tmp_path, job.output_path)
            self.written_count += 1

            if self.episode_store:
                self.episode_store.update_replay_info(
                    event_key=job.event_key,
                    replay_available=1,
                    replay_id=job.replay_id,
                    replay_path=str(job.output_path),
                    replay_source=job.replay_source,
                    capture_reason=job.capture_reason,
                    capture_status="available",
                )

            # Check retention limits and prune if necessary
            policy = get_global_capture_policy()
            if self.episode_store and hasattr(self.episode_store, "prune_replays"):
                self.episode_store.prune_replays(
                    max_files_per_stage=policy.max_replays_per_stage,
                    max_total_files=policy.max_total_replays,
                    max_disk_mb=policy.max_disk_mb,
                )
        except Exception as exc:
            self.failure_count += 1
            self.last_error = str(exc)
            logger.error("Failed to write replay file %s: %s", job.output_path, exc)
            if tmp_path is not None and tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            if self.episode_store:
                self.episode_store.update_replay_info(
                    event_key=job.event_key,
                    replay_available=0,
                    replay_id=job.replay_id,
                    replay_path=None,
                    replay_source=job.replay_source,
                    capture_reason=job.capture_reason,
                    capture_status="failed",
                )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
                if job is None:
                    break
                self._busy = True
                try:
                    self._write_job(job)
                finally:
                    self._busy = False
                    self._queue.task_done()
            except queue.Empty:
                pass
            except Exception as exc:
                self._busy = False
                self.failure_count += 1
                self.last_error = str(exc)
                logger.exception("Unexpected error in AsyncReplayWriter loop: %s", exc)

    def flush(self) -> None:
        """Synchronously flush remaining write jobs and wait for active writer."""
        while True:
            try:
                job = self._queue.get_nowait()
                if job is not None:
                    self._busy = True
                    try:
                        self._write_job(job)
                    finally:
                        self._busy = False
                        self._queue.task_done()
            except queue.Empty:
                break
        with contextlib.suppress(Exception):
            self._queue.join()
        while self._busy:
            time.sleep(0.01)

    def close(self) -> None:
        self.flush()
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)


_global_replay_writer: AsyncReplayWriter | None = None


def get_replay_writer(
    output_dir: str | Path = "artifacts/replays", episode_store: Any | None = None
) -> AsyncReplayWriter:
    global _global_replay_writer
    if _global_replay_writer is None:
        _global_replay_writer = AsyncReplayWriter(
            output_dir=output_dir, episode_store=episode_store
        )
    return _global_replay_writer
