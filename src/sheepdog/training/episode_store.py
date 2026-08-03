"""Durable SQLite storage for training episode telemetry in Sheepdog."""

from __future__ import annotations

import atexit
import datetime
import logging
import os
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EpisodeStore:
    """Thread-safe, queue-backed SQLite store for training episode telemetry."""

    def __init__(
        self,
        db_path: str | Path = "artifacts/training-telemetry.sqlite",
        max_queue_size: int = 10000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self.dropped_count: int = 0
        self.error_count: int = 0
        self.last_error: str | None = None

        self._init_db()
        self.start_worker()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    run_id TEXT,
                    session_id TEXT,
                    global_environment_episode INTEGER,
                    episode_in_stage INTEGER,
                    curriculum_stage INTEGER,
                    global_timestep INTEGER,
                    policy_version INTEGER,
                    completed_at TEXT,
                    active_runtime_seconds_total REAL,
                    reward REAL,
                    result TEXT,
                    success INTEGER,
                    timeout INTEGER,
                    stopped INTEGER,
                    sheep_penned INTEGER,
                    total_sheep INTEGER,
                    steps INTEGER,
                    seed INTEGER,
                    checkpoint_id TEXT,
                    created_at TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_id ON training_episodes(id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON training_episodes(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_stage ON training_episodes(curriculum_stage);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_global_ep ON training_episodes(global_environment_episode);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_global_ts ON training_episodes(global_timestep);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_completed_at ON training_episodes(completed_at);")
            conn.commit()

    def start_worker(self) -> None:
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name="EpisodeStoreWorker"
                )
                self._worker_thread.start()

    def add_episode(self, episode_data: dict[str, Any]) -> None:
        """Enqueue an episode record for asynchronous persistence without blocking training."""
        if not episode_data:
            return
        try:
            self._queue.put_nowait(episode_data)
        except queue.Full:
            self.dropped_count += 1
            logger.warning(
                "EpisodeStore queue is full (%d records). Dropped episode event key=%s",
                self._queue.maxsize,
                episode_data.get("event_key"),
            )

    def _build_event_key(self, record: dict[str, Any]) -> str:
        if record.get("event_key"):
            return str(record["event_key"])
        run_id = record.get("run_id") or "run_unknown"
        global_ep = record.get("global_environment_episode")
        if global_ep is None:
            global_ep = record.get("episode") or record.get("current_episode") or 0
        env_idx = record.get("env_index", 0)
        seed = record.get("seed", 0)
        return f"{run_id}_ep_{global_ep}_env_{env_idx}_seed_{seed}"

    def _insert_batch(self, conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
        inserted = 0
        sql = """
        INSERT OR IGNORE INTO training_episodes (
            event_key, run_id, session_id, global_environment_episode,
            episode_in_stage, curriculum_stage, global_timestep, policy_version,
            completed_at, active_runtime_seconds_total, reward, result,
            success, timeout, stopped, sheep_penned, total_sheep,
            steps, seed, checkpoint_id, created_at
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        params_list = []
        for r in records:
            event_key = self._build_event_key(r)
            run_id = r.get("run_id") or "run_unknown"
            session_id = r.get("session_id")
            global_ep = r.get("global_environment_episode")
            if global_ep is None:
                global_ep = r.get("episode") or r.get("current_episode") or 0
            ep_in_stage = r.get("episode_in_stage", global_ep)
            stage = r.get("curriculum_stage")
            if stage is None:
                stage = r.get("stage", 1)
            global_ts = r.get("global_timestep")
            policy_ver = r.get("policy_version")
            completed_at = r.get("completed_at") or r.get("timestamp") or now_iso
            runtime = r.get("active_runtime_seconds_total")
            reward = float(r.get("reward", 0.0))
            status = str(r.get("status") or r.get("result") or "UNKNOWN")
            success_bool = bool(r.get("success", False))
            success = 1 if success_bool else 0
            timeout = 1 if status == "TIMEOUT" else 0
            stopped = 1 if status == "STOPPED" else 0
            penned = int(r.get("penned", 0))
            total_sheep = int(r.get("total_sheep", 0))
            steps = int(r.get("length") or r.get("steps") or 0)
            seed = r.get("seed")
            checkpoint_id = r.get("checkpoint_id")
            created_at = r.get("created_at") or now_iso

            params_list.append((
                event_key, run_id, session_id, global_ep,
                ep_in_stage, stage, global_ts, policy_ver,
                completed_at, runtime, reward, status,
                success, timeout, stopped, penned, total_sheep,
                steps, seed, checkpoint_id, created_at
            ))

        try:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            inserted = cursor.rowcount if cursor.rowcount > 0 else 0
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to insert training episodes batch into SQLite: %s", exc)
        return inserted

    def _worker_loop(self) -> None:
        try:
            conn = self._get_connection()
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("EpisodeStore worker failed to open SQLite connection: %s", exc)
            return

        try:
            batch: list[dict[str, Any]] = []
            while not self._stop_event.is_set():
                try:
                    record = self._queue.get(timeout=0.2)
                    if record is None:
                        break
                    batch.append(record)
                    while len(batch) < 100:
                        try:
                            r = self._queue.get_nowait()
                            if r is None:
                                break
                            batch.append(r)
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass

                if batch:
                    self._insert_batch(conn, batch)
                    batch.clear()
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.exception("Unexpected error in EpisodeStore worker loop: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def flush(self) -> None:
        """Synchronously write all pending queued episodes to SQLite."""
        batch: list[dict[str, Any]] = []
        while True:
            try:
                record = self._queue.get_nowait()
                if record is not None:
                    batch.append(record)
            except queue.Empty:
                break

        if batch:
            try:
                with self._get_connection() as conn:
                    self._insert_batch(conn, batch)
            except Exception as exc:
                self.error_count += 1
                self.last_error = str(exc)
                logger.error("Failed to flush pending episodes to SQLite: %s", exc)

    def close(self) -> None:
        self.flush()
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def __enter__(self) -> EpisodeStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_episodes(
        self,
        after_id: int | None = None,
        before_id: int | None = None,
        stage: int | None = None,
        run_id: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Query episodes for Insights API with support for pagination and stage filtering."""
        limit = min(max(1, limit), 5000)
        conditions: list[str] = []
        params: list[Any] = []

        if after_id is not None:
            conditions.append("id > ?")
            params.append(after_id)
        if before_id is not None:
            conditions.append("id < ?")
            params.append(before_id)
        if stage is not None:
            conditions.append("curriculum_stage = ?")
            params.append(stage)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM training_episodes
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        """
        params.append(limit + 1)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                rows = [dict(row) for row in cursor.fetchall()]

                count_sql = f"SELECT COUNT(*), MIN(completed_at) FROM training_episodes {where_clause}"
                count_cursor = conn.execute(count_sql, params[:-1])
                total_matching, oldest_timestamp = count_cursor.fetchone()
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to query training episodes from SQLite: %s", exc)
            raise

        has_more = len(rows) > limit
        result_rows = rows[:limit]

        latest_id = result_rows[-1]["id"] if result_rows else (after_id or 0)
        next_after_id = latest_id

        # Convert integers back to boolean for frontend
        for r in result_rows:
            r["success"] = bool(r["success"])
            r["timeout"] = bool(r["timeout"])
            r["stopped"] = bool(r["stopped"])

        return {
            "episodes": result_rows,
            "latest_id": latest_id,
            "next_after_id": next_after_id,
            "has_more": has_more,
            "oldest_available_timestamp": oldest_timestamp or None,
            "total_matching": total_matching or 0,
        }

    def clear_store(self) -> None:
        """Clear all stored training episodes (for explicit journey resets)."""
        self.flush()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM training_episodes;")
            conn.commit()


_global_episode_store: EpisodeStore | None = None


def get_episode_store(db_path: str | Path = "artifacts/training-telemetry.sqlite") -> EpisodeStore:
    global _global_episode_store
    if _global_episode_store is None:
        _global_episode_store = EpisodeStore(db_path=db_path)
    return _global_episode_store


@atexit.register
def _on_exit_flush() -> None:
    global _global_episode_store
    if _global_episode_store is not None:
        try:
            _global_episode_store.close()
        except Exception:
            pass
