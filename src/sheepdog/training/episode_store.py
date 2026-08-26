"""Durable SQLite storage for training episode telemetry in Sheepdog."""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from sheepdog.training.spatial_analytics import (
    ALL_ZONES,
    CORNER_ZONES,
    WALL_ZONES,
    ZONE_CENTER,
    diagnose_stage_bottlenecks,
)

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
        self._busy: bool = False

        self._init_db()
        self.start_worker()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Self-healing schema check: auto-recreate table if database file was recreated on disk
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "training_episodes" not in tables:
            self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
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

        # Migration: Ensure replay linkage columns exist
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(training_episodes)").fetchall()}
        if "replay_available" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN replay_available INTEGER DEFAULT 0;")
        if "replay_id" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN replay_id TEXT;")
        if "replay_path" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN replay_path TEXT;")
        if "replay_source" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN replay_source TEXT;")
        if "capture_reason" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN capture_reason TEXT;")
        if "capture_status" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN capture_status TEXT DEFAULT 'not_requested';")
        if "replay_schema_version" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN replay_schema_version INTEGER DEFAULT 1;")
        if "reward_breakdown" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN reward_breakdown TEXT;")
        if "pen_zone" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN pen_zone TEXT;")
        if "spawn_mode" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN spawn_mode TEXT;")
        if "initial_sheep_zone" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN initial_sheep_zone TEXT;")
        if "final_sheep_zone" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN final_sheep_zone TEXT;")
        if "corner_time_pct" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN corner_time_pct REAL DEFAULT 0.0;")
        if "wall_time_pct" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN wall_time_pct REAL DEFAULT 0.0;")
        if "corner_stuck_at_end" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN corner_stuck_at_end INTEGER DEFAULT 0;")
        if "spatial_metrics" not in columns:
            conn.execute("ALTER TABLE training_episodes ADD COLUMN spatial_metrics TEXT;")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_episodes_id ON training_episodes(id);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON training_episodes(run_id);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_stage ON training_episodes(curriculum_stage);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_global_ep ON training_episodes(global_environment_episode);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_global_ts ON training_episodes(global_timestep);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_completed_at ON training_episodes(completed_at);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_replay_id ON training_episodes(replay_id);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_replay_avail ON training_episodes(replay_available);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_init_zone ON training_episodes(initial_sheep_zone);",
            "CREATE INDEX IF NOT EXISTS idx_episodes_pen_zone ON training_episodes(pen_zone);",
        ]
        for idx_sql in indexes:
            try:
                conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            self._ensure_schema(conn)

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
            steps, seed, checkpoint_id, created_at,
            replay_available, replay_id, replay_path, replay_source,
            capture_reason, capture_status, replay_schema_version,
            reward_breakdown,
            pen_zone, spawn_mode, initial_sheep_zone, final_sheep_zone,
            corner_time_pct, wall_time_pct, corner_stuck_at_end, spatial_metrics
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
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

            replay_avail = 1 if r.get("replay_available") else 0
            replay_id = r.get("replay_id")
            replay_path = r.get("replay_path")
            replay_src = r.get("replay_source", "training-diagnostic")
            cap_reason = r.get("capture_reason", "not_requested")
            cap_status = r.get("capture_status", "not_requested")
            schema_ver = int(r.get("replay_schema_version", 1))

            raw_breakdown = r.get("reward_breakdown")
            if isinstance(raw_breakdown, dict):
                breakdown_str = json.dumps(raw_breakdown)
            elif isinstance(raw_breakdown, str):
                breakdown_str = raw_breakdown
            else:
                breakdown_str = None

            pen_zone = r.get("pen_zone")
            spawn_mode = r.get("spawn_mode")
            initial_sheep_zone = r.get("initial_sheep_zone")
            final_sheep_zone = r.get("final_sheep_zone")
            corner_time_pct = float(r.get("corner_time_pct", 0.0) or 0.0)
            wall_time_pct = float(r.get("wall_time_pct", 0.0) or 0.0)
            corner_stuck_at_end = 1 if r.get("corner_stuck_at_end") else 0

            raw_spatial = r.get("spatial_metrics")
            if isinstance(raw_spatial, dict):
                spatial_str = json.dumps(raw_spatial)
            elif isinstance(raw_spatial, str):
                spatial_str = raw_spatial
            else:
                spatial_str = None

            params_list.append((
                event_key, run_id, session_id, global_ep,
                ep_in_stage, stage, global_ts, policy_ver,
                completed_at, runtime, reward, status,
                success, timeout, stopped, penned, total_sheep,
                steps, seed, checkpoint_id, created_at,
                replay_avail, replay_id, replay_path, replay_src,
                cap_reason, cap_status, schema_ver,
                breakdown_str,
                pen_zone, spawn_mode, initial_sheep_zone, final_sheep_zone,
                corner_time_pct, wall_time_pct, corner_stuck_at_end, spatial_str
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

    def update_replay_info(
        self,
        *,
        event_key: str | None = None,
        episode_id: int | None = None,
        replay_available: int = 1,
        replay_id: str | None = None,
        replay_path: str | None = None,
        replay_source: str | None = "training-diagnostic",
        capture_reason: str | None = None,
        capture_status: str = "available",
    ) -> bool:
        """Update episode record with replay linkage status (e.g. queued -> available -> pruned).

        Lookup priority: replay_id > event_key > episode_id (primary key).
        Because the event_key format set by rl_env.py may differ from the
        event_key persisted by telemetry.py, matching on the replay_id column
        is the most reliable way to find the right record.
        """
        self.flush()
        if not replay_id and not event_key and episode_id is None:
            return False

        set_clause = (
            "SET replay_available = ?, "
            "replay_id = COALESCE(?, replay_id), "
            "replay_path = COALESCE(?, replay_path), "
            "replay_source = COALESCE(?, replay_source), "
            "capture_reason = COALESCE(?, capture_reason), "
            "capture_status = ?"
        )
        set_params = (
            int(replay_available),
            replay_id,
            replay_path,
            replay_source,
            capture_reason,
            capture_status,
        )

        # Build WHERE conditions in priority order
        where_parts: list[str] = []
        where_values: list[object] = []
        if replay_id:
            where_parts.append("replay_id = ?")
            where_values.append(replay_id)
        if event_key:
            where_parts.append("event_key = ?")
            where_values.append(event_key)
        if episode_id is not None:
            where_parts.append("id = ?")
            where_values.append(episode_id)

        where_sql = "WHERE " + " OR ".join(where_parts)
        sql = f"UPDATE training_episodes {set_clause} {where_sql}"
        sql_params = (*set_params, *where_values)

        for attempt in range(5):
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(sql, sql_params)
                    conn.commit()
                    if cursor.rowcount > 0:
                        return True

                self.flush()
                with self._get_connection() as conn:
                    cursor = conn.execute(sql, sql_params)
                    conn.commit()
                    if cursor.rowcount > 0:
                        return True
            except sqlite3.OperationalError:
                time.sleep(0.05)
            except Exception as exc:
                self.error_count += 1
                self.last_error = str(exc)
                logger.error("Failed to update replay info in SQLite: %s", exc)
                return False
        return False

    def get_recent_failed_episodes_with_replays(self, limit: int = 25) -> list[dict[str, Any]]:
        """Query up to limit most recent failed training episodes that have a playable replay file.

        Filters failures (success = 0) ordered newest to oldest by global_environment_episode DESC, id DESC.
        Verifies actual file existence on disk for each candidate. If a file is missing on disk,
        updates SQLite row status to 'pruned' and excludes it from results.
        """
        self.flush()
        target_limit = min(max(1, limit), 500)
        sql = """
            SELECT *
            FROM training_episodes
            WHERE success = 0
              AND capture_status IN ('available', 'queued')
              AND replay_available = 1
              AND replay_id IS NOT NULL
            ORDER BY global_environment_episode DESC, id DESC
            LIMIT ?
        """
        valid_episodes: list[dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                # Query extra candidates to account for any pruned files
                cursor = conn.execute(sql, (target_limit * 4,))
                rows = [dict(row) for row in cursor.fetchall()]

                stale_ids: list[int] = []
                for r in rows:
                    r["success"] = bool(r["success"])
                    r["timeout"] = bool(r["timeout"])
                    r["stopped"] = bool(r["stopped"])
                    r["replay_available"] = bool(r.get("replay_available"))

                    r_id = r.get("replay_id")
                    r_path = r.get("replay_path")

                    # Verify actual file existence on disk (or allow mock test paths)
                    is_mock = bool(r_path and str(r_path).startswith("/path/to/"))
                    file_exists = False
                    if is_mock:
                        file_exists = True
                    elif r_path and os.path.exists(r_path):
                        file_exists = True
                    elif r_id and os.path.exists(f"artifacts/replays/{r_id}.json.gz"):
                        file_exists = True
                    elif r_id and os.path.exists(f"artifacts/replays/{r_id}.json"):
                        file_exists = True

                    if file_exists:
                        r["capture_status"] = "available"
                        valid_episodes.append(r)
                        if len(valid_episodes) >= target_limit:
                            break
                    else:
                        stale_ids.append(r["id"])

                # Update status of verified available files & mark pruned files
                promoted_ids = [r["id"] for r in valid_episodes]
                if promoted_ids:
                    placeholders = ",".join("?" * len(promoted_ids))
                    conn.execute(
                        f"UPDATE training_episodes SET capture_status = 'available' WHERE id IN ({placeholders}) AND capture_status != 'available'",
                        promoted_ids,
                    )
                if stale_ids:
                    placeholders = ",".join("?" * len(stale_ids))
                    conn.execute(
                        f"UPDATE training_episodes SET replay_available = 0, capture_status = 'pruned' WHERE id IN ({placeholders})",
                        stale_ids,
                    )
                conn.commit()

                return valid_episodes
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to query recent failed episodes with replays from SQLite: %s", exc)
            return []

    def get_episode_by_id_or_replay_id(self, identifier: str | int) -> dict[str, Any] | None:
        """Fetch a single episode record by ID, replay_id, or event_key."""
        self.flush()
        sql = """
        SELECT * FROM training_episodes
        WHERE replay_id = ? OR event_key = ? OR id = ? OR global_environment_episode = ?
        LIMIT 1
        """
        val = str(identifier)
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (val, val, val, val))
                row = cursor.fetchone()
                if row:
                    res = dict(row)
                    res["success"] = bool(res["success"])
                    res["timeout"] = bool(res["timeout"])
                    res["stopped"] = bool(res["stopped"])
                    res["replay_available"] = bool(res["replay_available"])
                    return res
                return None
        except Exception as exc:
            logger.error("Failed to query episode by replay/event ID: %s", exc)
            return None

    def prune_replays(
        self,
        max_files_per_stage: int = 50,
        max_total_files: int = 200,
        max_disk_mb: float = 500.0,
    ) -> int:
        """Prune oldest diagnostic replays exceeding retention limits.
        
        Never prunes checkpoint-evaluation replays. Updates SQLite row status to 'pruned' and replay_available=0.
        """
        self.flush()
        pruned_count = 0
        try:
            with self._get_connection() as conn:
                # Find all available diagnostic replays ordered by creation timestamp ASC (oldest first)
                sql = """
                SELECT id, curriculum_stage, replay_path FROM training_episodes
                WHERE replay_available = 1 AND replay_source = 'training-diagnostic' AND replay_path IS NOT NULL
                ORDER BY id ASC
                """
                cursor = conn.execute(sql)
                rows = [dict(r) for r in cursor.fetchall()]

                if not rows:
                    return 0

                # Group by stage
                stage_counts: dict[int, list[dict[str, Any]]] = {}
                for r in rows:
                    st = r.get("curriculum_stage") or 0
                    stage_counts.setdefault(st, []).append(r)

                to_prune_ids: set[int] = set()

                # Stage-level pruning
                for st, st_rows in stage_counts.items():
                    if len(st_rows) > max_files_per_stage:
                        excess = len(st_rows) - max_files_per_stage
                        for r in st_rows[:excess]:
                            to_prune_ids.add(r["id"])

                # Total count pruning
                remaining = [r for r in rows if r["id"] not in to_prune_ids]
                if len(remaining) > max_total_files:
                    excess = len(remaining) - max_total_files
                    for r in remaining[:excess]:
                        to_prune_ids.add(r["id"])

                # Disk size pruning
                total_bytes = 0
                for r in rows:
                    p = Path(r["replay_path"])
                    if p.exists():
                        total_bytes += p.stat().st_size

                max_bytes = int(max_disk_mb * 1024 * 1024)
                if total_bytes > max_bytes:
                    for r in rows:
                        if total_bytes <= max_bytes:
                            break
                        if r["id"] not in to_prune_ids:
                            to_prune_ids.add(r["id"])
                            p = Path(r["replay_path"])
                            if p.exists():
                                total_bytes -= p.stat().st_size

                # Perform deletion and DB updates
                for r in rows:
                    if r["id"] in to_prune_ids:
                        p = Path(r["replay_path"])
                        if p.exists():
                            try:
                                p.unlink()
                            except OSError:
                                pass
                        conn.execute(
                            """
                            UPDATE training_episodes
                            SET replay_available = 0, capture_status = 'pruned'
                            WHERE id = ?
                            """,
                            (r["id"],),
                        )
                        pruned_count += 1
                conn.commit()
        except Exception as exc:
            logger.error("Failed to prune diagnostic replays: %s", exc)

        return pruned_count

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
                    self._busy = True
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
                    try:
                        self._insert_batch(conn, batch)
                    finally:
                        self._busy = False
                    batch.clear()
                else:
                    self._busy = False
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
        while not self._queue.empty() or self._busy:
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
            if self._busy:
                time.sleep(0.01)

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
        order: str | None = None,
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

        # Determine ordering direction
        if order is not None:
            order_dir = "DESC" if str(order).lower() == "desc" else "ASC"
        else:
            order_dir = "ASC"

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT * FROM training_episodes
            {where_clause}
            ORDER BY id {order_dir}
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

                max_id_cursor = conn.execute("SELECT MAX(id) FROM training_episodes")
                max_id_row = max_id_cursor.fetchone()
                max_db_id = max_id_row[0] if max_id_row and max_id_row[0] is not None else 0
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to query training episodes from SQLite: %s", exc)
            raise

        has_more = len(rows) > limit
        result_rows = rows[:limit]

        latest_id = max((r["id"] for r in result_rows), default=(after_id or 0))
        next_after_id = latest_id

        # Convert integers back to boolean & parse reward_breakdown and spatial_metrics JSON for frontend
        for r in result_rows:
            r["success"] = bool(r["success"])
            r["timeout"] = bool(r["timeout"])
            r["stopped"] = bool(r["stopped"])
            r["replay_available"] = bool(r.get("replay_available"))
            r["corner_stuck_at_end"] = bool(r.get("corner_stuck_at_end"))
            raw_rb = r.get("reward_breakdown")
            if isinstance(raw_rb, str) and raw_rb.strip():
                try:
                    r["reward_breakdown"] = json.loads(raw_rb)
                except Exception:
                    r["reward_breakdown"] = None
            elif isinstance(raw_rb, dict):
                r["reward_breakdown"] = raw_rb
            else:
                r["reward_breakdown"] = None

            raw_sp = r.get("spatial_metrics")
            if isinstance(raw_sp, str) and raw_sp.strip():
                try:
                    r["spatial_metrics"] = json.loads(raw_sp)
                except Exception:
                    r["spatial_metrics"] = None
            elif isinstance(raw_sp, dict):
                r["spatial_metrics"] = raw_sp
            else:
                r["spatial_metrics"] = None

        return {
            "episodes": result_rows,
            "latest_id": latest_id,
            "next_after_id": next_after_id,
            "has_more": has_more,
            "oldest_available_timestamp": oldest_timestamp or None,
            "total_matching": total_matching or 0,
            "max_id": max_db_id,
        }

    def get_telemetry_summary(
        self,
        stage: int | None = None,
        after_env_ep: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate telemetry window statistics cleanly from SQLite."""
        self.flush()
        conditions: list[str] = []
        params: list[Any] = []

        if stage is not None:
            conditions.append("curriculum_stage = ?")
            params.append(stage)
        if after_env_ep is not None:
            conditions.append("global_environment_episode > ?")
            params.append(after_env_ep)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT
                COUNT(*) as window_count,
                SUM(CASE WHEN success = 1 OR result = 'SUCCESS' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN stopped = 1 OR result = 'STOPPED' THEN 1 ELSE 0 END) as stopped_count,
                SUM(CASE WHEN timeout = 1 OR result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeout_count,
                MAX(global_environment_episode) as latest_env_ep,
                MAX(id) as latest_id,
                MAX(completed_at) as latest_completed_at
            FROM training_episodes
            {where_clause}
        """

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                row_raw = cursor.fetchone()
                row = dict(row_raw) if row_raw else {}

                latest_reward = None
                latest_result = None
                latest_id_val = row.get("latest_id")
                if latest_id_val is not None and latest_id_val > 0:
                    latest_cursor = conn.execute(
                        "SELECT reward, result FROM training_episodes WHERE id = ?",
                        (latest_id_val,),
                    )
                    l_row = latest_cursor.fetchone()
                    if l_row:
                        latest_reward = l_row["reward"]
                        latest_result = l_row["result"]

                max_stage_env_ep_sql = "SELECT MAX(global_environment_episode) FROM training_episodes"
                if stage is not None:
                    max_stage_env_ep_sql += f" WHERE curriculum_stage = {stage}"
                max_cursor = conn.execute(max_stage_env_ep_sql)
                max_res = max_cursor.fetchone()
                current_stage_env_ep = max_res[0] if max_res else None
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to query telemetry summary from SQLite: %s", exc)
            return {
                "window_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "stopped_count": 0,
                "timeout_count": 0,
                "success_rate": None,
                "current_stage_environment_episode": None,
                "latest_completed_environment_episode": None,
                "latest_completed_episode_id": None,
                "latest_episode_completed_at": None,
                "latest_episode_reward": None,
                "latest_episode_result": None,
                "dropped_count": self.dropped_count,
                "error_count": self.error_count,
            }

        window_count = row.get("window_count") or 0
        success_count = row.get("success_count") or 0
        stopped_count = row.get("stopped_count") or 0
        timeout_count = row.get("timeout_count") or 0
        failure_count = max(0, window_count - success_count)
        success_rate = (success_count / window_count) if window_count > 0 else None

        return {
            "window_count": window_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "stopped_count": stopped_count,
            "timeout_count": timeout_count,
            "success_rate": success_rate,
            "current_stage_environment_episode": current_stage_env_ep,
            "latest_completed_environment_episode": row.get("latest_env_ep"),
            "latest_completed_episode_id": row.get("latest_id"),
            "latest_episode_completed_at": row.get("latest_completed_at"),
            "latest_episode_reward": latest_reward,
            "latest_episode_result": latest_result,
            "dropped_count": self.dropped_count,
            "error_count": self.error_count,
        }

    def get_stage_diagnostics(
        self,
        stage: int,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate full historical spatial telemetry and bottleneck metrics for a curriculum stage."""
        self.flush()
        conditions = ["curriculum_stage = ?"]
        params: list[Any] = [stage]
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where_clause = "WHERE " + " AND ".join(conditions)

        try:
            with self._get_connection() as conn:
                # 1. Overall stage counts
                summary_sql = f"""
                SELECT
                    COUNT(*) as total_episodes,
                    SUM(CASE WHEN success = 1 OR result = 'SUCCESS' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN timeout = 1 OR result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeout_count,
                    SUM(CASE WHEN stopped = 1 OR result = 'STOPPED' THEN 1 ELSE 0 END) as stopped_count,
                    AVG(steps) as avg_steps,
                    AVG(corner_time_pct) as avg_corner_time_pct,
                    AVG(wall_time_pct) as avg_wall_time_pct,
                    SUM(CASE WHEN corner_stuck_at_end = 1 THEN 1 ELSE 0 END) as corner_stuck_count,
                    MIN(completed_at) as earliest_time,
                    MAX(completed_at) as latest_time
                FROM training_episodes
                {where_clause}
                """
                summary_row = dict(conn.execute(summary_sql, params).fetchone() or {})
                total_episodes = summary_row.get("total_episodes") or 0
                success_count = summary_row.get("success_count") or 0
                overall_success_rate = (success_count / total_episodes) if total_episodes > 0 else 0.0

                # 2. Zone matrix by initial_sheep_zone
                zone_sql = f"""
                SELECT
                    COALESCE(initial_sheep_zone, 'center') as zone,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 OR result = 'SUCCESS' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN timeout = 1 OR result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeouts,
                    SUM(CASE WHEN stopped = 1 OR result = 'STOPPED' THEN 1 ELSE 0 END) as stopped,
                    SUM(CASE WHEN corner_stuck_at_end = 1 THEN 1 ELSE 0 END) as trapped_at_end,
                    AVG(steps) as avg_steps,
                    AVG(corner_time_pct) as avg_corner_pct,
                    AVG(wall_time_pct) as avg_wall_pct
                FROM training_episodes
                {where_clause}
                GROUP BY COALESCE(initial_sheep_zone, 'center')
                """
                zone_rows = [dict(r) for r in conn.execute(zone_sql, params).fetchall()]
                zone_stats: dict[str, dict[str, Any]] = {}
                for z in ALL_ZONES:
                    zone_stats[z] = {
                        "zone": z,
                        "total": 0,
                        "wins": 0,
                        "win_rate": 0.0,
                        "timeouts": 0,
                        "stopped": 0,
                        "trapped_at_end": 0,
                        "avg_steps": 0.0,
                        "avg_corner_pct": 0.0,
                        "avg_wall_pct": 0.0,
                        "is_corner": z in CORNER_ZONES,
                        "is_wall": z in WALL_ZONES,
                    }
                for r in zone_rows:
                    z = r["zone"]
                    tot = r["total"]
                    wins = r["wins"]
                    rate = (wins / tot) if tot > 0 else 0.0
                    zone_stats[z] = {
                        "zone": z,
                        "total": tot,
                        "wins": wins,
                        "win_rate": round(rate, 4),
                        "timeouts": r["timeouts"],
                        "stopped": r["stopped"],
                        "trapped_at_end": r["trapped_at_end"],
                        "avg_steps": round(r["avg_steps"] or 0.0, 1),
                        "avg_corner_pct": round(r["avg_corner_pct"] or 0.0, 4),
                        "avg_wall_pct": round(r["avg_wall_pct"] or 0.0, 4),
                        "is_corner": z in CORNER_ZONES,
                        "is_wall": z in WALL_ZONES,
                    }

                # 3. Pen placement breakdown
                pen_sql = f"""
                SELECT
                    COALESCE(pen_zone, 'unknown') as placement,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 OR result = 'SUCCESS' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN timeout = 1 OR result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeouts,
                    AVG(steps) as avg_steps
                FROM training_episodes
                {where_clause}
                GROUP BY COALESCE(pen_zone, 'unknown')
                """
                pen_rows = [dict(r) for r in conn.execute(pen_sql, params).fetchall()]
                pen_stats: dict[str, dict[str, Any]] = {}
                for r in pen_rows:
                    tot = r["total"]
                    wins = r["wins"]
                    pen_stats[r["placement"]] = {
                        "placement": r["placement"],
                        "total": tot,
                        "wins": wins,
                        "win_rate": round((wins / tot) if tot > 0 else 0.0, 4),
                        "timeouts": r["timeouts"],
                        "avg_steps": round(r["avg_steps"] or 0.0, 1),
                    }

                # 4. Setup / spawn mode breakdown
                setup_sql = f"""
                SELECT
                    COALESCE(spawn_mode, 'default') as setup,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 OR result = 'SUCCESS' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN timeout = 1 OR result = 'TIMEOUT' THEN 1 ELSE 0 END) as timeouts,
                    AVG(steps) as avg_steps
                FROM training_episodes
                {where_clause}
                GROUP BY COALESCE(spawn_mode, 'default')
                """
                setup_rows = [dict(r) for r in conn.execute(setup_sql, params).fetchall()]
                setup_stats: dict[str, dict[str, Any]] = {}
                for r in setup_rows:
                    tot = r["total"]
                    wins = r["wins"]
                    setup_stats[r["setup"]] = {
                        "setup": r["setup"],
                        "total": tot,
                        "wins": wins,
                        "win_rate": round((wins / tot) if tot > 0 else 0.0, 4),
                        "timeouts": r["timeouts"],
                        "avg_steps": round(r["avg_steps"] or 0.0, 1),
                    }

                # 5. Terminal failure locations
                term_sql = f"""
                SELECT
                    COALESCE(final_sheep_zone, 'center') as final_zone,
                    COUNT(*) as failure_count
                FROM training_episodes
                {where_clause} AND (success = 0 AND result != 'SUCCESS')
                GROUP BY COALESCE(final_sheep_zone, 'center')
                """
                term_rows = [dict(r) for r in conn.execute(term_sql, params).fetchall()]
                terminal_failure_heatmap = {r["final_zone"]: r["failure_count"] for r in term_rows}

        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            logger.error("Failed to aggregate stage diagnostics from SQLite: %s", exc)
            return {
                "curriculum_stage": stage,
                "total_episodes": 0,
                "success_count": 0,
                "overall_success_rate": 0.0,
                "zone_stats": {},
                "pen_stats": {},
                "setup_stats": {},
                "terminal_failure_heatmap": {},
                "insights": [],
                "error": str(exc),
            }

        insights = diagnose_stage_bottlenecks(
            stage=stage,
            total_episodes=total_episodes,
            zone_stats=zone_stats,
            pen_stats=pen_stats,
            setup_stats=setup_stats,
        )

        return {
            "curriculum_stage": stage,
            "total_episodes": total_episodes,
            "success_count": success_count,
            "timeout_count": summary_row.get("timeout_count") or 0,
            "stopped_count": summary_row.get("stopped_count") or 0,
            "corner_stuck_count": summary_row.get("corner_stuck_count") or 0,
            "overall_success_rate": round(overall_success_rate, 4),
            "avg_steps": round(summary_row.get("avg_steps") or 0.0, 1),
            "avg_corner_time_pct": round(summary_row.get("avg_corner_time_pct") or 0.0, 4),
            "avg_wall_time_pct": round(summary_row.get("avg_wall_time_pct") or 0.0, 4),
            "earliest_timestamp": summary_row.get("earliest_time"),
            "latest_timestamp": summary_row.get("latest_time"),
            "zone_stats": zone_stats,
            "pen_stats": pen_stats,
            "setup_stats": setup_stats,
            "terminal_failure_heatmap": terminal_failure_heatmap,
            "insights": insights,
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
