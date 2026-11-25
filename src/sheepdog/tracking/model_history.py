"""Durable SQLite model-lineage and per-stage training history store."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheepdog.tracking.registry import classify_assistance_mode
from sheepdog.training.stage_timing import (
    EFFECTIVE_CONFIG_FILENAME,
    RUNTIME_LEDGER_FILENAME,
    STAGE_TIMING_STATE_FILENAME,
    _run_id_start,
    _timesteps_from_event,
    config_snapshot,
    git_info,
    hardware_info,
    read_records,
    software_info,
)

MODEL_HISTORY_DB_FILENAME = "model-history.sqlite"
SCHEMA_VERSION = 1

_COMPLETED_STAGE_STATUSES = frozenset({"promoted", "mastered", "completed"})
_PHASE_KEYS = (
    "training_seconds",
    "evaluation_seconds",
    "checkpoint_save_seconds",
    "replay_serialization_seconds",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_passes (
    pass_id TEXT PRIMARY KEY,
    label TEXT,
    run_id TEXT,
    parent_pass_id TEXT REFERENCES model_passes(pass_id),
    parent_checkpoint_id TEXT,
    parent_checkpoint_path TEXT,
    parent_checkpoint_sha256 TEXT,
    resulting_checkpoint_id TEXT,
    resulting_checkpoint_path TEXT,
    resulting_checkpoint_sha256 TEXT,
    pass_type TEXT NOT NULL,
    stage_spec TEXT NOT NULL,
    planned_stages_json TEXT NOT NULL,
    completed_stages_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_training_seconds REAL NOT NULL DEFAULT 0.0,
    total_evaluation_seconds REAL NOT NULL DEFAULT 0.0,
    total_checkpoint_save_seconds REAL NOT NULL DEFAULT 0.0,
    total_replay_serialization_seconds REAL NOT NULL DEFAULT 0.0,
    total_active_seconds REAL NOT NULL DEFAULT 0.0,
    total_wall_seconds REAL,
    total_policy_timesteps INTEGER NOT NULL DEFAULT 0,
    assistance_mode TEXT,
    observation_mode TEXT,
    env_config_version TEXT,
    config_hash TEXT,
    git_revision TEXT,
    git_dirty INTEGER,
    hardware_json TEXT,
    software_json TEXT,
    source TEXT NOT NULL DEFAULT 'recorded',
    notes TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS stage_runs (
    stage_run_id TEXT PRIMARY KEY,
    pass_id TEXT NOT NULL REFERENCES model_passes(pass_id),
    run_id TEXT,
    stage INTEGER NOT NULL,
    sequence_order INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    start_checkpoint_id TEXT,
    start_checkpoint_sha256 TEXT,
    end_checkpoint_id TEXT,
    end_checkpoint_path TEXT,
    end_checkpoint_sha256 TEXT,
    start_policy_timesteps INTEGER,
    end_policy_timesteps INTEGER,
    policy_timesteps INTEGER,
    episodes INTEGER,
    training_seconds REAL,
    evaluation_seconds REAL,
    checkpoint_save_seconds REAL,
    replay_serialization_seconds REAL,
    active_seconds REAL,
    wall_seconds REAL,
    steps_per_second REAL,
    initial_success_rate REAL,
    final_success_rate REAL,
    best_success_rate REAL,
    final_avg_steps REAL,
    final_avg_reward REAL,
    eval_seed_count INTEGER,
    promotion_event_id TEXT,
    source TEXT NOT NULL DEFAULT 'recorded',
    baseline_runtime_json TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_passes_parent_pass_id ON model_passes(parent_pass_id);
CREATE INDEX IF NOT EXISTS idx_model_passes_run_id ON model_passes(run_id);
CREATE INDEX IF NOT EXISTS idx_model_passes_resulting_sha ON model_passes(resulting_checkpoint_sha256);
CREATE INDEX IF NOT EXISTS idx_model_passes_started_at ON model_passes(started_at);
CREATE INDEX IF NOT EXISTS idx_stage_runs_pass_seq ON stage_runs(pass_id, sequence_order);
CREATE INDEX IF NOT EXISTS idx_stage_runs_stage ON stage_runs(stage);
CREATE INDEX IF NOT EXISTS idx_stage_runs_run_id ON stage_runs(run_id);
"""


def parse_stage_spec(
    spec: str | Sequence[int], *, min_stage: int = 1, max_stage: int = 38
) -> list[int]:
    """Parse a stage specification string or integer sequence into an ordered list of stages."""
    if isinstance(spec, str):
        normalized = spec.replace("\u2013", "-")
        if not normalized.strip():
            raise ValueError("Stage list cannot be empty")
        raw_tokens = re.split(r"[,;]", normalized)
        if len(raw_tokens) > 1 and raw_tokens[-1].strip() == "":
            raw_tokens = raw_tokens[:-1]
        if not raw_tokens or all(not t.strip() for t in raw_tokens):
            raise ValueError("Stage list cannot be empty")

        stages: list[int] = []
        for raw in raw_tokens:
            token = raw.strip()
            if not token:
                raise ValueError(f"Empty stage token in specification: {spec!r}")
            range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", token)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start > end:
                    raise ValueError(f"Inverted stage range {token!r}: {start} > {end}")
                for value in range(start, end + 1):
                    if not (min_stage <= value <= max_stage):
                        raise ValueError(
                            f"Stage {value} in range {token!r} is outside [{min_stage}, {max_stage}]"
                        )
                    stages.append(value)
                continue
            single_match = re.fullmatch(r"-?\d+", token)
            if single_match:
                value = int(token)
                if not (min_stage <= value <= max_stage):
                    raise ValueError(f"Stage {value} is outside [{min_stage}, {max_stage}]")
                stages.append(value)
                continue
            raise ValueError(f"Invalid stage token {token!r}")
        return stages

    if not isinstance(spec, Sequence) or isinstance(spec, bytes | bytearray):
        raise ValueError(f"Unsupported stage specification type: {type(spec).__name__}")
    if len(spec) == 0:
        raise ValueError("Stage list cannot be empty")
    stages = []
    for item in spec:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Invalid stage value {item!r}: expected int")
        if not (min_stage <= item <= max_stage):
            raise ValueError(f"Stage {item} is outside [{min_stage}, {max_stage}]")
        stages.append(int(item))
    return stages


def format_stage_spec(stages: Sequence[int]) -> str:
    """Format an ordered sequence of stages, compressing ascending runs of length >= 2."""
    items = [int(s) for s in stages]
    if not items:
        return ""
    parts: list[str] = []
    run_start = items[0]
    run_end = items[0]
    for stage in items[1:]:
        if stage == run_end + 1:
            run_end = stage
        else:
            parts.append(f"{run_start}-{run_end}" if run_end > run_start else str(run_start))
            run_start = stage
            run_end = stage
    parts.append(f"{run_start}-{run_end}" if run_end > run_start else str(run_start))
    return ", ".join(parts)


def classify_pass_type(
    stages: Sequence[int], *, min_stage: int = 1, max_stage: int = 38
) -> str:
    """Classify a pass as 'full_curriculum', 'single_stage', or 'targeted_stages'."""
    stage_list = [int(s) for s in stages]
    if stage_list == list(range(min_stage, max_stage + 1)):
        return "full_curriculum"
    if len(stage_list) == 1:
        return "single_stage"
    return "targeted_stages"


def compute_file_sha256(path: str | Path | None) -> str | None:
    """Compute hex SHA-256 of a file in chunks, returning None if missing or unreadable."""
    if path is None:
        return None
    target = Path(path)
    hasher = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def _parse_iso_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _elapsed_seconds(started_at: str | None, ended_at: str | None) -> float | None:
    start = _parse_iso_time(started_at)
    end = _parse_iso_time(ended_at)
    if start is None or end is None or end < start:
        return None
    return round((end - start).total_seconds(), 3)


def _compute_steps_per_second(
    policy_timesteps: int | None,
    training_seconds: float | None,
    wall_seconds: float | None,
) -> float | None:
    if policy_timesteps is None:
        return None
    if training_seconds is not None and training_seconds > 0:
        return round(policy_timesteps / training_seconds, 3)
    if wall_seconds is not None and wall_seconds > 0:
        return round(policy_timesteps / wall_seconds, 3)
    return None


def _pct_delta(new: float | int | None, ref: float | int | None) -> float | None:
    if new is None or ref is None or ref == 0:
        return None
    return round((float(new) - float(ref)) / abs(float(ref)) * 100.0, 2)


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _from_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _resolve_artifact_path(output_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path or not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    norm = raw_path.replace("\\", "/")
    under_root = output_root / norm
    if under_root.is_file():
        return under_root
    if norm.startswith(output_root.name + "/"):
        under_parent = output_root.parent / norm
        if under_parent.is_file():
            return under_parent
    if candidate.is_file():
        return candidate
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _decode_stage_run_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data.pop("baseline_runtime_json", None)
    metadata_raw = data.pop("metadata_json", None)
    data["metadata"] = _from_json(metadata_raw)
    return data


def _decode_pass_row(row: sqlite3.Row, stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    data = dict(row)
    planned_raw = data.pop("planned_stages_json", "[]")
    hardware_raw = data.pop("hardware_json", None)
    software_raw = data.pop("software_json", None)
    metadata_raw = data.pop("metadata_json", None)
    dirty_raw = data.get("git_dirty")

    data["planned_stages"] = _from_json(planned_raw) or []
    data["hardware"] = _from_json(hardware_raw)
    data["software"] = _from_json(software_raw)
    data["metadata"] = _from_json(metadata_raw)
    data["git_dirty"] = None if dirty_raw is None else bool(dirty_raw)
    data["stage_runs"] = stage_runs
    return data


class ModelHistoryStore:
    """Thread-safe SQLite store for model-pass lineage and per-stage training runs."""

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        """Initialize the store, creating the schema if writable."""
        self.db_path = Path(db_path)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        if not self.read_only:
            with self._lock:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    conn.executescript(_SCHEMA_SQL)
                    row = conn.execute("PRAGMA user_version;").fetchone()
                    current_ver = int(row[0]) if row else 0
                    if current_ver < SCHEMA_VERSION:
                        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
                    conn.commit()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError(f"ModelHistoryStore is read-only: {self.db_path}")

    def _recompute_pass_rollups(self, conn: sqlite3.Connection, pass_id: str) -> None:
        rows = conn.execute(
            """
            SELECT status, training_seconds, evaluation_seconds, checkpoint_save_seconds,
                   replay_serialization_seconds, active_seconds, policy_timesteps
            FROM stage_runs
            WHERE pass_id = ?
            """,
            (pass_id,),
        ).fetchall()

        total_train = 0.0
        total_eval = 0.0
        total_ckpt = 0.0
        total_replay = 0.0
        total_active = 0.0
        total_steps = 0
        completed_count = 0

        for row in rows:
            t_s = float(row["training_seconds"] or 0.0)
            e_s = float(row["evaluation_seconds"] or 0.0)
            c_s = float(row["checkpoint_save_seconds"] or 0.0)
            r_s = float(row["replay_serialization_seconds"] or 0.0)
            a_s = (
                float(row["active_seconds"])
                if row["active_seconds"] is not None
                else (t_s + e_s + c_s + r_s)
            )
            total_train += t_s
            total_eval += e_s
            total_ckpt += c_s
            total_replay += r_s
            total_active += a_s
            total_steps += int(row["policy_timesteps"] or 0)
            if row["status"] in _COMPLETED_STAGE_STATUSES:
                completed_count += 1

        conn.execute(
            """
            UPDATE model_passes
            SET completed_stages_count = ?,
                total_training_seconds = ?,
                total_evaluation_seconds = ?,
                total_checkpoint_save_seconds = ?,
                total_replay_serialization_seconds = ?,
                total_active_seconds = ?,
                total_policy_timesteps = ?
            WHERE pass_id = ?
            """,
            (
                completed_count,
                total_train,
                total_eval,
                total_ckpt,
                total_replay,
                total_active,
                total_steps,
                pass_id,
            ),
        )

    def _fetch_stage_runs_for_pass(
        self, conn: sqlite3.Connection, pass_id: str
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM stage_runs WHERE pass_id = ? ORDER BY sequence_order ASC, rowid ASC",
            (pass_id,),
        ).fetchall()
        return [_decode_stage_run_row(r) for r in rows]

    def find_parent_pass_id(
        self,
        parent_checkpoint_sha256: str | None,
        *,
        parent_checkpoint_path: str | None = None,
        run_id: str | None = None,
    ) -> str | None:
        """Resolve the most recent ancestor pass matching checkpoint SHA, path, or run_id."""
        with self._lock:
            if self.read_only and not self.db_path.exists():
                return None
            with self._connect() as conn:
                if parent_checkpoint_sha256:
                    row = conn.execute(
                        """
                        SELECT pass_id FROM (
                            SELECT pass_id, COALESCE(ended_at, started_at) AS ts, rowid AS rid
                            FROM model_passes
                            WHERE resulting_checkpoint_sha256 = ?
                            UNION ALL
                            SELECT mp.pass_id,
                                   COALESCE(sr.ended_at, sr.started_at, mp.started_at) AS ts,
                                   mp.rowid AS rid
                            FROM stage_runs sr
                            JOIN model_passes mp ON mp.pass_id = sr.pass_id
                            WHERE sr.end_checkpoint_sha256 = ?
                        )
                        ORDER BY ts DESC, rid DESC
                        LIMIT 1
                        """,
                        (parent_checkpoint_sha256, parent_checkpoint_sha256),
                    ).fetchone()
                    if row:
                        return str(row["pass_id"])

                if parent_checkpoint_path:
                    norm_path = str(parent_checkpoint_path).replace("\\", "/")
                    row = conn.execute(
                        """
                        SELECT pass_id FROM model_passes
                        WHERE resulting_checkpoint_path = ?
                           OR REPLACE(resulting_checkpoint_path, '\\', '/') = ?
                        ORDER BY COALESCE(ended_at, started_at) DESC, rowid DESC
                        LIMIT 1
                        """,
                        (str(parent_checkpoint_path), norm_path),
                    ).fetchone()
                    if row:
                        return str(row["pass_id"])

                    row = conn.execute(
                        """
                        SELECT mp.pass_id
                        FROM stage_runs sr
                        JOIN model_passes mp ON mp.pass_id = sr.pass_id
                        WHERE sr.end_checkpoint_path = ?
                           OR REPLACE(sr.end_checkpoint_path, '\\', '/') = ?
                        ORDER BY COALESCE(sr.ended_at, sr.started_at, mp.started_at) DESC, mp.rowid DESC
                        LIMIT 1
                        """,
                        (str(parent_checkpoint_path), norm_path),
                    ).fetchone()
                    if row:
                        return str(row["pass_id"])

                if run_id:
                    row = conn.execute(
                        """
                        SELECT pass_id FROM model_passes
                        WHERE run_id = ?
                        ORDER BY started_at DESC, rowid DESC
                        LIMIT 1
                        """,
                        (run_id,),
                    ).fetchone()
                    if row:
                        return str(row["pass_id"])

                return None

    def start_pass(
        self,
        *,
        pass_id: str | None = None,
        label: str | None = None,
        run_id: str | None = None,
        parent_pass_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        parent_checkpoint_path: str | Path | None = None,
        parent_checkpoint_sha256: str | None = None,
        stage_spec: str | Sequence[int] | None = None,
        planned_stages: Sequence[int] | None = None,
        pass_type: str | None = None,
        status: str = "running",
        started_at: str | None = None,
        assistance_mode: str | None = None,
        observation_mode: str | None = None,
        env_config_version: str | None = None,
        config_hash: str | None = None,
        git_revision: str | None = None,
        git_dirty: bool | int | None = None,
        hardware: dict[str, Any] | None = None,
        software: dict[str, Any] | None = None,
        output_root: Path | str | None = None,
        source: str = "recorded",
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new model pass row and auto-resolve its parent pass if applicable."""
        self._require_writable()
        if planned_stages is not None:
            stages = parse_stage_spec(planned_stages)
        elif stage_spec is not None:
            stages = parse_stage_spec(stage_spec)
        else:
            stages = list(range(1, 39))

        formatted_spec = format_stage_spec(stages)
        resolved_pass_type = pass_type or classify_pass_type(stages)
        resolved_pass_id = pass_id or f"pass_{uuid.uuid4().hex[:16]}"
        resolved_started_at = started_at or datetime.now(UTC).isoformat()

        parent_path_str = str(parent_checkpoint_path) if parent_checkpoint_path is not None else None
        resolved_parent_sha = parent_checkpoint_sha256
        if resolved_parent_sha is None and parent_path_str is not None:
            resolved_parent_sha = compute_file_sha256(parent_path_str)

        if output_root is not None:
            cfg = config_snapshot(Path(output_root))
            if config_hash is None:
                config_hash = cfg.get("hash")
            if observation_mode is None:
                observation_mode = cfg.get("assistance_mode")

        if assistance_mode is None or (
            observation_mode == "guided" and assistance_mode == "autonomous"
        ):
            resolved_assistance = classify_assistance_mode(observation_mode)
        else:
            resolved_assistance = assistance_mode

        if git_revision is None or git_dirty is None:
            git = git_info()
            if git_revision is None:
                git_revision = git.get("revision")
            if git_dirty is None:
                git_dirty = git.get("dirty")
        dirty_int = None if git_dirty is None else (1 if git_dirty else 0)

        resolved_hardware = hardware if hardware is not None else hardware_info()
        resolved_software = software if software is not None else software_info()

        with self._lock:
            resolved_parent_pass_id = parent_pass_id
            if resolved_parent_pass_id is None and (
                resolved_parent_sha is not None or parent_path_str is not None or run_id is not None
            ):
                resolved_parent_pass_id = self.find_parent_pass_id(
                    resolved_parent_sha,
                    parent_checkpoint_path=parent_path_str,
                    run_id=run_id,
                )

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO model_passes (
                        pass_id, label, run_id, parent_pass_id,
                        parent_checkpoint_id, parent_checkpoint_path, parent_checkpoint_sha256,
                        pass_type, stage_spec, planned_stages_json,
                        status, started_at,
                        assistance_mode, observation_mode, env_config_version, config_hash,
                        git_revision, git_dirty, hardware_json, software_json,
                        source, notes, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_pass_id,
                        label,
                        run_id,
                        resolved_parent_pass_id,
                        parent_checkpoint_id,
                        parent_path_str,
                        resolved_parent_sha,
                        resolved_pass_type,
                        formatted_spec,
                        json.dumps(stages),
                        status,
                        resolved_started_at,
                        resolved_assistance,
                        observation_mode,
                        env_config_version,
                        config_hash,
                        git_revision,
                        dirty_int,
                        _to_json(resolved_hardware),
                        _to_json(resolved_software),
                        source,
                        notes,
                        _to_json(metadata),
                    ),
                )
                conn.commit()
        return resolved_pass_id

    def start_stage_run(
        self,
        pass_id: str,
        *,
        run_id: str | None = None,
        stage: int,
        sequence_order: int,
        started_at: str | None = None,
        start_checkpoint_id: str | None = None,
        start_checkpoint_sha256: str | None = None,
        start_policy_timesteps: int | None = None,
        baseline_runtime: dict[str, Any] | None = None,
        source: str = "recorded",
        stage_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a stage run within a model pass, storing any runtime baseline."""
        self._require_writable()
        resolved_id = stage_run_id or f"sr_{uuid.uuid4().hex[:16]}"
        resolved_started_at = started_at or datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_runs (
                    stage_run_id, pass_id, run_id, stage, sequence_order,
                    status, started_at, start_checkpoint_id, start_checkpoint_sha256,
                    start_policy_timesteps, source, baseline_runtime_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    pass_id,
                    run_id,
                    int(stage),
                    int(sequence_order),
                    resolved_started_at,
                    start_checkpoint_id,
                    start_checkpoint_sha256,
                    start_policy_timesteps,
                    source,
                    _to_json(baseline_runtime),
                    _to_json(metadata),
                ),
            )
            conn.commit()
        return resolved_id

    def _apply_stage_run_update(
        self,
        conn: sqlite3.Connection,
        stage_run_id: str,
        *,
        runtime_totals: dict[str, Any] | None = None,
        current_policy_timesteps: int | None = None,
        episodes: int | None = None,
        success_rate: float | None = None,
        avg_steps: float | None = None,
        avg_reward: float | None = None,
        eval_seed_count: int | None = None,
        checkpoint_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        status: str | None = None,
        ended_at: str | None = None,
        wall_seconds: float | None = None,
        promotion_event_id: str | None = None,
    ) -> None:
        row = conn.execute(
            "SELECT * FROM stage_runs WHERE stage_run_id = ?",
            (stage_run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Stage run not found: {stage_run_id}")

        pass_id = str(row["pass_id"])
        baseline = _from_json(row["baseline_runtime_json"])
        if not isinstance(baseline, dict):
            baseline = {}

        train_sec = _coerce_float(row["training_seconds"])
        eval_sec = _coerce_float(row["evaluation_seconds"])
        ckpt_sec = _coerce_float(row["checkpoint_save_seconds"])
        replay_sec = _coerce_float(row["replay_serialization_seconds"])
        active_sec = _coerce_float(row["active_seconds"])

        if runtime_totals is not None:
            train_sec = max(
                0.0,
                float(runtime_totals.get("training_seconds") or 0.0)
                - float(baseline.get("training_seconds") or 0.0),
            )
            eval_sec = max(
                0.0,
                float(runtime_totals.get("evaluation_seconds") or 0.0)
                - float(baseline.get("evaluation_seconds") or 0.0),
            )
            ckpt_sec = max(
                0.0,
                float(runtime_totals.get("checkpoint_save_seconds") or 0.0)
                - float(baseline.get("checkpoint_save_seconds") or 0.0),
            )
            replay_sec = max(
                0.0,
                float(runtime_totals.get("replay_serialization_seconds") or 0.0)
                - float(baseline.get("replay_serialization_seconds") or 0.0),
            )
            active_sec = train_sec + eval_sec + ckpt_sec + replay_sec

        initial_sr = _coerce_float(row["initial_success_rate"])
        final_sr = _coerce_float(row["final_success_rate"])
        best_sr = _coerce_float(row["best_success_rate"])
        if success_rate is not None:
            sr_val = float(success_rate)
            if initial_sr is None:
                initial_sr = sr_val
            best_sr = sr_val if best_sr is None else max(best_sr, sr_val)
            final_sr = sr_val

        final_avg_steps = float(avg_steps) if avg_steps is not None else _coerce_float(row["final_avg_steps"])
        final_avg_reward = float(avg_reward) if avg_reward is not None else _coerce_float(row["final_avg_reward"])
        new_eval_seed_count = (
            int(eval_seed_count) if eval_seed_count is not None else _coerce_int(row["eval_seed_count"])
        )
        new_episodes = int(episodes) if episodes is not None else _coerce_int(row["episodes"])

        new_ckpt_id = checkpoint_id if checkpoint_id is not None else row["end_checkpoint_id"]
        new_ckpt_path = (
            str(checkpoint_path) if checkpoint_path is not None else row["end_checkpoint_path"]
        )
        new_ckpt_sha = checkpoint_sha256 if checkpoint_sha256 is not None else row["end_checkpoint_sha256"]
        if checkpoint_sha256 is None and checkpoint_path is not None:
            computed = compute_file_sha256(checkpoint_path)
            if computed is not None:
                new_ckpt_sha = computed

        start_ts = _coerce_int(row["start_policy_timesteps"])
        end_ts = _coerce_int(row["end_policy_timesteps"])
        policy_ts = _coerce_int(row["policy_timesteps"])
        if current_policy_timesteps is not None:
            end_ts = int(current_policy_timesteps)
            if start_ts is not None:
                policy_ts = max(0, end_ts - start_ts)

        new_status = status if status is not None else str(row["status"])
        new_ended_at = ended_at if ended_at is not None else row["ended_at"]
        new_wall = wall_seconds if wall_seconds is not None else _coerce_float(row["wall_seconds"])
        if new_wall is None and new_ended_at is not None:
            new_wall = _elapsed_seconds(row["started_at"], new_ended_at)

        new_promo_id = (
            promotion_event_id if promotion_event_id is not None else row["promotion_event_id"]
        )

        sps = _compute_steps_per_second(policy_ts, train_sec, new_wall)

        conn.execute(
            """
            UPDATE stage_runs
            SET status = ?,
                ended_at = ?,
                end_checkpoint_id = ?,
                end_checkpoint_path = ?,
                end_checkpoint_sha256 = ?,
                end_policy_timesteps = ?,
                policy_timesteps = ?,
                episodes = ?,
                training_seconds = ?,
                evaluation_seconds = ?,
                checkpoint_save_seconds = ?,
                replay_serialization_seconds = ?,
                active_seconds = ?,
                wall_seconds = ?,
                steps_per_second = ?,
                initial_success_rate = ?,
                final_success_rate = ?,
                best_success_rate = ?,
                final_avg_steps = ?,
                final_avg_reward = ?,
                eval_seed_count = ?,
                promotion_event_id = ?
            WHERE stage_run_id = ?
            """,
            (
                new_status,
                new_ended_at,
                new_ckpt_id,
                new_ckpt_path,
                new_ckpt_sha,
                end_ts,
                policy_ts,
                new_episodes,
                train_sec,
                eval_sec,
                ckpt_sec,
                replay_sec,
                active_sec,
                new_wall,
                sps,
                initial_sr,
                final_sr,
                best_sr,
                final_avg_steps,
                final_avg_reward,
                new_eval_seed_count,
                new_promo_id,
                stage_run_id,
            ),
        )
        self._recompute_pass_rollups(conn, pass_id)

    def update_stage_run(
        self,
        stage_run_id: str,
        *,
        runtime_totals: dict[str, Any] | None = None,
        current_policy_timesteps: int | None = None,
        episodes: int | None = None,
        success_rate: float | None = None,
        avg_steps: float | None = None,
        avg_reward: float | None = None,
        eval_seed_count: int | None = None,
        checkpoint_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        status: str | None = None,
    ) -> None:
        """Update metrics, runtime deltas, and parent pass rollups for an active stage run."""
        self._require_writable()
        with self._lock, self._connect() as conn:
            self._apply_stage_run_update(
                conn,
                stage_run_id,
                runtime_totals=runtime_totals,
                current_policy_timesteps=current_policy_timesteps,
                episodes=episodes,
                success_rate=success_rate,
                avg_steps=avg_steps,
                avg_reward=avg_reward,
                eval_seed_count=eval_seed_count,
                checkpoint_id=checkpoint_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                status=status,
            )
            conn.commit()

    def finish_stage_run(
        self,
        stage_run_id: str,
        *,
        status: str,
        ended_at: str | None = None,
        promotion_event_id: str | None = None,
        runtime_totals: dict[str, Any] | None = None,
        current_policy_timesteps: int | None = None,
        episodes: int | None = None,
        success_rate: float | None = None,
        avg_steps: float | None = None,
        avg_reward: float | None = None,
        eval_seed_count: int | None = None,
        checkpoint_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        end_checkpoint_id: str | None = None,
        end_checkpoint_path: str | Path | None = None,
        end_checkpoint_sha256: str | None = None,
        wall_seconds: float | None = None,
    ) -> None:
        """Finalize a stage run, compute wall time, and update parent pass rollups."""
        self._require_writable()
        resolved_ended_at = ended_at or datetime.now(UTC).isoformat()
        eff_ckpt_id = end_checkpoint_id if end_checkpoint_id is not None else checkpoint_id
        eff_ckpt_path = end_checkpoint_path if end_checkpoint_path is not None else checkpoint_path
        eff_ckpt_sha = end_checkpoint_sha256 if end_checkpoint_sha256 is not None else checkpoint_sha256

        with self._lock, self._connect() as conn:
            self._apply_stage_run_update(
                conn,
                stage_run_id,
                runtime_totals=runtime_totals,
                current_policy_timesteps=current_policy_timesteps,
                episodes=episodes,
                success_rate=success_rate,
                avg_steps=avg_steps,
                avg_reward=avg_reward,
                eval_seed_count=eval_seed_count,
                checkpoint_id=eff_ckpt_id,
                checkpoint_path=eff_ckpt_path,
                checkpoint_sha256=eff_ckpt_sha,
                status=status,
                ended_at=resolved_ended_at,
                wall_seconds=wall_seconds,
                promotion_event_id=promotion_event_id,
            )
            conn.commit()

    def finish_pass(
        self,
        pass_id: str,
        *,
        status: str,
        ended_at: str | None = None,
        resulting_checkpoint_id: str | None = None,
        resulting_checkpoint_path: str | Path | None = None,
        resulting_checkpoint_sha256: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Finalize a model pass with status, wall time, resulting checkpoint, and rollups."""
        self._require_writable()
        resolved_ended_at = ended_at or datetime.now(UTC).isoformat()
        res_path_str = (
            str(resulting_checkpoint_path) if resulting_checkpoint_path is not None else None
        )
        res_sha = resulting_checkpoint_sha256
        if res_sha is None and res_path_str is not None:
            res_sha = compute_file_sha256(res_path_str)

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_passes WHERE pass_id = ?",
                (pass_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Model pass not found: {pass_id}")

            wall = _elapsed_seconds(row["started_at"], resolved_ended_at)
            eff_ckpt_id = (
                resulting_checkpoint_id
                if resulting_checkpoint_id is not None
                else row["resulting_checkpoint_id"]
            )
            eff_ckpt_path = (
                res_path_str if res_path_str is not None else row["resulting_checkpoint_path"]
            )
            eff_ckpt_sha = (
                res_sha if res_sha is not None else row["resulting_checkpoint_sha256"]
            )
            eff_notes = notes if notes is not None else row["notes"]

            conn.execute(
                """
                UPDATE model_passes
                SET status = ?,
                    ended_at = ?,
                    total_wall_seconds = ?,
                    resulting_checkpoint_id = ?,
                    resulting_checkpoint_path = ?,
                    resulting_checkpoint_sha256 = ?,
                    notes = ?
                WHERE pass_id = ?
                """,
                (
                    status,
                    resolved_ended_at,
                    wall,
                    eff_ckpt_id,
                    eff_ckpt_path,
                    eff_ckpt_sha,
                    eff_notes,
                    pass_id,
                ),
            )
            self._recompute_pass_rollups(conn, pass_id)
            conn.commit()

    def get_stage_run(self, stage_run_id: str) -> dict[str, Any] | None:
        """Return a single stage run dictionary (excluding internal baseline), or None."""
        with self._lock:
            if self.read_only and not self.db_path.exists():
                return None
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM stage_runs WHERE stage_run_id = ?",
                    (stage_run_id,),
                ).fetchone()
                if row is None:
                    return None
                return _decode_stage_run_row(row)

    def get_pass(self, pass_id: str) -> dict[str, Any] | None:
        """Return a model pass dictionary with ordered stage_runs, or None."""
        with self._lock:
            if self.read_only and not self.db_path.exists():
                return None
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM model_passes WHERE pass_id = ?",
                    (pass_id,),
                ).fetchone()
                if row is None:
                    return None
                stage_runs = self._fetch_stage_runs_for_pass(conn, pass_id)
                return _decode_pass_row(row, stage_runs)

    def list_passes(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return model passes ordered newest-first, each with its ordered stage_runs."""
        with self._lock:
            if self.read_only and not self.db_path.exists():
                return []
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM model_passes ORDER BY started_at DESC, rowid DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
                result: list[dict[str, Any]] = []
                for row in rows:
                    stage_runs = self._fetch_stage_runs_for_pass(conn, str(row["pass_id"]))
                    result.append(_decode_pass_row(row, stage_runs))
                return result

    def get_lineage(self, pass_id: str) -> dict[str, Any] | None:
        """Walk parent_pass_id chain from root to pass_id and compute cumulative/stage metrics."""
        with self._lock:
            if self.read_only and not self.db_path.exists():
                return None
            with self._connect() as conn:
                chain_reversed: list[dict[str, Any]] = []
                visited: set[str] = set()
                current_id: str | None = pass_id
                while current_id and current_id not in visited:
                    visited.add(current_id)
                    row = conn.execute(
                        "SELECT * FROM model_passes WHERE pass_id = ?",
                        (current_id,),
                    ).fetchone()
                    if row is None:
                        break
                    stage_runs = self._fetch_stage_runs_for_pass(conn, str(row["pass_id"]))
                    chain_reversed.append(_decode_pass_row(row, stage_runs))
                    current_id = row["parent_pass_id"]

                if not chain_reversed:
                    return None

                lineage = list(reversed(chain_reversed))

        cumulative = {
            "pass_count": len(lineage),
            "stage_runs_count": sum(len(p["stage_runs"]) for p in lineage),
            "completed_stage_runs_count": sum(
                1
                for p in lineage
                for sr in p["stage_runs"]
                if sr.get("status") in _COMPLETED_STAGE_STATUSES
            ),
            "total_training_seconds": sum(float(p.get("total_training_seconds") or 0.0) for p in lineage),
            "total_evaluation_seconds": sum(float(p.get("total_evaluation_seconds") or 0.0) for p in lineage),
            "total_checkpoint_save_seconds": sum(
                float(p.get("total_checkpoint_save_seconds") or 0.0) for p in lineage
            ),
            "total_replay_serialization_seconds": sum(
                float(p.get("total_replay_serialization_seconds") or 0.0) for p in lineage
            ),
            "total_active_seconds": sum(float(p.get("total_active_seconds") or 0.0) for p in lineage),
            "total_wall_seconds": sum(float(p.get("total_wall_seconds") or 0.0) for p in lineage),
            "total_policy_timesteps": sum(int(p.get("total_policy_timesteps") or 0) for p in lineage),
        }

        by_stage: dict[int, list[dict[str, Any]]] = {}
        for pass_index, pass_item in enumerate(lineage):
            for sr in pass_item["stage_runs"]:
                st = int(sr["stage"])
                by_stage.setdefault(st, []).append(
                    {
                        "pass_id": pass_item["pass_id"],
                        "pass_index": pass_index,
                        "label": pass_item["label"],
                        "sequence_order": sr["sequence_order"],
                        "status": sr["status"],
                        "active_seconds": sr["active_seconds"],
                        "training_seconds": sr["training_seconds"],
                        "evaluation_seconds": sr["evaluation_seconds"],
                        "wall_seconds": sr["wall_seconds"],
                        "policy_timesteps": sr["policy_timesteps"],
                        "steps_per_second": sr["steps_per_second"],
                        "initial_success_rate": sr["initial_success_rate"],
                        "final_success_rate": sr["final_success_rate"],
                        "best_success_rate": sr["best_success_rate"],
                        "final_avg_steps": sr["final_avg_steps"],
                    }
                )

        stage_comparison: list[dict[str, Any]] = []
        for stage_num in sorted(by_stage):
            runs = by_stage[stage_num]
            if len(runs) >= 2:
                latest = runs[-1]
                first = runs[0]
                prev = runs[-2]
                delta_vs_first = {
                    "active_seconds_pct": _pct_delta(
                        latest["active_seconds"], first["active_seconds"]
                    ),
                    "training_seconds_pct": _pct_delta(
                        latest["training_seconds"], first["training_seconds"]
                    ),
                    "policy_timesteps_pct": _pct_delta(
                        latest["policy_timesteps"], first["policy_timesteps"]
                    ),
                }
                delta_vs_previous = {
                    "active_seconds_pct": _pct_delta(
                        latest["active_seconds"], prev["active_seconds"]
                    ),
                    "training_seconds_pct": _pct_delta(
                        latest["training_seconds"], prev["training_seconds"]
                    ),
                    "policy_timesteps_pct": _pct_delta(
                        latest["policy_timesteps"], prev["policy_timesteps"]
                    ),
                }
            else:
                delta_vs_first = {
                    "active_seconds_pct": None,
                    "training_seconds_pct": None,
                    "policy_timesteps_pct": None,
                }
                delta_vs_previous = {
                    "active_seconds_pct": None,
                    "training_seconds_pct": None,
                    "policy_timesteps_pct": None,
                }
            stage_comparison.append(
                {
                    "stage": stage_num,
                    "runs": runs,
                    "delta_vs_first": delta_vs_first,
                    "delta_vs_previous": delta_vs_previous,
                }
            )

        return {
            "target_pass_id": pass_id,
            "lineage": lineage,
            "cumulative": cumulative,
            "stage_comparison": stage_comparison,
        }

    def build_api_payload(
        self,
        *,
        pass_id: str | None = None,
        stage: int | None = None,
        limit: int = 50,
        active_pass_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the API response payload with passes, selected lineage, and stage matrix."""
        passes = self.list_passes(limit=limit)
        selected_id = pass_id or active_pass_id or (passes[0]["pass_id"] if passes else None)
        selected_lineage = self.get_lineage(selected_id) if selected_id else None

        by_stage: dict[int, list[dict[str, Any]]] = {}
        for pass_item in passes:
            for sr in pass_item["stage_runs"]:
                st = int(sr["stage"])
                if stage is not None and st != stage:
                    continue
                by_stage.setdefault(st, []).append(
                    {
                        "pass_id": pass_item["pass_id"],
                        "label": pass_item["label"],
                        "run_id": pass_item["run_id"],
                        "started_at": sr["started_at"],
                        "ended_at": sr["ended_at"],
                        "sequence_order": sr["sequence_order"],
                        "status": sr["status"],
                        "active_seconds": sr["active_seconds"],
                        "training_seconds": sr["training_seconds"],
                        "evaluation_seconds": sr["evaluation_seconds"],
                        "checkpoint_save_seconds": sr["checkpoint_save_seconds"],
                        "replay_serialization_seconds": sr["replay_serialization_seconds"],
                        "wall_seconds": sr["wall_seconds"],
                        "policy_timesteps": sr["policy_timesteps"],
                        "steps_per_second": sr["steps_per_second"],
                        "initial_success_rate": sr["initial_success_rate"],
                        "final_success_rate": sr["final_success_rate"],
                        "best_success_rate": sr["best_success_rate"],
                        "final_avg_steps": sr["final_avg_steps"],
                        "source": sr["source"],
                    }
                )

        stage_matrix = [
            {"stage": st, "run_count": len(by_stage[st]), "runs": by_stage[st]}
            for st in sorted(by_stage)
        ]

        return {
            "schema_version": SCHEMA_VERSION,
            "passes": passes,
            "active_pass_id": active_pass_id,
            "selected_lineage": selected_lineage,
            "stage_matrix": stage_matrix,
        }

    def backfill_from_artifacts(self, output_root: Path | str) -> dict[str, Any]:
        """Idempotently backfill historical runs from artifact files into SQLite."""
        self._require_writable()
        root = Path(output_root)

        runtime_ledger = _read_json_file(root / RUNTIME_LEDGER_FILENAME)
        sessions: list[dict[str, Any]] = (
            [s for s in runtime_ledger.get("sessions", []) if isinstance(s, dict)]
            if isinstance(runtime_ledger, dict) and isinstance(runtime_ledger.get("sessions"), list)
            else []
        )

        timing_records = read_records(root)
        if not timing_records:
            promo_raw = _read_json_file(root / "promotion-history.json")
            if isinstance(promo_raw, list):
                timing_records = _records_from_promotion_history(promo_raw)

        timing_state = _read_json_file(root / STAGE_TIMING_STATE_FILENAME)
        if not isinstance(timing_state, dict):
            timing_state = {}

        training_state = _read_json_file(root / "training-state.json")
        if not isinstance(training_state, dict):
            training_state = {}

        run_state = _read_json_file(root / "run-state.json")
        if not isinstance(run_state, dict):
            run_state = {}

        stage_history = _read_json_file(root / "stage-history.json")
        if not isinstance(stage_history, dict):
            stage_history = {}

        cfg = (
            config_snapshot(root)
            if (root / EFFECTIVE_CONFIG_FILENAME).is_file()
            else {"snapshot": None, "hash": None, "assistance_mode": None}
        )

        # Group sessions and promotion rows by run_id
        sessions_by_run: dict[str, list[dict[str, Any]]] = {}
        for sess in sessions:
            rid = sess.get("run_id")
            if isinstance(rid, str) and rid:
                sessions_by_run.setdefault(rid, []).append(sess)

        records_by_run: dict[str, list[dict[str, Any]]] = {}
        for rec in timing_records:
            rid = rec.get("run_id")
            if isinstance(rid, str) and rid and isinstance(rec.get("stage"), int):
                records_by_run.setdefault(rid, []).append(rec)

        all_run_ids: set[str] = set(sessions_by_run) | set(records_by_run)
        state_run_id = timing_state.get("run_id")
        if isinstance(state_run_id, str) and state_run_id:
            all_run_ids.add(state_run_id)

        def _run_sort_key(rid: str) -> str:
            candidates: list[str] = []
            for s in sessions_by_run.get(rid, []):
                if isinstance(s.get("started_at"), str):
                    candidates.append(s["started_at"])
            for r in records_by_run.get(rid, []):
                if isinstance(r.get("started_at"), str):
                    candidates.append(r["started_at"])
                if isinstance(r.get("ended_at"), str):
                    candidates.append(r["ended_at"])
            if timing_state.get("run_id") == rid and isinstance(timing_state.get("started_at"), str):
                candidates.append(timing_state["started_at"])
            if candidates:
                return min(candidates)
            parsed_start = _run_id_start(rid)
            return parsed_start.isoformat() if parsed_start else rid

        ordered_run_ids = sorted(all_run_ids, key=_run_sort_key)

        imported_passes = 0
        skipped_passes = 0
        imported_stage_runs = 0
        imported_pass_ids: list[str] = []

        active_run_id = training_state.get("run_id") or run_state.get("run_id")

        with self._lock, self._connect() as conn:
            for rid in ordered_run_ids:
                pass_id = f"pass_backfill_{rid}"
                existing = conn.execute(
                    "SELECT 1 FROM model_passes WHERE pass_id = ?",
                    (pass_id,),
                ).fetchone()
                if existing is not None:
                    skipped_passes += 1
                    continue

                run_sessions = sessions_by_run.get(rid, [])
                run_records = sorted(
                    records_by_run.get(rid, []),
                    key=lambda r: (
                        str(r.get("ended_at") or r.get("started_at") or ""),
                        int(r["stage"]),
                    ),
                )

                # Sum per-stage phase totals from training-runtime.json for this run_id
                runtime_by_stage: dict[int, dict[str, float]] = {}
                for sess in run_sessions:
                    stage_map = sess.get("stage_seconds")
                    if not isinstance(stage_map, dict):
                        continue
                    for stage_key, block in stage_map.items():
                        if not isinstance(block, dict):
                            continue
                        try:
                            st_num = int(stage_key)
                        except ValueError:
                            continue
                        acc = runtime_by_stage.setdefault(
                            st_num, {k: 0.0 for k in _PHASE_KEYS}
                        )
                        for k in _PHASE_KEYS:
                            val = _coerce_float(block.get(k))
                            if val is not None:
                                acc[k] += val

                stage_count_in_records: dict[int, int] = {}
                for rec in run_records:
                    st = int(rec["stage"])
                    stage_count_in_records[st] = stage_count_in_records.get(st, 0) + 1

                stage_entries: list[dict[str, Any]] = []
                prev_ckpt_id: str | None = None
                prev_ended_at: str | None = None
                prev_timesteps: int | None = None

                for rec in run_records:
                    st = int(rec["stage"])
                    end_ts = _coerce_int(rec.get("trigger_timesteps"))
                    pts = _coerce_int(rec.get("policy_timesteps"))
                    start_ts: int | None = None
                    if end_ts is not None and pts is not None:
                        start_ts = max(0, end_ts - pts)
                    elif prev_timesteps is not None:
                        start_ts = prev_timesteps
                        if end_ts is not None:
                            pts = max(0, end_ts - start_ts)

                    started_at = rec.get("started_at") or prev_ended_at
                    if not started_at and st == 1:
                        if run_sessions and isinstance(run_sessions[0].get("started_at"), str):
                            started_at = run_sessions[0]["started_at"]
                        else:
                            parsed = _run_id_start(rid)
                            if parsed is not None:
                                started_at = parsed.isoformat()
                    ended_at = rec.get("ended_at")
                    if not started_at:
                        started_at = ended_at or datetime.now(UTC).isoformat()

                    wall_sec = _coerce_float(rec.get("wall_seconds"))
                    if wall_sec is None and ended_at is not None:
                        wall_sec = _elapsed_seconds(started_at, ended_at)

                    rt_block = (
                        runtime_by_stage.get(st)
                        if stage_count_in_records.get(st, 0) == 1
                        else None
                    )
                    if rt_block is not None:
                        train_sec = rt_block["training_seconds"]
                        eval_sec = rt_block["evaluation_seconds"]
                        ckpt_sec = rt_block["checkpoint_save_seconds"]
                        replay_sec = rt_block["replay_serialization_seconds"]
                        active_sec = train_sec + eval_sec + ckpt_sec + replay_sec
                    else:
                        train_sec = _coerce_float(rec.get("training_seconds"))
                        eval_sec = _coerce_float(rec.get("evaluation_seconds"))
                        ckpt_sec = None
                        replay_sec = None
                        if train_sec is not None or eval_sec is not None:
                            active_sec = float(train_sec or 0.0) + float(eval_sec or 0.0)
                        else:
                            active_sec = None

                    episodes = _coerce_int(rec.get("episodes"))
                    if episodes is None and rid == active_run_id:
                        episodes = _coerce_int(stage_history.get(str(st)))

                    sr_val = _coerce_float(rec.get("success_rate"))
                    best_sr = _coerce_float(rec.get("best_success"))
                    if sr_val is not None:
                        best_sr = sr_val if best_sr is None else max(best_sr, sr_val)

                    end_ckpt_id = rec.get("trigger_checkpoint_id")
                    sps = _compute_steps_per_second(pts, train_sec, wall_sec)

                    stage_entries.append(
                        {
                            "stage": st,
                            "status": "promoted",
                            "started_at": started_at,
                            "ended_at": ended_at,
                            "start_checkpoint_id": prev_ckpt_id,
                            "start_checkpoint_sha256": None,
                            "end_checkpoint_id": end_ckpt_id,
                            "end_checkpoint_path": None,
                            "end_checkpoint_sha256": None,
                            "start_policy_timesteps": start_ts,
                            "end_policy_timesteps": end_ts,
                            "policy_timesteps": pts,
                            "episodes": episodes,
                            "training_seconds": train_sec,
                            "evaluation_seconds": eval_sec,
                            "checkpoint_save_seconds": ckpt_sec,
                            "replay_serialization_seconds": replay_sec,
                            "active_seconds": active_sec,
                            "wall_seconds": wall_sec,
                            "steps_per_second": sps,
                            "initial_success_rate": None,
                            "final_success_rate": sr_val,
                            "best_success_rate": best_sr,
                            "final_avg_steps": None,
                            "final_avg_reward": None,
                            "eval_seed_count": _coerce_int(rec.get("evaluation_seed_count")),
                            "promotion_event_id": rec.get("promotion_event_id"),
                            "source": "backfilled",
                        }
                    )
                    prev_ckpt_id = end_ckpt_id
                    prev_ended_at = ended_at
                    if end_ts is not None:
                        prev_timesteps = end_ts

                # Check final non-promoted stage from stage-timing-state.json
                last_session = run_sessions[-1] if run_sessions else None
                promoted_stages = {entry["stage"] for entry in stage_entries}

                if timing_state.get("run_id") == rid and isinstance(timing_state.get("stage"), int):
                    final_stage = int(timing_state["stage"])
                    state_started = timing_state.get("started_at")
                    already_promoted_after = any(
                        e["stage"] == final_stage
                        and (not state_started or str(e["started_at"]) >= str(state_started))
                        for e in stage_entries
                    )
                    if not already_promoted_after:
                        start_ts = _coerce_int(timing_state.get("start_timesteps"))
                        if start_ts is None:
                            start_ts = prev_timesteps
                        end_ts = (
                            _coerce_int(training_state.get("total_timesteps"))
                            if training_state.get("run_id") == rid
                            else None
                        )
                        pts = (
                            max(0, end_ts - start_ts)
                            if end_ts is not None and start_ts is not None
                            else None
                        )
                        started_at = state_started or prev_ended_at
                        if not started_at and last_session:
                            started_at = last_session.get("started_at")
                        if not started_at:
                            started_at = datetime.now(UTC).isoformat()

                        ended_at = None
                        if last_session is not None:
                            ended_at = last_session.get("ended_at") or last_session.get(
                                "last_heartbeat_at"
                            )
                        elif run_state.get("run_id") == rid:
                            ended_at = run_state.get("updated_at")

                        wall_sec = (
                            _elapsed_seconds(started_at, ended_at)
                            if ended_at is not None
                            else None
                        )

                        if last_session is not None:
                            sess_status = last_session.get("status")
                            end_reason = last_session.get("end_reason")
                            if sess_status == "ended" and end_reason in (None, "completed"):
                                final_status = "mastered"
                            elif sess_status in ("running", "paused"):
                                final_status = "running"
                            else:
                                final_status = "stopped"
                        else:
                            final_status = "stopped"

                        rt_block = runtime_by_stage.get(final_stage)
                        if rt_block is not None:
                            train_sec = rt_block["training_seconds"]
                            eval_sec = rt_block["evaluation_seconds"]
                            ckpt_sec = rt_block["checkpoint_save_seconds"]
                            replay_sec = rt_block["replay_serialization_seconds"]
                            active_sec = train_sec + eval_sec + ckpt_sec + replay_sec
                        else:
                            train_sec = None
                            eval_sec = None
                            ckpt_sec = None
                            replay_sec = None
                            active_sec = None

                        episodes = (
                            _coerce_int(stage_history.get(str(final_stage)))
                            if rid == active_run_id
                            else None
                        )

                        end_ckpt_id = (
                            run_state.get("active_checkpoint_id")
                            if run_state.get("run_id") == rid
                            and run_state.get("active_curriculum_stage") == final_stage
                            else None
                        )

                        end_ckpt_path: str | None = None
                        end_ckpt_sha: str | None = None
                        raw_candidates: list[str] = []
                        if training_state.get("run_id") == rid:
                            for key in ("best_model_path", "policy_state_path"):
                                val = training_state.get(key)
                                if isinstance(val, str) and val:
                                    raw_candidates.append(val)
                        if run_state.get("run_id") == rid:
                            val = run_state.get("active_model_path")
                            if isinstance(val, str) and val:
                                raw_candidates.append(val)

                        for raw_p in raw_candidates:
                            resolved = _resolve_artifact_path(root, raw_p)
                            if resolved is not None:
                                end_ckpt_path = raw_p
                                end_ckpt_sha = compute_file_sha256(resolved)
                                break
                        if end_ckpt_path is None and raw_candidates:
                            end_ckpt_path = raw_candidates[0]

                        best_sr = None
                        final_avg_steps = None
                        final_avg_reward = None
                        if (
                            training_state.get("run_id") == rid
                            and training_state.get("best_model_curriculum_stage") == final_stage
                        ):
                            best_sr = _coerce_float(training_state.get("best_success_rate"))
                            final_avg_steps = _coerce_float(
                                training_state.get("best_completion_steps")
                            )
                            final_avg_reward = _coerce_float(
                                training_state.get("best_average_reward")
                            )

                        sps = _compute_steps_per_second(pts, train_sec, wall_sec)
                        stage_entries.append(
                            {
                                "stage": final_stage,
                                "status": final_status,
                                "started_at": started_at,
                                "ended_at": ended_at,
                                "start_checkpoint_id": prev_ckpt_id,
                                "start_checkpoint_sha256": None,
                                "end_checkpoint_id": end_ckpt_id,
                                "end_checkpoint_path": end_ckpt_path,
                                "end_checkpoint_sha256": end_ckpt_sha,
                                "start_policy_timesteps": start_ts,
                                "end_policy_timesteps": end_ts,
                                "policy_timesteps": pts,
                                "episodes": episodes,
                                "training_seconds": train_sec,
                                "evaluation_seconds": eval_sec,
                                "checkpoint_save_seconds": ckpt_sec,
                                "replay_serialization_seconds": replay_sec,
                                "active_seconds": active_sec,
                                "wall_seconds": wall_sec,
                                "steps_per_second": sps,
                                "initial_success_rate": None,
                                "final_success_rate": None,
                                "best_success_rate": best_sr,
                                "final_avg_steps": final_avg_steps,
                                "final_avg_reward": final_avg_reward,
                                "eval_seed_count": None,
                                "promotion_event_id": None,
                                "source": "backfilled",
                            }
                        )
                elif not stage_entries and runtime_by_stage:
                    # Runtime-only run without promotions or stage-timing-state
                    for st_num in sorted(runtime_by_stage):
                        if st_num in promoted_stages:
                            continue
                        rt_block = runtime_by_stage[st_num]
                        train_sec = rt_block["training_seconds"]
                        eval_sec = rt_block["evaluation_seconds"]
                        ckpt_sec = rt_block["checkpoint_save_seconds"]
                        replay_sec = rt_block["replay_serialization_seconds"]
                        active_sec = train_sec + eval_sec + ckpt_sec + replay_sec
                        st_started = (
                            run_sessions[0].get("started_at")
                            if run_sessions
                            else datetime.now(UTC).isoformat()
                        )
                        st_ended = (
                            last_session.get("ended_at") or last_session.get("last_heartbeat_at")
                            if last_session
                            else None
                        )
                        if last_session is not None:
                            if last_session.get("status") == "ended" and last_session.get(
                                "end_reason"
                            ) in (None, "completed"):
                                st_status = "completed"
                            elif last_session.get("status") in ("running", "paused"):
                                st_status = "running"
                            else:
                                st_status = "stopped"
                        else:
                            st_status = "stopped"
                        stage_entries.append(
                            {
                                "stage": st_num,
                                "status": st_status,
                                "started_at": st_started,
                                "ended_at": st_ended,
                                "start_checkpoint_id": None,
                                "start_checkpoint_sha256": None,
                                "end_checkpoint_id": None,
                                "end_checkpoint_path": None,
                                "end_checkpoint_sha256": None,
                                "start_policy_timesteps": None,
                                "end_policy_timesteps": None,
                                "policy_timesteps": None,
                                "episodes": None,
                                "training_seconds": train_sec,
                                "evaluation_seconds": eval_sec,
                                "checkpoint_save_seconds": ckpt_sec,
                                "replay_serialization_seconds": replay_sec,
                                "active_seconds": active_sec,
                                "wall_seconds": None,
                                "steps_per_second": None,
                                "initial_success_rate": None,
                                "final_success_rate": None,
                                "best_success_rate": None,
                                "final_avg_steps": None,
                                "final_avg_reward": None,
                                "eval_seed_count": None,
                                "promotion_event_id": None,
                                "source": "backfilled",
                            }
                        )

                if not stage_entries:
                    continue

                stages_list = [int(e["stage"]) for e in stage_entries]
                stage_spec = format_stage_spec(stages_list)
                pass_type = classify_pass_type(stages_list)
                label = f"Backfilled {stage_spec} ({rid})"

                pass_started_at = min(
                    [str(e["started_at"]) for e in stage_entries if e.get("started_at")]
                    + (
                        [str(s["started_at"]) for s in run_sessions if s.get("started_at")]
                        if run_sessions
                        else []
                    )
                )
                ended_candidates = [
                    str(e["ended_at"]) for e in stage_entries if e.get("ended_at")
                ]
                if last_session and (last_session.get("ended_at") or last_session.get("last_heartbeat_at")):
                    ended_candidates.append(
                        str(last_session.get("ended_at") or last_session.get("last_heartbeat_at"))
                    )
                pass_ended_at = max(ended_candidates) if ended_candidates else None
                pass_wall = (
                    _elapsed_seconds(pass_started_at, pass_ended_at)
                    if pass_ended_at is not None
                    else None
                )

                if last_session is not None:
                    sess_status = last_session.get("status")
                    end_reason = last_session.get("end_reason")
                    if sess_status == "ended" and end_reason in (None, "completed"):
                        pass_status = "completed"
                    elif sess_status in ("running", "paused"):
                        pass_status = "running"
                    else:
                        pass_status = "stopped"
                else:
                    pass_status = "completed"

                obs_mode: str | None = None
                if rid == active_run_id:
                    obs_mode = cfg.get("assistance_mode")
                    if obs_mode is None and isinstance(training_state.get("policy_config"), dict):
                        obs_mode = training_state["policy_config"].get("observation_mode")
                if obs_mode is None:
                    for rec in reversed(run_records):
                        if rec.get("assistance_mode"):
                            obs_mode = str(rec["assistance_mode"])
                            break
                if obs_mode is None:
                    obs_mode = cfg.get("assistance_mode")

                assistance_mode = classify_assistance_mode(obs_mode)

                cfg_hash = None
                git_rev = None
                git_dirty_int: int | None = None
                hw_data = None
                sw_data = None
                for rec in reversed(run_records):
                    if cfg_hash is None and rec.get("config_hash"):
                        cfg_hash = rec.get("config_hash")
                    if git_rev is None and rec.get("git_revision"):
                        git_rev = rec.get("git_revision")
                    if git_dirty_int is None and rec.get("git_dirty") is not None:
                        git_dirty_int = 1 if rec.get("git_dirty") else 0
                    if hw_data is None and rec.get("hardware") is not None:
                        hw_data = rec.get("hardware")
                    if sw_data is None and rec.get("software") is not None:
                        sw_data = rec.get("software")
                if cfg_hash is None and rid == active_run_id:
                    cfg_hash = cfg.get("hash")

                last_entry = stage_entries[-1]
                conn.execute(
                    """
                    INSERT INTO model_passes (
                        pass_id, label, run_id, parent_pass_id,
                        resulting_checkpoint_id, resulting_checkpoint_path, resulting_checkpoint_sha256,
                        pass_type, stage_spec, planned_stages_json,
                        status, started_at, ended_at, total_wall_seconds,
                        assistance_mode, observation_mode, config_hash,
                        git_revision, git_dirty, hardware_json, software_json,
                        source
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'backfilled')
                    """,
                    (
                        pass_id,
                        label,
                        rid,
                        last_entry["end_checkpoint_id"],
                        last_entry["end_checkpoint_path"],
                        last_entry["end_checkpoint_sha256"],
                        pass_type,
                        stage_spec,
                        json.dumps(stages_list),
                        pass_status,
                        pass_started_at,
                        pass_ended_at,
                        pass_wall,
                        assistance_mode,
                        obs_mode,
                        cfg_hash,
                        git_rev,
                        git_dirty_int,
                        _to_json(hw_data),
                        _to_json(sw_data),
                    ),
                )

                for seq_idx, entry in enumerate(stage_entries):
                    sr_id = f"sr_backfill_{rid}_{seq_idx}_{entry['stage']}"
                    conn.execute(
                        """
                        INSERT INTO stage_runs (
                            stage_run_id, pass_id, run_id, stage, sequence_order,
                            status, started_at, ended_at,
                            start_checkpoint_id, start_checkpoint_sha256,
                            end_checkpoint_id, end_checkpoint_path, end_checkpoint_sha256,
                            start_policy_timesteps, end_policy_timesteps, policy_timesteps,
                            episodes, training_seconds, evaluation_seconds,
                            checkpoint_save_seconds, replay_serialization_seconds,
                            active_seconds, wall_seconds, steps_per_second,
                            initial_success_rate, final_success_rate, best_success_rate,
                            final_avg_steps, final_avg_reward, eval_seed_count,
                            promotion_event_id, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sr_id,
                            pass_id,
                            rid,
                            entry["stage"],
                            seq_idx,
                            entry["status"],
                            entry["started_at"],
                            entry["ended_at"],
                            entry["start_checkpoint_id"],
                            entry["start_checkpoint_sha256"],
                            entry["end_checkpoint_id"],
                            entry["end_checkpoint_path"],
                            entry["end_checkpoint_sha256"],
                            entry["start_policy_timesteps"],
                            entry["end_policy_timesteps"],
                            entry["policy_timesteps"],
                            entry["episodes"],
                            entry["training_seconds"],
                            entry["evaluation_seconds"],
                            entry["checkpoint_save_seconds"],
                            entry["replay_serialization_seconds"],
                            entry["active_seconds"],
                            entry["wall_seconds"],
                            entry["steps_per_second"],
                            entry["initial_success_rate"],
                            entry["final_success_rate"],
                            entry["best_success_rate"],
                            entry["final_avg_steps"],
                            entry["final_avg_reward"],
                            entry["eval_seed_count"],
                            entry["promotion_event_id"],
                            entry["source"],
                        ),
                    )
                    imported_stage_runs += 1

                self._recompute_pass_rollups(conn, pass_id)
                imported_passes += 1
                imported_pass_ids.append(pass_id)

            conn.commit()

        return {
            "imported_passes": imported_passes,
            "skipped_passes": skipped_passes,
            "imported_stage_runs": imported_stage_runs,
            "pass_ids": imported_pass_ids,
        }


def _records_from_promotion_history(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build stage-timing-like dicts from promotion-history.json when stage-timing.jsonl is absent."""
    ordered = sorted(
        (
            e
            for e in events
            if isinstance(e, dict)
            and e.get("event_type") == "promotion"
            and e.get("from_stage") is not None
        ),
        key=lambda e: str(e.get("promoted_at") or e.get("timestamp") or ""),
    )
    records: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for event in ordered:
        event_id = event.get("promotion_event_id")
        if isinstance(event_id, str) and event_id:
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
        run_id = event.get("run_id")
        stage = int(event["from_stage"])
        ended_at = event.get("promoted_at") or event.get("timestamp")
        end_ts = _timesteps_from_event(event)

        started_at: str | None = None
        start_ts: int | None = None
        for prev in reversed(records):
            if prev.get("run_id") == run_id and prev.get("to_stage") == stage:
                started_at = prev.get("ended_at")
                start_ts = prev.get("trigger_timesteps")
                break
        if started_at is None and stage == 1:
            parsed = _run_id_start(run_id)
            if parsed is not None:
                started_at = parsed.isoformat()
                start_ts = 0

        policy_timesteps = (
            max(0, end_ts - int(start_ts))
            if end_ts is not None and start_ts is not None
            else None
        )
        wall_seconds = _elapsed_seconds(started_at, ended_at)
        records.append(
            {
                "run_id": run_id,
                "stage": stage,
                "to_stage": event.get("to_stage"),
                "started_at": started_at,
                "ended_at": ended_at,
                "wall_seconds": wall_seconds,
                "trigger_timesteps": end_ts,
                "policy_timesteps": policy_timesteps,
                "trigger_checkpoint_id": event.get("trigger_checkpoint_id")
                or event.get("checkpoint_id"),
                "episodes": None,
                "success_rate": event.get("success_rate"),
                "best_success": event.get("best_success"),
                "evaluation_seed_count": event.get("evaluation_seed_count"),
                "promotion_event_id": event_id,
            }
        )
    return records
