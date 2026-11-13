"""Database schema and migrations for the Sheepdog experiment and change registry."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

MIGRATION_V1_SQL = """
-- 1. Changes
CREATE TABLE IF NOT EXISTS registry_changes (
    change_id TEXT PRIMARY KEY,
    implemented_at TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    hypothesis TEXT,
    rationale TEXT,
    category TEXT,
    source_revision TEXT,
    dirty_worktree INTEGER DEFAULT 0,
    worktree_patch_hash TEXT,
    config_diff TEXT,
    metadata TEXT
);

-- 2. Experiments
CREATE TABLE IF NOT EXISTS registry_experiments (
    experiment_id TEXT PRIMARY KEY,
    change_id TEXT REFERENCES registry_changes(change_id),
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT,
    parent_checkpoint_id TEXT,
    parent_checkpoint_path TEXT,
    parent_checkpoint_sha256 TEXT,
    assistance_mode TEXT NOT NULL,
    training_seeds TEXT,
    evaluation_seeds TEXT,
    training_budget_transitions INTEGER,
    training_budget_world_steps INTEGER,
    target_stage INTEGER,
    environment_config TEXT,
    hyperparameters TEXT,
    status TEXT DEFAULT 'planned',
    metadata TEXT
);

-- 3. Activations
CREATE TABLE IF NOT EXISTS registry_activations (
    activation_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES registry_experiments(experiment_id),
    change_id TEXT REFERENCES registry_changes(change_id),
    run_id TEXT,
    session_id TEXT,
    status TEXT NOT NULL,
    activation_method TEXT,
    verified_at TEXT,
    activated_at_episode INTEGER,
    activated_at_checkpoint TEXT,
    activated_at_policy_transition INTEGER,
    activated_at_world_step INTEGER,
    verified_source_revision TEXT,
    verification_notes TEXT,
    metadata TEXT
);

-- 4. Checkpoints
CREATE TABLE IF NOT EXISTS registry_checkpoints (
    checkpoint_record_id TEXT PRIMARY KEY,
    checkpoint_id TEXT NOT NULL,
    experiment_id TEXT REFERENCES registry_experiments(experiment_id),
    activation_id TEXT REFERENCES registry_activations(activation_id),
    run_id TEXT,
    checkpoint_path TEXT,
    checkpoint_sha256 TEXT,
    parent_checkpoint_id TEXT,
    global_environment_episode INTEGER,
    checkpoint_label_episode INTEGER,
    policy_version INTEGER,
    policy_transitions INTEGER,
    world_steps INTEGER,
    created_at TEXT,
    assistance_mode TEXT,
    curriculum_stage INTEGER,
    metadata TEXT
);

-- 5. Evaluations
CREATE TABLE IF NOT EXISTS registry_evaluations (
    registry_eval_id TEXT PRIMARY KEY,
    legacy_evaluation_id TEXT,
    source_file TEXT,
    experiment_id TEXT REFERENCES registry_experiments(experiment_id),
    checkpoint_id TEXT,
    checkpoint_record_id TEXT REFERENCES registry_checkpoints(checkpoint_record_id),
    evaluated_at TEXT NOT NULL,
    evaluation_mode TEXT,
    assistance_mode TEXT,
    model_path TEXT,
    model_sha256 TEXT,
    environment_config_hash TEXT,
    seed_set_id TEXT,
    seed_count INTEGER,
    seeds TEXT,
    success_rate REAL,
    average_reward REAL,
    average_steps REAL,
    average_sheep_penned REAL,
    timeout_rate REAL,
    stopped_rate REAL,
    per_seed_results TEXT,
    diagnostics TEXT,
    replay_paths TEXT,
    provenance_notes TEXT,
    metadata TEXT
);

-- 6. Conclusions
CREATE TABLE IF NOT EXISTS registry_conclusions (
    conclusion_id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES registry_experiments(experiment_id),
    recorded_at TEXT NOT NULL,
    author TEXT,
    status TEXT NOT NULL,
    claim TEXT NOT NULL,
    evidence_summary TEXT NOT NULL,
    matched_control_eval_id TEXT,
    candidate_eval_id TEXT,
    limitations TEXT,
    is_autonomous_claim INTEGER DEFAULT 0,
    metadata TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reg_changes_cat ON registry_changes(category);
CREATE INDEX IF NOT EXISTS idx_reg_exp_change ON registry_experiments(change_id);
CREATE INDEX IF NOT EXISTS idx_reg_act_exp ON registry_activations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_reg_act_change ON registry_activations(change_id);
CREATE INDEX IF NOT EXISTS idx_reg_act_status ON registry_activations(status);
CREATE INDEX IF NOT EXISTS idx_reg_chk_id ON registry_checkpoints(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_reg_chk_exp ON registry_checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_reg_chk_act ON registry_checkpoints(activation_id);
CREATE INDEX IF NOT EXISTS idx_reg_chk_parent ON registry_checkpoints(parent_checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_reg_eval_exp ON registry_evaluations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_reg_eval_chk_id ON registry_evaluations(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_reg_eval_chk_rec ON registry_evaluations(checkpoint_record_id);
CREATE INDEX IF NOT EXISTS idx_reg_eval_time ON registry_evaluations(evaluated_at);
CREATE INDEX IF NOT EXISTS idx_reg_conc_exp ON registry_conclusions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_reg_conc_status ON registry_conclusions(status);
"""


def apply_migrations(conn_or_path: sqlite3.Connection | str | Path) -> None:
    """Apply additive schema migrations up to SCHEMA_VERSION idempotently."""
    close_after = False
    if isinstance(conn_or_path, str | Path):
        conn = sqlite3.connect(str(conn_or_path), timeout=30.0)
        close_after = True
    else:
        conn = conn_or_path

    try:
        cur = conn.cursor()
        current_version_row = cur.execute("PRAGMA user_version;").fetchone()
        current_version = current_version_row[0] if current_version_row else 0

        if current_version < 1:
            conn.executescript(MIGRATION_V1_SQL)
            conn.execute("PRAGMA user_version = 1;")
            conn.commit()
            logger.info("Migrated schema user_version from %d to 1", current_version)
        else:
            # Idempotently ensure tables and indexes exist even if version was already set
            conn.executescript(MIGRATION_V1_SQL)
            conn.commit()
    finally:
        if close_after:
            with contextlib.suppress(Exception):
                conn.close()


def verify_migration_safety(db_path: str | Path) -> bool:
    """Verify on an isolated copy that applying migrations preserves all existing records.

    Uses SQLite backup API from a read-only connection to create an isolated point-in-time
    snapshot, ensuring safe verification even under concurrent background writes (WAL mode).

    Returns True if migration is verified safe, raises an exception or returns False otherwise.
    """
    path = Path(db_path).resolve()
    if not path.exists():
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = Path(tmpdir) / "test_migration.sqlite"

        # Create isolated snapshot using SQLite's backup API from a read-only connection.
        # This properly handles WAL mode and prevents race conditions with concurrent writers.
        escaped_src_uri = f"{path.resolve().as_uri()}?mode=ro"
        with (
            contextlib.closing(
                sqlite3.connect(escaped_src_uri, uri=True, timeout=30.0)
            ) as src_conn,
            contextlib.closing(sqlite3.connect(str(temp_db))) as dst_conn,
        ):
            src_conn.backup(dst_conn)

        # Inspect pre-migration state directly on the isolated snapshot
        with contextlib.closing(sqlite3.connect(str(temp_db))) as pre_conn:
            has_episodes = bool(
                pre_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='training_episodes';"
                ).fetchone()
            )
            before_count = (
                pre_conn.execute("SELECT COUNT(*) FROM training_episodes;").fetchone()[0]
                if has_episodes
                else 0
            )

        # Apply migrations on the isolated copy
        apply_migrations(temp_db)

        # Inspect post-migration state on the isolated copy
        with contextlib.closing(sqlite3.connect(str(temp_db))) as migrated_conn:
            migrated_version = migrated_conn.execute("PRAGMA user_version;").fetchone()[0]
            if migrated_version < 1:
                raise RuntimeError(f"Migration failed to update user_version: {migrated_version}")

            if has_episodes:
                after_count = migrated_conn.execute("SELECT COUNT(*) FROM training_episodes;").fetchone()[0]
                if after_count != before_count:
                    raise RuntimeError(
                        f"Migration data loss in training_episodes: before={before_count}, after={after_count}"
                    )

            # Verify all registry tables exist
            required_tables = [
                "registry_changes",
                "registry_experiments",
                "registry_activations",
                "registry_checkpoints",
                "registry_evaluations",
                "registry_conclusions",
            ]
            found_tables = {
                row[0]
                for row in migrated_conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            }
            for tbl in required_tables:
                if tbl not in found_tables:
                    raise RuntimeError(f"Required table {tbl} missing after migration")

        return True
