"""Registry API for managing changes, experiments, activations, checkpoints, evaluations, and conclusions."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sheepdog.tracking.schema import apply_migrations

logger = logging.getLogger(__name__)


class MigrationRequiredError(Exception):
    """Raised when the database schema requires migration before read-only access."""


def _is_valid_git_hash(rev: str | None) -> bool:
    if not rev or not isinstance(rev, str):
        return False
    if not (7 <= len(rev) <= 40):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in rev)


def _to_json(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val, sort_keys=True)


def _from_json(val: str | None) -> Any:
    if val is None:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


@dataclass
class ChangeRecord:
    """Record of an intentional code, strategy, or configuration change."""

    change_id: str
    implemented_at: str
    title: str
    description: str | None = None
    hypothesis: str | None = None
    rationale: str | None = None
    category: str | None = None
    source_revision: str | None = None
    dirty_worktree: int | None = 0
    worktree_patch_hash: str | None = None
    config_diff: str | dict[str, Any] | None = None
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["config_diff"] = _to_json(self.config_diff)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class ExperimentRecord:
    """Record of a planned or executed experiment testing a change."""

    experiment_id: str
    created_at: str
    name: str
    change_id: str | None = None
    hypothesis: str | None = None
    parent_checkpoint_id: str | None = None
    parent_checkpoint_path: str | None = None
    parent_checkpoint_sha256: str | None = None
    assistance_mode: str = "guided"
    training_seeds: str | list[int] | None = None
    evaluation_seeds: str | list[int] | None = None
    training_budget_transitions: int | None = None
    training_budget_world_steps: int | None = None
    target_stage: int | None = None
    environment_config: str | dict[str, Any] | None = None
    hyperparameters: str | dict[str, Any] | None = None
    status: str = "planned"
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["training_seeds"] = _to_json(self.training_seeds)
        d["evaluation_seeds"] = _to_json(self.evaluation_seeds)
        d["environment_config"] = _to_json(self.environment_config)
        d["hyperparameters"] = _to_json(self.hyperparameters)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class ActivationRecord:
    """Record verifying when and how a change was activated at runtime."""

    activation_id: str
    status: str  # 'active', 'pending_restart', 'historical_inferred', 'unverified'
    experiment_id: str | None = None
    change_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    activation_method: str | None = None  # 'trainer_startup', 'runtime_handshake', 'historical_import'
    verified_at: str | None = None
    activated_at_episode: int | None = None
    activated_at_checkpoint: str | None = None
    activated_at_policy_transition: int | None = None
    activated_at_world_step: int | None = None
    verified_source_revision: str | None = None
    verification_notes: str | None = None
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class CheckpointRecord:
    """Record capturing checkpoint identity, counters, and lineage."""

    checkpoint_record_id: str
    checkpoint_id: str
    experiment_id: str | None = None
    activation_id: str | None = None
    run_id: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    parent_checkpoint_id: str | None = None
    global_environment_episode: int | None = None
    checkpoint_label_episode: int | None = None
    policy_version: int | None = None
    policy_transitions: int | None = None
    world_steps: int | None = None
    created_at: str | None = None
    assistance_mode: str | None = None
    curriculum_stage: int | None = None
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class EvaluationRecordEntry:
    """Record linking formal evaluation results to a checkpoint and experiment."""

    registry_eval_id: str
    evaluated_at: str
    legacy_evaluation_id: str | None = None
    source_file: str | None = None
    experiment_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_record_id: str | None = None
    evaluation_mode: str | None = None
    assistance_mode: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    environment_config_hash: str | None = None
    seed_set_id: str | None = None
    seed_count: int | None = None
    seeds: str | list[int] | None = None
    success_rate: float | None = None
    average_reward: float | None = None
    average_steps: float | None = None
    average_sheep_penned: float | None = None
    timeout_rate: float | None = None
    stopped_rate: float | None = None
    per_seed_results: str | dict[str, Any] | list[Any] | None = None
    diagnostics: str | dict[str, Any] | None = None
    replay_paths: str | list[str] | None = None
    provenance_notes: str | None = None
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["seeds"] = _to_json(self.seeds)
        d["per_seed_results"] = _to_json(self.per_seed_results)
        d["diagnostics"] = _to_json(self.diagnostics)
        d["replay_paths"] = _to_json(self.replay_paths)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class ConclusionRecord:
    """Record storing an evidence-backed finding or conclusion for an experiment."""

    conclusion_id: str
    recorded_at: str
    claim: str
    evidence_summary: str
    status: str  # 'hypothesis_supported', 'hypothesis_rejected', 'inconclusive', 'regression_observed'
    experiment_id: str | None = None
    author: str | None = None
    matched_control_eval_id: str | None = None
    candidate_eval_id: str | None = None
    limitations: str | None = None
    is_autonomous_claim: int = 0
    metadata: str | dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        """Convert record fields to database row dictionary with JSON serialized attributes."""
        d = asdict(self)
        d["metadata"] = _to_json(self.metadata)
        return d


@dataclass
class CheckpointLineage:
    """Aggregated lineage information answering the five core tracking questions."""

    checkpoint_id: str
    checkpoint: CheckpointRecord | None = None
    experiment: ExperimentRecord | None = None
    change: ChangeRecord | None = None
    activation: ActivationRecord | None = None
    evaluations: list[EvaluationRecordEntry] = field(default_factory=list)
    conclusions: list[ConclusionRecord] = field(default_factory=list)
    has_complete_provenance: bool = False
    missing_provenance_reasons: list[str] = field(default_factory=list)


def _is_excluded_untracked_path(rel_path: str) -> bool:
    """Check if an untracked path should be excluded from provenance tracking."""
    norm = rel_path.replace("\\", "/").strip().strip('"')
    lower = norm.lower()
    parts = lower.split("/")

    # Exclude common environment, build, and artifact directories
    excluded_dirs = (
        ".venv",
        "venv",
        "env",
        "node_modules",
        "artifacts",
        "web/dist",
        "web/build",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    )
    for ed in excluded_dirs:
        if norm == ed or norm.startswith(ed + "/") or f"/{ed}/" in f"/{norm}/":
            return True

    # Exclude secrets and credentials
    base_name = parts[-1]
    if base_name.startswith(".env") or "credential" in base_name or "secret" in base_name:
        return True
    if base_name.endswith((".pem", ".key", ".pfx", ".p12")) or base_name == "id_rsa":
        return True
    if any(p.endswith(".egg-info") for p in parts):
        return True

    # Exclude large binary formats and database files
    binary_exts = (".zip", ".sqlite", ".sqlite3", ".db", ".tar", ".gz", ".bin", ".pt", ".onnx")
    return any(lower.endswith(ext) for ext in binary_exts)


def _is_relevant_source_file(rel_path: str) -> bool:
    """Check if an untracked path qualifies as a tracked source/configuration/test file."""
    norm = rel_path.replace("\\", "/").strip().strip('"')
    lower = norm.lower()

    source_prefixes = ("src/", "tests/", "scripts/", "docs/", "web/src/")
    if any(norm.startswith(p) for p in source_prefixes):
        return True

    source_exts = (".py", ".ts", ".tsx", ".json", ".toml", ".md", ".yaml", ".yml", ".sh", ".ps1")
    return any(lower.endswith(ext) for ext in source_exts)


def capture_current_git_provenance(cwd: str | Path | None = None) -> tuple[str | None, int, str | None]:
    """Capture current git commit, dirty worktree flag, and worktree patch hash.

    Accounts for both tracked diffs and untracked source/test/config files while
    excluding secrets, large experiment binaries, and environment directories.
    """
    repo_dir = Path(cwd) if cwd is not None else Path(".")
    try:
        rev_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        rev = rev_proc.stdout.strip() if rev_proc.returncode == 0 else None

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )

        has_tracked_changes = False
        untracked_source_files: dict[str, bytes] = {}

        if status_proc.returncode == 0 and status_proc.stdout:
            for line in status_proc.stdout.splitlines():
                if not line.strip():
                    continue
                if line.startswith("?? "):
                    raw_rel = line[3:].strip().strip('"')
                    target = repo_dir / raw_rel
                    if target.is_dir():
                        for child in target.rglob("*"):
                            if child.is_file():
                                rel_child = str(child.relative_to(repo_dir)).replace("\\", "/")
                                if not _is_excluded_untracked_path(rel_child) and _is_relevant_source_file(rel_child):
                                    try:
                                        if child.stat().st_size <= 10 * 1024 * 1024:
                                            untracked_source_files[rel_child] = child.read_bytes()
                                    except OSError:
                                        pass
                    elif target.is_file():
                        rel_file = raw_rel.replace("\\", "/")
                        if not _is_excluded_untracked_path(rel_file) and _is_relevant_source_file(rel_file):
                            try:
                                if target.stat().st_size <= 10 * 1024 * 1024:
                                    untracked_source_files[rel_file] = target.read_bytes()
                            except OSError:
                                pass
                else:
                    has_tracked_changes = True

        is_dirty = 1 if (has_tracked_changes or len(untracked_source_files) > 0) else 0

        patch_hash = None
        if is_dirty:
            hasher = hashlib.sha256()
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                check=False,
            )
            if diff_proc.returncode == 0 and diff_proc.stdout:
                hasher.update(diff_proc.stdout)

            for rel_path in sorted(untracked_source_files.keys()):
                hasher.update(f"\nuntracked:{rel_path}\n".encode())
                hasher.update(untracked_source_files[rel_path])

            patch_hash = hasher.hexdigest()

        return rev, is_dirty, patch_hash
    except Exception as exc:
        logger.warning("Failed to capture git provenance: %s", exc)
        return None, 0, None


def classify_assistance_mode(obs_mode: str | None) -> str:
    """Classify assistance mode strictly without assuming autonomous.

    Only 'guided' and 'emergent' are recognized modes; everything else is 'unknown'.
    An explicit audit is required before any mode may be claimed autonomous.
    """
    if obs_mode == "guided":
        return "guided"
    if obs_mode == "emergent":
        return "emergent"
    return "unknown"


class TrackingRegistry:
    """Manages the experiment and change tracking database."""

    def __init__(
        self,
        db_path: str | Path | sqlite3.Connection = "artifacts/training-telemetry.sqlite",
        read_only: bool = False,
    ) -> None:
        """Initialize registry connection and apply schema migrations if writable."""
        self.read_only = read_only
        if isinstance(db_path, sqlite3.Connection):
            self._external_conn: sqlite3.Connection | None = db_path
            self.db_path: Path | None = None
        else:
            self._external_conn = None
            self.db_path = Path(db_path)
            if self.read_only:
                if not self.db_path.exists():
                    raise FileNotFoundError(f"Database does not exist: {self.db_path}")
            else:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In read_only mode, verify schema user_version >= 1 without running migrations
        with self._get_connection() as conn:
            if self.read_only:
                cur = conn.cursor()
                row = cur.execute("PRAGMA user_version;").fetchone()
                current_ver = row[0] if row else 0
                if current_ver < 1:
                    raise MigrationRequiredError(
                        f"Database at {self.db_path} requires migration (user_version {current_ver} < 1). "
                        f"Run 'python scripts/import_historical_trials.py --db {self.db_path}' to apply migrations."
                    )
            else:
                apply_migrations(conn)

    @contextlib.contextmanager
    def _get_connection(self):
        if self._external_conn is not None:
            self._external_conn.row_factory = sqlite3.Row
            self._external_conn.execute("PRAGMA foreign_keys = ON;")
            yield self._external_conn
        else:
            if self.read_only:
                if not self.db_path.exists():
                    raise FileNotFoundError(f"Database does not exist: {self.db_path}")
                escaped_uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
                conn = sqlite3.connect(
                    escaped_uri,
                    uri=True,
                    timeout=30.0,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                try:
                    yield conn
                finally:
                    conn.close()
            else:
                conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode = WAL;")
                try:
                    yield conn
                finally:
                    conn.close()

    def record_change(self, change: ChangeRecord) -> None:
        """Record or replace a change entry in registry_changes."""
        row = change.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_changes ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def record_experiment(self, experiment: ExperimentRecord) -> None:
        """Record or replace an experiment entry in registry_experiments."""
        row = experiment.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_experiments ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def record_activation(self, activation: ActivationRecord) -> None:
        """Record or replace an activation entry in registry_activations."""
        row = activation.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_activations ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def record_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        """Record or replace a checkpoint entry in registry_checkpoints."""
        row = checkpoint.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_checkpoints ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def record_evaluation(self, evaluation: EvaluationRecordEntry) -> None:
        """Record or replace an evaluation entry in registry_evaluations."""
        row = evaluation.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_evaluations ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def record_conclusion(self, conclusion: ConclusionRecord) -> None:
        """Record or replace a conclusion entry in registry_conclusions."""
        row = conclusion.to_row()
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT OR REPLACE INTO registry_conclusions ({cols}) VALUES ({placeholders});"
        with self._get_connection() as conn:
            conn.execute(sql, row)
            conn.commit()

    def get_change(self, change_id: str) -> ChangeRecord | None:
        """Retrieve a change record by change_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM registry_changes WHERE change_id = ?;", (change_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["config_diff"] = _from_json(d.get("config_diff"))
            d["metadata"] = _from_json(d.get("metadata"))
            return ChangeRecord(**d)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve an experiment record by experiment_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM registry_experiments WHERE experiment_id = ?;", (experiment_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["training_seeds"] = _from_json(d.get("training_seeds"))
            d["evaluation_seeds"] = _from_json(d.get("evaluation_seeds"))
            d["environment_config"] = _from_json(d.get("environment_config"))
            d["hyperparameters"] = _from_json(d.get("hyperparameters"))
            d["metadata"] = _from_json(d.get("metadata"))
            return ExperimentRecord(**d)

    def get_activation(self, activation_id: str) -> ActivationRecord | None:
        """Retrieve an activation record by activation_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM registry_activations WHERE activation_id = ?;", (activation_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["metadata"] = _from_json(d.get("metadata"))
            return ActivationRecord(**d)

    def get_checkpoint(self, checkpoint_record_id: str) -> CheckpointRecord | None:
        """Retrieve a checkpoint record by checkpoint_record_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM registry_checkpoints WHERE checkpoint_record_id = ?;", (checkpoint_record_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["metadata"] = _from_json(d.get("metadata"))
            return CheckpointRecord(**d)

    def get_checkpoint_by_id(self, checkpoint_id: str) -> CheckpointRecord | None:
        """Retrieve a checkpoint record by checkpoint_id or checkpoint_record_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM registry_checkpoints WHERE checkpoint_id = ? OR checkpoint_record_id = ? ORDER BY ROWID DESC LIMIT 1;",
                (checkpoint_id, checkpoint_id),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["metadata"] = _from_json(d.get("metadata"))
            return CheckpointRecord(**d)

    def list_checkpoints_for_experiment(self, experiment_id: str) -> list[CheckpointRecord]:
        """List all checkpoint records associated with an experiment."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM registry_checkpoints WHERE experiment_id = ? ORDER BY global_environment_episode ASC, ROWID ASC;",
                (experiment_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["metadata"] = _from_json(d.get("metadata"))
                results.append(CheckpointRecord(**d))
            return results

    def list_evaluations_for_checkpoint(self, checkpoint_id_or_record_id: str) -> list[EvaluationRecordEntry]:
        """List all evaluation records linked to a checkpoint."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM registry_evaluations WHERE checkpoint_id = ? OR checkpoint_record_id = ? ORDER BY evaluated_at ASC, ROWID ASC;",
                (checkpoint_id_or_record_id, checkpoint_id_or_record_id),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["seeds"] = _from_json(d.get("seeds"))
                d["per_seed_results"] = _from_json(d.get("per_seed_results"))
                d["diagnostics"] = _from_json(d.get("diagnostics"))
                d["replay_paths"] = _from_json(d.get("replay_paths"))
                d["metadata"] = _from_json(d.get("metadata"))
                results.append(EvaluationRecordEntry(**d))
            return results

    def list_conclusions_for_experiment(self, experiment_id: str) -> list[ConclusionRecord]:
        """List all conclusion records associated with an experiment."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM registry_conclusions WHERE experiment_id = ? ORDER BY recorded_at ASC, ROWID ASC;",
                (experiment_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["metadata"] = _from_json(d.get("metadata"))
                results.append(ConclusionRecord(**d))
            return results

    def get_history_for_checkpoint(self, checkpoint_id_or_record_id: str) -> CheckpointLineage:
        """Trace the full lineage for a checkpoint and evaluate provenance completeness."""
        chk = self.get_checkpoint_by_id(checkpoint_id_or_record_id)
        if chk is None:
            chk = self.get_checkpoint(checkpoint_id_or_record_id)

        missing_reasons: list[str] = []
        if chk is None:
            missing_reasons.append(f"Checkpoint record not found for query '{checkpoint_id_or_record_id}'")
            return CheckpointLineage(
                checkpoint_id=checkpoint_id_or_record_id,
                checkpoint=None,
                experiment=None,
                change=None,
                activation=None,
                evaluations=[],
                conclusions=[],
                has_complete_provenance=False,
                missing_provenance_reasons=missing_reasons,
            )

        if not chk.parent_checkpoint_id:
            missing_reasons.append("Parent checkpoint is unrecorded")

        exp = self.get_experiment(chk.experiment_id) if chk.experiment_id else None
        if not exp:
            missing_reasons.append("Experiment record missing or unlinked")

        chg = None
        if exp and exp.change_id:
            chg = self.get_change(exp.change_id)
        if not chg:
            missing_reasons.append("Change record missing or unlinked")
        else:
            if not chg.source_revision or not _is_valid_git_hash(chg.source_revision):
                missing_reasons.append("Change source_revision is unrecorded or not a valid Git commit")
            if chg.dirty_worktree is None:
                missing_reasons.append("Worktree dirty status at implementation was unrecorded")

        act = None
        if not chk.activation_id:
            missing_reasons.append("Checkpoint has no direct activation link; cannot infer which activation produced it")
        else:
            act = self.get_activation(chk.activation_id)
            if not act:
                missing_reasons.append(f"Activation record '{chk.activation_id}' not found in registry")
            elif act.status == "historical_inferred":
                missing_reasons.append(
                    "Activation status is historical_inferred (inferred from file timestamps; running trainer did not verify load)"
                )
            elif act.status == "pending_restart":
                missing_reasons.append(
                    "Activation status is pending_restart (trainer runtime has not restarted to load new code)"
                )
            elif act.status == "unverified":
                missing_reasons.append(
                    "Activation status is unverified (code load was not verified by running process)"
                )
            elif act.status != "active":
                missing_reasons.append(f"Activation status is {act.status}")

        evals = self.list_evaluations_for_checkpoint(chk.checkpoint_id)
        if not evals and chk.checkpoint_record_id:
            evals = self.list_evaluations_for_checkpoint(chk.checkpoint_record_id)

        conclusions = []
        if exp:
            conclusions = self.list_conclusions_for_experiment(exp.experiment_id)

        has_complete = len(missing_reasons) == 0

        return CheckpointLineage(
            checkpoint_id=chk.checkpoint_id,
            checkpoint=chk,
            experiment=exp,
            change=chg,
            activation=act,
            evaluations=evals,
            conclusions=conclusions,
            has_complete_provenance=has_complete,
            missing_provenance_reasons=missing_reasons,
        )
