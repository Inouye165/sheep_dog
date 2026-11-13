"""Tests for experiment and change tracking registry, migrations, and provenance reporting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sheepdog.tracking.importer import import_historical_trials, repair_historical_registry
from sheepdog.tracking.registry import (
    ActivationRecord,
    ChangeRecord,
    CheckpointRecord,
    ConclusionRecord,
    EvaluationRecordEntry,
    ExperimentRecord,
    MigrationRequiredError,
    TrackingRegistry,
    classify_assistance_mode,
)
from sheepdog.tracking.report import format_checkpoint_history_report
from sheepdog.tracking.schema import SCHEMA_VERSION, apply_migrations, verify_migration_safety
from sheepdog.training.episode_store import EpisodeStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_tracking.sqlite"


def test_schema_migration_from_empty(temp_db_path: Path):
    """Applying migrations to an empty database initializes user_version to 1 and creates tables."""
    conn = sqlite3.connect(str(temp_db_path))
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 0
    apply_migrations(conn)
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    for expected in [
        "registry_changes",
        "registry_experiments",
        "registry_activations",
        "registry_checkpoints",
        "registry_evaluations",
        "registry_conclusions",
    ]:
        assert expected in tables
    conn.close()


def test_schema_migration_preserves_existing_episodes(temp_db_path: Path):
    """Applying migrations to a database with existing training_episodes preserves all data."""
    conn = sqlite3.connect(str(temp_db_path))
    conn.execute(
        """
        CREATE TABLE training_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL UNIQUE,
            reward REAL,
            steps INTEGER
        );
        """
    )
    conn.execute("INSERT INTO training_episodes (event_key, reward, steps) VALUES ('ep1', 123.4, 250);")
    conn.execute("INSERT INTO training_episodes (event_key, reward, steps) VALUES ('ep2', -45.6, 600);")
    conn.commit()
    conn.close()

    # Safety verification
    assert verify_migration_safety(temp_db_path) is True

    # Apply migration directly
    apply_migrations(temp_db_path)

    conn = sqlite3.connect(str(temp_db_path))
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1
    count = conn.execute("SELECT COUNT(*) FROM training_episodes;").fetchone()[0]
    assert count == 2
    row = conn.execute("SELECT reward, steps FROM training_episodes WHERE event_key = 'ep1';").fetchone()
    assert row[0] == 123.4
    assert row[1] == 250
    conn.close()


def test_episode_store_automatically_applies_registry_migrations(tmp_path: Path):
    """EpisodeStore initialization must automatically run apply_migrations."""
    store_db = tmp_path / "store_test.sqlite"
    store = EpisodeStore(db_path=store_db)
    try:
        conn = sqlite3.connect(str(store_db))
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        assert "registry_changes" in tables
        assert "training_episodes" in tables
        conn.close()
    finally:
        store.close()


def test_full_relational_lineage_and_foreign_keys(temp_db_path: Path):
    """Full relationship chain (change -> experiment -> activation -> checkpoint -> evaluation -> conclusion) works with FKs."""
    registry = TrackingRegistry(temp_db_path)

    # 1. Change
    change = ChangeRecord(
        change_id="chg_001",
        implemented_at="2026-11-10T12:00:00Z",
        title="Flanking Angle Refinement",
        description="Refined flanker angle from 45 to 30 degrees",
        hypothesis="Smaller flanking angle reduces sheep escape margin",
        category="strategy",
        source_revision="abc1234",
        dirty_worktree=0,
    )
    registry.record_change(change)

    # 2. Experiment
    exp = ExperimentRecord(
        experiment_id="exp_001",
        change_id="chg_001",
        created_at="2026-11-10T12:10:00Z",
        name="Flanking Angle 30deg Pilot",
        parent_checkpoint_id="checkpoint-040000",
        assistance_mode="guided",
        training_seeds=[101, 102],
        evaluation_seeds=[11, 23, 37],
        training_budget_transitions=10000,
        training_budget_world_steps=3333,
    )
    registry.record_experiment(exp)

    # 3. Activation
    act = ActivationRecord(
        activation_id="act_001",
        experiment_id="exp_001",
        change_id="chg_001",
        status="active",
        activation_method="trainer_startup",
        verified_at="2026-11-10T12:15:00Z",
        activated_at_episode=40001,
        activated_at_checkpoint="checkpoint-040001",
        activated_at_policy_transition=10000,
        activated_at_world_step=3333,
    )
    registry.record_activation(act)

    # 4. Checkpoint
    chk = CheckpointRecord(
        checkpoint_record_id="chk_rec_001",
        checkpoint_id="checkpoint-040001",
        parent_checkpoint_id="checkpoint-040000",
        experiment_id="exp_001",
        activation_id="act_001",
        global_environment_episode=40001,
        checkpoint_label_episode=40001,
        policy_version=10500,
        policy_transitions=10000,
        world_steps=3333,
        created_at="2026-11-10T12:30:00Z",
        assistance_mode="guided",
        curriculum_stage=34,
    )
    registry.record_checkpoint(chk)

    # 5. Evaluation
    eval_entry = EvaluationRecordEntry(
        registry_eval_id="eval_reg_001",
        evaluated_at="2026-11-10T12:35:00Z",
        legacy_evaluation_id="eval_legacy_001",
        experiment_id="exp_001",
        checkpoint_id="checkpoint-040001",
        checkpoint_record_id="chk_rec_001",
        success_rate=0.9,
        average_reward=150.0,
        average_steps=300.0,
        average_sheep_penned=5.6,
        timeout_rate=0.1,
        stopped_rate=0.0,
        seeds=[11, 23, 37],
    )
    registry.record_evaluation(eval_entry)

    # 6. Conclusion
    conc = ConclusionRecord(
        conclusion_id="conc_001",
        experiment_id="exp_001",
        recorded_at="2026-11-10T12:40:00Z",
        author="evaluator",
        status="hypothesis_supported",
        claim="Flanking angle 30deg increased success to 90%",
        evidence_summary="9/10 seeds passed with avg steps 300 vs baseline 450",
        is_autonomous_claim=0,
    )
    registry.record_conclusion(conc)

    # Verify history resolution
    lineage = registry.get_history_for_checkpoint("checkpoint-040001")
    assert lineage.has_complete_provenance is True
    assert lineage.missing_provenance_reasons == []
    assert lineage.checkpoint.checkpoint_id == "checkpoint-040001"
    assert lineage.experiment.experiment_id == "exp_001"
    assert lineage.change.change_id == "chg_001"
    assert lineage.activation.status == "active"
    assert len(lineage.evaluations) == 1
    assert lineage.evaluations[0].registry_eval_id == "eval_reg_001"
    assert len(lineage.conclusions) == 1
    assert lineage.conclusions[0].conclusion_id == "conc_001"

    # Format report and verify contents
    report = format_checkpoint_history_report(lineage)
    assert "Flanking Angle Refinement" in report
    assert "2026-11-10T12:00:00Z" in report
    assert "Runtime Activation Status**: `active`" in report
    assert "90.0%" in report
    assert "NO (Assisted/Guided)" in report


def test_separation_of_implementation_time_and_runtime_activation(temp_db_path: Path):
    """Registry distinguishes implementation time from runtime activation time (pending_restart)."""
    registry = TrackingRegistry(temp_db_path)

    change = ChangeRecord(
        change_id="chg_pending_01",
        implemented_at="2026-11-10T14:00:00Z",
        title="Unactivated Feature",
    )
    registry.record_change(change)

    exp = ExperimentRecord(
        experiment_id="exp_pending_01",
        change_id="chg_pending_01",
        created_at="2026-11-10T14:05:00Z",
        name="Pending Experiment",
    )
    registry.record_experiment(exp)

    # Runtime is pending restart (code modified on disk, running server not restarted)
    act = ActivationRecord(
        activation_id="act_pending_01",
        experiment_id="exp_pending_01",
        change_id="chg_pending_01",
        status="pending_restart",
        activation_method="trainer_startup",
        verified_at=None,
    )
    registry.record_activation(act)

    chk = CheckpointRecord(
        checkpoint_record_id="chk_pending_01",
        checkpoint_id="checkpoint-pending",
        experiment_id="exp_pending_01",
        activation_id="act_pending_01",
    )
    registry.record_checkpoint(chk)

    lineage = registry.get_history_for_checkpoint("checkpoint-pending")
    assert lineage.has_complete_provenance is False
    assert any("pending_restart" in r for r in lineage.missing_provenance_reasons)

    report = format_checkpoint_history_report(lineage)
    assert "pending_restart" in report
    assert "running training server process has not restarted" in report


def test_counter_distinctions_and_null_handling(temp_db_path: Path):
    """Episode count, checkpoint label, policy transitions, and world steps are clearly distinguished."""
    registry = TrackingRegistry(temp_db_path)

    chk = CheckpointRecord(
        checkpoint_record_id="chk_counters_01",
        checkpoint_id="checkpoint-counter-test",
        checkpoint_label_episode=40400,
        global_environment_episode=392363,
        policy_version=10277,
        policy_transitions=336684452,
        world_steps=112228150,
        curriculum_stage=34,
    )
    registry.record_checkpoint(chk)

    retrieved = registry.get_checkpoint("chk_counters_01")
    assert retrieved.checkpoint_label_episode == 40400
    assert retrieved.global_environment_episode == 392363
    assert retrieved.policy_transitions == 336684452
    assert retrieved.world_steps == 112228150


def test_collision_handling_for_phase4_and_phase4_1(temp_db_path: Path):
    """Importing Phase 4 and Phase 4.1 with colliding legacy evaluation IDs creates two distinct registry evaluations."""
    registry = TrackingRegistry(temp_db_path)

    # Insert shared experiment and changes
    chg4 = ChangeRecord(change_id="chg_p4", implemented_at="2026-11-08T22:50:00Z", title="Phase 4 Decay 60")
    exp4 = ExperimentRecord(experiment_id="exp_p4", change_id="chg_p4", created_at="2026-11-08T22:55:00Z", name="Phase 4")
    chk4 = CheckpointRecord(checkpoint_record_id="chk_p4", checkpoint_id="checkpoint-p4", experiment_id="exp_p4")

    chg4_1 = ChangeRecord(change_id="chg_p4_1", implemented_at="2026-11-08T23:12:00Z", title="Phase 4.1 Decay 150")
    exp4_1 = ExperimentRecord(experiment_id="exp_p4_1", change_id="chg_p4_1", created_at="2026-11-08T23:13:00Z", name="Phase 4.1")
    chk4_1 = CheckpointRecord(checkpoint_record_id="chk_p4_1", checkpoint_id="checkpoint-p4_1", experiment_id="exp_p4_1")

    registry.record_change(chg4)
    registry.record_experiment(exp4)
    registry.record_checkpoint(chk4)

    registry.record_change(chg4_1)
    registry.record_experiment(exp4_1)
    registry.record_checkpoint(chk4_1)

    # Both evaluations have the EXACT SAME legacy evaluation ID: eval_pilot_chk00057344_confidence_ff85663fc223
    legacy_id = "eval_pilot_chk00057344_confidence_ff85663fc223"

    eval1 = EvaluationRecordEntry(
        registry_eval_id="eval_reg_20261108_phase4_thresh60",
        legacy_evaluation_id=legacy_id,
        experiment_id="exp_p4",
        checkpoint_id="checkpoint-p4",
        checkpoint_record_id="chk_p4",
        evaluated_at="2026-11-08T23:05:15Z",
        success_rate=0.9,
    )
    eval2 = EvaluationRecordEntry(
        registry_eval_id="eval_reg_20261108_phase4_1_thresh150",
        legacy_evaluation_id=legacy_id,
        experiment_id="exp_p4_1",
        checkpoint_id="checkpoint-p4_1",
        checkpoint_record_id="chk_p4_1",
        evaluated_at="2026-11-08T23:14:13Z",
        success_rate=0.9,
    )

    registry.record_evaluation(eval1)
    registry.record_evaluation(eval2)

    with registry._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM registry_evaluations;").fetchone()[0]
        assert count == 2

    # Querying evaluations for each checkpoint returns the respective distinct entry
    evals_p4 = registry.list_evaluations_for_checkpoint("checkpoint-p4")
    assert len(evals_p4) == 1
    assert evals_p4[0].registry_eval_id == "eval_reg_20261108_phase4_thresh60"
    assert evals_p4[0].legacy_evaluation_id == legacy_id

    evals_p4_1 = registry.list_evaluations_for_checkpoint("checkpoint-p4_1")
    assert len(evals_p4_1) == 1
    assert evals_p4_1[0].registry_eval_id == "eval_reg_20261108_phase4_1_thresh150"
    assert evals_p4_1[0].legacy_evaluation_id == legacy_id


def test_import_historical_trials_idempotency(temp_db_path: Path):
    """Running import_historical_trials twice produces identical counts with zero duplicate errors."""
    counts1 = import_historical_trials(temp_db_path)
    assert counts1["changes"] >= 5
    assert counts1["experiments"] >= 5
    assert counts1["activations"] >= 5
    assert counts1["checkpoints"] >= 50
    assert counts1["evaluations"] >= 50
    assert counts1["conclusions"] >= 5

    # Run again - must be idempotent
    counts2 = import_historical_trials(temp_db_path)
    assert counts2 == counts1

    # Verify database table row counts match exactly
    conn = sqlite3.connect(str(temp_db_path))
    chg_count = conn.execute("SELECT COUNT(*) FROM registry_changes;").fetchone()[0]
    exp_count = conn.execute("SELECT COUNT(*) FROM registry_experiments;").fetchone()[0]
    act_count = conn.execute("SELECT COUNT(*) FROM registry_activations;").fetchone()[0]
    chk_count = conn.execute("SELECT COUNT(*) FROM registry_checkpoints;").fetchone()[0]
    eval_count = conn.execute("SELECT COUNT(*) FROM registry_evaluations;").fetchone()[0]
    conc_count = conn.execute("SELECT COUNT(*) FROM registry_conclusions;").fetchone()[0]
    conn.close()

    assert chg_count == counts1["changes"]
    assert exp_count == counts1["experiments"]
    assert act_count == counts1["activations"]
    assert chk_count == counts1["checkpoints"]
    assert eval_count == counts1["evaluations"]
    assert conc_count == counts1["conclusions"]


def test_incomplete_provenance_reporting(temp_db_path: Path):
    """Checkpoint without linked records reports has_complete_provenance=False and lists missing reasons."""
    registry = TrackingRegistry(temp_db_path)

    # Checkpoint completely unknown
    lineage_unknown = registry.get_history_for_checkpoint("checkpoint-nonexistent")
    assert lineage_unknown.has_complete_provenance is False
    assert any("not found" in r for r in lineage_unknown.missing_provenance_reasons)
    report_unknown = format_checkpoint_history_report(lineage_unknown)
    assert "No checkpoint record found" in report_unknown

    # Checkpoint without linked experiment
    chk_orphan = CheckpointRecord(
        checkpoint_record_id="chk_orphan_01",
        checkpoint_id="checkpoint-orphan",
        experiment_id=None,
    )
    registry.record_checkpoint(chk_orphan)

    lineage_orphan = registry.get_history_for_checkpoint("checkpoint-orphan")
    assert lineage_orphan.has_complete_provenance is False
    assert "Experiment record missing or unlinked" in lineage_orphan.missing_provenance_reasons
    assert "Change record missing or unlinked" in lineage_orphan.missing_provenance_reasons
    assert any("activation" in r.lower() for r in lineage_orphan.missing_provenance_reasons)

    report_orphan = format_checkpoint_history_report(lineage_orphan)
    assert "INCOMPLETE OR UNVERIFIED" in report_orphan
    assert "Experiment record missing or unlinked" in report_orphan


def test_checkpoint_history_report_on_imported_trials(temp_db_path: Path):
    """Verify generated history reports on imported Phase 2, Phase 3, Phase 4.1, and Phase 5 checkpoints."""
    import_historical_trials(temp_db_path)
    registry = TrackingRegistry(temp_db_path)

    # Test Phase 4.1 checkpoint
    lineage_p4_1 = registry.get_history_for_checkpoint("checkpoint-040400-phase4_1")
    assert lineage_p4_1.checkpoint is not None
    assert lineage_p4_1.experiment is not None
    assert lineage_p4_1.change is not None
    assert lineage_p4_1.activation is not None
    assert len(lineage_p4_1.evaluations) >= 1
    assert len(lineage_p4_1.conclusions) >= 1

    report_p4_1 = format_checkpoint_history_report(lineage_p4_1)
    assert "Role Stagnation Threshold Relaxation to 150 Steps" in report_p4_1
    assert "eval_reg_20261108_phase4_1_thresh150" in report_p4_1
    assert "eval_pilot_chk00057344_confidence_ff85663fc223" in report_p4_1
    assert "Seed 41" in report_p4_1
    assert "historical_inferred" in report_p4_1


def test_read_only_mode_non_existent_db_fails_without_creating_file(tmp_path: Path):
    """TrackingRegistry in read_only=True raises FileNotFoundError on missing file without creating it."""
    missing_db = tmp_path / "subdir" / "does_not_exist.sqlite"
    assert missing_db.exists() is False

    with pytest.raises(FileNotFoundError):
        TrackingRegistry(missing_db, read_only=True)

    assert missing_db.exists() is False
    assert missing_db.parent.exists() is False


def test_read_only_mode_unmigrated_db_raises_migration_required_error(tmp_path: Path):
    """TrackingRegistry in read_only=True raises MigrationRequiredError without altering the DB."""
    unmigrated = tmp_path / "unmigrated.sqlite"
    conn = sqlite3.connect(str(unmigrated))
    conn.execute("CREATE TABLE some_table (id INT);")
    conn.commit()
    conn.close()

    with pytest.raises(MigrationRequiredError) as exc_info:
        TrackingRegistry(unmigrated, read_only=True)
    assert "requires migration" in str(exc_info.value)

    # Verify user_version is still 0
    conn = sqlite3.connect(str(unmigrated))
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 0
    conn.close()


def test_provenance_completeness_strict_rejection_rules(temp_db_path: Path):
    """historical_inferred, unverified, pending_restart, and unlinked activation reject complete provenance."""
    registry = TrackingRegistry(temp_db_path)

    change = ChangeRecord(
        change_id="chg_valid_01",
        implemented_at="2026-11-10T12:00:00Z",
        title="Valid Change",
        source_revision="0123456789abcdef0123456789abcdef01234567",
        dirty_worktree=0,
    )
    registry.record_change(change)

    exp = ExperimentRecord(
        experiment_id="exp_valid_01",
        change_id="chg_valid_01",
        created_at="2026-11-10T12:05:00Z",
        name="Valid Exp",
        parent_checkpoint_id="chk_parent",
    )
    registry.record_experiment(exp)

    # 1. Missing activation on checkpoint
    chk_no_act = CheckpointRecord(
        checkpoint_record_id="chk_rec_no_act",
        checkpoint_id="checkpoint-no-act",
        experiment_id="exp_valid_01",
        activation_id=None,
        parent_checkpoint_id="chk_parent",
    )
    registry.record_checkpoint(chk_no_act)
    lin_no_act = registry.get_history_for_checkpoint("checkpoint-no-act")
    assert lin_no_act.has_complete_provenance is False
    assert any("no direct activation link" in r for r in lin_no_act.missing_provenance_reasons)

    # 2. Historical inferred activation
    act_hist = ActivationRecord(
        activation_id="act_hist_01",
        experiment_id="exp_valid_01",
        change_id="chg_valid_01",
        status="historical_inferred",
    )
    registry.record_activation(act_hist)
    chk_hist = CheckpointRecord(
        checkpoint_record_id="chk_rec_hist",
        checkpoint_id="checkpoint-hist",
        experiment_id="exp_valid_01",
        activation_id="act_hist_01",
        parent_checkpoint_id="chk_parent",
    )
    registry.record_checkpoint(chk_hist)
    lin_hist = registry.get_history_for_checkpoint("checkpoint-hist")
    assert lin_hist.has_complete_provenance is False
    assert any("historical_inferred" in r for r in lin_hist.missing_provenance_reasons)

    # 3. Pending restart activation
    act_pending = ActivationRecord(
        activation_id="act_pending_02",
        experiment_id="exp_valid_01",
        change_id="chg_valid_01",
        status="pending_restart",
    )
    registry.record_activation(act_pending)
    chk_pending = CheckpointRecord(
        checkpoint_record_id="chk_rec_pending",
        checkpoint_id="checkpoint-pending-02",
        experiment_id="exp_valid_01",
        activation_id="act_pending_02",
        parent_checkpoint_id="chk_parent",
    )
    registry.record_checkpoint(chk_pending)
    lin_pending = registry.get_history_for_checkpoint("checkpoint-pending-02")
    assert lin_pending.has_complete_provenance is False
    assert any("pending_restart" in r for r in lin_pending.missing_provenance_reasons)

    # 4. Unverified activation
    act_unv = ActivationRecord(
        activation_id="act_unv_01",
        experiment_id="exp_valid_01",
        change_id="chg_valid_01",
        status="unverified",
    )
    registry.record_activation(act_unv)
    chk_unv = CheckpointRecord(
        checkpoint_record_id="chk_rec_unv",
        checkpoint_id="checkpoint-unv",
        experiment_id="exp_valid_01",
        activation_id="act_unv_01",
        parent_checkpoint_id="chk_parent",
    )
    registry.record_checkpoint(chk_unv)
    lin_unv = registry.get_history_for_checkpoint("checkpoint-unv")
    assert lin_unv.has_complete_provenance is False
    assert any("unverified" in r for r in lin_unv.missing_provenance_reasons)


def test_repair_historical_registry(temp_db_path: Path):
    """repair_historical_registry replaces 'historical_nov08' with NULL and updates audit trail."""
    registry = TrackingRegistry(temp_db_path)

    # Insert historical change with legacy values
    chg = ChangeRecord(
        change_id="chg_20261108_team_strategy_stray_2sheep",
        implemented_at="2026-11-08T18:00:00Z",
        title="Legacy Nov 8 Change",
        source_revision="historical_nov08",
        dirty_worktree=0,
        metadata={"phase": "phase_2"},
    )
    exp = ExperimentRecord(
        experiment_id="exp_20261108_stage34_recovery_pilot_v1",
        change_id="chg_20261108_team_strategy_stray_2sheep",
        created_at="2026-11-08T20:30:00Z",
        name="Phase 3 Pilot",
        training_budget_transitions=336663972,
    )
    chk = CheckpointRecord(
        checkpoint_record_id="chk_rec_task412_ep_040400",
        checkpoint_id="checkpoint-040400",
        experiment_id="exp_20261108_stage34_recovery_pilot_v1",
        policy_transitions=336684452,
        world_steps=112228150,
    )
    act = ActivationRecord(
        activation_id="act_20261108_stray_fix_eval",
        experiment_id="exp_20261108_stage34_recovery_pilot_v1",
        status="historical_inferred",
        activated_at_episode=40398,
        activated_at_world_step=112221324,
    )
    registry.record_change(chg)
    registry.record_experiment(exp)
    registry.record_checkpoint(chk)
    registry.record_activation(act)

    # Run repair
    repair_counts = repair_historical_registry(temp_db_path)
    assert repair_counts["changes_repaired"] >= 1
    assert repair_counts["experiments_repaired"] >= 1
    assert repair_counts["checkpoints_repaired"] >= 1
    assert repair_counts["activations_repaired"] >= 1

    # Verify repaired state in DB
    repaired_chg = registry.get_change("chg_20261108_team_strategy_stray_2sheep")
    assert repaired_chg.source_revision is None
    assert repaired_chg.dirty_worktree is None
    assert isinstance(repaired_chg.metadata, dict)
    assert repaired_chg.metadata["historical_source_label"] == "historical_nov08"
    assert repaired_chg.metadata["source_revision_status"] == "unrecorded_historical"
    assert repaired_chg.metadata["dirty_worktree_status"] == "unrecorded_historical_inferred_dirty"
    assert "repaired_at" in repaired_chg.metadata

    repaired_exp = registry.get_experiment("exp_20261108_stage34_recovery_pilot_v1")
    assert repaired_exp.training_budget_transitions == 20480
    assert isinstance(repaired_exp.metadata, dict)
    assert repaired_exp.metadata["cumulative_transitions_at_start"] == 336663972

    repaired_chk = registry.get_checkpoint("chk_rec_task412_ep_040400")
    assert repaired_chk.world_steps is None
    assert isinstance(repaired_chk.metadata, dict)
    assert repaired_chk.metadata["world_steps_status"] == "unmeasured"
    assert "local_estimated_world_steps" not in repaired_chk.metadata

    repaired_act = registry.get_activation("act_20261108_stray_fix_eval")
    assert repaired_act.activated_at_episode is None
    assert repaired_act.activated_at_world_step is None
    assert isinstance(repaired_act.metadata, dict)
    assert repaired_act.metadata["checkpoint_label_episode"] == 40398
    assert repaired_act.metadata["world_steps_status"] == "unmeasured"
    assert "local_estimated_world_steps" not in repaired_act.metadata
    assert "repair_audit_trail" in repaired_act.metadata
    assert len(repaired_act.metadata["repair_audit_trail"]) == 1
    assert repaired_act.metadata["repair_audit_trail"][0]["previous_activated_at_episode"] == 40398


def test_evidence_backed_historical_repair_per_record(tmp_path: Path):
    """Repair uses explicit evidence mapping, audits episode corrections, and preserves unrelated records."""
    db_file = tmp_path / "legacy_repair_test.sqlite"
    registry = TrackingRegistry(db_file)

    # Insert legacy records with blanket 392352
    act_phase2_legacy = ActivationRecord(
        activation_id="act_20261108_stray_fix_eval",
        status="historical_inferred",
        activated_at_episode=392352,  # blanket episode to be corrected to None
        activated_at_world_step=112221324,
        metadata={"checkpoint_label_episode": 40398},
    )
    act_phase5_legacy = ActivationRecord(
        activation_id="act_20261109_task412_launch",
        status="historical_inferred",
        activated_at_episode=392352,  # verified evidence cited from training-summary.json
        activated_at_world_step=112228150,
        metadata={"checkpoint_label_episode": 40400},  # incorrect label episode, should be 40398
    )
    act_unrelated = ActivationRecord(
        activation_id="act_runtime_live_run_999",
        status="active",
        activated_at_episode=12345,
        activated_at_world_step=None,
        metadata={"checkpoint_label_episode": 100},
    )

    registry.record_activation(act_phase2_legacy)
    registry.record_activation(act_phase5_legacy)
    registry.record_activation(act_unrelated)

    # Run repair
    repair_counts1 = repair_historical_registry(db_file)
    assert repair_counts1["activations_repaired"] == 2

    # Verify per-record results
    repaired_phase2 = registry.get_activation("act_20261108_stray_fix_eval")
    assert repaired_phase2.activated_at_episode is None
    assert repaired_phase2.activated_at_world_step is None
    assert repaired_phase2.metadata["checkpoint_label_episode"] == 40398
    assert "repair_audit_trail" in repaired_phase2.metadata
    assert repaired_phase2.metadata["repair_audit_trail"][-1]["previous_activated_at_episode"] == 392352

    repaired_phase5 = registry.get_activation("act_20261109_task412_launch")
    assert repaired_phase5.activated_at_episode == 392352
    assert repaired_phase5.activated_at_world_step is None
    assert repaired_phase5.metadata["checkpoint_label_episode"] == 40398

    # Verify unrelated record is completely untouched
    untouched = registry.get_activation("act_runtime_live_run_999")
    assert untouched.activated_at_episode == 12345
    assert untouched.status == "active"
    assert untouched.metadata.get("repair_audit_trail") is None

    # Idempotent: running repair again produces zero updates
    repair_counts2 = repair_historical_registry(db_file)
    assert repair_counts2["activations_repaired"] == 0

    repaired_phase2_second = registry.get_activation("act_20261108_stray_fix_eval")
    assert len(repaired_phase2_second.metadata["repair_audit_trail"]) == 1


def test_world_steps_unjustified_estimate_removal(tmp_path: Path):
    """Checkpoints without measured world steps store NULL and report format indicates unmeasured."""
    db_file = tmp_path / "world_steps_test.sqlite"
    registry = TrackingRegistry(db_file)

    # Checkpoint with no measured world steps and no estimate
    chk_unmeasured = CheckpointRecord(
        checkpoint_record_id="chk_unmeas_01",
        checkpoint_id="checkpoint-unmeasured",
        world_steps=None,
        metadata={"world_steps_status": "unmeasured"},
    )
    registry.record_checkpoint(chk_unmeasured)

    lin_unmeas = registry.get_history_for_checkpoint("checkpoint-unmeasured")
    rep_unmeas = format_checkpoint_history_report(lin_unmeas)
    assert "- **World Steps**: Unknown (unmeasured cumulative world steps)" in rep_unmeas

    # Checkpoint with measured world steps
    chk_meas = CheckpointRecord(
        checkpoint_record_id="chk_meas_01",
        checkpoint_id="checkpoint-measured",
        world_steps=50000,
        metadata={"world_steps_status": "measured"},
    )
    registry.record_checkpoint(chk_meas)

    lin_meas = registry.get_history_for_checkpoint("checkpoint-measured")
    rep_meas = format_checkpoint_history_report(lin_meas)
    assert "- **World Steps**: 50000" in rep_meas

    # Checkpoint without measured world steps never prints sprint-step variance or local interval estimate
    assert "sprint-step variance" not in rep_unmeas
    assert "local interval estimated" not in rep_unmeas


def test_read_only_and_safety_verification_with_special_characters_in_path(tmp_path: Path):
    """Paths with spaces, '#' delimiters, and non-ASCII unicode characters must escape safely."""
    special_dir = tmp_path / "test dir #1"
    special_dir.mkdir(parents=True, exist_ok=True)
    special_db = special_dir / "non_ascii_ünicode.sqlite"

    # Initialize writable DB with schema
    reg = TrackingRegistry(special_db, read_only=False)
    chg = ChangeRecord(
        change_id="chg_special_path",
        implemented_at="2026-11-10T12:00:00Z",
        title="Special Path Test",
    )
    reg.record_change(chg)

    # 1. Read-only mode connection with URI escaping
    ro_reg = TrackingRegistry(special_db, read_only=True)
    loaded_chg = ro_reg.get_change("chg_special_path")
    assert loaded_chg is not None
    assert loaded_chg.title == "Special Path Test"

    # 2. verify_migration_safety on path with special characters
    assert verify_migration_safety(special_db) is True


def test_assistance_mode_classification():
    """Verify strict assistance mode classification rules."""
    assert classify_assistance_mode("guided") == "guided"
    assert classify_assistance_mode("emergent") == "emergent"
    assert classify_assistance_mode(None) == "unknown"
    assert classify_assistance_mode("") == "unknown"
    assert classify_assistance_mode("autonomous") == "unknown"
    assert classify_assistance_mode("random") == "unknown"


def test_cumulative_transitions_produce_no_invented_local_estimate(tmp_path: Path):
    """Large cumulative transition count (e.g. 336,663,972) produces no invented local estimate in checkpoint record or report."""
    from sheepdog.checkpoints.store import CheckpointMetadata, CheckpointStore

    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir()
    db_file = tmp_path / "store_test.sqlite"
    reg = TrackingRegistry(db_file)
    store = CheckpointStore(chk_dir, tracking_registry=reg)

    meta = CheckpointMetadata(
        checkpoint_episode=40400,
        total_training_episodes=40400,
        policy_name="test_policy",
        seed=42,
        success_rate=0.9,
        average_completion_steps=250.0,
        timeout_rate=0.1,
        average_sheep_penned=5.8,
        average_reward=150.0,
        environment_config={"dogs": 3, "sheep": 6},
        reward_config={},
        checkpoint_id="checkpoint-040400",
        created_timestamp="2026-11-10T12:00:00Z",
        global_timestep=336663972,
    )
    chk_path = store.write(meta)
    assert chk_path.exists()

    rec = reg.get_checkpoint_by_id("checkpoint-040400")
    assert rec is not None
    assert rec.world_steps is None
    assert "local_estimated_world_steps" not in (rec.metadata or {})
    assert (rec.metadata or {}).get("world_steps_status") == "unmeasured"

    lin = reg.get_history_for_checkpoint("checkpoint-040400")
    rep = format_checkpoint_history_report(lin)
    assert "- **World Steps**: Unknown (unmeasured cumulative world steps)" in rep
    assert "sprint-step variance" not in rep
    assert "local interval estimated as" not in rep


def test_historical_repair_strictly_scopes_allowlists_and_handles_conflicts(tmp_path: Path):
    """Repair uses explicit allowlists: unlisted prefix records with genuine git hashes remain untouched, unexpected values report conflicts, and second run is idempotent with 0 modifications and no duplicate audit entries."""
    db_file = tmp_path / "scoped_repair_test.sqlite"
    reg = TrackingRegistry(db_file)

    # 1. Unlisted record with matching prefix and genuine git hash
    unlisted_chg = ChangeRecord(
        change_id="chg_20261108_unrelated_feature",
        implemented_at="2026-11-08T15:00:00Z",
        title="Unrelated Feature",
        source_revision="abcdef0123456789abcdef0123456789abcdef01",
        dirty_worktree=0,
    )
    # 2. Listed record needing repair
    listed_chg = ChangeRecord(
        change_id="chg_20261108_team_strategy_stray_2sheep",
        implemented_at="2026-11-08T18:00:00Z",
        title="Phase 2 Fix",
        source_revision="historical_nov08",
        dirty_worktree=0,
    )
    # 3. Listed record with unexpected legacy value
    conflict_exp = ExperimentRecord(
        experiment_id="exp_20261108_stage34_recovery_pilot_v1",
        created_at="2026-11-08T20:30:00Z",
        name="Phase 3 Pilot",
        training_budget_transitions=99999999,  # unexpected value!
    )
    # 4. Listed activation
    listed_act = ActivationRecord(
        activation_id="act_20261108_stray_fix_eval",
        status="historical_inferred",
        activated_at_episode=392352,
        activated_at_world_step=112221324,
    )
    reg.record_change(unlisted_chg)
    reg.record_change(listed_chg)
    reg.record_experiment(conflict_exp)
    reg.record_activation(listed_act)

    # Run repair pass 1
    counts1 = repair_historical_registry(db_file)
    assert counts1["changes_repaired"] == 1
    assert counts1["activations_repaired"] == 1
    assert len(counts1.get("conflicts", [])) >= 1

    # Verify unlisted record untouched
    unlisted_after = reg.get_change("chg_20261108_unrelated_feature")
    assert unlisted_after.source_revision == "abcdef0123456789abcdef0123456789abcdef01"
    assert unlisted_after.dirty_worktree == 0

    # Verify listed record repaired
    listed_after = reg.get_change("chg_20261108_team_strategy_stray_2sheep")
    assert listed_after.source_revision is None
    assert listed_after.dirty_worktree is None

    # Verify conflict record not overwritten
    conflict_after = reg.get_experiment("exp_20261108_stage34_recovery_pilot_v1")
    assert conflict_after.training_budget_transitions == 99999999

    # Verify listed activation repaired with audit trail
    act_after = reg.get_activation("act_20261108_stray_fix_eval")
    assert act_after.activated_at_episode is None
    audit_len_1 = len(act_after.metadata.get("repair_audit_trail", []))
    assert audit_len_1 == 1

    # Run repair pass 2 (idempotency check)
    counts2 = repair_historical_registry(db_file)
    assert counts2["changes_repaired"] == 0
    assert counts2["activations_repaired"] == 0

    act_after_2 = reg.get_activation("act_20261108_stray_fix_eval")
    audit_len_2 = len(act_after_2.metadata.get("repair_audit_trail", []))
    assert audit_len_2 == 1  # No duplicate audit entry!


def test_honest_report_formatting_values(tmp_path: Path):
    """Report formats dirty_worktree as Clean, Dirty, or Unknown, and implementation timestamps as Verified, Estimated, or Unknown."""
    db_file = tmp_path / "honest_report_test.sqlite"
    reg = TrackingRegistry(db_file)

    # 1. Clean worktree + verified timestamp
    chg_clean = ChangeRecord(
        change_id="chg_clean",
        implemented_at="2026-11-10T12:00:00Z",
        title="Clean Change",
        source_revision="1111111111111111111111111111111111111111",
        dirty_worktree=0,
        metadata={"implemented_at_status": "verified"},
    )
    exp_1 = ExperimentRecord(
        experiment_id="exp_1",
        change_id="chg_clean",
        created_at="2026-11-10T12:00:00Z",
        name="Clean Exp",
    )
    chk_1 = CheckpointRecord(
        checkpoint_record_id="chk_rec_1",
        checkpoint_id="chk-1",
        experiment_id="exp_1",
    )
    reg.record_change(chg_clean)
    reg.record_experiment(exp_1)
    reg.record_checkpoint(chk_1)

    rep1 = format_checkpoint_history_report(reg.get_history_for_checkpoint("chk-1"))
    assert "(Dirty Worktree: Clean)" in rep1
    assert "- **Implementation Timestamp (UTC)**: `2026-11-10T12:00:00Z` (Verified)" in rep1

    # 2. Dirty worktree + estimated timestamp
    chg_dirty = ChangeRecord(
        change_id="chg_dirty",
        implemented_at="2026-11-08T18:00:00Z",
        title="Dirty Change",
        source_revision="2222222222222222222222222222222222222222",
        dirty_worktree=1,
        metadata={"implemented_at_status": "estimated_from_eval_file_timestamp"},
    )
    exp_2 = ExperimentRecord(
        experiment_id="exp_2",
        change_id="chg_dirty",
        created_at="2026-11-08T18:00:00Z",
        name="Dirty Exp",
    )
    chk_2 = CheckpointRecord(
        checkpoint_record_id="chk_rec_2",
        checkpoint_id="chk-2",
        experiment_id="exp_2",
    )
    reg.record_change(chg_dirty)
    reg.record_experiment(exp_2)
    reg.record_checkpoint(chk_2)

    rep2 = format_checkpoint_history_report(reg.get_history_for_checkpoint("chk-2"))
    assert "(Dirty Worktree: Dirty)" in rep2
    assert "- **Implementation Timestamp (UTC)**: `2026-11-08T18:00:00Z` (Estimated)" in rep2

    # 3. Unknown dirty worktree + None timestamp
    chg_unknown = ChangeRecord(
        change_id="chg_unknown",
        implemented_at=None,
        title="Unknown Change",
        source_revision=None,
        dirty_worktree=None,
    )
    exp_3 = ExperimentRecord(
        experiment_id="exp_3",
        change_id="chg_unknown",
        created_at="2026-11-08T18:00:00Z",
        name="Unknown Exp",
    )
    chk_3 = CheckpointRecord(
        checkpoint_record_id="chk_rec_3",
        checkpoint_id="chk-3",
        experiment_id="exp_3",
    )
    from sheepdog.tracking.registry import CheckpointLineage
    lin_unknown = CheckpointLineage(
        checkpoint_id="chk-3",
        checkpoint=chk_3,
        experiment=exp_3,
        change=chg_unknown,
    )
    rep3 = format_checkpoint_history_report(lin_unknown)
    assert "(Dirty Worktree: Unknown)" in rep3
    assert "- **Implementation Timestamp (UTC)**: Unknown" in rep3

    # 4. Unknown verification timestamp
    chg_unver = ChangeRecord(
        change_id="chg_unver",
        implemented_at="2026-11-09T10:00:00Z",
        title="Unverified Timestamp Change",
        dirty_worktree=0,
    )
    exp_4 = ExperimentRecord(
        experiment_id="exp_4",
        change_id="chg_unver",
        created_at="2026-11-09T10:00:00Z",
        name="Unverified Exp",
    )
    chk_4 = CheckpointRecord(
        checkpoint_record_id="chk_rec_4",
        checkpoint_id="chk-4",
        experiment_id="exp_4",
    )
    reg.record_change(chg_unver)
    reg.record_experiment(exp_4)
    reg.record_checkpoint(chk_4)

    rep4 = format_checkpoint_history_report(reg.get_history_for_checkpoint("chk-4"))
    assert "- **Implementation Timestamp (UTC)**: `2026-11-09T10:00:00Z` (Unknown verification)" in rep4

