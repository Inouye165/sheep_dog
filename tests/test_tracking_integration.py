"""Integration tests for automated execution, checkpoint, and evaluation tracking."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from sheepdog.checkpoints.store import CheckpointMetadata, CheckpointStore
from sheepdog.config import EnvironmentConfig, LabConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.base import Policy
from sheepdog.tracking.execution import (
    RuntimeExecutionTracker,
    establish_process_startup_boundary,
    get_process_provenance,
    get_process_startup_provenance,
    is_process_startup_boundary_established,
    reset_process_startup_boundary,
    set_process_provenance,
)
from sheepdog.tracking.registry import TrackingRegistry, capture_current_git_provenance
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


class DummyPolicy(Policy):
    """Minimal policy for integration testing."""

    def __init__(self, name: str = "dummy_policy") -> None:
        self._name = name
        self.policy_version = 1
        self.trainer_type = "maskable_ppo"
        self.policy_type = "neural"
        self.policy_state_path = None

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, observation, valid_actions=None):
        return "wait"

    def select_actions(self, environment, deterministic: bool = True, **kwargs):
        return tuple("wait" for _ in range(environment.dog_count))


def test_automated_execution_checkpoint_evaluation_lifecycle():
    """Verify that execution tracker, checkpoint store, and evaluator automatically link records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "telemetry.sqlite"
        chk_dir = tmp_path / "checkpoints"
        eval_dir = tmp_path / "evaluations"

        # Initialize registry
        registry = TrackingRegistry(db_path)

        # 1. Startup execution capture
        establish_process_startup_boundary()
        exec_tracker = RuntimeExecutionTracker(registry=registry)
        run_id = "test_run_20261110_001"
        ctx = exec_tracker.initialize_or_resume(
            run_id=run_id,
            experiment_id="exp_test_integration",
            change_id="chg_test_integration",
            starting_episode=0,
            starting_policy_transition=0,
            starting_world_step=0,
            policy_state_path=None,
            config_hash="cfg_hash_test_123",
            is_resumed=False,
            target_stage=34,
            assistance_mode="guided",
        )

        assert ctx.experiment_id == "exp_test_integration"
        assert ctx.activation_id.startswith("act_test_run_")
        assert ctx.is_resumed is False

        # Verify activation record in database
        act_rec = registry.get_activation(ctx.activation_id)
        assert act_rec is not None
        assert act_rec.status == "active"
        assert act_rec.activation_method == "trainer_startup"
        assert act_rec.activated_at_episode == 0

        # 2. Checkpoint persistence with tracking linkage
        store = CheckpointStore(chk_dir, tracking_registry=registry)
        meta = CheckpointMetadata(
            checkpoint_episode=40400,
            total_training_episodes=40400,
            policy_name="dummy_policy",
            seed=42,
            success_rate=0.9,
            average_completion_steps=250.0,
            timeout_rate=0.1,
            average_sheep_penned=5.8,
            average_reward=150.0,
            environment_config={"dogs": 3, "sheep": 6, "width": 150, "height": 105},
            reward_config={},
            run_id=run_id,
            checkpoint_id="checkpoint-040400-test",
            parent_checkpoint_id="checkpoint-040399-test",
            global_timestep=120000,
            curriculum_stage=34,
            experiment_id=ctx.experiment_id,
            activation_id=ctx.activation_id,
            change_id=ctx.change_id,
        )

        chk_path = store.write(meta)
        assert chk_path.exists()

        chk_rec = registry.get_checkpoint_by_id("checkpoint-040400-test")
        assert chk_rec is not None
        assert chk_rec.experiment_id == ctx.experiment_id
        assert chk_rec.activation_id == ctx.activation_id
        assert chk_rec.parent_checkpoint_id == "checkpoint-040399-test"
        assert chk_rec.policy_transitions == 120000
        assert chk_rec.world_steps is None
        assert "local_estimated_world_steps" not in (chk_rec.metadata or {})
        assert chk_rec.metadata.get("world_steps_status") == "unmeasured"

        # 3. Evaluator evaluation with tracking linkage
        config = LabConfig()
        evaluator = Evaluator(config, eval_dir, tracking_registry=registry)
        dummy_pol = DummyPolicy()

        summary, json_p, csv_p = evaluator.evaluate(
            dummy_pol,
            seeds=(11, 23),
            checkpoint_episode=40400,
            capture_replays=False,
            capture_step_telemetry=False,
            evaluation_mode="confidence",
            run_id=run_id,
            checkpoint_id="checkpoint-040400-test",
            experiment_id=ctx.experiment_id,
            activation_id=ctx.activation_id,
            checkpoint_record_id=chk_rec.checkpoint_record_id,
        )

        assert json_p.exists()
        assert csv_p.exists()

        evals = registry.list_evaluations_for_checkpoint("checkpoint-040400-test")
        assert len(evals) == 1
        assert evals[0].experiment_id == ctx.experiment_id
        assert evals[0].checkpoint_record_id == chk_rec.checkpoint_record_id
        assert evals[0].seed_count == 2
        assert evals[0].success_rate is not None

        # 4. Verify full history lineage resolution
        lineage = registry.get_history_for_checkpoint("checkpoint-040400-test")
        assert lineage.checkpoint is not None
        assert lineage.experiment is not None
        assert lineage.activation is not None
        assert len(lineage.evaluations) == 1


def test_run_resume_preserves_experiment_and_records_new_activation():
    """Verify that resuming an existing run preserves experiment identity without duplicate records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "telemetry.sqlite"
        registry = TrackingRegistry(db_path)

        establish_process_startup_boundary()
        exec_tracker = RuntimeExecutionTracker(registry=registry)
        run_id = "test_run_20261110_resume"

        # Session 1: initial startup
        ctx1 = exec_tracker.initialize_or_resume(
            run_id=run_id,
            experiment_id="exp_resume_test",
            starting_episode=0,
            starting_policy_transition=0,
            starting_world_step=0,
            is_resumed=False,
        )
        assert ctx1.is_resumed is False

        # Session 2: resume same run at later episode
        ctx2 = exec_tracker.initialize_or_resume(
            run_id=run_id,
            experiment_id="exp_resume_test",
            starting_episode=50,
            starting_policy_transition=100000,
            starting_world_step=33333,
            is_resumed=True,
        )
        assert ctx2.is_resumed is True
        assert ctx2.experiment_id == ctx1.experiment_id
        assert ctx2.activation_id != ctx1.activation_id

        # Verify only one experiment record was created
        with registry._get_connection() as conn:
            exp_count = conn.execute(
                "SELECT COUNT(*) FROM registry_experiments WHERE experiment_id = 'exp_resume_test';"
            ).fetchone()[0]
            assert exp_count == 1

            # Verify two distinct activations exist
            act_rows = conn.execute(
                "SELECT activation_id, activation_method, activated_at_episode FROM registry_activations WHERE experiment_id = 'exp_resume_test' ORDER BY ROWID ASC;"
            ).fetchall()
            assert len(act_rows) == 2
            assert act_rows[0]["activation_method"] == "trainer_startup"
            assert act_rows[0]["activated_at_episode"] == 0
            assert act_rows[1]["activation_method"] == "trainer_resume"
            assert act_rows[1]["activated_at_episode"] == 50


def test_runtime_provenance_workspace_divergence_and_honest_claims():
    """Workspace divergence from process start marks activation unverified without false claims."""
    orig_proc = get_process_provenance()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_path = tmp_path / "telemetry.sqlite"
            registry = TrackingRegistry(db_path)
            tracker = RuntimeExecutionTracker(registry=registry)

            # 1. Simulate process initialization on an earlier commit that diverges from disk
            diverged_rev = "0000000000000000000000000000000000000000"
            set_process_provenance((diverged_rev, 0, None))

            ctx = tracker.initialize_or_resume(
                run_id="run_diverged_test",
                policy_state_path="artifacts/models/test-model.zip",
                weights_sha256=None,  # unverified weights SHA
                starting_episode=10,
                checkpoint_label_episode=2,
            )

            assert ctx.status == "unverified"
            assert ctx.verified_at == ""
            assert "Workspace diverged from process start" in (ctx.verification_notes or "")
            assert "weights SHA unverified" in (ctx.verification_notes or "")

            # Verify activation in DB
            act = registry.get_activation(ctx.activation_id)
            assert act is not None
            assert act.status == "unverified"
            assert act.verified_at is None
            assert act.verified_source_revision == diverged_rev
            assert ctx.source_revision == diverged_rev
            assert isinstance(act.metadata, dict)
            assert act.metadata["workspace_diverged"] is True
            assert act.metadata["checkpoint_label_episode"] == 2
            assert act.metadata["world_steps_status"] == "unmeasured"

            # 2. When weights_sha256 is explicitly provided
            ctx_hashed = tracker.initialize_or_resume(
                run_id="run_hashed_test",
                policy_state_path="artifacts/models/test-model.zip",
                weights_sha256="abc123def456",
            )
            assert "weights SHA-256: abc123def456" in (ctx_hashed.verification_notes or "")

    finally:
        set_process_provenance(orig_proc)


def test_maskable_ppo_trainer_tracking_wiring(tmp_path: Path):
    """MaskablePPOTrainer automatically tracks startup, resume, real episode counters, and guided assistance."""
    import sheepdog.training.episode_store as es

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    web_dir = tmp_path / "web"

    # Reset any cached global episode store
    if es._global_episode_store is not None:
        es._global_episode_store.close()
        es._global_episode_store = None

    try:
        config = LabConfig(
            training=TrainingConfig(
                output_dir=str(artifacts),
                web_export_dir=str(web_dir),
                total_timesteps=1000,
                checkpoint_episodes=(1,),
                observation_mode="guided",
            )
        )
        trainer = MaskablePPOTrainer(config, str(artifacts))

        # 1. Startup run with bounded loop (should_stop=True)
        trainer.train(should_stop=lambda: True)

        db_path = artifacts / "training-telemetry.sqlite"
        assert db_path.exists()

        reg = TrackingRegistry(db_path, read_only=True)
        with reg._get_connection() as conn:
            exps = conn.execute("SELECT * FROM registry_experiments;").fetchall()
            assert len(exps) == 1
            exp_row = dict(exps[0])
            assert exp_row["assistance_mode"] == "guided"

            acts = conn.execute("SELECT * FROM registry_activations;").fetchall()
            assert len(acts) == 1
            act_row = dict(acts[0])
            assert act_row["activation_method"] == "trainer_startup"
            assert act_row["activated_at_episode"] == 0
            assert act_row["activated_at_world_step"] is None

        # 2. Resumed run reusing existing experiment without duplicate experiment records
        models_dir = artifacts / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        dummy_model = models_dir / "checkpoint.zip"
        dummy_model.write_bytes(b"PK\x03\x04dummy")

        trainer._loaded_state["total_environment_episodes"] = 25000
        trainer._loaded_state["total_episodes_trained"] = 120
        trainer._loaded_state["total_timesteps"] = 150000
        trainer._loaded_state["policy_state_path"] = str(dummy_model)
        trainer._loaded_state["training_signature"] = trainer._training_signature()

        from unittest.mock import MagicMock, patch
        with patch.object(trainer.POLICY_CLASS, "load") as mock_load:
            mock_pol = MagicMock()
            mock_pol.name = "neural_ppo"
            mock_pol.trainer_type = "maskable_ppo"
            mock_pol.policy_type = "neural"
            mock_pol.observation_mode = "guided"
            mock_load.return_value = mock_pol

            trainer.train(should_stop=lambda: True)

        with reg._get_connection() as conn:
            exps_after = conn.execute("SELECT * FROM registry_experiments;").fetchall()
            assert len(exps_after) == 1  # Reused existing experiment, NO duplicates

            acts_after = conn.execute(
                "SELECT * FROM registry_activations ORDER BY ROWID ASC;"
            ).fetchall()
            assert len(acts_after) == 2
            resume_act = dict(acts_after[1])
            assert resume_act["activation_method"] == "trainer_resume"
            # Real cumulative environment episode counter, not checkpoint label!
            assert resume_act["activated_at_episode"] == 25000
            assert resume_act["activated_at_world_step"] is None

    finally:
        if es._global_episode_store is not None:
            es._global_episode_store.close()
            es._global_episode_store = None


def test_untracked_source_files_affect_patch_hash(tmp_path: Path):
    """Creating an untracked source file changes patch hash and dirty status; untracked excluded files do not."""
    # Initialize a temporary git repository
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), capture_output=True, check=True)

    # Initial tracked commit
    init_file = tmp_path / "README.md"
    init_file.write_text("# Repo", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), capture_output=True, check=True)

    # Clean repo state
    rev0, dirty0, patch0 = capture_current_git_provenance(cwd=tmp_path)
    assert rev0 is not None
    assert dirty0 == 0
    assert patch0 is None

    # 1. Add untracked excluded files (e.g. artifacts/telemetry.sqlite, model.zip)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "telemetry.sqlite").write_bytes(b"dummy_sqlite_data")
    (tmp_path / "model.zip").write_bytes(b"PK000dummy_zip")

    # Excluded files must NOT make the worktree dirty or change patch hash
    rev1, dirty1, patch1 = capture_current_git_provenance(cwd=tmp_path)
    assert dirty1 == 0
    assert patch1 is None

    # 2. Add an untracked source file (e.g. src/sheepdog/new_feature.py)
    src_dir = tmp_path / "src" / "sheepdog"
    src_dir.mkdir(parents=True)
    source_file = src_dir / "new_feature.py"
    source_file.write_text("print('hello world')", encoding="utf-8")

    # Untracked source file MUST mark worktree dirty and produce a patch hash
    rev2, dirty2, patch2 = capture_current_git_provenance(cwd=tmp_path)
    assert dirty2 == 1
    assert patch2 is not None

    # Modifying untracked source file changes patch hash
    source_file.write_text("print('modified source code')", encoding="utf-8")
    rev3, dirty3, patch3 = capture_current_git_provenance(cwd=tmp_path)
    assert dirty3 == 1
    assert patch3 != patch2


def test_process_startup_boundary_and_divergence(tmp_path: Path):
    """Edits made after startup boundary mark activation as unverified and retain process startup revision; boundary unestablished marks unverified; repeated calls retain process identity."""
    reset_process_startup_boundary()
    try:
        db_path = tmp_path / "divergence_test.sqlite"
        registry = TrackingRegistry(db_path)
        tracker = RuntimeExecutionTracker(registry=registry)

        # 1. Boundary not established -> status is unverified
        ctx_unest = tracker.initialize_or_resume(run_id="run_unest")
        assert ctx_unest.status == "unverified"
        assert "Process startup boundary was not explicitly established" in (ctx_unest.verification_notes or "")

        # 2. Establish boundary with simulated provenance
        proc_rev = "1111111111111111111111111111111111111111"
        set_process_provenance((proc_rev, 0, None))

        # Repeated training calls in one process retain process identity
        ctx1 = tracker.initialize_or_resume(run_id="run_proc_1")
        ctx2 = tracker.initialize_or_resume(run_id="run_proc_2")
        assert get_process_startup_provenance()[0] == proc_rev
        assert ctx1.status == "unverified"  # Because disk differs from proc_rev!
        assert ctx1.source_revision == proc_rev  # Retains process startup revision!
        assert ctx2.source_revision == proc_rev  # Retains process startup revision across repeated calls!

        act1 = registry.get_activation(ctx1.activation_id)
        assert act1.status == "unverified"
        assert act1.verified_source_revision == proc_rev
        assert act1.metadata["process_source_revision"] == proc_rev
        assert "Workspace diverged from process start" in (act1.verification_notes or "")

    finally:
        reset_process_startup_boundary()


def test_process_startup_boundary_immutability(tmp_path: Path):
    """Later calls to establish_process_startup_boundary cannot overwrite the startup identity."""
    reset_process_startup_boundary()
    try:
        repo1 = tmp_path / "repo1"
        repo1.mkdir()
        prov1 = establish_process_startup_boundary(repo_dir=repo1)
        assert is_process_startup_boundary_established()

        # Attempt second establishment with different repo/provenance
        repo2 = tmp_path / "repo2"
        repo2.mkdir()
        prov2 = establish_process_startup_boundary(repo_dir=repo2)
        assert prov2 == prov1
        assert get_process_startup_provenance() == prov1
    finally:
        reset_process_startup_boundary()


def test_train_without_prior_boundary_produces_unverified(tmp_path: Path):
    """train() without prior boundary establishment produces unverified status."""
    reset_process_startup_boundary()
    try:
        assert not is_process_startup_boundary_established()
        artifacts = tmp_path / "artifacts_no_bound"
        config = LabConfig(
            training=TrainingConfig(
                output_dir=str(artifacts),
                web_export_dir=str(tmp_path / "web"),
                total_timesteps=1000,
                checkpoint_episodes=(1,),
                observation_mode="guided",
            )
        )
        trainer = MaskablePPOTrainer(config, str(artifacts))

        # Run train with stop condition
        trainer.train(should_stop=lambda: True)

        reg = TrackingRegistry(artifacts / "training-telemetry.sqlite")
        with reg._get_connection() as conn:
            exps = conn.execute("SELECT * FROM registry_experiments;").fetchall()
            assert len(exps) == 1
            acts = conn.execute("SELECT * FROM registry_activations;").fetchall()
            assert len(acts) == 1
            assert acts[0]["status"] == "unverified"
            assert "Process startup boundary was not explicitly established" in (acts[0]["verification_notes"] or "")
    finally:
        reset_process_startup_boundary()


def test_server_startup_boundary_real_entry_path_divergence(tmp_path: Path, monkeypatch):
    """Server initialization establishes boundary; post-startup edit causes subsequent train() call to detect workspace divergence, retain process startup revision, and set status to 'unverified'."""
    reset_process_startup_boundary()
    try:
        startup_prov = ("2222222222222222222222222222222222222222", 0, None)
        diverged_prov = ("3333333333333333333333333333333333333333", 1, "abcd1234")

        curr_prov = startup_prov
        monkeypatch.setattr(
            "sheepdog.tracking.execution.capture_current_git_provenance",
            lambda cwd=None: curr_prov,
        )

        # Server startup establishes boundary
        from sheepdog.tracking.execution import establish_process_startup_boundary
        establish_process_startup_boundary()
        assert is_process_startup_boundary_established()
        assert get_process_startup_provenance() == startup_prov

        # Source edit occurs after server startup
        curr_prov = diverged_prov

        # Subsequent train() call
        artifacts = tmp_path / "artifacts_server_test"
        config = LabConfig(
            training=TrainingConfig(
                output_dir=str(artifacts),
                web_export_dir=str(tmp_path / "web"),
                total_timesteps=1000,
                checkpoint_episodes=(1,),
                observation_mode="guided",
            )
        )
        trainer = MaskablePPOTrainer(config, str(artifacts))
        trainer.train(should_stop=lambda: True)

        reg = TrackingRegistry(artifacts / "training-telemetry.sqlite")
        with reg._get_connection() as conn:
            exps = conn.execute("SELECT * FROM registry_experiments;").fetchall()
            assert len(exps) == 1
            acts = conn.execute("SELECT * FROM registry_activations;").fetchall()
            assert len(acts) == 1
            assert acts[0]["status"] == "unverified"
            assert acts[0]["verified_source_revision"] == startup_prov[0]
            assert "Workspace diverged from process start" in (acts[0]["verification_notes"] or "")
    finally:
        reset_process_startup_boundary()


def test_evaluator_handles_missing_foreign_keys_and_links_existing_parents(tmp_path: Path):
    """Evaluator.evaluate nulls out absent FK parents instead of failing, and links valid parents."""
    db_path = tmp_path / "telemetry.sqlite"
    eval_dir = tmp_path / "evaluations"
    registry = TrackingRegistry(db_path)

    config = LabConfig(
        environment=EnvironmentConfig(max_steps=5),
        training=TrainingConfig(observation_mode="guided"),
    )
    evaluator = Evaluator(config=config, output_root=eval_dir, tracking_registry=registry)
    policy = DummyPolicy()

    # 1. Evaluate with non-existent experiment_id and checkpoint_record_id
    summary_orphan, json_orphan, _ = evaluator.evaluate(
        policy,
        (101,),
        checkpoint_episode=10,
        capture_replays=False,
        capture_step_telemetry=False,
        evaluation_mode="stage_transition_baseline",
        run_id="run_missing",
        checkpoint_id="chk_missing_parent",
        experiment_id="exp_does_not_exist",
        checkpoint_record_id="chk_rec_does_not_exist",
    )
    assert json_orphan.exists()

    with registry._get_connection() as conn:
        rows = conn.execute("SELECT * FROM registry_evaluations;").fetchall()
        assert len(rows) == 1
        assert rows[0]["experiment_id"] is None
        assert rows[0]["checkpoint_record_id"] is None
        assert rows[0]["checkpoint_id"] == "chk_missing_parent"
        assert rows[0]["legacy_evaluation_id"] == summary_orphan.evaluation_id

    # 2. Create valid experiment and checkpoint records, then evaluate again
    from sheepdog.tracking.registry import CheckpointRecord, ExperimentRecord

    registry.record_experiment(
        ExperimentRecord(
            experiment_id="exp_valid",
            created_at="2026-11-18T00:00:00Z",
            name="Valid Experiment",
            assistance_mode="guided",
        )
    )
    registry.record_checkpoint(
        CheckpointRecord(
            checkpoint_record_id="chk_rec_valid",
            checkpoint_id="chk_valid",
            experiment_id="exp_valid",
        )
    )

    summary_linked, _, _ = evaluator.evaluate(
        policy,
        (102,),
        checkpoint_episode=20,
        capture_replays=False,
        capture_step_telemetry=False,
        evaluation_mode="confidence",
        run_id="run_valid",
        checkpoint_id="chk_valid",
        experiment_id="exp_valid",
        checkpoint_record_id="chk_rec_valid",
    )

    evals = registry.list_evaluations_for_checkpoint("chk_rec_valid")
    assert len(evals) == 1
    assert evals[0].experiment_id == "exp_valid"
    assert evals[0].checkpoint_record_id == "chk_rec_valid"
    assert evals[0].legacy_evaluation_id == summary_linked.evaluation_id


def test_quick_and_confidence_evaluations_recorded_after_checkpoint_write(tmp_path: Path):
    """Quick and confidence evaluations deferred during evaluate() are both linked after checkpoint_store.write()."""
    db_path = tmp_path / "telemetry.sqlite"
    chk_dir = tmp_path / "checkpoints"
    eval_dir = tmp_path / "evaluations"
    registry = TrackingRegistry(db_path)

    from sheepdog.tracking.registry import ExperimentRecord

    registry.record_experiment(
        ExperimentRecord(
            experiment_id="exp_deferred",
            created_at="2026-11-18T00:00:00Z",
            name="Deferred Eval Test",
            assistance_mode="guided",
        )
    )

    config = LabConfig(
        environment=EnvironmentConfig(max_steps=5),
        training=TrainingConfig(observation_mode="guided"),
    )
    evaluator = Evaluator(config=config, output_root=eval_dir, tracking_registry=registry)
    store = CheckpointStore(chk_dir, tracking_registry=registry)
    policy = DummyPolicy()

    chk_id = "chk_run_deferred_pv_1_ts_100"
    quick_summary, quick_json, _ = evaluator.evaluate(
        policy,
        (101,),
        checkpoint_episode=100,
        capture_replays=False,
        capture_step_telemetry=False,
        evaluation_mode="quick",
        run_id="run_deferred",
        checkpoint_id=chk_id,
        experiment_id="exp_deferred",
        checkpoint_record_id=f"chk_rec_{chk_id}",
        record_in_registry=False,
    )
    conf_summary, conf_json, _ = evaluator.evaluate(
        policy,
        (101, 102),
        checkpoint_episode=100,
        capture_replays=False,
        capture_step_telemetry=False,
        evaluation_mode="confidence",
        run_id="run_deferred",
        checkpoint_id=chk_id,
        experiment_id="exp_deferred",
        checkpoint_record_id=f"chk_rec_{chk_id}",
        record_in_registry=False,
    )

    # Nothing in registry_evaluations yet
    assert registry.list_evaluations_for_checkpoint(chk_id) == []

    # Now write checkpoint metadata (inserts chk_rec_{chk_id} into registry_checkpoints)
    meta = CheckpointMetadata(
        checkpoint_episode=100,
        total_training_episodes=100,
        policy_name="dummy_policy",
        seed=42,
        success_rate=conf_summary.success_rate,
        average_completion_steps=conf_summary.average_completion_steps,
        timeout_rate=conf_summary.timeout_rate,
        average_sheep_penned=conf_summary.average_sheep_penned,
        average_reward=conf_summary.average_reward,
        environment_config={"dogs": 3, "sheep": 6, "width": 150, "height": 105},
        reward_config={},
        run_id="run_deferred",
        checkpoint_id=chk_id,
        global_timestep=100,
        experiment_id="exp_deferred",
    )
    store.write(meta)

    assert evaluator.record_evaluation_in_registry(
        summary=quick_summary,
        json_path=quick_json,
        policy=policy,
        experiment_id="exp_deferred",
        checkpoint_record_id=f"chk_rec_{chk_id}",
    )
    assert evaluator.record_evaluation_in_registry(
        summary=conf_summary,
        json_path=conf_json,
        policy=policy,
        experiment_id="exp_deferred",
        checkpoint_record_id=f"chk_rec_{chk_id}",
    )

    recorded = registry.list_evaluations_for_checkpoint(chk_id)
    assert len(recorded) == 2
    assert {e.evaluation_mode for e in recorded} == {"quick", "confidence"}
    assert all(e.experiment_id == "exp_deferred" for e in recorded)
    assert all(e.checkpoint_record_id == f"chk_rec_{chk_id}" for e in recorded)


def test_record_checkpoint_upsert_preserves_referenced_evaluations(tmp_path: Path):
    """Updating an existing CheckpointRecord does not fail when registry_evaluations references it."""
    from sheepdog.tracking.registry import (
        CheckpointRecord,
        EvaluationRecordEntry,
        ExperimentRecord,
    )

    registry = TrackingRegistry(tmp_path / "telemetry.sqlite")
    registry.record_experiment(
        ExperimentRecord(
            experiment_id="exp_upsert",
            created_at="2026-11-18T00:00:00Z",
            name="Upsert Test",
            assistance_mode="guided",
        )
    )
    registry.record_checkpoint(
        CheckpointRecord(
            checkpoint_record_id="chk_rec_upsert",
            checkpoint_id="chk_upsert",
            experiment_id="exp_upsert",
            world_steps=100,
        )
    )
    registry.record_evaluation(
        EvaluationRecordEntry(
            registry_eval_id="eval_reg_upsert",
            evaluated_at="2026-11-18T00:01:00Z",
            experiment_id="exp_upsert",
            checkpoint_id="chk_upsert",
            checkpoint_record_id="chk_rec_upsert",
        )
    )

    # Updating the checkpoint must succeed under PRAGMA foreign_keys = ON
    registry.record_checkpoint(
        CheckpointRecord(
            checkpoint_record_id="chk_rec_upsert",
            checkpoint_id="chk_upsert",
            experiment_id="exp_upsert",
            world_steps=250,
        )
    )

    updated = registry.get_checkpoint("chk_rec_upsert")
    assert updated is not None
    assert updated.world_steps == 250
    evals = registry.list_evaluations_for_checkpoint("chk_rec_upsert")
    assert len(evals) == 1
    assert evals[0].registry_eval_id == "eval_reg_upsert"


