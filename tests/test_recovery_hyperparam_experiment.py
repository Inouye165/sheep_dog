"""Focused tests for the controlled three-dog recovery hyperparameter experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sheepdog.config import LabConfig
from sheepdog.entities import Pen, Point
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.recovery_hyperparam_experiment import (
    DEFAULT_LATEST_PARENT_CHECKPOINT,
    DEFAULT_PAIRED_TRAINING_SEEDS,
    EXPERIMENT_ARMS,
    TRANSITIONS_PER_WORLD_STEP,
    build_arm_config,
    build_pilot_budget_and_stopping_proposal,
    build_transition_budget,
    collect_git_provenance,
    evaluate_experiment_promotion_guardrails,
    evaluate_policy_on_recovery_suite,
    get_development_recovery_scenarios,
    get_held_out_recovery_scenarios,
    inspect_runtime_observation_schemas,
    prepare_experiment_workspace,
    run_controlled_arm_seed,
    save_lab_config,
    sha256_file,
    verify_loaded_arm_models,
    verify_seed_and_scenario_isolation,
    verify_single_parameter_arm_diffs,
)
from sheepdog.training.maskable_ppo import MaskablePPOTrainer


def _create_tiny_parent_checkpoint(tmp_path: Path) -> tuple[Path, LabConfig]:
    """Create a tiny disposable parent checkpoint with gamma=0.99 and ent_coef=0.01."""
    cfg_dict = LabConfig().to_dict()
    cfg_dict["environment"]["max_steps"] = 20
    cfg_dict["training"]["rollout_steps"] = 32
    cfg_dict["training"]["batch_size"] = 16
    cfg_dict["training"]["gamma"] = 0.99
    cfg_dict["training"]["gae_lambda"] = 0.98
    cfg_dict["training"]["entropy_coef"] = 0.01
    cfg_dict["training"]["learning_rate"] = 6.5e-5
    cfg_dict["training"]["learning_rate_final"] = 6.5e-5
    cfg_dict["training"]["train_seed"] = 7
    cfg_dict["training"]["replay_export_on_new_best"] = False
    cfg_dict["training"]["replay_export_on_promotion"] = False
    cfg_dict["training"]["replay_export_on_final"] = False
    cfg_dict["training"]["replay_export_on_failed_diagnostic"] = False
    cfg_dict["training"]["backup_enabled"] = False
    cfg_dict["training"]["output_dir"] = str(tmp_path / "parent_init")
    cfg_dict["rewards"]["instincts"]["curriculum_stage"] = 34
    config = LabConfig.from_dict(cfg_dict)

    policy = NeuralPolicy.initialize(config)
    parent_zip = tmp_path / "parent_source" / "parent-stage34.zip"
    policy.save(parent_zip)
    venv = policy.model.get_env()
    if venv is not None:
        venv.close()
    return parent_zip, config


def test_transition_budget_and_three_dog_semantics() -> None:
    """Budgets must be expressed in both policy transitions and world steps (3 transitions = 1 world step)."""
    budget = build_transition_budget(
        policy_transitions_per_arm_seed=90_112,
        checkpoint_count=4,
        rollout_steps=2_048,
        num_arms=3,
        num_seeds=3,
    )
    assert TRANSITIONS_PER_WORLD_STEP == 3
    assert budget.policy_transitions_per_arm_seed == 90_112
    assert budget.whole_world_steps_per_arm_seed == 30_037
    assert budget.rollouts_per_arm_seed == 44
    assert budget.checkpoint_policy_transitions == (22_528, 45_056, 67_584, 90_112)
    assert budget.total_policy_transitions_all_arms_seeds == 90_112 * 9

    with pytest.raises(ValueError, match="multiple of rollout_steps"):
        build_transition_budget(
            policy_transitions_per_arm_seed=5_000,
            checkpoint_count=2,
            rollout_steps=2_048,
        )


def test_arms_differ_only_in_single_intended_setting(tmp_path: Path) -> None:
    """Arms A, B, and C must be identical except for entropy_coef (Arm B) and gamma (Arm C)."""
    base_cfg = LabConfig()
    budget = build_transition_budget()
    arm_configs = {
        arm.arm_id: build_arm_config(
            base_cfg,
            arm,
            arm_output_dir=tmp_path / arm.arm_id,
            train_seed=101,
            locked_learning_rate=6.5e-5,
            budget=budget,
        )
        for arm in EXPERIMENT_ARMS
    }
    diff_report = verify_single_parameter_arm_diffs(arm_configs)
    assert diff_report["verified"] is True
    assert diff_report["arm_b_vs_control_diffs"] == {
        "training.entropy_coef": {"control": 0.01, "arm_b": 0.02}
    }
    assert diff_report["arm_c_vs_control_diffs"] == {
        "training.gamma": {"control": 0.99, "arm_c": 0.997}
    }


def test_seed_and_scenario_isolation_covers_all_unpenned_counts() -> None:
    """Training, development, and held-out seeds/scenarios must be disjoint and cover 1..6 unpenned sheep."""
    report = verify_seed_and_scenario_isolation(DEFAULT_PAIRED_TRAINING_SEEDS)
    assert report["verified"] is True
    assert report["development_unpenned_counts_covered"] == [1, 2, 3, 4, 5, 6]
    assert report["held_out_unpenned_counts_covered"] == [1, 2, 3, 4, 5, 6]

    dev_scenarios = get_development_recovery_scenarios()
    held_out_scenarios = get_held_out_recovery_scenarios()
    assert set(dev_scenarios.keys()).isdisjoint(set(held_out_scenarios.keys()))


def test_resumed_model_applies_gamma_entropy_lr_seed_and_resets_rollout_buffer(
    tmp_path: Path,
) -> None:
    """Resumed SB3 models must update both model.gamma and rollout_buffer.gamma while preserving evaluation checkpoint_gamma."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    budget = build_transition_budget(
        policy_transitions_per_arm_seed=64,
        checkpoint_count=2,
        rollout_steps=32,
    )
    arm_c_spec = next(arm for arm in EXPERIMENT_ARMS if arm.arm_id == "arm_c_gamma_0997")
    arm_c_cfg = build_arm_config(
        base_cfg,
        arm_c_spec,
        arm_output_dir=tmp_path / "arm_c",
        train_seed=202,
        locked_learning_rate=6.5e-5,
        budget=budget,
    )

    loaded_policy = NeuralPolicy.load(parent_zip, arm_c_cfg)
    try:
        # Read-only load preserves original checkpoint gamma (0.99) for Evaluator mismatch checks
        assert loaded_policy.checkpoint_gamma == pytest.approx(0.99)
        before_params = loaded_policy.live_training_parameters()
        assert before_params["gamma"] == pytest.approx(0.99)
        assert before_params["rollout_buffer_gamma"] == pytest.approx(0.99)

        # Simulate stale rollout buffer position before configure_for_training
        loaded_policy.model.rollout_buffer.pos = 5

        after_params = loaded_policy.configure_for_training(
            arm_c_cfg,
            effective_learning_rate=arm_c_cfg.training.learning_rate,
            effective_entropy_coef=arm_c_cfg.training.entropy_coef,
            reset_rollout_buffer=True,
            set_seed=True,
        )
        assert after_params["gamma"] == pytest.approx(0.997)
        assert after_params["rollout_buffer_gamma"] == pytest.approx(0.997)
        assert after_params["ent_coef"] == pytest.approx(0.01)
        assert after_params["learning_rate"] == pytest.approx(6.5e-5)
        assert after_params["optimizer_learning_rates"] == [pytest.approx(6.5e-5)]
        assert after_params["seed"] == 202
        assert after_params["rollout_buffer_pos"] == 0
        assert after_params["rollout_buffer_full"] is False
        assert loaded_policy.checkpoint_gamma == pytest.approx(0.997)
    finally:
        venv = loaded_policy.model.get_env()
        if venv is not None:
            venv.close()


def test_disposable_smoke_run_isolates_outputs_and_preserves_parent_checkpoint(
    tmp_path: Path,
) -> None:
    """A tiny disposable smoke run must leave parent checkpoint untouched and propagate live settings in trainer."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)
    parent_sha_before = sha256_file(parent_zip)

    smoke_budget = build_transition_budget(
        policy_transitions_per_arm_seed=32,
        checkpoint_count=1,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "smoke_experiment"
    manifest = prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=smoke_budget,
        evaluation_seeds=(11,),
    )
    assert manifest["status"] == "prepared_not_started"

    loaded_verification = verify_loaded_arm_models(exp_root)
    assert loaded_verification["verified"] is True
    assert loaded_verification["arms"]["arm_a_control"]["after_configure_for_training"][
        "ent_coef"
    ] == pytest.approx(0.01)
    assert loaded_verification["arms"]["arm_b_entropy_002"][
        "after_configure_for_training"
    ]["ent_coef"] == pytest.approx(0.02)
    assert loaded_verification["arms"]["arm_c_gamma_0997"][
        "after_configure_for_training"
    ]["gamma"] == pytest.approx(0.997)
    assert loaded_verification["arms"]["arm_c_gamma_0997"][
        "after_configure_for_training"
    ]["rollout_buffer_gamma"] == pytest.approx(0.997)

    # Run a tiny 32-transition smoke rollout for Arm B and Arm C without full 2320-step eval
    for arm_id in ("arm_a_control", "arm_b_entropy_002", "arm_c_gamma_0997"):
        res = run_controlled_arm_seed(
            exp_root,
            arm_id=arm_id,
            train_seed=101,
            evaluate_development=False,
            evaluate_held_out=False,
        )
        assert res["completed"] is True
        assert res["trained_policy_transitions"] == 32
        assert res["live_parameter_verification_passed"] is True
        assert Path(res["final_checkpoint_path"]).is_relative_to(exp_root)

    # Parent checkpoint must be bitwise unchanged
    assert sha256_file(parent_zip) == parent_sha_before


def test_promotion_guardrails_reject_incomplete_and_reward_only_outcomes() -> None:
    """Incomplete runs, checksum drift, or reward-only gains without success/recovery improvement cannot pass."""
    fake_episodes_ctrl = [
        {"case_id": "c1", "full_flock_success": True},
        {"case_id": "c2", "full_flock_success": False},
    ]
    fake_episodes_cand_same_success = [
        {"case_id": "c1", "full_flock_success": True},
        {"case_id": "c2", "full_flock_success": False},
    ]

    def _make_run(
        *,
        completed: bool = True,
        transitions: int = 90_112,
        sr: float = 0.5,
        rep_steps: float = 40.0,
        rec_rate: float = 0.25,
        reward: float = 100.0,
        episodes: list[dict] = fake_episodes_ctrl,
    ) -> dict:
        return {
            "completed": completed,
            "trained_policy_transitions": transitions,
            "live_parameter_verification_passed": True,
            "evaluations": {
                "development": {
                    "decision_mode": "deterministic",
                    "full_flock_success_rate": sr,
                    "repeated_movement_steps_mean": rep_steps,
                    "recovery_outcome_rate": rec_rate,
                    "average_reward_diagnostic_only": reward,
                    "episodes": episodes,
                }
            },
        }

    # 1. Incomplete seed -> rejected as INVALID_OR_INCOMPLETE
    incomplete_results = {
        "arm_a_control": {101: _make_run()},
        "arm_b_entropy_002": {101: _make_run(completed=False)},
        "arm_c_gamma_0997": {101: _make_run()},
    }
    guard_incomplete = evaluate_experiment_promotion_guardrails(
        parent_sha256_before="abc",
        parent_sha256_after="abc",
        expected_seeds=(101,),
        expected_transitions_per_arm_seed=90_112,
        arm_seed_results=incomplete_results,
    )
    assert guard_incomplete["comparable"] is False
    assert guard_incomplete["status"] == "INVALID_OR_INCOMPLETE"

    # 2. Higher reward (+500) but NO increase in full-flock success rate -> NO_VERIFIED_IMPROVEMENT
    reward_only_results = {
        "arm_a_control": {101: _make_run(sr=0.5, reward=100.0)},
        "arm_b_entropy_002": {
            101: _make_run(
                sr=0.5,
                rep_steps=30.0,
                rec_rate=0.5,
                reward=600.0,
                episodes=fake_episodes_cand_same_success,
            )
        },
        "arm_c_gamma_0997": {101: _make_run(sr=0.5, reward=500.0)},
    }
    guard_reward_only = evaluate_experiment_promotion_guardrails(
        parent_sha256_before="abc",
        parent_sha256_after="abc",
        expected_seeds=(101,),
        expected_transitions_per_arm_seed=90_112,
        arm_seed_results=reward_only_results,
    )
    assert guard_reward_only["comparable"] is True
    assert (
        guard_reward_only["arm_comparisons"]["arm_b_entropy_002"]["verdict"]
        == "NO_VERIFIED_IMPROVEMENT"
    )


def test_real_stage34_parent_checkpoint_preparation_if_present(tmp_path: Path) -> None:
    """If the active Stage 34 checkpoint exists on disk, verify its checksum and settings without modifying it."""
    if not DEFAULT_LATEST_PARENT_CHECKPOINT.exists():
        pytest.skip("Stage 34 parent checkpoint not present in workspace.")

    sha_before = sha256_file(DEFAULT_LATEST_PARENT_CHECKPOINT)
    exp_root = tmp_path / "real_prep"
    manifest = prepare_experiment_workspace(exp_root)
    assert manifest["parent_checkpoint"]["verified_entropy_coef_matches_assumed_001"] is True
    assert manifest["parent_checkpoint"]["verified_gamma_matches_assumed_099"] is True
    assert sha256_file(DEFAULT_LATEST_PARENT_CHECKPOINT) == sha_before

    proposal = build_pilot_budget_and_stopping_proposal()
    assert proposal["budget"]["policy_transitions_per_arm_seed"] == 90_112


def test_git_provenance_handles_non_ascii_and_failed_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Git provenance must preserve raw diff bytes, handle non-ASCII content, capture untracked files, and raise on failure."""
    import hashlib
    import subprocess

    non_ascii_diff_bytes = b"diff --git a/test.txt b/test.txt\n--- a/test.txt\n+++ b/test.txt\n@@ -1 +1 @@\n-\xc3\xa9\xc3\xa0\n+\xff\xfe\n"
    expected_sha = hashlib.sha256(non_ascii_diff_bytes).hexdigest()

    # Create dummy untracked file
    untracked = tmp_path / "scratch_file.py"
    untracked.write_text("# untracked script\n", encoding="utf-8")
    untracked_sha = sha256_file(untracked)

    def mock_run_success(args, cwd=None, capture_output=True, check=False):  # noqa: ANN001, ARG001
        cmd = args[1:]
        if cmd == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"1234567890abcdef\n", stderr=b"")
        if cmd == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"main\n", stderr=b"")
        if cmd == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout=b" M tracked_file.py\n?? scratch_file.py\n", stderr=b"")
        if cmd == ["diff", "--stat"]:
            return subprocess.CompletedProcess(args, 0, stdout=b" 1 file changed\n", stderr=b"")
        if cmd == ["diff", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=non_ascii_diff_bytes, stderr=b"")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run_success)
    prov = collect_git_provenance(tmp_path)
    assert prov["head_commit"] == "1234567890abcdef"
    assert prov["branch"] == "main"
    assert prov["is_dirty"] is True
    assert prov["head_diff_sha256"] == expected_sha
    assert prov["raw_head_diff_bytes_count"] == len(non_ascii_diff_bytes)
    assert any(f["path"] == "scratch_file.py" and f["sha256"] == untracked_sha for f in prov["untracked_files"])

    def mock_run_failure(args, cwd=None, capture_output=True, check=False):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(args, 128, stdout=b"", stderr=b"fatal: not a git repository\n")

    monkeypatch.setattr(subprocess, "run", mock_run_failure)
    with pytest.raises(RuntimeError, match="Git command .* failed with exit code 128"):
        collect_git_provenance(tmp_path)


def test_runtime_observation_schemas_and_weight_shape_compatibility() -> None:
    """RoleAwareObservationBuilder has 54 features, Emergent has 50 features, creating an architectural shape mismatch."""
    schemas = inspect_runtime_observation_schemas()
    guided = schemas["guided"]
    emergent = schemas["emergent"]

    assert guided["builder_class"] == "RoleAwareObservationBuilder"
    assert guided["feature_count"] == 54
    assert emergent["builder_class"] == "EmergentObservationBuilder"
    assert emergent["feature_count"] == 50

    # Guided has role one-hot, target, distance_to_target, focus_sheep, stray_sheep
    assert "target_x" in guided["feature_names"]
    assert "distance_to_target" in guided["feature_names"]
    assert "role_rear_pressure" in guided["feature_names"]
    assert "focus_sheep_dx" in guided["feature_names"]

    # Emergent omits role/target and includes dog_id_slot and nearest/farthest unpenned sheep
    assert "target_x" not in emergent["feature_names"]
    assert "distance_to_target" not in emergent["feature_names"]
    assert "role_rear_pressure" not in emergent["feature_names"]
    assert "dog_id_slot_0" in emergent["feature_names"]
    assert "nearest_unpenned_dx" in emergent["feature_names"]
    assert "farthest_unpenned_dx" in emergent["feature_names"]

    # Check comparison
    comp = schemas["comparison"]
    assert comp["guided_count"] == 54
    assert comp["emergent_count"] == 50
    assert comp["shape_match"] is False
    assert comp["zero_shot_compatible"] is False

    # Check parent checkpoint layer weight shape if present
    if DEFAULT_LATEST_PARENT_CHECKPOINT.exists():
        loaded = NeuralPolicy.load(DEFAULT_LATEST_PARENT_CHECKPOINT, LabConfig())
        try:
            first_layer_w = loaded.model.policy.mlp_extractor.policy_net[0].weight
            assert list(first_layer_w.shape) == [256, 54]
        finally:
            venv = loaded.model.get_env()
            if venv is not None:
                venv.close()


def test_operational_run_ordering_enforced(tmp_path: Path) -> None:
    """Running Arm B or C before paired Control Arm A has completed on the same seed must raise RuntimeError."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)

    budget = build_transition_budget(
        policy_transitions_per_arm_seed=32,
        checkpoint_count=1,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "ordering_experiment"
    prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=budget,
        evaluation_seeds=(11,),
    )

    # Attempting to run Arm B without Arm A having run must fail
    with pytest.raises(RuntimeError, match="Operational run ordering violation"):
        run_controlled_arm_seed(
            exp_root,
            arm_id="arm_b_entropy_002",
            train_seed=101,
            evaluate_development=False,
            evaluate_held_out=False,
        )


def test_bounded_checkpoint_progression_and_consumption_recording(tmp_path: Path) -> None:
    """Trainer must progress across checkpoint boundaries, record actual integer consumption, and enforce stopping rules."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)

    # 2 checkpoints of 32 transitions each = 64 transitions
    budget = build_transition_budget(
        policy_transitions_per_arm_seed=64,
        checkpoint_count=2,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "progression_experiment"
    prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=budget,
        evaluation_seeds=(11,),
    )

    # 1. Run Control Arm A first (required by run ordering)
    res_a = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_a_control",
        train_seed=101,
        evaluate_development=False,
        evaluate_held_out=False,
    )
    assert res_a["completed"] is True
    assert res_a["stopped_early"] is False
    assert len(res_a["checkpoints_evaluated"]) == 2
    cons_a = res_a["consumption"]
    assert cons_a["actual_additional_policy_transitions"] >= 64
    assert isinstance(cons_a["actual_completed_world_steps"], int)
    assert cons_a["actual_completed_world_steps"] == cons_a["actual_additional_policy_transitions"] // 3
    assert cons_a["partial_team_steps"] == cons_a["actual_additional_policy_transitions"] % 3
    assert cons_a["stopped_early"] is False

    # 2. Run Candidate Arm B now that Control Arm A has run
    res_b = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_b_entropy_002",
        train_seed=101,
        evaluate_development=False,
        evaluate_held_out=False,
    )
    assert res_b["completed"] is True
    assert res_b["stopped_early"] is False
    assert len(res_b["checkpoints_evaluated"]) == 2
    cons_b = res_b["consumption"]
    assert cons_b["actual_completed_world_steps"] == cons_b["actual_additional_policy_transitions"] // 3


def test_ordinary_resume_behavior_protection(tmp_path: Path) -> None:
    """Proves: read-only load preserves checkpoint_gamma, overrides reach model/rollout/optimizer, Adam buffers preserved, parent file byte-identical, active artifacts untouched."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    parent_sha_orig = sha256_file(parent_zip)

    # 1. Read-only evaluation load preserves checkpoint_gamma
    eval_cfg = LabConfig()
    eval_cfg_dict = eval_cfg.to_dict()
    eval_cfg_dict["training"]["gamma"] = 0.95  # Different requested gamma
    eval_policy = NeuralPolicy.load(parent_zip, LabConfig.from_dict(eval_cfg_dict))
    try:
        assert eval_policy.checkpoint_gamma == pytest.approx(0.99)
        assert eval_policy.model.gamma == pytest.approx(0.99)
    finally:
        venv = eval_policy.model.get_env()
        if venv is not None:
            venv.close()

    # 2. Training overrides update model.gamma, rollout_buffer.gamma, ent_coef, and optimizer lr
    train_cfg_dict = eval_cfg.to_dict()
    train_cfg_dict["training"]["gamma"] = 0.997
    train_cfg_dict["training"]["entropy_coef"] = 0.02
    train_cfg_dict["training"]["learning_rate"] = 1.2e-4
    train_policy = NeuralPolicy.load(parent_zip, LabConfig.from_dict(train_cfg_dict))
    try:
        # Populate dummy optimizer momentum (exp_avg) and second-moment (exp_avg_sq) buffers
        opt = train_policy.model.policy.optimizer
        param = next(iter(train_policy.model.policy.parameters()))
        import torch
        opt.state[param]["exp_avg"] = torch.ones_like(param) * 0.123
        opt.state[param]["exp_avg_sq"] = torch.ones_like(param) * 0.456

        params_after = train_policy.configure_for_training(
            LabConfig.from_dict(train_cfg_dict),
            effective_learning_rate=1.2e-4,
            effective_entropy_coef=0.02,
            reset_rollout_buffer=True,
            set_seed=True,
        )
        assert params_after["gamma"] == pytest.approx(0.997)
        assert params_after["rollout_buffer_gamma"] == pytest.approx(0.997)
        assert params_after["ent_coef"] == pytest.approx(0.02)
        assert params_after["learning_rate"] == pytest.approx(1.2e-4)
        assert all(pytest.approx(lr) == 1.2e-4 for lr in params_after["optimizer_learning_rates"])

        # Optimizer momentum and second-moment buffers must be preserved
        assert torch.allclose(opt.state[param]["exp_avg"], torch.ones_like(param) * 0.123)
        assert torch.allclose(opt.state[param]["exp_avg_sq"], torch.ones_like(param) * 0.456)
    finally:
        venv = train_policy.model.get_env()
        if venv is not None:
            venv.close()

    # 3. Parent checkpoint file remains byte-identical
    assert sha256_file(parent_zip) == parent_sha_orig


def test_integration_pipeline_with_dev_eval_and_comparison(tmp_path: Path) -> None:
    """Exercise real training runner -> checkpoint -> dev eval -> temporal analysis -> persisted results -> comparison."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)
    parent_sha_before = sha256_file(parent_zip)

    # 2 checkpoints of 32 transitions each = 64 transitions
    tiny_budget = build_transition_budget(
        policy_transitions_per_arm_seed=64,
        checkpoint_count=2,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "integration_experiment"
    manifest = prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=tiny_budget,
        evaluation_seeds=(11,),
    )
    assert manifest["status"] == "prepared_not_started"

    # 1. Run Control Arm A with real development evaluation enabled (bounded step budget 6 per episode)
    res_a = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_a_control",
        train_seed=101,
        evaluate_development=True,
        evaluate_held_out=False,
        eval_max_steps_override=6,
    )
    assert res_a["completed"] is True
    assert res_a["trained_policy_transitions"] == 64
    assert len(res_a["checkpoints_evaluated"]) == 2

    # Check that development evaluation and temporal analysis ran and persisted real metrics
    chk1 = res_a["checkpoints_evaluated"][0]
    dev1 = chk1["development_evaluation"]
    assert dev1 is not None
    assert dev1["suite"] == "development"
    assert dev1["total_episodes"] > 0
    assert "repeated_movement_fraction_mean" in dev1
    assert "total_stall_intervals" in dev1
    assert "by_unpenned_sheep_count" in dev1
    # Check that case_geometry is recorded for every episode
    for ep in dev1["episodes"]:
        assert "case_geometry" in ep
        assert ep["case_geometry"]["width"] in (51, 80, 144)
        assert ep["case_geometry"]["height"] in (31, 60, 104)
        assert ep["case_geometry"]["arena_type"] in (
            "full_size_training_arena",
            "compact_diagnostic_arena",
            "procedural_arena",
            "scenario_arena",
        )

    # 2. Run Candidate Arm B and Arm C paired with Control Arm A
    res_b = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_b_entropy_002",
        train_seed=101,
        evaluate_development=True,
        evaluate_held_out=False,
        eval_max_steps_override=6,
    )
    assert res_b["completed"] is True
    assert res_b["trained_policy_transitions"] == 64

    res_c = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_c_gamma_0997",
        train_seed=101,
        evaluate_development=True,
        evaluate_held_out=False,
        eval_max_steps_override=6,
    )
    assert res_c["completed"] is True
    assert res_c["trained_policy_transitions"] == 64

    # 3. Evaluate promotion guardrails comparing Arms B and C against Control Arm A
    comparison = evaluate_experiment_promotion_guardrails(
        parent_sha256_before=parent_sha_before,
        parent_sha256_after=sha256_file(parent_zip),
        expected_seeds=(101,),
        expected_transitions_per_arm_seed=64,
        arm_seed_results={
            "arm_a_control": {101: res_a},
            "arm_b_entropy_002": {101: res_b},
            "arm_c_gamma_0997": {101: res_c},
        },
        suite="development",
    )
    assert comparison["status"] == "COMPARABLE"
    assert "arm_b_entropy_002" in comparison["arm_comparisons"]
    assert "arm_c_gamma_0997" in comparison["arm_comparisons"]
    b_comp = comparison["arm_comparisons"]["arm_b_entropy_002"]
    assert "control_mean_full_flock_success_rate" in b_comp
    assert "candidate_mean_full_flock_success_rate" in b_comp
    assert "delta_full_flock_success_rate" in b_comp
    assert "verdict" in b_comp

    # 4. Compare command CLI execution
    import subprocess
    cmd = [
        sys.executable,
        "scripts/run_recovery_hyperparam_experiment.py",
        "compare",
        "--experiment-root",
        str(exp_root),
        "--suite",
        "development",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0
    saved_cmp_file = exp_root / "comparison_development.json"
    assert saved_cmp_file.exists()
    cmp_json = json.loads(saved_cmp_file.read_text(encoding="utf-8"))
    assert cmp_json["status"] == "COMPARABLE"


def test_repeated_movement_measurement_on_known_cycling_sequence() -> None:
    """Proves: a known cycling sequence produces nonzero measured repeated movement."""
    from sheepdog.environment import (
        AgentSnapshot,
        EnvironmentSnapshot,
        RewardBreakdown,
        StepRecord,
    )

    pen = Pen(origin=Point(30, 20), width=10, height=10, opening="left")
    frames = []
    for step in range(1, 11):
        dog0_x = 10 if step % 2 == 1 else 11
        dogs = (
            AgentSnapshot(index=0, x=dog0_x, y=10, last_action="move_right" if step % 2 == 0 else "move_left"),
            AgentSnapshot(index=1, x=5, y=5, last_action="wait"),
            AgentSnapshot(index=2, x=6, y=5, last_action="wait"),
        )
        sheep = tuple(
            AgentSnapshot(index=i, x=20 + i, y=20, penned=False)
            for i in range(6)
        )
        snap = EnvironmentSnapshot(
            step=step,
            simulated_seconds=float(step),
            grid_width=51,
            grid_height=31,
            field_width=51,
            field_height=31,
            dogs=dogs,
            sheep=sheep,
            pen=pen,
            fence_cells=tuple((p.x, p.y) for p in pen.fence_cells()),
            penned_count=0,
            average_distance_to_pen=15.0,
            flock_spread=3.0,
            no_progress_steps=step,
            terminated=False,
            timeout=False,
            stopped=False,
            success=False,
            status="running",
        )
        frames.append(
            StepRecord(
                step=step,
                actions=("move_right", "wait", "wait"),
                snapshot=snap,
                reward=RewardBreakdown(),
            )
        )

    dog_positions_history: dict[int, list[Point]] = {d: [] for d in range(3)}
    ep_repeated_steps = 0
    for frame in frames:
        any_dog_in_loop = False
        for dog_snap in frame.snapshot.dogs:
            hist = dog_positions_history.setdefault(dog_snap.index, [])
            hist.append(Point(dog_snap.x, dog_snap.y))
            if len(hist) >= 4:
                tail = hist[-4:]
                if tail[-1] == tail[-3] and tail[-2] == tail[-4] and len({*tail}) <= 2:
                    any_dog_in_loop = True
        if any_dog_in_loop:
            ep_repeated_steps += 1

    assert ep_repeated_steps >= 6, f"Expected at least 6 repeated steps, got {ep_repeated_steps}"
    rep_fraction = ep_repeated_steps / len(frames)
    assert rep_fraction >= 0.60


def test_distinguish_missing_metrics_from_measured_zero() -> None:
    """Proves: empty denominators produce None (unavailable), strictly distinguished from measured 0.0."""
    cfg_dict = LabConfig().to_dict()
    cfg_dict["environment"]["max_steps"] = 10
    eval_cfg = LabConfig.from_dict(cfg_dict)
    policy = NeuralPolicy.initialize(eval_cfg)
    try:
        eval_result = evaluate_policy_on_recovery_suite(
            policy,
            eval_cfg,
            suite="development",
            deterministic=True,
            max_steps_override=2,
        )
        if eval_result["total_stall_intervals"] == 0:
            assert eval_result["recovery_outcome_rate"] is None
        else:
            assert isinstance(eval_result["recovery_outcome_rate"], float)

        for _k_str, bucket in eval_result["by_unpenned_sheep_count"].items():
            if bucket["stall_intervals"] == 0:
                assert bucket["recovery_rate"] is None
            else:
                assert isinstance(bucket["recovery_rate"], float)
    finally:
        venv = policy.model.get_env()
        if venv is not None:
            venv.close()


def test_enforced_safety_stopping_halts_training_and_skips_subsequent_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves: an enforced arm-level stopping condition halts the arm immediately, skipping subsequent checkpoints."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)

    # 4 planned checkpoints
    budget = build_transition_budget(
        policy_transitions_per_arm_seed=128,
        checkpoint_count=4,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "stop_experiment"
    prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=budget,
        evaluation_seeds=(11,),
    )

    original_train = MaskablePPOTrainer.train

    def patched_train(self, *args, **kwargs):
        orig_callback = kwargs.get("progress_callback")
        call_count = 0

        def wrapped_callback(payload):
            nonlocal call_count
            if payload.get("phase") in ("checkpoint", "checkpoint_saved"):
                call_count += 1
                if call_count >= 2:
                    payload["loss"] = float("nan")
                    payload["approx_kl"] = 0.01
                    payload["value_loss"] = 0.5
            if orig_callback:
                orig_callback(payload)

        kwargs["progress_callback"] = wrapped_callback
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(MaskablePPOTrainer, "train", patched_train)

    res = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_a_control",
        train_seed=101,
        evaluate_development=False,
        evaluate_held_out=False,
    )
    assert res["completed"] is False
    assert res["stopped_early"] is True
    assert "numerical_instability" in str(res["stop_reason"])
    # Crucial proof: checkpoint 3 and 4 were SKIPPED and never executed!
    assert len(res["checkpoints_evaluated"]) == 2


def test_missing_safety_diagnostics_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves: if required PPO safety diagnostics are absent at a checkpoint boundary after updates, the runner fails closed."""
    parent_zip, base_cfg = _create_tiny_parent_checkpoint(tmp_path)
    base_cfg_path = tmp_path / "base_config.json"
    save_lab_config(base_cfg, base_cfg_path)

    budget = build_transition_budget(
        policy_transitions_per_arm_seed=64,
        checkpoint_count=2,
        rollout_steps=32,
        num_seeds=1,
    )
    exp_root = tmp_path / "fail_closed_experiment"
    prepare_experiment_workspace(
        exp_root,
        parent_checkpoint_path=parent_zip,
        base_config_path=base_cfg_path,
        training_state_path=tmp_path / "nonexistent_state.json",
        checkpoint_metadata_path=None,
        paired_seeds=(101,),
        budget=budget,
        evaluation_seeds=(11,),
    )

    original_train = MaskablePPOTrainer.train

    def patched_train(self, *args, **kwargs):
        orig_callback = kwargs.get("progress_callback")

        def wrapped_callback(payload):
            if payload.get("phase") in ("checkpoint", "checkpoint_saved"):
                payload.pop("approx_kl", None)
                payload.pop("loss", None)
                payload.pop("value_loss", None)
                if "checkpoint_payload" in payload and isinstance(payload["checkpoint_payload"], dict):
                    payload["checkpoint_payload"].pop("approx_kl", None)
                    payload["checkpoint_payload"].pop("loss", None)
                    payload["checkpoint_payload"].pop("value_loss", None)
                chk_p = payload.get("checkpoint_path")
                if chk_p:
                    chk_path = Path(chk_p)
                    if not chk_path.is_absolute():
                        chk_path = Path.cwd() / chk_path
                    if chk_path.exists() and chk_path.suffix == ".json":
                        m = json.loads(chk_path.read_text(encoding="utf-8"))
                        m.pop("approx_kl", None)
                        m.pop("loss", None)
                        m.pop("value_loss", None)
                        chk_path.write_text(json.dumps(m), encoding="utf-8")
            if orig_callback:
                orig_callback(payload)

        kwargs["progress_callback"] = wrapped_callback
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(MaskablePPOTrainer, "train", patched_train)

    res = run_controlled_arm_seed(
        exp_root,
        arm_id="arm_a_control",
        train_seed=101,
        evaluate_development=False,
        evaluate_held_out=False,
    )
    assert res["completed"] is False
    assert res["stopped_early"] is True
    assert "missing_safety_diagnostics" in str(res["stop_reason"])
    assert len(res["checkpoints_evaluated"]) == 1
