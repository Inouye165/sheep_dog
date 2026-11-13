"""Importer for historical trials into the experiment and change tracking registry."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheepdog.tracking.registry import (
    ActivationRecord,
    ChangeRecord,
    CheckpointRecord,
    ConclusionRecord,
    EvaluationRecordEntry,
    ExperimentRecord,
    TrackingRegistry,
)

logger = logging.getLogger(__name__)

HISTORICAL_CHANGE_IDS: tuple[str, ...] = (
    "chg_20261108_team_strategy_stray_2sheep",
    "chg_20261108_recovery_entropy_gamma_pilot",
    "chg_20261108_role_stagnation_decay_60",
    "chg_20261108_role_stagnation_decay_150",
    "chg_20261109_extended_training_batch_task412",
)

HISTORICAL_EXPERIMENT_IDS: tuple[str, ...] = (
    "exp_20261108_stage34_recovery_pilot_v1",
)

HISTORICAL_ACTIVATION_IDS: tuple[str, ...] = (
    "act_20261108_stray_fix_eval",
    "act_20261108_pilot_training",
    "act_20261108_phase4_eval",
    "act_20261108_phase4_1_eval",
    "act_20261109_task412_launch",
)

HISTORICAL_CHECKPOINT_IDS: tuple[str, ...] = (
    "checkpoint-040398",
    "checkpoint-040399",
    "checkpoint-040400-phase4",
    "checkpoint-040400-phase4_1",
    *(f"checkpoint-{ep:06d}" for ep in range(40400, 40448)),
)

HISTORICAL_CHECKPOINT_RECORD_IDS: tuple[str, ...] = (
    "chk_rec_checkpoint-040398",
    "chk_rec_checkpoint-040399",
    "chk_rec_checkpoint-040400-phase4",
    "chk_rec_checkpoint-040400-phase4_1",
    *(f"chk_rec_task412_ep_{ep:06d}" for ep in range(40400, 40448)),
)

HISTORICAL_PROVENANCE_NOTE = (
    "Historical import from handoff/eval JSON: SHA verified; activation time inferred from file mtime; "
    "running trainer did not verify code load; assistance mode: guided"
)

HISTORICAL_ACTIVATION_EVIDENCE: dict[str, dict[str, Any]] = {
    "act_20261108_stray_fix_eval": {
        "checkpoint_label_episode": 40398,
        "activated_at_episode": None,
        "activated_at_policy_transition": 336663972,
        "activated_at_world_step": None,
        "verification_notes": "Offline evaluation execution on checkpoint-040398; no training episodes run",
        "repair_reason": "Offline evaluation on checkpoint-040398; no training episodes run",
    },
    "act_20261108_pilot_training": {
        "checkpoint_label_episode": 40398,
        "activated_at_episode": None,
        "activated_at_policy_transition": 336663972,
        "activated_at_world_step": None,
        "verification_notes": (
            "Offline pilot training execution in artifacts/experiments/stage34_recovery_pilot_v1 "
            "started from parent checkpoint 40398; cumulative episodes unrecorded in pilot state; "
            "running trainer did not verify code load; assistance mode: guided"
        ),
        "repair_reason": "Cumulative environment episodes not recorded at pilot start in pilot state; store unknown",
    },
    "act_20261108_phase4_eval": {
        "checkpoint_label_episode": 40400,
        "activated_at_episode": None,
        "activated_at_policy_transition": 57344,
        "activated_at_world_step": None,
        "verification_notes": "Offline evaluation execution on pilot checkpoint 40400; no training episodes run",
        "repair_reason": "Offline evaluation on pilot checkpoint 40400; no training episodes run",
    },
    "act_20261108_phase4_1_eval": {
        "checkpoint_label_episode": 40400,
        "activated_at_episode": None,
        "activated_at_policy_transition": 57344,
        "activated_at_world_step": None,
        "verification_notes": "Offline evaluation execution on pilot checkpoint 40400; no training episodes run",
        "repair_reason": "Offline evaluation on pilot checkpoint 40400; no training episodes run",
    },
    "act_20261109_task412_launch": {
        "checkpoint_label_episode": 40398,
        "activated_at_episode": 392352,
        "activated_at_policy_transition": 336663972,
        "activated_at_world_step": None,
        "verification_notes": (
            "Continuous training run via scripts/launch_extended_training.py starting from "
            "checkpoint 40398 (episode 392352, transition 336663972 cited from "
            "artifacts/training-summary.json); running trainer did not verify code load; assistance mode: guided"
        ),
        "repair_reason": (
            "Starting at checkpoint 40398; activated_at_episode=392352 cited from "
            "artifacts/training-summary.json entry for checkpoint 40398"
        ),
    },
}


def _load_json_file(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        logger.warning("Historical file not found: %s", file_path)
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to parse JSON from %s: %s", file_path, exc)
        return None


def _extract_eval_entry(
    registry_eval_id: str,
    source_file: Path,
    experiment_id: str,
    checkpoint_id: str,
    checkpoint_record_id: str,
    provenance_notes: str = HISTORICAL_PROVENANCE_NOTE,
    assistance_mode: str = "guided",
) -> EvaluationRecordEntry | None:
    data = _load_json_file(source_file)
    if not data:
        return None

    records = data.get("records", [])
    seeds = [r.get("seed") for r in records if "seed" in r]
    per_seed = [
        {
            "seed": r.get("seed"),
            "success": r.get("success"),
            "steps": r.get("steps"),
            "reward": r.get("reward_total"),
            "sheep_penned": r.get("sheep_penned"),
            "stop_reason": r.get("stop_reason"),
        }
        for r in records
    ]
    replays = [r.get("replay_path") for r in records if r.get("replay_path")]

    return EvaluationRecordEntry(
        registry_eval_id=registry_eval_id,
        legacy_evaluation_id=data.get("evaluation_id"),
        source_file=str(source_file).replace("\\", "/"),
        experiment_id=experiment_id,
        checkpoint_id=checkpoint_id,
        checkpoint_record_id=checkpoint_record_id,
        evaluated_at=data.get("evaluation_timestamp") or "2026-11-08T00:00:00Z",
        evaluation_mode=data.get("evaluation_mode", "confidence"),
        assistance_mode=assistance_mode,
        model_path=data.get("policy_state_path"),
        model_sha256=None,
        environment_config_hash=data.get("environment_config_hash"),
        seed_set_id=data.get("evaluation_seed_set_id"),
        seed_count=data.get("evaluation_seed_count") or len(seeds),
        seeds=seeds,
        success_rate=data.get("success_rate"),
        average_reward=data.get("average_reward"),
        average_steps=data.get("average_completion_steps"),
        average_sheep_penned=data.get("average_sheep_penned"),
        timeout_rate=data.get("timeout_rate"),
        stopped_rate=data.get("stopped_rate"),
        per_seed_results=per_seed,
        diagnostics={"records_count": len(records)},
        replay_paths=replays,
        provenance_notes=provenance_notes,
        metadata={"imported": True},
    )


def import_historical_trials(
    registry_or_db: TrackingRegistry | sqlite3.Connection | str | Path,
    repo_root: str | Path = ".",
) -> dict[str, int]:
    """Import the November 8-9 historical trials into the registry idempotently.

    Returns a dictionary of counts for each record type imported.
    """
    root = Path(repo_root)
    if isinstance(registry_or_db, TrackingRegistry):
        registry = registry_or_db
    else:
        registry = TrackingRegistry(registry_or_db)

    counts = {
        "changes": 0,
        "experiments": 0,
        "activations": 0,
        "checkpoints": 0,
        "evaluations": 0,
        "conclusions": 0,
    }

    # =========================================================================
    # Phase 2 (Nov 8): Two-Sheep Stray Blind Spot Fix
    # =========================================================================
    chg_phase2 = ChangeRecord(
        change_id="chg_20261108_team_strategy_stray_2sheep",
        implemented_at="2026-11-08T18:00:00Z",
        title="TeamStrategy Two-Sheep Stray Blind Spot Fix",
        description="Fixed mathematical impossibility in TeamStrategy._stray_sheep when len(sheep) == 2 where flock_spread < max(5.0, flock_spread + 2.0) was never satisfied. Evaluates stray sheep farther from pen and designates collector role.",
        hypothesis="Explicitly handling two unpenned sheep by pairwise separation and assigning the farther sheep to a collector dog will prevent dog clustering and eliminate role thrashing.",
        rationale="When 2 sheep remain unpenned, midpoint geometry prevented collector role assignment, causing dogs to cluster or thrash roles rather than dispatching a collector.",
        category="strategy",
        source_revision=None,
        dirty_worktree=None,
        worktree_patch_hash=None,
        config_diff=None,
        metadata={
            "phase": "phase_2",
            "author": "implementation_agent",
            "historical_source_label": "historical_nov08",
            "source_revision_status": "unrecorded_historical",
            "dirty_worktree_status": "unrecorded_historical_inferred_dirty",
            "implemented_at_status": "estimated_from_eval_file_timestamp",
        },
    )
    registry.record_change(chg_phase2)
    counts["changes"] += 1

    exp_phase2 = ExperimentRecord(
        experiment_id="exp_20261108_team_strategy_stray_eval",
        change_id=chg_phase2.change_id,
        created_at="2026-11-08T18:30:00Z",
        name="Phase 2 TeamStrategy Stray Blind Spot Evaluation",
        hypothesis="TeamStrategy._stray_sheep 2-sheep distance handling eliminates role thrashing and reduces penalties on strays.",
        parent_checkpoint_id="checkpoint-040398",
        parent_checkpoint_path="artifacts/checkpoints/stage34/maskable-ppo-stage34-336663972.zip",
        parent_checkpoint_sha256="8cf211a57e28413a1ee1d6f1fc1a47fad26deb00844185248fa05f22ba0a3997",
        assistance_mode="guided",
        training_seeds=None,
        evaluation_seeds=[11, 23, 37, 41, 53, 59, 61, 67, 71, 73],
        training_budget_transitions=0,
        training_budget_world_steps=0,
        target_stage=34,
        environment_config=None,
        hyperparameters=None,
        status="completed",
        metadata={"phase": "phase_2"},
    )
    registry.record_experiment(exp_phase2)
    counts["experiments"] += 1

    act_phase2 = ActivationRecord(
        activation_id="act_20261108_stray_fix_eval",
        experiment_id=exp_phase2.experiment_id,
        change_id=chg_phase2.change_id,
        status="historical_inferred",
        activation_method="historical_import",
        verified_at=None,
        activated_at_episode=None,
        activated_at_checkpoint="checkpoint-040398",
        activated_at_policy_transition=336663972,
        activated_at_world_step=None,
        verified_source_revision=None,
        verification_notes="Offline evaluation execution on checkpoint-040398; no training episodes run",
        metadata={
            "phase": "phase_2",
            "checkpoint_label_episode": 40398,
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_activation(act_phase2)
    counts["activations"] += 1

    chk_phase2 = CheckpointRecord(
        checkpoint_record_id="chk_rec_checkpoint-040398",
        checkpoint_id="checkpoint-040398",
        experiment_id=exp_phase2.experiment_id,
        activation_id=act_phase2.activation_id,
        checkpoint_path="artifacts/checkpoints/stage34/maskable-ppo-stage34-336663972.zip",
        checkpoint_sha256="8cf211a57e28413a1ee1d6f1fc1a47fad26deb00844185248fa05f22ba0a3997",
        parent_checkpoint_id=None,
        global_environment_episode=392352,
        checkpoint_label_episode=40398,
        policy_version=10275,
        policy_transitions=336663972,
        world_steps=None,
        created_at="2026-11-08T18:00:00Z",
        assistance_mode="guided",
        curriculum_stage=34,
        metadata={
            "phase": "phase_2",
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_checkpoint(chk_phase2)
    counts["checkpoints"] += 1

    eval_phase2_path = root / "artifacts/evaluations/eval_maskable-ppo-stage34-336663972_confidence_ff85663fc223.json"
    eval_phase2 = _extract_eval_entry(
        registry_eval_id="eval_reg_20261108_stray_2sheep_eval",
        source_file=eval_phase2_path,
        experiment_id=exp_phase2.experiment_id,
        checkpoint_id="checkpoint-040398",
        checkpoint_record_id=chk_phase2.checkpoint_record_id,
    )
    if eval_phase2:
        registry.record_evaluation(eval_phase2)
        counts["evaluations"] += 1

    conc_phase2 = ConclusionRecord(
        conclusion_id="conc_20261108_stray_2sheep_fix",
        experiment_id=exp_phase2.experiment_id,
        recorded_at="2026-11-08T19:00:00Z",
        author="implementation_agent",
        status="hypothesis_supported",
        claim="Two-sheep stray blind spot fix maintains 80% success (8/10) with 0 regressions, reducing Seed 59 penalty by 55.8% (-3808.72 vs -8618.09) and eliminating role thrashing.",
        evidence_summary="Deterministic 10-seed evaluation on checkpoint 40398 showed zero regressions on 8 previously successful seeds, average reward improved by +1232.22 across 10 seeds, and Seed 59 role switches dropped from 60+ to 5.",
        matched_control_eval_id=None,
        candidate_eval_id="eval_reg_20261108_stray_2sheep_eval",
        limitations="Guided policy under TeamStrategy heuristic; Seed 59 still timed out at 2320 steps (0 penned) because policy weights trained under old blind spot required retraining to learn stray retrieval.",
        is_autonomous_claim=0,
        metadata={"phase": "phase_2"},
    )
    registry.record_conclusion(conc_phase2)
    counts["conclusions"] += 1

    # =========================================================================
    # Phase 3 (Nov 8): Stage 34 Recovery Pilot (stage34_recovery_entropy_gamma_v1)
    # =========================================================================
    chg_phase3 = ChangeRecord(
        change_id="chg_20261108_recovery_entropy_gamma_pilot",
        implemented_at="2026-11-08T20:00:00Z",
        title="Stage 34 Recovery Hyperparameter Pilot (Entropy 0.02, Gamma 0.997)",
        description="Bounded training pilot from checkpoint 40398 with entropy_coef=0.02 and gamma=0.997 to test if higher entropy exploration and longer discount horizon enable learning stray retrieval.",
        hypothesis="Pairing higher entropy exploration (0.02) and longer credit horizon (gamma 0.997) with Phase 2 collector heuristics will allow the policy to adapt to the corrected stray role and solve Seed 59.",
        rationale="Seed 59 timeout under fixed policy weights required exploration to discover collector guidance.",
        category="hyperparameter",
        source_revision=None,
        dirty_worktree=None,
        worktree_patch_hash=None,
        config_diff=json.dumps({"entropy_coef": {"before": 0.01, "after": 0.02}, "gamma": {"before": 0.99, "after": 0.997}}),
        metadata={
            "phase": "phase_3",
            "historical_source_label": "historical_nov08",
            "source_revision_status": "unrecorded_historical",
            "dirty_worktree_status": "unrecorded_historical_inferred_dirty",
            "implemented_at_status": "estimated_from_eval_file_timestamp",
        },
    )
    registry.record_change(chg_phase3)
    counts["changes"] += 1

    exp_phase3 = ExperimentRecord(
        experiment_id="exp_20261108_stage34_recovery_pilot_v1",
        change_id=chg_phase3.change_id,
        created_at="2026-11-08T20:30:00Z",
        name="Phase 3 Stage 34 Recovery Pilot (Entropy 0.02, Gamma 0.997)",
        hypothesis="Increasing entropy to 0.02 and gamma to 0.997 enables policy to learn stray retrieval on Seed 59.",
        parent_checkpoint_id="checkpoint-040398",
        parent_checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/parent/maskable-ppo-stage34-336663972.zip",
        parent_checkpoint_sha256="8cf211a57e28413a1ee1d6f1fc1a47fad26deb00844185248fa05f22ba0a3997",
        assistance_mode="guided",
        training_seeds=[101],
        evaluation_seeds=[11, 23, 37, 41, 53, 59, 61, 67, 71, 73],
        training_budget_transitions=20480,
        training_budget_world_steps=6827,
        target_stage=34,
        environment_config=None,
        hyperparameters={"entropy_coef": 0.02, "gamma": 0.997, "learning_rate": 5e-05},
        status="completed",
        metadata={
            "phase": "phase_3",
            "cumulative_transitions_at_start": 336663972,
            "cumulative_transitions_at_end": 336684452,
        },
    )
    registry.record_experiment(exp_phase3)
    counts["experiments"] += 1

    act_phase3 = ActivationRecord(
        activation_id="act_20261108_pilot_training",
        experiment_id=exp_phase3.experiment_id,
        change_id=chg_phase3.change_id,
        status="historical_inferred",
        activation_method="historical_import",
        verified_at=None,
        activated_at_episode=None,
        activated_at_checkpoint="checkpoint-040398",
        activated_at_policy_transition=336663972,
        activated_at_world_step=None,
        verified_source_revision=None,
        verification_notes=(
            "Offline pilot training execution in artifacts/experiments/stage34_recovery_pilot_v1 "
            "started from parent checkpoint 40398; cumulative episodes unrecorded in pilot state; "
            "running trainer did not verify code load; assistance mode: guided"
        ),
        metadata={
            "phase": "phase_3",
            "checkpoint_label_episode": 40398,
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_activation(act_phase3)
    counts["activations"] += 1

    chk_phase3 = CheckpointRecord(
        checkpoint_record_id="chk_rec_checkpoint-040399",
        checkpoint_id="checkpoint-040399",
        experiment_id=exp_phase3.experiment_id,
        activation_id=act_phase3.activation_id,
        checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip",
        checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        parent_checkpoint_id="checkpoint-040398",
        global_environment_episode=392353,
        checkpoint_label_episode=40399,
        policy_version=10276,
        policy_transitions=57344,
        world_steps=None,
        created_at="2026-11-08T22:30:00Z",
        assistance_mode="guided",
        curriculum_stage=34,
        metadata={
            "phase": "phase_3",
            "model_alias": "maskable-ppo-stage34-00057344",
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_checkpoint(chk_phase3)
    counts["checkpoints"] += 1

    eval_phase3_path = root / "artifacts/experiments/stage34_recovery_pilot_v1/evaluations/eval_maskable-ppo-00057344_confidence_ff85663fc223.json"
    eval_phase3 = _extract_eval_entry(
        registry_eval_id="eval_reg_20261108_stage34_pilot_v1",
        source_file=eval_phase3_path,
        experiment_id=exp_phase3.experiment_id,
        checkpoint_id="checkpoint-040399",
        checkpoint_record_id=chk_phase3.checkpoint_record_id,
    )
    if eval_phase3:
        registry.record_evaluation(eval_phase3)
        counts["evaluations"] += 1

    conc_phase3 = ConclusionRecord(
        conclusion_id="conc_20261108_stage34_pilot_v1",
        experiment_id=exp_phase3.experiment_id,
        recorded_at="2026-11-08T22:45:00Z",
        author="implementation_agent",
        status="inconclusive",
        claim="Stage 34 recovery pilot solved Seed 59 in 336 steps (+176.98 reward) and improved efficiency on Seeds 37/41/53, but caused regression on Seed 11 (timeout at 2320 steps).",
        evidence_summary="Seed 59 completely converted from failure (2320 steps timeout, -3808.72 reward) to full success in 336 steps (+176.98 reward). However, Seed 11 regressed to timeout (reward -6492.31) due to dog 2 sprint oscillation along wall.",
        matched_control_eval_id="eval_reg_20261108_stray_2sheep_eval",
        candidate_eval_id="eval_reg_20261108_stage34_pilot_v1",
        limitations="Guided policy; Seed 73 and Seed 11 suffered from persistent role-lock preventing re-flanking.",
        is_autonomous_claim=0,
        metadata={"phase": "phase_3"},
    )
    registry.record_conclusion(conc_phase3)
    counts["conclusions"] += 1

    # =========================================================================
    # Phase 4 (Nov 8): Role-Lock Stagnation Breaker (threshold 60)
    # =========================================================================
    chg_phase4 = ChangeRecord(
        change_id="chg_20261108_role_stagnation_decay_60",
        implemented_at="2026-11-08T22:50:00Z",
        title="TeamStrategy Role Stagnation Decay (Threshold 60) and Boundary Backoff",
        description="Implemented role-stagnation decay in TeamStrategy._role_assignment_cost (decay stickiness when steps_in_role > 60) and 3-cell boundary backoff in _collector_target when stray wall margin <= 3.",
        hypothesis="Decaying assignment stickiness for long role holds (>60 steps) will break multi-hundred-step role-locks, allowing closer or fresh dogs to take over and break corner/wall traps.",
        rationale="Seed 73 and Seed 11 were trapped in persistent role-locks with 1000 hold steps.",
        category="strategy",
        source_revision=None,
        dirty_worktree=None,
        worktree_patch_hash=None,
        config_diff=None,
        metadata={
            "phase": "phase_4",
            "historical_source_label": "historical_nov08",
            "source_revision_status": "unrecorded_historical",
            "dirty_worktree_status": "unrecorded_historical_inferred_dirty",
            "implemented_at_status": "estimated_from_eval_file_timestamp",
        },
    )
    registry.record_change(chg_phase4)
    counts["changes"] += 1

    exp_phase4 = ExperimentRecord(
        experiment_id="exp_20261108_phase4_stagnation_breaker",
        change_id=chg_phase4.change_id,
        created_at="2026-11-08T22:55:00Z",
        name="Phase 4 Role-Lock Stagnation Breaker Evaluation",
        hypothesis="Role-stagnation decay threshold 60 breaks persistent role-locks on Seed 73 and Seed 11.",
        parent_checkpoint_id="checkpoint-040399",
        parent_checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip",
        parent_checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        assistance_mode="guided",
        training_seeds=None,
        evaluation_seeds=[11, 23, 37, 41, 53, 59, 61, 67, 71, 73],
        training_budget_transitions=0,
        training_budget_world_steps=0,
        target_stage=34,
        environment_config=None,
        hyperparameters=None,
        status="completed",
        metadata={"phase": "phase_4"},
    )
    registry.record_experiment(exp_phase4)
    counts["experiments"] += 1

    act_phase4 = ActivationRecord(
        activation_id="act_20261108_phase4_eval",
        experiment_id=exp_phase4.experiment_id,
        change_id=chg_phase4.change_id,
        status="historical_inferred",
        activation_method="historical_import",
        verified_at=None,
        activated_at_episode=None,
        activated_at_checkpoint="checkpoint-040400-phase4",
        activated_at_policy_transition=57344,
        activated_at_world_step=None,
        verified_source_revision=None,
        verification_notes="Offline evaluation execution on pilot checkpoint 40400; no training episodes run",
        metadata={
            "phase": "phase_4",
            "checkpoint_label_episode": 40400,
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_activation(act_phase4)
    counts["activations"] += 1

    chk_phase4 = CheckpointRecord(
        checkpoint_record_id="chk_rec_checkpoint-040400-phase4",
        checkpoint_id="checkpoint-040400-phase4",
        experiment_id=exp_phase4.experiment_id,
        activation_id=act_phase4.activation_id,
        checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip",
        checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        parent_checkpoint_id="checkpoint-040399",
        global_environment_episode=392354,
        checkpoint_label_episode=40400,
        policy_version=10276,
        policy_transitions=57344,
        world_steps=None,
        created_at="2026-11-08T23:05:00Z",
        assistance_mode="guided",
        curriculum_stage=34,
        metadata={
            "phase": "phase_4",
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_checkpoint(chk_phase4)
    counts["checkpoints"] += 1

    eval_phase4_path = root / "artifacts/experiments/stage34_recovery_pilot_v1/evaluations_phase4/eval_pilot_chk00057344_confidence_ff85663fc223.json"
    eval_phase4 = _extract_eval_entry(
        registry_eval_id="eval_reg_20261108_phase4_thresh60",
        source_file=eval_phase4_path,
        experiment_id=exp_phase4.experiment_id,
        checkpoint_id="checkpoint-040400-phase4",
        checkpoint_record_id=chk_phase4.checkpoint_record_id,
    )
    if eval_phase4:
        registry.record_evaluation(eval_phase4)
        counts["evaluations"] += 1

    conc_phase4 = ConclusionRecord(
        conclusion_id="conc_20261108_phase4_thresh60",
        experiment_id=exp_phase4.experiment_id,
        recorded_at="2026-11-08T23:10:00Z",
        author="implementation_agent",
        status="hypothesis_supported",
        claim="Role-stagnation decay at threshold 60 achieved 90% success (9/10), recovering Seed 11 (793 steps, +5747.38 reward improvement) while keeping Seed 59 solved (925 steps).",
        evidence_summary="Seed 11 converted from timeout to full-flock success in 793 steps. Seed 73 role switches increased from 2 to 42, breaking role-lock saturation. Only Seed 73 failed.",
        matched_control_eval_id="eval_reg_20261108_stage34_pilot_v1",
        candidate_eval_id="eval_reg_20261108_phase4_thresh60",
        limitations="Guided policy; premature role decay at step 60 disrupted wide flanking sweeps on Seeds 41, 61, 67, increasing their completion times.",
        is_autonomous_claim=0,
        metadata={"phase": "phase_4"},
    )
    registry.record_conclusion(conc_phase4)
    counts["conclusions"] += 1

    # =========================================================================
    # Phase 4.1 (Nov 8): Role-Stagnation Decay Threshold Relaxation (150 steps)
    # =========================================================================
    chg_phase4_1 = ChangeRecord(
        change_id="chg_20261108_role_stagnation_decay_150",
        implemented_at="2026-11-08T23:12:00Z",
        title="TeamStrategy Role Stagnation Threshold Relaxation to 150 Steps",
        description="Relaxed role-stagnation decay onset threshold from 60 steps to 150 steps in TeamStrategy._role_assignment_cost, preserving boundary backoff.",
        hypothesis="Relaxing threshold to 150 steps allows dogs sufficient uninterrupted time for wide flanking sweeps while still preventing multi-hundred-step indefinite role-locks.",
        rationale="Threshold 60 caused premature role swaps during wide flanking maneuvers.",
        category="strategy",
        source_revision=None,
        dirty_worktree=None,
        worktree_patch_hash=None,
        config_diff=None,
        metadata={
            "phase": "phase_4_1",
            "historical_source_label": "historical_nov08",
            "source_revision_status": "unrecorded_historical",
            "dirty_worktree_status": "unrecorded_historical_inferred_dirty",
            "implemented_at_status": "estimated_from_eval_file_timestamp",
        },
    )
    registry.record_change(chg_phase4_1)
    counts["changes"] += 1

    exp_phase4_1 = ExperimentRecord(
        experiment_id="exp_20261108_phase4_1_threshold150",
        change_id=chg_phase4_1.change_id,
        created_at="2026-11-08T23:13:00Z",
        name="Phase 4.1 Role Stagnation Threshold Relaxation (150 Steps) Evaluation",
        hypothesis="Relaxing decay onset to 150 steps restores sweep speed and high rewards on Seeds 41, 61, 67 while preserving 90% success.",
        parent_checkpoint_id="checkpoint-040399",
        parent_checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip",
        parent_checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        assistance_mode="guided",
        training_seeds=None,
        evaluation_seeds=[11, 23, 37, 41, 53, 59, 61, 67, 71, 73],
        training_budget_transitions=0,
        training_budget_world_steps=0,
        target_stage=34,
        environment_config=None,
        hyperparameters=None,
        status="completed",
        metadata={"phase": "phase_4_1"},
    )
    registry.record_experiment(exp_phase4_1)
    counts["experiments"] += 1

    act_phase4_1 = ActivationRecord(
        activation_id="act_20261108_phase4_1_eval",
        experiment_id=exp_phase4_1.experiment_id,
        change_id=chg_phase4_1.change_id,
        status="historical_inferred",
        activation_method="historical_import",
        verified_at=None,
        activated_at_episode=None,
        activated_at_checkpoint="checkpoint-040400-phase4_1",
        activated_at_policy_transition=57344,
        activated_at_world_step=None,
        verified_source_revision=None,
        verification_notes="Offline evaluation execution on pilot checkpoint 40400; no training episodes run",
        metadata={
            "phase": "phase_4_1",
            "checkpoint_label_episode": 40400,
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_activation(act_phase4_1)
    counts["activations"] += 1

    chk_phase4_1 = CheckpointRecord(
        checkpoint_record_id="chk_rec_checkpoint-040400-phase4_1",
        checkpoint_id="checkpoint-040400-phase4_1",
        experiment_id=exp_phase4_1.experiment_id,
        activation_id=act_phase4_1.activation_id,
        checkpoint_path="artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip",
        checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        parent_checkpoint_id="checkpoint-040399",
        global_environment_episode=392354,
        checkpoint_label_episode=40400,
        policy_version=10276,
        policy_transitions=57344,
        world_steps=None,
        created_at="2026-11-08T23:14:00Z",
        assistance_mode="guided",
        curriculum_stage=34,
        metadata={
            "phase": "phase_4_1",
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_checkpoint(chk_phase4_1)
    counts["checkpoints"] += 1

    eval_phase4_1_path = root / "artifacts/experiments/stage34_recovery_pilot_v1/evaluations_phase4_1/eval_pilot_chk00057344_confidence_ff85663fc223.json"
    eval_phase4_1 = _extract_eval_entry(
        registry_eval_id="eval_reg_20261108_phase4_1_thresh150",
        source_file=eval_phase4_1_path,
        experiment_id=exp_phase4_1.experiment_id,
        checkpoint_id="checkpoint-040400-phase4_1",
        checkpoint_record_id=chk_phase4_1.checkpoint_record_id,
    )
    if eval_phase4_1:
        registry.record_evaluation(eval_phase4_1)
        counts["evaluations"] += 1

    conc_phase4_1 = ConclusionRecord(
        conclusion_id="conc_20261108_phase4_1_thresh150",
        experiment_id=exp_phase4_1.experiment_id,
        recorded_at="2026-11-08T23:20:00Z",
        author="implementation_agent",
        status="hypothesis_supported",
        claim="Threshold relaxation to 150 steps maintained 90% success (9/10) while restoring sweep speeds and high positive rewards on Seeds 41 (+282.50 reward, 201 steps), 61 (+347.63 reward, 245 steps), and 67 (+196.04 reward, 281 steps).",
        evidence_summary="Seed 41 completed in 201 steps (down from 322 in Phase 4), Seed 61 in 245 steps (down from 412 in Phase 4), Seed 67 in 281 steps (down from 501 in Phase 4). Seeds 11 and 59 remained fully solved.",
        matched_control_eval_id="eval_reg_20261108_phase4_thresh60",
        candidate_eval_id="eval_reg_20261108_phase4_1_thresh150",
        limitations="Guided policy under TeamStrategy heuristic; Seed 73 still fails.",
        is_autonomous_claim=0,
        metadata={"phase": "phase_4_1"},
    )
    registry.record_conclusion(conc_phase4_1)
    counts["conclusions"] += 1

    # =========================================================================
    # Phase 5 (Nov 9): Extended Stage 34 Training Batch (task-412)
    # =========================================================================
    chg_phase5 = ChangeRecord(
        change_id="chg_20261109_extended_training_batch_task412",
        implemented_at="2026-11-08T23:30:00Z",
        title="Continuous Extended Stage 34 Training Batch (task-412)",
        description="Continuous training launcher script (scripts/launch_extended_training.py) running 1,024,000 transitions across checkpoints 40400-40449 with gamma=0.997, entropy_coef=0.015, and periodic 10-seed formal confidence evaluations.",
        hypothesis="Extended training with gamma 0.997 and entropy 0.015 will consolidate recovery behaviors across varied spawn geometries.",
        rationale="Extended training was needed to consolidate the stray recovery behaviors discovered in Phase 3.",
        category="training",
        source_revision=None,
        dirty_worktree=None,
        worktree_patch_hash=None,
        config_diff=json.dumps({"gamma": 0.997, "entropy_coef": 0.015, "learning_rate": 5e-05}),
        metadata={
            "phase": "phase_5",
            "historical_source_label": "historical_nov08",
            "source_revision_status": "unrecorded_historical",
            "dirty_worktree_status": "unrecorded_historical_inferred_dirty",
            "implemented_at_status": "estimated_from_eval_file_timestamp",
        },
    )
    registry.record_change(chg_phase5)
    counts["changes"] += 1

    exp_phase5 = ExperimentRecord(
        experiment_id="exp_20261109_extended_stage34_task412",
        change_id=chg_phase5.change_id,
        created_at="2026-11-09T00:00:00Z",
        name="Extended Stage 34 Training Batch task-412",
        hypothesis="Extended training with gamma 0.997 and entropy 0.015 consolidates recovery behaviors across Stage 34 checkpoints 40400-40449.",
        parent_checkpoint_id="checkpoint-040399",
        parent_checkpoint_path="artifacts/checkpoints/stage34/maskable-ppo-stage34-pilot-00057344.zip",
        parent_checkpoint_sha256="903b20dc9ad1a84265a8f311e75507ec7fe130edf1f00a2312c3c866a9271f2f",
        assistance_mode="guided",
        training_seeds=None,
        evaluation_seeds=[11, 23, 37, 41, 53, 59, 61, 67, 71, 73],
        training_budget_transitions=1024000,
        training_budget_world_steps=341333,
        target_stage=34,
        environment_config=None,
        hyperparameters={"gamma": 0.997, "entropy_coef": 0.015, "learning_rate": 5e-05},
        status="completed",
        metadata={"phase": "phase_5"},
    )
    registry.record_experiment(exp_phase5)
    counts["experiments"] += 1

    act_phase5 = ActivationRecord(
        activation_id="act_20261109_task412_launch",
        experiment_id=exp_phase5.experiment_id,
        change_id=chg_phase5.change_id,
        status="historical_inferred",
        activation_method="historical_import",
        verified_at=None,
        activated_at_episode=392352,
        activated_at_checkpoint="checkpoint-040398",
        activated_at_policy_transition=336663972,
        activated_at_world_step=None,
        verified_source_revision=None,
        verification_notes=(
            "Continuous training run via scripts/launch_extended_training.py starting from "
            "checkpoint 40398 (episode 392352, transition 336663972 cited from "
            "artifacts/training-summary.json); running trainer did not verify code load; assistance mode: guided"
        ),
        metadata={
            "phase": "phase_5",
            "checkpoint_label_episode": 40398,
            "world_steps_status": "unmeasured",
        },
    )
    registry.record_activation(act_phase5)
    counts["activations"] += 1

    # Checkpoints 40400 through 40449 from training-summary.json
    summary_path = root / "artifacts/training-summary.json"
    summary_data = _load_json_file(summary_path) or {}
    checkpoints_list = summary_data.get("checkpoints", [])

    for c in checkpoints_list:
        ep = c.get("checkpoint_episode")
        if ep is None or not (40400 <= ep <= 40449):
            continue

        chk_rec_id = f"chk_rec_task412_ep_{ep:06d}"
        chk_id = f"checkpoint-{ep:06d}"
        policy_trans = c.get("global_timestep")

        chk_entry = CheckpointRecord(
            checkpoint_record_id=chk_rec_id,
            checkpoint_id=chk_id,
            experiment_id=exp_phase5.experiment_id,
            activation_id=act_phase5.activation_id,
            run_id=c.get("run_id"),
            checkpoint_path=c.get("policy_state_path"),
            checkpoint_sha256=None,
            parent_checkpoint_id=c.get("parent_checkpoint_id") or "checkpoint-040399",
            global_environment_episode=c.get("environment_episodes_total"),
            checkpoint_label_episode=ep,
            policy_version=c.get("policy_version"),
            policy_transitions=policy_trans,
            world_steps=None,
            created_at=c.get("created_timestamp") or c.get("recorded_at"),
            assistance_mode="guided",
            curriculum_stage=34,
            metadata={
                "phase": "phase_5",
                "raw_checkpoint": c.get("checkpoint"),
                "success_rate": c.get("success_rate"),
                "average_reward": c.get("average_reward"),
                "world_steps_status": "unmeasured",
            },
        )
        registry.record_checkpoint(chk_entry)
        counts["checkpoints"] += 1

        eval_filename = c.get("evaluation")
        if eval_filename:
            eval_path = root / "artifacts/evaluations" / eval_filename
            if eval_path.exists():
                eval_entry = _extract_eval_entry(
                    registry_eval_id=f"eval_reg_task412_ep_{ep:06d}",
                    source_file=eval_path,
                    experiment_id=exp_phase5.experiment_id,
                    checkpoint_id=chk_id,
                    checkpoint_record_id=chk_rec_id,
                )
                if eval_entry:
                    registry.record_evaluation(eval_entry)
                    counts["evaluations"] += 1

    conc_phase5 = ConclusionRecord(
        conclusion_id="conc_20261109_extended_stage34_task412",
        experiment_id=exp_phase5.experiment_id,
        recorded_at="2026-11-09T18:00:00Z",
        author="implementation_agent",
        status="hypothesis_supported",
        claim="Extended Stage 34 training batch task-412 successfully completed 1,024,000 transitions across checkpoints 40400-40449, exhibiting strong recovery behavior with multiple 100% full-success checkpoints and up to 90% pass rates.",
        evidence_summary="Checkpoints 40416, 40429, 40439 achieved 1.0 (100%) success rate on the 10 standard evaluation seeds. Final model saved at artifacts/models/maskable-ppo-337687972.zip.",
        matched_control_eval_id="eval_reg_20261108_phase4_1_thresh150",
        candidate_eval_id="eval_reg_task412_ep_040439",
        limitations="Guided policy under TeamStrategy heuristic; recovery behavior was trained and evaluated under scripted guidance.",
        is_autonomous_claim=0,
        metadata={"phase": "phase_5"},
    )
    registry.record_conclusion(conc_phase5)
    counts["conclusions"] += 1

    # Also apply repairs to any existing historical rows in the DB
    with registry._get_connection() as conn:
        repair_counts = repair_historical_registry(conn)
        counts["repaired"] = sum(v for k, v in repair_counts.items() if isinstance(v, int))

    logger.info("Import completed with counts: %s", counts)
    return counts


def repair_historical_registry(conn_or_path: sqlite3.Connection | str | Path) -> dict[str, Any]:
    """Idempotently repair historical November 8-9 records in SQLite database.

    Replaces unverified 'historical_nov08' with NULL, resets unrecorded dirty_worktree
    flags to NULL, updates audit metadata, and ensures accurate transition counters.
    Strictly scoped to explicit allowlists of known historical IDs.
    Does NOT modify training_episodes, delete records, or touch unlisted records.
    """
    import contextlib

    close_after = False
    if isinstance(conn_or_path, str | Path):
        conn = sqlite3.connect(str(conn_or_path), timeout=30.0)
        close_after = True
    else:
        conn = conn_or_path

    conn.row_factory = sqlite3.Row
    conflicts: list[str] = []
    counts: dict[str, Any] = {
        "changes_repaired": 0,
        "experiments_repaired": 0,
        "checkpoints_repaired": 0,
        "activations_repaired": 0,
        "conflicts": conflicts,
    }

    try:
        cur = conn.cursor()
        now_utc = datetime.now(UTC).isoformat()

        # 1. Repair registry_changes using strict allowlist
        placeholders_chg = ",".join("?" for _ in HISTORICAL_CHANGE_IDS)
        changes = cur.execute(
            f"SELECT change_id, source_revision, dirty_worktree, metadata FROM registry_changes WHERE change_id IN ({placeholders_chg});",
            HISTORICAL_CHANGE_IDS,
        ).fetchall()
        for row in changes:
            c_id = row["change_id"]
            s_rev = row["source_revision"]
            d_work = row["dirty_worktree"]
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if meta_raw else {}

            # Check expected legacy values before modifying
            if s_rev is not None and s_rev != "historical_nov08":
                conflicts.append(
                    f"Change {c_id}: unexpected source_revision {s_rev!r}; expected 'historical_nov08' or None. Skipped."
                )
                continue

            need_update = False
            if s_rev == "historical_nov08":
                s_rev = None
                meta["historical_source_label"] = "historical_nov08"
                meta["source_revision_status"] = "unrecorded_historical"
                need_update = True
            if d_work is not None:
                d_work = None
                meta["dirty_worktree_status"] = "unrecorded_historical_inferred_dirty"
                need_update = True
            if meta.get("implemented_at_status") != "estimated_from_eval_file_timestamp":
                meta["implemented_at_status"] = "estimated_from_eval_file_timestamp"
                need_update = True
            if need_update:
                meta["repaired_at"] = now_utc
                cur.execute(
                    "UPDATE registry_changes SET source_revision = ?, dirty_worktree = ?, metadata = ? WHERE change_id = ?;",
                    (s_rev, d_work, json.dumps(meta, sort_keys=True), c_id),
                )
                counts["changes_repaired"] += 1

        # 2. Repair registry_experiments using strict allowlist
        placeholders_exp = ",".join("?" for _ in HISTORICAL_EXPERIMENT_IDS)
        exps = cur.execute(
            f"SELECT experiment_id, training_budget_transitions, metadata FROM registry_experiments WHERE experiment_id IN ({placeholders_exp});",
            HISTORICAL_EXPERIMENT_IDS,
        ).fetchall()
        for row in exps:
            e_id = row["experiment_id"]
            budget = row["training_budget_transitions"]
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if meta_raw else {}

            # Check expected legacy values
            if budget not in (336663972, 20480):
                conflicts.append(
                    f"Experiment {e_id}: unexpected training_budget_transitions {budget}; expected 336663972 or 20480. Skipped."
                )
                continue

            if budget == 336663972 or "cumulative_transitions_at_start" not in meta:
                meta["cumulative_transitions_at_start"] = 336663972
                meta["cumulative_transitions_at_end"] = 336684452
                meta["repaired_at"] = now_utc
                cur.execute(
                    "UPDATE registry_experiments SET training_budget_transitions = ?, metadata = ? WHERE experiment_id = ?;",
                    (20480, json.dumps(meta, sort_keys=True), e_id),
                )
                counts["experiments_repaired"] += 1

        # 3. Repair registry_checkpoints metadata and remove unjustified world steps using strict allowlist
        placeholders_chk_id = ",".join("?" for _ in HISTORICAL_CHECKPOINT_IDS)
        placeholders_chk_rec = ",".join("?" for _ in HISTORICAL_CHECKPOINT_RECORD_IDS)
        chks = cur.execute(
            f"SELECT checkpoint_record_id, checkpoint_id, world_steps, metadata FROM registry_checkpoints "
            f"WHERE checkpoint_record_id IN ({placeholders_chk_rec}) OR checkpoint_id IN ({placeholders_chk_id});",
            (*HISTORICAL_CHECKPOINT_RECORD_IDS, *HISTORICAL_CHECKPOINT_IDS),
        ).fetchall()
        for row in chks:
            rec_id = row["checkpoint_record_id"]
            w_steps = row["world_steps"]
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if meta_raw else {}

            # Check for conflict: if explicitly marked as measured
            if meta.get("world_steps_status") == "measured":
                conflicts.append(
                    f"Checkpoint {rec_id}: has measured world_steps_status. Skipped."
                )
                continue

            need_update = False
            if w_steps is not None:
                w_steps = None
                need_update = True
            if "local_estimated_world_steps" in meta:
                meta.pop("local_estimated_world_steps", None)
                need_update = True
            if meta.get("world_steps_status") != "unmeasured":
                meta["world_steps_status"] = "unmeasured"
                need_update = True
            if need_update:
                meta["repaired_at"] = now_utc
                cur.execute(
                    "UPDATE registry_checkpoints SET world_steps = NULL, metadata = ? WHERE checkpoint_record_id = ?;",
                    (json.dumps(meta, sort_keys=True), rec_id),
                )
                counts["checkpoints_repaired"] += 1

        # 4. Repair registry_activations using explicit per-record evidence and strict allowlist
        placeholders_act = ",".join("?" for _ in HISTORICAL_ACTIVATION_IDS)
        acts = cur.execute(
            f"SELECT activation_id, activated_at_episode, activated_at_world_step, verification_notes, metadata "
            f"FROM registry_activations WHERE activation_id IN ({placeholders_act});",
            HISTORICAL_ACTIVATION_IDS,
        ).fetchall()
        for row in acts:
            a_id = row["activation_id"]
            ep = row["activated_at_episode"]
            w_step = row["activated_at_world_step"]
            notes = row["verification_notes"]
            meta_raw = row["metadata"]
            meta = json.loads(meta_raw) if meta_raw else {}

            ev = HISTORICAL_ACTIVATION_EVIDENCE.get(a_id)
            if not ev:
                conflicts.append(f"Activation {a_id}: not found in HISTORICAL_ACTIVATION_EVIDENCE. Skipped.")
                continue

            target_ep = ev["activated_at_episode"]
            target_label_ep = ev["checkpoint_label_episode"]
            target_notes = ev["verification_notes"]
            repair_reason = ev["repair_reason"]

            # Check expected legacy values
            if ep is not None and ep not in (392352, 40398, 40400, target_ep):
                conflicts.append(
                    f"Activation {a_id}: unexpected activated_at_episode {ep}. Skipped."
                )
                continue

            need_update = False
            if ep != target_ep:
                audit_trail = meta.setdefault("repair_audit_trail", [])
                audit_trail.append({
                    "previous_activated_at_episode": ep,
                    "repair_reason": repair_reason,
                    "repaired_at": now_utc,
                })
                ep = target_ep
                need_update = True

            if w_step is not None:
                w_step = None
                need_update = True

            if "local_estimated_world_steps" in meta:
                meta.pop("local_estimated_world_steps", None)
                need_update = True
            if "estimated_world_steps" in meta:
                meta.pop("estimated_world_steps", None)
                need_update = True

            if meta.get("checkpoint_label_episode") != target_label_ep:
                meta["checkpoint_label_episode"] = target_label_ep
                need_update = True

            if meta.get("world_steps_status") != "unmeasured":
                meta["world_steps_status"] = "unmeasured"
                need_update = True

            if notes != target_notes:
                notes = target_notes
                need_update = True

            if need_update:
                meta["repaired_at"] = now_utc
                cur.execute(
                    "UPDATE registry_activations SET activated_at_episode = ?, activated_at_world_step = ?, "
                    "verification_notes = ?, metadata = ? WHERE activation_id = ?;",
                    (ep, w_step, notes, json.dumps(meta, sort_keys=True), a_id),
                )
                counts["activations_repaired"] += 1

        conn.commit()
    finally:
        if close_after:
            with contextlib.suppress(Exception):
                conn.close()

    return counts
