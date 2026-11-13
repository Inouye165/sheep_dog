"""Report generator for checkpoint lineage and experiment history."""

from __future__ import annotations

import json

from sheepdog.tracking.registry import CheckpointLineage


def format_checkpoint_history_report(lineage: CheckpointLineage) -> str:
    """Format a comprehensive Markdown report answering the 5 core lineage questions."""
    lines: list[str] = []

    lines.append(f"# Checkpoint Provenance and History Report: `{lineage.checkpoint_id}`\n")

    # Provenance Summary Banner
    if lineage.has_complete_provenance:
        lines.append("> [!NOTE]")
        lines.append("> **Provenance Status**: COMPLETE. All lineage links (Change -> Experiment -> Activation -> Checkpoint) are verified.\n")
    elif lineage.activation and lineage.activation.status == "historical_inferred":
        lines.append("> [!WARNING]")
        lines.append("> **Provenance Status**: HISTORICAL_INFERRED (Unverified Runtime Activation).")
        lines.append("> Code changes were inferred from historical files; the executing trainer did not record a code-load verification handshake.")
        for reason in lineage.missing_provenance_reasons:
            lines.append(f"> - {reason}")
        lines.append("")
    elif lineage.activation and lineage.activation.status == "pending_restart":
        lines.append("> [!WARNING]")
        lines.append("> **Provenance Status**: PENDING_RESTART (Trainer Has Not Restarted).")
        lines.append("> Code changes are present in the workspace, but the running training process has not restarted to load new code.")
        for reason in lineage.missing_provenance_reasons:
            lines.append(f"> - {reason}")
        lines.append("")
    else:
        lines.append("> [!WARNING]")
        lines.append("> **Provenance Status**: INCOMPLETE OR UNVERIFIED.")
        for reason in lineage.missing_provenance_reasons:
            lines.append(f"> - {reason}")
        lines.append("")

    # If checkpoint itself is missing
    if lineage.checkpoint is None:
        lines.append("## Checkpoint Details")
        lines.append(f"No checkpoint record found for identifier `{lineage.checkpoint_id}`.\n")
        lines.append("### Answers to Core Provenance Questions:")
        lines.append("1. **What did we change, and why?** Unknown (checkpoint record not tracked in registry).")
        lines.append("2. **When was it implemented?** Unknown.")
        lines.append("3. **When did the running trainer actually start using it?** Unknown.")
        lines.append("4. **At which training episode, checkpoint, policy transition, and world step?** Unknown.")
        lines.append("5. **What results followed, and what evidence shows improvement or regression?** Unknown.\n")
        return "\n".join(lines)

    chk = lineage.checkpoint
    chg = lineage.change
    exp = lineage.experiment
    act = lineage.activation

    # 1. What did we change, and why?
    lines.append("## 1. What Did We Change, and Why?")
    if chg:
        lines.append(f"- **Change ID**: `{chg.change_id}`")
        lines.append(f"- **Title**: {chg.title}")
        lines.append(f"- **Category**: {chg.category or 'Unspecified'}")
        lines.append(f"- **Description**: {chg.description or 'None provided'}")
        lines.append(f"- **Hypothesis**: {chg.hypothesis or 'None recorded'}")
        lines.append(f"- **Rationale**: {chg.rationale or 'None recorded'}")
        if chg.dirty_worktree is None:
            dirty_str = "Unknown"
        elif chg.dirty_worktree in (1, True):
            dirty_str = "Dirty"
        elif chg.dirty_worktree in (0, False):
            dirty_str = "Clean"
        else:
            dirty_str = "Unknown"
        lines.append(f"- **Source Revision**: `{chg.source_revision or 'Unknown'}` (Dirty Worktree: {dirty_str})")
        if chg.config_diff:
            lines.append(f"- **Config Diff**: `{chg.config_diff}`")
    else:
        lines.append("*(No change record linked to this checkpoint)*")
    lines.append("")

    # 2. When was it implemented?
    lines.append("## 2. When Was It Implemented?")
    if chg:
        if chg.implemented_at is None:
            lines.append("- **Implementation Timestamp (UTC)**: Unknown")
        else:
            chg_meta = json.loads(chg.metadata) if isinstance(chg.metadata, str) else (chg.metadata or {})
            impl_status = str(chg_meta.get("implemented_at_status") or "")
            if impl_status == "estimated_from_eval_file_timestamp" or "estimated" in impl_status.lower():
                status_label = "(Estimated)"
            elif impl_status in ("verified", "git_commit") or "verified" in impl_status.lower():
                status_label = "(Verified)"
            else:
                status_label = "(Unknown verification)"
            lines.append(f"- **Implementation Timestamp (UTC)**: `{chg.implemented_at}` {status_label}")
    else:
        lines.append("*(Implementation time unknown - no linked change record)*")
    lines.append("")

    # 3. When did the running trainer actually start using it?
    lines.append("## 3. When Did the Running Trainer Actually Start Using It?")
    if act:
        lines.append(f"- **Activation ID**: `{act.activation_id}`")
        lines.append(f"- **Runtime Activation Status**: `{act.status}`")
        lines.append(f"- **Activation Method**: `{act.activation_method or 'Unspecified'}`")
        if act.verified_at:
            lines.append(f"- **Verified Active At**: `{act.verified_at}`")
        else:
            lines.append("- **Verified Active At**: `Unverified / Pending`")
        if act.activated_at_episode is not None:
            lines.append(f"- **Activated At Global Environment Episode**: {act.activated_at_episode}")
        act_meta = json.loads(act.metadata) if isinstance(act.metadata, str) else (act.metadata or {})
        label_ep = act_meta.get("checkpoint_label_episode")
        if label_ep is not None:
            lines.append(f"- **Activated At Checkpoint Label Episode**: {label_ep}")
        if act.status == "pending_restart":
            lines.append("> [!IMPORTANT]")
            lines.append("> **Runtime Notice**: The change is implemented on disk, but the running training server process has not restarted. It is NOT yet active in the current trainer.")
        elif act.status == "historical_inferred":
            lines.append("- **Inference Disclosure**: Activation status is inferred from historical trial records/file timestamps; the running trainer did not record a code-load handshake.")
        elif act.status == "unverified":
            lines.append("> [!WARNING]")
            lines.append("> **Unverified Runtime Notice**: Workspace diverged from process start or loaded code identity could not be verified.")
        if act.verification_notes:
            lines.append(f"- **Verification Notes**: {act.verification_notes}")
    else:
        lines.append("*(Runtime activation unrecorded or unlinked)*")
    lines.append("")

    # 4. At which training episode, checkpoint, policy transition, and world step?
    lines.append("## 4. Episode, Checkpoint, Transition, and Step Counts")
    lines.append(f"- **Checkpoint Record ID**: `{chk.checkpoint_record_id}`")
    lines.append(f"- **Checkpoint Identifier**: `{chk.checkpoint_id}`")
    lines.append(f"- **Checkpoint Label Episode**: {chk.checkpoint_label_episode if chk.checkpoint_label_episode is not None else 'Unknown'}")
    lines.append(f"- **Global Environment Episode Count**: {chk.global_environment_episode if chk.global_environment_episode is not None else 'Unknown'}")
    lines.append(f"- **Policy Version**: {chk.policy_version if chk.policy_version is not None else 'Unknown'}")
    lines.append(f"- **Policy Transitions (PPO Timesteps)**: {chk.policy_transitions if chk.policy_transitions is not None else 'Unknown'}")

    if chk.world_steps is not None:
        lines.append(f"- **World Steps**: {chk.world_steps}")
    else:
        lines.append("- **World Steps**: Unknown (unmeasured cumulative world steps)")

    lines.append(f"- **Curriculum Stage**: {chk.curriculum_stage if chk.curriculum_stage is not None else 'Unknown'}")
    lines.append(f"- **Assistance Mode**: `{chk.assistance_mode or 'Unknown'}`")
    if chk.parent_checkpoint_id:
        lines.append(f"- **Parent Checkpoint**: `{chk.parent_checkpoint_id}`")
    if chk.checkpoint_path:
        lines.append(f"- **Checkpoint Path**: `{chk.checkpoint_path}`")
    if chk.checkpoint_sha256:
        lines.append(f"- **Checkpoint SHA-256**: `{chk.checkpoint_sha256}`")
    lines.append("")

    # 5. What results followed, and what evidence shows improvement or regression?
    lines.append("## 5. What Results Followed, and What Evidence Shows Improvement or Regression?")
    if exp:
        lines.append(f"- **Experiment**: `{exp.experiment_id}` ({exp.name})")
        lines.append(f"- **Experiment Assistance Mode**: `{exp.assistance_mode}`")
        if exp.training_budget_transitions:
            lines.append(f"- **Training Budget**: {exp.training_budget_transitions:,} transitions ({exp.training_budget_world_steps:,} world steps)")

    if lineage.evaluations:
        lines.append("\n### Linked Formal Evaluations:")
        for idx, ev in enumerate(lineage.evaluations, 1):
            lines.append(f"#### Evaluation {idx}: `{ev.registry_eval_id}`")
            if ev.legacy_evaluation_id:
                lines.append(f"- **Legacy Evaluation ID**: `{ev.legacy_evaluation_id}`")
            lines.append(f"- **Evaluated At (UTC)**: `{ev.evaluated_at}`")
            lines.append(f"- **Evaluation Mode**: `{ev.evaluation_mode or 'standard'}`")
            lines.append(f"- **Assistance Mode**: `{ev.assistance_mode or 'guided'}`")
            if ev.success_rate is not None:
                lines.append(f"- **Full-Flock Success Rate**: {ev.success_rate * 100:.1f}% ({int(round(ev.success_rate * (ev.seed_count or 10)))}/{ev.seed_count or 10})")
            if ev.average_reward is not None:
                lines.append(f"- **Average Total Reward**: {ev.average_reward:.2f}")
            if ev.average_steps is not None:
                lines.append(f"- **Average Completion Steps**: {ev.average_steps:.1f}")
            if ev.average_sheep_penned is not None:
                lines.append(f"- **Average Sheep Penned**: {ev.average_sheep_penned:.2f} / 6")
            if ev.timeout_rate is not None:
                lines.append(f"- **Timeout Rate**: {ev.timeout_rate * 100:.1f}%")
            if ev.stopped_rate is not None:
                lines.append(f"- **Stopped (No-Progress) Rate**: {ev.stopped_rate * 100:.1f}%")
            if ev.provenance_notes:
                lines.append(f"- **Provenance Notes**: *{ev.provenance_notes}*")
            if ev.source_file:
                lines.append(f"- **Source File**: `{ev.source_file}`")
            if ev.per_seed_results and isinstance(ev.per_seed_results, list):
                lines.append("- **Per-Seed Breakdown**:")
                lines.append("  | Seed | Success | Steps | Reward | Penned | Stop Reason |")
                lines.append("  | :--- | :--- | :--- | :--- | :--- | :--- |")
                for s in ev.per_seed_results:
                    succ = "PASS" if s.get("success") else "FAIL"
                    steps = s.get("steps", "-")
                    rew = f"{s.get('reward', 0.0):.1f}" if s.get("reward") is not None else "-"
                    penned = s.get("sheep_penned", "-")
                    reason = s.get("stop_reason", "-")
                    lines.append(f"  | {s.get('seed')} | {succ} | {steps} | {rew} | {penned} | {reason} |")
    else:
        lines.append("\n*(No formal evaluations linked to this checkpoint in registry)*")

    if lineage.conclusions:
        lines.append("\n### Recorded Evidence-Backed Conclusions:")
        for idx, conc in enumerate(lineage.conclusions, 1):
            lines.append(f"#### Conclusion {idx}: `{conc.conclusion_id}` ({conc.status.upper()})")
            lines.append(f"- **Claim**: {conc.claim}")
            lines.append(f"- **Evidence Summary**: {conc.evidence_summary}")
            if conc.limitations:
                lines.append(f"- **Limitations**: {conc.limitations}")
            lines.append(f"- **Autonomous Claim**: {'YES (Audited)' if conc.is_autonomous_claim else 'NO (Assisted/Guided)'}")
            if conc.candidate_eval_id:
                lines.append(f"- **Candidate Evaluation**: `{conc.candidate_eval_id}`")
            if conc.matched_control_eval_id:
                lines.append(f"- **Matched Control Evaluation**: `{conc.matched_control_eval_id}`")
    else:
        lines.append("\n*(No formal conclusions recorded for this experiment)*")

    lines.append("")
    return "\n".join(lines)
