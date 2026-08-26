"""Cross-checkpoint evaluation seed matrix and streak tracker."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from sheepdog.training.diagnostics.config import DeterministicDiagnosticsConfig
from sheepdog.training.diagnostics.schemas import (
    SeedCheckpointOutcome,
    SeedEvaluationSummary,
    SeedMatrixReport,
)
from sheepdog.training.diagnostics.signatures import classify_failure_signature


def build_seed_matrix_report(
    evaluations: Sequence[dict[str, Any]],
    curriculum_stage: int,
    config: DeterministicDiagnosticsConfig | None = None,
) -> SeedMatrixReport:
    """Build a cross-checkpoint seed performance matrix grouping evaluations by unique training checkpoint.

    Args:
        evaluations: List of evaluation summary dicts ordered chronologically.
        curriculum_stage: Target curriculum stage being analyzed.
        config: Centralized diagnostics configuration constants.
    """
    cfg = config or DeterministicDiagnosticsConfig()

    stage_evals = [
        ev for ev in evaluations
        if ev.get("curriculum_stage") == curriculum_stage and ev.get("records")
    ]
    stage_evals.sort(key=lambda x: x.get("checkpoint_episode", 0))

    # Group evaluation runs by unique training checkpoint
    evals_by_cp: dict[int, list[dict[str, Any]]] = {}
    for ev in stage_evals:
        cp = int(ev.get("checkpoint_episode", 0) or 0)
        evals_by_cp.setdefault(cp, []).append(ev)

    # Truncate to window of unique checkpoints
    all_unique_cps = sorted(evals_by_cp.keys())
    window_cps = tuple(all_unique_cps[-cfg.window_checkpoints_max:]) if all_unique_cps else ()
    window_eval_count = sum(len(evals_by_cp[cp]) for cp in window_cps)

    # Collect all seeds tested in this window
    all_seeds: set[int] = set()
    for cp in window_cps:
        for ev in evals_by_cp[cp]:
            for r in ev.get("records", []):
                s = r.get("seed")
                if s is not None:
                    all_seeds.add(int(s))

    seed_summaries: dict[int, SeedEvaluationSummary] = {}
    watch_seeds: list[int] = []
    persistent_candidate_seeds: list[int] = []
    strong_persistence_seeds: list[int] = []
    outlier_seeds: list[int] = []
    recently_recovered_seeds: list[int] = []

    # Overall stage pass rate across all seeds in window
    total_eval_runs_all = 0
    total_eval_wins_all = 0

    for s in all_seeds:
        for cp in window_cps:
            for ev in evals_by_cp[cp]:
                for r in ev.get("records", []):
                    if r.get("seed") == s:
                        total_eval_runs_all += 1
                        if bool(r.get("success", False)):
                            total_eval_wins_all += 1

    stage_pass_rate = (total_eval_wins_all / total_eval_runs_all) if total_eval_runs_all > 0 else 0.0

    for seed in sorted(all_seeds):
        cp_outcomes: list[SeedCheckpointOutcome] = []
        total_eval_runs = 0
        failed_eval_runs = 0
        all_seed_signatures: list[str] = []

        for cp in window_cps:
            cp_evals = evals_by_cp[cp]
            eval_mode_res: dict[str, str] = {}
            cp_signatures: list[str] = []
            cp_runs_tested = 0
            cp_runs_failed = 0

            for ev in cp_evals:
                mode = str(ev.get("evaluation_mode") or ev.get("eval_mode") or "eval")
                for r in ev.get("records", []):
                    if r.get("seed") == seed:
                        cp_runs_tested += 1
                        total_eval_runs += 1
                        is_succ = bool(r.get("success", False))
                        if is_succ:
                            eval_mode_res[mode] = "pass"
                        else:
                            cp_runs_failed += 1
                            failed_eval_runs += 1
                            eval_mode_res[mode] = "fail"
                            sig = classify_failure_signature(r)
                            cp_signatures.append(sig)
                            all_seed_signatures.append(sig)

            if cp_runs_tested > 0:
                if cp_runs_failed == 0:
                    chk_status = "pass"
                elif cp_runs_failed == cp_runs_tested:
                    chk_status = "fail"
                else:
                    chk_status = "mixed"

                cp_outcomes.append(
                    SeedCheckpointOutcome(
                        checkpoint_episode=cp,
                        evaluation_runs_count=cp_runs_tested,
                        evaluation_runs_failed=cp_runs_failed,
                        checkpoint_status=chk_status,
                        eval_mode_results=eval_mode_res,
                        failure_signatures=tuple(cp_signatures),
                    )
                )

        unique_cps_tested = len(cp_outcomes)
        unique_cps_passed = sum(1 for cpo in cp_outcomes if cpo.checkpoint_status == "pass")
        cp_pass_rate = (unique_cps_passed / unique_cps_tested) if unique_cps_tested > 0 else 0.0
        eval_run_pass_rate = ((total_eval_runs - failed_eval_runs) / total_eval_runs) if total_eval_runs > 0 else 0.0

        # Trailing streak of checkpoints without a clean pass
        current_failed_cps = 0
        for cpo in reversed(cp_outcomes):
            if cpo.checkpoint_status != "pass":
                current_failed_cps += 1
            else:
                break

        # Max historical failed checkpoint streak in window
        max_cp_streak = 0
        temp_streak = 0
        for cpo in cp_outcomes:
            if cpo.checkpoint_status != "pass":
                temp_streak += 1
                if temp_streak > max_cp_streak:
                    max_cp_streak = temp_streak
            else:
                temp_streak = 0

        # Current persistence severity
        if current_failed_cps >= cfg.strong_persistence_streak_threshold:
            curr_severity = "strong_persistence"
            strong_persistence_seeds.append(seed)
        elif current_failed_cps >= cfg.persistent_candidate_streak_threshold:
            curr_severity = "persistent_candidate"
            persistent_candidate_seeds.append(seed)
        elif current_failed_cps >= cfg.watch_checkpoint_streak_threshold:
            curr_severity = "watch"
            watch_seeds.append(seed)
        else:
            curr_severity = "normal"

        # Historical persistence severity
        if max_cp_streak >= cfg.strong_persistence_streak_threshold:
            hist_severity = "strong_persistence"
        elif max_cp_streak >= cfg.persistent_candidate_streak_threshold:
            hist_severity = "persistent_candidate"
        elif max_cp_streak >= cfg.watch_checkpoint_streak_threshold:
            hist_severity = "watch"
        else:
            hist_severity = "normal"

        # Recently recovered check: exhibited persistent failure streak (>= 3) historically but passed latest checkpoint
        recently_recovered = (max_cp_streak >= cfg.persistent_candidate_streak_threshold and current_failed_cps == 0)
        if recently_recovered:
            recently_recovered_seeds.append(seed)

        # Trailing failure signature stability across checkpoints
        dominant_sig: str | None = None
        consecutive_sig_cp_streak = 0
        is_stable_sig = False

        if all_seed_signatures:
            sig_counts = Counter(all_seed_signatures)
            dominant_sig = sig_counts.most_common(1)[0][0]

            if current_failed_cps > 0:
                trailing_failed_cpos = cp_outcomes[-current_failed_cps:]
                trailing_cp_sigs: list[str] = []
                for cpo in trailing_failed_cpos:
                    if cpo.failure_signatures:
                        cp_dom = Counter(cpo.failure_signatures).most_common(1)[0][0]
                        trailing_cp_sigs.append(cp_dom)
                    else:
                        trailing_cp_sigs.append("unknown")

                if trailing_cp_sigs:
                    last_sig = trailing_cp_sigs[-1]
                    streak_count = 0
                    for sig_val in reversed(trailing_cp_sigs):
                        if sig_val == last_sig and sig_val not in ("unknown", "multiple_candidate_causes", "insufficient_telemetry"):
                            streak_count += 1
                        else:
                            break
                    consecutive_sig_cp_streak = streak_count
                    is_stable_sig = (streak_count >= 2) and (streak_count == current_failed_cps)

        # Deterministic Outlier check
        is_outlier = (
            (eval_run_pass_rate <= cfg.outlier_max_pass_rate)
            and (unique_cps_tested >= cfg.outlier_min_checkpoints)
            and (stage_pass_rate >= cfg.outlier_min_stage_pass_rate)
        )
        if is_outlier:
            outlier_seeds.append(seed)

        seed_summaries[seed] = SeedEvaluationSummary(
            seed=seed,
            unique_checkpoints_tested=unique_cps_tested,
            unique_checkpoints_passed=unique_cps_passed,
            checkpoint_pass_rate=round(cp_pass_rate, 4),
            current_failure_streak=current_failed_cps,
            current_persistence_severity=curr_severity,
            max_failure_streak=max_cp_streak,
            historical_persistence_severity=hist_severity,
            recently_recovered=recently_recovered,
            failed_evaluation_runs=failed_eval_runs,
            total_evaluation_runs=total_eval_runs,
            evaluation_run_pass_rate=round(eval_run_pass_rate, 4),
            dominant_failure_signature=dominant_sig,
            consecutive_signature_checkpoint_streak=consecutive_sig_cp_streak,
            is_stable_signature_failure=is_stable_sig,
            is_deterministic_outlier=is_outlier,
            checkpoint_history=tuple(cp_outcomes),
        )

    return SeedMatrixReport(
        curriculum_stage=curriculum_stage,
        unique_checkpoints=window_cps,
        evaluation_run_count=window_eval_count,
        seed_summaries=seed_summaries,
        watch_seeds=tuple(watch_seeds),
        persistent_candidate_seeds=tuple(persistent_candidate_seeds),
        strong_persistence_seeds=tuple(strong_persistence_seeds),
        deterministic_outlier_seeds=tuple(outlier_seeds),
        recently_recovered_seeds=tuple(recently_recovered_seeds),
    )
