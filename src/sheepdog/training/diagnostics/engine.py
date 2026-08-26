"""Core deterministic training diagnostics engine."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, variance
from typing import Any, Sequence

from sheepdog.curriculum import CURRICULUM_STAGES
from sheepdog.training.diagnostics.config import DeterministicDiagnosticsConfig
from sheepdog.training.diagnostics.schemas import (
    DeterministicDiagnosticReport,
    DiagnosticFinding,
    DiagnosticFindingSupport,
    ProgressWithinFailuresSummary,
    SeedMatrixReport,
)
from sheepdog.training.diagnostics.seed_tracker import build_seed_matrix_report
from sheepdog.training.diagnostics.signatures import classify_failure_signature

CORNER_ZONES = frozenset({"top_left", "top_right", "bottom_left", "bottom_right"})
WALL_ZONES = frozenset({"top_wall", "bottom_wall", "left_wall", "right_wall"})


class DeterministicDiagnosticsEngine:
    """Computes comprehensive objective training diagnostics without an LLM."""

    def __init__(self, config: DeterministicDiagnosticsConfig | None = None) -> None:
        self.config = config or DeterministicDiagnosticsConfig()

    def analyze_stage_window(
        self,
        evaluations: Sequence[dict[str, Any]],
        curriculum_stage: int,
        target_checkpoint_episode: int | None = None,
    ) -> DeterministicDiagnosticReport:
        """Run full deterministic diagnostic analysis on a stage evaluation window."""
        cfg = self.config

        # 1. Filter evaluations matching stage and up to target checkpoint
        stage_evals = [
            ev for ev in evaluations
            if ev.get("curriculum_stage") == curriculum_stage and ev.get("records")
        ]
        if target_checkpoint_episode is not None:
            stage_evals = [
                ev for ev in stage_evals
                if (ev.get("checkpoint_episode") or 0) <= target_checkpoint_episode
            ]
        stage_evals.sort(key=lambda x: x.get("checkpoint_episode", 0))

        # Group runs by unique checkpoint_episode
        evals_by_cp: dict[int, list[dict[str, Any]]] = {}
        for ev in stage_evals:
            cp = int(ev.get("checkpoint_episode", 0) or 0)
            evals_by_cp.setdefault(cp, []).append(ev)

        all_unique_cps = sorted(evals_by_cp.keys())
        window_unique_cps = tuple(all_unique_cps[-cfg.window_checkpoints_max:]) if all_unique_cps else ()
        n_unique_cps = len(window_unique_cps)

        # Collect evaluation runs and episode records in the unique checkpoint window
        window_eval_runs: list[dict[str, Any]] = []
        for cp in window_unique_cps:
            window_eval_runs.extend(evals_by_cp[cp])

        n_eval_runs = len(window_eval_runs)
        total_samples = sum(len(ev.get("records", [])) for ev in window_eval_runs)
        latest_cp = window_unique_cps[-1] if window_unique_cps else (target_checkpoint_episode or 0)

        # Data Adequacy calculation (pure sample volume & depth, independent of failure patterns)
        c_cps = min(1.0, n_unique_cps / 6.0)
        c_runs = min(1.0, n_eval_runs / 12.0)
        c_samples = min(1.0, total_samples / 60.0)
        data_adequacy_score = round((0.40 * c_cps) + (0.30 * c_runs) + (0.30 * c_samples), 3)

        if data_adequacy_score >= 0.90:
            data_adequacy_level = "strong"
        elif data_adequacy_score >= 0.70:
            data_adequacy_level = "adequate"
        elif data_adequacy_score >= 0.40:
            data_adequacy_level = "limited"
        else:
            data_adequacy_level = "insufficient"

        # Minimum-history safeguards
        if n_unique_cps < cfg.min_unique_checkpoints_required or total_samples < cfg.min_episodes_for_trend:
            empty_seed_matrix = build_seed_matrix_report([], curriculum_stage, config=cfg)
            empty_fail_progress = ProgressWithinFailuresSummary(
                failed_episodes_count=0,
                avg_sheep_penned_at_failure=0.0,
                penned_ratio_at_failure=0.0,
                avg_min_distance_to_pen=0.0,
                stagnant_steps_ratio=0.0,
                improving_proximity=False,
                high_partial_success_rate=False,
                failure_progress_score=0.0,
                failure_progress_trend="insufficient_history",
                data_quality_warnings=(),
            )
            latest_s = float(window_eval_runs[-1].get("success_rate", 0.0)) if window_eval_runs else 0.0
            return DeterministicDiagnosticReport(
                stage=curriculum_stage,
                checkpoint_episode=latest_cp,
                unique_checkpoints=window_unique_cps,
                unique_checkpoint_count=n_unique_cps,
                evaluation_run_count=n_eval_runs,
                episode_sample_count=total_samples,
                performance_band="low" if latest_s < cfg.moderate_performance_threshold else "moderate",
                success_rate_latest=latest_s,
                rolling_success_rate=latest_s,
                success_trend="insufficient_history",
                success_trend_slope=0.0,
                step_efficiency_trend="insufficient_history",
                step_reduction_pct=None,
                avg_steps_success=None,
                failure_progress=empty_fail_progress,
                seed_matrix=empty_seed_matrix,
                dominant_failure_signature=None,
                signature_distribution={},
                findings=(),
                spatial_bottlenecks={},
                observation_health={},
                data_adequacy_score=data_adequacy_score,
                data_adequacy_level=data_adequacy_level,
                requires_investigation=False,
                summary_message=f"Insufficient history ({n_unique_cps}/{cfg.min_unique_checkpoints_required} checkpoints, {total_samples}/{cfg.min_episodes_for_trend} samples) to determine trends.",
            )

        # 2. Checkpoint-level success rates (average rate per unique checkpoint)
        cp_success_rates: list[float] = []
        for cp in window_unique_cps:
            cp_rates = [float(ev.get("success_rate", 0.0)) for ev in evals_by_cp[cp]]
            cp_success_rates.append(fmean(cp_rates))

        latest_success = cp_success_rates[-1]
        rolling_success = fmean(cp_success_rates)
        max_prior_success = max(cp_success_rates[:-1]) if len(cp_success_rates) > 1 else latest_success
        all_time_max = max(cp_success_rates)

        # Linear regression slope over unique training checkpoint index
        x_vals = list(range(n_unique_cps))
        x_mean = fmean(x_vals)
        y_mean = rolling_success
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, cp_success_rates))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = (numerator / denominator) if denominator > 0 else 0.0

        rate_variance = variance(cp_success_rates) if len(cp_success_rates) > 1 else 0.0

        # 3. Performance Band (Absolute level) vs Success Trend (Directional)
        if rolling_success >= cfg.near_mastery_threshold:
            perf_band = "near_mastery"
        elif rolling_success >= cfg.high_performance_threshold:
            perf_band = "high"
        elif rolling_success >= cfg.moderate_performance_threshold:
            perf_band = "moderate"
        else:
            perf_band = "low"

        if all_time_max >= cfg.cliff_all_time_min_max and latest_success <= cfg.cliff_latest_max:
            trend = "cliff"
        elif (max_prior_success - latest_success >= 0.20) or (slope <= cfg.regressing_slope_threshold):
            trend = "regressing"
        elif abs(slope) < cfg.plateau_max_slope and (rate_variance < cfg.plateau_variance_threshold or n_unique_cps >= 4):
            trend = "stable"
        elif slope >= cfg.improving_slope_threshold:
            trend = "improving"
        else:
            trend = "stable"

        # 4. Step-Efficiency Trend (across unique checkpoints)
        step_efficiency_trend, step_reduction_pct, avg_steps_success = self._compute_step_efficiency_trend(
            window_unique_cps, evals_by_cp
        )

        # 5. Progress Within Failures Analysis with Stage Sheep Integrity Validation
        fail_progress = self._compute_failure_progress(window_unique_cps, evals_by_cp, curriculum_stage)

        # 6. Seed Matrix & Failure Streaks (Unique checkpoints)
        seed_matrix = build_seed_matrix_report(window_eval_runs, curriculum_stage, config=cfg)

        # 7. Failure Signatures Distribution
        dominant_sig, sig_distribution = self._compute_failure_signatures(window_eval_runs)

        # 8. Spatial Bottlenecks & Observation Health
        spatial_bottlenecks = self._extract_spatial_bottlenecks(window_eval_runs, curriculum_stage)
        observation_health = self._extract_observation_health(window_eval_runs[-1])

        # 9. Structured Diagnostic Findings
        findings = self._build_findings(
            window_unique_cps=window_unique_cps,
            window_eval_runs=window_eval_runs,
            seed_matrix=seed_matrix,
            spatial_bottlenecks=spatial_bottlenecks,
            sig_distribution=sig_distribution,
        )

        active_outliers = [
            s for s in seed_matrix.deterministic_outlier_seeds
            if seed_matrix.seed_summaries[s].current_failure_streak > 0
        ]

        requires_investigation = (
            trend in ("regressing", "cliff")
            or len(seed_matrix.persistent_candidate_seeds) > 0
            or len(seed_matrix.strong_persistence_seeds) > 0
            or len(active_outliers) > 0
            or bool(spatial_bottlenecks.get("critical_bottleneck"))
            or (trend == "stable" and perf_band in ("low", "moderate"))
        )

        summary_msg = self._build_summary_message(
            perf_band=perf_band,
            trend=trend,
            rolling_success=rolling_success,
            slope=slope,
            step_trend=step_efficiency_trend,
            seed_matrix=seed_matrix,
            dominant_sig=dominant_sig,
            fail_progress=fail_progress,
        )

        return DeterministicDiagnosticReport(
            stage=curriculum_stage,
            checkpoint_episode=latest_cp,
            unique_checkpoints=window_unique_cps,
            unique_checkpoint_count=n_unique_cps,
            evaluation_run_count=n_eval_runs,
            episode_sample_count=total_samples,
            performance_band=perf_band,
            success_rate_latest=round(latest_success, 4),
            rolling_success_rate=round(rolling_success, 4),
            success_trend=trend,
            success_trend_slope=round(slope, 4),
            step_efficiency_trend=step_efficiency_trend,
            step_reduction_pct=round(step_reduction_pct, 4) if step_reduction_pct is not None else None,
            avg_steps_success=round(avg_steps_success, 1) if avg_steps_success is not None else None,
            failure_progress=fail_progress,
            seed_matrix=seed_matrix,
            dominant_failure_signature=dominant_sig,
            signature_distribution=sig_distribution,
            findings=tuple(findings),
            spatial_bottlenecks=spatial_bottlenecks,
            observation_health=observation_health,
            data_adequacy_score=data_adequacy_score,
            data_adequacy_level=data_adequacy_level,
            requires_investigation=requires_investigation,
            summary_message=summary_msg,
        )

    def _compute_step_efficiency_trend(
        self, window_unique_cps: tuple[int, ...], evals_by_cp: dict[int, list[dict[str, Any]]]
    ) -> tuple[str, float | None, float | None]:
        cfg = self.config
        cp_succ_steps: list[float] = []

        for cp in window_unique_cps:
            succ_steps_at_cp = []
            for ev in evals_by_cp[cp]:
                for r in ev.get("records", []):
                    if bool(r.get("success", False)) and r.get("steps"):
                        succ_steps_at_cp.append(float(r["steps"]))
            if succ_steps_at_cp:
                cp_succ_steps.append(fmean(succ_steps_at_cp))

        if len(cp_succ_steps) < 2:
            latest_avg = cp_succ_steps[-1] if cp_succ_steps else None
            return "insufficient_history", None, latest_avg

        prev_steps = cp_succ_steps[-2]
        curr_steps = cp_succ_steps[-1]
        reduction_pct = (prev_steps - curr_steps) / prev_steps if prev_steps > 0 else 0.0

        if reduction_pct >= cfg.step_reduction_threshold_pct and (prev_steps - curr_steps) >= cfg.step_reduction_min_steps:
            trend = "improving"
        elif curr_steps > prev_steps * (1.0 + cfg.step_regression_threshold_pct) and (curr_steps - prev_steps) >= cfg.step_regression_min_steps:
            trend = "regressing"
        else:
            trend = "stable"

        return trend, reduction_pct, curr_steps

    def _compute_failure_progress(
        self,
        window_unique_cps: tuple[int, ...],
        evals_by_cp: dict[int, list[dict[str, Any]]],
        curriculum_stage: int,
    ) -> ProgressWithinFailuresSummary:
        failed_records: list[dict[str, Any]] = []
        per_cp_failures: list[list[dict[str, Any]]] = []
        warnings: list[str] = []

        # Determine authoritative stage sheep count from curriculum configuration
        stage_cfg_sheep = None
        if curriculum_stage in CURRICULUM_STAGES:
            stage_cfg_sheep = CURRICULUM_STAGES[curriculum_stage].get("sheep")

        for cp in window_unique_cps:
            cp_fails = []
            for ev in evals_by_cp[cp]:
                for r in ev.get("records", []):
                    if not bool(r.get("success", False)):
                        cp_fails.append(r)
            if cp_fails:
                per_cp_failures.append(cp_fails)
            failed_records.extend(cp_fails)

        if not failed_records:
            return ProgressWithinFailuresSummary(
                failed_episodes_count=0,
                avg_sheep_penned_at_failure=0.0,
                penned_ratio_at_failure=0.0,
                avg_min_distance_to_pen=0.0,
                stagnant_steps_ratio=0.0,
                improving_proximity=False,
                high_partial_success_rate=False,
                failure_progress_score=0.0,
                failure_progress_trend="insufficient_history",
                data_quality_warnings=(),
            )

        penned_counts: list[float] = []
        ratios: list[float] = []

        for r in failed_records:
            raw_penned = r.get("sheep_penned", 0)
            if raw_penned is None or (isinstance(raw_penned, float) and math.isnan(raw_penned)):
                raw_penned = 0
                warnings.append("NaN/None sheep_penned converted to 0")

            penned_val = max(0.0, float(raw_penned))

            # Determine total sheep count
            total_s = r.get("total_sheep") or r.get("num_sheep")
            if total_s is None:
                final_positions = r.get("final_sheep_positions")
                if final_positions and isinstance(final_positions, (list, tuple)):
                    total_s = len(final_positions)
                elif stage_cfg_sheep is not None:
                    total_s = stage_cfg_sheep
                else:
                    total_s = max(1.0, penned_val)

            total_s = max(1.0, float(total_s))

            if penned_val > total_s:
                warnings.append(f"sheep_penned ({penned_val}) exceeded total_sheep ({total_s}); clamped to total_sheep")
                penned_val = total_s

            ratio = max(0.0, min(1.0, penned_val / total_s))
            penned_counts.append(penned_val)
            ratios.append(ratio)

        min_dists = []
        for r in failed_records:
            val = r.get("min_sheep_distance_to_pen")
            if val is not None:
                try:
                    f_val = float(val)
                    if not math.isnan(f_val) and not math.isinf(f_val):
                        min_dists.append(f_val)
                except (ValueError, TypeError):
                    pass

        no_prog = [max(0.0, float(r.get("no_progress_steps", 0) or 0)) for r in failed_records]
        steps = [max(1.0, float(r.get("steps", 1) or 1)) for r in failed_records]
        stagnant_ratios = [min(1.0, np / st) for np, st in zip(no_prog, steps)]

        avg_penned = fmean(penned_counts) if penned_counts else 0.0
        avg_ratio = fmean(ratios) if ratios else 0.0
        avg_min_dist = fmean(min_dists) if min_dists else 0.0
        avg_stagnant = fmean(stagnant_ratios) if stagnant_ratios else 0.0

        first_penned: float | None = None
        latest_penned: float | None = None
        penned_delta: float | None = None
        first_min_d: float | None = None
        latest_min_d: float | None = None
        min_d_delta: float | None = None
        failure_progress_trend = "stable"

        if len(per_cp_failures) >= 2:
            first_fails = per_cp_failures[0]
            latest_fails = per_cp_failures[-1]

            first_penned = fmean(max(0.0, float(r.get("sheep_penned", 0) or 0)) for r in first_fails)
            latest_penned = fmean(max(0.0, float(r.get("sheep_penned", 0) or 0)) for r in latest_fails)
            penned_delta = round(latest_penned - first_penned, 2)
            first_penned = round(first_penned, 2)
            latest_penned = round(latest_penned, 2)

            f_dists = [float(r.get("min_sheep_distance_to_pen", 99.0) or 99.0) for r in first_fails if r.get("min_sheep_distance_to_pen") is not None]
            l_dists = [float(r.get("min_sheep_distance_to_pen", 99.0) or 99.0) for r in latest_fails if r.get("min_sheep_distance_to_pen") is not None]
            if f_dists and l_dists:
                first_min_d = round(fmean(f_dists), 2)
                latest_min_d = round(fmean(l_dists), 2)
                min_d_delta = round(latest_min_d - first_min_d, 2)

            if (penned_delta is not None and penned_delta >= 0.5) or (min_d_delta is not None and min_d_delta <= -2.0):
                failure_progress_trend = "improving"
            elif (penned_delta is not None and penned_delta <= -0.5) or (min_d_delta is not None and min_d_delta >= 2.0):
                failure_progress_trend = "regressing"
            else:
                failure_progress_trend = "stable"
        else:
            failure_progress_trend = "insufficient_history"

        improving_prox = bool(min_d_delta is not None and min_d_delta < 0.0) or (len(per_cp_failures) < 2 and avg_min_dist < 5.0)
        high_partial = avg_ratio >= 0.60

        progress_score = (
            (avg_ratio * 0.45)
            + ((1.0 - min(1.0, avg_min_dist / 20.0)) * 0.35)
            + ((1.0 - avg_stagnant) * 0.20)
        )
        progress_score = max(0.0, min(1.0, progress_score))

        return ProgressWithinFailuresSummary(
            failed_episodes_count=len(failed_records),
            avg_sheep_penned_at_failure=round(avg_penned, 2),
            penned_ratio_at_failure=round(avg_ratio, 3),
            avg_min_distance_to_pen=round(avg_min_dist, 2),
            stagnant_steps_ratio=round(avg_stagnant, 3),
            improving_proximity=improving_prox,
            high_partial_success_rate=high_partial,
            failure_progress_score=round(progress_score, 3),
            failure_progress_trend=failure_progress_trend,
            first_checkpoint_avg_penned=first_penned,
            latest_checkpoint_avg_penned=latest_penned,
            penned_delta=penned_delta,
            first_checkpoint_avg_min_dist=first_min_d,
            latest_checkpoint_avg_min_dist=latest_min_d,
            min_dist_delta=min_d_delta,
            data_quality_warnings=tuple(warnings[:5]),
        )

    def _compute_failure_signatures(
        self, window_eval_runs: list[dict[str, Any]]
    ) -> tuple[str | None, dict[str, int]]:
        signatures: list[str] = []
        for ev in window_eval_runs:
            for r in ev.get("records", []):
                if not bool(r.get("success", False)):
                    sig = classify_failure_signature(r)
                    signatures.append(sig)

        if not signatures:
            return None, {}

        counts = Counter(signatures)
        dominant = counts.most_common(1)[0][0]
        return dominant, dict(counts)

    def _extract_spatial_bottlenecks(
        self, window_eval_runs: list[dict[str, Any]], curriculum_stage: int
    ) -> dict[str, Any]:
        cfg = self.config
        corner_stuck_count = 0
        total_records = 0
        zone_counts: dict[str, int] = {}
        zone_wins: dict[str, int] = {}

        for ev in window_eval_runs:
            for r in ev.get("records", []):
                total_records += 1
                if bool(r.get("corner_stuck_at_end", False)):
                    corner_stuck_count += 1
                z = str(r.get("initial_sheep_zone") or r.get("final_sheep_zone") or "center")
                zone_counts[z] = zone_counts.get(z, 0) + 1
                if bool(r.get("success", False)):
                    zone_wins[z] = zone_wins.get(z, 0) + 1

        zone_rates = {
            z: (zone_wins.get(z, 0) / count)
            for z, count in zone_counts.items()
            if count >= 2
        }

        critical_bottleneck = False
        bottleneck_reason = None

        center_rate = zone_rates.get("center")
        corner_rates = [r for z, r in zone_rates.items() if z in CORNER_ZONES]
        if center_rate is not None and corner_rates:
            worst_corner_rate = min(corner_rates)
            if center_rate - worst_corner_rate >= cfg.spatial_disparity_threshold:
                critical_bottleneck = True
                bottleneck_reason = f"Severe corner entrapment gap: {worst_corner_rate:.1%} in corners vs {center_rate:.1%} in center."

        return {
            "total_records_analyzed": total_records,
            "corner_stuck_count": corner_stuck_count,
            "zone_win_rates": {z: round(r, 3) for z, r in zone_rates.items()},
            "critical_bottleneck": critical_bottleneck,
            "bottleneck_reason": bottleneck_reason,
        }

    def _extract_observation_health(self, latest_eval: dict[str, Any]) -> dict[str, Any]:
        records = latest_eval.get("records", [])
        for r in records:
            obs_diag = r.get("observation_diagnostics")
            if obs_diag and isinstance(obs_diag, dict):
                return {
                    "constant_features": obs_diag.get("constant_features", []),
                    "saturated_features": obs_diag.get("saturated_features", []),
                    "nan_or_inf_features": obs_diag.get("nan_or_inf_features", []),
                    "vector_length": obs_diag.get("vector_length", 0),
                }
        return {"constant_features": [], "saturated_features": [], "nan_or_inf_features": []}

    def _build_findings(
        self,
        window_unique_cps: tuple[int, ...],
        window_eval_runs: list[dict[str, Any]],
        seed_matrix: SeedMatrixReport,
        spatial_bottlenecks: dict[str, Any],
        sig_distribution: dict[str, int],
    ) -> list[DiagnosticFinding]:
        findings: list[DiagnosticFinding] = []
        n_unique_cps = len(window_unique_cps)
        n_runs = len(window_eval_runs)
        total_failures = sum(sig_distribution.values())

        if total_failures == 0:
            return findings

        # 1. Check for dominant failure signature finding
        for sig, count in sig_distribution.items():
            if sig in ("unknown", "multiple_candidate_causes", "insufficient_telemetry"):
                continue
            if count >= 3 and (count / total_failures) >= 0.35:
                affected_seeds = [
                    seed for seed, summary in seed_matrix.seed_summaries.items()
                    if summary.dominant_failure_signature == sig and summary.total_evaluation_runs > 0
                ]
                max_streak = max((seed_matrix.seed_summaries[s].current_failure_streak for s in affected_seeds), default=0)
                ev_level = "strong" if (count / total_failures >= 0.60 and len(affected_seeds) >= 2) else "moderate"

                support = DiagnosticFindingSupport(
                    affected_failures=count,
                    total_failures=total_failures,
                    checkpoints_observed=n_unique_cps,
                    evaluation_runs_observed=n_runs,
                    affected_seeds=tuple(affected_seeds),
                    longest_seed_checkpoint_streak=max_streak,
                )
                findings.append(
                    DiagnosticFinding(
                        finding_type="dominant_failure_mode",
                        target=sig,
                        evidence_level=ev_level,
                        description=f"Behavioral failure signature '{sig}' accounted for {count}/{total_failures} failures across {n_unique_cps} checkpoints ({n_runs} evaluation runs).",
                        support=support,
                    )
                )

        # 2. Check for persistent seed failures (current or severe historical)
        persistent_all = set(seed_matrix.persistent_candidate_seeds) | set(seed_matrix.strong_persistence_seeds)
        for s in sorted(persistent_all):
            summary = seed_matrix.seed_summaries[s]
            ev_level = "strong" if summary.current_persistence_severity == "strong_persistence" and summary.is_stable_signature_failure else "moderate"
            sig_note = f" (stable signature: {summary.dominant_failure_signature})" if summary.is_stable_signature_failure else " (varying failure modes)"

            support = DiagnosticFindingSupport(
                affected_failures=summary.failed_evaluation_runs,
                total_failures=total_failures,
                checkpoints_observed=summary.unique_checkpoints_tested,
                evaluation_runs_observed=summary.total_evaluation_runs,
                affected_seeds=(s,),
                longest_seed_checkpoint_streak=summary.current_failure_streak,
                success_rate_with_condition=summary.checkpoint_pass_rate,
            )
            findings.append(
                DiagnosticFinding(
                    finding_type="persistent_seed_failure",
                    target=f"seed_{s}",
                    evidence_level=ev_level,
                    description=f"Seed {s} has failed in {summary.current_failure_streak} consecutive training checkpoints ({summary.failed_evaluation_runs}/{summary.total_evaluation_runs} evaluation runs){sig_note}.",
                    support=support,
                )
            )

        # 3. Check for critical spatial disparity
        if spatial_bottlenecks.get("critical_bottleneck"):
            zone_rates = spatial_bottlenecks.get("zone_win_rates", {})
            corner_fails = spatial_bottlenecks.get("corner_stuck_count", 0)
            center_rate = zone_rates.get("center")
            corner_rates = [r for z, r in zone_rates.items() if z in CORNER_ZONES]
            worst_c_rate = min(corner_rates) if corner_rates else 0.0

            support = DiagnosticFindingSupport(
                affected_failures=corner_fails,
                total_failures=total_failures,
                checkpoints_observed=n_unique_cps,
                evaluation_runs_observed=n_runs,
                affected_seeds=(),
                longest_seed_checkpoint_streak=0,
                success_rate_with_condition=worst_c_rate,
                success_rate_without_condition=center_rate,
            )
            findings.append(
                DiagnosticFinding(
                    finding_type="spatial_disparity",
                    target="corners",
                    evidence_level="strong",
                    description=str(spatial_bottlenecks.get("bottleneck_reason") or "Spatial corner entrapment bottleneck observed."),
                    support=support,
                )
            )

        return findings

    def _build_summary_message(
        self,
        perf_band: str,
        trend: str,
        rolling_success: float,
        slope: float,
        step_trend: str,
        seed_matrix: SeedMatrixReport,
        dominant_sig: str | None,
        fail_progress: ProgressWithinFailuresSummary,
    ) -> str:
        band_labels = {
            "near_mastery": "Near Mastery",
            "high": "High Performance",
            "moderate": "Moderate Performance",
            "low": "Low Performance",
        }
        band_str = band_labels.get(perf_band, "Performance")

        if trend == "improving":
            msg = f"{band_str} with positive progression ({rolling_success:.1%} success rate, slope +{slope:.3f})."
            if step_trend == "improving":
                msg += " Completion steps on successful episodes are actively decreasing."
            return msg

        if trend == "cliff":
            return f"Abrupt performance drop observed: success rate decreased sharply to {rolling_success:.1%} ({band_str.lower()})."

        if trend == "regressing":
            return f"Negative performance trend observed (slope {slope:.3f}, rolling success rate {rolling_success:.1%}, {band_str.lower()})."

        # Stable case
        msg = f"{band_str} is stable at {rolling_success:.1%} success rate."
        persistent = seed_matrix.persistent_candidate_seeds + seed_matrix.strong_persistence_seeds
        if persistent:
            seeds_str = ", ".join(str(s) for s in persistent)
            msg += f" Consecutive checkpoint failures observed on seed(s) {seeds_str}."
        if dominant_sig and dominant_sig not in ("unknown", "multiple_candidate_causes", "insufficient_telemetry"):
            sig_name = dominant_sig.replace("_", " ").title()
            msg += f" Most frequent failure signature: {sig_name}."

        if fail_progress.failure_progress_score >= 0.70:
            msg += f" Failed episodes exhibit high partial penning ({fail_progress.penned_ratio_at_failure:.1%}) and close pen proximity."

        return msg
