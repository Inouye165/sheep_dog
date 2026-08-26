"""Dataclasses and schemas for deterministic training diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressWithinFailuresSummary:
    """Detailed progress indicators for failed episodes within an evaluation window."""

    failed_episodes_count: int
    avg_sheep_penned_at_failure: float
    penned_ratio_at_failure: float
    avg_min_distance_to_pen: float
    stagnant_steps_ratio: float
    improving_proximity: bool
    high_partial_success_rate: bool
    failure_progress_score: float  # [0.0 - 1.0] Objective evidence of proximity/penning progress during failures
    failure_progress_trend: str  # "insufficient_history", "improving", "stable", "regressing"

    # Explicit Window Progression Endpoints & Deltas
    first_checkpoint_avg_penned: float | None = None
    latest_checkpoint_avg_penned: float | None = None
    penned_delta: float | None = None
    first_checkpoint_avg_min_dist: float | None = None
    latest_checkpoint_avg_min_dist: float | None = None
    min_dist_delta: float | None = None

    # Telemetry Data Quality Flag
    data_quality_warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedCheckpointOutcome:
    """Per-training-checkpoint outcome for an individual seed, combining all evaluation runs."""

    checkpoint_episode: int
    evaluation_runs_count: int
    evaluation_runs_failed: int
    checkpoint_status: str  # "pass", "fail", "mixed", "insufficient"
    eval_mode_results: dict[str, str]  # e.g. {"quick": "fail", "confidence": "fail"}
    failure_signatures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_episode": self.checkpoint_episode,
            "evaluation_runs_count": self.evaluation_runs_count,
            "evaluation_runs_failed": self.evaluation_runs_failed,
            "checkpoint_status": self.checkpoint_status,
            "eval_mode_results": dict(self.eval_mode_results),
            "failure_signatures": list(self.failure_signatures),
        }


@dataclass(frozen=True, slots=True)
class SeedEvaluationSummary:
    """Historical multi-checkpoint summary for an individual evaluation seed."""

    seed: int
    unique_checkpoints_tested: int
    unique_checkpoints_passed: int
    checkpoint_pass_rate: float

    # Current vs Historical Persistence Separation
    current_failure_streak: int  # Trailing consecutive checkpoints without a clean pass
    current_persistence_severity: str  # "normal", "watch", "persistent_candidate", "strong_persistence"
    max_failure_streak: int  # All-time maximum consecutive failed checkpoints
    historical_persistence_severity: str  # "normal", "watch", "persistent_candidate", "strong_persistence"
    recently_recovered: bool  # True if had persistent failure in history but passed latest checkpoint

    # Evaluation Run metrics
    failed_evaluation_runs: int
    total_evaluation_runs: int
    evaluation_run_pass_rate: float

    # Failure Signatures & Outliers
    dominant_failure_signature: str | None
    consecutive_signature_checkpoint_streak: int
    is_stable_signature_failure: bool
    is_deterministic_outlier: bool
    checkpoint_history: tuple[SeedCheckpointOutcome, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "unique_checkpoints_tested": self.unique_checkpoints_tested,
            "unique_checkpoints_passed": self.unique_checkpoints_passed,
            "checkpoint_pass_rate": self.checkpoint_pass_rate,
            "current_failure_streak": self.current_failure_streak,
            "current_persistence_severity": self.current_persistence_severity,
            "max_failure_streak": self.max_failure_streak,
            "historical_persistence_severity": self.historical_persistence_severity,
            "recently_recovered": self.recently_recovered,
            "failed_evaluation_runs": self.failed_evaluation_runs,
            "total_evaluation_runs": self.total_evaluation_runs,
            "evaluation_run_pass_rate": self.evaluation_run_pass_rate,
            "dominant_failure_signature": self.dominant_failure_signature,
            "consecutive_signature_checkpoint_streak": self.consecutive_signature_checkpoint_streak,
            "is_stable_signature_failure": self.is_stable_signature_failure,
            "is_deterministic_outlier": self.is_deterministic_outlier,
            "checkpoint_history": [ch.to_dict() for ch in self.checkpoint_history],
        }


@dataclass(frozen=True, slots=True)
class SeedMatrixReport:
    """Cross-checkpoint evaluation seed matrix."""

    curriculum_stage: int
    unique_checkpoints: tuple[int, ...]
    evaluation_run_count: int
    seed_summaries: dict[int, SeedEvaluationSummary]
    watch_seeds: tuple[int, ...]
    persistent_candidate_seeds: tuple[int, ...]
    strong_persistence_seeds: tuple[int, ...]
    deterministic_outlier_seeds: tuple[int, ...]
    recently_recovered_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_stage": self.curriculum_stage,
            "unique_checkpoints": list(self.unique_checkpoints),
            "evaluation_run_count": self.evaluation_run_count,
            "seed_summaries": {str(k): v.to_dict() for k, v in self.seed_summaries.items()},
            "watch_seeds": list(self.watch_seeds),
            "persistent_candidate_seeds": list(self.persistent_candidate_seeds),
            "strong_persistence_seeds": list(self.strong_persistence_seeds),
            "deterministic_outlier_seeds": list(self.deterministic_outlier_seeds),
            "recently_recovered_seeds": list(self.recently_recovered_seeds),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFindingSupport:
    """Empirical grounding and support metrics for a specific diagnostic finding."""

    affected_failures: int
    total_failures: int
    checkpoints_observed: int  # Unique checkpoints observed
    evaluation_runs_observed: int  # Evaluation runs observed
    affected_seeds: tuple[int, ...]
    longest_seed_checkpoint_streak: int
    success_rate_with_condition: float | None = None
    success_rate_without_condition: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_failures": self.affected_failures,
            "total_failures": self.total_failures,
            "checkpoints_observed": self.checkpoints_observed,
            "evaluation_runs_observed": self.evaluation_runs_observed,
            "affected_seeds": list(self.affected_seeds),
            "longest_seed_checkpoint_streak": self.longest_seed_checkpoint_streak,
            "success_rate_with_condition": self.success_rate_with_condition,
            "success_rate_without_condition": self.success_rate_without_condition,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """A major structured observation grounded by empirical telemetry."""

    finding_type: str  # e.g. "dominant_failure_mode", "persistent_seed_failure", "spatial_disparity"
    target: str  # e.g. "top_left", "seed_103", "gate_corridor"
    evidence_level: str  # "weak", "moderate", "strong" (local evidentiary strength)
    description: str
    support: DiagnosticFindingSupport

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "target": self.target,
            "evidence_level": self.evidence_level,
            "description": self.description,
            "support": self.support.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DeterministicDiagnosticReport:
    """Comprehensive top-level deterministic diagnostic report."""

    stage: int
    checkpoint_episode: int
    unique_checkpoints: tuple[int, ...]
    unique_checkpoint_count: int
    evaluation_run_count: int
    episode_sample_count: int

    # Absolute Performance vs Directional Trend Separation
    performance_band: str  # "low", "moderate", "high", "near_mastery"
    success_rate_latest: float
    rolling_success_rate: float
    success_trend: str  # "insufficient_history", "improving", "stable", "regressing", "cliff"
    success_trend_slope: float  # Linear regression slope over unique training checkpoints

    # Step Efficiency Trend
    step_efficiency_trend: str  # "insufficient_history", "improving", "stable", "regressing"
    step_reduction_pct: float | None
    avg_steps_success: float | None

    # Progress Within Failures
    failure_progress: ProgressWithinFailuresSummary

    # Seed Analysis
    seed_matrix: SeedMatrixReport

    # Failure Signatures
    dominant_failure_signature: str | None
    signature_distribution: dict[str, int]

    # Empirical Structured Findings
    findings: tuple[DiagnosticFinding, ...]

    # Spatial Bottlenecks & Observation Health
    spatial_bottlenecks: dict[str, Any]
    observation_health: dict[str, Any]

    # Data Adequacy (Volume & Depth of evaluation data, NOT AI confidence)
    data_adequacy_score: float  # [0.0 - 1.0]
    data_adequacy_level: str  # "insufficient", "limited", "adequate", "strong"

    # Action triggers & message
    requires_investigation: bool
    summary_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "checkpoint_episode": self.checkpoint_episode,
            "unique_checkpoints": list(self.unique_checkpoints),
            "unique_checkpoint_count": self.unique_checkpoint_count,
            "evaluation_run_count": self.evaluation_run_count,
            "episode_sample_count": self.episode_sample_count,
            "performance_band": self.performance_band,
            "success_rate_latest": self.success_rate_latest,
            "rolling_success_rate": self.rolling_success_rate,
            "success_trend": self.success_trend,
            "success_trend_slope": self.success_trend_slope,
            "step_efficiency_trend": self.step_efficiency_trend,
            "step_reduction_pct": self.step_reduction_pct,
            "avg_steps_success": self.avg_steps_success,
            "failure_progress": self.failure_progress.to_dict(),
            "seed_matrix": self.seed_matrix.to_dict(),
            "dominant_failure_signature": self.dominant_failure_signature,
            "signature_distribution": dict(self.signature_distribution),
            "findings": [f.to_dict() for f in self.findings],
            "spatial_bottlenecks": dict(self.spatial_bottlenecks),
            "observation_health": dict(self.observation_health),
            "data_adequacy_score": self.data_adequacy_score,
            "data_adequacy_level": self.data_adequacy_level,
            "requires_investigation": self.requires_investigation,
            "summary_message": self.summary_message,
        }
