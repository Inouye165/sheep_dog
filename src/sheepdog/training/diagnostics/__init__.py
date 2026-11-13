"""Deterministic training diagnostics subpackage for sheep_dog."""

from sheepdog.training.diagnostics.config import DeterministicDiagnosticsConfig
from sheepdog.training.diagnostics.engine import DeterministicDiagnosticsEngine
from sheepdog.training.diagnostics.repetition_recovery import (
    METRICS_VERSION as REPETITION_RECOVERY_METRICS_VERSION,
)
from sheepdog.training.diagnostics.repetition_recovery import (
    DogMovementType,
    RepetitionRecoveryAnalysis,
    StallOutcome,
    StallOutcomeRecord,
    UnpennedBucketMetrics,
    analyze_repetition_and_recovery,
    classify_dog_movement_pattern,
)
from sheepdog.training.diagnostics.schemas import (
    DeterministicDiagnosticReport,
    DiagnosticFinding,
    DiagnosticFindingSupport,
    ProgressWithinFailuresSummary,
    SeedEvaluationSummary,
    SeedMatrixReport,
)
from sheepdog.training.diagnostics.seed_tracker import build_seed_matrix_report
from sheepdog.training.diagnostics.signatures import (
    classify_failure_signature,
    extract_failure_candidate_causes,
)
from sheepdog.training.diagnostics.temporal_interaction import (
    INTERVAL_READABLE_LABELS,
    DogIntervalSummary,
    EpisodeTemporalAnalysis,
    FailureRecoveryComparison,
    IntervalContextWindowSummary,
    ReplayEvidenceAudit,
    RewardIntervalSummary,
    SheepIntervalSummary,
    TemporalInteractionInterval,
    analyze_episode_temporal_interactions,
    audit_replay_evidence,
    build_interval_plain_language_explanation,
    build_step_instrumentation_payload,
    build_temporal_diagnostic_payload,
    compare_failure_and_recovery_episodes,
    format_temporal_diagnostic_report,
    label_dog_observation_fields,
    load_replay_bundle,
)

__all__ = [
    "INTERVAL_READABLE_LABELS",
    "DeterministicDiagnosticsConfig",
    "DeterministicDiagnosticsEngine",
    "DeterministicDiagnosticReport",
    "DiagnosticFinding",
    "DiagnosticFindingSupport",
    "DogIntervalSummary",
    "EpisodeTemporalAnalysis",
    "FailureRecoveryComparison",
    "IntervalContextWindowSummary",
    "ProgressWithinFailuresSummary",
    "ReplayEvidenceAudit",
    "RewardIntervalSummary",
    "SeedEvaluationSummary",
    "SeedMatrixReport",
    "SheepIntervalSummary",
    "TemporalInteractionInterval",
    "analyze_episode_temporal_interactions",
    "audit_replay_evidence",
    "build_interval_plain_language_explanation",
    "build_seed_matrix_report",
    "build_step_instrumentation_payload",
    "build_temporal_diagnostic_payload",
    "classify_failure_signature",
    "compare_failure_and_recovery_episodes",
    "extract_failure_candidate_causes",
    "format_temporal_diagnostic_report",
    "label_dog_observation_fields",
    "load_replay_bundle",
    "REPETITION_RECOVERY_METRICS_VERSION",
    "DogMovementType",
    "RepetitionRecoveryAnalysis",
    "StallOutcome",
    "StallOutcomeRecord",
    "UnpennedBucketMetrics",
    "analyze_repetition_and_recovery",
    "classify_dog_movement_pattern",
]
