"""Deterministic training diagnostics subpackage for sheep_dog."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "INTERVAL_READABLE_LABELS": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "INTERVAL_READABLE_LABELS",
    ),
    "DeterministicDiagnosticsConfig": (
        "sheepdog.training.diagnostics.config",
        "DeterministicDiagnosticsConfig",
    ),
    "DeterministicDiagnosticsEngine": (
        "sheepdog.training.diagnostics.engine",
        "DeterministicDiagnosticsEngine",
    ),
    "DeterministicDiagnosticReport": (
        "sheepdog.training.diagnostics.schemas",
        "DeterministicDiagnosticReport",
    ),
    "DiagnosticFinding": (
        "sheepdog.training.diagnostics.schemas",
        "DiagnosticFinding",
    ),
    "DiagnosticFindingSupport": (
        "sheepdog.training.diagnostics.schemas",
        "DiagnosticFindingSupport",
    ),
    "DogIntervalSummary": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "DogIntervalSummary",
    ),
    "EpisodeTemporalAnalysis": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "EpisodeTemporalAnalysis",
    ),
    "FailureRecoveryComparison": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "FailureRecoveryComparison",
    ),
    "IntervalContextWindowSummary": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "IntervalContextWindowSummary",
    ),
    "ProgressWithinFailuresSummary": (
        "sheepdog.training.diagnostics.schemas",
        "ProgressWithinFailuresSummary",
    ),
    "ReplayEvidenceAudit": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "ReplayEvidenceAudit",
    ),
    "RewardIntervalSummary": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "RewardIntervalSummary",
    ),
    "SeedEvaluationSummary": (
        "sheepdog.training.diagnostics.schemas",
        "SeedEvaluationSummary",
    ),
    "SeedMatrixReport": (
        "sheepdog.training.diagnostics.schemas",
        "SeedMatrixReport",
    ),
    "SheepIntervalSummary": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "SheepIntervalSummary",
    ),
    "TemporalInteractionInterval": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "TemporalInteractionInterval",
    ),
    "analyze_episode_temporal_interactions": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "analyze_episode_temporal_interactions",
    ),
    "audit_replay_evidence": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "audit_replay_evidence",
    ),
    "build_interval_plain_language_explanation": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "build_interval_plain_language_explanation",
    ),
    "build_seed_matrix_report": (
        "sheepdog.training.diagnostics.seed_tracker",
        "build_seed_matrix_report",
    ),
    "build_step_instrumentation_payload": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "build_step_instrumentation_payload",
    ),
    "build_temporal_diagnostic_payload": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "build_temporal_diagnostic_payload",
    ),
    "classify_failure_signature": (
        "sheepdog.training.diagnostics.signatures",
        "classify_failure_signature",
    ),
    "compare_failure_and_recovery_episodes": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "compare_failure_and_recovery_episodes",
    ),
    "extract_failure_candidate_causes": (
        "sheepdog.training.diagnostics.signatures",
        "extract_failure_candidate_causes",
    ),
    "format_temporal_diagnostic_report": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "format_temporal_diagnostic_report",
    ),
    "label_dog_observation_fields": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "label_dog_observation_fields",
    ),
    "load_replay_bundle": (
        "sheepdog.training.diagnostics.temporal_interaction",
        "load_replay_bundle",
    ),
    "REPETITION_RECOVERY_METRICS_VERSION": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "METRICS_VERSION",
    ),
    "DogMovementType": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "DogMovementType",
    ),
    "RepetitionRecoveryAnalysis": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "RepetitionRecoveryAnalysis",
    ),
    "StallOutcome": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "StallOutcome",
    ),
    "StallOutcomeRecord": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "StallOutcomeRecord",
    ),
    "UnpennedBucketMetrics": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "UnpennedBucketMetrics",
    ),
    "analyze_repetition_and_recovery": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "analyze_repetition_and_recovery",
    ),
    "classify_dog_movement_pattern": (
        "sheepdog.training.diagnostics.repetition_recovery",
        "classify_dog_movement_pattern",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve and cache exported diagnostic symbols on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including lazily exported symbols."""
    return sorted(set(globals()) | set(__all__))
