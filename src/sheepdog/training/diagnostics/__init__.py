"""Deterministic training diagnostics subpackage for sheep_dog."""

from sheepdog.training.diagnostics.config import DeterministicDiagnosticsConfig
from sheepdog.training.diagnostics.engine import DeterministicDiagnosticsEngine
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

__all__ = [
    "DeterministicDiagnosticsConfig",
    "DeterministicDiagnosticsEngine",
    "DeterministicDiagnosticReport",
    "DiagnosticFinding",
    "DiagnosticFindingSupport",
    "ProgressWithinFailuresSummary",
    "SeedEvaluationSummary",
    "SeedMatrixReport",
    "build_seed_matrix_report",
    "classify_failure_signature",
    "extract_failure_candidate_causes",
]
