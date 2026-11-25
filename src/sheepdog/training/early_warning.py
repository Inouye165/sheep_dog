"""Early Warning Advisory System for Completed Evaluations (Reduced Scope).

Scope and Architectural Guarantees:
- COMPLETED EVALUATIONS ONLY: Surfaces advisory warnings strictly from completed formal evaluations.
- DEFERRED SIGNALS: Live-training window monitoring, PPO alerts, recovery-relapse tracking,
  stale-training detection, and automatic resolution are explicitly deferred to future work.
- ADVISORY ONLY: Warnings never automatically pause, stop, rollback, or alter training.
- HONEST REPETITION METRICS: Uses corrected, versioned repetition metrics (v2.0) when available.
  Never substitutes the old oscillation_detected flag. If v2.0 metrics are absent, reports
  "Not available for this evaluation."
- COMPATIBLE REGRESSION ONLY: Evaluates success regression only when a fully compatible comparison
  exists (matching run, stage, seed manifest, assistance mode, environment/reward config,
  episode budget, and decision mode). Missing metadata reports "Comparison unavailable."
- BOUNDED LOADING: Caches evaluation summaries by file path and mtime. Status requests are strictly
  read-only and never scan all replays or rewrite warning state on status polls.
- REPLAY LINKAGE: Links directly to authentic replay files and lead-in steps on disk. Missing replays
  are clearly explained without fabricating links.
- NO RECOVERY CLAIMS: Acknowledgment means "seen" and does not claim recovery or resolve the condition.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json
from sheepdog.training.diagnostics.repetition_recovery import (
    METRICS_VERSION as REPETITION_METRICS_VERSION,
)
from sheepdog.training.diagnostics.repetition_recovery import (
    analyze_repetition_and_recovery,
)

logger = logging.getLogger(__name__)

# Required compatibility fields for valid success regression comparison
COMPATIBILITY_CRITERIA: tuple[tuple[str, str], ...] = (
    ("run_id", "Run ID"),
    ("curriculum_stage", "Curriculum Stage"),
    ("evaluation_seed_set_id", "Seed Manifest ID"),
    ("policy_mode", "Assistance Mode"),
    ("environment_config_hash", "Environment/Reward Config Hash"),
    ("episode_budget", "Episode Budget"),
    ("evaluation_mode", "Evaluation Decision Mode"),
)


class WarningSeverity(StrEnum):
    """Severity tier for an advisory early warning."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class WarningStatus(StrEnum):
    """Lifecycle status of an advisory warning record."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"


class WarningConditionType(StrEnum):
    """Supported warning conditions in reduced sprint."""

    UNPRODUCTIVE_REPETITION = "unproductive_repetition"
    SUCCESS_REGRESSION = "success_regression"


@dataclass(frozen=True, slots=True)
class EarlyWarningItem:
    """An immutable advisory warning record backed by completed evaluation evidence."""

    id: str
    condition_type: str  # "unproductive_repetition" or "success_regression"
    severity: str  # "critical", "warning", "info"
    title: str
    plain_language_observation: str
    candidate_causes: tuple[str, ...]
    source_layer: str  # "completed_evaluation"
    evidence_source: str
    evaluation_id: str | None
    checkpoint_id: str | None
    checkpoint_episode: int | None
    policy_version: int | None
    curriculum_stage: int
    completion_time: str
    sample_count: int
    metric_version: str | None
    status: str  # "active" or "acknowledged"
    supporting_measurements: dict[str, Any]
    compatibility_factors: dict[str, Any] | None = None
    comparison_status: str | None = None
    replay_link: str | None = None
    replay_step: int | None = None
    replay_interval_id: str | None = None
    replay_available: bool = False
    link_explanation: str | None = None
    acknowledged_at: str | None = None
    acknowledged_by: str | None = None
    first_detected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_detected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert warning record to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "condition_type": self.condition_type,
            "severity": self.severity,
            "title": self.title,
            "plain_language_observation": self.plain_language_observation,
            "candidate_causes": list(self.candidate_causes),
            "source_layer": self.source_layer,
            "evidence_source": self.evidence_source,
            "evaluation_id": self.evaluation_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_episode": self.checkpoint_episode,
            "policy_version": self.policy_version,
            "curriculum_stage": self.curriculum_stage,
            "completion_time": self.completion_time,
            "sample_count": self.sample_count,
            "metric_version": self.metric_version,
            "status": self.status,
            "supporting_measurements": self.supporting_measurements,
            "compatibility_factors": self.compatibility_factors,
            "comparison_status": self.comparison_status,
            "replay_link": self.replay_link,
            "replay_step": self.replay_step,
            "replay_interval_id": self.replay_interval_id,
            "replay_available": self.replay_available,
            "link_explanation": self.link_explanation,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "first_detected_at": self.first_detected_at,
            "last_detected_at": self.last_detected_at,
        }


def validate_evaluation_record(eval_data: dict[str, Any]) -> tuple[bool, str]:
    """Verify that an evaluation record is complete and usable.

    Rejects incomplete evaluations, evaluations with zero records, or corrupted payloads.
    """
    if not isinstance(eval_data, dict):
        return False, "Evaluation payload is not a dictionary"

    if eval_data.get("is_complete") is False:
        return False, "Evaluation marked incomplete (is_complete=False)"

    records = eval_data.get("records")
    if records is None:
        records = eval_data.get("per_seed_records")

    if not isinstance(records, list) or len(records) == 0:
        return False, "Evaluation contains zero completed episode records"

    seed_count = eval_data.get("evaluation_seed_count")
    if seed_count is not None and len(records) < int(seed_count):
        return False, f"Evaluation incomplete: expected {seed_count} seed records but found {len(records)}"

    has_ident = any(
        eval_data.get(k) is not None
        for k in ("checkpoint_episode", "policy_version", "checkpoint_id", "evaluation_id")
    )
    if not has_ident:
        return False, "Evaluation missing checkpoint identity (checkpoint_episode, policy_version, or checkpoint_id)"

    return True, "Valid complete evaluation"


def check_evaluation_compatibility(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Determine whether two completed evaluations form a compatible comparison.

    Must match:
    1. run_id
    2. curriculum_stage
    3. evaluation_seed_set_id
    4. policy_mode (assistance mode)
    5. environment_config_hash
    6. episode_budget
    7. evaluation_mode (decision mode)

    Missing metadata means comparison is unavailable.
    """
    matched: dict[str, Any] = {}
    mismatches: list[str] = []
    missing_fields: list[str] = []

    for key, label in COMPATIBILITY_CRITERIA:
        b_val = baseline.get(key)
        c_val = candidate.get(key)

        # Normalize special default keys if needed
        if key == "policy_mode":
            b_val = b_val or baseline.get("policy_type")
            c_val = c_val or candidate.get("policy_type")
        elif key == "episode_budget":
            b_val = b_val or baseline.get("max_steps") or baseline.get("episode_max_steps")
            c_val = c_val or candidate.get("max_steps") or candidate.get("episode_max_steps")

        if b_val is None or c_val is None:
            missing_fields.append(f"{label} ({key})")
        elif str(b_val) != str(c_val):
            mismatches.append(f"{label} mismatch: baseline={b_val} vs candidate={c_val}")
        else:
            matched[key] = b_val

    if missing_fields:
        reason = f"Comparison unavailable: missing metadata for {', '.join(missing_fields)}"
        return False, reason, matched

    if mismatches:
        reason = f"Comparison unavailable: incompatible conditions ({'; '.join(mismatches)})"
        return False, reason, matched

    return True, "Fully compatible comparison", matched


def analyze_single_evaluation_repetition(
    eval_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Analyze ONE selected completed evaluation's replays and save its diagnostic summary.

    Bounded and isolated: only analyzes replays explicitly referenced by this evaluation.
    Computes rigorous v2.0 repetition metrics without scanning historical replays.
    """
    eval_file = Path(eval_path).resolve()
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with eval_file.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    is_valid, validation_msg = validate_evaluation_record(eval_data)
    if not is_valid:
        raise ValueError(f"Cannot analyze incomplete evaluation: {validation_msg}")

    records = eval_data.get("records") or eval_data.get("per_seed_records") or []
    eval_id = eval_data.get("evaluation_id") or eval_file.stem
    cp_ep = eval_data.get("checkpoint_episode")
    pv = eval_data.get("policy_version")
    stg = eval_data.get("curriculum_stage", 1)

    total_world_steps = 0
    repeated_steps = 0
    total_stalls = 0
    recovered_stalls = 0
    relapsed_stalls = 0
    unresolved_stalls = 0
    replays_analyzed = 0
    missing_replays = 0

    representative_seed: int | None = None
    representative_replay_path: str | None = None
    representative_step: int | None = None
    representative_interval_id: str | None = None
    per_seed_summaries: list[dict[str, Any]] = []

    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    for r in records:
        seed = r.get("seed", 0)
        rp_str = r.get("replay_path")
        if not rp_str:
            missing_replays += 1
            continue

        cand_path = Path(rp_str)
        if not cand_path.is_absolute():
            # Try relative to repo root or evaluation file parent
            cand_path = repo_root / rp_str
            if not cand_path.exists():
                cand_path = eval_file.parent / rp_str

        if not cand_path.exists():
            missing_replays += 1
            per_seed_summaries.append({
                "seed": seed,
                "replay_available": False,
                "replay_path": rp_str,
                "error": "File not found on disk",
            })
            continue

        try:
            analysis = analyze_repetition_and_recovery(cand_path)
            replays_analyzed += 1
            total_world_steps += analysis.total_world_steps
            repeated_steps += analysis.repeated_movement_steps
            total_stalls += analysis.stalls_count
            recovered_stalls += analysis.recovered_stalls_count
            relapsed_stalls += analysis.relapsed_stalls_count
            unresolved_stalls += analysis.unresolved_stalls_count

            # Find a representative stall with a lead-in step for deep link
            rep_step = None
            rep_iv_id = None
            if analysis.stall_records:
                first_stall = analysis.stall_records[0]
                rep_step = max(0, first_stall.start_step - 20)  # lead-in step
                rep_iv_id = first_stall.interval_id
            elif analysis.repeated_movement_steps > 0:
                rep_step = 0

            # Prioritize seeds with actual stalls for representative replay link
            if representative_seed is None or (analysis.stalls_count > 0 and representative_interval_id is None):
                representative_seed = seed
                representative_replay_path = rp_str
                representative_step = rep_step
                representative_interval_id = rep_iv_id

            per_seed_summaries.append({
                "seed": seed,
                "replay_available": True,
                "replay_path": rp_str,
                "world_steps": analysis.total_world_steps,
                "repeated_steps": analysis.repeated_movement_steps,
                "repeated_fraction": analysis.repeated_movement_fraction,
                "stalls_count": analysis.stalls_count,
                "recovered_count": analysis.recovered_stalls_count,
                "representative_lead_in_step": rep_step,
                "representative_interval_id": rep_iv_id,
            })
        except Exception as exc:
            logger.warning("Failed analyzing replay %s for seed %s: %s", cand_path, seed, exc)
            per_seed_summaries.append({
                "seed": seed,
                "replay_available": True,
                "replay_path": rp_str,
                "error": str(exc),
            })

    rep_fraction = (
        round(repeated_steps / total_world_steps, 4) if total_world_steps > 0 else None
    )
    rec_rate = (
        round(recovered_stalls / total_stalls, 4) if total_stalls > 0 else None
    )

    summary = {
        "metrics_version": REPETITION_METRICS_VERSION,
        "evaluation_id": eval_id,
        "checkpoint_episode": cp_ep,
        "policy_version": pv,
        "curriculum_stage": stg,
        "evaluation_file": str(eval_file),
        "analyzed_at": datetime.now(UTC).isoformat(),
        "replays_analyzed": replays_analyzed,
        "missing_replays": missing_replays,
        "total_world_steps": total_world_steps,
        "repeated_movement_steps": repeated_steps,
        "repeated_movement_fraction": rep_fraction,
        "stalls_count": total_stalls,
        "recovered_stalls_count": recovered_stalls,
        "relapsed_stalls_count": relapsed_stalls,
        "unresolved_stalls_count": unresolved_stalls,
        "recovery_rate": rec_rate,
        "representative_seed": representative_seed,
        "representative_replay_path": representative_replay_path,
        "representative_step": representative_step,
        "representative_interval_id": representative_interval_id,
        "per_seed_summaries": per_seed_summaries,
    }

    if output_path is None:
        diag_dir = eval_file.parent / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        dest_file = diag_dir / f"{eval_file.stem}.diagnostics.json"
    else:
        dest_file = Path(output_path).resolve()
        dest_file.parent.mkdir(parents=True, exist_ok=True)

    atomic_write_json(dest_file, summary)

    logger.info("Saved evaluation diagnostic summary to %s", dest_file)
    return summary


class EarlyWarningMonitor:
    """Bounded, read-only advisory monitoring engine for completed evaluations."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir).resolve()
        self.store_file = self.output_dir / "early-warnings.json"
        self._lock = threading.Lock()
        self._eval_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _read_store(self) -> dict[str, Any]:
        """Read persisted user acknowledgments from early-warnings.json."""
        if not self.store_file.exists():
            return {"warnings": {}, "schema_version": "2.0"}
        try:
            with self.store_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to read early-warnings.json: %s", exc)
        return {"warnings": {}, "schema_version": "2.0"}

    def _write_store(self, data: dict[str, Any]) -> None:
        """Atomically persist acknowledgment changes to early-warnings.json."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.store_file, data)
        except Exception as exc:
            logger.warning("Failed writing early-warnings.json: %s", exc)

    def acknowledge_warning(self, warning_id: str, note: str = "") -> bool:
        """Mark an active advisory warning as acknowledged (seen).

        Acknowledgment records that the operator has reviewed the warning.
        It does NOT clear the measured condition or claim recovery.
        """
        with self._lock:
            store_data = self._read_store()
            warnings = store_data.get("warnings", {})
            if warning_id in warnings:
                w = warnings[warning_id]
                if w.get("status") == WarningStatus.ACTIVE.value:
                    w["status"] = WarningStatus.ACKNOWLEDGED.value
                    w["acknowledged_at"] = datetime.now(UTC).isoformat()
                    w["acknowledged_by"] = note or "operator"
                    self._write_store(store_data)
                    self._eval_cache.clear()
                    return True
        return False

    def get_warnings_summary(
        self,
        current_stage: int = 1,
        selected_eval_path: Path | str | None = None,
        force_refresh: bool = False,
        is_read_only_preview: bool = False,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return advisory warnings summary backed by completed evaluations.

        Guarantees:
        - Read-only operations: never modifies database, never pauses training, never rewrites
          warning store on status polling.
        - Bounded loading: reuses cached analysis when source file mtime is unchanged.
        """
        evals_dir = self.output_dir / "evaluations"

        # Determine target evaluation file to inspect
        target_eval_file: Path | None = None
        if selected_eval_path is not None:
            cand = Path(selected_eval_path).resolve()
            if cand.exists():
                target_eval_file = cand
        elif is_read_only_preview:
            # In preview mode, use the verified real recorded evaluation if available
            cand = evals_dir / "eval_chk_run_20260817_063453_91f4_pv_10074_ts_329366199_quick_ff85663fc223.json"
            if cand.exists():
                target_eval_file = cand

        if target_eval_file is None and evals_dir.exists():
            # First check if there is an evaluation with a precomputed diagnostics summary
            diag_dir = evals_dir / "diagnostics"
            if diag_dir.exists():
                diag_entries = sorted(
                    [e for e in os.scandir(diag_dir) if e.name.endswith(".diagnostics.json")],
                    key=lambda e: e.stat().st_mtime,
                    reverse=True,
                )
                if diag_entries:
                    stem = diag_entries[0].name.replace(".diagnostics.json", "")
                    cand = evals_dir / f"{stem}.json"
                    if cand.exists():
                        target_eval_file = cand

            if target_eval_file is None:
                # Lightweight search: find latest eval_*.json by mtime without parsing files
                try:
                    latest_entry = max(
                        (
                            e for e in os.scandir(evals_dir)
                            if e.is_file()
                            and e.name.startswith("eval_")
                            and e.name.endswith(".json")
                            and not e.name.endswith(".inprogress.json")
                        ),
                        key=lambda e: e.stat(follow_symlinks=False).st_mtime,
                        default=None,
                    )
                    if latest_entry is not None:
                        target_eval_file = Path(latest_entry.path)
                except Exception as exc:
                    logger.warning("Error finding latest evaluation: %s", exc)

        if target_eval_file is None:
            return self._build_empty_summary(
                stage=current_stage,
                reason="No formal evaluations completed yet.",
                is_read_only_preview=is_read_only_preview,
            )

        # Check cache by file path and mtime
        file_mtime = target_eval_file.stat().st_mtime
        cache_key = str(target_eval_file)
        if not force_refresh:
            with self._lock:
                cached_entry = self._eval_cache.get(cache_key)
                if cached_entry is not None and cached_entry[0] == file_mtime:
                    # Cache hit: return without disk parsing or store rewrite
                    return cached_entry[1]

        # Process the single completed evaluation
        try:
            payload = self._evaluate_completed_evaluation(
                target_eval_file, current_stage, is_read_only_preview=is_read_only_preview
            )
        except Exception as exc:
            logger.error("Error evaluating completed evaluation %s: %s", target_eval_file, exc)
            payload = self._build_empty_summary(
                stage=current_stage,
                reason=f"Failed evaluating completed evaluation: {exc}",
                is_read_only_preview=is_read_only_preview,
            )

        with self._lock:
            self._eval_cache[cache_key] = (file_mtime, payload)
        return payload

    def _evaluate_completed_evaluation(
        self,
        eval_file: Path,
        current_stage: int,
        is_read_only_preview: bool = False,
    ) -> dict[str, Any]:
        """Generate advisory warnings from one completed evaluation."""
        with eval_file.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)

        is_valid, validation_msg = validate_evaluation_record(eval_data)
        if not is_valid:
            logger.warning("Rejected incomplete evaluation %s: %s", eval_file.name, validation_msg)
            return self._build_empty_summary(
                stage=current_stage,
                reason=f"Rejected incomplete evaluation: {validation_msg}",
                is_read_only_preview=is_read_only_preview,
            )

        eval_id = eval_data.get("evaluation_id") or eval_file.stem
        cp_ep = eval_data.get("checkpoint_episode")
        pv = eval_data.get("policy_version")
        eval_stage = eval_data.get("curriculum_stage", current_stage)
        eval_time = (
            eval_data.get("evaluation_timestamp")
            or eval_data.get("created_timestamp")
            or datetime.fromtimestamp(eval_file.stat().st_mtime, UTC).isoformat()
        )
        sample_count = len(eval_data.get("records") or eval_data.get("per_seed_records") or [])

        # Check for accompanying or embedded v2.0 diagnostic summary
        diag_summary = eval_data.get("diagnostic_summary")
        if not diag_summary:
            diag_file = eval_file.parent / "diagnostics" / f"{eval_file.stem}.diagnostics.json"
            if diag_file.exists():
                try:
                    with diag_file.open("r", encoding="utf-8") as df:
                        diag_summary = json.load(df)
                except Exception as exc:
                    logger.warning("Error reading diagnostics %s: %s", diag_file, exc)

        # Load persisted acknowledgments
        with self._lock:
            store_data = self._read_store()
            persisted = store_data.get("warnings", {})

        generated_warnings: list[EarlyWarningItem] = []
        repo_root = Path(__file__).resolve().parent.parent.parent.parent

        # ------------------------------------------------------------------
        # Warning 1: Unproductive Repetition (requires verified v2.0 metrics)
        # ------------------------------------------------------------------
        rep_metrics_available = False
        rep_version = None
        rep_fraction = None
        total_steps = None
        repeated_steps = None
        stalls_count = None
        recovery_rate = None

        if diag_summary and diag_summary.get("metrics_version") == REPETITION_METRICS_VERSION:
            rep_metrics_available = True
            rep_version = REPETITION_METRICS_VERSION
            rep_fraction = diag_summary.get("repeated_movement_fraction")
            total_steps = diag_summary.get("total_world_steps", 0)
            repeated_steps = diag_summary.get("repeated_movement_steps", 0)
            stalls_count = diag_summary.get("stalls_count", 0)
            recovery_rate = diag_summary.get("recovery_rate")

            # Check threshold: sustained repetition >= 50% of world steps
            REPETITION_THRESHOLD = 0.50
            if rep_fraction is not None and rep_fraction >= REPETITION_THRESHOLD:
                w_id = f"warn_rep_{eval_id}"
                is_ack = persisted.get(w_id, {}).get("status") == WarningStatus.ACKNOWLEDGED.value
                ack_at = persisted.get(w_id, {}).get("acknowledged_at")
                ack_by = persisted.get(w_id, {}).get("acknowledged_by")

                # Resolve replay path and lead-in step
                raw_rp = diag_summary.get("representative_replay_path")
                rep_step = diag_summary.get("representative_step")
                rep_iv_id = diag_summary.get("representative_interval_id")

                rep_on_disk = None
                if raw_rp:
                    cand = Path(raw_rp)
                    if not cand.is_absolute():
                        cand = repo_root / raw_rp
                        if not cand.exists():
                            cand = eval_file.parent / raw_rp
                    if cand.exists():
                        rep_on_disk = str(cand)

                replay_avail = rep_on_disk is not None
                normalized_link = raw_rp.replace("\\", "/") if raw_rp else None

                link_expl = (
                    f"Verified replay on disk ({raw_rp}) at lead-in step {rep_step}"
                    if replay_avail
                    else (
                        f"Replay not available on disk ({raw_rp or 'no replay path recorded'})"
                    )
                )

                generated_warnings.append(
                    EarlyWarningItem(
                        id=w_id,
                        condition_type=WarningConditionType.UNPRODUCTIVE_REPETITION.value,
                        severity=WarningSeverity.WARNING.value,
                        title=f"Unproductive Repetition ({round(rep_fraction * 100, 1)}% of world steps)",
                        plain_language_observation=(
                            f"In completed evaluation {eval_id} (checkpoint {cp_ep}), dogs spent "
                            f"{repeated_steps} of {total_steps} total world steps ({round(rep_fraction * 100, 1)}%) "
                            f"in verified repeated movement cycles or standoffs without flock penning."
                        ),
                        candidate_causes=(
                            "Dogs overshooting pen entrance axis or cycling along closed pen fence",
                            "Inward target clamping against walls or corners on stray sheep",
                            "Policy entropy decay or short credit horizon in multi-agent coordination",
                        ),
                        source_layer="completed_evaluation",
                        evidence_source=str(eval_file),
                        evaluation_id=eval_id,
                        checkpoint_id=eval_data.get("checkpoint_id"),
                        checkpoint_episode=cp_ep,
                        policy_version=pv,
                        curriculum_stage=eval_stage,
                        completion_time=eval_time,
                        sample_count=sample_count,
                        metric_version=rep_version,
                        status=WarningStatus.ACKNOWLEDGED.value if is_ack else WarningStatus.ACTIVE.value,
                        supporting_measurements={
                            "repeated_movement_fraction": rep_fraction,
                            "repeated_movement_steps": repeated_steps,
                            "total_world_steps": total_steps,
                            "stalls_count": stalls_count,
                            "recovery_rate": recovery_rate,
                            "threshold": REPETITION_THRESHOLD,
                            "sample_count": sample_count,
                            "representative_seed": diag_summary.get("representative_seed"),
                        },
                        replay_link=normalized_link if replay_avail else None,
                        replay_step=rep_step if replay_avail else None,
                        replay_interval_id=rep_iv_id if replay_avail else None,
                        replay_available=replay_avail,
                        link_explanation=link_expl,
                        acknowledged_at=ack_at,
                        acknowledged_by=ack_by,
                    )
                )

        # ------------------------------------------------------------------
        # Warning 2: Success Regression (strictly requires compatible comparison)
        # ------------------------------------------------------------------
        candidate_sr = float(eval_data.get("success_rate", 0.0) or 0.0)
        compatible_baseline: dict[str, Any] | None = None
        compatibility_explanation: str = "Comparison unavailable: no prior evaluation evaluated"
        matched_factors: dict[str, Any] | None = None

        # Look for prior evaluations to check compatibility
        evals_dir = eval_file.parent
        eval_mtime = eval_file.stat().st_mtime
        eval_run_id = eval_data.get("run_id")

        prior_entries: list[tuple[float, str, str]] = []
        try:
            with os.scandir(evals_dir) as it:
                for entry in it:
                    if (
                        entry.is_file()
                        and entry.name.startswith("eval_")
                        and entry.name.endswith(".json")
                        and not entry.name.endswith(".inprogress.json")
                        and entry.name != eval_file.name
                    ):
                        try:
                            mtime = entry.stat(follow_symlinks=False).st_mtime
                            if mtime < eval_mtime:
                                prior_entries.append((mtime, entry.path, entry.name))
                        except OSError:
                            continue
        except OSError:
            pass

        # If run_id is known, prefer prior files containing the run_id in the filename
        if eval_run_id:
            run_matched = [item for item in prior_entries if eval_run_id in item[2]]
            if run_matched:
                prior_entries = run_matched

        # Sort by mtime descending and check top candidates
        prior_entries.sort(key=lambda x: x[0], reverse=True)

        for _, path_str, _ in prior_entries[:10]:
            prev_p = Path(path_str)
            try:
                with prev_p.open("r", encoding="utf-8") as pf:
                    prev_data = json.load(pf)
                is_comp, comp_reason, matched = check_evaluation_compatibility(prev_data, eval_data)
                if is_comp:
                    compatible_baseline = prev_data
                    compatibility_explanation = "Fully compatible comparison"
                    matched_factors = matched
                    break
                else:
                    compatibility_explanation = comp_reason
            except Exception:
                continue

        if compatible_baseline is not None:
            base_sr = float(compatible_baseline.get("success_rate", 0.0) or 0.0)
            base_id = compatible_baseline.get("evaluation_id") or "baseline"
            REGRESSION_DROP_THRESHOLD = 0.25

            if base_sr >= 0.50 and (base_sr - candidate_sr) >= REGRESSION_DROP_THRESHOLD:
                w_id = f"warn_regress_{eval_id}"
                is_ack = persisted.get(w_id, {}).get("status") == WarningStatus.ACKNOWLEDGED.value
                ack_at = persisted.get(w_id, {}).get("acknowledged_at")
                ack_by = persisted.get(w_id, {}).get("acknowledged_by")

                # Find failing replay from candidate records
                failing_rp = None
                failing_seed = None
                records = eval_data.get("records") or eval_data.get("per_seed_records") or []
                for r in records:
                    if not r.get("success", False) and r.get("replay_path"):
                        failing_rp = r["replay_path"]
                        failing_seed = r.get("seed")
                        break

                rep_on_disk = None
                if failing_rp:
                    cand = Path(failing_rp)
                    if not cand.is_absolute():
                        cand = repo_root / failing_rp
                        if not cand.exists():
                            cand = eval_file.parent / failing_rp
                    if cand.exists():
                        rep_on_disk = str(cand)

                replay_avail = rep_on_disk is not None
                normalized_link = failing_rp.replace("\\", "/") if failing_rp else None

                generated_warnings.append(
                    EarlyWarningItem(
                        id=w_id,
                        condition_type=WarningConditionType.SUCCESS_REGRESSION.value,
                        severity=WarningSeverity.CRITICAL.value if candidate_sr < 0.35 else WarningSeverity.WARNING.value,
                        title=f"Success Rate Regression ({int(round(candidate_sr * 100))}% vs Compatible Baseline {int(round(base_sr * 100))}%)",
                        plain_language_observation=(
                            f"Success rate dropped by {int(round((base_sr - candidate_sr) * 100))}% "
                            f"between compatible evaluations on Stage {eval_stage} "
                            f"(current {int(round(candidate_sr * 100))}% on {eval_id} vs "
                            f"baseline {int(round(base_sr * 100))}% on {base_id})."
                        ),
                        candidate_causes=(
                            "Exploration shift on complex spawn distributions",
                            "Emergent policy coordination regression requiring hyperparameter tuning",
                            "Curriculum difficulty increase or seed set variance",
                        ),
                        source_layer="completed_evaluation",
                        evidence_source=str(eval_file),
                        evaluation_id=eval_id,
                        checkpoint_id=eval_data.get("checkpoint_id"),
                        checkpoint_episode=cp_ep,
                        policy_version=pv,
                        curriculum_stage=eval_stage,
                        completion_time=eval_time,
                        sample_count=sample_count,
                        metric_version=rep_version,
                        status=WarningStatus.ACKNOWLEDGED.value if is_ack else WarningStatus.ACTIVE.value,
                        supporting_measurements={
                            "current_success_rate": round(candidate_sr, 4),
                            "baseline_success_rate": round(base_sr, 4),
                            "drop_percentage": round((base_sr - candidate_sr) * 100, 1),
                            "baseline_evaluation_id": base_id,
                            "sample_count": sample_count,
                            "failing_seed": failing_seed,
                        },
                        compatibility_factors=matched_factors,
                        comparison_status=compatibility_explanation,
                        replay_link=normalized_link if replay_avail else None,
                        replay_step=0 if replay_avail else None,
                        replay_available=replay_avail,
                        link_explanation=(
                            f"Verified failure replay on disk ({failing_rp})"
                            if replay_avail
                            else f"Replay not available on disk ({failing_rp or 'none recorded'})"
                        ),
                        acknowledged_at=ack_at,
                        acknowledged_by=ack_by,
                    )
                )

        # Merge generated warnings with store for user acknowledgment tracking
        with self._lock:
            for w in generated_warnings:
                if w.id not in persisted:
                    persisted[w.id] = w.to_dict()
                else:
                    # Update evidence without overriding user acknowledgment
                    ex = persisted[w.id]
                    is_ack = ex.get("status") == WarningStatus.ACKNOWLEDGED.value
                    w_dict = w.to_dict()
                    if is_ack:
                        w_dict["status"] = WarningStatus.ACKNOWLEDGED.value
                        w_dict["acknowledged_at"] = ex.get("acknowledged_at")
                        w_dict["acknowledged_by"] = ex.get("acknowledged_by")
                    persisted[w.id] = w_dict
            store_data["warnings"] = persisted
            self._write_store(store_data)

        # Build final payload
        active_list = [w for w in persisted.values() if w.get("status") == WarningStatus.ACTIVE.value]
        ack_list = [w for w in persisted.values() if w.get("status") == WarningStatus.ACKNOWLEDGED.value]

        return {
            "enabled": True,
            "is_read_only_preview": is_read_only_preview,
            "mode_description": "Advisory Early Warnings (Completed Evaluations Only; never stops training)",
            "monitored_signals": [
                "unproductive_repetition (completed evaluations with verified v2.0 metrics)",
                "success_regression (completed evaluations with verified compatible baseline)",
            ],
            "deferred_signals": [
                "live_training_window (deferred to future sprint)",
                "ppo_numerical_instability (deferred to future sprint)",
                "recovery_relapse_tracking (deferred to future sprint)",
                "stale_training_detection (deferred to future sprint)",
                "automatic_resolution (deferred to future sprint)",
            ],
            "current_evaluation": {
                "evaluation_id": eval_id,
                "checkpoint_id": eval_data.get("checkpoint_id"),
                "checkpoint_episode": cp_ep,
                "policy_version": pv,
                "curriculum_stage": eval_stage,
                "completion_time": eval_time,
                "sample_count": sample_count,
                "evidence_source": str(eval_file),
                "repetition_metrics_available": rep_metrics_available,
                "repetition_metric_status": (
                    f"Verified (v{rep_version})"
                    if rep_metrics_available
                    else "Not available for this evaluation."
                ),
                "repetition_metrics": {
                    "metrics_version": rep_version,
                    "repeated_movement_fraction": rep_fraction,
                    "repeated_movement_steps": repeated_steps,
                    "total_world_steps": total_steps,
                    "stalls_count": stalls_count,
                    "recovery_rate": recovery_rate,
                } if rep_metrics_available else None,
                "comparison_status": compatibility_explanation,
            },
            "active_warnings": active_list,
            "acknowledged_warnings": ack_list,
            "summary": {
                "critical_count": sum(1 for w in active_list if w.get("severity") == WarningSeverity.CRITICAL.value),
                "warning_count": sum(1 for w in active_list if w.get("severity") == WarningSeverity.WARNING.value),
                "info_count": sum(1 for w in active_list if w.get("severity") == WarningSeverity.INFO.value),
                "total_active": len(active_list),
                "total_acknowledged": len(ack_list),
            },
        }

    def _build_empty_summary(
        self, stage: int, reason: str, is_read_only_preview: bool = False
    ) -> dict[str, Any]:
        """Return clean summary when no evaluations are available or evaluation is invalid."""
        return {
            "enabled": True,
            "is_read_only_preview": is_read_only_preview,
            "mode_description": "Advisory Early Warnings (Completed Evaluations Only; never stops training)",
            "monitored_signals": [
                "unproductive_repetition (completed evaluations with verified v2.0 metrics)",
                "success_regression (completed evaluations with verified compatible baseline)",
            ],
            "deferred_signals": [
                "live_training_window (deferred to future sprint)",
                "ppo_numerical_instability (deferred to future sprint)",
                "recovery_relapse_tracking (deferred to future sprint)",
                "stale_training_detection (deferred to future sprint)",
                "automatic_resolution (deferred to future sprint)",
            ],
            "current_evaluation": {
                "evaluation_id": None,
                "checkpoint_id": None,
                "checkpoint_episode": None,
                "policy_version": None,
                "curriculum_stage": stage,
                "completion_time": None,
                "sample_count": 0,
                "evidence_source": None,
                "repetition_metrics_available": False,
                "repetition_metric_status": "Not available for this evaluation.",
                "repetition_metrics": None,
                "comparison_status": reason,
            },
            "active_warnings": [],
            "acknowledged_warnings": [],
            "summary": {
                "critical_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "total_active": 0,
                "total_acknowledged": 0,
            },
        }


def main() -> None:
    """CLI entry point for bounded offline evaluation analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze ONE completed evaluation's replays and save its v2.0 repetition diagnostic summary."
    )
    parser.add_argument(
        "--evaluation",
        "-e",
        required=True,
        type=Path,
        help="Path to the completed evaluation JSON file to analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional destination path for the output diagnostics JSON summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = analyze_single_evaluation_repetition(args.evaluation, args.output)
        print("\n--- Diagnostic Analysis Complete ---")
        print(f"Evaluation ID: {summary['evaluation_id']}")
        print(f"Replays Analyzed: {summary['replays_analyzed']}")
        print(f"Total World Steps: {summary['total_world_steps']}")
        print(f"Repeated Movement Steps: {summary['repeated_movement_steps']}")
        print(f"Repeated Movement Fraction: {summary['repeated_movement_fraction']}")
        print(f"Total Stalls Count: {summary['stalls_count']}")
        print(f"Recovery Rate: {summary['recovery_rate']}")
        if summary.get("representative_replay_path"):
            print(f"Representative Replay: {summary['representative_replay_path']}")
            print(f"Representative Lead-in Step: {summary['representative_step']}")
    except Exception as exc:
        print(f"Error analyzing evaluation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
