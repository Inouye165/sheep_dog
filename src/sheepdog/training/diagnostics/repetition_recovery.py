"""Production Repetition and Recovery Metrics Engine (v2.0).

Provides rigorous, testable metrics for:
1. Multi-agent repeated movement and oscillation (distinguishing stationary holding,
   productive waiting, active multi-step periodic loops, and productive repositioning).
2. Sustained recovery vs relapse vs unresolved stalls (distinguishing true sustained exit
   into lasting route/penning progress from transient distance jitter, recurrence, or
   incidental eventual episode success).

Preserves versioned metric contracts across ordinary training early warning monitoring
and hyperparameter experiment safeguards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from sheepdog.training.diagnostics.temporal_interaction import (
    EpisodeTemporalAnalysis,
    TemporalInteractionInterval,
    analyze_episode_temporal_interactions,
    load_replay_bundle,
)

METRICS_VERSION = "2.0"


class DogMovementType(StrEnum):
    """Fine-grained classification of an individual dog's movement."""

    STATIONARY_HOLDING = "stationary_holding"
    PRODUCTIVE_WAITING = "productive_waiting"
    ACTIVE_REPETITION = "active_repetition"
    PRODUCTIVE_REPOSITIONING = "productive_repositioning"
    EXPLORING_OR_ADVANCING = "exploring_or_advancing"


class StallOutcome(StrEnum):
    """Mutually exclusive outcome classification for a candidate stall interval."""

    RECOVERED = "recovered"
    RELAPSED = "relapsed"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class StallOutcomeRecord:
    """Detailed record of a single stall interval and its aftermath outcome."""

    interval_id: str
    classification: str
    unpenned_sheep_count: int
    start_step: int
    end_step: int
    duration_steps: int
    outcome: StallOutcome
    lasting_progress_evidence: str
    relapse_step: int | None = None
    eventual_episode_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert stall record to dictionary."""
        return {
            "interval_id": self.interval_id,
            "classification": self.classification,
            "unpenned_sheep_count": self.unpenned_sheep_count,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration_steps": self.duration_steps,
            "outcome": self.outcome.value,
            "lasting_progress_evidence": self.lasting_progress_evidence,
            "relapse_step": self.relapse_step,
            "eventual_episode_success": self.eventual_episode_success,
        }


@dataclass(frozen=True, slots=True)
class UnpennedBucketMetrics:
    """Metrics aggregated for a specific unpenned sheep count (1..6)."""

    unpenned_count: int
    episodes_observed: int = 0
    world_steps: int = 0
    repeated_movement_steps: int = 0
    repeated_movement_fraction: float | None = None
    stall_intervals: int = 0
    recovered_intervals: int = 0
    relapsed_intervals: int = 0
    unresolved_intervals: int = 0
    recovery_rate: float | None = None
    relapse_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stall record to dictionary."""
        return {
            "unpenned_count": self.unpenned_count,
            "episodes_observed": self.episodes_observed,
            "world_steps": self.world_steps,
            "repeated_movement_steps": self.repeated_movement_steps,
            "repeated_movement_fraction": (
                round(self.repeated_movement_fraction, 4)
                if self.repeated_movement_fraction is not None
                else None
            ),
            "stall_intervals": self.stall_intervals,
            "recovered_intervals": self.recovered_intervals,
            "relapsed_intervals": self.relapsed_intervals,
            "unresolved_intervals": self.unresolved_intervals,
            "recovery_rate": (
                round(self.recovery_rate, 4) if self.recovery_rate is not None else None
            ),
            "relapse_rate": (
                round(self.relapse_rate, 4) if self.relapse_rate is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RepetitionRecoveryAnalysis:
    """Complete, verified repetition and recovery evaluation for an episode."""

    metrics_version: str
    total_world_steps: int
    repeated_movement_steps: int
    repeated_movement_fraction: float | None
    stalls_count: int
    recovered_stalls_count: int
    relapsed_stalls_count: int
    unresolved_stalls_count: int
    recovery_rate: float | None
    relapse_rate: float | None
    unresolved_rate: float | None
    eventual_episode_success: bool
    stall_records: tuple[StallOutcomeRecord, ...] = field(default_factory=tuple)
    by_unpenned_count: dict[str, UnpennedBucketMetrics] = field(default_factory=dict)
    temporal_analysis: EpisodeTemporalAnalysis | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stall record to dictionary."""
        return {
            "metrics_version": self.metrics_version,
            "total_world_steps": self.total_world_steps,
            "repeated_movement_steps": self.repeated_movement_steps,
            "repeated_movement_fraction": (
                round(self.repeated_movement_fraction, 4)
                if self.repeated_movement_fraction is not None
                else None
            ),
            "stalls_count": self.stalls_count,
            "recovered_stalls_count": self.recovered_stalls_count,
            "relapsed_stalls_count": self.relapsed_stalls_count,
            "unresolved_stalls_count": self.unresolved_stalls_count,
            "recovery_rate": (
                round(self.recovery_rate, 4) if self.recovery_rate is not None else None
            ),
            "relapse_rate": (
                round(self.relapse_rate, 4) if self.relapse_rate is not None else None
            ),
            "unresolved_rate": (
                round(self.unresolved_rate, 4) if self.unresolved_rate is not None else None
            ),
            "eventual_episode_success": self.eventual_episode_success,
            "stall_records": [r.to_dict() for r in self.stall_records],
            "by_unpenned_count": {k: v.to_dict() for k, v in self.by_unpenned_count.items()},
        }


def _point_to_tuple(p: Any) -> tuple[int, int]:
    """Convert Point or tuple or dict to (x, y)."""
    if hasattr(p, "x") and hasattr(p, "y"):
        return (int(p.x), int(p.y))
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        return (int(p[0]), int(p[1]))
    if isinstance(p, dict) and "x" in p and "y" in p:
        return (int(p["x"]), int(p["y"]))
    return (0, 0)


def classify_dog_movement_pattern(
    positions_history: Sequence[Any],
    team_flock_progress: float = 0.0,
    team_penned_sheep: bool = False,
    in_productive_repositioning: bool = False,
) -> DogMovementType:
    """Classify the movement pattern of an individual dog over its recent position tail.

    Distinguishes:
    1. Stationary holding: dog stayed at exactly one coordinate (len(set) == 1).
    2. Productive waiting: dog is stationary WHILE teammates make flock progress or pen sheep.
    3. Active repetition: dog moves across 2, 3, or 4 coordinates in a sustained periodic cycle
       without team progress.
    4. Productive repositioning: dog moves and improves alignment or advances flock.
    5. Exploring / advancing: non-repeating movement.
    """
    if len(positions_history) < 4:
        return DogMovementType.EXPLORING_OR_ADVANCING

    tail = [_point_to_tuple(p) for p in positions_history[-8:]]
    last_4 = tail[-4:]
    unique_last_4 = set(last_4)

    # 1. Stationary check: Exactly one position in last 4 steps
    if len(unique_last_4) == 1:
        if team_flock_progress >= 0.5 or team_penned_sheep or in_productive_repositioning:
            return DogMovementType.PRODUCTIVE_WAITING
        return DogMovementType.STATIONARY_HOLDING

    # If the team is making verified progress or in productive repositioning,
    # the dog is not unproductively repeating
    if team_flock_progress >= 1.0 or team_penned_sheep or in_productive_repositioning:
        return DogMovementType.PRODUCTIVE_REPOSITIONING

    # 2. Check for active multi-step periodic loops:
    # Period 2: A-B-A-B over last 4 steps
    if (
        len(unique_last_4) == 2
        and last_4[0] == last_4[2]
        and last_4[1] == last_4[3]
        and last_4[0] != last_4[1]
    ):
        return DogMovementType.ACTIVE_REPETITION

    # Period 3: A-B-C-A-B-C over last 6 steps
    if len(tail) >= 6:
        last_6 = tail[-6:]
        unique_6 = set(last_6)
        if (
            len(unique_6) == 3
            and last_6[0] == last_6[3]
            and last_6[1] == last_6[4]
            and last_6[2] == last_6[5]
            and last_6[0] != last_6[1]
            and last_6[1] != last_6[2]
        ):
            return DogMovementType.ACTIVE_REPETITION

    # Period 4: A-B-C-D-A-B-C-D over last 8 steps
    if len(tail) >= 8:
        unique_8 = set(tail)
        if (
            len(unique_8) == 4
            and tail[0] == tail[4]
            and tail[1] == tail[5]
            and tail[2] == tail[6]
            and tail[3] == tail[7]
            and len({tail[0], tail[1], tail[2], tail[3]}) == 4
        ):
            return DogMovementType.ACTIVE_REPETITION

    return DogMovementType.EXPLORING_OR_ADVANCING


def analyze_repetition_and_recovery(
    replay_bundle_or_result: Any,
    dog_count: int = 3,
    aftermath_window_steps: int = 20,
) -> RepetitionRecoveryAnalysis:
    """Analyze an episode for multi-agent repeated movement and recovery outcomes.

    Reuses the verified temporal interaction analyzer for whole-episode interval detection,
    and applies rigorous definitions:
    - Stationary holding is never counted as an oscillation.
    - Productive waiting while teammates make progress is never counted as repetition.
    - Sustained multi-step loops (2, 3, or 4 cells) are identified.
    - Recovery requires sustained exit from the stall into lasting progress (penning,
      net route distance reduction, or collection/cohesion without immediate relapse).
    - Eventual episode success is recorded but does NOT credit stalls that relapsed or
      timed out at their observation window.
    """
    # 1. Obtain or build the replay bundle and run temporal analysis
    if isinstance(replay_bundle_or_result, (str, Path)):
        bundle, _, _ = load_replay_bundle(replay_bundle_or_result)
        frames = bundle.get("frames", [])
        stats = bundle.get("stats", {})
        eventual_success = bool(stats.get("success", False)) if isinstance(stats, dict) else False
    elif isinstance(replay_bundle_or_result, dict) and "frames" in replay_bundle_or_result:
        bundle = replay_bundle_or_result
        frames = bundle.get("frames", [])
        stats = bundle.get("stats", {})
        eventual_success = bool(stats.get("success", False)) if isinstance(stats, dict) else False
    elif hasattr(replay_bundle_or_result, "replay"):
        # EvaluationResult object
        res = replay_bundle_or_result
        eventual_success = bool(getattr(res.stats, "success", False))
        frames = [f.to_dict() if hasattr(f, "to_dict") else f for f in res.replay]
        bundle = {
            "policy_name": getattr(res, "policy_name", "evaluation"),
            "seed": getattr(res, "seed", 0),
            "environment": getattr(res, "environment_config", {}),
            "final_snapshot": res.final_snapshot.to_dict() if hasattr(res, "final_snapshot") else {},
            "initial_snapshot": res.initial_snapshot.to_dict() if hasattr(res, "initial_snapshot") else {},
            "stats": asdict(res.stats) if hasattr(res.stats, "__dataclass_fields__") else (res.stats if isinstance(res.stats, dict) else {}),
            "frames": frames,
            "strategy_version": "v1.4",
            "discount_timebase": "sequential_shared_dog",
            "discount_factor": 0.99,
        }
    else:
        bundle = {"frames": [], "stats": {"success": False}}
        frames = []
        eventual_success = False

    temporal: EpisodeTemporalAnalysis | None = None
    intervals: list[TemporalInteractionInterval] = []
    try:
        temporal = analyze_episode_temporal_interactions(bundle)
        intervals = list(temporal.intervals)
    except Exception:
        intervals = []

    # Map intervals by step and classification
    oscillation_intervals = [iv for iv in intervals if iv.classification == "oscillation_loop"]
    repositioning_intervals = [iv for iv in intervals if iv.classification == "productive_repositioning"]

    osc_steps: set[int] = set()
    for iv in oscillation_intervals:
        for s in range(iv.start_step, iv.end_step + 1):
            osc_steps.add(s)

    reposition_steps: set[int] = set()
    for iv in repositioning_intervals:
        for s in range(iv.start_step, iv.end_step + 1):
            reposition_steps.add(s)

    total_world_steps = len(frames)
    if total_world_steps == 0 and temporal is not None:
        total_world_steps = temporal.total_steps

    # 2. Step-by-step dog position tracking and repetition classification
    dog_histories: dict[int, list[tuple[int, int]]] = {d: [] for d in range(dog_count)}
    steps_by_unpenned: dict[int, int] = {k: 0 for k in range(1, 7)}
    repeated_steps_by_unpenned: dict[int, int] = {k: 0 for k in range(1, 7)}

    prev_avg_dist: float | None = None
    prev_penned_count: int | None = None
    unproductive_repetition_step_indices: set[int] = set()

    for idx, frame in enumerate(frames):
        snap = frame.get("snapshot", frame) if isinstance(frame, dict) else frame
        step_num = int(frame.get("step", idx)) if isinstance(frame, dict) else idx

        # Unpenned sheep count
        sheep_list = snap.get("sheep", []) if isinstance(snap, dict) else getattr(snap, "sheep", [])
        unpenned_count = sum(1 for s in sheep_list if not (s.get("penned", False) if isinstance(s, dict) else getattr(s, "penned", False)))
        penned_count = len(sheep_list) - unpenned_count

        if 1 <= unpenned_count <= 6:
            steps_by_unpenned[unpenned_count] += 1

        # Track flock route distance
        curr_dist = float(snap.get("average_distance_to_pen", 0.0) if isinstance(snap, dict) else getattr(snap, "average_distance_to_pen", 0.0))
        dist_prog = (prev_avg_dist - curr_dist) if prev_avg_dist is not None else 0.0
        penned_delta = (penned_count - prev_penned_count) if prev_penned_count is not None else 0
        prev_avg_dist = curr_dist
        prev_penned_count = penned_count

        # In productive repositioning?
        in_reposition = step_num in reposition_steps

        # Check dog movement patterns
        dogs_snap = snap.get("dogs", []) if isinstance(snap, dict) else getattr(snap, "dogs", [])
        dog_patterns: list[DogMovementType] = []

        for d_idx, d_snap in enumerate(dogs_snap[:dog_count]):
            pt = _point_to_tuple(d_snap)
            hist = dog_histories.setdefault(d_idx, [])
            hist.append(pt)
            pat = classify_dog_movement_pattern(
                hist,
                team_flock_progress=dist_prog,
                team_penned_sheep=penned_delta > 0,
                in_productive_repositioning=in_reposition,
            )
            dog_patterns.append(pat)

        # Team repetition rule:
        # 1. Must NOT be in verified productive repositioning.
        # 2. Must NOT be making active flock progress (dist_prog >= 1.0 or penned_delta > 0).
        # 3. Step is counted as unproductive repetition if:
        #    a) Step belongs to a confirmed oscillation_loop interval, OR
        #    b) At least one dog is in ACTIVE_REPETITION and no dog is making verified progress.
        # 4. Stationary holding alone does NOT trigger active repetition.
        has_active_dog_loop = any(p == DogMovementType.ACTIVE_REPETITION for p in dog_patterns)
        is_osc_step = step_num in osc_steps

        is_team_progressing = (dist_prog >= 1.0) or (penned_delta > 0) or in_reposition

        if (is_osc_step or has_active_dog_loop) and not is_team_progressing:
            unproductive_repetition_step_indices.add(step_num)
            if 1 <= unpenned_count <= 6:
                repeated_steps_by_unpenned[unpenned_count] += 1

    ep_repeated_steps = len(unproductive_repetition_step_indices)
    rep_fraction = (
        ep_repeated_steps / total_world_steps if total_world_steps > 0 else None
    )

    # 3. Analyze candidate stalls and recovery vs relapse outcomes
    candidate_stalls = [
        iv
        for iv in intervals
        if iv.classification in ("oscillation_loop", "counterproductive_pressure", "stationary_hold")
    ]

    stall_records: list[StallOutcomeRecord] = []
    stalls_by_unpenned: dict[int, list[StallOutcomeRecord]] = {k: [] for k in range(1, 7)}

    for iv_idx, iv in enumerate(candidate_stalls):
        iv_id = iv.interval_id or f"interval_{iv_idx}"
        k_unpenned = iv.unpenned_sheep_count

        # Examine aftermath window
        after_end = min(iv.end_step + aftermath_window_steps, total_world_steps)

        # Check whether any subsequent stall begins in this aftermath window
        subsequent_stall_in_window = next(
            (
                s_iv
                for s_iv in candidate_stalls
                if s_iv.start_step > iv.end_step and s_iv.start_step <= after_end
            ),
            None,
        )

        penned_in_aftermath = iv.aftermath.sheep_penned_delta > 0
        route_prog_aftermath = iv.aftermath.net_route_progress
        has_lasting_route_progress = route_prog_aftermath >= 2.0

        # Successful collection / flock cohesion progress:
        # e.g. flock spread dropped significantly while maintaining alignment, followed by progress
        spread_delta = iv.aftermath.flock_spread_start - iv.aftermath.flock_spread_end
        is_successful_collection = (
            spread_delta >= 2.0 and route_prog_aftermath >= 0.0
        )

        outcome: StallOutcome
        relapse_step: int | None = None
        evidence_text: str

        if subsequent_stall_in_window is not None:
            # Re-entered a stall within aftermath window -> Relapse
            outcome = StallOutcome.RELAPSED
            relapse_step = subsequent_stall_in_window.start_step
            evidence_text = (
                f"Relapsed into {subsequent_stall_in_window.classification} at step {relapse_step} "
                f"after temporary exit (aftermath route progress: {route_prog_aftermath:+.1f} cells)."
            )
        elif penned_in_aftermath or has_lasting_route_progress or is_successful_collection:
            # Sustained exit into lasting progress
            outcome = StallOutcome.RECOVERED
            evidence_parts = []
            if penned_in_aftermath:
                evidence_parts.append(f"{iv.aftermath.sheep_penned_delta} sheep penned")
            if has_lasting_route_progress:
                evidence_parts.append(f"{route_prog_aftermath:+.1f} route progress cells toward pen")
            if is_successful_collection:
                evidence_parts.append(f"flock regrouped (spread reduced by {spread_delta:.1f} cells)")
            evidence_text = (
                f"Sustained recovery confirmed: {', '.join(evidence_parts)} without recurrence in aftermath."
            )
        else:
            # Neither recovered into lasting progress nor verified relapse; remained unresolved
            outcome = StallOutcome.UNRESOLVED
            evidence_text = (
                f"Unresolved stall: aftermath achieved {route_prog_aftermath:+.1f} cells progress "
                f"with 0 sheep penned; episode ended or stalled."
            )

        rec = StallOutcomeRecord(
            interval_id=iv_id,
            classification=iv.classification,
            unpenned_sheep_count=k_unpenned,
            start_step=iv.start_step,
            end_step=iv.end_step,
            duration_steps=iv.duration_steps,
            outcome=outcome,
            lasting_progress_evidence=evidence_text,
            relapse_step=relapse_step,
            eventual_episode_success=eventual_success,
        )
        stall_records.append(rec)
        if 1 <= k_unpenned <= 6:
            stalls_by_unpenned[k_unpenned].append(rec)

    # 4. Aggregations
    total_stalls = len(stall_records)
    recovered_count = sum(1 for r in stall_records if r.outcome == StallOutcome.RECOVERED)
    relapsed_count = sum(1 for r in stall_records if r.outcome == StallOutcome.RELAPSED)
    unresolved_count = sum(1 for r in stall_records if r.outcome == StallOutcome.UNRESOLVED)

    recovery_rate = round(recovered_count / total_stalls, 4) if total_stalls > 0 else None
    relapse_rate = round(relapsed_count / total_stalls, 4) if total_stalls > 0 else None
    unresolved_rate = round(unresolved_count / total_stalls, 4) if total_stalls > 0 else None

    # Per-unpenned bucket summaries
    by_unpenned_metrics: dict[str, UnpennedBucketMetrics] = {}
    for k in range(1, 7):
        k_str = str(k)
        w_steps = steps_by_unpenned[k]
        r_steps = repeated_steps_by_unpenned[k]
        k_rep_frac = round(r_steps / w_steps, 4) if w_steps > 0 else None

        k_stalls = stalls_by_unpenned[k]
        n_stalls = len(k_stalls)
        n_rec = sum(1 for r in k_stalls if r.outcome == StallOutcome.RECOVERED)
        n_rel = sum(1 for r in k_stalls if r.outcome == StallOutcome.RELAPSED)
        n_unres = sum(1 for r in k_stalls if r.outcome == StallOutcome.UNRESOLVED)

        k_rec_rate = round(n_rec / n_stalls, 4) if n_stalls > 0 else None
        k_rel_rate = round(n_rel / n_stalls, 4) if n_stalls > 0 else None

        by_unpenned_metrics[k_str] = UnpennedBucketMetrics(
            unpenned_count=k,
            episodes_observed=1 if w_steps > 0 else 0,
            world_steps=w_steps,
            repeated_movement_steps=r_steps,
            repeated_movement_fraction=k_rep_frac,
            stall_intervals=n_stalls,
            recovered_intervals=n_rec,
            relapsed_intervals=n_rel,
            unresolved_intervals=n_unres,
            recovery_rate=k_rec_rate,
            relapse_rate=k_rel_rate,
        )

    return RepetitionRecoveryAnalysis(
        metrics_version=METRICS_VERSION,
        total_world_steps=total_world_steps,
        repeated_movement_steps=ep_repeated_steps,
        repeated_movement_fraction=rep_fraction,
        stalls_count=total_stalls,
        recovered_stalls_count=recovered_count,
        relapsed_stalls_count=relapsed_count,
        unresolved_stalls_count=unresolved_count,
        recovery_rate=recovery_rate,
        relapse_rate=relapse_rate,
        unresolved_rate=unresolved_rate,
        eventual_episode_success=eventual_success,
        stall_records=tuple(stall_records),
        by_unpenned_count=by_unpenned_metrics,
        temporal_analysis=temporal,
    )
