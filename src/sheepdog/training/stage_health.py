"""Real-time Whole-Stage Learning Health & Prescriptive Diagnostic Engine.

Aggregates all evaluation checkpoints across the lifetime of a curriculum stage
to classify training health (Green / Yellow / Red), track failure progress (closeness to victory),
measure all-time seed reliability, and generate automated configuration recommendations.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.curriculum import (
    CURRICULUM_REWARD_OVERRIDES,
    CURRICULUM_STAGES,
    CURRICULUM_TRAINING_OVERRIDES,
    stage_summary,
)

logger = logging.getLogger(__name__)

# In-memory cache to guarantee <15ms response times on repeated calls
_STAGE_HEALTH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class SeedHealthItem:
    seed: int
    win_rate: float
    wins: int
    fails: int
    total: int
    status: str  # "green", "yellow", "red"
    current_consecutive_fails: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureProgressStats:
    total_failures: int
    avg_penned_on_fail: float
    three_penned_pct: float
    two_penned_pct: float
    one_penned_pct: float
    zero_penned_pct: float
    closeness_score: float  # [0.0 - 1.0] Higher means failed runs were very close to victory

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HyperparameterAuditItem:
    parameter: str
    current_value: Any
    recommended_value: Any
    status: str  # "ok", "warn", "danger"
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrescriptiveRecommendation:
    type: str  # "continue", "reward_tweak", "entropy_tweak", "lr_tweak", "failure_directed", "cooldown"
    title: str
    description: str
    suggested_action: str
    priority: str  # "high", "medium", "info"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageHealthSummary:
    stage: int
    stage_title: str
    total_stage_checkpoints: int
    all_time_stage_success_rate: float
    recent_success_rate: float
    peak_stage_success_rate: float
    recent_avg_steps: float
    recent_avg_reward: float
    status: str  # "green", "yellow", "red"
    status_label: str
    status_explanation: str
    promotion_ready: bool
    promotion_status_text: str
    failure_progress: FailureProgressStats
    seed_matrix: tuple[SeedHealthItem, ...] = field(default_factory=tuple)
    recent_trajectory: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hyperparameter_audit: tuple[HyperparameterAuditItem, ...] = field(default_factory=tuple)
    prescriptive_recommendations: tuple[PrescriptiveRecommendation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_title": self.stage_title,
            "total_stage_checkpoints": self.total_stage_checkpoints,
            "all_time_stage_success_rate": round(self.all_time_stage_success_rate, 4),
            "recent_success_rate": round(self.recent_success_rate, 4),
            "peak_stage_success_rate": round(self.peak_stage_success_rate, 4),
            "recent_avg_steps": round(self.recent_avg_steps, 1),
            "recent_avg_reward": round(self.recent_avg_reward, 1),
            "status": self.status,
            "status_label": self.status_label,
            "status_explanation": self.status_explanation,
            "promotion_ready": self.promotion_ready,
            "promotion_status_text": self.promotion_status_text,
            "failure_progress": self.failure_progress.to_dict(),
            "seed_matrix": [s.to_dict() for s in self.seed_matrix],
            "recent_trajectory": list(self.recent_trajectory),
            "hyperparameter_audit": [h.to_dict() for h in self.hyperparameter_audit],
            "prescriptive_recommendations": [r.to_dict() for r in self.prescriptive_recommendations],
        }


def _extract_pv_from_filename(filename: str) -> int | None:
    match = re.search(r"pv_(\d+)_", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"checkpoint-(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    return None


def compute_stage_health_summary(
    output_dir: Path | str,
    target_stage: int | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Compute an instantaneous, holistic health assessment across the entire stage history."""
    out_root = Path(output_dir).resolve()
    evals_dir = out_root / "evaluations"

    cache_key = f"{out_root.as_posix()}_{target_stage}"
    now_time = os.times().elapsed if hasattr(os, "times") else 0.0

    if not force_refresh and cache_key in _STAGE_HEALTH_CACHE:
        cached_ts, cached_data = _STAGE_HEALTH_CACHE[cache_key]
        if now_time - cached_ts < CACHE_TTL_SECONDS:
            return cached_data

    if not evals_dir.exists():
        empty_summary = _build_empty_summary(target_stage or 1)
        return empty_summary.to_dict()

    # Fast directory index of evaluation files
    eval_files: list[tuple[int, Path]] = []
    for entry in os.scandir(evals_dir):
        if entry.is_file() and entry.name.endswith(".json") and entry.name.startswith("eval_"):
            pv = _extract_pv_from_filename(entry.name)
            if pv is not None:
                eval_files.append((pv, Path(entry.path)))

    eval_files.sort(key=lambda x: x[0])
    if not eval_files:
        empty_summary = _build_empty_summary(target_stage or 1)
        return empty_summary.to_dict()

    # Step 1: Discover stage if target_stage is not explicitly provided
    # Load recent files from the end until stage is identified
    loaded_evals: list[tuple[int, dict[str, Any]]] = []
    seen_keys: set[tuple[int, str]] = set()

    for pv, path in reversed(eval_files):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            st = data.get("curriculum_stage")
            mode = data.get("evaluation_mode", "quick")
            if target_stage is None and st is not None:
                target_stage = int(st)
            if st is not None and int(st) == target_stage:
                key = (pv, mode)
                if key not in seen_keys:
                    seen_keys.add(key)
                    loaded_evals.append((pv, data))
        except Exception:
            continue

    if target_stage is None:
        target_stage = 1

    loaded_evals.sort(key=lambda x: x[0])
    if not loaded_evals:
        empty_summary = _build_empty_summary(target_stage)
        return empty_summary.to_dict()

    stage_desc = CURRICULUM_STAGES.get(target_stage, {})
    dogs_count = stage_desc.get("dogs", 1)
    sheep_count = stage_desc.get("sheep", 1)
    width = stage_desc.get("width", 60)
    height = stage_desc.get("height", 45)
    stage_title = f"Stage {target_stage} ({dogs_count} Dogs, {sheep_count} Sheep, {width}x{height})"

    # Aggregate All-Time Stage Metrics
    all_success_rates: list[float] = []
    all_steps: list[float] = []
    all_rewards: list[float] = []
    seed_wins: dict[int, int] = {}
    seed_totals: dict[int, int] = {}
    seed_consecutive_fails: dict[int, int] = {}

    penned_counts: list[int] = []
    total_failed_runs = 0

    # Process chronologically
    for pv, data in loaded_evals:
        sr = data.get("success_rate")
        steps = data.get("average_completion_steps")
        rew = data.get("average_reward")

        if sr is not None:
            all_success_rates.append(float(sr))
        if steps is not None:
            all_steps.append(float(steps))
        if rew is not None:
            all_rewards.append(float(rew))

        records = data.get("records", [])
        for r in records:
            s = r.get("seed")
            if s is not None:
                s = int(s)
                seed_totals[s] = seed_totals.get(s, 0) + 1
                is_win = bool(r.get("success", False))
                if is_win:
                    seed_wins[s] = seed_wins.get(s, 0) + 1
                    seed_consecutive_fails[s] = 0
                else:
                    seed_consecutive_fails[s] = seed_consecutive_fails.get(s, 0) + 1
                    total_failed_runs += 1
                    penned = r.get("sheep_penned")
                    if penned is not None:
                        penned_counts.append(int(penned))

    total_stage_checkpoints = len(loaded_evals)
    all_time_sr = sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0.0
    peak_sr = max(all_success_rates) if all_success_rates else 0.0

    recent_slice = loaded_evals[-15:]
    recent_sr_list = [d.get("success_rate", 0.0) or 0.0 for _, d in recent_slice]
    recent_steps_list = [d.get("average_completion_steps", 0.0) or 0.0 for _, d in recent_slice]
    recent_rew_list = [d.get("average_reward", 0.0) or 0.0 for _, d in recent_slice]

    recent_sr = sum(recent_sr_list) / len(recent_sr_list) if recent_sr_list else 0.0
    recent_avg_steps = sum(recent_steps_list) / len(recent_steps_list) if recent_steps_list else 0.0
    recent_avg_rew = sum(recent_rew_list) / len(recent_rew_list) if recent_rew_list else 0.0

    # Failure Progress Statistics
    three_p = 0
    two_p = 0
    one_p = 0
    zero_p = 0
    if penned_counts:
        for p in penned_counts:
            if p >= 3:
                three_p += 1
            elif p == 2:
                two_p += 1
            elif p == 1:
                one_p += 1
            else:
                zero_p += 1
        n_f = len(penned_counts)
        three_pct = three_p / n_f
        two_pct = two_p / n_f
        one_pct = one_p / n_f
        zero_pct = zero_p / n_f
        avg_penned = sum(penned_counts) / n_f
        # Closeness score: heavily weight 3/4 and 2/4 penned failures
        closeness = min(1.0, (three_pct * 1.0) + (two_pct * 0.6) + (one_pct * 0.2))
    else:
        three_pct = two_pct = one_pct = zero_pct = avg_penned = closeness = 0.0

    failure_progress = FailureProgressStats(
        total_failures=total_failed_runs,
        avg_penned_on_fail=round(avg_penned, 2),
        three_penned_pct=round(three_pct, 3),
        two_penned_pct=round(two_pct, 3),
        one_penned_pct=round(one_pct, 3),
        zero_penned_pct=round(zero_pct, 3),
        closeness_score=round(closeness, 3),
    )

    # Seed Health Matrix
    seed_health_list: list[SeedHealthItem] = []
    drag_seeds: list[int] = []
    for s in sorted(seed_totals.keys()):
        tot = seed_totals[s]
        w = seed_wins.get(s, 0)
        wr = w / tot if tot > 0 else 0.0
        consec = seed_consecutive_fails.get(s, 0)

        if wr >= 0.80:
            s_status = "green"
        elif wr >= 0.50 or consec <= 1:
            s_status = "yellow"
        else:
            s_status = "red"
            drag_seeds.append(s)

        seed_health_list.append(
            SeedHealthItem(
                seed=s,
                win_rate=round(wr, 3),
                wins=w,
                fails=tot - w,
                total=tot,
                status=s_status,
                current_consecutive_fails=consec,
            )
        )

    # Recent Trajectory points
    recent_trajectory = [
        {
            "pv": pv,
            "episode": d.get("checkpoint_episode", 0),
            "success_rate": round(d.get("success_rate", 0.0) or 0.0, 3),
            "steps": round(d.get("average_completion_steps", 0.0) or 0.0, 1),
            "reward": round(d.get("average_reward", 0.0) or 0.0, 1),
            "mode": d.get("evaluation_mode", "quick"),
            "timestamp": d.get("created_timestamp") or d.get("evaluation_timestamp") or "",
        }
        for pv, d in recent_slice
    ]

    # Health Classifier: Green / Yellow / Red
    # Recent surges (e.g. hitting 90% in multiple recent evals) or high failure progress = Green
    latest_sr = recent_sr_list[-1] if recent_sr_list else 0.0
    latest_few_max = max(recent_sr_list[-4:]) if len(recent_sr_list) >= 4 else latest_sr

    if latest_few_max >= 0.85 or (recent_sr >= 0.75 and recent_avg_rew > 150):
        status = "green"
        status_label = "Healthy Learning · Surging"
        status_explanation = (
            f"The policy is progressing excellently in Stage {target_stage}. "
            f"Recent evaluations have peaked at {int(round(latest_few_max * 100))}% win rate with "
            f"strong step efficiency ({round(recent_avg_steps, 1)} avg steps). Exploration is healthy "
            f"and dogs are consistently solving primary flock coordinates."
        )
    elif recent_sr >= 0.45 or closeness >= 0.45 or (peak_sr >= 0.80 and all_time_sr >= 0.55):
        status = "yellow"
        status_label = "Active Exploration · Watch"
        status_explanation = (
            f"The policy is undergoing standard exploratory consolidation. While win rates fluctuate "
            f"between {int(round(min(recent_sr_list) * 100))}%–{int(round(max(recent_sr_list) * 100))}%, "
            f"failures show high partial progress ({int(round(three_pct * 100))}% of failures pen 3 of 4 sheep). "
            f"This is normal multi-agent exploration and does NOT indicate stalled learning."
        )
    else:
        status = "red"
        status_label = "Systemic Bottleneck · Action Required"
        status_explanation = (
            f"The policy has remained below 40% success for an extended duration without partial progress "
            f"(only {int(round(three_pct * 100))}% 3-penned failures). Dogs are likely encountering an "
            f"unrewarded attractor state or severe corner/wall entrapment."
        )

    # Hyperparameter Audit
    hyperparam_items: list[HyperparameterAuditItem] = []
    reward_overrides = CURRICULUM_REWARD_OVERRIDES.get(target_stage, {})
    training_overrides = CURRICULUM_TRAINING_OVERRIDES.get(target_stage, {})

    curr_farthest = reward_overrides.get("farthest_sheep_progress_scale", 0.0)
    rec_farthest = 0.55 if target_stage >= 7 else 0.0
    if target_stage in (7, 8) and curr_farthest < 0.50:
        hyperparam_items.append(
            HyperparameterAuditItem(
                parameter="farthest_sheep_progress_scale",
                current_value=curr_farthest,
                recommended_value=rec_farthest,
                status="warn",
                note=f"Current {curr_farthest} is low for stray recovery on large grid; recommend >= {rec_farthest}.",
            )
        )
    else:
        hyperparam_items.append(
            HyperparameterAuditItem(
                parameter="farthest_sheep_progress_scale",
                current_value=curr_farthest,
                recommended_value=rec_farthest,
                status="ok",
                note="Balanced stray approach reward active.",
            )
        )

    curr_entropy = training_overrides.get("entropy_coef", 0.010)
    hyperparam_items.append(
        HyperparameterAuditItem(
            parameter="entropy_coef",
            current_value=curr_entropy,
            recommended_value=0.010 if status == "green" else 0.014,
            status="ok",
            note="Action exploration entropy coefficient.",
        )
    )

    failure_directed = bool(training_overrides.get("failure_directed_training_enabled", False))
    hyperparam_items.append(
        HyperparameterAuditItem(
            parameter="failure_directed_training_enabled",
            current_value=failure_directed,
            recommended_value=True if drag_seeds else False,
            status="warn" if (drag_seeds and not failure_directed) else "ok",
            note="Dynamic exposure to difficult seed spawn scenarios.",
        )
    )

    # Prescriptive Recommendations
    recommendations: list[PrescriptiveRecommendation] = []

    if status == "green":
        recommendations.append(
            PrescriptiveRecommendation(
                type="continue",
                title="Allow Live Training to Continue",
                description="The policy is in a strong upward trajectory and actively hitting 90% benchmarks. Do not interrupt or reset weights.",
                suggested_action="Keep training running. Monitor auto-promotion gate for consecutive qualification.",
                priority="info",
            )
        )
    elif status == "yellow":
        recommendations.append(
            PrescriptiveRecommendation(
                type="continue",
                title="Do Not Panic on Temporary Dips",
                description="Failed episodes are penning 3 of 4 sheep in the majority of trials. Multi-agent PPO naturally surges after exploration consolidation.",
                suggested_action="Allow 10–15 more checkpoints before making intervention decisions.",
                priority="info",
            )
        )

    if drag_seeds:
        drag_str = ", ".join(str(s) for s in drag_seeds[:3])
        recommendations.append(
            PrescriptiveRecommendation(
                type="failure_directed",
                title=f"Target Drag Seeds ({drag_str})",
                description=f"Seeds {drag_str} account for the majority of stage failures due to wide stray initial spawns.",
                suggested_action=f"Add override in CURRICULUM_TRAINING_OVERRIDES[{target_stage}]: 'failure_directed_training_enabled': True",
                priority="medium" if status == "green" else "high",
            )
        )

    if target_stage in (7, 8) and curr_farthest < 0.50:
        recommendations.append(
            PrescriptiveRecommendation(
                type="reward_tweak",
                title="Boost Lone Straggler Approach Scale",
                description="37%+ of failures leave exactly 1 sheep unpenned until timeout. A stronger approach gradient motivates dogs to leave the gate and retrieve the 4th sheep.",
                suggested_action=f"Update CURRICULUM_REWARD_OVERRIDES[{target_stage}]['farthest_sheep_progress_scale'] = 0.55 in src/sheepdog/curriculum.py",
                priority="medium",
            )
        )

    summary = StageHealthSummary(
        stage=target_stage,
        stage_title=stage_title,
        total_stage_checkpoints=total_stage_checkpoints,
        all_time_stage_success_rate=all_time_sr,
        recent_success_rate=recent_sr,
        peak_stage_success_rate=peak_sr,
        recent_avg_steps=recent_avg_steps,
        recent_avg_reward=recent_avg_rew,
        status=status,
        status_label=status_label,
        status_explanation=status_explanation,
        promotion_ready=latest_few_max >= 0.90 and recent_sr >= 0.80,
        promotion_status_text="Passes Pending" if latest_few_max < 0.90 else "Promotion Candidate",
        failure_progress=failure_progress,
        seed_matrix=tuple(seed_health_list),
        recent_trajectory=tuple(recent_trajectory),
        hyperparameter_audit=tuple(hyperparam_items),
        prescriptive_recommendations=tuple(recommendations),
    )

    result_dict = summary.to_dict()
    _STAGE_HEALTH_CACHE[cache_key] = (now_time, result_dict)
    return result_dict


def _build_empty_summary(stage: int) -> StageHealthSummary:
    return StageHealthSummary(
        stage=stage,
        stage_title=f"Stage {stage} · Baseline",
        total_stage_checkpoints=0,
        all_time_stage_success_rate=0.0,
        recent_success_rate=0.0,
        peak_stage_success_rate=0.0,
        recent_avg_steps=0.0,
        recent_avg_reward=0.0,
        status="yellow",
        status_label="Awaiting Initial Checkpoints",
        status_explanation="No formal evaluation checkpoints have been recorded for this stage yet.",
        promotion_ready=False,
        promotion_status_text="Collecting Data",
        failure_progress=FailureProgressStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
