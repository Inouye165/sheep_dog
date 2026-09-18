"""Deterministic checkpoint evaluation."""

from __future__ import annotations

import contextlib
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from sheepdog.config import LabConfig
from sheepdog.environment import EpisodeResult, SheepdogEnvironment
from sheepdog.policies.base import Policy
from sheepdog.replay.store import ReplayStore
from sheepdog.training.runtime import TrainingRuntimeTracker
from sheepdog.evaluation.retention import EvaluationReplayRetentionManager


import logging
import types
from sheepdog.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


class EvaluationInterruptedError(Exception):
    """Raised when an evaluation run is interrupted by a stop or pause request."""

    def __init__(self, message: str, inprogress_path: Path | None = None) -> None:
        super().__init__(message)
        self.inprogress_path = inprogress_path


class _RestoredStats:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name, 0.0)


class _RestoredResult:
    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = _RestoredStats(stats)


def _policy_metadata(
    policy_name: str, *, trainer_type: str | None, policy_type: str | None
) -> tuple[str, str, str]:
    """Return normalized trainer, policy, and replay-mode labels for replay export."""

    normalized_trainer = trainer_type or "baseline"
    normalized_policy_type = policy_type or "instinct"
    replay_mode = "baseline"
    if policy_name == "joint_team_policy" or normalized_trainer == "joint_maskable_ppo":
        normalized_trainer = "joint_maskable_ppo"
        normalized_policy_type = "neural"
        replay_mode = "joint_team_ppo"
    elif policy_name == "neural_policy" or normalized_trainer == "maskable_ppo":
        normalized_trainer = "maskable_ppo"
        normalized_policy_type = "neural"
        replay_mode = "neural_ppo"
    elif policy_name == "trained_policy":
        normalized_trainer = "hill_climb"
        normalized_policy_type = "linear"
        replay_mode = "trained_linear"
    elif policy_name == "heuristic_expert":
        normalized_trainer = "baseline"
        normalized_policy_type = "heuristic"
    elif policy_name in {"random_untrained", "random_policy"}:
        normalized_trainer = "baseline"
        normalized_policy_type = "random"
    else:
        normalized_trainer = "baseline"
        normalized_policy_type = "instinct"
    return normalized_trainer, normalized_policy_type, replay_mode


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Per-seed evaluation results."""

    seed: int
    success: bool
    timeout: bool
    stopped: bool
    steps: int
    simulated_seconds: float
    sheep_penned: int
    final_sheep_distance_to_pen: float
    final_flock_spread: float
    no_progress_steps: int
    stop_reason: str
    spawn_mode: str
    reward_total: float
    final_farthest_distance_to_pen: float
    final_farthest_distance_to_flock_center: float
    role_switches: int
    collector_activations: int
    blocker_activations: int
    cumulative_gate_progress: float
    controlled_stall_steps: int
    left_flank_occupancy_steps: int
    right_flank_occupancy_steps: int
    gate_corridor_occupancy_peak: float
    gate_corridor_failure_steps: int
    dog_role_occupancy: dict[str, dict[str, int]]
    reward_breakdown: dict[str, float]
    replay_path: str
    policy_version: int | None = None
    initial_sheep_distance_to_pen: float | None = None
    min_sheep_distance_to_pen: float | None = None
    final_dog_to_sheep_distance: float | None = None
    final_dog_positions: list[tuple[float, float]] | None = None
    final_sheep_positions: list[tuple[float, float]] | None = None
    pen_position: tuple[float, float] | None = None
    num_waits: int | None = None
    num_sprints: int | None = None
    num_invalid_actions: int | None = None
    most_frequent_action: str | None = None
    oscillation_detected: bool | None = None
    observation_diagnostics: dict[str, Any] | None = None
    failed_trajectory_summary: list[dict[str, Any]] | None = None
    last_actions_before_failure: list[list[str]] | None = None
    pen_zone: str | None = None
    initial_sheep_zone: str | None = None
    final_sheep_zone: str | None = None
    corner_steps_total: int | None = None
    corner_time_pct: float | None = None
    wall_steps_total: int | None = None
    wall_time_pct: float | None = None
    corner_stuck_at_end: bool | None = None
    corner_entered: bool | None = None
    corner_extracted: bool | None = None
    spatial_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate checkpoint evaluation output."""

    checkpoint_episode: int
    policy_name: str
    records: tuple[EvaluationRecord, ...]
    success_rate: float
    timeout_rate: float
    average_completion_steps: float
    average_completion_seconds: float
    average_sheep_penned: float
    average_reward: float
    trainer_type: str = "hill_climb"
    policy_type: str = "linear"
    average_distance_to_pen: float = 0.0
    average_flock_spread: float = 0.0
    stopped_rate: float = 0.0
    average_no_progress_steps: float = 0.0
    average_farthest_distance_to_pen: float = 0.0
    average_farthest_distance_to_flock_center: float = 0.0
    average_role_switches: float = 0.0
    average_collector_activations: float = 0.0
    average_blocker_activations: float = 0.0
    average_gate_progress: float = 0.0
    average_controlled_stall_steps: float = 0.0
    average_left_flank_occupancy_steps: float = 0.0
    average_right_flank_occupancy_steps: float = 0.0
    average_gate_corridor_occupancy_peak: float = 0.0
    average_gate_corridor_failure_steps: float = 0.0
    curriculum_stage: int = 1
    run_id: str | None = None
    checkpoint_id: str | None = None
    policy_version: int | None = None
    evaluation_timestamp: str | None = None
    evaluation_seed_set_id: str | None = None
    evaluation_seed_count: int | None = None
    environment_config_hash: str | None = None
    observation_schema_hash: str | None = None
    action_space_hash: str | None = None
    evaluation_id: str | None = None
    evaluation_mode: str = "confidence"
    promotion_eligible: bool = True
    evaluation_index: int | None = None
    retention_status: str | None = None
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


class Evaluator:
    """Run checkpoint comparisons on fixed evaluation seeds."""

    def __init__(self, config: LabConfig, output_root: str | Path) -> None:
        self.config = config
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.replay_store = ReplayStore(self.output_root / "replays")
        self.retention_manager = EvaluationReplayRetentionManager(self.output_root)
        self.runtime_tracker: TrainingRuntimeTracker | None = None
        self._evaluation_count = 0
        self.progress_callback: Callable[[dict[str, Any]], None] | None = None

    def evaluate(
        self,
        policy: Policy,
        seeds: tuple[int, ...],
        checkpoint_episode: int,
        deterministic: bool = True,
        *,
        capture_replays: bool = True,
        evaluation_mode: str = "confidence",
        run_id: str | None = None,
        checkpoint_id: str | None = None,
        policy_version: int | None = None,
        curriculum_stage: int | None = None,
        evaluation_index: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[EvaluationSummary, Path, Path]:
        """Run the policy on each seed and optionally capture full replays."""
        if evaluation_index is not None:
            eval_idx = int(evaluation_index)
            self._evaluation_count = max(self._evaluation_count, eval_idx)
        else:
            self._evaluation_count += 1
            eval_idx = self._evaluation_count

        import datetime

        from sheepdog.checkpoints.store import (
            compute_env_config_hash,
            compute_seed_set_id,
            get_action_space_hash,
            get_observation_schema_hash,
        )

        active_curriculum_stage = curriculum_stage if curriculum_stage is not None else self.config.rewards.instincts.curriculum_stage
        active_run_id = run_id
        active_checkpoint_id = checkpoint_id
        active_policy_version = policy_version if policy_version is not None else getattr(policy, "policy_version", None)

        callback = progress_callback or self.progress_callback
        eval_mode_label = evaluation_mode or "confidence"
        total_seeds = len(seeds)
        seeds_list = list(seeds)
        success_count = 0
        timeout_count = 0
        recent_results: list[dict[str, Any]] = []

        evaluation_seed_set_id = compute_seed_set_id(seeds)
        evaluation_seed_count = len(seeds)
        evaluation_id = (
            f"eval_{active_checkpoint_id}_{evaluation_mode}_{evaluation_seed_set_id[:12]}"
            if active_checkpoint_id
            else None
        )
        artifact_name = evaluation_id or f"evaluation-checkpoint-{checkpoint_episode:06d}"
        inprogress_path = self.output_root / f"{artifact_name}.inprogress.json"

        # Check for existing partial evaluation progress to resume
        resumed_records: dict[int, EvaluationRecord] = {}
        resumed_stats: dict[int, dict[str, Any]] = {}
        resumed_replay_paths_by_seed: dict[int, Path] = {}
        resumed_seed_results: dict[int, dict[str, Any]] = {}

        if inprogress_path.exists():
            try:
                with inprogress_path.open("r", encoding="utf-8") as handle:
                    inprog_data = json.load(handle)
                if inprog_data.get("seeds") == seeds_list:
                    for entry in inprog_data.get("completed_seeds_data", []):
                        s = entry.get("seed")
                        if s is not None and s in seeds:
                            rec_dict = entry.get("record")
                            if rec_dict:
                                resumed_records[s] = EvaluationRecord(**rec_dict)
                            st_dict = entry.get("stats")
                            if st_dict:
                                resumed_stats[s] = st_dict
                            rep_str = entry.get("replay_path")
                            if rep_str and Path(rep_str).exists():
                                resumed_replay_paths_by_seed[s] = Path(rep_str)
                            s_res = entry.get("seed_result")
                            if s_res:
                                resumed_seed_results[s] = s_res
                    if resumed_records:
                        now_res_str = datetime.datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[Sheepdog] [{now_res_str}] Resuming evaluation [{eval_mode_label}]: "
                            f"{len(resumed_records)}/{total_seeds} seeds already completed on disk.",
                            flush=True,
                        )
            except Exception as exc:
                logger.warning("Could not read in-progress evaluation cache %s: %s", inprogress_path, exc)

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        seed_preview = str(seeds_list[:6]) if total_seeds <= 6 else f"{seeds_list[:5]}... (+{total_seeds - 5} more)"
        print(
            f"[Sheepdog] [{now_str}] Evaluation [{eval_mode_label}] started | "
            f"Evaluating {total_seeds} seeds: {seed_preview}",
            flush=True,
        )

        if callback:
            with contextlib.suppress(Exception):
                callback({
                    "phase": "evaluation",
                    "evaluation_in_progress": True,
                    "evaluation_mode": eval_mode_label,
                    "evaluation_seed": seeds[0] if seeds else None,
                    "evaluation_seed_index": 1 if seeds else 0,
                    "evaluation_total_seeds": total_seeds,
                    "evaluation_completed_seeds": len(resumed_records),
                    "evaluation_success_count": sum(1 for r in resumed_records.values() if r.success),
                    "evaluation_timeout_count": sum(1 for r in resumed_records.values() if r.timeout),
                    "evaluation_success_rate": round(sum(1 for r in resumed_records.values() if r.success) / len(resumed_records), 4) if resumed_records else 0.0,
                    "evaluation_seeds": seeds_list,
                    "evaluation_message": f"Starting {eval_mode_label} evaluation on {total_seeds} seeds...",
                    "evaluation_latest_result": None,
                    "evaluation_recent_results": [],
                })

        results: list[Any] = []
        records: list[EvaluationRecord] = []
        saved_replay_paths: list[Path] = []
        completed_entries: list[tuple[EvaluationRecord, dict[str, Any], Path | None, dict[str, Any]]] = []

        def _save_in_progress() -> None:
            payload = {
                "evaluation_id": evaluation_id,
                "artifact_name": artifact_name,
                "checkpoint_episode": checkpoint_episode,
                "evaluation_mode": eval_mode_label,
                "seeds": seeds_list,
                "success_count": success_count,
                "timeout_count": timeout_count,
                "recent_results": recent_results,
                "completed_seeds_data": [
                    {
                        "seed": rec.seed,
                        "record": rec.to_dict(),
                        "stats": st_dict,
                        "replay_path": str(rep) if rep else None,
                        "seed_result": s_res,
                    }
                    for rec, st_dict, rep, s_res in completed_entries
                ],
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            try:
                atomic_write_json(inprogress_path, payload)
            except Exception as exc:
                logger.warning("Failed to save evaluation progress to %s: %s", inprogress_path, exc)

        for idx, seed in enumerate(seeds, start=1):
            if should_stop is not None and should_stop():
                if completed_entries:
                    _save_in_progress()
                raise EvaluationInterruptedError(
                    f"Evaluation [{eval_mode_label}] stopped before seed {seed}",
                    inprogress_path=inprogress_path,
                )

            # Check if this seed was previously completed and can be reused
            if seed in resumed_records:
                rec = resumed_records[seed]
                records.append(rec)
                st = resumed_stats.get(seed, {})
                results.append(_RestoredResult(st))
                rep_path = resumed_replay_paths_by_seed.get(seed)
                if rep_path:
                    saved_replay_paths.append(rep_path)
                s_res = resumed_seed_results.get(seed)
                if s_res:
                    recent_results.append(s_res)
                    if s_res.get("success"):
                        success_count += 1
                    if s_res.get("timeout"):
                        timeout_count += 1
                else:
                    if rec.success:
                        success_count += 1
                    if rec.timeout:
                        timeout_count += 1

                completed_entries.append((rec, st, rep_path, s_res or {}))

                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                rate_so_far = round((success_count / idx) * 100, 1)
                res_str = rec.stop_reason.upper() if rec.stop_reason else ("SUCCESS" if rec.success else "TIMEOUT")
                seed_done_msg = (
                    f"Evaluation [{eval_mode_label}] seed {seed} ({idx}/{total_seeds}) {res_str} (cached) | "
                    f"Penned: {rec.sheep_penned}/{int(self.config.environment.sheep)} | "
                    f"Reward: {rec.reward_total:.2f} | Steps: {rec.steps} | "
                    f"Score: {success_count}/{idx} ({rate_so_far:.0f}%)"
                )
                print(f"[Sheepdog] [{now_str}] {seed_done_msg}", flush=True)

                if callback:
                    with contextlib.suppress(Exception):
                        callback({
                            "phase": "evaluation",
                            "evaluation_in_progress": True,
                            "evaluation_mode": eval_mode_label,
                            "evaluation_seed": seed,
                            "evaluation_seed_index": idx,
                            "evaluation_total_seeds": total_seeds,
                            "evaluation_completed_seeds": idx,
                            "evaluation_success_count": success_count,
                            "evaluation_timeout_count": timeout_count,
                            "evaluation_success_rate": round(success_count / idx, 4),
                            "evaluation_seeds": seeds_list,
                            "evaluation_message": seed_done_msg,
                            "evaluation_latest_result": s_res,
                            "evaluation_recent_results": list(recent_results),
                        })
                continue

            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            in_prog_msg = f"Evaluation [{eval_mode_label}] in progress: seed {seed} ({idx}/{total_seeds})"
            print(f"[Sheepdog] [{now_str}] {in_prog_msg}...", flush=True)

            if callback:
                with contextlib.suppress(Exception):
                    callback({
                        "phase": "evaluation",
                        "evaluation_in_progress": True,
                        "evaluation_mode": eval_mode_label,
                        "evaluation_seed": seed,
                        "evaluation_seed_index": idx,
                        "evaluation_total_seeds": total_seeds,
                        "evaluation_completed_seeds": idx - 1,
                        "evaluation_success_count": success_count,
                        "evaluation_timeout_count": timeout_count,
                        "evaluation_success_rate": round(success_count / (idx - 1), 4) if (idx > 1) else 0.0,
                        "evaluation_seeds": seeds_list,
                        "evaluation_message": in_prog_msg,
                        "evaluation_latest_result": recent_results[-1] if recent_results else None,
                        "evaluation_recent_results": list(recent_results),
                    })

            environment = SheepdogEnvironment(self.config)
            capture_phase = (
                self.runtime_tracker.phase("replay_capture")
                if capture_replays and self.runtime_tracker is not None
                else contextlib.nullcontext()
            )
            with capture_phase:
                result = environment.run_policy(
                    policy,
                    seed,
                    capture_replay=capture_replays,
                    deterministic=deterministic,
                )
            results.append(result)

            is_success = bool(result.stats.success)
            is_timeout = bool(result.stats.timeout)
            if is_success:
                success_count += 1
            if is_timeout:
                timeout_count += 1

            total_sheep = int(self.config.environment.sheep)
            res_str = "SUCCESS" if is_success else ("TIMEOUT" if is_timeout else "FAILED")
            seed_result = {
                "seed": int(seed),
                "success": is_success,
                "timeout": is_timeout,
                "status": res_str,
                "reward": round(float(result.stats.reward_total), 2),
                "penned": int(result.stats.sheep_penned),
                "total_sheep": total_sheep,
                "steps": int(result.stats.steps),
            }
            recent_results.append(seed_result)

            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            rate_so_far = round((success_count / idx) * 100, 1)
            seed_done_msg = (
                f"Evaluation [{eval_mode_label}] seed {seed} ({idx}/{total_seeds}) {res_str} | "
                f"Penned: {result.stats.sheep_penned}/{total_sheep} | "
                f"Reward: {result.stats.reward_total:.2f} | Steps: {result.stats.steps} | "
                f"Score: {success_count}/{idx} ({rate_so_far:.0f}%)"
            )
            print(f"[Sheepdog] [{now_str}] {seed_done_msg}", flush=True)

            if callback:
                with contextlib.suppress(Exception):
                    callback({
                        "phase": "evaluation",
                        "evaluation_in_progress": True,
                        "evaluation_mode": eval_mode_label,
                        "evaluation_seed": seed,
                        "evaluation_seed_index": idx,
                        "evaluation_total_seeds": total_seeds,
                        "evaluation_completed_seeds": idx,
                        "evaluation_success_count": success_count,
                        "evaluation_timeout_count": timeout_count,
                        "evaluation_success_rate": round(success_count / idx, 4),
                        "evaluation_seeds": seeds_list,
                        "evaluation_message": seed_done_msg,
                        "evaluation_latest_result": seed_result,
                        "evaluation_recent_results": list(recent_results),
                    })

            trainer_type, policy_type, replay_mode = _policy_metadata(
                result.policy_name,
                trainer_type=getattr(policy, "trainer_type", None),
                policy_type=getattr(policy, "policy_type", None),
            )
            replay_path: Path | None = None
            if capture_replays:
                serialization_phase = (
                    self.runtime_tracker.phase("replay_serialization")
                    if self.runtime_tracker is not None
                    else contextlib.nullcontext()
                )
                with serialization_phase:
                    replay_path = self.replay_store.write(
                        f"checkpoint-{checkpoint_episode:06d}-seed-{seed:06d}.json",
                        {
                    "seed": result.seed,
                    "policy_name": result.policy_name,
                    "trainer_type": trainer_type,
                    "policy_type": policy_type,
                    "policy_mode": result.policy_name,
                    "replay_mode": replay_mode,
                    "environment": {
                        "dogs": self.config.environment.dogs,
                        "sheep": self.config.environment.sheep,
                        "width": self.config.environment.width,
                        "height": self.config.environment.height,
                        "curriculum_stage": self.config.rewards.instincts.curriculum_stage,
                        "enable_instinct_rewards": (
                            self.config.rewards.instincts.enable_instinct_rewards
                        ),
                    },
                    "final_snapshot": result.final_snapshot.to_dict(),
                    "stats": asdict(result.stats),
                        "frames": [frame.to_dict() for frame in result.replay],
                        },
                    )
                    if replay_path is not None:
                        saved_replay_paths.append(replay_path)
            p_ver = policy_version if policy_version is not None else getattr(policy, "policy_version", None)
            new_record = EvaluationRecord(
                **{
                    **self._record_from_result(result, policy_version=p_ver).to_dict(),
                    "replay_path": str(replay_path) if replay_path is not None else "",
                }
            )
            records.append(new_record)
            completed_entries.append((new_record, asdict(result.stats), replay_path, seed_result))
            _save_in_progress()

            if should_stop is not None and should_stop():
                raise EvaluationInterruptedError(
                    f"Evaluation [{eval_mode_label}] stopped after seed {seed}",
                    inprogress_path=inprogress_path,
                )

        summary_trainer_type, summary_policy_type, _summary_replay_mode = _policy_metadata(
            policy.name,
            trainer_type=getattr(policy, "trainer_type", None),
            policy_type=getattr(policy, "policy_type", None),
        )

        evaluation_timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        if hasattr(self.config, "to_dict"):
            env_dict = self.config.to_dict()["environment"]
        else:
            env_dict = asdict(self.config.environment)
        environment_config_hash = compute_env_config_hash(env_dict)

        try:
            observation_schema_hash = get_observation_schema_hash(self.config)
        except Exception:  # pylint: disable=broad-exception-caught
            observation_schema_hash = None

        try:
            action_space_hash = get_action_space_hash()
        except Exception:  # pylint: disable=broad-exception-caught
            action_space_hash = None

        summary = EvaluationSummary(
            checkpoint_episode=checkpoint_episode,
            policy_name=policy.name,
            trainer_type=summary_trainer_type,
            policy_type=summary_policy_type,
            records=tuple(records),
            success_rate=fmean(1.0 if record.success else 0.0 for record in records),
            timeout_rate=fmean(1.0 if record.timeout else 0.0 for record in records),
            average_completion_steps=fmean(record.steps for record in records),
            average_completion_seconds=fmean(record.simulated_seconds for record in records),
            average_sheep_penned=fmean(record.sheep_penned for record in records),
            average_reward=fmean(record.reward_total for record in records),
            average_distance_to_pen=fmean(record.final_sheep_distance_to_pen for record in records),
            average_flock_spread=fmean(record.final_flock_spread for record in records),
            stopped_rate=fmean(1.0 if record.stopped else 0.0 for record in records),
            average_no_progress_steps=fmean(record.no_progress_steps for record in records),
            average_farthest_distance_to_pen=fmean(
                record.final_farthest_distance_to_pen for record in records
            ),
            average_farthest_distance_to_flock_center=fmean(
                record.final_farthest_distance_to_flock_center for record in records
            ),
            average_role_switches=fmean(getattr(result.stats, "role_switches", 0.0) for result in results),
            average_collector_activations=fmean(
                getattr(result.stats, "collector_activations", 0.0) for result in results
            ),
            average_blocker_activations=fmean(
                getattr(result.stats, "blocker_activations", 0.0) for result in results
            ),
            average_gate_progress=fmean(
                getattr(result.stats, "cumulative_gate_progress", 0.0) for result in results
            ),
            average_controlled_stall_steps=fmean(
                getattr(result.stats, "controlled_stall_steps", 0.0) for result in results
            ),
            average_left_flank_occupancy_steps=fmean(
                getattr(result.stats, "left_flank_occupancy_steps", 0.0) for result in results
            ),
            average_right_flank_occupancy_steps=fmean(
                getattr(result.stats, "right_flank_occupancy_steps", 0.0) for result in results
            ),
            average_gate_corridor_occupancy_peak=fmean(
                getattr(result.stats, "gate_corridor_occupancy_peak", 0.0) for result in results
            ),
            average_gate_corridor_failure_steps=fmean(
                getattr(result.stats, "gate_corridor_failure_steps", 0.0) for result in results
            ),
            curriculum_stage=active_curriculum_stage,
            run_id=active_run_id,
            checkpoint_id=active_checkpoint_id,
            policy_version=active_policy_version,
            evaluation_timestamp=evaluation_timestamp,
            evaluation_seed_set_id=evaluation_seed_set_id,
            evaluation_seed_count=evaluation_seed_count,
            environment_config_hash=environment_config_hash,
            observation_schema_hash=observation_schema_hash,
            action_space_hash=action_space_hash,
            evaluation_id=evaluation_id,
            evaluation_mode=evaluation_mode,
            promotion_eligible=evaluation_mode == "confidence",
            evaluation_index=eval_idx,
            retention_status=None,
            pinned=False,
        )

        retention_status = None
        pinned = self.retention_manager.is_pinned(artifact_name)
        if capture_replays and saved_replay_paths:
            retention_res = self.retention_manager.register_and_prune(
                evaluation_id=artifact_name,
                evaluation_index=eval_idx,
                checkpoint_episode=checkpoint_episode,
                replay_paths=saved_replay_paths,
            )
            retention_status = retention_res.get("status")
            pinned = self.retention_manager.is_pinned(artifact_name)

        summary = EvaluationSummary(
            **{
                **summary.to_dict(),
                "records": summary.records,
                "retention_status": retention_status,
                "pinned": pinned,
            }
        )

        json_path = self.output_root / f"{artifact_name}.json"
        csv_path = self.output_root / f"{artifact_name}.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].to_dict().keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

        # Clean up the inprogress file since evaluation completed fully
        if inprogress_path.exists():
            with contextlib.suppress(OSError):
                inprogress_path.unlink()

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        summary_msg = (
            f"Evaluation [{eval_mode_label}] complete | "
            f"Success Rate: {summary.success_rate * 100:.1f}% ({success_count}/{total_seeds}) | "
            f"Avg Reward: {summary.average_reward:.2f} | "
            f"Avg Penned: {summary.average_sheep_penned:.1f}/{self.config.environment.sheep}"
        )
        print(f"[Sheepdog] [{now_str}] {summary_msg}", flush=True)

        if callback:
            with contextlib.suppress(Exception):
                callback({
                    "phase": "evaluation_complete",
                    "evaluation_in_progress": False,
                    "evaluation_mode": eval_mode_label,
                    "evaluation_seed": seeds[-1] if seeds else None,
                    "evaluation_seed_index": total_seeds,
                    "evaluation_total_seeds": total_seeds,
                    "evaluation_completed_seeds": total_seeds,
                    "evaluation_success_count": success_count,
                    "evaluation_timeout_count": timeout_count,
                    "evaluation_success_rate": round(summary.success_rate, 4),
                    "evaluation_seeds": seeds_list,
                    "evaluation_message": summary_msg,
                    "evaluation_latest_result": recent_results[-1] if recent_results else None,
                    "evaluation_recent_results": list(recent_results),
                    "summary": summary.to_dict() if hasattr(summary, "to_dict") else asdict(summary),
                })

        return summary, json_path, csv_path

    def export_replay(
        self,
        policy: Policy,
        seed: int,
        checkpoint_episode: int,
        deterministic: bool = True,
    ) -> Path:
        """Capture and serialize one replay for an important checkpoint."""
        environment = SheepdogEnvironment(self.config)
        capture_phase = (
            self.runtime_tracker.phase("replay_capture")
            if self.runtime_tracker is not None
            else contextlib.nullcontext()
        )
        with capture_phase:
            result = environment.run_policy(
                policy,
                seed,
                capture_replay=True,
                deterministic=deterministic,
            )
        trainer_type, policy_type, replay_mode = _policy_metadata(
            result.policy_name,
            trainer_type=getattr(policy, "trainer_type", None),
            policy_type=getattr(policy, "policy_type", None),
        )
        serialization_phase = (
            self.runtime_tracker.phase("replay_serialization")
            if self.runtime_tracker is not None
            else contextlib.nullcontext()
        )
        with serialization_phase:
            return self.replay_store.write(
                f"checkpoint-{checkpoint_episode:06d}-seed-{seed:06d}.json",
                {
                    "seed": result.seed,
                    "policy_name": result.policy_name,
                    "trainer_type": trainer_type,
                    "policy_type": policy_type,
                    "replay_mode": replay_mode,
                    "environment": self.config.to_dict()["environment"],
                    "final_snapshot": result.final_snapshot.to_dict(),
                    "stats": asdict(result.stats),
                    "frames": [frame.to_dict() for frame in result.replay],
                },
            )

    def _record_from_result(self, result: EpisodeResult, policy_version: int | None = None) -> EvaluationRecord:
        snapshot = result.final_snapshot

        obs_diag = None
        if hasattr(result, "observations") and result.observations:
            obs_diag = self._compute_observation_diagnostics(result.observations)

        failed_traj = None
        last_actions = None
        if not result.stats.success:
            failed_traj = self._compute_failed_trajectory_summary(result)
            last_actions = [list(frame.actions) for frame in result.replay[-20:]]

        return EvaluationRecord(
            seed=result.seed,
            success=result.stats.success,
            timeout=result.stats.timeout,
            stopped=result.stats.stopped,
            steps=result.stats.steps,
            simulated_seconds=result.stats.simulated_seconds,
            sheep_penned=result.stats.sheep_penned,
            final_sheep_distance_to_pen=snapshot.average_distance_to_pen,
            final_flock_spread=snapshot.flock_spread,
            no_progress_steps=result.stats.no_progress_steps,
            stop_reason=result.stats.stop_reason,
            spawn_mode=result.stats.spawn_mode,
            reward_total=result.stats.reward_total,
            final_farthest_distance_to_pen=result.stats.final_farthest_distance_to_pen,
            final_farthest_distance_to_flock_center=result.stats.final_farthest_distance_to_flock_center,
            role_switches=result.stats.role_switches,
            collector_activations=result.stats.collector_activations,
            blocker_activations=result.stats.blocker_activations,
            cumulative_gate_progress=result.stats.cumulative_gate_progress,
            controlled_stall_steps=result.stats.controlled_stall_steps,
            left_flank_occupancy_steps=result.stats.left_flank_occupancy_steps,
            right_flank_occupancy_steps=result.stats.right_flank_occupancy_steps,
            gate_corridor_occupancy_peak=result.stats.gate_corridor_occupancy_peak,
            gate_corridor_failure_steps=result.stats.gate_corridor_failure_steps,
            dog_role_occupancy=result.stats.dog_role_occupancy,
            reward_breakdown=result.stats.final_reward_breakdown,
            replay_path="",
            policy_version=policy_version,
            initial_sheep_distance_to_pen=getattr(result.stats, "initial_sheep_distance_to_pen", None),
            min_sheep_distance_to_pen=getattr(result.stats, "min_sheep_distance_to_pen", None),
            final_dog_to_sheep_distance=getattr(result.stats, "final_dog_to_sheep_distance", None),
            final_dog_positions=getattr(result.stats, "final_dog_positions", None),
            final_sheep_positions=getattr(result.stats, "final_sheep_positions", None),
            pen_position=getattr(result.stats, "pen_position", None),
            num_waits=getattr(result.stats, "num_waits", None),
            num_sprints=getattr(result.stats, "num_sprints", None),
            num_invalid_actions=getattr(result.stats, "num_invalid_actions", None),
            most_frequent_action=getattr(result.stats, "most_frequent_action", None),
            oscillation_detected=getattr(result.stats, "oscillation_detected", None),
            observation_diagnostics=obs_diag,
            failed_trajectory_summary=failed_traj,
            last_actions_before_failure=last_actions,
            pen_zone=getattr(result.stats, "pen_zone", None),
            initial_sheep_zone=getattr(result.stats, "initial_sheep_zone", None),
            final_sheep_zone=getattr(result.stats, "final_sheep_zone", None),
            corner_steps_total=getattr(result.stats, "corner_steps_total", None),
            corner_time_pct=getattr(result.stats, "corner_time_pct", None),
            wall_steps_total=getattr(result.stats, "wall_steps_total", None),
            wall_time_pct=getattr(result.stats, "wall_time_pct", None),
            corner_stuck_at_end=getattr(result.stats, "corner_stuck_at_end", None),
            corner_entered=getattr(result.stats, "corner_entered", None),
            corner_extracted=getattr(result.stats, "corner_extracted", None),
            spatial_metrics=getattr(result.stats, "spatial_metrics", None),
        )

    def _compute_observation_diagnostics(self, observations: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
        if not observations:
            return {}
        first_step = observations[0]
        if not first_step:
            return {}
        feature_names = list(first_step[0].feature_names)
        num_features = len(feature_names)

        min_vals = []
        max_vals = []
        mean_vals = []
        std_vals = []
        constant_features = []
        nan_or_inf_features = []
        saturated_features = []

        import math

        for i, name in enumerate(feature_names):
            values = []
            for step_obs in observations:
                for dog_obs in step_obs:
                    if i < len(dog_obs.values):
                        values.append(dog_obs.values[i])

            if not values:
                min_vals.append(0.0)
                max_vals.append(0.0)
                mean_vals.append(0.0)
                std_vals.append(0.0)
                constant_features.append(name)
                continue

            has_nan = any(math.isnan(v) or math.isinf(v) for v in values)
            if has_nan:
                nan_or_inf_features.append(name)
                values = [v for v in values if not (math.isnan(v) or math.isinf(v))]

            if not values:
                min_vals.append(0.0)
                max_vals.append(0.0)
                mean_vals.append(0.0)
                std_vals.append(0.0)
                constant_features.append(name)
                continue

            min_v = min(values)
            max_v = max(values)
            mean_v = sum(values) / len(values)
            std_v = math.sqrt(sum((v - mean_v)**2 for v in values) / len(values))

            min_vals.append(min_v)
            max_vals.append(max_v)
            mean_vals.append(mean_v)
            std_vals.append(std_v)

            if std_v < 1e-6:
                constant_features.append(name)

            is_sat = False
            for bound in (-1.0, 0.0, 1.0):
                if all(abs(v - bound) < 1e-4 for v in values):
                    is_sat = True
                    break
            if is_sat and name not in constant_features:
                saturated_features.append(name)

        return {
            "feature_names": feature_names,
            "vector_length": num_features,
            "min_values": min_vals,
            "max_values": max_vals,
            "mean_values": mean_vals,
            "std_values": std_vals,
            "constant_features": constant_features,
            "nan_or_inf_features": nan_or_inf_features,
            "saturated_features": saturated_features,
            "bounds_mismatch": False,
        }

    def _compute_failed_trajectory_summary(self, result: EpisodeResult) -> list[dict[str, Any]]:
        replay = result.replay
        if not replay:
            return []
        steps_count = len(replay)

        min_dist_steps = set()
        running_min_dist = float("inf")
        for idx, frame in enumerate(replay):
            dist = frame.snapshot.average_distance_to_pen
            if dist < running_min_dist:
                running_min_dist = dist
                min_dist_steps.add(idx)

        key_steps = {0, steps_count - 1} | min_dist_steps
        target_count = 20
        step_interval = max(1, steps_count // target_count)
        for idx in range(0, steps_count, step_interval):
            key_steps.add(idx)

        sorted_steps = sorted(list(key_steps))

        summary_rows = []
        for step_idx in sorted_steps:
            if step_idx >= steps_count:
                continue
            frame = replay[step_idx]
            snap = frame.snapshot

            unpenned_sheep = [s for s in snap.sheep if not s.penned]
            avg_dog_to_sheep = 0.0
            if unpenned_sheep and snap.dogs:
                from sheepdog.entities import Point
                total_d = 0.0
                for d in snap.dogs:
                    d_pos = Point(d.x, d.y)
                    dog_d = sum(d_pos.distance_to(Point(s.x, s.y)) for s in unpenned_sheep) / len(unpenned_sheep)
                    total_d += dog_d
                avg_dog_to_sheep = total_d / len(snap.dogs)

            event = ""
            if step_idx == 0:
                event = "initial"
            elif step_idx == steps_count - 1:
                event = f"termination: {snap.status or result.stats.stop_reason}"
            elif step_idx in min_dist_steps:
                event = "new_min_distance"

            summary_rows.append({
                "step": frame.step,
                "dog_positions": [(d.x, d.y) for d in snap.dogs],
                "sheep_positions": [(s.x, s.y) for s in snap.sheep],
                "sheep_distance_to_pen": snap.average_distance_to_pen,
                "dog_to_sheep_distance": avg_dog_to_sheep,
                "selected_actions": list(frame.actions),
                "reward": frame.reward.total if hasattr(frame.reward, "total") else 0.0,
                "reward_breakdown": asdict(frame.reward) if hasattr(frame.reward, "to_dict") or hasattr(frame.reward, "__dataclass_fields__") else {},
                "no_progress_counter": snap.no_progress_steps,
                "event": event
            })

        return summary_rows

