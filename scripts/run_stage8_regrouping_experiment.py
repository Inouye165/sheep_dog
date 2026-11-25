"""Stage 8 Controlled 3-Arm Regrouping Experiment (stage8_regrouping_v1)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.evaluation.evaluator import EvaluationRecord, EvaluationSummary, Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy

PARENT_MODEL = Path("artifacts/models/maskable-ppo-05065228.zip")
EXPECTED_PARENT_SHA256 = (
    "da8ce02fcdcd9100601def6fb733762c5c9ce134d11d07058e57919763530e60"
)
EFFECTIVE_CONFIG_PATH = Path("artifacts/effective-training-config.json")
TRAINING_STATE_PATH = Path("artifacts/training-state.json")
TELEMETRY_DB_PATH = Path("artifacts/training-telemetry.sqlite")
EXPERIMENT_ROOT = Path("artifacts/experiments/stage8_regrouping_v1")

TRAIN_SEED = 7
SEGMENTS = 10
SEGMENT_TIMESTEPS = 32_768
TOTAL_TIMESTEPS = SEGMENTS * SEGMENT_TIMESTEPS  # 327,680
DEV_EVAL_SEGMENTS: tuple[int, ...] = (2, 4, 6, 8, 10)
EFFECTIVE_LR = 0.00015
EFFECTIVE_ENTROPY_COEF = 0.020
PPO_ENV_WORKERS = 16

DEV_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
DIAG_20_SEEDS: tuple[int, ...] = (
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
)
DIAG_MILD_SEEDS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
DIAG_SAFE_SEEDS: tuple[int, ...] = (2031, 2032, 2033, 2034, 2035)
CONFIRM_MILD_SEEDS: tuple[int, ...] = tuple(range(4001, 4011))
CONFIRM_SAFE_SEEDS: tuple[int, ...] = tuple(range(4011, 4021))

DEV_SPAWN_MIX: dict[str, float] = {
    "fixed_easy": 0.30,
    "randomized_flock": 0.55,
    "nearby_stray": 0.15,
}

# Arm B's 50% wrong_side_mild + 50% Treatment-v2 base
REGROUP_SPAWN_MIX: dict[str, float] = {
    "randomized_flock": 0.245,
    "nearby_stray": 0.082,
    "split_flock": 0.082,
    "coherent_wall_recovery": 0.054,
    "fixed_easy": 0.037,
    "wrong_side_mild": 0.50,
}

ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "arm_a",
        "label": "Arm A (Control: scale=0.0, reset=False)",
        "regrouping_approach_scale": 0.0,
        "regroup_no_progress_reset": False,
        "regroup_no_progress_max_resets": 80,
    },
    {
        "arm_id": "arm_b",
        "label": "Arm B (Reward Credit Only: scale=2.0, reset=False)",
        "regrouping_approach_scale": 2.0,
        "regroup_no_progress_reset": False,
        "regroup_no_progress_max_resets": 80,
    },
    {
        "arm_id": "arm_c",
        "label": "Arm C (No-Progress Accounting Only: scale=0.0, reset=True)",
        "regrouping_approach_scale": 0.0,
        "regroup_no_progress_reset": True,
        "regroup_no_progress_max_resets": 80,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_arm_config(
    base_payload: dict[str, Any],
    arm_dir: Path,
    spawn_mix: dict[str, float],
    *,
    train_seed: int | None = None,
    env_workers: int | None = None,
    entropy_coef: float | None = None,
    regrouping_approach_scale: float = 0.0,
    regroup_no_progress_reset: bool = False,
    regroup_no_progress_max_resets: int = 80,
) -> LabConfig:
    payload = copy.deepcopy(base_payload)
    payload["environment"]["spawn_mix"] = dict(spawn_mix)
    payload["environment"]["regroup_no_progress_reset"] = bool(regroup_no_progress_reset)
    payload["environment"]["regroup_no_progress_max_resets"] = int(
        regroup_no_progress_max_resets
    )
    payload["rewards"]["regrouping_approach_scale"] = float(regrouping_approach_scale)
    payload["training"]["output_dir"] = str(arm_dir)
    payload["training"]["web_export_dir"] = str(arm_dir / "web_export")
    payload["training"]["backup_dir"] = str(arm_dir / "backups")
    payload["training"]["backup_enabled"] = False
    payload["training"]["wandb_enabled"] = False
    if train_seed is not None:
        payload["training"]["train_seed"] = int(train_seed)
    if env_workers is not None:
        payload["training"]["ppo_env_workers"] = int(env_workers)
    if entropy_coef is not None:
        payload["training"]["entropy_coef"] = float(entropy_coef)
    return LabConfig.from_dict(payload)


def _summarize_records(records: list[EvaluationRecord]) -> dict[str, Any]:
    total = len(records)
    successes = sum(1 for r in records if r.success)
    timeouts = sum(1 for r in records if r.timeout)
    stopped = sum(1 for r in records if r.stopped)
    avg_steps = sum(r.steps for r in records) / total if total else 0.0
    successful_steps = [r.steps for r in records if r.success]
    avg_success_steps = (
        sum(successful_steps) / len(successful_steps) if successful_steps else None
    )
    avg_reward = sum(r.reward_total for r in records) / total if total else 0.0
    avg_sheep_penned = sum(r.sheep_penned for r in records) / total if total else 0.0
    avg_no_progress_steps = sum(r.no_progress_steps for r in records) / total if total else 0.0

    failures = [r for r in records if not r.success]
    no_progress_count = sum(1 for r in failures if r.stop_reason == "no-progress")
    timeout_count = sum(1 for r in failures if r.stop_reason == "timeout" or r.timeout)
    osc_all = sum(1 for r in records if bool(r.oscillation_detected))
    osc_failures = sum(1 for r in failures if bool(r.oscillation_detected))
    corner_all = sum(1 for r in records if bool(r.corner_stuck_at_end))
    corner_failures = sum(1 for r in failures if bool(r.corner_stuck_at_end))

    per_seed = {
        str(r.seed): {
            "seed": r.seed,
            "success": r.success,
            "steps": r.steps,
            "reward_total": r.reward_total,
            "sheep_penned": r.sheep_penned,
            "no_progress_steps": r.no_progress_steps,
            "stop_reason": r.stop_reason,
            "timeout": r.timeout,
            "oscillation_detected": bool(r.oscillation_detected),
            "corner_stuck_at_end": bool(r.corner_stuck_at_end),
            "spawn_mode": r.spawn_mode,
        }
        for r in records
    }

    return {
        "total_episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "timeout_rate": timeouts / total if total else 0.0,
        "stopped_rate": stopped / total if total else 0.0,
        "average_reward": avg_reward,
        "average_steps": avg_steps,
        "average_completion_steps_success_only": avg_success_steps,
        "average_sheep_penned": avg_sheep_penned,
        "average_no_progress_steps": avg_no_progress_steps,
        "failure_count": len(failures),
        "failure_categories": {
            "no-progress": no_progress_count,
            "timeout": timeout_count,
            "oscillation_detected_failures": osc_failures,
            "oscillation_detected_all": osc_all,
            "corner_stuck_at_end_failures": corner_failures,
            "corner_stuck_at_end_all": corner_all,
        },
        "per_seed": per_seed,
    }


def _evaluate_seeds(
    policy: NeuralPolicy,
    eval_config: LabConfig,
    seeds: tuple[int, ...],
    eval_dir: Path,
    mode_label: str,
) -> tuple[EvaluationSummary, list[EvaluationRecord]]:
    evaluator = Evaluator(eval_config, eval_dir, tracking_registry=None)
    try:
        summary, _, _ = evaluator.evaluate(
            policy,
            seeds,
            checkpoint_episode=0,
            deterministic=True,
            capture_replays=False,
            capture_step_telemetry=False,
            evaluation_mode=mode_label,
            curriculum_stage=eval_config.environment.curriculum_stage,
            record_in_registry=False,
        )
    finally:
        evaluator.close()
    return summary, list(summary.records)


def _compute_step0_entropy(
    policy: NeuralPolicy,
    env: SheepdogEnvironment,
    seed: int,
) -> dict[str, Any]:
    """Compute policy action entropy at step 0 across all 3 dogs for a given seed."""
    env.reset(seed=seed)
    if hasattr(env, "prepare_policy_step"):
        env.prepare_policy_step()
    reserved_positions: set[Any] = set()
    dog_entropies: list[float] = []
    for dog_index in range(env.dog_count):
        obs_obj = env.build_observation_for_dog(dog_index)
        obs = np.asarray(obs_obj.values, dtype=np.float32)
        mask_map = env.action_mask_for_dog(
            dog_index,
            reserved_positions=reserved_positions,
        )
        action_masks = np.asarray([mask_map[act] for act in ACTION_ORDER], dtype=bool)
        with torch.no_grad():
            obs_tensor, _ = policy.model.policy.obs_to_tensor(obs.reshape(1, -1))
            masks_tensor = torch.as_tensor(
                action_masks.reshape(1, -1),
                device=policy.model.policy.device,
            )
            dist = policy.model.policy.get_distribution(obs_tensor, action_masks=masks_tensor)
            ent = float(dist.distribution.entropy().mean().item())
        dog_entropies.append(ent)
        action_index, _ = policy.model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        action = ACTION_ORDER[int(action_index)]
        reserved_positions.add(env.project_dog_action(dog_index, action))
    return {
        "seed": seed,
        "mean_entropy": statistics.mean(dog_entropies),
        "per_dog_entropy": dog_entropies,
    }


def _measure_recovery_diagnostics(
    policy: NeuralPolicy,
    probe_cfg: LabConfig,
    seeds: tuple[int, ...],
    entropy_seed: int,
) -> dict[str, Any]:
    """Run deterministic rollouts on diagnostic recovery seeds to measure flanking & early sheep displacement."""
    env = SheepdogEnvironment(probe_cfg)
    flank_reached_flags: list[bool] = []
    early_disp_means: list[float] = []
    per_seed_diag: dict[str, Any] = {}

    for seed in seeds:
        result = env.run_policy(policy, seed=seed, capture_replay=True, deterministic=True)
        frames = []
        if result.initial_snapshot is not None:
            frames.append(result.initial_snapshot)
        frames.extend(rec.snapshot for rec in result.replay)

        flank_reached = False
        for snap in frames:
            unpenned = [s for s in snap.sheep if not s.penned]
            sheep_list = unpenned if unpenned else list(snap.sheep)
            fx = sum(s.x for s in sheep_list) / len(sheep_list)
            fy = sum(s.y for s in sheep_list) / len(sheep_list)
            pen_c = snap.pen.center
            dx = float(pen_c.x) - fx
            dy = float(pen_c.y) - fy
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                ux, uy = dx / norm, dy / norm
                for dog in snap.dogs:
                    forward_proj = (dog.x - fx) * ux + (dog.y - fy) * uy
                    if forward_proj < 0.0:
                        flank_reached = True
                        break
            if flank_reached:
                break

        max_frame_idx = min(40, len(frames) - 1)
        sheep_disps: list[float] = []
        if max_frame_idx >= 1:
            num_sheep = len(frames[0].sheep)
            for s_idx in range(num_sheep):
                total_d = 0.0
                for t in range(1, max_frame_idx + 1):
                    prev_s = frames[t - 1].sheep[s_idx]
                    curr_s = frames[t].sheep[s_idx]
                    total_d += math.hypot(curr_s.x - prev_s.x, curr_s.y - prev_s.y)
                sheep_disps.append(total_d)
        mean_ep_disp = statistics.mean(sheep_disps) if sheep_disps else 0.0

        flank_reached_flags.append(flank_reached)
        early_disp_means.append(mean_ep_disp)
        per_seed_diag[str(seed)] = {
            "flank_reached": flank_reached,
            "early_sheep_disp_40": mean_ep_disp,
            "success": result.stats.success,
            "steps": result.stats.steps,
            "stop_reason": result.stats.stop_reason,
            "no_progress_steps": result.stats.no_progress_steps,
        }

    step0_ent = _compute_step0_entropy(policy, env, entropy_seed)
    return {
        "flank_reached_pct": sum(1 for f in flank_reached_flags if f) / len(flank_reached_flags),
        "flank_reached_count": sum(1 for f in flank_reached_flags if f),
        "early_sheep_disp": statistics.mean(early_disp_means) if early_disp_means else 0.0,
        "step0_entropy": step0_ent,
        "per_seed_diag": per_seed_diag,
    }


def _evaluate_checkpoint_screen(
    model_path: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    eval_root: Path,
    label: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on Dev-10, Diag Mild-5, and Diag Safe-5 using standard eval rules."""
    eval_start = time.perf_counter()

    dev_cfg = _build_arm_config(base_payload, eval_root / "dev", DEV_SPAWN_MIX, env_workers=1)
    dev_policy = NeuralPolicy.load(model_path, dev_cfg, policy_config=policy_config)
    try:
        _, dev_records = _evaluate_seeds(
            dev_policy, dev_cfg, DEV_SEEDS, eval_root / "dev", f"regroup_{label}_dev"
        )
    finally:
        v_env = dev_policy.model.get_env()
        if v_env is not None:
            v_env.close()
    dev_summary = _summarize_records(dev_records)

    mild_cfg = _build_arm_config(
        base_payload, eval_root / "diag_mild", {"wrong_side_mild": 1.0}, env_workers=1
    )
    mild_policy = NeuralPolicy.load(model_path, mild_cfg, policy_config=policy_config)
    try:
        _, mild_records = _evaluate_seeds(
            mild_policy,
            mild_cfg,
            DIAG_MILD_SEEDS,
            eval_root / "diag_mild",
            f"regroup_{label}_diag_mild",
        )
        mild_diag = _measure_recovery_diagnostics(
            mild_policy, mild_cfg, DIAG_MILD_SEEDS, entropy_seed=2024
        )
    finally:
        m_env = mild_policy.model.get_env()
        if m_env is not None:
            m_env.close()
    mild_summary = _summarize_records(mild_records)
    mild_summary.update(mild_diag)

    safe_cfg = _build_arm_config(
        base_payload, eval_root / "diag_safe", {"wrong_side_safe_radius": 1.0}, env_workers=1
    )
    safe_policy = NeuralPolicy.load(model_path, safe_cfg, policy_config=policy_config)
    try:
        _, safe_records = _evaluate_seeds(
            safe_policy,
            safe_cfg,
            DIAG_SAFE_SEEDS,
            eval_root / "diag_safe",
            f"regroup_{label}_diag_safe",
        )
        safe_diag = _measure_recovery_diagnostics(
            safe_policy, safe_cfg, DIAG_SAFE_SEEDS, entropy_seed=2034
        )
    finally:
        s_env = safe_policy.model.get_env()
        if s_env is not None:
            s_env.close()
    safe_summary = _summarize_records(safe_records)
    safe_summary.update(safe_diag)

    rec_successes = mild_summary["successes"] + safe_summary["successes"]
    rec_reward = 0.5 * (mild_summary["average_reward"] + safe_summary["average_reward"])
    return {
        "eval_wall_seconds": time.perf_counter() - eval_start,
        "dev_10": dev_summary,
        "diag_wrong_side_mild": mild_summary,
        "diag_wrong_side_safe_radius": safe_summary,
        "combined_recovery_successes": rec_successes,
        "combined_score_key": [
            dev_summary["success_rate"] + (rec_successes / 10.0),
            dev_summary["average_reward"] + rec_reward,
        ],
    }


def _evaluate_diag20(
    model_path: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    eval_root: Path,
    label: str,
) -> dict[str, Any]:
    cfg = _build_arm_config(base_payload, eval_root / "diag20", DEV_SPAWN_MIX, env_workers=1)
    policy = NeuralPolicy.load(model_path, cfg, policy_config=policy_config)
    try:
        _, records = _evaluate_seeds(
            policy, cfg, DIAG_20_SEEDS, eval_root / "diag20", f"regroup_{label}_diag20"
        )
    finally:
        v_env = policy.model.get_env()
        if v_env is not None:
            v_env.close()
    return _summarize_records(records)


def _evaluate_confirmation(
    model_path: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    eval_root: Path,
    label: str,
) -> dict[str, Any]:
    mild_cfg = _build_arm_config(
        base_payload, eval_root / "confirm_mild", {"wrong_side_mild": 1.0}, env_workers=1
    )
    mild_policy = NeuralPolicy.load(model_path, mild_cfg, policy_config=policy_config)
    try:
        _, mild_records = _evaluate_seeds(
            mild_policy,
            mild_cfg,
            CONFIRM_MILD_SEEDS,
            eval_root / "confirm_mild",
            f"regroup_{label}_confirm_mild",
        )
    finally:
        m_env = mild_policy.model.get_env()
        if m_env is not None:
            m_env.close()

    safe_cfg = _build_arm_config(
        base_payload, eval_root / "confirm_safe", {"wrong_side_safe_radius": 1.0}, env_workers=1
    )
    safe_policy = NeuralPolicy.load(model_path, safe_cfg, policy_config=policy_config)
    try:
        _, safe_records = _evaluate_seeds(
            safe_policy,
            safe_cfg,
            CONFIRM_SAFE_SEEDS,
            eval_root / "confirm_safe",
            f"regroup_{label}_confirm_safe",
        )
    finally:
        s_env = safe_policy.model.get_env()
        if s_env is not None:
            s_env.close()

    mild_summary = _summarize_records(mild_records)
    safe_summary = _summarize_records(safe_records)
    total_confirm_successes = mild_summary["successes"] + safe_summary["successes"]
    return {
        "confirm_wrong_side_mild_10": mild_summary,
        "confirm_wrong_side_safe_radius_10": safe_summary,
        "confirm_20_successes": total_confirm_successes,
        "confirm_20_success_rate": total_confirm_successes / 20.0,
    }


def _train_and_eval_arm(
    arm_spec: dict[str, Any],
    parent_copy: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    arm_id = arm_spec["arm_id"]
    arm_dir = EXPERIMENT_ROOT / arm_id
    result_path = arm_dir / "arm_result.json"
    if result_path.exists():
        print(f"  [{arm_id}] Reusing cached {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    train_config = _build_arm_config(
        base_payload,
        arm_dir,
        REGROUP_SPAWN_MIX,
        train_seed=TRAIN_SEED,
        env_workers=PPO_ENV_WORKERS,
        entropy_coef=EFFECTIVE_ENTROPY_COEF,
        regrouping_approach_scale=arm_spec["regrouping_approach_scale"],
        regroup_no_progress_reset=arm_spec["regroup_no_progress_reset"],
        regroup_no_progress_max_resets=arm_spec["regroup_no_progress_max_resets"],
    )
    atomic_write_json(arm_dir / "effective_config.json", train_config.to_dict())

    set_random_seed(TRAIN_SEED)
    train_start = time.perf_counter()
    policy = NeuralPolicy.load(parent_copy, train_config, policy_config=policy_config)
    policy.model.tensorboard_log = None
    policy.model.verbose = 0

    segment_paths: dict[int, Path] = {}
    try:
        for seg_idx in range(1, SEGMENTS + 1):
            is_first = seg_idx == 1
            policy.configure_for_training(
                train_config,
                effective_learning_rate=EFFECTIVE_LR,
                effective_entropy_coef=EFFECTIVE_ENTROPY_COEF,
                reset_rollout_buffer=is_first,
                set_seed=is_first,
            )
            policy.model.learn(
                total_timesteps=SEGMENT_TIMESTEPS,
                reset_num_timesteps=True,
                progress_bar=False,
            )
            if seg_idx in DEV_EVAL_SEGMENTS:
                seg_path = arm_dir / f"checkpoint_seg{seg_idx}.zip"
                policy.save(seg_path)
                segment_paths[seg_idx] = seg_path
    finally:
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()

    train_wall_seconds = time.perf_counter() - train_start
    print(f"  [{arm_id}] Training finished in {train_wall_seconds:.1f}s")

    checkpoints: dict[str, Any] = {}
    best_seg: int | None = None
    best_key: tuple[float, float] | None = None
    for seg_idx in DEV_EVAL_SEGMENTS:
        seg_path = segment_paths[seg_idx]
        screen = _evaluate_checkpoint_screen(
            seg_path,
            base_payload,
            policy_config,
            arm_dir / f"eval_seg{seg_idx}",
            f"{arm_id}_seg{seg_idx}",
        )
        checkpoints[f"seg_{seg_idx}"] = {
            "segment": seg_idx,
            "timesteps": seg_idx * SEGMENT_TIMESTEPS,
            "checkpoint_path": str(seg_path),
            **screen,
        }
        key = (
            float(screen["combined_score_key"][0]),
            float(screen["combined_score_key"][1]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_seg = seg_idx
        print(
            f"    [{arm_id} seg_{seg_idx}] "
            f"dev={screen['dev_10']['successes']}/10 ({screen['dev_10']['average_reward']:.1f}) | "
            f"mild={screen['diag_wrong_side_mild']['successes']}/5 "
            f"(flank={screen['diag_wrong_side_mild']['flank_reached_pct']*100:.0f}%, "
            f"disp={screen['diag_wrong_side_mild']['early_sheep_disp']:.2f}, "
            f"H0={screen['diag_wrong_side_mild']['step0_entropy']['mean_entropy']:.3f}) | "
            f"safe={screen['diag_wrong_side_safe_radius']['successes']}/5 "
            f"(flank={screen['diag_wrong_side_safe_radius']['flank_reached_pct']*100:.0f}%, "
            f"disp={screen['diag_wrong_side_safe_radius']['early_sheep_disp']:.2f}, "
            f"H0={screen['diag_wrong_side_safe_radius']['step0_entropy']['mean_entropy']:.3f})"
        )

    assert best_seg is not None
    best_src = segment_paths[best_seg]
    best_dst = arm_dir / "best_combined_model.zip"
    shutil.copy2(best_src, best_dst)

    diag20_best = _evaluate_diag20(
        best_dst, base_payload, policy_config, arm_dir / "diag20_best", f"{arm_id}_best"
    )
    confirmation_20 = _evaluate_confirmation(
        best_dst, base_payload, policy_config, arm_dir / "confirm", f"{arm_id}_best"
    )

    arm_result = {
        "arm_id": arm_id,
        "label": arm_spec["label"],
        "regrouping_approach_scale": arm_spec["regrouping_approach_scale"],
        "regroup_no_progress_reset": arm_spec["regroup_no_progress_reset"],
        "regroup_no_progress_max_resets": arm_spec["regroup_no_progress_max_resets"],
        "spawn_mix": REGROUP_SPAWN_MIX,
        "train_seed": TRAIN_SEED,
        "entropy_coef": EFFECTIVE_ENTROPY_COEF,
        "train_wall_seconds": train_wall_seconds,
        "selected_best_segment": best_seg,
        "selected_best_timesteps": best_seg * SEGMENT_TIMESTEPS,
        "selected_best_path": str(best_dst),
        "checkpoints": checkpoints,
        "diag_20_best": diag20_best,
        "confirmation_20": confirmation_20,
    }
    atomic_write_json(result_path, arm_result)
    return arm_result


def main() -> None:
    """Execute the 3-arm Stage 8 Regrouping Experiment (stage8_regrouping_v1)."""
    overall_start = time.perf_counter()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    set_global_capture_policy(CapturePolicy(mode="off"))

    parent_sha_before = _sha256(PARENT_MODEL)
    if parent_sha_before != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"Parent SHA mismatch: {parent_sha_before} != {EXPECTED_PARENT_SHA256}"
        )
    eff_cfg_sha_before = _sha256(EFFECTIVE_CONFIG_PATH)
    state_sha_before = _sha256(TRAINING_STATE_PATH)
    db_sha_before = _sha256(TELEMETRY_DB_PATH) if TELEMETRY_DB_PATH.exists() else None

    parent_copy = EXPERIMENT_ROOT / "parent_ckpt613.zip"
    if not parent_copy.exists():
        shutil.copy2(PARENT_MODEL, parent_copy)

    base_payload = json.loads(EFFECTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    state_payload = json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    policy_config = dict(state_payload.get("policy_config", {}))

    print("=== Stage 8 Regrouping Experiment (stage8_regrouping_v1) ===")
    arms_out: dict[str, Any] = {}
    for spec in ARM_SPECS:
        print(f"Running {spec['label']}...")
        arms_out[spec["arm_id"]] = _train_and_eval_arm(
            spec, parent_copy, base_payload, policy_config
        )

    parent_sha_after = _sha256(PARENT_MODEL)
    eff_cfg_sha_after = _sha256(EFFECTIVE_CONFIG_PATH)
    state_sha_after = _sha256(TRAINING_STATE_PATH)
    db_sha_after = _sha256(TELEMETRY_DB_PATH) if TELEMETRY_DB_PATH.exists() else None

    results = {
        "experiment_id": "stage8_regrouping_v1",
        "parent_checkpoint": {
            "path": str(PARENT_MODEL),
            "sha256_before": parent_sha_before,
            "sha256_after": parent_sha_after,
            "unchanged": parent_sha_before == parent_sha_after == EXPECTED_PARENT_SHA256,
        },
        "protected_artifacts_unchanged": {
            "effective_training_config": eff_cfg_sha_before == eff_cfg_sha_after,
            "training_state": state_sha_before == state_sha_after,
            "telemetry_db": db_sha_before == db_sha_after,
        },
        "protocol": {
            "train_seed": TRAIN_SEED,
            "segments": SEGMENTS,
            "segment_timesteps": SEGMENT_TIMESTEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_segments": list(DEV_EVAL_SEGMENTS),
            "effective_learning_rate": EFFECTIVE_LR,
            "effective_entropy_coef": EFFECTIVE_ENTROPY_COEF,
            "ppo_env_workers": PPO_ENV_WORKERS,
            "spawn_mix": REGROUP_SPAWN_MIX,
            "dev_seeds": list(DEV_SEEDS),
            "diag_20_seeds": list(DIAG_20_SEEDS),
            "diag_mild_seeds": list(DIAG_MILD_SEEDS),
            "diag_safe_seeds": list(DIAG_SAFE_SEEDS),
            "confirm_mild_seeds": list(CONFIRM_MILD_SEEDS),
            "confirm_safe_seeds": list(CONFIRM_SAFE_SEEDS),
        },
        "arms": arms_out,
        "total_wall_seconds": time.perf_counter() - overall_start,
    }
    atomic_write_json(EXPERIMENT_ROOT / "regrouping_results.json", results)
    print(
        f"Completed in {results['total_wall_seconds']:.1f}s -> "
        f"{EXPERIMENT_ROOT / 'regrouping_results.json'}"
    )


if __name__ == "__main__":
    main()
