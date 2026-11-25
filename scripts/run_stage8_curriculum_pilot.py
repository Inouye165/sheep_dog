"""Controlled Stage 8 curriculum A/B pilot: Control vs Treatment from Ckpt 613."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.utils import set_random_seed

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import EvaluationSummary, Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy

PARENT_MODEL = Path("artifacts/models/maskable-ppo-05065228.zip")
EFFECTIVE_CONFIG_PATH = Path("artifacts/effective-training-config.json")
TRAINING_STATE_PATH = Path("artifacts/training-state.json")
EXPERIMENT_ROOT = Path("artifacts/experiments/stage8_curriculum_pilot_v1")

DEV_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
HOLDOUT_SEEDS: tuple[int, ...] = (
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
)

CONTROL_SPAWN_MIX: dict[str, float] = {
    "fixed_easy": 0.30,
    "randomized_flock": 0.55,
    "nearby_stray": 0.15,
}
TREATMENT_SPAWN_MIX: dict[str, float] = {
    "randomized_flock": 0.35,
    "nearby_stray": 0.15,
    "split_flock": 0.15,
    "wall_recovery": 0.10,
    "fixed_easy": 0.10,
    "wrong_side_recovery": 0.15,
}

SEGMENTS = 3
SEGMENT_TIMESTEPS = 32_768
TOTAL_TIMESTEPS = SEGMENTS * SEGMENT_TIMESTEPS  # 98,304


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_arm_config(
    base_payload: dict[str, Any],
    arm_dir: Path,
    spawn_mix: dict[str, float],
    *,
    env_workers: int | None = None,
) -> LabConfig:
    payload = copy.deepcopy(base_payload)
    payload["environment"]["spawn_mix"] = dict(spawn_mix)
    payload["training"]["output_dir"] = str(arm_dir)
    payload["training"]["web_export_dir"] = str(arm_dir / "web_export")
    payload["training"]["backup_dir"] = str(arm_dir / "backups")
    payload["training"]["backup_enabled"] = False
    payload["training"]["wandb_enabled"] = False
    if env_workers is not None:
        payload["training"]["ppo_env_workers"] = int(env_workers)
    return LabConfig.from_dict(payload)


def _summarize_evaluation(summary: EvaluationSummary) -> dict[str, Any]:
    records = summary.records
    total = len(records)
    successes = sum(1 for r in records if r.success)
    avg_steps = sum(r.steps for r in records) / total if total else 0.0
    successful_steps = [r.steps for r in records if r.success]
    avg_success_steps = (
        sum(successful_steps) / len(successful_steps) if successful_steps else None
    )

    failures = [r for r in records if not r.success]
    failure_categories = {
        "no-progress": sum(1 for r in failures if r.stop_reason == "no-progress"),
        "timeout": sum(1 for r in failures if r.stop_reason == "timeout" or r.timeout),
        "oscillation_detected": sum(1 for r in failures if bool(r.oscillation_detected)),
        "corner_stuck_at_end": sum(1 for r in failures if bool(r.corner_stuck_at_end)),
    }
    per_seed = {
        str(r.seed): {
            "seed": r.seed,
            "success": r.success,
            "steps": r.steps,
            "reward_total": r.reward_total,
            "sheep_penned": r.sheep_penned,
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
        "success_rate": summary.success_rate,
        "timeout_rate": summary.timeout_rate,
        "stopped_rate": summary.stopped_rate,
        "average_reward": summary.average_reward,
        "average_steps": avg_steps,
        "average_completion_steps_success_only": avg_success_steps,
        "average_sheep_penned": summary.average_sheep_penned,
        "failure_count": len(failures),
        "failure_categories": failure_categories,
        "per_seed": per_seed,
    }


def _evaluate_model(
    model_path: Path,
    eval_config: LabConfig,
    policy_config: dict[str, Any],
    eval_dir: Path,
    label: str,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    policy = NeuralPolicy.load(model_path, eval_config, policy_config=policy_config)
    evaluator = Evaluator(eval_config, eval_dir, tracking_registry=None)
    try:
        dev_summary, _, _ = evaluator.evaluate(
            policy,
            DEV_SEEDS,
            checkpoint_episode=0,
            deterministic=True,
            capture_replays=False,
            capture_step_telemetry=False,
            evaluation_mode=f"stage8_pilot_{label}_dev",
            curriculum_stage=eval_config.environment.curriculum_stage,
            record_in_registry=False,
        )
        holdout_summary, _, _ = evaluator.evaluate(
            policy,
            HOLDOUT_SEEDS,
            checkpoint_episode=0,
            deterministic=True,
            capture_replays=False,
            capture_step_telemetry=False,
            evaluation_mode=f"stage8_pilot_{label}_holdout",
            curriculum_stage=eval_config.environment.curriculum_stage,
            record_in_registry=False,
        )
    finally:
        evaluator.close()
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()

    return {
        "eval_wall_seconds": time.perf_counter() - eval_start,
        "dev": _summarize_evaluation(dev_summary),
        "holdout": _summarize_evaluation(holdout_summary),
    }


def _train_arm(
    name: str,
    parent_copy: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    spawn_mix: dict[str, float],
    effective_lr: float,
    effective_ent: float,
) -> tuple[Path, float]:
    arm_dir = EXPERIMENT_ROOT / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    train_config = _build_arm_config(base_payload, arm_dir, spawn_mix)
    atomic_write_json(arm_dir / "effective_config.json", train_config.to_dict())

    train_seed = int(train_config.training.train_seed)
    set_random_seed(train_seed)

    train_start = time.perf_counter()
    policy = NeuralPolicy.load(parent_copy, train_config, policy_config=policy_config)
    policy.model.tensorboard_log = None
    policy.model.verbose = 0
    try:
        for seg_idx in range(SEGMENTS):
            is_first = seg_idx == 0
            policy.configure_for_training(
                train_config,
                effective_learning_rate=effective_lr,
                effective_entropy_coef=effective_ent,
                reset_rollout_buffer=is_first,
                set_seed=is_first,
            )
            policy.model.learn(
                total_timesteps=SEGMENT_TIMESTEPS,
                reset_num_timesteps=True,
                progress_bar=False,
            )
            seg_path = arm_dir / f"checkpoint_seg{seg_idx + 1}.zip"
            policy.save(seg_path)
        final_model_path = arm_dir / "final_model.zip"
        policy.save(final_model_path)
    finally:
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()

    return final_model_path, time.perf_counter() - train_start


def main() -> None:
    """Run the controlled Stage 8 curriculum A/B pilot and save results."""
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    parent_sha = _sha256(PARENT_MODEL)
    parent_copy = EXPERIMENT_ROOT / "parent_ckpt613.zip"
    shutil.copy2(PARENT_MODEL, parent_copy)

    base_payload = json.loads(EFFECTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    training_state = json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    adaptive_state = training_state.get("adaptive_step_state", {})
    effective_lr = float(adaptive_state.get("effective_learning_rate", 0.00015))
    effective_ent = float(adaptive_state.get("effective_entropy_coef", 0.005))
    policy_config = dict(training_state.get("policy_config", {}))

    eval_config = _build_arm_config(
        base_payload,
        EXPERIMENT_ROOT / "eval_tmp",
        CONTROL_SPAWN_MIX,
        env_workers=1,
    )

    overall_start = time.perf_counter()

    print("Evaluating Baseline (Ckpt 613)...")
    baseline_eval = _evaluate_model(
        parent_copy, eval_config, policy_config, EXPERIMENT_ROOT / "baseline_eval", "baseline"
    )

    print("Training Control arm (98,304 timesteps)...")
    control_model, control_train_s = _train_arm(
        "control",
        parent_copy,
        base_payload,
        policy_config,
        CONTROL_SPAWN_MIX,
        effective_lr,
        effective_ent,
    )
    print(f"Control training done in {control_train_s:.1f}s. Evaluating Control...")
    control_eval = _evaluate_model(
        control_model, eval_config, policy_config, EXPERIMENT_ROOT / "control" / "eval", "control"
    )

    print("Training Treatment arm (98,304 timesteps)...")
    treatment_model, treatment_train_s = _train_arm(
        "treatment",
        parent_copy,
        base_payload,
        policy_config,
        TREATMENT_SPAWN_MIX,
        effective_lr,
        effective_ent,
    )
    print(f"Treatment training done in {treatment_train_s:.1f}s. Evaluating Treatment...")
    treatment_eval = _evaluate_model(
        treatment_model,
        eval_config,
        policy_config,
        EXPERIMENT_ROOT / "treatment" / "eval",
        "treatment",
    )

    results = {
        "experiment": "stage8_curriculum_pilot_v1",
        "parent_model": str(PARENT_MODEL),
        "parent_sha256": parent_sha,
        "parent_unchanged": _sha256(PARENT_MODEL) == parent_sha,
        "total_timesteps_per_arm": TOTAL_TIMESTEPS,
        "segments": SEGMENTS,
        "segment_timesteps": SEGMENT_TIMESTEPS,
        "effective_learning_rate": effective_lr,
        "effective_entropy_coef": effective_ent,
        "train_seed": int(base_payload["training"]["train_seed"]),
        "ppo_env_workers": int(base_payload["training"]["ppo_env_workers"]),
        "dev_seeds": list(DEV_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "control_spawn_mix": CONTROL_SPAWN_MIX,
        "treatment_spawn_mix": TREATMENT_SPAWN_MIX,
        "wall_clock": {
            "total_seconds": time.perf_counter() - overall_start,
            "control_train_seconds": control_train_s,
            "treatment_train_seconds": treatment_train_s,
            "baseline_eval_seconds": baseline_eval["eval_wall_seconds"],
            "control_eval_seconds": control_eval["eval_wall_seconds"],
            "treatment_eval_seconds": treatment_eval["eval_wall_seconds"],
        },
        "arms": {
            "baseline": baseline_eval,
            "control": control_eval,
            "treatment": treatment_eval,
        },
    }
    output_path = EXPERIMENT_ROOT / "pilot_results.json"
    atomic_write_json(output_path, results)
    print(f"Saved pilot results to {output_path}")
    print(json.dumps({
        "wall_clock": results["wall_clock"],
        "baseline_dev_sr": baseline_eval["dev"]["success_rate"],
        "control_dev_sr": control_eval["dev"]["success_rate"],
        "treatment_dev_sr": treatment_eval["dev"]["success_rate"],
        "baseline_holdout_sr": baseline_eval["holdout"]["success_rate"],
        "control_holdout_sr": control_eval["holdout"]["success_rate"],
        "treatment_holdout_sr": treatment_eval["holdout"]["success_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
