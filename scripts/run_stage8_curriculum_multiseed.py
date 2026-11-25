"""Controlled multi-seed Stage 8 curriculum comparison: Control vs Treatment from Ckpt 613."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.utils import set_random_seed

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import EvaluationRecord, EvaluationSummary, Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy

PARENT_MODEL = Path("artifacts/models/maskable-ppo-05065228.zip")
EFFECTIVE_CONFIG_PATH = Path("artifacts/effective-training-config.json")
TRAINING_STATE_PATH = Path("artifacts/training-state.json")
EXPERIMENT_ROOT = Path("artifacts/experiments/stage8_curriculum_multiseed_v1")

TRAIN_SEEDS: tuple[int, ...] = (7, 101, 202)
DEV_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
DIAG_SEEDS: tuple[int, ...] = (
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
)
UNTOUCHED_TEST_SEEDS: tuple[int, ...] = (
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061,
    1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123,
    1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
)
STRESS_PROBE_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("wrong_side_recovery", (2001, 2002, 2003, 2004, 2005)),
    ("split_flock", (2006, 2007, 2008, 2009, 2010)),
    ("wall_recovery", (2011, 2012, 2013, 2014, 2015)),
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

SEGMENTS = 10
SEGMENT_TIMESTEPS = 32_768
TOTAL_TIMESTEPS = SEGMENTS * SEGMENT_TIMESTEPS  # 327,680
DEV_EVAL_SEGMENTS: tuple[int, ...] = (2, 4, 6, 8, 10)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_arm_config(
    base_payload: dict[str, Any],
    arm_dir: Path,
    spawn_mix: dict[str, float],
    *,
    train_seed: int | None = None,
    env_workers: int | None = None,
) -> LabConfig:
    payload = copy.deepcopy(base_payload)
    payload["environment"]["spawn_mix"] = dict(spawn_mix)
    payload["training"]["output_dir"] = str(arm_dir)
    payload["training"]["web_export_dir"] = str(arm_dir / "web_export")
    payload["training"]["backup_dir"] = str(arm_dir / "backups")
    payload["training"]["backup_enabled"] = False
    payload["training"]["wandb_enabled"] = False
    if train_seed is not None:
        payload["training"]["train_seed"] = int(train_seed)
    if env_workers is not None:
        payload["training"]["ppo_env_workers"] = int(env_workers)
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

    by_category: dict[str, list[EvaluationRecord]] = {}
    for r in records:
        by_category.setdefault(r.spawn_mode, []).append(r)

    per_category: dict[str, dict[str, Any]] = {}
    for cat, cat_records in sorted(by_category.items()):
        cat_n = len(cat_records)
        cat_succ = sum(1 for r in cat_records if r.success)
        cat_succ_steps = [r.steps for r in cat_records if r.success]
        per_category[cat] = {
            "count": cat_n,
            "successes": cat_succ,
            "success_rate": cat_succ / cat_n if cat_n else 0.0,
            "average_sheep_penned": sum(r.sheep_penned for r in cat_records) / cat_n,
            "average_steps": sum(r.steps for r in cat_records) / cat_n,
            "average_completion_steps_success_only": (
                sum(cat_succ_steps) / len(cat_succ_steps) if cat_succ_steps else None
            ),
            "average_reward": sum(r.reward_total for r in cat_records) / cat_n,
            "average_no_progress_steps": (
                sum(r.no_progress_steps for r in cat_records) / cat_n
            ),
            "no_progress_count": sum(
                1 for r in cat_records if not r.success and r.stop_reason == "no-progress"
            ),
            "oscillation_detected_count": sum(
                1 for r in cat_records if bool(r.oscillation_detected)
            ),
            "corner_stuck_at_end_count": sum(
                1 for r in cat_records if bool(r.corner_stuck_at_end)
            ),
        }

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
        "per_category": per_category,
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


def _evaluate_dev_only(
    model_path: Path,
    eval_config: LabConfig,
    policy_config: dict[str, Any],
    eval_dir: Path,
    label: str,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    policy = NeuralPolicy.load(model_path, eval_config, policy_config=policy_config)
    try:
        summary, records = _evaluate_seeds(
            policy, eval_config, DEV_SEEDS, eval_dir, f"stage8_ms_{label}_dev"
        )
    finally:
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()
    result = _summarize_records(records)
    result["eval_wall_seconds"] = time.perf_counter() - eval_start
    result["selection_key"] = [
        summary.success_rate,
        summary.average_reward,
        -summary.average_completion_steps,
    ]
    return result


def _evaluate_full_suite(
    model_path: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    eval_root: Path,
    label: str,
    *,
    precomputed_dev: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    eval_config = _build_arm_config(
        base_payload,
        eval_root / "cfg",
        CONTROL_SPAWN_MIX,
        env_workers=1,
    )
    policy = NeuralPolicy.load(model_path, eval_config, policy_config=policy_config)
    try:
        if precomputed_dev is not None:
            dev_dict = precomputed_dev
        else:
            _, dev_records = _evaluate_seeds(
                policy, eval_config, DEV_SEEDS, eval_root / "dev", f"stage8_ms_{label}_dev"
            )
            dev_dict = _summarize_records(dev_records)

        _, diag_records = _evaluate_seeds(
            policy, eval_config, DIAG_SEEDS, eval_root / "diag", f"stage8_ms_{label}_diag"
        )
        diag_dict = _summarize_records(diag_records)

        _, test_records = _evaluate_seeds(
            policy,
            eval_config,
            UNTOUCHED_TEST_SEEDS,
            eval_root / "untouched_test",
            f"stage8_ms_{label}_untouched_test",
        )
        test_dict = _summarize_records(test_records)
    finally:
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()

    # Category stress probe: temporary spawn_mix override per category
    probe_records_all: list[EvaluationRecord] = []
    probe_by_category: dict[str, Any] = {}
    for cat_name, cat_seeds in STRESS_PROBE_SPECS:
        probe_cfg = _build_arm_config(
            base_payload,
            eval_root / f"probe_{cat_name}",
            {cat_name: 1.0},
            env_workers=1,
        )
        probe_policy = NeuralPolicy.load(model_path, probe_cfg, policy_config=policy_config)
        try:
            _, cat_records = _evaluate_seeds(
                probe_policy,
                probe_cfg,
                cat_seeds,
                eval_root / f"probe_{cat_name}",
                f"stage8_ms_{label}_probe_{cat_name}",
            )
        finally:
            p_env = probe_policy.model.get_env()
            if p_env is not None:
                p_env.close()
        probe_records_all.extend(cat_records)
        probe_by_category[cat_name] = _summarize_records(cat_records)

    probe_dict = _summarize_records(probe_records_all)
    probe_dict["by_probe_category"] = probe_by_category

    return {
        "eval_wall_seconds": time.perf_counter() - eval_start,
        "dev": dev_dict,
        "diagnostic_20": diag_dict,
        "untouched_test_30": test_dict,
        "stress_probe_15": probe_dict,
    }


def _train_and_select_arm(
    arm_name: str,
    train_seed: int,
    parent_copy: Path,
    base_payload: dict[str, Any],
    policy_config: dict[str, Any],
    spawn_mix: dict[str, float],
    effective_lr: float,
    effective_ent: float,
) -> dict[str, Any]:
    arm_dir = EXPERIMENT_ROOT / f"seed_{train_seed}" / arm_name
    run_result_path = arm_dir / "run_result.json"
    if run_result_path.exists():
        print(f"  [seed={train_seed} {arm_name}] Reusing cached {run_result_path}")
        return json.loads(run_result_path.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    eval_config = _build_arm_config(
        base_payload,
        arm_dir / "dev_eval_cfg",
        CONTROL_SPAWN_MIX,
        env_workers=1,
    )

    expected_ckpts = {s: arm_dir / f"checkpoint_seg{s}.zip" for s in DEV_EVAL_SEGMENTS}
    if all(p.exists() for p in expected_ckpts.values()) and (arm_dir / "best_dev_model.zip").exists():
        segment_paths = expected_ckpts
        cfg_path = arm_dir / "effective_config.json"
        if cfg_path.exists():
            train_wall_seconds = max(
                0.0,
                expected_ckpts[DEV_EVAL_SEGMENTS[-1]].stat().st_mtime - cfg_path.stat().st_mtime,
            )
        else:
            train_wall_seconds = 0.0
        print(
            f"  [seed={train_seed} {arm_name}] Reusing completed checkpoints "
            f"(estimated train={train_wall_seconds:.1f}s)"
        )
    else:
        train_config = _build_arm_config(
            base_payload,
            arm_dir,
            spawn_mix,
            train_seed=train_seed,
        )
        atomic_write_json(arm_dir / "effective_config.json", train_config.to_dict())

        set_random_seed(train_seed)
        train_start = time.perf_counter()
        policy = NeuralPolicy.load(parent_copy, train_config, policy_config=policy_config)
        policy.model.tensorboard_log = None
        policy.model.verbose = 0

        segment_paths = {}
        try:
            for seg_idx in range(1, SEGMENTS + 1):
                is_first = seg_idx == 1
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
                if seg_idx in DEV_EVAL_SEGMENTS:
                    seg_path = arm_dir / f"checkpoint_seg{seg_idx}.zip"
                    policy.save(seg_path)
                    segment_paths[seg_idx] = seg_path
        finally:
            vec_env = policy.model.get_env()
            if vec_env is not None:
                vec_env.close()

        train_wall_seconds = time.perf_counter() - train_start

    # Multi-checkpoint dev validation
    dev_checkpoints: dict[str, Any] = {}
    best_seg: int | None = None
    best_key: tuple[float, float, float] | None = None
    for seg_idx in DEV_EVAL_SEGMENTS:
        seg_path = segment_paths[seg_idx]
        dev_res = _evaluate_dev_only(
            seg_path,
            eval_config,
            policy_config,
            arm_dir / f"dev_eval_seg{seg_idx}",
            f"s{train_seed}_{arm_name}_seg{seg_idx}",
        )
        key = (
            float(dev_res["selection_key"][0]),
            float(dev_res["selection_key"][1]),
            float(dev_res["selection_key"][2]),
        )
        dev_checkpoints[f"seg_{seg_idx}"] = {
            "segment": seg_idx,
            "timesteps": seg_idx * SEGMENT_TIMESTEPS,
            "checkpoint_path": str(seg_path),
            "dev_summary": dev_res,
        }
        if best_key is None or key > best_key:
            best_key = key
            best_seg = seg_idx

    assert best_seg is not None
    best_src = segment_paths[best_seg]
    best_dst = arm_dir / "best_dev_model.zip"
    shutil.copy2(best_src, best_dst)

    print(
        f"  [seed={train_seed} {arm_name}] train={train_wall_seconds:.1f}s | "
        f"best=seg_{best_seg} ({best_seg * SEGMENT_TIMESTEPS} ts) "
        f"dev_sr={dev_checkpoints[f'seg_{best_seg}']['dev_summary']['success_rate']:.2f} "
        f"dev_rew={dev_checkpoints[f'seg_{best_seg}']['dev_summary']['average_reward']:.1f}"
    )

    full_eval = _evaluate_full_suite(
        best_dst,
        base_payload,
        policy_config,
        arm_dir / "final_eval",
        f"s{train_seed}_{arm_name}_best",
        precomputed_dev=dev_checkpoints[f"seg_{best_seg}"]["dev_summary"],
    )

    run_result = {
        "train_seed": train_seed,
        "arm": arm_name,
        "train_wall_seconds": train_wall_seconds,
        "selected_segment": best_seg,
        "selected_timesteps": best_seg * SEGMENT_TIMESTEPS,
        "selected_checkpoint_path": str(best_dst),
        "dev_trajectory": dev_checkpoints,
        "evaluation": full_eval,
    }
    atomic_write_json(run_result_path, run_result)
    return run_result


def _compute_forgetting(
    baseline_eval: dict[str, Any],
    run_eval: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suite_key in ("dev", "diagnostic_20", "untouched_test_30"):
        base_seeds = baseline_eval[suite_key]["per_seed"]
        run_seeds = run_eval[suite_key]["per_seed"]
        base_passed = sorted(int(s) for s, rec in base_seeds.items() if rec["success"])
        regressed = sorted(
            int(s) for s in base_passed if not run_seeds[str(s)]["success"]
        )
        gained = sorted(
            int(s)
            for s, rec in base_seeds.items()
            if not rec["success"] and run_seeds[str(s)]["success"]
        )
        out[suite_key] = {
            "baseline_passed_count": len(base_passed),
            "regressed_count": len(regressed),
            "regressed_seeds": regressed,
            "gained_count": len(gained),
            "gained_seeds": gained,
            "net_delta": len(gained) - len(regressed),
        }
    return out


def _agg_stats(values: list[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) >= 2 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _aggregate_arm(
    runs: list[dict[str, Any]],
    baseline_eval: dict[str, Any],
) -> dict[str, Any]:
    suites = ("dev", "diagnostic_20", "untouched_test_30", "stress_probe_15")
    aggregated: dict[str, Any] = {
        "train_wall_seconds": _agg_stats([r["train_wall_seconds"] for r in runs]),
        "selected_timesteps": _agg_stats([r["selected_timesteps"] for r in runs]),
        "suites": {},
        "forgetting": {},
    }

    for suite in suites:
        s_list = [r["evaluation"][suite] for r in runs]
        all_cats = sorted({cat for s in s_list for cat in s["per_category"]})
        cat_agg: dict[str, Any] = {}
        for cat in all_cats:
            cat_entries = [s["per_category"][cat] for s in s_list if cat in s["per_category"]]
            cat_agg[cat] = {
                "success_rate": _agg_stats([c["success_rate"] for c in cat_entries]),
                "average_sheep_penned": _agg_stats(
                    [c["average_sheep_penned"] for c in cat_entries]
                ),
                "average_steps": _agg_stats([c["average_steps"] for c in cat_entries]),
                "average_completion_steps_success_only": _agg_stats(
                    [c["average_completion_steps_success_only"] for c in cat_entries]
                ),
                "average_reward": _agg_stats([c["average_reward"] for c in cat_entries]),
                "average_no_progress_steps": _agg_stats(
                    [c["average_no_progress_steps"] for c in cat_entries]
                ),
                "no_progress_count": _agg_stats(
                    [c["no_progress_count"] for c in cat_entries]
                ),
                "oscillation_detected_count": _agg_stats(
                    [c["oscillation_detected_count"] for c in cat_entries]
                ),
                "corner_stuck_at_end_count": _agg_stats(
                    [c["corner_stuck_at_end_count"] for c in cat_entries]
                ),
            }

        aggregated["suites"][suite] = {
            "success_rate": _agg_stats([s["success_rate"] for s in s_list]),
            "timeout_rate": _agg_stats([s["timeout_rate"] for s in s_list]),
            "stopped_rate": _agg_stats([s["stopped_rate"] for s in s_list]),
            "average_sheep_penned": _agg_stats([s["average_sheep_penned"] for s in s_list]),
            "average_steps": _agg_stats([s["average_steps"] for s in s_list]),
            "average_completion_steps_success_only": _agg_stats(
                [s["average_completion_steps_success_only"] for s in s_list]
            ),
            "average_reward": _agg_stats([s["average_reward"] for s in s_list]),
            "average_no_progress_steps": _agg_stats(
                [s["average_no_progress_steps"] for s in s_list]
            ),
            "no_progress_count": _agg_stats(
                [s["failure_categories"]["no-progress"] for s in s_list]
            ),
            "oscillation_detected_failures": _agg_stats(
                [s["failure_categories"]["oscillation_detected_failures"] for s in s_list]
            ),
            "oscillation_detected_all": _agg_stats(
                [s["failure_categories"]["oscillation_detected_all"] for s in s_list]
            ),
            "corner_stuck_at_end_failures": _agg_stats(
                [s["failure_categories"]["corner_stuck_at_end_failures"] for s in s_list]
            ),
            "corner_stuck_at_end_all": _agg_stats(
                [s["failure_categories"]["corner_stuck_at_end_all"] for s in s_list]
            ),
            "per_category": cat_agg,
        }

    for suite_key in ("dev", "diagnostic_20", "untouched_test_30"):
        f_list = [_compute_forgetting(baseline_eval, r["evaluation"])[suite_key] for r in runs]
        aggregated["forgetting"][suite_key] = {
            "regressed_count": _agg_stats([f["regressed_count"] for f in f_list]),
            "gained_count": _agg_stats([f["gained_count"] for f in f_list]),
            "net_delta": _agg_stats([f["net_delta"] for f in f_list]),
        }

    return aggregated


def main() -> None:
    """Run the controlled multi-seed Stage 8 curriculum comparison."""
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    parent_sha = _sha256(PARENT_MODEL)
    parent_copy = EXPERIMENT_ROOT / "parent_ckpt613.zip"
    shutil.copy2(PARENT_MODEL, parent_copy)
    if _sha256(parent_copy) != parent_sha:
        raise RuntimeError("Parent model SHA-256 verification failed after copy.")

    base_payload = json.loads(EFFECTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    training_state = json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))
    adaptive_state = training_state.get("adaptive_step_state", {})
    effective_lr = float(adaptive_state.get("effective_learning_rate", 0.00015))
    effective_ent = float(adaptive_state.get("effective_entropy_coef", 0.005))
    policy_config = dict(training_state.get("policy_config", {}))

    overall_start = time.perf_counter()

    baseline_cache = EXPERIMENT_ROOT / "baseline_eval.json"
    if baseline_cache.exists():
        print(f"Reusing cached Baseline evaluation from {baseline_cache}")
        baseline_eval = json.loads(baseline_cache.read_text(encoding="utf-8"))
    else:
        print("Evaluating Baseline (Ckpt 613) on Dev-10, Diag-20, Untouched-30, StressProbe-15...")
        baseline_eval = _evaluate_full_suite(
            parent_copy,
            base_payload,
            policy_config,
            EXPERIMENT_ROOT / "baseline_eval",
            "baseline",
        )
        atomic_write_json(baseline_cache, baseline_eval)
    print(
        f"Baseline done in {baseline_eval['eval_wall_seconds']:.1f}s | "
        f"dev={baseline_eval['dev']['success_rate']:.2f}, "
        f"diag={baseline_eval['diagnostic_20']['success_rate']:.2f}, "
        f"test={baseline_eval['untouched_test_30']['success_rate']:.3f}, "
        f"probe={baseline_eval['stress_probe_15']['success_rate']:.3f}"
    )

    runs: dict[str, list[dict[str, Any]]] = {"control": [], "treatment": []}
    for train_seed in TRAIN_SEEDS:
        for arm_name, spawn_mix in (
            ("control", CONTROL_SPAWN_MIX),
            ("treatment", TREATMENT_SPAWN_MIX),
        ):
            print(
                f"Running seed={train_seed}, arm={arm_name} "
                f"({TOTAL_TIMESTEPS} timesteps, {SEGMENTS} segments)..."
            )
            run_result = _train_and_select_arm(
                arm_name,
                train_seed,
                parent_copy,
                base_payload,
                policy_config,
                spawn_mix,
                effective_lr,
                effective_ent,
            )
            run_result["forgetting_vs_baseline"] = _compute_forgetting(
                baseline_eval, run_result["evaluation"]
            )
            runs[arm_name].append(run_result)

    aggregated = {
        "control": _aggregate_arm(runs["control"], baseline_eval),
        "treatment": _aggregate_arm(runs["treatment"], baseline_eval),
    }

    total_wall = time.perf_counter() - overall_start
    results = {
        "experiment": "stage8_curriculum_multiseed_v1",
        "parent_model": str(PARENT_MODEL),
        "parent_sha256": parent_sha,
        "parent_unchanged": _sha256(PARENT_MODEL) == parent_sha,
        "train_seeds": list(TRAIN_SEEDS),
        "segments": SEGMENTS,
        "segment_timesteps": SEGMENT_TIMESTEPS,
        "total_timesteps_per_arm_per_seed": TOTAL_TIMESTEPS,
        "dev_eval_segments": list(DEV_EVAL_SEGMENTS),
        "effective_learning_rate": effective_lr,
        "effective_entropy_coef": effective_ent,
        "ppo_env_workers": int(base_payload["training"]["ppo_env_workers"]),
        "dev_seeds": list(DEV_SEEDS),
        "diagnostic_20_seeds": list(DIAG_SEEDS),
        "untouched_test_30_seeds": list(UNTOUCHED_TEST_SEEDS),
        "stress_probe_specs": {k: list(v) for k, v in STRESS_PROBE_SPECS},
        "control_spawn_mix": CONTROL_SPAWN_MIX,
        "treatment_spawn_mix": TREATMENT_SPAWN_MIX,
        "wall_clock_total_seconds": total_wall,
        "baseline": baseline_eval,
        "runs": runs,
        "aggregated": aggregated,
    }
    output_path = EXPERIMENT_ROOT / "multiseed_results.json"
    atomic_write_json(output_path, results)
    print(f"Saved multi-seed results to {output_path} (total wall={total_wall:.1f}s)")


if __name__ == "__main__":
    main()
