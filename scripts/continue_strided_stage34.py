"""Bounded, isolated Stage 34 continuation test on 232-input strided checkpoints.

Continues ``arm_a_control_zero_gate/seed_101`` and ``seed_202`` from
``artifacts/experiments/strided_gate_run_20261117_085600`` for 3 x 65,536
transitions (196,608 total per seed) while preserving their populated Adam
optimizer state, under ``ENV_CONFIG_VERSION="1.1"`` and 50% failure-directed
starts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.environment import ENV_CONFIG_VERSION
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.gate_information_preparation import DEV_SEEDS
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    compute_observation_schema_version,
)
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions, sha256_file
from sheepdog.tracking.registry import capture_current_git_provenance
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter

PROTOCOL = "stage34_strided_gate_continuation_v1"
DEFAULT_SOURCE_DIR = Path("artifacts/experiments/strided_gate_run_20261117_085600")
DEFAULT_ARM = "arm_a_control_zero_gate"
DEFAULT_SEEDS: tuple[int, ...] = (101, 202)
DEFAULT_SEGMENTS = 3
DEFAULT_SEGMENT_TRANSITIONS = 65_536
DEFAULT_MAX_SEED_SECONDS = 500.0
DEFAULT_MAX_TOTAL_SECONDS = 1100.0
DEFAULT_FAILURE_TARGET_RATIO = 0.5
DEFAULT_FAILURE_WEIGHTS: dict[str, float] = {
    "boundary_split_flock": 1.0,
    "wall_stray_extraction": 1.0,
    "high_pen_standoff": 1.0,
    "gate_wall_flock": 1.0,
}

EXPECTED_SOURCE_MODEL_SHA256: dict[int, str] = {
    101: "1b53d865487eecd4631c0f332daa932389e7c666567b2d0ed4009a5232178b9c",
    202: "5471cf401ea652bddb37c639a37b150d1115ad11aac542bd205dc6c67efd4454",
}


class _RolloutBudgetCallback(BaseCallback):
    """Track transitions and world steps and enforce wall-clock deadline."""

    def __init__(self, deadline: float, transition_limit: int) -> None:
        super().__init__()
        self.deadline = deadline
        self.transition_limit = transition_limit
        self.transitions = 0
        self.world_steps = 0
        self.episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.transitions += len(infos)
        for info in infos:
            if info.get("team_step_completed"):
                self.world_steps += 1
            if "episode" in info:
                self.episodes.append(info["episode"])
        return self.transitions < self.transition_limit and time.monotonic() < self.deadline


def _find_baseline_eval_json(source_seed_dir: Path) -> Path:
    eval_dir = source_seed_dir / "evaluation"
    matches = sorted(
        p for p in eval_dir.glob("eval_*.json") if p.name != "eval_retention_index.json"
    )
    if not matches:
        raise FileNotFoundError(f"No baseline evaluation JSON found in {eval_dir}")
    return matches[0]


def _format_eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(r["seed"]),
            "success": bool(r["success"]),
            "sheep_penned": int(r["sheep_penned"]),
            "steps": int(r["steps"]),
            "stop_reason": str(r["stop_reason"]),
            "reward": float(r["reward_total"]),
            "replay_path": r.get("replay_path"),
        }
        for r in records
    ]


def _run_eval(
    model: MaskablePPO,
    model_path: Path,
    cfg: LabConfig,
    policy_cfg: NeuralPolicyConfig,
    *,
    eval_dir: Path,
    eval_seeds: tuple[int, ...],
    run_id: str,
    checkpoint_id: str,
) -> dict[str, Any]:
    policy = NeuralPolicy(model, model_path, policy_cfg)
    eval_summary, _, _ = Evaluator(cfg, eval_dir).evaluate(
        policy,
        eval_seeds,
        checkpoint_episode=0,
        deterministic=True,
        capture_replays=True,
        capture_step_telemetry=False,
        evaluation_mode="development_strided_gate_cont_v1",
        curriculum_stage=cfg.environment.curriculum_stage,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
    )
    per_seed = [
        {
            "seed": int(r.seed),
            "success": bool(r.success),
            "sheep_penned": int(r.sheep_penned),
            "steps": int(r.steps),
            "stop_reason": str(r.stop_reason),
            "reward": float(r.reward_total),
            "replay_path": r.replay_path,
        }
        for r in eval_summary.records
    ]
    return {
        "success_count": sum(1 for r in eval_summary.records if r.success),
        "episode_count": len(eval_summary.records),
        "success_rate": float(eval_summary.success_rate),
        "average_reward": float(eval_summary.average_reward),
        "per_seed": per_seed,
    }


def run_seed_continuation(
    *,
    source_dir: Path,
    arm: str,
    seed: int,
    output_seed_dir: Path,
    segments: int,
    segment_transitions: int,
    eval_seeds: tuple[int, ...],
    max_seed_seconds: float,
    global_deadline: float,
    run_id: str,
) -> dict[str, Any]:
    """Copy source checkpoint/config, verify Adam state, and run segmented continuation."""
    src_seed_dir = source_dir / arm / f"seed_{seed}"
    src_model = src_seed_dir / "final_model.zip"
    src_config = src_seed_dir / "config.json"
    src_eval_json = _find_baseline_eval_json(src_seed_dir)

    src_model_sha = sha256_file(src_model)
    expected_sha = EXPECTED_SOURCE_MODEL_SHA256.get(seed)
    if expected_sha is not None and src_model_sha != expected_sha:
        raise ValueError(
            f"Source model SHA-256 mismatch for seed {seed}: got {src_model_sha}, expected {expected_sha}"
        )
    src_config_sha = sha256_file(src_config)

    output_seed_dir.mkdir(parents=True, exist_ok=True)
    copied_model = output_seed_dir / "baseline_model.zip"
    copied_config = output_seed_dir / "config.json"
    copied_eval = output_seed_dir / "baseline_eval.json"
    shutil.copy2(src_model, copied_model)
    shutil.copy2(src_config, copied_config)
    shutil.copy2(src_eval_json, copied_eval)

    if sha256_file(copied_model) != src_model_sha:
        raise ValueError(f"Copied model SHA-256 mismatch for seed {seed}")
    if sha256_file(copied_config) != src_config_sha:
        raise ValueError(f"Copied config SHA-256 mismatch for seed {seed}")

    cfg_dict = json.loads(copied_config.read_text(encoding="utf-8"))
    total_budget = int(segments * segment_transitions)
    cfg_dict["training"]["total_timesteps"] = total_budget
    cfg_dict["training"]["failure_directed_target_ratio"] = float(DEFAULT_FAILURE_TARGET_RATIO)
    cfg_dict["training"]["output_dir"] = str(output_seed_dir / "training_output")
    cfg_dict["training"]["backup_enabled"] = False
    cfg_dict["training"]["backup_dir"] = str(output_seed_dir / "backups")
    cfg_dict["training"]["web_export_dir"] = str(output_seed_dir / "web")
    cfg_dict["training"]["wandb_enabled"] = False
    cfg = LabConfig.from_dict(cfg_dict)

    baseline_payload = json.loads(copied_eval.read_text(encoding="utf-8"))
    baseline_records = _format_eval_records(baseline_payload["records"])
    baseline_success_count = sum(1 for r in baseline_records if r["success"])
    baseline_avg_reward = (
        sum(r["reward"] for r in baseline_records) / len(baseline_records)
        if baseline_records
        else 0.0
    )

    checkpoints: list[dict[str, Any]] = [
        {
            "transitions": 0,
            "world_steps": 0,
            "source": "copied_baseline",
            "model_path": str(copied_model),
            "model_sha256": src_model_sha,
            "success_count": baseline_success_count,
            "episode_count": len(baseline_records),
            "success_rate": baseline_success_count / len(baseline_records),
            "average_reward": baseline_avg_reward,
            "per_seed": baseline_records,
        }
    ]

    t0 = time.monotonic()
    seed_deadline = min(t0 + float(max_seed_seconds), global_deadline)
    set_random_seed(int(seed))
    venv = DummyVecEnv([lambda c=cfg: SheepdogRLAdapter(c)])
    total_transitions = 0
    total_world_steps = 0
    total_episodes = 0
    stop_reason = "completed_budget"
    initial_optimizer_state_len = 0
    sampling_usage: list[dict[str, Any]] = []

    try:
        venv.env_method("update_failure_weights", dict(DEFAULT_FAILURE_WEIGHTS))
        model = MaskablePPO.load(str(copied_model), env=venv, device="cpu")
        initial_optimizer_state_len = len(model.policy.optimizer.state)
        if initial_optimizer_state_len == 0:
            raise AssertionError(
                f"Expected non-empty Adam optimizer state for seed {seed}, got 0"
            )
        if (
            model.n_envs != 1
            or segment_transitions <= 0
            or segment_transitions % model.n_steps != 0
        ):
            raise ValueError(
                f"segment_transitions ({segment_transitions}) must be a positive multiple of n_steps ({model.n_steps})"
            )

        model.tensorboard_log = None
        model.verbose = 0
        model.set_random_seed(int(seed))

        policy_cfg = NeuralPolicyConfig(
            hidden_sizes=tuple(int(x) for x in cfg.training.neural_hidden_sizes),
            observation_size=int(model.observation_space.shape[0]),
            env_workers=1,
            observation_mode=str(cfg.training.observation_mode),
            observation_schema_version=compute_observation_schema_version(
                str(cfg.training.observation_mode),
                tuple(cfg.training.observation_history_strides),
                bool(cfg.training.include_loop_flags_in_observation),
            ),
            observation_history_strides=tuple(cfg.training.observation_history_strides),
            include_loop_flags_in_observation=bool(
                cfg.training.include_loop_flags_in_observation
            ),
        )

        milestones_dir = output_seed_dir / "milestones"
        milestones_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx in range(1, segments + 1):
            if time.monotonic() >= seed_deadline:
                stop_reason = "seed_or_global_time_budget_exhausted"
                break

            callback = _RolloutBudgetCallback(
                deadline=seed_deadline,
                transition_limit=int(segment_transitions),
            )
            model.learn(
                total_timesteps=int(segment_transitions),
                reset_num_timesteps=False,
                callback=callback,
            )
            total_transitions += int(callback.transitions)
            total_world_steps += int(callback.world_steps)
            total_episodes += len(callback.episodes)

            milestone_path = milestones_dir / f"model_seg{seg_idx}_ts_{total_transitions}.zip"
            model.save(str(milestone_path))
            eval_result = _run_eval(
                model,
                milestone_path,
                cfg,
                policy_cfg,
                eval_dir=output_seed_dir / "evaluation" / f"ts_{total_transitions}",
                eval_seeds=eval_seeds,
                run_id=run_id,
                checkpoint_id=f"cont_s{seed}_ts{total_transitions}",
            )
            checkpoints.append(
                {
                    "segment": seg_idx,
                    "transitions": total_transitions,
                    "segment_transitions": int(callback.transitions),
                    "world_steps": total_world_steps,
                    "elapsed_seconds": round(time.monotonic() - t0, 3),
                    "optimizer_state_len": len(model.policy.optimizer.state),
                    "source": "segment_milestone",
                    "model_path": str(milestone_path),
                    "model_sha256": sha256_file(milestone_path),
                    **eval_result,
                }
            )
            if callback.transitions < int(segment_transitions) or time.monotonic() >= seed_deadline:
                stop_reason = "seed_or_global_time_budget_exhausted"
                break

        sampling_usage = venv.env_method("get_scenario_sampler_summary")
    finally:
        venv.close()

    return {
        "train_seed": int(seed),
        "arm": arm,
        "workspace": str(output_seed_dir),
        "source_model_sha256": src_model_sha,
        "source_config_sha256": src_config_sha,
        "optimizer_state_preserved": True,
        "initial_optimizer_state_len": initial_optimizer_state_len,
        "stop_reason": stop_reason,
        "transitions_completed": total_transitions,
        "transitions_budget": total_budget,
        "world_steps_completed": total_world_steps,
        "training_episodes_completed": total_episodes,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "sampling_usage": sampling_usage,
        "checkpoints": checkpoints,
    }


def run_continuation(
    output_dir: Path,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    arm: str = DEFAULT_ARM,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    segments: int = DEFAULT_SEGMENTS,
    segment_transitions: int = DEFAULT_SEGMENT_TRANSITIONS,
    max_seed_seconds: float = DEFAULT_MAX_SEED_SECONDS,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
) -> dict[str, Any]:
    """Run the isolated Stage 34 continuation experiment and write manifest.json."""
    if ENV_CONFIG_VERSION != "1.1":
        raise AssertionError(f"Expected ENV_CONFIG_VERSION=='1.1', got {ENV_CONFIG_VERSION!r}")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))

    t_start = time.monotonic()
    global_deadline = t_start + float(max_total_seconds)
    git_revision, git_dirty, dirty_patch_hash = capture_current_git_provenance()

    seed_results: dict[str, Any] = {}
    for seed in seeds:
        if time.monotonic() >= global_deadline:
            seed_results[f"seed_{seed}"] = {
                "train_seed": int(seed),
                "stop_reason": "skipped_global_time_budget_exhausted",
                "transitions_completed": 0,
            }
            continue
        seed_results[f"seed_{seed}"] = run_seed_continuation(
            source_dir=source_dir,
            arm=arm,
            seed=int(seed),
            output_seed_dir=output_dir / f"seed_{seed}",
            segments=int(segments),
            segment_transitions=int(segment_transitions),
            eval_seeds=tuple(DEV_SEEDS),
            max_seed_seconds=float(max_seed_seconds),
            global_deadline=global_deadline,
            run_id=output_dir.name,
        )

    manifest = {
        "protocol": PROTOCOL,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": git_revision,
        "git_dirty": bool(git_dirty),
        "dirty_patch_hash": dirty_patch_hash,
        "dependencies": collect_dependency_versions(),
        "env_config_version": ENV_CONFIG_VERSION,
        "assistance_mode": "guided",
        "assistance_disclosure": (
            "Guided/assisted training and evaluation: observation_mode='guided_gate_control' "
            "uses TeamStrategy role/target guidance, loop flags, and simulation gate-surge physics."
        ),
        "source_experiment_dir": str(source_dir),
        "source_arm": arm,
        "failure_directed_weights": dict(DEFAULT_FAILURE_WEIGHTS),
        "failure_directed_target_ratio": float(DEFAULT_FAILURE_TARGET_RATIO),
        "eval_seeds": [int(s) for s in DEV_SEEDS],
        "budget": {
            "seeds": [int(s) for s in seeds],
            "segments": int(segments),
            "segment_transitions": int(segment_transitions),
            "transitions_per_seed": int(segments * segment_transitions),
            "max_seed_seconds": float(max_seed_seconds),
            "max_total_seconds": float(max_total_seconds),
        },
        "elapsed_seconds": round(time.monotonic() - t_start, 3),
        "seed_results": seed_results,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parse_int_list(raw: str) -> tuple[int, ...]:
    items = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    return tuple(items)


def main() -> None:
    """CLI entrypoint for bounded Stage 34 strided continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = (
        Path("artifacts/experiments")
        / f"strided_gate_cont_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--arm", type=str, default=DEFAULT_ARM)
    parser.add_argument("--seeds", type=str, default="101,202")
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--segment-transitions", type=int, default=DEFAULT_SEGMENT_TRANSITIONS)
    parser.add_argument("--max-seed-seconds", type=float, default=DEFAULT_MAX_SEED_SECONDS)
    parser.add_argument("--max-total-seconds", type=float, default=DEFAULT_MAX_TOTAL_SECONDS)
    args = parser.parse_args()

    manifest = run_continuation(
        args.output_dir,
        source_dir=args.source_dir,
        arm=args.arm,
        seeds=_parse_int_list(args.seeds),
        segments=args.segments,
        segment_transitions=args.segment_transitions,
        max_seed_seconds=args.max_seed_seconds,
        max_total_seconds=args.max_total_seconds,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "seeds": {
                    k: [
                        {
                            "transitions": c["transitions"],
                            "success_count": c["success_count"],
                            "episode_count": c["episode_count"],
                            "average_reward": round(c["average_reward"], 2),
                        }
                        for c in v.get("checkpoints", [])
                    ]
                    for k, v in manifest["seed_results"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
