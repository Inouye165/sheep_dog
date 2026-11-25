"""Bounded, isolated clean-sheet from-scratch pilot at Stage 1 with raw emergent_gate observations.

Uses:
- ``observation_mode="emergent_gate"``
- ``observation_history_strides=(0, 4, 8, 16)``
- ``include_loop_flags_in_observation=False``
- ``enable_instinct_rewards=False``
- ``52 * 4 = 208`` inputs, zero ``TeamStrategy`` guidance, zero legacy loop flags.
"""

from __future__ import annotations

import argparse
import json
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
from sheepdog.curriculum import CURRICULUM_STAGES, apply_training_profile
from sheepdog.environment import ENV_CONFIG_VERSION
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.gate_information_preparation import DEV_SEEDS
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    compute_observation_schema_version,
)
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions, sha256_file
from sheepdog.server import _get_success_threshold
from sheepdog.tracking.registry import capture_current_git_provenance
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter

PROTOCOL = "clean_sheet_emergent_gate_pilot_v1"
OBS_MODE = "emergent_gate"
STRIDES: tuple[int, ...] = (0, 4, 8, 16)
INCLUDE_LOOP_FLAGS = False
HIDDEN_SIZES: tuple[int, ...] = (256, 256, 128)
EXPECTED_OBS_DIM = 208
DEFAULT_SEEDS: tuple[int, ...] = (101, 202)
DEFAULT_TRANSITIONS_PER_SEED = 65_536
DEFAULT_EVAL_INTERVAL = 8_192
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ENABLE_INSTINCT_REWARDS = False
DEFAULT_MAX_SEED_SECONDS = 300.0
DEFAULT_MAX_TOTAL_SECONDS = 600.0


def build_clean_sheet_stage_config(
    stage: int,
    *,
    ws: Path,
    train_seed: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    enable_instinct_rewards: bool = DEFAULT_ENABLE_INSTINCT_REWARDS,
) -> LabConfig:
    """Build an isolated LabConfig for *stage* with raw emergent_gate (208-dim) settings."""
    profiled = apply_training_profile(
        LabConfig(),
        enable_instinct_rewards=bool(enable_instinct_rewards),
        curriculum_stage=int(stage),
    )
    payload = profiled.to_dict()
    training = payload["training"]
    training["trainer_type"] = "maskable_ppo"
    training["policy_type"] = "neural"
    training["observation_mode"] = OBS_MODE
    training["observation_history_strides"] = list(STRIDES)
    training["include_loop_flags_in_observation"] = INCLUDE_LOOP_FLAGS
    training["neural_hidden_sizes"] = list(HIDDEN_SIZES)
    training["learning_rate"] = float(learning_rate)
    training["ppo_env_workers"] = 1
    training["output_dir"] = str(ws)
    training["backup_enabled"] = False
    training["backup_dir"] = str(ws / "backups")
    training["web_export_dir"] = str(ws / "web_export")
    training["train_seed"] = int(train_seed)
    return LabConfig.from_dict(payload)


class _RolloutBudgetCallback(BaseCallback):
    """Track transitions/world steps and stop cleanly at deadline or transition limit."""

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


def _save_milestone(
    model: MaskablePPO,
    policy_cfg: NeuralPolicyConfig,
    model_path: Path,
    *,
    stage: int,
    transitions: int,
    eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a milestone model ZIP, sidecar metadata JSON, and SHA-256."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    sidecar_path = model_path.with_suffix(".sidecar.json")
    sidecar_payload = {
        "policy_config": policy_cfg.to_dict(),
        "curriculum_stage": int(stage),
        "transitions": int(transitions),
        "env_config_version": ENV_CONFIG_VERSION,
        "eval_summary": eval_summary,
    }
    atomic_write_json(sidecar_path, sidecar_payload)
    return {
        "stage": int(stage),
        "transitions": int(transitions),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "eval_summary": eval_summary,
    }


def _evaluate_stage(
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
    """Run deterministic evaluation on *eval_seeds* without step telemetry."""
    policy = NeuralPolicy(model, model_path, policy_cfg)
    summary, _, _ = Evaluator(cfg, eval_dir).evaluate(
        policy,
        eval_seeds,
        checkpoint_episode=0,
        deterministic=True,
        capture_replays=False,
        capture_step_telemetry=False,
        evaluation_mode="development_clean_sheet_emergent_v1",
        curriculum_stage=cfg.environment.curriculum_stage,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
    )
    threshold = float(_get_success_threshold(cfg.environment.curriculum_stage))
    per_seed = [
        {
            "seed": int(rec.seed),
            "success": bool(rec.success),
            "sheep_penned": int(rec.sheep_penned),
            "steps": int(rec.steps),
            "stop_reason": rec.stop_reason,
            "reward": float(rec.reward_total),
        }
        for rec in summary.records
    ]
    success_count = sum(1 for rec in summary.records if rec.success)
    episode_count = len(summary.records)
    avg_steps = (
        sum(float(rec.steps) for rec in summary.records) / episode_count
        if episode_count > 0
        else 0.0
    )
    return {
        "stage": int(cfg.environment.curriculum_stage),
        "dogs": int(cfg.environment.dogs),
        "sheep": int(cfg.environment.sheep),
        "success_threshold": threshold,
        "success_rate": float(summary.success_rate),
        "success_count": int(success_count),
        "episode_count": int(episode_count),
        "avg_reward": float(summary.average_reward),
        "avg_steps": float(avg_steps),
        "promotion_qualified": bool(summary.success_rate >= threshold),
        "per_seed": per_seed,
    }


def run_seed_pilot(
    ws: Path,
    *,
    train_seed: int,
    transitions_budget: int,
    eval_interval: int,
    eval_seeds: tuple[int, ...],
    max_seed_seconds: float,
    global_deadline: float,
    run_id: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    enable_instinct_rewards: bool = DEFAULT_ENABLE_INSTINCT_REWARDS,
) -> dict[str, Any]:
    """Train one clean-sheet seed from Stage 1 with stage-promotion evaluations."""
    ws.mkdir(parents=True, exist_ok=True)
    milestones_dir = ws / "milestones"
    eval_dir = ws / "evaluation"
    milestones_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    seed_deadline = min(t0 + float(max_seed_seconds), global_deadline)
    set_random_seed(int(train_seed))

    stage = 1
    cfg = build_clean_sheet_stage_config(
        stage,
        ws=ws,
        train_seed=train_seed,
        learning_rate=learning_rate,
        enable_instinct_rewards=enable_instinct_rewards,
    )
    atomic_write_json(ws / "stage_01_config.json", cfg.to_dict())

    schema_version = compute_observation_schema_version(OBS_MODE, STRIDES, INCLUDE_LOOP_FLAGS)
    policy_cfg = NeuralPolicyConfig(
        hidden_sizes=HIDDEN_SIZES,
        observation_size=EXPECTED_OBS_DIM,
        env_workers=1,
        observation_mode=OBS_MODE,
        observation_schema_version=schema_version,
        observation_history_strides=STRIDES,
        include_loop_flags_in_observation=INCLUDE_LOOP_FLAGS,
    )

    venv: DummyVecEnv | None = DummyVecEnv([lambda c=cfg: SheepdogRLAdapter(c)])
    milestones: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    total_transitions = 0
    total_world_steps = 0
    total_episodes = 0
    stop_reason = "budget_completed"

    try:
        obs_shape = getattr(venv.observation_space, "shape", None)
        if obs_shape != (EXPECTED_OBS_DIM,):
            raise RuntimeError(f"Expected observation shape ({EXPECTED_OBS_DIM},), got {obs_shape}")

        model = MaskablePPO(
            "MlpPolicy",
            venv,
            learning_rate=cfg.training.learning_rate,
            n_steps=cfg.training.rollout_steps,
            batch_size=cfg.training.batch_size,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
            clip_range=cfg.training.clip_range,
            ent_coef=cfg.training.entropy_coef,
            vf_coef=cfg.training.value_coef,
            target_kl=getattr(cfg.training, "target_kl", None),
            seed=int(train_seed),
            policy_kwargs={"net_arch": list(HIDDEN_SIZES)},
            verbose=0,
            tensorboard_log=None,
            device="cpu",
        )

        init_path = milestones_dir / "stage_01_init_ts_0.zip"
        milestones.append(
            _save_milestone(model, policy_cfg, init_path, stage=stage, transitions=0)
        )

        while total_transitions < transitions_budget:
            if time.monotonic() >= seed_deadline:
                stop_reason = "time_budget_reached"
                break

            remaining = transitions_budget - total_transitions
            segment_steps = min(int(eval_interval), int(remaining))
            cb = _RolloutBudgetCallback(seed_deadline, segment_steps)
            model.learn(
                total_timesteps=segment_steps,
                reset_num_timesteps=False,
                callback=cb,
            )
            total_transitions += cb.transitions
            total_world_steps += cb.world_steps
            total_episodes += len(cb.episodes)

            ckpt_path = milestones_dir / f"stage_{stage:02d}_ts_{total_transitions}.zip"
            model.save(str(ckpt_path))
            eval_info = _evaluate_stage(
                model,
                ckpt_path,
                cfg,
                policy_cfg,
                eval_dir=eval_dir / f"stage_{stage:02d}_ts_{total_transitions}",
                eval_seeds=eval_seeds,
                run_id=run_id,
                checkpoint_id=f"seed{train_seed}_s{stage}_ts{total_transitions}",
            )
            eval_info["transitions"] = int(total_transitions)
            eval_info["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            evaluations.append(eval_info)

            if eval_info["promotion_qualified"] and (stage + 1) in CURRICULUM_STAGES:
                promoted_path = milestones_dir / f"stage_{stage:02d}_promoted_ts_{total_transitions}.zip"
                milestones.append(
                    _save_milestone(
                        model,
                        policy_cfg,
                        promoted_path,
                        stage=stage,
                        transitions=total_transitions,
                        eval_summary=eval_info,
                    )
                )
                next_stage = stage + 1
                promotions.append(
                    {
                        "from_stage": stage,
                        "to_stage": next_stage,
                        "transitions": int(total_transitions),
                        "elapsed_seconds": round(time.monotonic() - t0, 3),
                        "eval_success_rate": eval_info["success_rate"],
                        "eval_success_threshold": eval_info["success_threshold"],
                        "checkpoint_path": str(promoted_path),
                    }
                )
                stage = next_stage
                cfg = build_clean_sheet_stage_config(
                    stage,
                    ws=ws,
                    train_seed=train_seed,
                    learning_rate=learning_rate,
                    enable_instinct_rewards=enable_instinct_rewards,
                )
                atomic_write_json(ws / f"stage_{stage:02d}_config.json", cfg.to_dict())
                venv.close()
                venv = DummyVecEnv([lambda c=cfg: SheepdogRLAdapter(c)])
                model.set_env(venv)

        final_path = milestones_dir / f"final_stage_{stage:02d}_ts_{total_transitions}.zip"
        final_eval = (
            evaluations[-1]
            if evaluations and evaluations[-1]["stage"] == stage and evaluations[-1]["transitions"] == total_transitions
            else _evaluate_stage(
                model,
                final_path,
                cfg,
                policy_cfg,
                eval_dir=eval_dir / f"final_stage_{stage:02d}_ts_{total_transitions}",
                eval_seeds=eval_seeds,
                run_id=run_id,
                checkpoint_id=f"seed{train_seed}_final_s{stage}_ts{total_transitions}",
            )
        )
        milestones.append(
            _save_milestone(
                model,
                policy_cfg,
                final_path,
                stage=stage,
                transitions=total_transitions,
                eval_summary=final_eval,
            )
        )
    finally:
        if venv is not None:
            venv.close()

    return {
        "train_seed": int(train_seed),
        "workspace": str(ws),
        "stop_reason": stop_reason,
        "starting_stage": 1,
        "final_stage": int(stage),
        "transitions_completed": int(total_transitions),
        "transitions_budget": int(transitions_budget),
        "world_steps_completed": int(total_world_steps),
        "training_episodes_completed": int(total_episodes),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "stage_promotions": promotions,
        "milestones": milestones,
        "evaluations": evaluations,
        "final_evaluation": final_eval,
    }


def run_pilot(
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    transitions_per_seed: int = DEFAULT_TRANSITIONS_PER_SEED,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    eval_seeds: tuple[int, ...] = tuple(DEV_SEEDS),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    enable_instinct_rewards: bool = DEFAULT_ENABLE_INSTINCT_REWARDS,
    max_seed_seconds: float = DEFAULT_MAX_SEED_SECONDS,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
) -> dict[str, Any]:
    """Run the bounded clean-sheet emergent_gate pilot and write manifest.json."""
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))

    t_start = time.monotonic()
    global_deadline = t_start + float(max_total_seconds)
    git_revision, git_dirty, dirty_patch_hash = capture_current_git_provenance()
    sample_cfg = build_clean_sheet_stage_config(
        1,
        ws=output_dir / "_sample",
        train_seed=seeds[0],
        learning_rate=learning_rate,
        enable_instinct_rewards=enable_instinct_rewards,
    )

    seed_results: dict[str, Any] = {}
    for seed in seeds:
        if time.monotonic() >= global_deadline:
            seed_results[f"seed_{seed}"] = {
                "train_seed": int(seed),
                "stop_reason": "skipped_global_time_budget_exhausted",
                "transitions_completed": 0,
            }
            continue
        seed_results[f"seed_{seed}"] = run_seed_pilot(
            output_dir / f"seed_{seed}",
            train_seed=int(seed),
            transitions_budget=int(transitions_per_seed),
            eval_interval=int(eval_interval),
            eval_seeds=eval_seeds,
            max_seed_seconds=float(max_seed_seconds),
            global_deadline=global_deadline,
            run_id=output_dir.name,
            learning_rate=float(learning_rate),
            enable_instinct_rewards=bool(enable_instinct_rewards),
        )

    manifest = {
        "protocol": PROTOCOL,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": git_revision,
        "git_dirty": bool(git_dirty),
        "dirty_patch_hash": dirty_patch_hash,
        "dependencies": collect_dependency_versions(),
        "env_config_version": ENV_CONFIG_VERSION,
        "observation_mode": OBS_MODE,
        "observation_history_strides": list(STRIDES),
        "include_loop_flags_in_observation": INCLUDE_LOOP_FLAGS,
        "observation_size": EXPECTED_OBS_DIM,
        "observation_schema_version": compute_observation_schema_version(
            OBS_MODE, STRIDES, INCLUDE_LOOP_FLAGS
        ),
        "learning_rate": float(learning_rate),
        "enable_instinct_rewards": bool(enable_instinct_rewards),
        "assistance_mode": "autonomous_emergent_gate",
        "assistance_disclosure": {
            "policy_guidance": (
                "none (emergent_gate raw observations, zero TeamStrategy role/target inputs, "
                f"zero legacy loop flags, enable_instinct_rewards={bool(enable_instinct_rewards)})"
            ),
            "physics_assistance": {
                "sheep_surge_multiplier": float(sample_cfg.environment.sheep_surge_multiplier),
                "sheep_d_gate": float(sample_cfg.environment.sheep_d_gate),
                "note": "EnvironmentConfig sheep gate-surge physics remain active in simulation physics.",
            },
        },
        "stage_1_config": sample_cfg.to_dict(),
        "budget": {
            "seeds": [int(s) for s in seeds],
            "transitions_per_seed": int(transitions_per_seed),
            "eval_interval": int(eval_interval),
            "eval_seeds": [int(s) for s in eval_seeds],
            "learning_rate": float(learning_rate),
            "enable_instinct_rewards": bool(enable_instinct_rewards),
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
    """Parse CLI arguments and run the bounded clean-sheet emergent_gate pilot."""
    parser = argparse.ArgumentParser(description="Run clean-sheet Stage 1 emergent_gate pilot.")
    default_dir = Path("artifacts/experiments") / f"clean_sheet_emergent_gate_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--seeds", type=str, default="101,202")
    parser.add_argument(
        "--transitions",
        "--transitions-per-seed",
        dest="transitions_per_seed",
        type=int,
        default=DEFAULT_TRANSITIONS_PER_SEED,
    )
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--enable-instinct-rewards",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_INSTINCT_REWARDS,
    )
    parser.add_argument("--max-seed-seconds", type=float, default=DEFAULT_MAX_SEED_SECONDS)
    parser.add_argument("--max-total-seconds", type=float, default=DEFAULT_MAX_TOTAL_SECONDS)
    args = parser.parse_args()

    manifest = run_pilot(
        args.output_dir,
        seeds=_parse_int_list(args.seeds),
        transitions_per_seed=args.transitions_per_seed,
        eval_interval=args.eval_interval,
        learning_rate=args.lr,
        enable_instinct_rewards=args.enable_instinct_rewards,
        max_seed_seconds=args.max_seed_seconds,
        max_total_seconds=args.max_total_seconds,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
