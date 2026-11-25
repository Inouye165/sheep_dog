"""Prepare, verify, and optionally run strided (0, 4, 8, 16) gate-information validation.

Default parent is immutable pv_10483 (checkpoint 40897 / timestep 343787786).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.environment import ENV_CONFIG_VERSION, SheepdogEnvironment
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.gate_information_preparation import (
    ARCH,
    ARMS,
    DEV_SEEDS,
    SEEDS,
    arm_config,
    inspect_parent,
)
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    compute_observation_schema_version,
    expand_policy_observation_inputs,
)
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions, sha256_file
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter

PROTOCOL = "stage34_strided_gate_validation_v1"
STRIDES: tuple[int, ...] = (0, 4, 8, 16)
BASE_DIM_WITH_GATE = 58
STACKED_DIM = BASE_DIM_WITH_GATE * len(STRIDES)  # 232
DEFAULT_FAILURE_WEIGHTS: dict[str, float] = {
    "boundary_split_flock": 1.0,
    "wall_stray_extraction": 1.0,
    "high_pen_standoff": 1.0,
    "gate_wall_flock": 1.0,
}

DEFAULT_PARENT_MODEL = Path(
    "artifacts/experiments/stopped_adoption_20261113_171410/artifacts/checkpoints/stage34/maskable-ppo-stage34-343787786.zip"
)
DEFAULT_PARENT_METADATA = Path(
    "artifacts/experiments/stopped_adoption_20261113_171410/artifacts/checkpoints/checkpoint-040897.json"
)
DEFAULT_PARENT_SHA256 = "26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2"


def _strided_arm_config(
    meta: dict[str, Any],
    archive: dict[str, Any],
    workspace: Path,
    seed: int,
    mode: str,
) -> LabConfig:
    base_cfg = arm_config(meta, archive, workspace, seed, mode)
    payload = base_cfg.to_dict()
    payload["training"]["observation_history_strides"] = list(STRIDES)
    payload["training"]["include_loop_flags_in_observation"] = True
    return LabConfig.from_dict(payload)


def verify_numerical_logit_value_equivalence(
    source_path: Path,
    target_path: Path,
    *,
    old_dim: int = 54,
    new_dim: int = STACKED_DIM,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify logit and critic value equivalence on random 54-dim inputs zero-padded to *new_dim*."""
    src_model = MaskablePPO.load(str(source_path), device="cpu")
    dst_model = MaskablePPO.load(str(target_path), device="cpu")
    rng = np.random.default_rng(53)
    obs_54 = rng.uniform(-1.0, 1.0, size=(16, old_dim)).astype(np.float32)
    obs_new = np.zeros((16, new_dim), dtype=np.float32)
    obs_new[:, :old_dim] = obs_54

    with torch.no_grad():
        t_54 = torch.as_tensor(obs_54)
        t_new = torch.as_tensor(obs_new)
        dist_54 = src_model.policy.get_distribution(t_54)
        dist_new = dst_model.policy.get_distribution(t_new)
        val_54 = src_model.policy.predict_values(t_54)
        val_new = dst_model.policy.predict_values(t_new)
        max_abs_logit_diff = float(
            torch.max(torch.abs(dist_54.distribution.logits - dist_new.distribution.logits)).item()
        )
        max_abs_value_diff = float(torch.max(torch.abs(val_54 - val_new)).item())

    if max_abs_logit_diff > atol or max_abs_value_diff > atol:
        raise AssertionError(
            f"Numerical equivalence failed: logit_diff={max_abs_logit_diff}, value_diff={max_abs_value_diff}, atol={atol}"
        )
    return {
        "max_abs_logit_diff": max_abs_logit_diff,
        "max_abs_value_diff": max_abs_value_diff,
        "atol": atol,
    }


def verify_strided_20_step_equivalence(
    parent_path: Path,
    models: dict[str, Path],
    base_config: LabConfig,
) -> dict[str, Any]:
    """Verify exact parameter preservation, logit/value parity, and 20-step seed-53 rollout equivalence."""
    traces: dict[str, list[tuple[list[str], dict[str, Any], dict[str, Any]]]] = {}
    parent_tensors: dict[str, torch.Tensor] | None = None
    numerical_checks: dict[str, dict[str, Any]] = {}

    for label, path, mode, strides, obs_dim in [
        ("parent", parent_path, "guided", (0,), 54),
        *((arm, models[arm], mode, STRIDES, STACKED_DIM) for arm, mode in ARMS.items()),
    ]:
        payload = base_config.to_dict()
        payload["training"]["observation_mode"] = mode
        payload["training"]["observation_history_strides"] = list(strides)
        payload["training"]["include_loop_flags_in_observation"] = True
        cfg = LabConfig.from_dict(payload)
        env = SheepdogEnvironment(cfg)
        env.reset(seed=53)
        model = MaskablePPO.load(str(path), device="cpu")
        if label == "parent":
            parent_tensors = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        else:
            assert parent_tensors is not None
            for name, tensor in model.policy.state_dict().items():
                original = parent_tensors[name]
                if tensor.shape == original.shape:
                    preserved = torch.equal(tensor, original)
                elif (
                    tensor.ndim == 2
                    and original.ndim == 2
                    and tensor.shape == (original.shape[0], STACKED_DIM)
                    and original.shape[1] == 54
                ):
                    preserved = torch.equal(tensor[:, :54], original) and bool(
                        torch.all(tensor[:, 54:] == 0)
                    )
                else:
                    preserved = False
                if not preserved:
                    raise AssertionError(f"Parameter mismatch after expansion: {label}/{name}")

            # Verify numerical logit & value equivalence on zero-padded 232-dim inputs
            num_check = verify_numerical_logit_value_equivalence(
                parent_path,
                path,
                old_dim=54,
                new_dim=STACKED_DIM,
                atol=1e-5,
            )
            numerical_checks[label] = num_check

            # Verify NeuralPolicy.load succeeds cleanly with strict schema checks
            loaded_policy = NeuralPolicy.load(
                path,
                cfg,
                policy_config=NeuralPolicyConfig(
                    hidden_sizes=ARCH,
                    observation_size=STACKED_DIM,
                    env_workers=1,
                    observation_mode=mode,
                    observation_schema_version=compute_observation_schema_version(
                        mode, STRIDES, True
                    ),
                    observation_history_strides=STRIDES,
                    include_loop_flags_in_observation=True,
                ).to_dict(),
            )
            del loaded_policy

        policy = NeuralPolicy(
            model,
            path,
            NeuralPolicyConfig(
                hidden_sizes=ARCH,
                observation_size=obs_dim,
                observation_mode=mode,
                observation_schema_version=compute_observation_schema_version(mode, strides, True),
                observation_history_strides=strides,
                include_loop_flags_in_observation=True,
            ),
        )
        trace: list[tuple[list[str], dict[str, Any], dict[str, Any]]] = []
        for _ in range(20):
            obs0 = env.build_observation_for_dog(0)
            if len(obs0.values) != obs_dim:
                raise AssertionError(f"{label}: expected obs_dim={obs_dim}, got {len(obs0.values)}")
            actions = policy.select_actions(env, deterministic=True)
            snapshot, reward = env.step(actions)
            trace.append((list(actions), snapshot.to_dict(), reward.to_dict()))
            if snapshot.terminated:
                break
        traces[label] = trace

    if len(traces["parent"]) != 20 or any(t != traces["parent"] for t in traces.values()):
        raise AssertionError("Exact 20-step seed-53 rollout equivalence failed")

    return {
        "seed": 53,
        "world_steps": 20,
        "exact_actions_snapshots_rewards": True,
        "all_existing_parameters_preserved": True,
        "numerical_checks": numerical_checks,
        "parent_trace_sha256": hashlib.sha256(
            json.dumps(traces["parent"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def prepare(
    root: Path,
    parent: Path = DEFAULT_PARENT_MODEL,
    metadata: Path = DEFAULT_PARENT_METADATA,
    expected_sha256: str = DEFAULT_PARENT_SHA256,
) -> dict[str, Any]:
    """Prepare 232-input strided workspaces in *root* and verify exact equivalence; never train."""
    if root.exists():
        raise FileExistsError(f"Preparation requires a new directory; {root} already exists")
    meta, archive = inspect_parent(parent, metadata, expected_sha256)
    root.mkdir(parents=True)
    frozen = root / "parent" / parent.name
    frozen.parent.mkdir()
    shutil.copy2(parent, frozen)
    shutil.copy2(metadata, frozen.parent / "metadata.json")
    if sha256_file(frozen) != expected_sha256:
        raise ValueError("Copied parent SHA-256 mismatch")

    torch.set_num_threads(1)
    entries: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {}
    for seed in SEEDS:
        models: dict[str, Path] = {}
        sample_cfg: LabConfig | None = None
        for arm, mode in ARMS.items():
            ws = root / arm / f"seed_{seed}"
            ws.mkdir(parents=True)
            cfg = _strided_arm_config(meta, archive, ws, seed, mode)
            if sample_cfg is None:
                sample_cfg = cfg
            atomic_write_json(ws / "config.json", cfg.to_dict())
            model_path = ws / "initial_model.zip"
            conversion = expand_policy_observation_inputs(
                frozen,
                model_path,
                old_dim=54,
                new_dim=STACKED_DIM,
                learning_rate=float(archive["learning_rate"]),
                observation_mode=mode,
                observation_history_strides=STRIDES,
                include_loop_flags_in_observation=True,
            )
            models[arm] = model_path
            entries[f"{arm}/seed_{seed}"] = {
                "arm": arm,
                "seed": seed,
                "observation_mode": mode,
                "observation_history_strides": list(STRIDES),
                "observation_size": STACKED_DIM,
                "workspace": str(ws),
                "config_path": str(ws / "config.json"),
                "config_sha256": sha256_file(ws / "config.json"),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "conversion": conversion,
            }
        assert sample_cfg is not None
        checks[f"seed_{seed}"] = verify_strided_20_step_equivalence(frozen, models, sample_cfg)

    manifest = {
        "protocol": PROTOCOL,
        "status": "prepared_verified_not_run",
        "strides": list(STRIDES),
        "stacked_observation_size": STACKED_DIM,
        "parent": {
            "model_path": str(frozen),
            "metadata_path": str(frozen.parent / "metadata.json"),
            "sha256": expected_sha256,
        },
        "dependencies": collect_dependency_versions(),
        "entries": entries,
        "equivalence_checks": checks,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def verify(root: Path) -> dict[str, Any]:
    """Re-verify integrity, strict schema loading, and 20-step equivalence for a prepared *root*."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = Path(manifest["parent"]["model_path"])
    if sha256_file(frozen) != manifest["parent"]["sha256"]:
        raise ValueError("Parent SHA-256 mismatch in workspace")

    entries = manifest["entries"]
    for key, entry in entries.items():
        if sha256_file(Path(entry["config_path"])) != entry["config_sha256"]:
            raise ValueError(f"Config SHA-256 mismatch for {key}")
        if sha256_file(Path(entry["model_path"])) != entry["model_sha256"]:
            raise ValueError(f"Model SHA-256 mismatch for {key}")

    torch.set_num_threads(1)
    rechecks: dict[str, Any] = {}
    for seed in SEEDS:
        models = {arm: Path(entries[f"{arm}/seed_{seed}"]["model_path"]) for arm in ARMS}
        sample_cfg_dict = json.loads(
            Path(entries[f"arm_a_control_zero_gate/seed_{seed}"]["config_path"]).read_text(
                encoding="utf-8"
            )
        )
        rechecks[f"seed_{seed}"] = verify_strided_20_step_equivalence(
            frozen,
            models,
            LabConfig.from_dict(sample_cfg_dict),
        )
    return {
        "status": "verified",
        "protocol": manifest["protocol"],
        "stacked_observation_size": manifest["stacked_observation_size"],
        "seeds_verified": list(SEEDS),
        "rechecks": rechecks,
    }


class _RolloutBudgetCallback(BaseCallback):
    """Count executed transitions and world steps and enforce wall-clock deadline."""

    def __init__(self, deadline: float, transitions: int) -> None:
        super().__init__()
        self.deadline = deadline
        self.limit = transitions
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
        # Allow the final transition of an exact rollout multiple so SB3 optimizes it.
        return self.transitions <= self.limit and time.monotonic() < self.deadline


def _parse_str_list(values: list[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    items = [part.strip() for raw in values for part in raw.split(",") if part.strip()]
    return tuple(items) if items else None


def _parse_int_list(values: list[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    items = [int(part.strip()) for raw in values for part in raw.split(",") if part.strip()]
    return tuple(items) if items else None


def _evaluate_checkpoint(
    model: MaskablePPO,
    model_path: Path,
    cfg: LabConfig,
    policy_cfg: NeuralPolicyConfig,
    *,
    eval_dir: Path,
    eval_seeds: tuple[int, ...],
    run_id: str,
) -> dict[str, Any]:
    """Run deterministic evaluation on *eval_seeds* and format summary + per-seed metrics."""
    policy = NeuralPolicy(model, model_path, policy_cfg)
    eval_summary, _, _ = Evaluator(cfg, eval_dir).evaluate(
        policy,
        eval_seeds,
        checkpoint_episode=0,
        deterministic=True,
        capture_replays=True,
        capture_step_telemetry=False,
        evaluation_mode="development_strided_gate_v1",
        curriculum_stage=cfg.environment.curriculum_stage,
        run_id=run_id,
        checkpoint_id=f"strided_{sha256_file(model_path)[:16]}",
    )
    per_seed = [
        {
            "seed": int(r.seed),
            "success": bool(r.success),
            "sheep_penned": int(r.sheep_penned),
            "steps": int(r.steps),
            "stop_reason": r.stop_reason,
            "reward": float(r.reward_total),
            "replay_path": r.replay_path,
        }
        for r in eval_summary.records
    ]
    return {
        "success_rate": float(eval_summary.success_rate),
        "average_reward": float(eval_summary.average_reward),
        "success_count": sum(1 for r in eval_summary.records if r.success),
        "episode_count": len(eval_summary.records),
        "per_seed": per_seed,
    }


def run(
    root: Path,
    *,
    transitions_per_arm_seed: int = 12_288,
    max_arm_seconds: float = 600.0,
    arms: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    target_ratio: float = 0.25,
    eval_seeds: tuple[int, ...] | list[int] | None = None,
    lr: float | None = None,
    target_kl: float | None = None,
    eval_interval: int | None = None,
) -> dict[str, Any]:
    """Execute bounded paired continuation training on a verified workspace (isolated from live artifacts)."""
    verification = verify(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["entries"]

    arm_filter = set(arms) if arms is not None else None
    if arm_filter is not None:
        unknown_arms = arm_filter - set(ARMS)
        if unknown_arms:
            raise ValueError(f"Unknown arm filter(s): {sorted(unknown_arms)}")

    seed_filter = {int(s) for s in seeds} if seeds is not None else None
    if seed_filter is not None:
        available_seeds = {int(e["seed"]) for e in entries.values()}
        unknown_seeds = seed_filter - available_seeds
        if unknown_seeds:
            raise ValueError(f"Unknown seed filter(s): {sorted(unknown_seeds)}")

    selected = [
        (key, entry)
        for key, entry in entries.items()
        if (arm_filter is None or entry["arm"] in arm_filter)
        and (seed_filter is None or int(entry["seed"]) in seed_filter)
    ]
    if not selected:
        raise ValueError("No prepared arm/seed entries matched the requested filters")

    if eval_interval is not None and int(eval_interval) <= 0:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}")

    active_eval_seeds = tuple(int(s) for s in eval_seeds) if eval_seeds is not None else tuple(DEV_SEEDS)
    failure_weights = dict(DEFAULT_FAILURE_WEIGHTS)

    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))

    results: dict[str, Any] = {}
    for key, entry in selected:
        ws = Path(entry["workspace"])
        train_seed = int(entry["seed"])
        cfg_dict = json.loads(Path(entry["config_path"]).read_text(encoding="utf-8"))
        cfg_dict["training"]["total_timesteps"] = int(transitions_per_arm_seed)
        cfg_dict["training"]["failure_directed_target_ratio"] = float(target_ratio)
        if lr is not None:
            cfg_dict["training"]["learning_rate"] = float(lr)
        cfg_dict["training"]["output_dir"] = str(ws / "training_output")
        cfg_dict["training"]["backup_dir"] = str(ws / "backups")
        cfg_dict["training"]["web_export_dir"] = str(ws / "web")
        cfg = LabConfig.from_dict(cfg_dict)

        set_random_seed(train_seed)
        venv = DummyVecEnv([lambda cfg=cfg: SheepdogRLAdapter(cfg)])
        t0 = time.monotonic()
        deadline = t0 + float(max_arm_seconds)
        total_transitions = 0
        total_world_steps = 0
        per_segment: list[dict[str, Any]] = []
        policy_cfg: NeuralPolicyConfig | None = None
        try:
            venv.env_method("update_failure_weights", failure_weights)
            model = MaskablePPO.load(str(entry["model_path"]), env=venv, device="cpu")
            if (
                model.n_envs != 1
                or transitions_per_arm_seed <= 0
                or transitions_per_arm_seed % model.n_steps != 0
            ):
                raise ValueError(
                    f"transitions_per_arm_seed ({transitions_per_arm_seed}) must be a positive "
                    f"multiple of single-env rollout size ({model.n_steps})"
                )
            if eval_interval is not None and int(eval_interval) % model.n_steps != 0:
                raise ValueError(
                    f"eval_interval ({eval_interval}) must be a positive multiple of "
                    f"single-env rollout size ({model.n_steps})"
                )
            if lr is not None:
                fixed_lr = float(lr)
                model.learning_rate = fixed_lr
                model.lr_schedule = lambda _, value=fixed_lr: value
            if target_kl is not None:
                model.target_kl = float(target_kl)
            model.tensorboard_log = None
            model.verbose = 0
            model.set_random_seed(train_seed)

            policy_cfg = NeuralPolicyConfig.from_dict(
                entry["conversion"]["policy_config"],
                observation_size=int(model.observation_space.shape[0]),
            )

            if eval_interval is None:
                callback = _RolloutBudgetCallback(
                    deadline=deadline,
                    transitions=int(transitions_per_arm_seed),
                )
                model.learn(
                    total_timesteps=int(transitions_per_arm_seed),
                    reset_num_timesteps=True,
                    callback=callback,
                )
                total_transitions = callback.transitions
                total_world_steps = callback.world_steps
            else:
                milestones_dir = ws / "milestones"
                milestones_dir.mkdir(parents=True, exist_ok=True)
                remaining = int(transitions_per_arm_seed)
                seg_idx = 0
                while remaining > 0 and time.monotonic() < deadline:
                    seg_idx += 1
                    seg_budget = min(int(eval_interval), remaining)
                    callback = _RolloutBudgetCallback(
                        deadline=deadline,
                        transitions=seg_budget,
                    )
                    model.learn(
                        total_timesteps=seg_budget,
                        reset_num_timesteps=(seg_idx == 1),
                        callback=callback,
                    )
                    total_transitions += callback.transitions
                    total_world_steps += callback.world_steps
                    remaining -= seg_budget

                    milestone_path = milestones_dir / f"model_seg{seg_idx}_ts_{total_transitions}.zip"
                    model.save(str(milestone_path))
                    seg_eval = _evaluate_checkpoint(
                        model,
                        milestone_path,
                        cfg,
                        policy_cfg,
                        eval_dir=ws / "evaluation" / f"ts_{total_transitions}",
                        eval_seeds=active_eval_seeds,
                        run_id=root.name,
                    )
                    per_segment.append(
                        {
                            "segment": seg_idx,
                            "transitions": total_transitions,
                            "segment_transitions": callback.transitions,
                            "world_steps": total_world_steps,
                            "elapsed_seconds": round(time.monotonic() - t0, 3),
                            "optimizer_state_len": len(model.policy.optimizer.state),
                            "model_path": str(milestone_path),
                            "model_sha256": sha256_file(milestone_path),
                            **seg_eval,
                        }
                    )
                    if callback.transitions < seg_budget or time.monotonic() >= deadline:
                        break

            final_model = ws / "final_model.zip"
            model.save(str(final_model))
            sampling_usage = venv.env_method("get_scenario_sampler_summary")
        finally:
            venv.close()

        assert policy_cfg is not None
        if per_segment and per_segment[-1]["transitions"] == total_transitions:
            last_seg = per_segment[-1]
            final_eval = {
                "success_rate": last_seg["success_rate"],
                "average_reward": last_seg["average_reward"],
                "success_count": last_seg["success_count"],
                "episode_count": last_seg["episode_count"],
                "per_seed": last_seg["per_seed"],
            }
        else:
            final_eval = _evaluate_checkpoint(
                model,
                final_model,
                cfg,
                policy_cfg,
                eval_dir=ws / "evaluation",
                eval_seeds=active_eval_seeds,
                run_id=root.name,
            )

        entry_result: dict[str, Any] = {
            "arm": entry["arm"],
            "seed": train_seed,
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "transitions": total_transitions,
            "world_steps": total_world_steps,
            "complete_training_budget": total_transitions == int(transitions_per_arm_seed),
            "effective_learning_rate": float(cfg.training.learning_rate),
            "effective_target_kl": model.target_kl,
            "sampling_usage": sampling_usage,
            "failure_directed_weights": failure_weights,
            "failure_directed_target_ratio": float(target_ratio),
            "final_model_path": str(final_model),
            "final_model_sha256": sha256_file(final_model),
            **final_eval,
        }
        if eval_interval is not None:
            entry_result["eval_interval"] = int(eval_interval)
            entry_result["per_segment"] = per_segment
        results[key] = entry_result

    run_summary = {
        "status": "completed",
        "env_config_version": ENV_CONFIG_VERSION,
        "verification": verification,
        "transitions_per_arm_seed": int(transitions_per_arm_seed),
        "max_arm_seconds": float(max_arm_seconds),
        "lr": float(lr) if lr is not None else None,
        "target_kl": float(target_kl) if target_kl is not None else None,
        "eval_interval": int(eval_interval) if eval_interval is not None else None,
        "failure_directed_weights": failure_weights,
        "failure_directed_target_ratio": float(target_ratio),
        "eval_seeds": list(active_eval_seeds),
        "results": results,
    }
    atomic_write_json(root / "run_results.json", run_summary)
    return run_summary


def main() -> None:
    """CLI entrypoint for prepare, verify, and run modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_PARENT_METADATA)
    parser.add_argument("--sha256", default=DEFAULT_PARENT_SHA256)
    parser.add_argument("--transitions", type=int, default=12_288)
    parser.add_argument("--max-arm-seconds", type=float, default=600.0)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--target-ratio", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    args = parser.parse_args()

    if args.command == "prepare":
        manifest = prepare(args.root, args.parent, args.metadata, args.sha256)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "stacked_observation_size": manifest["stacked_observation_size"],
                    "workspaces": len(manifest["entries"]),
                }
            )
        )
    elif args.command == "verify":
        print(json.dumps(verify(args.root)))
    else:
        print(
            json.dumps(
                run(
                    args.root,
                    transitions_per_arm_seed=args.transitions,
                    max_arm_seconds=args.max_arm_seconds,
                    arms=_parse_str_list(args.arms),
                    seeds=_parse_int_list(args.seeds),
                    target_ratio=args.target_ratio,
                    lr=args.lr,
                    target_kl=args.target_kl,
                    eval_interval=args.eval_interval,
                )
            )
        )


if __name__ == "__main__":
    main()
