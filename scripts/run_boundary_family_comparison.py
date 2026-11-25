"""Bounded comparison of near-pen versus boundary-split recovery curricula.

Both arms retain the same original 54-input policy and saved Adam state. Only
the procedurally sampled recovery family differs; no live job is resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from scripts.run_gate_current_comparison import evaluate
from scripts.run_recovery_mix_pilot import digest, source_record
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.gate_information_preparation import (
    DEV_SEEDS,
    SEEDS,
    arm_config,
    inspect_parent,
)
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter
from sheepdog.training.training_scenarios import list_scenario_types

PARENT_ROOT = Path("artifacts/experiments/stopped_adoption_20261113_171410/artifacts/checkpoints")
PARENT = PARENT_ROOT / "stage34/maskable-ppo-stage34-343787786.zip"
METADATA = PARENT_ROOT / "checkpoint-chk_run_20260817_063453_91f4_pv_10483_ts_343787786.json"
PARENT_SHA256 = "26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2"
ARMS = {"control_near_pen_stray": "wall_stray_extraction",
        "treatment_boundary_split": "boundary_split_flock"}
TRANSITIONS = 24_576


def optimizer_digest(model: MaskablePPO) -> str:
    """Hash every Adam tensor, scalar, parameter group and state key deterministically."""
    result = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            result.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                result.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            result.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            result.update(repr(value).encode())

    visit(model.policy.optimizer.state_dict())
    return result.hexdigest()


def load_continuation(path: Path, env: DummyVecEnv | None,
                      expected_optimizer: str) -> MaskablePPO:
    """Load the serialized optimizer without resetting or rebuilding its state."""
    model = MaskablePPO.load(str(path), env=env, device="cpu")
    if not model.policy.optimizer.state:
        raise ValueError("Expected populated saved Adam state")
    if optimizer_digest(model) != expected_optimizer:
        raise ValueError("Loaded Adam state does not match the immutable parent")
    model.tensorboard_log, model.verbose = None, 0
    return model


def comparison_config(meta: dict, archive: dict, output: Path, seed: int) -> LabConfig:
    """Both curricula share one configuration apart from output directories."""
    cfg = arm_config(meta, archive, output, seed, "guided")
    return replace(cfg, training=replace(cfg.training, failure_directed_target_ratio=0.5))


class ExposureAdapter(SheepdogRLAdapter):
    """Attach passive pre-step family labels before VecEnv's automatic reset."""

    def __init__(self, config: LabConfig, family: str):
        self.start_records: list[dict] = []
        super().__init__(config)
        self.family = family

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.start_records.append({
            "seed": int(self._latest_seed), "scenario_id": self._latest_scenario_id,
            "family": self.family if self._latest_scenario_id is not None else "normal",
            "initial_penned": sum(bool(s.penned) for s in self._environment.sheep),
            "total_sheep": len(self._environment.sheep),
        })
        return observation, info

    def step(self, action):
        family = self.family if self._latest_scenario_id is not None else "normal"
        observation, reward, terminated, truncated, info = super().step(action)
        info["comparison_start_family"] = family
        return observation, reward, terminated, truncated, info


class BudgetCallback(BaseCallback):
    """Count actual transitions and world steps with passive family exposure."""

    def __init__(self, deadline: float, transitions: int = TRANSITIONS):
        super().__init__()
        self.deadline, self.limit = deadline, transitions
        self.transitions = 0
        self.world_steps = 0
        self.family_world_steps: dict[str, int] = {}
        self.episodes: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.transitions += len(infos)
        for info in infos:
            if info.get("team_step_completed"):
                self.world_steps += 1
                name = info["comparison_start_family"]
                self.family_world_steps[name] = self.family_world_steps.get(name, 0) + 1
            if "episode" in info:
                self.episodes.append(info["episode"])
        # The learner's total_timesteps is an exact multiple of its single-env
        # rollout size. Allow the last step so that final rollout is optimized.
        return self.transitions <= self.limit and time.monotonic() < self.deadline


def validate_rollout_budget(model: MaskablePPO, transitions: int) -> None:
    """Prevent SB3 rounding the requested budget up to another rollout."""
    if model.n_envs != 1 or transitions <= 0 or transitions % model.n_steps:
        raise ValueError("Budget must be a positive whole number of single-env rollouts")


def run(root: Path) -> dict:
    """Run at most six bounded training arms and seventy ordinary evaluations."""
    root = root.resolve()
    if root.parent != Path("artifacts/experiments").resolve() or not root.name.startswith("boundary_family_run_"):
        raise ValueError("Use a new artifacts/experiments/boundary_family_run_<timestamp> directory")
    if root.exists():
        raise FileExistsError("Run requires a new output directory")
    if not set(ARMS.values()) <= set(list_scenario_types()):
        raise ValueError("Both tested scenario builders must be registered")
    meta, archive = inspect_parent(PARENT, METADATA, PARENT_SHA256)
    start, deadline = time.monotonic(), time.monotonic() + 900
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))
    root.mkdir(parents=True)
    frozen = root / "parent" / PARENT.name
    frozen.parent.mkdir()
    shutil.copy2(PARENT, frozen)
    shutil.copy2(METADATA, frozen.parent / "metadata.json")
    if digest(frozen) != PARENT_SHA256:
        raise ValueError("Copied parent SHA-256 mismatch")
    baseline_model = MaskablePPO.load(str(frozen), device="cpu")
    expected_optimizer = optimizer_digest(baseline_model)
    if not baseline_model.policy.optimizer.state:
        raise ValueError("Saved parent has no Adam state")
    manifest = {
        "protocol": "stage34_boundary_family_retained_adam_v1", "status": "running",
        "hypothesis": "Matched whole-flock boundary-split starts improve seed-59 recovery versus near-pen stray starts",
        "assistance": "guided; existing parent role/physics assistance retained",
        "parent_sha256": PARENT_SHA256, "parent_checkpoint_id": meta["checkpoint_id"],
        "parent_model": str(frozen.relative_to(root)), "optimizer_initial_sha256": expected_optimizer,
        "optimizer": "Populated saved Adam tensors, steps and parameter groups independently restored and verified in every arm",
        "training_seeds": list(SEEDS), "development_seeds": list(DEV_SEEDS), "held_out": "unused",
        "families": ARMS, "target_probability": 0.5,
        "live_default_probability": 0.25,
        "budget": {"transitions_per_arm": TRANSITIONS, "maximum_total_transitions": TRANSITIONS * 6,
                   "maximum_evaluation_episodes": 70, "train_arm_wall_seconds": 120,
                   "overall_wall_seconds": 900, "relative_to_previous_73728_transition_pilot": 2},
        "decision_rule": "Exploratory candidate only if seed59 solves more paired cases and pooled ordinary full-flock success is not worse; report every regression; no automatic promotion",
        "metric_definitions": "All-six success per10 ordinary development episodes/arm-seed; timeouts/no-progress per10; steps among successes; actual completed world steps per start family, including unfinished episodes",
        "uncertainty": "Shared parent and repeated development seeds are correlated; Wilson intervals are descriptive, not release evidence",
        "source": source_record(root), "versions": collect_dependency_versions(),
        "archive_hyperparameters": {k: archive[k] for k in
            ("gamma", "gae_lambda", "ent_coef", "learning_rate", "n_steps", "batch_size", "n_epochs")},
        "runs": [],
    }
    atomic_write_json(root / "manifest.json", manifest)
    try:
        cfg = comparison_config(meta, archive, root / "baseline", SEEDS[0])
        atomic_write_json(root / "baseline_config.json", cfg.to_dict())
        manifest["baseline"] = evaluate(baseline_model, cfg, root / "baseline", deadline, frozen)
        atomic_write_json(root / "manifest.json", manifest)
        print(json.dumps({"baseline_success": manifest["baseline"]["success_rate"]}), flush=True)
        del baseline_model
        for seed in SEEDS:
            for arm, family in ARMS.items():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Overall comparison wall budget exhausted")
                output = root / arm / f"seed_{seed}"
                output.mkdir(parents=True)
                cfg = comparison_config(meta, archive, output, seed)
                atomic_write_json(output / "config.json", cfg.to_dict())
                atomic_write_json(output / "curriculum_weights.json", {family: 1.0})
                set_random_seed(seed)
                env = DummyVecEnv([lambda cfg=cfg, family=family: ExposureAdapter(cfg, family)])
                try:
                    env.env_method("update_failure_weights", {family: 1.0})
                    model = load_continuation(frozen, env, expected_optimizer)
                    validate_rollout_budget(model, TRANSITIONS)
                    initial_updates = model._n_updates
                    expected_epochs = TRANSITIONS // model.n_steps * model.n_epochs
                    model.set_random_seed(seed)
                    callback = BudgetCallback(min(deadline, time.monotonic() + 120))
                    model.learn(total_timesteps=TRANSITIONS, reset_num_timesteps=True, callback=callback)
                    path = output / "model.zip"
                    model.save(str(path))
                    record = {"seed": seed, "arm": arm, "family": family,
                              "transitions": callback.transitions, "world_steps": callback.world_steps,
                              "family_world_steps": callback.family_world_steps,
                              "training_episodes": callback.episodes,
                              "started_episodes": env.get_attr("start_records")[0],
                              "complete_training_budget": callback.transitions == TRANSITIONS and
                              model._n_updates - initial_updates == expected_epochs,
                              "optimizer_initial_sha256": expected_optimizer,
                              "optimizer_epochs_completed": model._n_updates - initial_updates,
                              "optimizer_epochs_expected": expected_epochs,
                              "model_sha256": digest(path),
                              "sampling_usage": env.env_method("get_scenario_sampler_summary")}
                    manifest["runs"].append(record)
                    atomic_write_json(output / "training.json", record)
                    atomic_write_json(root / "manifest.json", manifest)
                    record["evaluation"] = evaluate(model, cfg, output / "evaluation", deadline, path)
                    atomic_write_json(root / "manifest.json", manifest)
                    print(json.dumps({"seed": seed, "arm": arm, "success": record["evaluation"]["success_rate"],
                                      "transitions": record["transitions"], "world_steps": record["world_steps"]}), flush=True)
                finally:
                    env.close()
        manifest["status"] = "complete" if all(r["complete_training_budget"] for r in manifest["runs"]) else "incomplete_training_budget"
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["parent_unchanged"] = digest(PARENT) == digest(frozen) == PARENT_SHA256
        manifest["checksums"] = {str(p.relative_to(root)): digest(p)
                                 for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
        atomic_write_json(root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args().output_dir)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
