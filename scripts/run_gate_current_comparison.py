"""Run the bounded current-parent gate comparison in an isolated process."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from scripts.run_recovery_mix_pilot import (
    FAILURE_WEIGHTS,
    TRANSITIONS,
    LimitCallback,
    digest,
    source_record,
    wilson,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.gate_information_preparation import ARMS, DEV_SEEDS, SEEDS, verify
from sheepdog.observations import GATE_OBSERVATION_SCHEMA_VERSION
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter

PREPARATION = Path("artifacts/experiments/gate_current_prep_20261113_173000")


def isolated_config(payload: dict, root: Path, mode: str | None = None) -> LabConfig:
    """Retain prepared settings, changing output paths and explicit mode only."""
    data = json.loads(json.dumps(payload))
    data["training"].update(output_dir=str(root / "training_output"),
                            backup_dir=str(root / "backups"), web_export_dir=str(root / "web"))
    if mode is not None:
        data["training"]["observation_mode"] = mode
    return LabConfig.from_dict(data)


class AuditedPolicy(NeuralPolicy):
    """Reject evaluation wiring mismatches before selecting any action."""

    def __init__(self, model, path: Path, config: LabConfig):
        mode = config.training.observation_mode
        size = int(model.observation_space.shape[0])
        super().__init__(model, path, NeuralPolicyConfig(
            hidden_sizes=(256, 256, 128), observation_size=size, action_size=9,
            observation_mode=mode,
            observation_schema_version=GATE_OBSERVATION_SCHEMA_VERSION if size == 58 else None))
        self.observation_dimensions: set[int] = set()
        self.episode_observation_checks = 0

    def select_actions(self, environment, deterministic=True):
        """Validate each new evaluation episode before normal learned selection."""
        if environment.step_count == 0:
            for dog in range(environment.dog_count):
                values = environment.build_observation_for_dog(dog).values
                self.observation_dimensions.add(len(values))
                if len(values) != self.config.observation_size:
                    raise ValueError("Evaluator observation size does not match checkpoint")
                if self.config.observation_mode == "guided_gate_control" and any(values[54:]):
                    raise ValueError("Control gate features must be zero")
                if self.config.observation_mode == "guided_gate_treatment" and not any(values[56:]):
                    raise ValueError("Treatment is missing the gate normal")
            self.episode_observation_checks += 1
        return super().select_actions(environment, deterministic=deterministic)


def evaluate(model, config: LabConfig, root: Path, deadline: float, model_path: Path) -> dict:
    """Use ordinary full-task starts, auditing the evaluator's real environment."""
    policy = AuditedPolicy(model, model_path, config)
    summary, _, _ = Evaluator(config, root).evaluate(
        policy, DEV_SEEDS, checkpoint_episode=0, deterministic=True,
        capture_replays=True, capture_step_telemetry=False,
        evaluation_mode="development_gate_information_v2", curriculum_stage=34,
        run_id=root.parent.name, checkpoint_id=f"gate_{digest(model_path)[:16]}",
        should_stop=lambda: time.monotonic() >= deadline)
    data = summary.to_dict()
    successes = [r.steps for r in summary.records if r.success]
    data.update(model_path=str(model_path), model_sha256=digest(model_path),
                observation_mode=config.training.observation_mode,
                audited_observation_dimensions=sorted(policy.observation_dimensions),
                audited_episodes=policy.episode_observation_checks,
                steps_among_successes=successes,
                mean_steps_among_successes=sum(successes) / len(successes) if successes else None,
                success_wilson_95=wilson(len(successes), len(summary.records)))
    return data


def preflight(model_path: Path, config: LabConfig) -> dict:
    """Exercise the same direct environment construction used by Evaluator."""
    model = MaskablePPO.load(str(model_path), device="cpu")
    env = SheepdogEnvironment(config)
    env.reset(seed=53)
    policy = AuditedPolicy(model, model_path, config)
    env.step(policy.select_actions(env))
    return {"mode": config.training.observation_mode,
            "dimensions": sorted(policy.observation_dimensions), "world_steps": 1}


def run(root: Path, preparation: Path = PREPARATION) -> dict:
    """Never resume the live job or mutate its checkpoint/configuration."""
    root = root.resolve()
    if root.parent != Path("artifacts/experiments").resolve() or not root.name.startswith("gate_current_run_"):
        raise ValueError("Use a new artifacts/experiments/gate_current_run_<timestamp> directory")
    if root.exists():
        raise FileExistsError("Run requires a new directory")
    verify(preparation)
    start = time.monotonic()
    deadline = start + 900
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))
    root.mkdir(parents=True)
    frozen = root / "prepared"
    shutil.copytree(preparation, frozen)
    verify(frozen)
    prep = json.loads((frozen / "manifest.json").read_text())
    manifest = {
        "protocol": "stage34_gate_information_current_parent_v2_bounded_run",
        "status": "running", "hypothesis": prep["hypothesis"],
        "assistance": prep["assistance"], "held_out": "unused",
        "preparation": str(preparation), "parent": prep["parent"],
        "source": source_record(root), "versions": collect_dependency_versions(),
        "training_seeds": list(SEEDS), "development_seeds": list(DEV_SEEDS),
        "optimizer": "identical fresh Adam states from zero-expanded prepared models",
        "sampling": prep["sampling"], "budget": prep["proposed_budget_only"],
        "metrics": "All-six success/10 ordinary development episodes per arm-seed; timeout/no-progress; successful completion steps; existing controlled-stall/gate-corridor definitions",
        "uncertainty": "Shared parent and repeated development seeds are correlated. Wilson intervals are descriptive, not release evidence.",
        "preflight": [], "runs": [],
    }
    atomic_write_json(root / "manifest.json", manifest)
    try:
        first = prep["entries"][f"arm_a_control_zero_gate/{SEEDS[0]}"]
        payload = json.loads((frozen / first["config"]).read_text())
        base_cfg = isolated_config(payload, root / "baseline", "guided")
        parent_path = frozen / prep["parent"]["model"]
        atomic_write_json(root / "baseline_config.json", base_cfg.to_dict())
        for arm in ARMS:
            entry = prep["entries"][f"{arm}/{SEEDS[0]}"]
            cfg = isolated_config(json.loads((frozen / entry["config"]).read_text()), root / "preflight" / arm)
            manifest["preflight"].append(preflight(frozen / entry["model"], cfg))
        model = MaskablePPO.load(str(parent_path), device="cpu")
        manifest["baseline"] = evaluate(model, base_cfg, root / "baseline", deadline, parent_path)
        atomic_write_json(root / "manifest.json", manifest)
        print(json.dumps({"baseline_success": manifest["baseline"]["success_rate"]}), flush=True)
        del model
        for seed in SEEDS:
            for arm, mode in ARMS.items():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Overall pilot budget exhausted")
                entry = prep["entries"][f"{arm}/{seed}"]
                arm_dir = root / arm / f"seed_{seed}"
                arm_dir.mkdir(parents=True)
                cfg = isolated_config(json.loads((frozen / entry["config"]).read_text()), arm_dir)
                atomic_write_json(arm_dir / "config.json", cfg.to_dict())
                set_random_seed(seed)
                env = DummyVecEnv([lambda cfg=cfg: SheepdogRLAdapter(cfg)])
                try:
                    env.env_method("update_failure_weights", FAILURE_WEIGHTS)
                    model = MaskablePPO.load(str(frozen / entry["model"]), env=env, device="cpu")
                    model.tensorboard_log, model.verbose = None, 0
                    model.set_random_seed(seed)
                    callback = LimitCallback(min(deadline, time.monotonic() + 120))
                    model.learn(total_timesteps=TRANSITIONS, reset_num_timesteps=True, callback=callback)
                    path = arm_dir / "model.zip"
                    model.save(str(path))
                    record = {"seed": seed, "arm": arm, "observation_mode": mode,
                              "transitions": callback.transitions, "world_steps": callback.world_steps,
                              "complete_training_budget": callback.transitions == TRANSITIONS,
                              "training_episodes": callback.episodes,
                              "sampling_usage": env.env_method("get_scenario_sampler_summary"),
                              "model_sha256": digest(path)}
                    manifest["runs"].append(record)
                    atomic_write_json(arm_dir / "training.json", record)
                    atomic_write_json(root / "manifest.json", manifest)
                    record["evaluation"] = evaluate(model, cfg, arm_dir / "evaluation", deadline, path)
                    atomic_write_json(root / "manifest.json", manifest)
                    print(json.dumps({"seed": seed, "arm": arm,
                                      "success": record["evaluation"]["success_rate"],
                                      "transitions": record["transitions"],
                                      "world_steps": record["world_steps"]}), flush=True)
                finally:
                    env.close()
        manifest["status"] = "complete" if all(r["complete_training_budget"] for r in manifest["runs"]) else "incomplete_training_budget"
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["prepared_parent_unchanged"] = digest(frozen / prep["parent"]["model"]) == prep["parent"]["sha256"]
        manifest["checksums"] = {str(p.relative_to(root)): digest(p)
                                 for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
        atomic_write_json(root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, default=PREPARATION)
    args = parser.parse_args()
    result = run(args.output_dir, args.preparation)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
