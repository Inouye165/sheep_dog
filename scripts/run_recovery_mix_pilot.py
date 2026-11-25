"""Isolated, bounded development comparison of 25% and 60% recovery starts.

No active training state is read for continuation or mutated. Both arms load the
same immutable copied parent, including its optimizer; only the sampling ratio
differs. This small pilot cannot establish release readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy
from sheepdog.training.rl_env import SheepdogRLAdapter

PARENT = Path("artifacts/checkpoints/stage34/maskable-ppo-stage34-343721120.zip")
METADATA = Path(
    "artifacts/checkpoints/checkpoint-chk_run_20260817_063453_91f4_pv_10481_ts_343721120.json"
)
DEV_SEEDS = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
TRAIN_SEEDS = (101, 202, 303)
RATIOS = (0.25, 0.60)
TRANSITIONS = 12_288
FAILURE_WEIGHTS = {
    "wall_stray_extraction": 1.0,
    "high_pen_standoff": 1.0,
    "gate_wall_flock": 1.0,
}


def digest(path: Path) -> str:
    """Return an artifact's SHA-256 identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int) -> list[float] | None:
    """Episode-level descriptive interval; shared seeds remain correlated."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / denom
    delta = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denom
    return [mid - delta, mid + delta]


def make_config(base: dict[str, Any], metadata: dict[str, Any], root: Path,
                seed: int, ratio: float) -> LabConfig:
    """Reuse checkpoint physics/rewards; isolate outputs and freeze adaptation."""
    payload = json.loads(json.dumps(base))
    payload["environment"] = metadata["environment_config"]
    payload["rewards"] = metadata["reward_config"]
    payload["training"].update({
        "train_seed": seed, "output_dir": str(root),
        "backup_dir": str(root / "backups"), "web_export_dir": str(root / "web"),
        "backup_enabled": False, "ppo_env_workers": 1,
        "enable_adaptive_learning": False, "auto_rollback_on_collapse": False,
        "scenario_training_enabled": False, "failure_directed_training_enabled": True,
        "failure_directed_target_ratio": ratio, "observation_mode": "guided",
    })
    config = LabConfig.from_dict(payload)
    if (config.environment.dogs, config.environment.sheep,
            config.environment.max_steps, config.environment.curriculum_stage) != (3, 6, 2320, 34):
        raise ValueError("Expected Stage 34, three dogs, six sheep, 2320 world steps")
    if metadata["policy_config"]["observation_size"] != 54:
        raise ValueError("Expected original 54-input guided checkpoint")
    return config


class LimitCallback(BaseCallback):
    """Count executed transitions/world steps and stop a timed-out training arm."""

    def __init__(self, deadline: float) -> None:
        super().__init__()
        self.deadline = deadline
        self.transitions = 0
        self.world_steps = 0
        self.episodes: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.transitions += len(infos)
        self.world_steps += sum(bool(info.get("team_step_completed")) for info in infos)
        self.episodes.extend(info["episode"] for info in infos if "episode" in info)
        return self.transitions < TRANSITIONS + 1 and time.monotonic() < self.deadline


def verify_parent(metadata: dict[str, Any], archive: dict[str, Any]) -> None:
    """Reject mismatched metadata and unsupported checkpoint architecture."""
    if metadata["curriculum_stage"] != 34:
        raise ValueError("Parent is not Stage 34")
    if Path(metadata["policy_state_path"].replace("\\", "/")).name != PARENT.name:
        raise ValueError("Metadata points to a different parent")
    if archive["n_steps"] != 2048 or archive["policy_kwargs"]["net_arch"] != [256, 256, 128]:
        raise ValueError("Unexpected rollout size or architecture")


def source_record(root: Path) -> dict[str, Any]:
    """Preserve the exact tracked patch and identities of untracked source."""
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    (root / "dirty.patch").write_bytes(patch)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked": {p: digest(Path(p)) for p in untracked.decode().split("\0") if p and Path(p).is_file()},
    }


def evaluate(model: MaskablePPO, config: LabConfig, root: Path,
             deadline: float, model_path: Path) -> dict[str, Any]:
    """Evaluate ordinary full-task starts using the existing evaluator."""
    policy = NeuralPolicy(model, model_path, NeuralPolicyConfig(
        hidden_sizes=(256, 256, 128), observation_size=54, action_size=9,
        observation_mode="guided"))
    summary, _, _ = Evaluator(config, root).evaluate(
        policy, DEV_SEEDS, checkpoint_episode=0, deterministic=True,
        capture_replays=True, capture_step_telemetry=False,
        evaluation_mode="development_recovery_mix_pilot", curriculum_stage=34,
        run_id=root.parent.name, checkpoint_id=f"pilot_{digest(model_path)[:16]}",
        should_stop=lambda: time.monotonic() >= deadline,
    )
    data = summary.to_dict()
    data["success_wilson_95"] = wilson(sum(r.success for r in summary.records), len(summary.records))
    successful_steps = [r.steps for r in summary.records if r.success]
    data["steps_among_successes"] = successful_steps
    data["mean_steps_among_successes"] = (
        sum(successful_steps) / len(successful_steps) if successful_steps else None
    )
    data["model_path"] = str(model_path)
    data["model_sha256"] = digest(model_path)
    return data


def run(root: Path, *, train: bool = False) -> dict[str, Any]:
    """Prepare and baseline by default; training requires explicit --train."""
    root = root.resolve()
    allowed = Path("artifacts/experiments").resolve()
    if root.parent != allowed or not root.name.startswith("recovery_mix_pilot_"):
        raise ValueError("Use a new artifacts/experiments/recovery_mix_pilot_<timestamp> directory")
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + 900
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))
    parent_hash = digest(PARENT)
    shutil.copy2(PARENT, root / "parent.zip")
    if digest(root / "parent.zip") != parent_hash:
        raise ValueError("Parent changed while being copied")
    shutil.copy2(METADATA, root / "parent_metadata.json")
    metadata = json.loads((root / "parent_metadata.json").read_text())
    base = json.loads(Path("artifacts/effective-training-config.json").read_text())
    with zipfile.ZipFile(root / "parent.zip") as z:
        archive = json.loads(z.read("data"))
    verify_parent(metadata, archive)
    # Config records match the serialized optimizer/model settings, rather than
    # a potentially newer active trainer's adaptive values.
    for config_key, model_key in (
        ("gamma", "gamma"), ("gae_lambda", "gae_lambda"),
        ("entropy_coef", "ent_coef"), ("learning_rate", "learning_rate"),
        ("rollout_steps", "n_steps"), ("batch_size", "batch_size"),
    ):
        base["training"][config_key] = archive[model_key]
    manifest: dict[str, Any] = {
        "status": "prepared", "hypothesis": "60% varied recovery starts versus 25% improves development full-flock completion",
        "assistance": "guided; existing physics assistance retained", "parent_sha256": parent_hash,
        "parent_checkpoint_id": metadata["checkpoint_id"], "parent_archive_hyperparameters": {
            k: archive[k] for k in ("gamma", "gae_lambda", "ent_coef", "learning_rate", "n_steps", "batch_size", "n_epochs")},
        "source": source_record(root), "versions": collect_dependency_versions(),
        "training_seeds": TRAIN_SEEDS, "development_seeds": DEV_SEEDS, "held_out": "unused",
        "failure_weights": FAILURE_WEIGHTS, "ratios": RATIOS,
        "sampling": "fixed development-directed classes, not closed-loop adaptation",
        "optimizer": "same saved parent Adam state independently restored in every arm",
        "budget": {"transitions_per_arm_seed": TRANSITIONS, "maximum_total_transitions": TRANSITIONS * 6,
                   "maximum_evaluation_episodes": 70, "arm_train_wall_seconds": 120, "overall_wall_seconds": 900},
        "metrics": "all-six completion/10 dev episodes; timeout and no-progress rates; steps among successes; gate-corridor and controlled-stall durations as existing evaluator definitions; no release promotion",
        "interval_limit": "Wilson intervals describe episodes; repeated development seeds and shared parent induce dependence, not independent release evidence",
        "runs": [],
    }
    atomic_write_json(root / "manifest.json", manifest)
    try:
        cfg = make_config(base, metadata, root / "baseline", 101, .25)
        atomic_write_json(root / "baseline_config.json", cfg.to_dict())
        baseline = MaskablePPO.load(str(root / "parent.zip"), device="cpu")
        manifest["baseline"] = evaluate(baseline, cfg, root / "baseline", deadline, root / "parent.zip")
        atomic_write_json(root / "manifest.json", manifest)
        del baseline
        if train:
            for seed in TRAIN_SEEDS:
                for ratio in RATIOS:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Overall pilot wall budget exhausted")
                    arm_dir = root / f"seed_{seed}_ratio_{ratio:.2f}"
                    arm_dir.mkdir()
                    cfg = make_config(base, metadata, arm_dir, seed, ratio)
                    atomic_write_json(arm_dir / "config.json", cfg.to_dict())
                    set_random_seed(seed)
                    env = DummyVecEnv([lambda cfg=cfg: SheepdogRLAdapter(cfg)])
                    try:
                        env.env_method("update_failure_weights", FAILURE_WEIGHTS)
                        model = MaskablePPO.load(str(root / "parent.zip"), env=env, device="cpu")
                        model.tensorboard_log = None
                        model.verbose = 0
                        model.set_random_seed(seed)
                        callback = LimitCallback(min(deadline, time.monotonic() + 120))
                        model.learn(total_timesteps=TRANSITIONS, reset_num_timesteps=True, callback=callback)
                        model_path = arm_dir / "model.zip"
                        model.save(str(model_path))
                        record = {
                            "seed": seed, "ratio": ratio, "transitions": callback.transitions,
                            "world_steps": callback.world_steps, "training_episodes": callback.episodes,
                            "sampling_usage": env.env_method("get_scenario_sampler_summary"),
                            "complete_training_budget": callback.transitions == TRANSITIONS,
                            "model_sha256": digest(model_path),
                        }
                        atomic_write_json(arm_dir / "training.json", record)
                        manifest["runs"].append(record)
                        atomic_write_json(root / "manifest.json", manifest)
                        record["evaluation"] = evaluate(model, cfg, arm_dir / "evaluation", deadline, model_path)
                        atomic_write_json(root / "manifest.json", manifest)
                    finally:
                        env.close()
            manifest["status"] = "complete" if all(r["complete_training_budget"] for r in manifest["runs"]) else "incomplete_training_budget"
        else:
            manifest["status"] = "baseline_only"
    except Exception as exc:
        manifest["status"] = "incomplete"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        manifest["source_parent_unchanged"] = digest(PARENT) == parent_hash
        manifest["checksums"] = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
        atomic_write_json(root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, train=args.train)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))
