"""Prepare a separately identified gate-information experiment from an explicit parent.

Historical v1 identity and optimization locks are deliberately untouched. This
module performs conversion and bounded equivalence checks, never optimization.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.gate_information_experiment import verify_converted_model_equivalence
from sheepdog.observations import (
    GATE_FEATURE_NAMES,
    GATE_OBSERVATION_SCHEMA_VERSION,
    build_observation_builder,
)
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    expand_policy_observation_inputs,
)
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions, sha256_file

PROTOCOL = "stage34_gate_information_current_parent_v2"
ARMS = {"arm_a_control_zero_gate": "guided_gate_control",
        "arm_b_treatment_gate_info": "guided_gate_treatment"}
SEEDS = (101, 202, 303)
DEV_SEEDS = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
ARCH = (256, 256, 128)


def inspect_parent(model: Path, metadata: Path, expected_sha256: str) -> tuple[dict, dict]:
    """Reject identity/task mismatches before model deserialization."""
    if sha256_file(model) != expected_sha256:
        raise ValueError("Parent SHA-256 mismatch")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if Path(meta["policy_state_path"].replace("\\", "/")).name != model.name:
        raise ValueError("Parent metadata references a different model")
    env, policy = meta["environment_config"], meta["policy_config"]
    if tuple(env.get(k) for k in ("dogs", "sheep", "curriculum_stage", "max_steps")) != (3, 6, 34, 2320):
        raise ValueError("Expected three dogs, six sheep, Stage 34, 2320 world steps")
    if policy.get("observation_size") != 54 or policy.get("action_size") != 9:
        raise ValueError("Expected guided 54-input, nine-action parent")
    if policy.get("observation_mode", "guided") != "guided":
        raise ValueError("Expected guided parent")
    with zipfile.ZipFile(model) as archive:
        if archive.testzip() is not None:
            raise ValueError("Parent ZIP failed integrity check")
        data = json.loads(archive.read("data"))
    if tuple(data["policy_kwargs"]["net_arch"]) != ARCH or tuple(policy["hidden_sizes"]) != ARCH:
        raise ValueError("Unexpected parent architecture")
    if data["n_steps"] != 2048 or data["batch_size"] != 1024:
        raise ValueError("Unexpected parent rollout or batch size")
    if not isinstance(data["learning_rate"], (float, int)):
        raise ValueError("Scheduled learning rates require an explicit protocol")
    return meta, data


def arm_config(meta: dict, archive: dict, workspace: Path, seed: int, mode: str) -> LabConfig:
    """Freeze parent physics/reward and archive-effective hyperparameters equally."""
    payload = LabConfig().to_dict()
    payload["environment"] = copy.deepcopy(meta["environment_config"])
    payload["rewards"] = copy.deepcopy(meta["reward_config"])
    training = payload["training"]
    training.update(train_seed=seed, observation_mode=mode, neural_hidden_sizes=list(ARCH),
                    output_dir=str(workspace / "training_output"),
                    backup_dir=str(workspace / "backups"), web_export_dir=str(workspace / "web"),
                    backup_enabled=False, wandb_enabled=False, ppo_env_workers=1,
                    enable_adaptive_learning=False, auto_rollback_on_collapse=False,
                    scenario_training_enabled=False, failure_directed_training_enabled=True,
                    failure_directed_target_ratio=0.25, evaluation_seeds=list(DEV_SEEDS))
    for config_key, model_key in (("gamma", "gamma"), ("gae_lambda", "gae_lambda"),
                                  ("entropy_coef", "ent_coef"), ("learning_rate", "learning_rate"),
                                  ("rollout_steps", "n_steps"), ("batch_size", "batch_size")):
        training[config_key] = archive[model_key]
    return LabConfig.from_dict(payload)


def exact_rollout_equivalence(parent: Path, models: dict[str, Path], config: LabConfig) -> dict:
    """Compare unrounded actions, complete snapshots and reward fields for 20 steps."""
    traces = {}
    parent_tensors = None
    for label, path, mode in [("parent", parent, "guided"),
                              *((arm, models[arm], mode) for arm, mode in ARMS.items())]:
        payload = config.to_dict()
        payload["training"]["observation_mode"] = mode
        env = SheepdogEnvironment(LabConfig.from_dict(payload))
        env._observation_builder = build_observation_builder(mode)
        env.reset(seed=53)
        model = MaskablePPO.load(str(path), device="cpu")
        if label == "parent":
            parent_tensors = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        else:
            for name, tensor in model.policy.state_dict().items():
                original = parent_tensors[name]
                if tensor.shape == original.shape:
                    preserved = torch.equal(tensor, original)
                elif (tensor.ndim == 2 and original.ndim == 2 and
                      tensor.shape == (original.shape[0], 58) and original.shape[1] == 54):
                    preserved = torch.equal(tensor[:, :54], original) and bool(torch.all(tensor[:, 54:] == 0))
                else:
                    preserved = False
                if not preserved:
                    raise AssertionError(f"Existing policy parameter changed: {label}/{name}")
        policy = NeuralPolicy(model, path, NeuralPolicyConfig(
            hidden_sizes=ARCH, observation_size=54 if label == "parent" else 58,
            observation_mode=mode))
        trace = []
        for _ in range(20):
            actions = policy.select_actions(env, deterministic=True)
            snapshot, reward = env.step(actions)
            trace.append((list(actions), snapshot.to_dict(), reward.to_dict()))
            if snapshot.terminated:
                break
        traces[label] = trace
    if len(traces["parent"]) != 20 or any(t != traces["parent"] for t in traces.values()):
        raise AssertionError("Exact 20-step seed-53 rollout equivalence failed")
    return {"seed": 53, "world_steps": 20, "exact_actions_snapshots_rewards": True,
            "all_existing_parameters_preserved": True,
            "parent_trace_sha256": hashlib.sha256(
                json.dumps(traces["parent"], sort_keys=True).encode()).hexdigest()}


def prepare(root: Path, parent: Path, metadata: Path, expected_sha256: str,
            baseline: Path | None = None) -> dict[str, Any]:
    """Prepare six models in a new directory and verify all six; never train."""
    if root.exists():
        raise FileExistsError("Preparation requires a new directory; no overwrites")
    meta, archive = inspect_parent(parent, metadata, expected_sha256)
    root.mkdir(parents=True)
    frozen = root / "parent" / parent.name
    frozen.parent.mkdir()
    shutil.copy2(parent, frozen)
    shutil.copy2(metadata, frozen.parent / "metadata.json")
    if sha256_file(frozen) != expected_sha256:
        raise ValueError("Copied parent SHA-256 mismatch")
    torch.set_num_threads(1)
    entries, checks = {}, {}
    for seed in SEEDS:
        models = {}
        for arm, mode in ARMS.items():
            ws = root / arm / f"seed_{seed}"
            ws.mkdir(parents=True)
            cfg = arm_config(meta, archive, ws, seed, mode)
            atomic_write_json(ws / "config.json", cfg.to_dict())
            models[arm] = ws / "initial_model.zip"
            conversion = expand_policy_observation_inputs(frozen, models[arm], old_dim=54,
                new_dim=58, learning_rate=float(archive["learning_rate"]))
            entries[f"{arm}/{seed}"] = {"model": str(models[arm].relative_to(root)),
                "config": str((ws / "config.json").relative_to(root)), "conversion": conversion}
        checks[str(seed)] = {
            "existing_numerical_verifier": verify_converted_model_equivalence(frozen, models, cfg),
            "exact_rollout": exact_rollout_equivalence(frozen, models, cfg)}
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    (root / "dirty.patch").write_bytes(patch)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    untracked_hashes = {p: sha256_file(Path(p)) for p in untracked.decode().split("\0")
                        if p and Path(p).is_file()}
    historical = None
    if baseline is not None:
        b = json.loads(baseline.read_text(encoding="utf-8"))
        if b["checkpoint_id"] != meta["checkpoint_id"]:
            raise ValueError("Baseline references a different checkpoint")
        shutil.copy2(baseline, root / "historical_development_baseline.json")
        historical = {"sha256": sha256_file(baseline), "success_rate": b["success_rate"],
                      "fresh_pretraining_baseline_required": True}
    manifest = {"protocol": PROTOCOL, "status": "prepared_verified_not_trained",
        "hypothesis": "Explicit gate location and outward normal improve recovery versus zero appended inputs",
        "assistance": "guided; existing parent physics assistance retained",
        "observation_schema_version": GATE_OBSERVATION_SCHEMA_VERSION,
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "parent": {"sha256": expected_sha256, "checkpoint_id": meta["checkpoint_id"],
                   "model": str(frozen.relative_to(root)), "metadata": "parent/metadata.json"},
        "archive_hyperparameters": {k: archive[k] for k in
            ("gamma", "gae_lambda", "ent_coef", "learning_rate", "n_steps", "batch_size", "n_epochs")},
        "compatibility": "New explicit 58-input modes; original weights/biases preserved; new columns zero; both Adam states reset. Approximate continuation, no old checkpoint reinterpretation.",
        "training_seeds": SEEDS, "development_seeds": DEV_SEEDS, "held_out": "unused",
        "proposed_budget_only": {"transitions_per_arm_seed": 12288, "total_transitions": 73728,
            "maximum_eval_episodes": 70, "arm_train_seconds": 120, "overall_seconds": 900},
        "sampling": {"target_ratio": 0.25, "fixed_same_weights_both_arms": {
            "wall_stray_extraction": 1.0, "high_pen_standoff": 1.0, "gate_wall_flock": 1.0}},
        "historical_parent_baseline": historical, "entries": entries, "equivalence": checks,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "untracked_hashes": untracked_hashes, "versions": collect_dependency_versions()}
    manifest["checksums"] = {str(p.relative_to(root)): sha256_file(p)
                             for p in root.rglob("*") if p.is_file()}
    atomic_write_json(root / "manifest.json", manifest)
    verify(root)
    return manifest


def verify(root: Path) -> dict:
    """Reject changed models/configs/parent and confirm arm parity."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL or len(manifest["entries"]) != 6:
        raise ValueError("Unexpected protocol or workspace count")
    for name, expected in manifest["checksums"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Integrity mismatch: {name}")
    parent = manifest["parent"]
    inspect_parent(root / parent["model"], root / parent["metadata"], parent["sha256"])
    for seed in SEEDS:
        configs = []
        for arm, mode in ARMS.items():
            entry = manifest["entries"][f"{arm}/{seed}"]
            cfg = json.loads((root / entry["config"]).read_text(encoding="utf-8"))
            if cfg["training"].pop("observation_mode") != mode:
                raise ValueError("Wrong observation mode")
            for key in ("output_dir", "backup_dir", "web_export_dir"):
                cfg["training"].pop(key)
            configs.append(cfg)
        if configs[0] != configs[1]:
            raise ValueError("Arm configs differ beyond observation mode/output paths")
    return {"status": "verified_not_trained", "workspaces": 6,
            "files_verified": len(manifest["checksums"])}
