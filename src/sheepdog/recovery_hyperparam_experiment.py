"""Controlled hyperparameter experiment for three-dog recovery from unproductive interactions.

Prepares, verifies, and executes isolated three-arm comparisons from a frozen
Stage 34 parent checkpoint:
  - Arm A (Control): unchanged baseline (entropy_coef=0.01, gamma=0.99)
  - Arm B (Entropy-only): increased exploration (entropy_coef=0.02, gamma=0.99)
  - Arm C (Gamma-only): longer-horizon credit assignment (entropy_coef=0.01, gamma=0.997)

All non-treatment settings (rewards, physics, guidance, observations, network
architecture, curriculum stage, termination rules, learning rate, and paired
training/evaluation seeds) are locked identically across arms.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import comb, log, sqrt
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.entities import Pen, Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.scenarios import AgentLayout, PenLayout, SavedScenario
from sheepdog.observations import EmergentObservationBuilder, RoleAwareObservationBuilder
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.diagnostics.repetition_recovery import (
    analyze_repetition_and_recovery,
)
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.replay_writer import (
    CapturePolicy,
    get_global_capture_policy,
    set_global_capture_policy,
)
from sheepdog.training.training_scenarios import get_scenario_builder

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "stage34_recovery_entropy_gamma_v1"

DOGS_PER_TEAM = 3
SHEEP_PER_FLOCK = 6
TRANSITIONS_PER_WORLD_STEP = DOGS_PER_TEAM
DISCOUNT_TIMEBASE = "sequential_shared_dog"
EPISODE_BUDGET_WORLD_STEPS = 2_320
EPISODE_BUDGET_POLICY_TRANSITIONS = EPISODE_BUDGET_WORLD_STEPS * TRANSITIONS_PER_WORLD_STEP

DEFAULT_LATEST_PARENT_CHECKPOINT = Path(
    "artifacts/checkpoints/stage34/maskable-ppo-stage34-331063991.zip"
)
DEFAULT_LATEST_PARENT_METADATA = Path("artifacts/checkpoints/checkpoint-039961.json")
ALTERNATE_BEST_REWARD_CHECKPOINT = Path(
    "artifacts/checkpoints/stage34/stage34-best-model.zip"
)

DEFAULT_PAIRED_TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303)
DEVELOPMENT_EVALUATION_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
HELD_OUT_EVALUATION_SEEDS: tuple[int, ...] = tuple(
    200_003 + index * 7_919 for index in range(30)
)

# Proposed bounded pilot budget:
# 90,112 policy transitions per arm per seed = 44 PPO rollouts of 2,048 transitions
# = 30,037.33 world steps (~13 full 2,320-step episodes per seed), divided into
# 4 checkpoints of 22,528 transitions (11 rollouts = 7,509.33 world steps each).
PROPOSED_PILOT_TRANSITIONS_PER_ARM_SEED = 90_112
PROPOSED_PILOT_CHECKPOINT_COUNT = 4
PROPOSED_PILOT_ROLLOUT_STEPS = 2_048


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Single-factor treatment specification for one experiment arm."""

    arm_id: str
    label: str
    hypothesis: str
    entropy_coef: float
    gamma: float
    is_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize arm specification to a dictionary."""
        return asdict(self)


EXPERIMENT_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        arm_id="arm_a_control",
        label="Arm A — Unchanged Control (entropy_coef=0.01, gamma=0.99)",
        hypothesis="Reference baseline preserving parent Stage 34 entropy and discount settings.",
        entropy_coef=0.01,
        gamma=0.99,
        is_control=True,
    ),
    ArmSpec(
        arm_id="arm_b_entropy_002",
        label="Arm B — Entropy-Only (entropy_coef=0.02, gamma=0.99)",
        hypothesis=(
            "Higher policy entropy (0.02 vs 0.01) increases action diversity during "
            "repeated dog-sheep loops and wall/gate stalls so dogs discover loop-breaking "
            "repositioning maneuvers."
        ),
        entropy_coef=0.02,
        gamma=0.99,
        is_control=False,
    ),
    ArmSpec(
        arm_id="arm_c_gamma_0997",
        label="Arm C — Gamma-Only (entropy_coef=0.01, gamma=0.997)",
        hypothesis=(
            "Longer discount horizon (gamma=0.997 vs 0.99 under sequential_shared_dog "
            "timebase, increasing world-step credit half-life from ~23.0 to ~76.9 world steps) "
            "values delayed penning rewards enough to justify temporary retreat/detour maneuvers."
        ),
        entropy_coef=0.01,
        gamma=0.997,
        is_control=False,
    ),
)


@dataclass(frozen=True, slots=True)
class TransitionBudget:
    """Explicit budget representation in both policy transitions and world steps."""

    policy_transitions_per_arm_seed: int
    world_steps_per_arm_seed: float
    whole_world_steps_per_arm_seed: int
    rollout_steps: int
    rollouts_per_arm_seed: int
    checkpoint_count: int
    checkpoint_policy_transitions: tuple[int, ...]
    checkpoint_world_steps: tuple[float, ...]
    full_episode_equivalents_per_arm_seed: float
    num_arms: int
    num_seeds: int
    total_policy_transitions_all_arms_seeds: int
    total_world_steps_all_arms_seeds: float
    discount_timebase: str = DISCOUNT_TIMEBASE
    transitions_per_world_step: int = TRANSITIONS_PER_WORLD_STEP
    episode_budget_world_steps: int = EPISODE_BUDGET_WORLD_STEPS
    episode_budget_policy_transitions: int = EPISODE_BUDGET_POLICY_TRANSITIONS

    def to_dict(self) -> dict[str, Any]:
        """Serialize arm specification to a dictionary."""
        return asdict(self)


def build_transition_budget(
    *,
    policy_transitions_per_arm_seed: int = PROPOSED_PILOT_TRANSITIONS_PER_ARM_SEED,
    checkpoint_count: int = PROPOSED_PILOT_CHECKPOINT_COUNT,
    rollout_steps: int = PROPOSED_PILOT_ROLLOUT_STEPS,
    num_arms: int = len(EXPERIMENT_ARMS),
    num_seeds: int = len(DEFAULT_PAIRED_TRAINING_SEEDS),
) -> TransitionBudget:
    """Construct and validate a transition/world-step budget."""
    if policy_transitions_per_arm_seed <= 0:
        raise ValueError("policy_transitions_per_arm_seed must be positive.")
    if checkpoint_count <= 0:
        raise ValueError("checkpoint_count must be positive.")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if policy_transitions_per_arm_seed % checkpoint_count != 0:
        raise ValueError(
            f"policy_transitions_per_arm_seed ({policy_transitions_per_arm_seed}) "
            f"must be divisible by checkpoint_count ({checkpoint_count})."
        )
    segment_transitions = policy_transitions_per_arm_seed // checkpoint_count
    if segment_transitions % rollout_steps != 0:
        raise ValueError(
            f"Per-checkpoint transitions ({segment_transitions}) must be an exact "
            f"multiple of rollout_steps ({rollout_steps}) so no partial rollout buffer "
            f"is truncated across checkpoints."
        )
    checkpoint_transitions = tuple(
        segment_transitions * idx for idx in range(1, checkpoint_count + 1)
    )
    checkpoint_world = tuple(
        round(steps / TRANSITIONS_PER_WORLD_STEP, 4) for steps in checkpoint_transitions
    )
    world_steps = policy_transitions_per_arm_seed / TRANSITIONS_PER_WORLD_STEP
    total_runs = num_arms * num_seeds
    return TransitionBudget(
        policy_transitions_per_arm_seed=int(policy_transitions_per_arm_seed),
        world_steps_per_arm_seed=round(world_steps, 4),
        whole_world_steps_per_arm_seed=int(policy_transitions_per_arm_seed // TRANSITIONS_PER_WORLD_STEP),
        rollout_steps=int(rollout_steps),
        rollouts_per_arm_seed=int(policy_transitions_per_arm_seed // rollout_steps),
        checkpoint_count=int(checkpoint_count),
        checkpoint_policy_transitions=checkpoint_transitions,
        checkpoint_world_steps=checkpoint_world,
        full_episode_equivalents_per_arm_seed=round(
            policy_transitions_per_arm_seed / EPISODE_BUDGET_POLICY_TRANSITIONS,
            4,
        ),
        num_arms=int(num_arms),
        num_seeds=int(num_seeds),
        total_policy_transitions_all_arms_seeds=int(
            policy_transitions_per_arm_seed * total_runs
        ),
        total_world_steps_all_arms_seeds=round(world_steps * total_runs, 4),
    )


def discount_horizon_summary(gamma: float) -> dict[str, float]:
    """Compute transition and world-step credit horizon metrics under sequential_shared_dog."""
    gamma_f = float(gamma)
    per_world_step_gamma = gamma_f**TRANSITIONS_PER_WORLD_STEP
    half_life_transitions = log(0.5) / log(gamma_f)
    half_life_world_steps = half_life_transitions / TRANSITIONS_PER_WORLD_STEP
    return {
        "gamma_per_policy_transition": gamma_f,
        "effective_gamma_per_world_step": round(per_world_step_gamma, 6),
        "same_world_step_dog0_discount": round(gamma_f ** (DOGS_PER_TEAM - 1), 6),
        "half_life_policy_transitions": round(half_life_transitions, 2),
        "half_life_world_steps": round(half_life_world_steps, 2),
        "weight_at_50_world_steps": round(gamma_f ** (50 * TRANSITIONS_PER_WORLD_STEP), 6),
        "weight_at_100_world_steps": round(gamma_f ** (100 * TRANSITIONS_PER_WORLD_STEP), 6),
        "weight_at_250_world_steps": round(gamma_f ** (250 * TRANSITIONS_PER_WORLD_STEP), 6),
    }


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lab_config(path: str | Path) -> LabConfig:
    """Load a LabConfig from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("training"), dict) and "checkpoint_timesteps" in payload["training"]:
        payload["training"]["checkpoint_timesteps"] = tuple(
            int(v) for v in payload["training"]["checkpoint_timesteps"]
        )
    return LabConfig.from_dict(payload)


def save_lab_config(config: LabConfig, path: str | Path) -> Path:
    """Save a LabConfig to a JSON file."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return resolved


def collect_git_provenance(repo_root: str | Path = Path(".")) -> dict[str, Any]:
    """Record current Git commit, branch, status, untracked files, and exact dirty-worktree diff checksum."""
    root = Path(repo_root).resolve()

    def _run_git_bytes(*args: str) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            err_msg = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Git command 'git {' '.join(args)}' failed with exit code {result.returncode}: {err_msg}"
            )
        return result.stdout

    head_sha = _run_git_bytes("rev-parse", "HEAD").decode("utf-8", errors="replace").strip()
    branch = _run_git_bytes("rev-parse", "--abbrev-ref", "HEAD").decode("utf-8", errors="replace").strip()
    status_raw = _run_git_bytes("status", "--porcelain")
    status_text = status_raw.decode("utf-8", errors="replace")
    diff_stat = _run_git_bytes("diff", "--stat").decode("utf-8", errors="replace").strip()

    # Exact patch bytes hashed directly with SHA-256 (no Windows ANSI/cp1252 decoding)
    raw_head_diff = _run_git_bytes("diff", "HEAD")
    head_diff_sha256 = hashlib.sha256(raw_head_diff).hexdigest()

    dirty_lines = [line.strip() for line in status_text.splitlines() if line.strip()]

    # Collect untracked implementation and configuration files with their paths and SHA-256
    untracked_records: list[dict[str, str]] = []
    for line in status_text.splitlines():
        if line.startswith("?? "):
            rel_str = line[3:].strip().strip('"')
            file_path = root / rel_str
            if file_path.is_file():
                untracked_records.append(
                    {
                        "path": rel_str.replace("\\", "/"),
                        "sha256": sha256_file(file_path),
                    }
                )

    return {
        "head_commit": head_sha,
        "branch": branch,
        "is_dirty": bool(dirty_lines),
        "dirty_file_count": len(dirty_lines),
        "dirty_files": dirty_lines,
        "untracked_file_count": len(untracked_records),
        "untracked_files": untracked_records,
        "diff_stat": diff_stat,
        "head_diff_sha256": head_diff_sha256,
        "raw_head_diff_bytes_count": len(raw_head_diff),
    }


def collect_dependency_versions() -> dict[str, str]:
    """Record exact Python and RL package versions."""
    packages = (
        "torch",
        "stable-baselines3",
        "sb3-contrib",
        "gymnasium",
        "numpy",
    )
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def inspect_checkpoint_archive(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path = Path("artifacts/effective-training-config.json"),
    training_state_path: str | Path = Path("artifacts/training-state.json"),
    checkpoint_metadata_path: str | Path | None = DEFAULT_LATEST_PARENT_METADATA,
) -> dict[str, Any]:
    """Inspect a saved SB3 checkpoint archive and compare with saved config/state."""
    resolved_ckpt = Path(checkpoint_path)
    if not resolved_ckpt.exists():
        raise FileNotFoundError(f"Parent checkpoint not found: {resolved_ckpt}")

    ckpt_sha256 = sha256_file(resolved_ckpt)
    ckpt_size = resolved_ckpt.stat().st_size

    config = load_lab_config(config_path) if Path(config_path).exists() else LabConfig()
    policy = NeuralPolicy.load(resolved_ckpt, config)
    live_params = policy.live_training_parameters()
    vec_env = policy.model.get_env()
    if vec_env is not None:
        vec_env.close()

    training_state: dict[str, Any] = {}
    if Path(training_state_path).exists():
        training_state = json.loads(Path(training_state_path).read_text(encoding="utf-8"))

    checkpoint_metadata: dict[str, Any] = {}
    if checkpoint_metadata_path is not None and Path(checkpoint_metadata_path).exists():
        checkpoint_metadata = json.loads(
            Path(checkpoint_metadata_path).read_text(encoding="utf-8")
        )

    adaptive_state = training_state.get("adaptive_step_state") or {}
    saved_cfg_training = config.to_dict().get("training", {})

    discrepancies: list[str] = []
    cfg_lr = float(saved_cfg_training.get("learning_rate", 0.0))
    live_lr = float(live_params.get("learning_rate") or 0.0)
    if abs(cfg_lr - live_lr) > 1e-10:
        discrepancies.append(
            f"effective-training-config.json has base learning_rate={cfg_lr:.6g}, "
            f"whereas the checkpoint archive and optimizer store live learning_rate={live_lr:.6g} "
            f"(AdaptiveLearningController stage={adaptive_state.get('stage')} multiplier={adaptive_state.get('lr_multiplier')})."
        )
    best_model_str = training_state.get("best_model_path")
    policy_state_str = training_state.get("policy_state_path")
    if best_model_str and policy_state_str and str(best_model_str) != str(policy_state_str):
        discrepancies.append(
            f"training-state.json points best_model_path to '{best_model_str}' "
            f"(earlier best-reward snapshot at episode 38933, lr=8.0e-5) while "
            f"policy_state_path points to '{policy_state_str}' "
            f"(latest checkpoint 39961, lr=6.5e-5)."
        )

    return {
        "checkpoint_path": str(resolved_ckpt),
        "checkpoint_sha256": ckpt_sha256,
        "checkpoint_size_bytes": ckpt_size,
        "loaded_model_parameters": live_params,
        "checkpoint_metadata_episode": checkpoint_metadata.get("checkpoint_episode"),
        "checkpoint_metadata_policy_version": checkpoint_metadata.get("policy_version"),
        "checkpoint_metadata_curriculum_stage": checkpoint_metadata.get("curriculum_stage"),
        "checkpoint_metadata_success_rate": checkpoint_metadata.get("success_rate"),
        "checkpoint_metadata_average_reward": checkpoint_metadata.get("average_reward"),
        "saved_effective_training_config": {
            "entropy_coef": saved_cfg_training.get("entropy_coef"),
            "gamma": saved_cfg_training.get("gamma"),
            "gae_lambda": saved_cfg_training.get("gae_lambda"),
            "learning_rate": saved_cfg_training.get("learning_rate"),
            "learning_rate_final": saved_cfg_training.get("learning_rate_final"),
            "batch_size": saved_cfg_training.get("batch_size"),
            "rollout_steps": saved_cfg_training.get("rollout_steps"),
            "target_kl": saved_cfg_training.get("target_kl"),
            "value_coef": saved_cfg_training.get("value_coef"),
            "enable_adaptive_learning": saved_cfg_training.get("enable_adaptive_learning"),
            "failure_directed_training_enabled": saved_cfg_training.get(
                "failure_directed_training_enabled"
            ),
            "auto_rollback_on_collapse": saved_cfg_training.get("auto_rollback_on_collapse"),
        },
        "training_state_adaptive_step_state": adaptive_state,
        "verified_entropy_coef_matches_assumed_001": abs(
            float(live_params.get("ent_coef") or 0.0) - 0.01
        )
        < 1e-9,
        "verified_gamma_matches_assumed_099": abs(
            float(live_params.get("gamma") or 0.0) - 0.99
        )
        < 1e-9,
        "discrepancies_detected": discrepancies,
    }


def build_arm_config(
    base_config: LabConfig,
    arm: ArmSpec,
    *,
    arm_output_dir: str | Path,
    train_seed: int,
    locked_learning_rate: float,
    budget: TransitionBudget,
    evaluation_seeds: tuple[int, ...] = DEVELOPMENT_EVALUATION_SEEDS,
) -> LabConfig:
    """Create a locked, single-factor LabConfig for one arm and training seed."""
    payload = copy.deepcopy(base_config.to_dict())
    train_dict = payload["training"]

    # Single-factor treatment settings:
    train_dict["entropy_coef"] = float(arm.entropy_coef)
    train_dict["gamma"] = float(arm.gamma)

    # Paired seed and isolated output directory:
    train_dict["train_seed"] = int(train_seed)
    train_dict["output_dir"] = str(Path(arm_output_dir))
    train_dict["rollout_steps"] = int(budget.rollout_steps)
    train_dict["total_timesteps"] = int(budget.policy_transitions_per_arm_seed)
    train_dict["checkpoint_timesteps"] = tuple(budget.checkpoint_policy_transitions)
    train_dict["checkpoint_episodes"] = tuple(range(1, budget.checkpoint_count + 1))
    train_dict["episodes"] = int(budget.checkpoint_count)
    train_dict["evaluation_seeds"] = tuple(int(s) for s in evaluation_seeds)
    train_dict["quick_evaluation_seed_count"] = len(evaluation_seeds)

    # Lock non-treatment schedules and dynamic overrides across all arms:
    # 1. Freeze learning rate to the parent checkpoint's live optimizer LR so
    #    Control Arm A is genuinely unchanged and no mid-run adaptive stage
    #    transition can damp entropy or alter LR differently across arms.
    train_dict["learning_rate"] = float(locked_learning_rate)
    train_dict["learning_rate_final"] = float(locked_learning_rate)
    train_dict["enable_adaptive_learning"] = False

    # 2. Freeze scenario distribution and curriculum promotion so checkpoint
    #    evaluation outcomes cannot alter training scenario sampling mid-run.
    train_dict["failure_directed_training_enabled"] = False
    train_dict["scenario_training_enabled"] = False
    train_dict["auto_rollback_on_collapse"] = False
    train_dict["wandb_enabled"] = False

    return LabConfig.from_dict(payload)


def verify_single_parameter_arm_diffs(
    arm_configs: dict[str, LabConfig],
) -> dict[str, Any]:
    """Verify that Arms A, B, and C differ ONLY in their intended hyperparameter and output_dir."""
    required_ids = {"arm_a_control", "arm_b_entropy_002", "arm_c_gamma_0997"}
    if set(arm_configs.keys()) != required_ids:
        raise ValueError(f"Expected arm configs {sorted(required_ids)}, got {sorted(arm_configs.keys())}")

    control_dict = arm_configs["arm_a_control"].to_dict()

    def _diff_dicts(left: dict[str, Any], right: dict[str, Any], prefix: str = "") -> dict[str, tuple[Any, Any]]:
        diffs: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(left.keys()) | set(right.keys())):
            path = f"{prefix}.{key}" if prefix else key
            l_val = left.get(key)
            r_val = right.get(key)
            if isinstance(l_val, dict) and isinstance(r_val, dict):
                diffs.update(_diff_dicts(l_val, r_val, path))
            elif l_val != r_val:
                diffs[path] = (l_val, r_val)
        return diffs

    b_diffs = _diff_dicts(control_dict, arm_configs["arm_b_entropy_002"].to_dict())
    c_diffs = _diff_dicts(control_dict, arm_configs["arm_c_gamma_0997"].to_dict())

    b_non_path_diffs = {k: v for k, v in b_diffs.items() if k != "training.output_dir"}
    c_non_path_diffs = {k: v for k, v in c_diffs.items() if k != "training.output_dir"}

    if set(b_non_path_diffs.keys()) != {"training.entropy_coef"}:
        raise ValueError(
            f"Arm B differs from Control Arm A in unexpected keys: {b_non_path_diffs}"
        )
    if set(c_non_path_diffs.keys()) != {"training.gamma"}:
        raise ValueError(
            f"Arm C differs from Control Arm A in unexpected keys: {c_non_path_diffs}"
        )

    return {
        "verified": True,
        "arm_b_vs_control_diffs": {
            k: {"control": v[0], "arm_b": v[1]} for k, v in b_non_path_diffs.items()
        },
        "arm_c_vs_control_diffs": {
            k: {"control": v[0], "arm_c": v[1]} for k, v in c_non_path_diffs.items()
        },
    }


def _make_saved_scenario(
    *,
    name: str,
    seed: int,
    pen: Pen,
    dog_positions: tuple[Point, ...],
    sheep_positions: tuple[Point, ...],
    grid_width: int = 51,
    grid_height: int = 31,
) -> SavedScenario:
    """Construct a SavedScenario from Pen and Point tuples."""
    return SavedScenario(
        id=name,
        name=name,
        created_at="2026-11-06T00:00:00+00:00",
        seed=int(seed),
        width=int(grid_width),
        height=int(grid_height),
        dogs=tuple(
            AgentLayout(index=idx, x=int(pt.x), y=int(pt.y))
            for idx, pt in enumerate(dog_positions)
        ),
        sheep=tuple(
            AgentLayout(index=idx, x=int(pt.x), y=int(pt.y), personality="obedient")
            for idx, pt in enumerate(sheep_positions)
        ),
        pen=PenLayout(
            origin_x=int(pen.origin.x),
            origin_y=int(pen.origin.y),
            width=int(pen.width),
            height=int(pen.height),
            opening=str(pen.opening),
        ),
        description=f"Controlled recovery scenario: {name}",
    )


def _scenario_initial_unpenned_count(scenario: SavedScenario) -> int:
    """Return the number of sheep starting outside the pen in a SavedScenario."""
    pen = Pen(
        Point(scenario.pen.origin_x, scenario.pen.origin_y),
        scenario.pen.width,
        scenario.pen.height,
        opening=scenario.pen.opening,
    )
    return sum(1 for s in scenario.sheep if not pen.contains(Point(s.x, s.y)))


def get_development_recovery_scenarios() -> dict[str, SavedScenario]:
    """Return development scenarios covering fences, walls/corners, split flocks, interference, and 1..6 unpenned sheep."""
    base_cfg = LabConfig()
    scenarios: dict[str, SavedScenario] = {}
    dev_archetypes = (
        "corner_huddle",
        "wall_stray_extraction",
        "gate_wall_flock",
        "split_flock",
        "subpen_flock",
        "high_pen_standoff",
    )
    for idx, sc_type in enumerate(dev_archetypes):
        sc_seed = 105_001 + idx
        builder = get_scenario_builder(sc_type)
        scenarios[f"dev_{sc_type}"] = builder(
            seed=sc_seed,
            config=base_cfg.environment,
        )

    # Standard 51x31 Stage 34 field with pen at (39, 11), width=10, height=9, opening="left".
    # Pen interior cells: x in [39..48], y in [11..19] (penned_sheep placed at x=44..47, y=13..18).
    pen = Pen(origin=Point(39, 11), width=10, height=9, opening="left")
    penned_slots: tuple[Point, ...] = (
        Point(46, 14),
        Point(46, 16),
        Point(45, 14),
        Point(45, 16),
        Point(44, 15),
    )

    unpenned_templates: tuple[tuple[str, int, tuple[Point, ...], tuple[Point, ...]], ...] = (
        (
            "dev_unpenned_1_gate_wing_fence",
            110_001,
            (Point(40, 9),),  # 1 unpenned sheep trapped against top pen fence wing
            (Point(37, 9), Point(38, 13), Point(35, 15)),
        ),
        (
            "dev_unpenned_2_corner_wall_pin",
            110_002,
            (Point(1, 1), Point(2, 1)),  # 2 unpenned sheep pinned in top-left corner
            (Point(4, 3), Point(5, 2), Point(6, 5)),
        ),
        (
            "dev_unpenned_3_blocked_gate_corridor",
            110_003,
            (Point(34, 14), Point(34, 16), Point(33, 15)),  # 3 unpenned sheep outside gate
            (Point(37, 15), Point(37, 14), Point(30, 15)),  # Dogs blocking gate corridor
        ),
        (
            "dev_unpenned_4_split_flock_stragglers",
            110_004,
            (Point(31, 14), Point(31, 16), Point(8, 6), Point(8, 25)),  # 4 unpenned split
            (Point(27, 15), Point(20, 10), Point(20, 20)),
        ),
        (
            "dev_unpenned_5_subpen_dog_interference",
            110_005,
            (Point(42, 24), Point(44, 24), Point(46, 24), Point(43, 26), Point(45, 26)),
            (Point(41, 22), Point(42, 22), Point(43, 22)),  # 3 dogs clustered below pen fence
        ),
        (
            "dev_unpenned_6_full_flock_wall_standoff",
            110_006,
            (
                Point(24, 1),
                Point(26, 1),
                Point(28, 1),
                Point(25, 2),
                Point(27, 2),
                Point(29, 2),
            ),
            (Point(25, 5), Point(27, 5), Point(29, 5)),
        ),
    )

    for name, seed, unpenned_pts, dog_pts in unpenned_templates:
        unpenned_count = len(unpenned_pts)
        penned_needed = SHEEP_PER_FLOCK - unpenned_count
        sheep_positions = tuple(unpenned_pts) + penned_slots[:penned_needed]
        scenarios[name] = _make_saved_scenario(
            name=name,
            seed=seed,
            pen=pen,
            dog_positions=dog_pts,
            sheep_positions=sheep_positions,
        )

    return scenarios


def get_held_out_recovery_scenarios() -> dict[str, SavedScenario]:
    """Return held-out scenarios strictly isolated from training and development tuning."""
    pen_right = Pen(origin=Point(39, 11), width=10, height=9, opening="left")
    pen_top = Pen(origin=Point(20, 1), width=10, height=9, opening="bottom")
    penned_right: tuple[Point, ...] = (
        Point(46, 14),
        Point(46, 16),
        Point(45, 14),
        Point(45, 16),
        Point(44, 15),
    )
    penned_top: tuple[Point, ...] = (
        Point(23, 3),
        Point(25, 3),
        Point(27, 3),
        Point(24, 4),
        Point(26, 4),
    )

    specs: tuple[tuple[str, int, Pen, tuple[Point, ...], tuple[Point, ...], tuple[Point, ...]], ...] = (
        # Unpenned = 1
        (
            "heldout_unpenned_1_south_fence_wing",
            300_001,
            pen_right,
            penned_right,
            (Point(41, 21),),
            (Point(38, 21), Point(37, 17), Point(35, 15)),
        ),
        (
            "heldout_unpenned_1_bottom_right_corner",
            300_002,
            pen_right,
            penned_right,
            (Point(49, 29),),
            (Point(46, 27), Point(47, 26), Point(44, 28)),
        ),
        # Unpenned = 2
        (
            "heldout_unpenned_2_bottom_left_corner_trap",
            300_003,
            pen_right,
            penned_right,
            (Point(1, 29), Point(2, 29)),
            (Point(4, 27), Point(5, 28), Point(6, 26)),
        ),
        (
            "heldout_unpenned_2_top_pen_wing_pin",
            300_004,
            pen_top,
            penned_top,
            (Point(18, 3), Point(31, 3)),
            (Point(18, 6), Point(31, 6), Point(25, 12)),
        ),
        # Unpenned = 3
        (
            "heldout_unpenned_3_dog_blocked_approach",
            300_005,
            pen_right,
            penned_right,
            (Point(33, 13), Point(33, 15), Point(33, 17)),
            (Point(37, 14), Point(37, 16), Point(36, 15)),
        ),
        (
            "heldout_unpenned_3_left_wall_pin",
            300_006,
            pen_right,
            penned_right,
            (Point(0, 12), Point(0, 14), Point(1, 13)),
            (Point(3, 12), Point(3, 14), Point(4, 13)),
        ),
        # Unpenned = 4
        (
            "heldout_unpenned_4_diagonal_split_flock",
            300_007,
            pen_right,
            penned_right,
            (Point(7, 5), Point(9, 6), Point(7, 25), Point(9, 24)),
            (Point(16, 14), Point(16, 15), Point(16, 16)),
        ),
        (
            "heldout_unpenned_4_above_pen_roof_trap",
            300_008,
            pen_right,
            penned_right,
            (Point(41, 7), Point(43, 7), Point(45, 7), Point(44, 6)),
            (Point(39, 8), Point(40, 8), Point(37, 10)),
        ),
        # Unpenned = 5
        (
            "heldout_unpenned_5_gate_corridor_jam",
            300_009,
            pen_right,
            penned_right,
            (Point(34, 13), Point(34, 15), Point(34, 17), Point(32, 14), Point(32, 16)),
            (Point(37, 15), Point(36, 13), Point(36, 17)),
        ),
        (
            "heldout_unpenned_5_top_pen_split_strays",
            300_010,
            pen_top,
            penned_top,
            (Point(23, 14), Point(25, 14), Point(27, 14), Point(6, 25), Point(44, 25)),
            (Point(25, 18), Point(18, 20), Point(32, 20)),
        ),
        # Unpenned = 6
        (
            "heldout_unpenned_6_bottom_wall_standoff",
            300_011,
            pen_right,
            penned_right,
            (
                Point(20, 29),
                Point(22, 29),
                Point(24, 29),
                Point(21, 28),
                Point(23, 28),
                Point(25, 28),
            ),
            (Point(21, 25), Point(23, 25), Point(25, 25)),
        ),
        (
            "heldout_unpenned_6_three_cluster_scatter",
            300_012,
            pen_right,
            penned_right,
            (
                Point(6, 6),
                Point(8, 7),
                Point(6, 24),
                Point(8, 23),
                Point(28, 4),
                Point(29, 26),
            ),
            (Point(18, 13), Point(18, 15), Point(18, 17)),
        ),
    )

    scenarios: dict[str, SavedScenario] = {}
    for name, seed, pen, penned_pool, unpenned_pts, dog_pts in specs:
        penned_needed = SHEEP_PER_FLOCK - len(unpenned_pts)
        sheep_positions = tuple(unpenned_pts) + penned_pool[:penned_needed]
        scenarios[name] = _make_saved_scenario(
            name=name,
            seed=seed,
            pen=pen,
            dog_positions=dog_pts,
            sheep_positions=sheep_positions,
        )
    return scenarios


def verify_seed_and_scenario_isolation(
    training_seeds: tuple[int, ...] = DEFAULT_PAIRED_TRAINING_SEEDS,
) -> dict[str, Any]:
    """Verify training, development, and held-out seed/scenario sets are strictly disjoint."""
    dev_scenarios = get_development_recovery_scenarios()
    held_out_scenarios = get_held_out_recovery_scenarios()

    train_set = set(int(s) for s in training_seeds)
    dev_seed_set = set(DEVELOPMENT_EVALUATION_SEEDS) | {
        int(sc.seed) for sc in dev_scenarios.values()
    }
    held_out_seed_set = set(HELD_OUT_EVALUATION_SEEDS) | {
        int(sc.seed) for sc in held_out_scenarios.values()
    }

    train_dev_overlap = sorted(train_set & dev_seed_set)
    train_heldout_overlap = sorted(train_set & held_out_seed_set)
    dev_heldout_overlap = sorted(dev_seed_set & held_out_seed_set)
    scenario_name_overlap = sorted(set(dev_scenarios.keys()) & set(held_out_scenarios.keys()))

    if train_dev_overlap or train_heldout_overlap or dev_heldout_overlap or scenario_name_overlap:
        raise ValueError(
            f"Seed/scenario overlap detected: train_dev={train_dev_overlap}, "
            f"train_heldout={train_heldout_overlap}, dev_heldout={dev_heldout_overlap}, "
            f"scenario_names={scenario_name_overlap}"
        )

    dev_unpenned_counts = sorted(
        {_scenario_initial_unpenned_count(sc) for sc in dev_scenarios.values()}
    )
    held_out_unpenned_counts = sorted(
        {_scenario_initial_unpenned_count(sc) for sc in held_out_scenarios.values()}
    )
    if dev_unpenned_counts != [1, 2, 3, 4, 5, 6]:
        raise ValueError(
            f"Development scenarios must cover all unpenned sheep counts 1..6, got {dev_unpenned_counts}"
        )
    if held_out_unpenned_counts != [1, 2, 3, 4, 5, 6]:
        raise ValueError(
            f"Held-out scenarios must cover all unpenned sheep counts 1..6, got {held_out_unpenned_counts}"
        )

    return {
        "verified": True,
        "training_seeds": list(training_seeds),
        "development_seed_count": len(DEVELOPMENT_EVALUATION_SEEDS),
        "development_scenario_count": len(dev_scenarios),
        "development_unpenned_counts_covered": dev_unpenned_counts,
        "held_out_seed_count": len(HELD_OUT_EVALUATION_SEEDS),
        "held_out_scenario_count": len(held_out_scenarios),
        "held_out_unpenned_counts_covered": held_out_unpenned_counts,
    }


def wilson_confidence_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    n = float(total)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (round(max(0.0, center - half_width), 6), round(min(1.0, center + half_width), 6))


def exact_mcnemar_pvalue(control_outcomes: list[bool], candidate_outcomes: list[bool]) -> float:
    """Two-sided exact McNemar p-value for paired binary episode outcomes."""
    if len(control_outcomes) != len(candidate_outcomes):
        raise ValueError("Paired outcome vectors must have equal length.")
    b = sum(1 for c, t in zip(control_outcomes, candidate_outcomes, strict=True) if c and not t)
    c = sum(1 for ctrl, cand in zip(control_outcomes, candidate_outcomes, strict=True) if not ctrl and cand)
    discordant = b + c
    if discordant == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(discordant, i) for i in range(0, k + 1)) * (0.5**discordant)
    return round(min(1.0, 2.0 * tail), 6)


def evaluate_policy_on_recovery_suite(
    policy: NeuralPolicy,
    config: LabConfig,
    *,
    suite: str = "development",
    deterministic: bool = True,
    replay_output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """Evaluate a loaded policy on the development or held-out suite with temporal interaction metrics."""
    if suite == "development":
        seeds = DEVELOPMENT_EVALUATION_SEEDS
        scenarios = get_development_recovery_scenarios()
    elif suite == "held_out":
        seeds = HELD_OUT_EVALUATION_SEEDS
        scenarios = get_held_out_recovery_scenarios()
    else:
        raise ValueError(f"Unknown evaluation suite '{suite}'. Expected 'development' or 'held_out'.")

    eval_cfg_dict = copy.deepcopy(config.to_dict())
    if max_steps_override is not None:
        eval_cfg_dict["environment"]["max_steps"] = int(max_steps_override)
    eval_config = LabConfig.from_dict(eval_cfg_dict)
    env = SheepdogEnvironment(eval_config)

    decision_mode = "deterministic" if deterministic else "sampled"
    episode_records: list[dict[str, Any]] = []
    unpenned_bucket_totals: dict[str, dict[str, float]] = {
        str(k): {
            "episodes_observed": 0.0,
            "world_steps": 0.0,
            "repeated_movement_steps": 0.0,
            "stall_intervals": 0.0,
            "recovered_intervals": 0.0,
        }
        for k in range(1, SHEEP_PER_FLOCK + 1)
    }

    saved_replays: dict[str, dict[str, Any]] = {}
    replay_dir = Path(replay_output_dir) if replay_output_dir is not None else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    # 1. Seeded procedural episodes
    eval_items: list[tuple[str, str, int, SavedScenario | None]] = [
        (f"seed_{seed}", "seeded_procedural", int(seed), None) for seed in seeds
    ]
    # 2. Named recovery/unpenned-count scenarios
    for sc_name, sc in scenarios.items():
        eval_items.append((sc_name, "recovery_scenario", int(sc.seed), sc))

    for case_id, case_kind, seed, scenario in eval_items:
        if scenario is None:
            case_width = int(eval_config.environment.width)
            case_height = int(eval_config.environment.height)
            initial_snapshot = env.reset(seed=seed)
            initial_unpenned = SHEEP_PER_FLOCK
            case_pen = env.pen
            arena_type = (
                "full_size_training_arena"
                if (case_width == 144 and case_height == 104)
                else "procedural_arena"
            )
            result = env._run_policy_loop(
                policy,
                seed=seed,
                capture_replay=True,
                deterministic=deterministic,
                capture_step_telemetry=False,
            )
        else:
            case_width = int(scenario.width)
            case_height = int(scenario.height)
            initial_snapshot = env.reset_from_scenario(scenario)
            initial_unpenned = _scenario_initial_unpenned_count(scenario)
            case_pen = Pen(
                Point(scenario.pen.origin_x, scenario.pen.origin_y),
                scenario.pen.width,
                scenario.pen.height,
                opening=scenario.pen.opening,
            )
            arena_type = (
                "compact_diagnostic_arena"
                if (case_width == 51 and case_height == 31)
                else "scenario_arena"
            )
            result = env._run_policy_loop(
                policy,
                seed=int(scenario.seed),
                capture_replay=True,
                deterministic=deterministic,
                capture_step_telemetry=False,
            )

        case_geometry = {
            "width": case_width,
            "height": case_height,
            "pen_origin": [int(case_pen.origin.x), int(case_pen.origin.y)],
            "pen_width": int(case_pen.width),
            "pen_height": int(case_pen.height),
            "pen_opening": str(case_pen.opening),
            "arena_type": arena_type,
        }

        # Build complete, valid replay bundle conforming to load_replay_bundle contract
        bundle: dict[str, Any] = {
            "replay_id": f"{suite}_{case_id}_{decision_mode}",
            "seed": int(seed),
            "checkpoint_id": None,
            "run_id": None,
            "policy_version": getattr(policy, "policy_version", None),
            "evaluation_mode": decision_mode,
            "policy_name": result.policy_name,
            "trainer_type": "maskable_ppo",
            "policy_type": getattr(policy, "policy_type", "neural"),
            "policy_mode": "guided",
            "replay_mode": "evaluation",
            "environment": {
                **asdict(eval_config.environment),
                "width": case_width,
                "height": case_height,
                "pen_origin_x": int(case_pen.origin.x),
                "pen_origin_y": int(case_pen.origin.y),
                "pen_width": int(case_pen.width),
                "pen_height": int(case_pen.height),
                "pen_opening": str(case_pen.opening),
                "curriculum_stage": eval_config.rewards.instincts.curriculum_stage,
                "enable_instinct_rewards": eval_config.rewards.instincts.enable_instinct_rewards,
            },
            "strategy_version": TeamStrategy.VERSION,
            "discount_timebase": "sequential_shared_dog",
            "discount_factor": eval_config.training.gamma,
            "checkpoint_gamma": getattr(policy, "checkpoint_gamma", None),
            "evaluation_gamma": eval_config.training.gamma,
            "gamma_mismatch": False,
            "dog_count": int(eval_config.environment.dogs),
            "final_snapshot": result.final_snapshot.to_dict(),
            "initial_snapshot": initial_snapshot.to_dict(),
            "stats": asdict(result.stats),
            "frames": [frame.to_dict() for frame in result.replay],
        }

        analysis = analyze_repetition_and_recovery(
            bundle, dog_count=int(eval_config.environment.dogs)
        )
        temporal = analysis.temporal_analysis
        intervals = list(temporal.intervals) if temporal is not None else []

        counterproductive_steps = sum(
            iv.duration_steps
            for iv in intervals
            if iv.classification == "counterproductive_pressure"
        )
        total_world_steps = int(result.stats.steps)
        ep_repeated_steps = analysis.repeated_movement_steps

        # Accumulate per-unpenned bucket metrics from verified analysis
        observed_unpenned_counts = []
        for k in range(1, SHEEP_PER_FLOCK + 1):
            k_str = str(k)
            b = analysis.by_unpenned_count.get(k_str)
            if b and b.world_steps > 0:
                observed_unpenned_counts.append(k)
                unpenned_bucket_totals[k_str]["episodes_observed"] += 1.0
                unpenned_bucket_totals[k_str]["world_steps"] += float(b.world_steps)
                unpenned_bucket_totals[k_str]["repeated_movement_steps"] += float(
                    b.repeated_movement_steps
                )
                unpenned_bucket_totals[k_str]["stall_intervals"] += float(b.stall_intervals)
                unpenned_bucket_totals[k_str]["recovered_intervals"] += float(b.recovered_intervals)

        ep_record = {
            "case_id": case_id,
            "case_kind": case_kind,
            "seed": int(seed),
            "initial_unpenned_sheep": int(initial_unpenned),
            "full_flock_success": bool(result.stats.success),
            "sheep_penned": int(result.stats.sheep_penned),
            "stop_reason": str(result.stats.stop_reason),
            "timeout": bool(result.stats.timeout),
            "no_progress_stopped": bool(result.stats.stopped),
            "world_steps": total_world_steps,
            "policy_transitions": total_world_steps * TRANSITIONS_PER_WORLD_STEP,
            "reward_total": round(float(result.stats.reward_total), 4),
            "repeated_movement_steps": ep_repeated_steps,
            "repeated_movement_fraction": round(
                ep_repeated_steps / max(1, total_world_steps),
                4,
            ),
            "counterproductive_pressure_steps": counterproductive_steps,
            "stall_interval_count": analysis.stalls_count,
            "recovered_interval_count": analysis.recovered_stalls_count,
            "recovery_outcome_rate": (
                round(analysis.recovery_rate, 4) if analysis.recovery_rate is not None else None
            ),
            "unpenned_counts_observed": observed_unpenned_counts,
            "case_geometry": case_geometry,
        }
        episode_records.append(ep_record)

        # Save one representative success replay and one representative failure replay linked to the checkpoint
        if replay_dir is not None:
            rep_type = "representative_success" if result.stats.success else "representative_failure"
            if rep_type not in saved_replays:
                replay_file = replay_dir / f"{suite}_{rep_type}_{case_id}_{decision_mode}.replay.json"
                replay_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "suite": suite,
                    "decision_mode": decision_mode,
                    "representative_kind": rep_type,
                    "case_id": case_id,
                    "seed": int(seed),
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else str(policy.model_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "episode_summary": ep_record,
                    "steps": [step.to_dict() for step in result.replay],
                }
                replay_file.write_text(json.dumps(replay_payload, indent=2), encoding="utf-8")
                saved_replays[rep_type] = {
                    "case_id": case_id,
                    "seed": int(seed),
                    "replay_path": str(replay_file),
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path else str(policy.model_path),
                    "checkpoint_sha256": checkpoint_sha256,
                }

    total_cases = len(episode_records)
    success_records = [ep for ep in episode_records if ep["full_flock_success"]]
    success_count = len(success_records)
    timeout_count = sum(1 for ep in episode_records if ep["timeout"])
    no_progress_count = sum(1 for ep in episode_records if ep["no_progress_stopped"])
    total_stalls = sum(ep["stall_interval_count"] for ep in episode_records)
    total_recovered = sum(ep["recovered_interval_count"] for ep in episode_records)

    budget_steps = int(eval_config.environment.max_steps)
    return {
        "suite": suite,
        "decision_mode": decision_mode,
        "episode_budget_world_steps": budget_steps,
        "episode_budget_policy_transitions": budget_steps * TRANSITIONS_PER_WORLD_STEP,
        "total_episodes": total_cases,
        "full_flock_success_count": success_count,
        "full_flock_success_rate": round(success_count / max(1, total_cases), 4) if total_cases > 0 else None,
        "full_flock_success_wilson_95ci": (
            list(wilson_confidence_interval(success_count, total_cases))
            if total_cases > 0
            else None
        ),
        "timeout_rate": round(timeout_count / max(1, total_cases), 4) if total_cases > 0 else None,
        "no_progress_stop_rate": round(no_progress_count / max(1, total_cases), 4) if total_cases > 0 else None,
        "steps_to_success_mean": (
            round(
                sum(ep["world_steps"] for ep in success_records) / len(success_records),
                2,
            )
            if success_records
            else None
        ),
        "penalized_steps_mean": (
            round(
                sum(
                    ep["world_steps"] if ep["full_flock_success"] else budget_steps
                    for ep in episode_records
                )
                / max(1, total_cases),
                2,
            )
            if total_cases > 0
            else None
        ),
        "repeated_movement_steps_mean": (
            round(sum(ep["repeated_movement_steps"] for ep in episode_records) / max(1, total_cases), 2)
            if total_cases > 0
            else None
        ),
        "repeated_movement_fraction_mean": (
            round(sum(ep["repeated_movement_fraction"] for ep in episode_records) / max(1, total_cases), 4)
            if total_cases > 0
            else None
        ),
        "counterproductive_pressure_steps_mean": (
            round(sum(ep["counterproductive_pressure_steps"] for ep in episode_records) / max(1, total_cases), 2)
            if total_cases > 0
            else None
        ),
        "total_stall_intervals": int(total_stalls),
        "total_recovered_intervals": int(total_recovered),
        "recovery_outcome_rate": (
            round(total_recovered / total_stalls, 4) if total_stalls > 0 else None
        ),
        "average_reward_diagnostic_only": (
            round(sum(ep["reward_total"] for ep in episode_records) / max(1, total_cases), 4)
            if total_cases > 0
            else None
        ),
        "by_unpenned_sheep_count": {
            k: {
                "episodes_observed": int(vals["episodes_observed"]),
                "world_steps": int(vals["world_steps"]),
                "repeated_movement_steps": int(vals["repeated_movement_steps"]),
                "repeated_movement_fraction": (
                    round(vals["repeated_movement_steps"] / max(1.0, vals["world_steps"]), 4)
                    if vals["world_steps"] > 0
                    else None
                ),
                "stall_intervals": int(vals["stall_intervals"]),
                "recovered_intervals": int(vals["recovered_intervals"]),
                "recovery_rate": (
                    round(vals["recovered_intervals"] / vals["stall_intervals"], 4)
                    if vals["stall_intervals"] > 0
                    else None
                ),
            }
            for k, vals in unpenned_bucket_totals.items()
        },
        "representative_replays": saved_replays,
        "episodes": episode_records,
    }


def evaluate_experiment_promotion_guardrails(
    *,
    parent_sha256_before: str,
    parent_sha256_after: str,
    expected_seeds: tuple[int, ...],
    expected_transitions_per_arm_seed: int,
    arm_seed_results: dict[str, dict[int, dict[str, Any]]],
    suite: str = "development",
    decision_mode: str = "deterministic",
    required_arms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate comparability and ensure incomplete or reward-only gains cannot be reported as improved."""
    reasons: list[str] = []
    if parent_sha256_before != parent_sha256_after:
        reasons.append(
            "Parent checkpoint checksum changed during experiment execution."
        )

    arms_to_check = (
        list(required_arms)
        if required_arms is not None
        else [arm.arm_id for arm in EXPERIMENT_ARMS]
    )
    for arm_id in arms_to_check:
        if arm_id not in arm_seed_results:
            reasons.append(f"Missing results for arm '{arm_id}'.")
            continue
        seed_map = arm_seed_results[arm_id]
        missing_seeds = [s for s in expected_seeds if s not in seed_map]
        if missing_seeds:
            reasons.append(f"Arm '{arm_id}' is missing paired seeds: {missing_seeds}.")
        for seed, run_data in seed_map.items():
            if not run_data.get("completed", False):
                reasons.append(f"Arm '{arm_id}' seed {seed} did not complete training.")
            if run_data.get("consumption", {}).get("stopped_early") or run_data.get("stopped_early"):
                reasons.append(
                    f"Arm '{arm_id}' seed {seed} stopped early ({run_data.get('consumption', {}).get('stop_reason') or run_data.get('stop_reason')}); "
                    "comparisons must identify incomplete or unequal-budget results and cannot declare early stops as improvement."
                )
            trained_transitions = int(run_data.get("trained_policy_transitions", 0))
            if trained_transitions != int(expected_transitions_per_arm_seed):
                reasons.append(
                    f"Arm '{arm_id}' seed {seed} trained {trained_transitions} transitions "
                    f"instead of expected {expected_transitions_per_arm_seed}."
                )
            if not run_data.get("live_parameter_verification_passed", False):
                reasons.append(
                    f"Arm '{arm_id}' seed {seed} failed live trainer parameter verification."
                )
            eval_block = run_data.get("evaluations", {}).get(suite)
            if not isinstance(eval_block, dict):
                reasons.append(f"Arm '{arm_id}' seed {seed} is missing '{suite}' evaluation.")
            elif eval_block.get("decision_mode") != decision_mode:
                reasons.append(
                    f"Arm '{arm_id}' seed {seed} evaluation decision_mode "
                    f"'{eval_block.get('decision_mode')}' != required '{decision_mode}'."
                )

    if reasons:
        return {
            "comparable": False,
            "status": "INVALID_OR_INCOMPLETE",
            "reasons": reasons,
            "arm_comparisons": {},
        }

    # Aggregate across seeds and compare B and C against Control Arm A:
    control_runs = arm_seed_results["arm_a_control"]
    control_outcomes: list[bool] = []
    control_success_rates: list[float] = []
    control_repeated_steps: list[float] = []
    control_recovery_rates: list[float] = []

    for seed in expected_seeds:
        ev = control_runs[seed]["evaluations"][suite]
        control_success_rates.append(float(ev["full_flock_success_rate"]))
        control_repeated_steps.append(float(ev["repeated_movement_steps_mean"]))
        if ev.get("recovery_outcome_rate") is not None:
            control_recovery_rates.append(float(ev["recovery_outcome_rate"]))
        control_outcomes.extend(bool(ep["full_flock_success"]) for ep in ev["episodes"])

    ctrl_mean_sr = sum(control_success_rates) / len(control_success_rates)
    ctrl_mean_rep = sum(control_repeated_steps) / len(control_repeated_steps)
    ctrl_mean_rec = (
        sum(control_recovery_rates) / len(control_recovery_rates)
        if control_recovery_rates
        else None
    )

    arm_comparisons: dict[str, Any] = {}
    for arm in EXPERIMENT_ARMS:
        if arm.is_control or arm.arm_id not in arms_to_check:
            continue
        cand_runs = arm_seed_results[arm.arm_id]
        cand_outcomes: list[bool] = []
        cand_success_rates: list[float] = []
        cand_repeated_steps: list[float] = []
        cand_recovery_rates: list[float] = []
        cand_rewards: list[float] = []

        for seed in expected_seeds:
            ev = cand_runs[seed]["evaluations"][suite]
            cand_success_rates.append(float(ev["full_flock_success_rate"]))
            cand_repeated_steps.append(float(ev["repeated_movement_steps_mean"]))
            if ev.get("recovery_outcome_rate") is not None:
                cand_recovery_rates.append(float(ev["recovery_outcome_rate"]))
            cand_rewards.append(float(ev["average_reward_diagnostic_only"]))
            cand_outcomes.extend(bool(ep["full_flock_success"]) for ep in ev["episodes"])

        cand_mean_sr = sum(cand_success_rates) / len(cand_success_rates)
        cand_mean_rep = sum(cand_repeated_steps) / len(cand_repeated_steps)
        cand_mean_rec = (
            sum(cand_recovery_rates) / len(cand_recovery_rates)
            if cand_recovery_rates
            else None
        )
        mcnemar_p = exact_mcnemar_pvalue(control_outcomes, cand_outcomes)
        pooled_successes = sum(1 for x in cand_outcomes if x)
        wilson_ci = wilson_confidence_interval(pooled_successes, len(cand_outcomes))

        # Strict decision rule: higher reward or earlier termination alone NEVER counts as improvement.
        # Improvement requires higher full-flock success rate AND either reduced repeated-movement
        # duration or higher recovery rate.
        rec_improved = (
            (cand_mean_rec > ctrl_mean_rec)
            if (cand_mean_rec is not None and ctrl_mean_rec is not None)
            else False
        )
        improved = (
            cand_mean_sr > ctrl_mean_sr
            and (cand_mean_rep < ctrl_mean_rep or rec_improved)
        )
        verdict = "IMPROVED" if improved else "NO_VERIFIED_IMPROVEMENT"
        arm_comparisons[arm.arm_id] = {
            "verdict": verdict,
            "control_mean_full_flock_success_rate": round(ctrl_mean_sr, 4),
            "candidate_mean_full_flock_success_rate": round(cand_mean_sr, 4),
            "delta_full_flock_success_rate": round(cand_mean_sr - ctrl_mean_sr, 4),
            "candidate_per_seed_success_rates": [round(x, 4) for x in cand_success_rates],
            "candidate_pooled_wilson_95ci": list(wilson_ci),
            "paired_mcnemar_p_value": mcnemar_p,
            "control_mean_repeated_movement_steps": round(ctrl_mean_rep, 2),
            "candidate_mean_repeated_movement_steps": round(cand_mean_rep, 2),
            "delta_repeated_movement_steps": round(cand_mean_rep - ctrl_mean_rep, 2),
            "control_mean_recovery_outcome_rate": (
                round(ctrl_mean_rec, 4) if ctrl_mean_rec is not None else None
            ),
            "candidate_mean_recovery_outcome_rate": (
                round(cand_mean_rec, 4) if cand_mean_rec is not None else None
            ),
            "candidate_mean_reward_diagnostic_only": round(
                sum(cand_rewards) / len(cand_rewards), 4
            ),
        }

    return {
        "comparable": True,
        "status": "COMPARABLE",
        "reasons": [],
        "arm_comparisons": arm_comparisons,
    }


def build_pilot_budget_and_stopping_proposal(
    budget: TransitionBudget | None = None,
) -> dict[str, Any]:
    """Return the concrete bounded pilot budget and stopping conditions for manager/user review."""
    active_budget = budget or build_transition_budget()
    dev_cases = len(DEVELOPMENT_EVALUATION_SEEDS) + len(get_development_recovery_scenarios())
    held_out_cases = len(HELD_OUT_EVALUATION_SEEDS) + len(get_held_out_recovery_scenarios())

    return {
        "budget": active_budget.to_dict(),
        "discount_horizon_comparison": {
            "arm_a_and_b_gamma_099": discount_horizon_summary(0.99),
            "arm_c_gamma_0997": discount_horizon_summary(0.997),
        },
        "estimated_compute_and_storage": {
            "rollout_throughput_estimate_transitions_per_sec_cpu": 95.0,
            "training_wall_time_minutes_per_arm_seed": 16.0,
            "development_eval_wall_time_minutes_per_arm_seed": 4.0,
            "total_pilot_wall_time_hours_all_9_runs_sequential_cpu": 3.0,
            "checkpoint_zip_size_mb": 2.78,
            "checkpoints_saved_per_arm_seed": active_budget.checkpoint_count,
            "estimated_storage_mb_per_arm_seed": 14.5,
            "estimated_total_storage_mb_all_9_runs": 135.0,
        },
        "evaluation_cost": {
            "development_cases_per_checkpoint": dev_cases,
            "development_checkpoints_evaluated_per_arm_seed": active_budget.checkpoint_count,
            "development_episodes_per_arm_seed": dev_cases * active_budget.checkpoint_count,
            "development_max_world_steps_per_arm_seed": (
                dev_cases * active_budget.checkpoint_count * EPISODE_BUDGET_WORLD_STEPS
            ),
            "held_out_cases_final_checkpoint_only": held_out_cases,
            "held_out_max_world_steps_per_arm_seed": (
                held_out_cases * EPISODE_BUDGET_WORLD_STEPS
            ),
        },
        "stopping_conditions": {
            "kl_divergence_guard": (
                "Each PPO update continues to enforce target_kl=0.03 early stopping inside "
                "Stable-Baselines3; if mean approx_kl exceeds 0.045 on 2 consecutive checkpoints, "
                "abort that arm-seed run as unstable."
            ),
            "severe_success_regression": (
                "If development full-flock success rate drops to <= 0.10 (<= 2/22 cases) on 2 "
                "consecutive checkpoints while Control Arm A remains >= 0.40 on the same paired "
                "seed, stop that arm-seed run early."
            ),
            "repeated_movement_explosion": (
                "If mean repeated-movement fraction exceeds 0.65 across the development suite "
                "for 2 consecutive checkpoints (indicating severe 2-cell oscillation lockup), "
                "stop that arm-seed run early."
            ),
            "numerical_instability": (
                "Immediate abort on any NaN/Inf in policy logits, value loss, or rollout advantages."
            ),
        },
    }


def inspect_runtime_observation_schemas(config: LabConfig | None = None) -> dict[str, Any]:
    """Inspect actual runtime feature counts and ordered feature names from observation builders."""
    base_cfg = config or LabConfig()
    env = SheepdogEnvironment(base_cfg)
    env.reset(seed=101)

    guided_builder = RoleAwareObservationBuilder()
    guided_obs = guided_builder.build(env, dog_index=0)

    emergent_builder = EmergentObservationBuilder()
    emergent_obs = emergent_builder.build(env, dog_index=0)

    guided_names = list(guided_obs.feature_names)
    emergent_names = list(emergent_obs.feature_names)
    in_guided_not_emergent = [name for name in guided_names if name not in emergent_names]
    in_emergent_not_guided = [name for name in emergent_names if name not in guided_names]

    return {
        "guided": {
            "builder_class": "RoleAwareObservationBuilder",
            "feature_count": len(guided_names),
            "feature_names": guided_names,
        },
        "emergent": {
            "builder_class": "EmergentObservationBuilder",
            "feature_count": len(emergent_names),
            "feature_names": emergent_names,
        },
        "comparison": {
            "guided_count": len(guided_names),
            "emergent_count": len(emergent_names),
            "shape_match": len(guided_names) == len(emergent_names),
            "features_in_guided_not_emergent": in_guided_not_emergent,
            "features_in_emergent_not_guided": in_emergent_not_guided,
            "parent_mlp_first_layer_shape": [256, len(guided_names)],
            "zero_shot_compatible": False,
        },
    }


def build_guidance_and_information_audit(config: LabConfig | None = None) -> dict[str, Any]:
    """Return the read-only audit of guidance and observation/memory limitations with follow-up proposals."""
    schemas = inspect_runtime_observation_schemas(config)
    guided_count = schemas["guided"]["feature_count"]
    emergent_count = schemas["emergent"]["feature_count"]
    guided_names = schemas["guided"]["feature_names"]
    emergent_names = schemas["emergent"]["feature_names"]

    return {
        "schemas": schemas,
        "audit_1_guidance": {
            "active_path_configuration": {
                "policy_mode": "neural_policy",
                "observation_mode": "guided",
                "observation_builder": "RoleAwareObservationBuilder",
                "observation_feature_count": guided_count,
                "allow_instinct_target_awareness": False,
                "handler_target_enabled": False,
                "enable_instinct_rewards": True,
                "curriculum_stage": 34,
            },
            "runtime_feature_comparison": {
                "guided_feature_count": guided_count,
                "emergent_feature_count": emergent_count,
                "guided_feature_names": guided_names,
                "emergent_feature_names": emergent_names,
                "features_in_guided_not_emergent": schemas["comparison"]["features_in_guided_not_emergent"],
                "features_in_emergent_not_guided": schemas["comparison"]["features_in_emergent_not_guided"],
            },
            "trace_in_code": [
                {
                    "component": "Step role & target assignment",
                    "location": "src/sheepdog/environment.py:725-762 (prepare_policy_step) -> src/sheepdog/team_strategy.py:39-145 (TeamStrategy.assign_roles)",
                    "behavior": (
                        "Before every policy decision step, TeamStrategy.assign_roles() deterministically "
                        "assigns each of the 3 dogs to REAR_PRESSURE, LEFT_FLANKER, RIGHT_FLANKER, "
                        "COLLECTOR, or BLOCKER and computes a scripted geometric target point "
                        "assignment.target (plus focus_sheep_index)."
                    ),
                },
                {
                    "component": "Observation builder schema (RoleAwareObservationBuilder vs EmergentObservationBuilder)",
                    "location": "src/sheepdog/observations.py:49-188 vs 324-470",
                    "behavior": (
                        f"In active Stage 34 training (observation_mode='guided'), RoleAwareObservationBuilder "
                        f"produces {guided_count} features, including scripted guidance features: "
                        "target_x, target_y, distance_to_target, 5-dim role one-hot (role_rear_pressure, "
                        "role_left_flanker, role_right_flanker, role_collector, role_blocker), focus_sheep_dx/dy/dist, "
                        "and stray_sheep_dx/dy. "
                        f"In contrast, EmergentObservationBuilder produces {emergent_count} features: "
                        "it does NOT merely zero dimensions, but completely omits role labels and assigned targets, "
                        "replacing them with dog identity one-hot (dog_id_slot_0..2), nearest unpenned sheep relative "
                        "position and distance, and farthest-from-pen unpenned sheep relative position and distance."
                    ),
                },
                {
                    "component": "Action legality mask",
                    "location": "src/sheepdog/environment.py:635-652 (action_mask_for_dog)",
                    "behavior": (
                        "For policy_mode='neural_policy', action_mask_for_dog() masks only physically "
                        "blocked moves (project_dog_action(dog, action) != dog.position) and always "
                        "allows 'wait'. Tactical wait gating is NOT applied to neural_policy."
                    ),
                },
                {
                    "component": "Reward shaping & wait penalty",
                    "location": "src/sheepdog/rewards.py:194-414 & src/sheepdog/environment.py:1068-1125, 2612-2616",
                    "behavior": (
                        "Stage 34 reward uses PenNavigation BFS route progress (progress_scale=16.5), "
                        "penning bonus (36.0), completion bonus (120.0), and instinct shaping "
                        "(behind_flock, dog_spacing, overpressure, flank_arc, pack_dispersion). "
                        "Notably, waited_without_reason calls _tactically_valid_wait(), which scores "
                        "actions against heuristic role/formation weights."
                    ),
                },
            ],
            "architectural_and_checkpoint_compatibility": {
                "parent_input_dimension": guided_count,
                "emergent_input_dimension": emergent_count,
                "compatible_zero_shot": False,
                "explanation": (
                    f"The Stage 34 parent checkpoint mlp_extractor input layer has shape [256, {guided_count}]. "
                    f"Loading an emergent observation of shape ({emergent_count},) into this model fails with a tensor "
                    "shape mismatch error. Emergent evaluation requires either training a fresh emergent policy "
                    f"from scratch or an explicit surgical adapter layer (e.g. projecting {emergent_count} -> {guided_count} "
                    "or surgical zero-padding), both of which induce severe distribution shifts that must be reported."
                ),
            },
            "smallest_controlled_followup_experiment": {
                "name": "guidance_ablation_guided_vs_emergent_v1",
                "design": (
                    f"Compare two matched arms with identical reward functions, physics, and action masks: "
                    f"Arm G1 (observation_mode='guided', {guided_count} features) vs Arm G2 (observation_mode='emergent', "
                    f"{emergent_count} features). Because of the {guided_count} vs {emergent_count} weight shape mismatch, "
                    "both arms must be trained under identical curricula from an identical stage initialization."
                ),
                "checkpoint_compatibility": (
                    "Incompatible with direct Stage 34 checkpoint loading without surgical projection; "
                    "must be initialized from a shared Stage 34 emergent-compatible trunk or evaluated on matched fresh runs."
                ),
                "required_evidence": (
                    "Report assisted (guided) vs autonomous (emergent) full-flock success rate, "
                    "recovery rate from loops, and target-tracking correlation across 3 paired seeds "
                    "on the development and held-out suites."
                ),
            },
        },
        "audit_2_information": {
            "trace_in_code": [
                {
                    "component": "Fence & route representation mismatch in observations",
                    "location": "src/sheepdog/observations.py:58-98 vs src/sheepdog/pen_navigation.py:11-104",
                    "finding": (
                        "Rewards and TeamStrategy route around closed pen fences via PenNavigation "
                        "(BFS shortest distance and first-corner waypoint navigation.target()), "
                        "but ObservationBuilder computes pen_x, pen_y using raw Euclidean pen.center, "
                        "and includes NO explicit pen-fence bounds, wall thickness, or pen_opening orientation feature. "
                        "In emergent mode (where assigned_target is absent), the policy has no direct observation "
                        "of which side of the pen is open or where the fence corners lie."
                    ),
                },
                {
                    "component": "History-derived indicators vs feedforward MLP memory",
                    "location": "src/sheepdog/environment.py:2981-2996, src/sheepdog/observations.py:89-118",
                    "finding": (
                        "The policy network is a purely feedforward MLP (256, 256, 128) with no "
                        "recurrence (LSTM/GRU) and no frame stacking. Its temporal history consists "
                        "exclusively of 4 scalar features in the observation: no_progress_steps, "
                        "blocked_steps, revisits_recent_position (checks only position[t] == position[t-2]), "
                        "and two_position_loop (checks strict A-B-A-B period-2 alternation over the last 4 steps). "
                        "It is completely blind to: (a) sheep velocity / whether sheep moved last step, (b) teammate "
                        "last actions or velocities, (c) loops of period >= 3, and (d) which direction escapes "
                        "the loop once two_position_loop flips to 1.0."
                    ),
                },
            ],
            "what_is_missing_vs_hard_to_learn": {
                "genuinely_missing_from_observation": [
                    "Pen opening orientation / fence-safe waypoint vector when scripted targets are absent.",
                    "Sheep recent displacement / velocity vector (distinguishing moving flock from wall-pinned stall).",
                    "Directional history of visited cells (knowing which move repeats vs breaks a cycle) and loops of period >= 3.",
                ],
                "present_but_hard_to_exploit": [
                    "Period-2 oscillation flag (two_position_loop) and no_progress_steps are present, "
                    "but because they are directionless scalars, a deterministic feedforward policy "
                    "often maps the (state, flag=1) pair to a single fixed action that immediately "
                    "re-enters the stall 2 steps later.",
                ],
            },
            "smallest_controlled_followup_experiment": {
                "name": "information_route_and_short_history_v1",
                "design": (
                    "Compare two minimal, separate arms from a matched checkpoint/stage: "
                    "(I1) replace Euclidean pen.center with the fence-safe PenNavigation.target() "
                    f"waypoint (preserving the {guided_count}-dim shape), and (I2) 2-frame observation stacking "
                    "(or adding 4 directional revisit indicators + flock displacement delta)."
                ),
                "checkpoint_compatibility": (
                    f"Arm I1 preserves the {guided_count}-dim input shape (requiring fine-tuning adaptation "
                    "due to semantic distribution change), whereas Arm I2 changes observation_size "
                    "(requiring input-layer weight expansion or fresh stage initialization)."
                ),
                "required_evidence": (
                    "Compare repeated-movement duration, fence-wing stall duration, and recovery "
                    "rate across unpenned sheep counts 1..6 on the development and held-out suites."
                ),
            },
        },
    }


def prepare_experiment_workspace(
    experiment_root: str | Path,
    *,
    parent_checkpoint_path: str | Path = DEFAULT_LATEST_PARENT_CHECKPOINT,
    base_config_path: str | Path = Path("artifacts/effective-training-config.json"),
    training_state_path: str | Path = Path("artifacts/training-state.json"),
    checkpoint_metadata_path: str | Path | None = DEFAULT_LATEST_PARENT_METADATA,
    paired_seeds: tuple[int, ...] = DEFAULT_PAIRED_TRAINING_SEEDS,
    budget: TransitionBudget | None = None,
    evaluation_seeds: tuple[int, ...] = DEVELOPMENT_EVALUATION_SEEDS,
) -> dict[str, Any]:
    """Materialize isolated experiment directories, frozen parent copy, arm configs, and manifest."""
    root = Path(experiment_root)
    root.mkdir(parents=True, exist_ok=True)

    active_budget = budget or build_transition_budget(num_seeds=len(paired_seeds))
    seed_isolation = verify_seed_and_scenario_isolation(paired_seeds)

    primary_inspection = inspect_checkpoint_archive(
        parent_checkpoint_path,
        config_path=base_config_path,
        training_state_path=training_state_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
    )
    alternate_inspection: dict[str, Any] | None = None
    if ALTERNATE_BEST_REWARD_CHECKPOINT.exists() and Path(parent_checkpoint_path).resolve() != ALTERNATE_BEST_REWARD_CHECKPOINT.resolve():
        alternate_inspection = inspect_checkpoint_archive(
            ALTERNATE_BEST_REWARD_CHECKPOINT,
            config_path=base_config_path,
            training_state_path=training_state_path,
            checkpoint_metadata_path=None,
        )

    # Copy parent checkpoint into isolated baseline directory so arms never write near active checkpoints
    baseline_dir = root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frozen_parent_copy = baseline_dir / Path(parent_checkpoint_path).name
    shutil.copy2(parent_checkpoint_path, frozen_parent_copy)
    if sha256_file(frozen_parent_copy) != primary_inspection["checkpoint_sha256"]:
        raise RuntimeError("Frozen parent checkpoint copy checksum mismatch.")

    base_config = load_lab_config(base_config_path) if Path(base_config_path).exists() else LabConfig()
    locked_lr = float(
        primary_inspection["loaded_model_parameters"].get("learning_rate")
        or base_config.training.learning_rate
    )

    arm_first_seed_configs: dict[str, LabConfig] = {}
    arm_entries: dict[str, Any] = {}

    for arm in EXPERIMENT_ARMS:
        arm_dir = root / "arms" / arm.arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        seed_entries: dict[str, Any] = {}
        for seed in paired_seeds:
            seed_dir = arm_dir / f"seed_{seed}"
            run_output_dir = seed_dir / "artifacts"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            arm_cfg = build_arm_config(
                base_config,
                arm,
                arm_output_dir=run_output_dir,
                train_seed=seed,
                locked_learning_rate=locked_lr,
                budget=active_budget,
                evaluation_seeds=evaluation_seeds,
            )
            if seed == paired_seeds[0]:
                arm_first_seed_configs[arm.arm_id] = arm_cfg

            cfg_file = seed_dir / "config.json"
            save_lab_config(arm_cfg, cfg_file)

            # Seed an isolated training-state.json that points BOTH policy_state_path
            # and best_model_path to the frozen parent checkpoint copy so MaskablePPOTrainer
            # resumes from the exact parent checkpoint without touching active artifacts/.
            parent_stage = int(
                primary_inspection.get("checkpoint_metadata_curriculum_stage")
                or arm_cfg.rewards.instincts.curriculum_stage
            )
            obs_schema_info = inspect_runtime_observation_schemas(arm_cfg)
            active_obs_size = int(obs_schema_info["guided"]["feature_count"])
            isolated_state = {
                "trainer_type": "maskable_ppo",
                "policy_type": "neural",
                "policy_state_path": str(frozen_parent_copy.resolve()),
                "best_model_path": str(frozen_parent_copy.resolve()),
                "best_model_curriculum_stage": parent_stage,
                "best_success_rate": float(
                    primary_inspection.get("checkpoint_metadata_success_rate") or 0.0
                ),
                "best_average_reward": float(
                    primary_inspection.get("checkpoint_metadata_average_reward") or 0.0
                ),
                "best_completion_steps": float(EPISODE_BUDGET_WORLD_STEPS),
                "total_episodes": 0,
                "environment_episodes_total": 0,
                "total_timesteps": int(
                    primary_inspection["loaded_model_parameters"].get("num_timesteps") or 0
                ),
                "policy_version": int(
                    primary_inspection.get("checkpoint_metadata_policy_version") or 0
                ),
                "policy_config": {
                    "hidden_sizes": list(arm_cfg.training.neural_hidden_sizes),
                    "observation_size": active_obs_size,
                    "action_size": 9,
                    "env_workers": 1,
                },
                "parent_checkpoint_source": str(Path(parent_checkpoint_path).resolve()),
                "parent_checkpoint_sha256": primary_inspection["checkpoint_sha256"],
            }
            (run_output_dir / "training-state.json").write_text(
                json.dumps(isolated_state, indent=2),
                encoding="utf-8",
            )
            save_lab_config(arm_cfg, run_output_dir / "effective-training-config.json")

            seed_entries[str(seed)] = {
                "train_seed": int(seed),
                "config_path": str(cfg_file),
                "output_dir": str(run_output_dir),
                "training_state_path": str(run_output_dir / "training-state.json"),
            }

        arm_entries[arm.arm_id] = {
            "spec": arm.to_dict(),
            "locked_learning_rate": locked_lr,
            "discount_horizon": discount_horizon_summary(arm.gamma),
            "seeds": seed_entries,
        }

    diff_verification = verify_single_parameter_arm_diffs(arm_first_seed_configs)

    protocol_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "seed_and_scenario_isolation": seed_isolation,
        "primary_decision_mode": "deterministic",
        "secondary_diagnostic_mode": "sampled",
        "episode_budget_world_steps": EPISODE_BUDGET_WORLD_STEPS,
        "episode_budget_policy_transitions": EPISODE_BUDGET_POLICY_TRANSITIONS,
        "development_seeds": list(DEVELOPMENT_EVALUATION_SEEDS),
        "development_scenarios": list(get_development_recovery_scenarios().keys()),
        "held_out_seeds": list(HELD_OUT_EVALUATION_SEEDS),
        "held_out_scenarios": list(get_held_out_recovery_scenarios().keys()),
    }
    protocol_path = root / "evaluation_protocol.json"
    protocol_path.write_text(json.dumps(protocol_payload, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "prepared_not_started",
        "experiment_root": str(root.resolve()),
        "git_provenance": collect_git_provenance(),
        "dependency_versions": collect_dependency_versions(),
        "assistance_mode": {
            "policy_mode": base_config.policy.policy_mode,
            "observation_mode": base_config.training.observation_mode,
            "allow_instinct_target_awareness": base_config.policy.allow_instinct_target_awareness,
            "handler_target_enabled": base_config.policy.handler_target_enabled,
            "enable_instinct_rewards": base_config.rewards.instincts.enable_instinct_rewards,
            "assistance_classification": (
                "guided_observations_and_instinct_reward_shaping "
                f"(RoleAwareObservationBuilder {active_obs_size} features including 5-dim role one-hot + "
                "assigned_target_dx/dy + focus_sheep_dx/dy/dist + stray_sheep_dx/dy; physical-only action mask)"
            ),
        },
        "runtime_observation_schemas": obs_schema_info,
        "parent_checkpoint": primary_inspection,
        "alternate_best_reward_checkpoint": alternate_inspection,
        "frozen_parent_copy_path": str(frozen_parent_copy.resolve()),
        "frozen_parent_copy_sha256": primary_inspection["checkpoint_sha256"],
        "paired_training_seeds": list(paired_seeds),
        "single_parameter_diff_verification": diff_verification,
        "budget": active_budget.to_dict(),
        "pilot_proposal": build_pilot_budget_and_stopping_proposal(active_budget),
        "evaluation_protocol_path": str(protocol_path),
        "arms": arm_entries,
        "guidance_and_information_audit": build_guidance_and_information_audit(base_config),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_loaded_arm_models(experiment_root: str | Path) -> dict[str, Any]:
    """Load the frozen parent checkpoint under each arm config and verify live SB3 parameters."""
    root = Path(experiment_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_path = Path(manifest["frozen_parent_copy_path"])
    parent_sha_before = sha256_file(parent_path)

    arm_verifications: dict[str, Any] = {}
    for arm_id, arm_info in manifest["arms"].items():
        first_seed_key = next(iter(arm_info["seeds"].keys()))
        seed_info = arm_info["seeds"][first_seed_key]
        cfg = load_lab_config(seed_info["config_path"])
        policy = NeuralPolicy.load(parent_path, cfg)
        before_configure = policy.live_training_parameters()
        after_configure = policy.configure_for_training(
            cfg,
            effective_learning_rate=cfg.training.learning_rate,
            effective_entropy_coef=cfg.training.entropy_coef,
            reset_rollout_buffer=True,
            set_seed=True,
        )
        vec_env = policy.model.get_env()
        if vec_env is not None:
            vec_env.close()

        matches = (
            abs(float(after_configure["gamma"]) - float(cfg.training.gamma)) < 1e-9
            and abs(float(after_configure["rollout_buffer_gamma"]) - float(cfg.training.gamma)) < 1e-9
            and abs(float(after_configure["gae_lambda"]) - float(cfg.training.gae_lambda)) < 1e-9
            and abs(float(after_configure["rollout_buffer_gae_lambda"]) - float(cfg.training.gae_lambda)) < 1e-9
            and abs(float(after_configure["ent_coef"]) - float(cfg.training.entropy_coef)) < 1e-9
            and abs(float(after_configure["learning_rate"]) - float(cfg.training.learning_rate)) < 1e-12
            and all(
                abs(float(lr) - float(cfg.training.learning_rate)) < 1e-12
                for lr in after_configure["optimizer_learning_rates"]
            )
            and int(after_configure["seed"]) == int(cfg.training.train_seed)
            and int(after_configure["rollout_buffer_pos"]) == 0
            and not bool(after_configure["rollout_buffer_full"])
        )
        if not matches:
            raise RuntimeError(
                f"Live trainer parameter verification failed for {arm_id}: {after_configure}"
            )
        arm_verifications[arm_id] = {
            "passed": True,
            "train_seed": int(cfg.training.train_seed),
            "before_configure_from_zip": before_configure,
            "after_configure_for_training": after_configure,
        }

    parent_sha_after = sha256_file(parent_path)
    if parent_sha_before != parent_sha_after:
        raise RuntimeError("Parent checkpoint checksum changed during verification!")

    return {
        "verified": True,
        "parent_checkpoint_sha256_unchanged": True,
        "parent_checkpoint_sha256": parent_sha_after,
        "arms": arm_verifications,
    }


def run_controlled_arm_seed(
    experiment_root: str | Path,
    *,
    arm_id: str,
    train_seed: int,
    evaluate_development: bool = True,
    evaluate_held_out: bool = False,
    eval_max_steps_override: int | None = None,
) -> dict[str, Any]:
    """Execute one isolated (arm, seed) training run and verify parent immutability, live parameters, and bounded progression."""
    root = Path(experiment_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    orig_parent = Path(manifest["parent_checkpoint"]["checkpoint_path"])
    frozen_parent = Path(manifest["frozen_parent_copy_path"])
    expected_sha = str(manifest["parent_checkpoint"]["checkpoint_sha256"])

    if orig_parent.exists() and sha256_file(orig_parent) != expected_sha:
        raise RuntimeError("Original parent checkpoint SHA-256 mismatch before run.")
    if sha256_file(frozen_parent) != expected_sha:
        raise RuntimeError("Frozen parent copy SHA-256 mismatch before run.")

    # Enforce operational run ordering: Control Arm A must be completed before candidate arms run
    paired_control_result: dict[str, Any] | None = None
    if arm_id != "arm_a_control":
        control_seed_result_path = (
            root / "arms" / "arm_a_control" / f"seed_{train_seed}" / "artifacts" / "arm_seed_result.json"
        )
        if not control_seed_result_path.exists():
            raise RuntimeError(
                f"Operational run ordering violation: Paired Control Arm A for seed {train_seed} "
                f"must be executed before candidate arm '{arm_id}' can run on seed {train_seed}. "
                f"Missing expected control result at: {control_seed_result_path}"
            )
        paired_control_result = json.loads(control_seed_result_path.read_text(encoding="utf-8"))

    seed_info = manifest["arms"][arm_id]["seeds"][str(train_seed)]
    cfg = load_lab_config(seed_info["config_path"])
    output_dir = Path(seed_info["output_dir"])

    starting_timesteps = int(
        manifest["parent_checkpoint"]["loaded_model_parameters"].get("num_timesteps") or 0
    )

    checkpoints_evaluated: list[dict[str, Any]] = []
    stop_requested = False
    stopped_early = False
    stop_reason: str | None = None
    consecutive_kl_divergences = 0
    consecutive_repeated_movement_high = 0
    consecutive_paired_regressions = 0

    def _progress_callback(payload: dict[str, Any]) -> None:
        nonlocal stop_requested, stopped_early, stop_reason
        nonlocal consecutive_kl_divergences, consecutive_repeated_movement_high, consecutive_paired_regressions

        phase = payload.get("phase")
        if phase not in ("checkpoint", "checkpoint_saved"):
            return

        chk_payload = payload.get("checkpoint_payload") or {}
        saved_model_str = chk_payload.get("policy_state_path")
        chk_json_str = payload.get("checkpoint_path")
        chk_meta: dict[str, Any] = {}
        if chk_json_str:
            chk_json_p = Path(chk_json_str)
            if not chk_json_p.is_absolute():
                chk_json_p = Path.cwd() / chk_json_p
            if chk_json_p.exists() and chk_json_p.suffix == ".json":
                chk_meta = json.loads(chk_json_p.read_text(encoding="utf-8"))
                if not saved_model_str:
                    saved_model_str = chk_meta.get("policy_state_path")

        if not saved_model_str:
            # Fall back to training-state.json
            state_p = output_dir / "training-state.json"
            if state_p.exists():
                st_data = json.loads(state_p.read_text(encoding="utf-8"))
                saved_model_str = st_data.get("policy_state_path")

        if not saved_model_str:
            return
        saved_model_p = Path(saved_model_str)
        if not saved_model_p.is_absolute():
            saved_model_p = Path.cwd() / saved_model_p
        chk_sha = sha256_file(saved_model_p) if saved_model_p.exists() else None

        chk_idx = len(checkpoints_evaluated)
        chk_ep = int(
            payload.get("checkpoint_episode")
            or chk_payload.get("checkpoint_episode")
            or chk_meta.get("checkpoint_episode", chk_idx + 1)
        )
        global_ts = int(
            payload.get("total_timesteps")
            or chk_payload.get("global_timestep")
            or chk_meta.get("global_timestep", 0)
        )
        actual_transitions = global_ts - starting_timesteps

        # Evaluate development suite at each checkpoint boundary if enabled
        dev_eval: dict[str, Any] | None = None
        if evaluate_development and saved_model_p.exists():
            eval_p = NeuralPolicy.load(saved_model_p, cfg)
            try:
                dev_eval = evaluate_policy_on_recovery_suite(
                    eval_p,
                    cfg,
                    suite="development",
                    deterministic=True,
                    replay_output_dir=output_dir / "eval_replays" / f"checkpoint_{chk_idx + 1}",
                    checkpoint_path=saved_model_p,
                    checkpoint_sha256=chk_sha,
                    max_steps_override=eval_max_steps_override,
                )
            finally:
                venv = eval_p.model.get_env()
                if venv is not None:
                    venv.close()

        # Extract PPO diagnostics representing missing as None (unavailable), never zero!
        def _get_metric(key: str) -> float | None:
            v = payload.get(key)
            if v is None:
                v = chk_payload.get(key)
            if v is None:
                v = chk_meta.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        approx_kl = _get_metric("approx_kl")
        loss_val = _get_metric("loss")
        value_loss = _get_metric("value_loss")
        pg_loss = _get_metric("policy_gradient_loss")
        entropy_loss = _get_metric("entropy_loss")
        clip_fraction = _get_metric("clip_fraction")

        # Check explicit stopping conditions
        # 1. Fail-closed: if actual policy transitions occurred (> 0) but required PPO safety diagnostics are unavailable
        if actual_transitions > 0:
            missing_required: list[str] = []
            if approx_kl is None:
                missing_required.append("approx_kl")
            if loss_val is None:
                missing_required.append("loss")
            if value_loss is None:
                missing_required.append("value_loss")
            if missing_required:
                stop_requested = True
                stopped_early = True
                stop_reason = (
                    f"missing_safety_diagnostics: required PPO telemetry {missing_required} "
                    f"unavailable at checkpoint {chk_idx + 1}"
                )

        # 2. Numerical instability (NaN / Inf)
        import math
        for diag_name, diag_val in [
            ("loss", loss_val),
            ("approx_kl", approx_kl),
            ("value_loss", value_loss),
            ("policy_gradient_loss", pg_loss),
        ]:
            if diag_val is not None and (math.isnan(diag_val) or math.isinf(diag_val)):
                stop_requested = True
                stopped_early = True
                stop_reason = f"numerical_instability: {diag_name}={diag_val} at checkpoint {chk_idx + 1}"
                break

        # 3. KL divergence guard (approx_kl > 0.045 for 2 consecutive checkpoints)
        if approx_kl is not None and approx_kl > 0.045:
            consecutive_kl_divergences += 1
            if consecutive_kl_divergences >= 2:
                stop_requested = True
                stopped_early = True
                stop_reason = (
                    f"kl_divergence_guard: approx_kl={approx_kl:.4f} exceeded 0.045 "
                    f"for 2 consecutive checkpoints"
                )
        else:
            consecutive_kl_divergences = 0

        # 4. Repeated movement explosion (repeated_movement_fraction > 0.65 for 2 consecutive checkpoints)
        if dev_eval is not None:
            rep_frac = dev_eval.get("repeated_movement_fraction_mean")
            if rep_frac is not None and float(rep_frac) > 0.65:
                consecutive_repeated_movement_high += 1
                if consecutive_repeated_movement_high >= 2:
                    stop_requested = True
                    stopped_early = True
                    stop_reason = (
                        f"repeated_movement_explosion: repeated_movement_fraction={float(rep_frac):.4f} "
                        f"exceeded 0.65 for 2 consecutive checkpoints"
                    )
            else:
                consecutive_repeated_movement_high = 0

            # 5. Paired-control regression rule (candidate <= 0.10 while Control Arm A >= 0.40 for 2 consecutive checkpoints)
            if paired_control_result is not None:
                ctrl_chk_evals = paired_control_result.get("checkpoints_evaluated", [])
                if chk_idx >= len(ctrl_chk_evals):
                    stop_requested = True
                    stopped_early = True
                    stop_reason = (
                        f"missing_paired_control_checkpoint: Control Arm A has only "
                        f"{len(ctrl_chk_evals)} checkpoints evaluated, but candidate reached {chk_idx + 1}"
                    )
                else:
                    ctrl_dev = ctrl_chk_evals[chk_idx].get("development_evaluation") or {}
                    ctrl_sr = ctrl_dev.get("full_flock_success_rate")
                    cand_sr = dev_eval.get("full_flock_success_rate")
                    if (
                        ctrl_sr is not None
                        and cand_sr is not None
                        and float(cand_sr) <= 0.10
                        and float(ctrl_sr) >= 0.40
                    ):
                        consecutive_paired_regressions += 1
                        if consecutive_paired_regressions >= 2:
                            stop_requested = True
                            stopped_early = True
                            stop_reason = (
                                f"severe_success_regression: candidate success {float(cand_sr):.2%} <= 10% "
                                f"while paired Control Arm A was {float(ctrl_sr):.2%} >= 40% for 2 consecutive checkpoints"
                            )
                    else:
                        consecutive_paired_regressions = 0

        chk_record = {
            "checkpoint_index": chk_idx + 1,
            "checkpoint_episode": chk_ep,
            "global_timestep": global_ts,
            "actual_additional_transitions": actual_transitions,
            "actual_completed_world_steps": actual_transitions // TRANSITIONS_PER_WORLD_STEP,
            "partial_team_steps": actual_transitions % TRANSITIONS_PER_WORLD_STEP,
            "checkpoint_path": str(saved_model_p),
            "checkpoint_sha256": chk_sha,
            "ppo_diagnostics": {
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "loss": loss_val,
                "value_loss": value_loss,
                "policy_gradient_loss": pg_loss,
                "entropy_loss": entropy_loss,
            },
            "development_evaluation": dev_eval,
            "stop_requested": bool(stop_requested),
            "stop_reason": stop_reason,
        }
        checkpoints_evaluated.append(chk_record)

    previous_capture_policy = get_global_capture_policy()
    set_global_capture_policy(CapturePolicy(mode="off"))
    try:
        trainer = MaskablePPOTrainer(
            cfg,
            output_root=output_dir,
        )
        summary = trainer.train(
            should_stop=lambda: stop_requested,
            progress_callback=_progress_callback,
        )
    finally:
        set_global_capture_policy(previous_capture_policy)

    if orig_parent.exists() and sha256_file(orig_parent) != expected_sha:
        raise RuntimeError("Original parent checkpoint was modified during arm run!")
    if sha256_file(frozen_parent) != expected_sha:
        raise RuntimeError("Frozen parent checkpoint copy was modified during arm run!")

    live_params = getattr(trainer, "last_configured_training_parameters", None) or {}
    live_verified = (
        abs(float(live_params.get("gamma", -1.0)) - float(cfg.training.gamma)) < 1e-9
        and abs(float(live_params.get("rollout_buffer_gamma", -1.0)) - float(cfg.training.gamma)) < 1e-9
        and abs(float(live_params.get("ent_coef", -1.0)) - float(cfg.training.entropy_coef)) < 1e-9
        and abs(float(live_params.get("learning_rate", -1.0)) - float(cfg.training.learning_rate)) < 1e-12
        and int(live_params.get("seed", -1)) == int(cfg.training.train_seed)
    )

    saved_state = json.loads((output_dir / "training-state.json").read_text(encoding="utf-8"))
    final_ckpt_path = Path(saved_state["policy_state_path"])
    if not final_ckpt_path.is_absolute():
        final_ckpt_path = Path.cwd() / final_ckpt_path
    final_ckpt_sha = sha256_file(final_ckpt_path)

    final_model_timesteps = int(saved_state.get("total_timesteps") or starting_timesteps)
    actual_additional_transitions = max(0, final_model_timesteps - starting_timesteps)
    completed_world_steps = actual_additional_transitions // TRANSITIONS_PER_WORLD_STEP
    partial_team_steps = actual_additional_transitions % TRANSITIONS_PER_WORLD_STEP
    requested_transitions = int(cfg.training.total_timesteps)
    rollout_overshoot = max(0, actual_additional_transitions - requested_transitions)

    # Evaluate held-out recovery suite ONLY on the final completed checkpoint if requested
    held_out_evaluation: dict[str, Any] | None = None
    if evaluate_held_out:
        eval_policy = NeuralPolicy.load(final_ckpt_path, cfg)
        try:
            held_out_evaluation = evaluate_policy_on_recovery_suite(
                eval_policy,
                cfg,
                suite="held_out",
                deterministic=True,
                replay_output_dir=output_dir / "eval_replays" / "final_held_out",
                checkpoint_path=final_ckpt_path,
                checkpoint_sha256=final_ckpt_sha,
                max_steps_override=eval_max_steps_override,
            )
        finally:
            venv = eval_policy.model.get_env()
            if venv is not None:
                venv.close()

    # Final development evaluation record from last checkpoint or separate eval
    final_dev_eval: dict[str, Any] | None = None
    if checkpoints_evaluated and checkpoints_evaluated[-1].get("development_evaluation") is not None:
        final_dev_eval = checkpoints_evaluated[-1]["development_evaluation"]
    elif evaluate_development:
        eval_policy = NeuralPolicy.load(final_ckpt_path, cfg)
        try:
            final_dev_eval = evaluate_policy_on_recovery_suite(
                eval_policy,
                cfg,
                suite="development",
                deterministic=True,
                replay_output_dir=output_dir / "eval_replays" / "final_development",
                checkpoint_path=final_ckpt_path,
                checkpoint_sha256=final_ckpt_sha,
                max_steps_override=eval_max_steps_override,
            )
        finally:
            venv = eval_policy.model.get_env()
            if venv is not None:
                venv.close()

    evaluations: dict[str, Any] = {}
    if final_dev_eval is not None:
        evaluations["development"] = final_dev_eval
    if held_out_evaluation is not None:
        evaluations["held_out"] = held_out_evaluation

    consumption = {
        "parent_starting_timesteps": starting_timesteps,
        "actual_final_timesteps": final_model_timesteps,
        "requested_policy_transitions": requested_transitions,
        "actual_additional_policy_transitions": actual_additional_transitions,
        "requested_world_steps_whole": requested_transitions // TRANSITIONS_PER_WORLD_STEP,
        "actual_completed_world_steps": completed_world_steps,
        "partial_team_steps": partial_team_steps,
        "rollout_overshoot_transitions": rollout_overshoot,
        "checkpoints_planned": len(cfg.training.checkpoint_timesteps),
        "checkpoints_completed": len(checkpoints_evaluated),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "ppo_per_update_kl_early_stops": "handled_per_update_by_target_kl",
        "arm_level_termination": bool(stopped_early),
    }

    run_record = {
        "arm_id": arm_id,
        "train_seed": int(train_seed),
        "completed": bool(not stopped_early),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "trained_policy_transitions": actual_additional_transitions,
        "actual_completed_world_steps": completed_world_steps,
        "partial_team_steps": partial_team_steps,
        "consumption": consumption,
        "live_configured_parameters": live_params,
        "live_parameter_verification_passed": bool(live_verified),
        "final_checkpoint_path": str(final_ckpt_path),
        "final_checkpoint_sha256": final_ckpt_sha,
        "parent_checkpoint_sha256_verified": expected_sha,
        "trainer_summary_total_episodes": int(getattr(summary, "total_episodes", 0)),
        "trainer_summary_total_timesteps": int(getattr(summary, "total_timesteps", 0)),
        "checkpoints_evaluated": checkpoints_evaluated,
        "evaluations": evaluations,
    }
    (output_dir / "arm_seed_result.json").write_text(
        json.dumps(run_record, indent=2),
        encoding="utf-8",
    )
    return run_record
