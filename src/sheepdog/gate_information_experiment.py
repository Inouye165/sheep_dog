"""Controlled gate-information observation ablation experiment (Stage 34).

Prepares, verifies, and runs an isolated two-arm comparison from a pinned
54-input Stage 34 parent checkpoint, widening the policy and value input layers
to 58 dimensions with bit-for-bit preservation of existing weights and exact
zero initialization on the 4 new columns:
  - Arm A (Control): ``observation_mode="guided_gate_control"``
    (54 existing guided features + 4 constant-zero features)
  - Arm B (Treatment): ``observation_mode="guided_gate_treatment"``
    (the same 54 existing guided features + 4 explicit gate features:
    ``gate_center_dx``, ``gate_center_dy``, ``gate_normal_x``, ``gate_normal_y``)

Both arms use identical optimizer reset (fresh Adam state), identical locked
hyperparameters, identical paired training seeds ``(101, 202, 303)``, and an
identical bounded pilot budget of ``90,112`` policy transitions per arm-seed.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from sheepdog.config import LabConfig
from sheepdog.entities import Pen, Point
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
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
from sheepdog.recovery_hyperparam_experiment import (
    build_transition_budget,
    collect_dependency_versions,
    collect_git_provenance,
    save_lab_config,
    sha256_file,
)
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.replay_writer import (
    CapturePolicy,
    get_global_capture_policy,
    set_global_capture_policy,
)

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "stage34_gate_information_ablation_v1"

DOGS_PER_TEAM = 3
SHEEP_PER_FLOCK = 6
TRANSITIONS_PER_WORLD_STEP = DOGS_PER_TEAM
DISCOUNT_TIMEBASE = "sequential_shared_dog"
EPISODE_BUDGET_WORLD_STEPS = 2_320
NO_PROGRESS_WINDOW_WORLD_STEPS = 780
CURRICULUM_STAGE = 34

OLD_OBSERVATION_DIM = 54
NEW_OBSERVATION_DIM = 58

DEFAULT_PARENT_CHECKPOINT = Path(
    "artifacts/checkpoints/stage34/maskable-ppo-stage34-339740239.zip"
)
EXPECTED_PARENT_SHA256 = (
    "bc6b9883ab5f9343255c15e59cbf0252e8268808731482fec2f5b417d3196a00"
)
DEFAULT_PARENT_METADATA = Path(
    "artifacts/checkpoints/checkpoint-chk_run_20260817_063453_91f4_pv_10369_ts_339740239.json"
)

EXPECTED_PARENT_ARCH: tuple[int, ...] = (256, 256, 128)
EXPECTED_PARENT_GAMMA = 0.997
EXPECTED_PARENT_GAE_LAMBDA = 0.98
EXPECTED_PARENT_ENT_COEF = 0.0075
EXPECTED_PARENT_LEARNING_RATE = 5e-5
EXPECTED_PARENT_N_STEPS = 2_048
EXPECTED_PARENT_BATCH_SIZE = 1_024

PAIRED_TRAINING_SEEDS: tuple[int, ...] = (101, 202, 303)
DEVELOPMENT_EVALUATION_SEEDS: tuple[int, ...] = (
    11,
    23,
    37,
    41,
    53,
    59,
    61,
    67,
    71,
    73,
)
TARGETED_FAILURE_SEEDS: tuple[int, ...] = (53, 67, 73)
PREVIOUSLY_SOLVED_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 59, 61, 71)
HELD_OUT_EVALUATION_STATUS = "pending_manifest_audit"

PILOT_TRANSITIONS_PER_ARM_SEED = 90_112
PILOT_ROLLOUT_STEPS = 2_048
PILOT_CHECKPOINT_COUNT = 4
MAX_WALL_SECONDS_PER_ARM_SEED = 1_800.0
MAX_TOTAL_WALL_SECONDS = 10_800.0

TELEMETRY_HEAD_STEPS = 200
TELEMETRY_TAIL_STEPS = 500
EQUIVALENCE_ATOL = 1e-5
EQUIVALENCE_PROB_ATOL = 1e-6


@dataclass(frozen=True, slots=True)
class GateArmSpec:
    """Single-factor observation treatment specification for one experiment arm."""

    arm_id: str
    label: str
    hypothesis: str
    observation_mode: str
    observation_size: int
    observation_schema_version: str
    is_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize arm specification to a dictionary."""
        return asdict(self)


GATE_EXPERIMENT_ARMS: tuple[GateArmSpec, ...] = (
    GateArmSpec(
        arm_id="arm_a_control_zero_gate",
        label="Arm A — 58-Input Zero-Padded Gate Control",
        hypothesis=(
            "Baseline control with identical 58-input network architecture, fresh "
            "optimizer state, and continued Stage 34 training budget, where the 4 "
            "appended gate features are constant 0.0."
        ),
        observation_mode="guided_gate_control",
        observation_size=NEW_OBSERVATION_DIM,
        observation_schema_version=GATE_OBSERVATION_SCHEMA_VERSION,
        is_control=True,
    ),
    GateArmSpec(
        arm_id="arm_b_treatment_gate_info",
        label="Arm B — 58-Input Explicit Gate-Information Treatment",
        hypothesis=(
            "Providing relative gate-center offsets (gate_center_dx, gate_center_dy) "
            "and outward gate normal (gate_normal_x, gate_normal_y) enables dogs to "
            "flank around closed exterior pen fences rather than pressing sheep into "
            "solid walls (especially seeds 53 and 67)."
        ),
        observation_mode="guided_gate_treatment",
        observation_size=NEW_OBSERVATION_DIM,
        observation_schema_version=GATE_OBSERVATION_SCHEMA_VERSION,
        is_control=False,
    ),
)


def get_gate_arm_spec(arm_id: str) -> GateArmSpec:
    """Return the GateArmSpec matching *arm_id*."""
    for arm in GATE_EXPERIMENT_ARMS:
        if arm.arm_id == arm_id:
            return arm
    raise KeyError(f"Unknown gate experiment arm: {arm_id!r}")


def verify_parent_checkpoint(
    parent_checkpoint: str | Path = DEFAULT_PARENT_CHECKPOINT,
    parent_metadata: str | Path = DEFAULT_PARENT_METADATA,
    *,
    expected_sha256: str = EXPECTED_PARENT_SHA256,
) -> dict[str, Any]:
    """Verify parent checkpoint SHA-256, SB3 hyperparameters, and Stage 34 metadata."""
    ckpt_path = Path(parent_checkpoint)
    meta_path = Path(parent_metadata)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Parent checkpoint not found: {ckpt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Parent metadata JSON not found: {meta_path}")

    actual_sha256 = sha256_file(ckpt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Parent checkpoint SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    env_cfg = meta_payload.get("environment_config", {})
    policy_cfg = meta_payload.get("policy_config", {})

    if int(env_cfg.get("dogs", -1)) != DOGS_PER_TEAM:
        raise ValueError(f"Expected {DOGS_PER_TEAM} dogs in parent metadata, got {env_cfg.get('dogs')}")
    if int(env_cfg.get("sheep", -1)) != SHEEP_PER_FLOCK:
        raise ValueError(f"Expected {SHEEP_PER_FLOCK} sheep in parent metadata, got {env_cfg.get('sheep')}")
    if int(env_cfg.get("max_steps", -1)) != EPISODE_BUDGET_WORLD_STEPS:
        raise ValueError(
            f"Expected max_steps={EPISODE_BUDGET_WORLD_STEPS}, got {env_cfg.get('max_steps')}"
        )
    if int(env_cfg.get("no_progress_window", -1)) != NO_PROGRESS_WINDOW_WORLD_STEPS:
        raise ValueError(
            f"Expected no_progress_window={NO_PROGRESS_WINDOW_WORLD_STEPS}, "
            f"got {env_cfg.get('no_progress_window')}"
        )
    if int(env_cfg.get("curriculum_stage", -1)) != CURRICULUM_STAGE:
        raise ValueError(
            f"Expected curriculum_stage={CURRICULUM_STAGE}, got {env_cfg.get('curriculum_stage')}"
        )

    model = MaskablePPO.load(str(ckpt_path), device="cpu")
    obs_shape = tuple(int(v) for v in (getattr(model.observation_space, "shape", ()) or ()))
    if obs_shape != (OLD_OBSERVATION_DIM,):
        raise ValueError(f"Expected parent observation_space ({OLD_OBSERVATION_DIM},), got {obs_shape}")
    if int(model.action_space.n) != len(ACTION_ORDER):
        raise ValueError(f"Expected action_space.n={len(ACTION_ORDER)}, got {model.action_space.n}")

    net_arch = tuple(
        int(v)
        for v in (
            model.policy_kwargs.get("net_arch", ())
            if isinstance(getattr(model, "policy_kwargs", None), dict)
            else ()
        )
    )
    if net_arch != EXPECTED_PARENT_ARCH:
        raise ValueError(f"Expected net_arch={list(EXPECTED_PARENT_ARCH)}, got {list(net_arch)}")

    lr_val = float(model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate)
    if not np.isclose(float(model.gamma), EXPECTED_PARENT_GAMMA, atol=1e-9):
        raise ValueError(f"Expected gamma={EXPECTED_PARENT_GAMMA}, got {model.gamma}")
    if not np.isclose(float(model.gae_lambda), EXPECTED_PARENT_GAE_LAMBDA, atol=1e-9):
        raise ValueError(f"Expected gae_lambda={EXPECTED_PARENT_GAE_LAMBDA}, got {model.gae_lambda}")
    if not np.isclose(float(model.ent_coef), EXPECTED_PARENT_ENT_COEF, atol=1e-9):
        raise ValueError(f"Expected ent_coef={EXPECTED_PARENT_ENT_COEF}, got {model.ent_coef}")
    if not np.isclose(lr_val, EXPECTED_PARENT_LEARNING_RATE, atol=1e-12):
        raise ValueError(f"Expected learning_rate={EXPECTED_PARENT_LEARNING_RATE}, got {lr_val}")
    if int(model.n_steps) != EXPECTED_PARENT_N_STEPS:
        raise ValueError(f"Expected n_steps={EXPECTED_PARENT_N_STEPS}, got {model.n_steps}")
    if int(model.batch_size) != EXPECTED_PARENT_BATCH_SIZE:
        raise ValueError(f"Expected batch_size={EXPECTED_PARENT_BATCH_SIZE}, got {model.batch_size}")

    return {
        "checkpoint_path": str(ckpt_path),
        "metadata_path": str(meta_path),
        "sha256": actual_sha256,
        "checkpoint_id": meta_payload.get("checkpoint_id"),
        "policy_version": meta_payload.get("policy_version"),
        "global_timestep": meta_payload.get("global_timestep"),
        "observation_size": OLD_OBSERVATION_DIM,
        "action_size": len(ACTION_ORDER),
        "net_arch": list(net_arch),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "ent_coef": float(model.ent_coef),
        "learning_rate": lr_val,
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "curriculum_stage": int(env_cfg["curriculum_stage"]),
        "dogs": int(env_cfg["dogs"]),
        "sheep": int(env_cfg["sheep"]),
        "max_steps": int(env_cfg["max_steps"]),
        "no_progress_window": int(env_cfg["no_progress_window"]),
        "metadata_policy_config": policy_cfg,
    }


def build_gate_arm_config(
    parent_metadata_payload: dict[str, Any],
    arm_spec: GateArmSpec,
    *,
    train_seed: int,
    output_dir: str | Path,
) -> LabConfig:
    """Build a locked Stage 34 LabConfig for one *(arm, seed)* workspace."""
    base = LabConfig()
    training_dict = copy.deepcopy(base.training)
    env_dict = dict(parent_metadata_payload.get("environment_config", {}))
    rewards_dict = copy.deepcopy(parent_metadata_payload.get("reward_config", {}))

    if "instincts" in rewards_dict and isinstance(rewards_dict["instincts"], dict):
        rewards_dict["instincts"]["curriculum_stage"] = CURRICULUM_STAGE
    env_dict["curriculum_stage"] = CURRICULUM_STAGE

    t_payload = base.to_dict()["training"]
    t_payload.update(
        {
            "train_seed": int(train_seed),
            "evaluation_seed": DEVELOPMENT_EVALUATION_SEEDS[0],
            "evaluation_seeds": list(DEVELOPMENT_EVALUATION_SEEDS),
            "neural_hidden_sizes": list(EXPECTED_PARENT_ARCH),
            "gamma": EXPECTED_PARENT_GAMMA,
            "gae_lambda": EXPECTED_PARENT_GAE_LAMBDA,
            "entropy_coef": EXPECTED_PARENT_ENT_COEF,
            "learning_rate": EXPECTED_PARENT_LEARNING_RATE,
            "rollout_steps": EXPECTED_PARENT_N_STEPS,
            "batch_size": EXPECTED_PARENT_BATCH_SIZE,
            "observation_mode": arm_spec.observation_mode,
            "enable_adaptive_learning": False,
            "auto_rollback_on_collapse": False,
            "backup_enabled": False,
            "wandb_enabled": False,
            "output_dir": str(output_dir),
            "episodes": max(1, PILOT_TRANSITIONS_PER_ARM_SEED),
        }
    )
    del training_dict
    payload = {
        "environment": env_dict,
        "rewards": rewards_dict,
        "training": t_payload,
        "policy": base.to_dict()["policy"],
    }
    return LabConfig.from_dict(payload)


def compute_fence_stall_duration(
    frames: list[dict[str, Any]],
    pen_origin: Point | tuple[int, int],
    pen_width: int,
    pen_height: int,
    pen_opening: str,
    *,
    progress_epsilon: float = 0.1,
) -> int:
    """Count world steps where an unpenned sheep is fence-adjacent with no penning or route progress.

    A sheep is fence-adjacent when its Chebyshev distance to any cell in
    ``Pen.fence_cells()`` is ``<= 1``. Progress requires either an increase in
    ``penned_count`` or a strict reduction in ``average_distance_to_pen`` below
    the running best by more than ``progress_epsilon``.
    """
    if not frames:
        return 0
    origin_pt = (
        pen_origin
        if isinstance(pen_origin, Point)
        else Point(int(pen_origin[0]), int(pen_origin[1]))
    )
    pen = Pen(
        origin=origin_pt,
        width=int(pen_width),
        height=int(pen_height),
        opening=str(pen_opening),
    )
    fence_cells = pen.fence_cells()

    stall_steps = 0
    prev_penned: int | None = None
    best_route_dist: float | None = None

    for frame in frames:
        snap = frame.get("snapshot", frame)
        sheep_list = snap.get("sheep", ())
        penned_count = snap.get("penned_count")
        if penned_count is None:
            penned_count = sum(1 for s in sheep_list if s.get("penned", False))
        penned_count = int(penned_count)

        raw_avg_dist = snap.get("average_distance_to_pen")
        avg_dist = float(raw_avg_dist) if raw_avg_dist is not None else None

        unpenned_positions: list[Point] = []
        for s in sheep_list:
            if s.get("penned", False):
                continue
            pos = s.get("position")
            if isinstance(pos, dict):
                sx, sy = int(pos["x"]), int(pos["y"])
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                sx, sy = int(pos[0]), int(pos[1])
            else:
                sx, sy = int(s["x"]), int(s["y"])
            unpenned_positions.append(Point(sx, sy))

        fence_adjacent = any(
            max(abs(sp.x - fc.x), abs(sp.y - fc.y)) <= 1
            for sp in unpenned_positions
            for fc in fence_cells
        )

        if prev_penned is None:
            prev_penned = penned_count
            if avg_dist is not None:
                best_route_dist = avg_dist
            continue

        penning_progress = penned_count > prev_penned
        if avg_dist is not None and best_route_dist is not None:
            route_progress = avg_dist < (best_route_dist - progress_epsilon)
        else:
            route_progress = False

        if penning_progress:
            prev_penned = penned_count
        if avg_dist is not None and (best_route_dist is None or avg_dist < best_route_dist):
            best_route_dist = avg_dist

        if fence_adjacent and not penning_progress and not route_progress:
            stall_steps += 1

    return stall_steps


def build_metric_and_decision_spec() -> dict[str, Any]:
    """Return explicit metrics, denominators, stopping conditions, and decision rules."""
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "comparison_design": (
            "Arm B (58-input explicit gate features) vs Arm A (equal-budget 58-input "
            "zero-padded gate control) from identical Stage 34 parent checkpoint."
        ),
        "metrics": {
            "full_flock_success_rate": {
                "definition": "Fraction of evaluation episodes with all 6/6 sheep penned within 2,320 world steps.",
                "denominator_per_training_seed": len(DEVELOPMENT_EVALUATION_SEEDS),
                "denominator_pooled_across_seeds": len(DEVELOPMENT_EVALUATION_SEEDS) * len(PAIRED_TRAINING_SEEDS),
            },
            "timeout_rate": {
                "definition": "Fraction of evaluation episodes reaching max_steps=2,320 without full-flock penning.",
                "denominator_per_training_seed": len(DEVELOPMENT_EVALUATION_SEEDS),
            },
            "no_progress_stop_rate": {
                "definition": "Fraction of evaluation episodes terminating early due to no_progress_window=780.",
                "denominator_per_training_seed": len(DEVELOPMENT_EVALUATION_SEEDS),
            },
            "steps_to_success": {
                "definition": "Mean/median world steps across successful episodes only; failures reported separately.",
            },
            "fence_stall_duration": {
                "definition": (
                    "World steps where at least one unpenned sheep is Chebyshev-adjacent (<=1) "
                    "to a closed Pen.fence_cells() cell while making neither penning nor route progress."
                ),
            },
            "targeted_failure_seeds": list(TARGETED_FAILURE_SEEDS),
            "previously_solved_regression_seeds": list(PREVIOUSLY_SOLVED_SEEDS),
            "uncertainty_reporting": "Per-training-seed (101, 202, 303) outcomes plus pooled Wilson 95% CI.",
        },
        "stopping_conditions": {
            "budget_completion": (
                f"Exact {PILOT_TRANSITIONS_PER_ARM_SEED} policy transitions "
                f"({PILOT_ROLLOUT_STEPS * 44} = 44 rollouts = 30,037.33 world steps) per arm-seed."
            ),
            "wall_clock_limit_per_arm_seed_seconds": MAX_WALL_SECONDS_PER_ARM_SEED,
            "total_wall_clock_limit_seconds": MAX_TOTAL_WALL_SECONDS,
            "numerical_instability": "Abort immediately on NaN/Inf loss or non-finite policy parameters.",
            "action_mask_violation": "Abort immediately if an illegal masked action is ever selected.",
        },
        "decision_rules": {
            "auto_promotion_enabled": False,
            "continue_to_full_run_if": (
                "Arm B outperforms equal-budget Arm A on pooled full-flock success rate across "
                "all 3 seeds, improves at least 1 of seeds {53, 67, 73} or reduces mean "
                "fence_stall_duration without regressing >1 previously solved seed."
            ),
            "reject_if": (
                "Arm B shows <= Arm A pooled success rate, zero improvement on fence_stall_duration, "
                "or net regressions on previously solved seeds {11, 23, 37, 41, 59, 61, 71}."
            ),
            "held_out_evaluation_status": HELD_OUT_EVALUATION_STATUS,
        },
    }


def verify_converted_model_equivalence(
    parent_checkpoint: str | Path,
    arm_model_paths: dict[str, str | Path],
    base_config: LabConfig,
    *,
    atol: float = EQUIVALENCE_ATOL,
) -> dict[str, Any]:
    """Verify bit-for-bit weight preservation, fresh optimizer state, and forward-pass equivalence."""
    parent_model = MaskablePPO.load(str(parent_checkpoint), device="cpu")
    parent_policy = parent_model.policy
    parent_pi_w = parent_policy.mlp_extractor.policy_net[0].weight.detach().clone()
    parent_pi_b = parent_policy.mlp_extractor.policy_net[0].bias.detach().clone()
    parent_vf_w = parent_policy.mlp_extractor.value_net[0].weight.detach().clone()
    parent_vf_b = parent_policy.mlp_extractor.value_net[0].bias.detach().clone()

    # Collect representative 54-dim observations from a deterministic Stage 34 episode.
    env54 = SheepdogEnvironment(base_config)
    env54._observation_builder = build_observation_builder("guided")
    env54.reset(seed=DEVELOPMENT_EVALUATION_SEEDS[0])
    rep_obs54: list[np.ndarray] = []
    rep_masks: list[np.ndarray] = []
    for dog_idx in range(DOGS_PER_TEAM):
        obs54 = np.asarray(env54.build_observation_for_dog(dog_idx).values, dtype=np.float32)
        mask_dict = env54.action_mask_for_dog(dog_idx, policy_mode="neural_policy")
        mask_arr = np.array([bool(mask_dict[act]) for act in ACTION_ORDER], dtype=bool)
        rep_obs54.append(obs54)
        rep_masks.append(mask_arr)

    # Also test a non-trivial partial legal mask.
    partial_mask = np.array(
        [True, False, True, True, False, True, False, True, False], dtype=bool
    )
    appended_variants = (
        np.zeros(4, dtype=np.float32),
        np.array([0.42, -0.73, 1.0, -1.0], dtype=np.float32),
    )

    arm_reports: dict[str, Any] = {}
    param_counts: dict[str, int] = {}

    for arm_id, model_path in arm_model_paths.items():
        arm_model = MaskablePPO.load(str(model_path), device="cpu")
        arm_policy = arm_model.policy

        if tuple(arm_model.observation_space.shape) != (NEW_OBSERVATION_DIM,):
            raise ValueError(f"{arm_id}: expected observation_space ({NEW_OBSERVATION_DIM},)")
        if tuple(arm_policy.observation_space.shape) != (NEW_OBSERVATION_DIM,):
            raise ValueError(f"{arm_id}: expected policy.observation_space ({NEW_OBSERVATION_DIM},)")

        pi_w = arm_policy.mlp_extractor.policy_net[0].weight.detach()
        pi_b = arm_policy.mlp_extractor.policy_net[0].bias.detach()
        vf_w = arm_policy.mlp_extractor.value_net[0].weight.detach()
        vf_b = arm_policy.mlp_extractor.value_net[0].bias.detach()

        if not torch.equal(pi_w[:, :OLD_OBSERVATION_DIM], parent_pi_w):
            raise AssertionError(f"{arm_id}: actor [:, :54] weights differ from parent")
        if not torch.equal(pi_b, parent_pi_b):
            raise AssertionError(f"{arm_id}: actor bias differs from parent")
        if not torch.all(pi_w[:, OLD_OBSERVATION_DIM:] == 0.0):
            raise AssertionError(f"{arm_id}: actor [:, 54:58] weights are not exact zero")
        if not torch.equal(vf_w[:, :OLD_OBSERVATION_DIM], parent_vf_w):
            raise AssertionError(f"{arm_id}: critic [:, :54] weights differ from parent")
        if not torch.equal(vf_b, parent_vf_b):
            raise AssertionError(f"{arm_id}: critic bias differs from parent")
        if not torch.all(vf_w[:, OLD_OBSERVATION_DIM:] == 0.0):
            raise AssertionError(f"{arm_id}: critic [:, 54:58] weights are not exact zero")

        if len(arm_policy.optimizer.state) != 0:
            raise AssertionError(f"{arm_id}: optimizer state is not empty after reset")

        param_counts[arm_id] = sum(p.numel() for p in arm_policy.parameters())

        max_logit_diff = 0.0
        max_prob_diff = 0.0
        max_val_diff = 0.0

        with torch.no_grad():
            for obs54, base_mask in zip(rep_obs54, rep_masks, strict=True):
                obs54_t = parent_policy.obs_to_tensor(obs54.reshape(1, -1))[0]
                p_val = float(parent_policy.predict_values(obs54_t).cpu().numpy().reshape(-1)[0])

                for extra in appended_variants:
                    obs58 = np.concatenate([obs54, extra]).astype(np.float32)
                    obs58_t = arm_policy.obs_to_tensor(obs58.reshape(1, -1))[0]
                    a_val = float(arm_policy.predict_values(obs58_t).cpu().numpy().reshape(-1)[0])
                    max_val_diff = max(max_val_diff, abs(p_val - a_val))

                    for mask in (base_mask, partial_mask):
                        m_arr = mask.reshape(1, -1)
                        p_dist = parent_policy.get_distribution(obs54_t, action_masks=m_arr)
                        a_dist = arm_policy.get_distribution(obs58_t, action_masks=m_arr)
                        p_logits = p_dist.distribution.logits.cpu().numpy().reshape(-1)
                        a_logits = a_dist.distribution.logits.cpu().numpy().reshape(-1)
                        p_probs = p_dist.distribution.probs.cpu().numpy().reshape(-1)
                        a_probs = a_dist.distribution.probs.cpu().numpy().reshape(-1)
                        finite_mask = np.isfinite(p_logits) | np.isfinite(a_logits)
                        if np.any(finite_mask):
                            max_logit_diff = max(
                                max_logit_diff,
                                float(np.max(np.abs(p_logits[finite_mask] - a_logits[finite_mask]))),
                            )
                        max_prob_diff = max(max_prob_diff, float(np.max(np.abs(p_probs - a_probs))))

        if (
            max_logit_diff > atol
            or max_prob_diff > EQUIVALENCE_PROB_ATOL
            or max_val_diff > atol
        ):
            raise AssertionError(
                f"{arm_id}: numerical equivalence exceeded tolerance "
                f"(atol={atol}, prob_atol={EQUIVALENCE_PROB_ATOL}): "
                f"logits={max_logit_diff}, probs={max_prob_diff}, val={max_val_diff}"
            )

        arm_reports[arm_id] = {
            "parameter_count": param_counts[arm_id],
            "optimizer_state_entries": len(arm_policy.optimizer.state),
            "max_abs_logit_diff": max_logit_diff,
            "max_abs_masked_prob_diff": max_prob_diff,
            "max_abs_value_diff": max_val_diff,
            "within_atol": True,
        }

    if len(set(param_counts.values())) != 1:
        raise AssertionError(f"Control and treatment parameter counts differ: {param_counts}")

    # Bounded deterministic rollout equivalence across parent (54), control (58), treatment (58).
    rollout_steps = 15
    rollout_traces: dict[str, list[tuple[Any, ...]]] = {}
    for label, obs_mode, mdl in (
        ("parent_54", "guided", parent_model),
        ("arm_a_control_zero_gate", "guided_gate_control", MaskablePPO.load(str(arm_model_paths["arm_a_control_zero_gate"]), device="cpu")),
        ("arm_b_treatment_gate_info", "guided_gate_treatment", MaskablePPO.load(str(arm_model_paths["arm_b_treatment_gate_info"]), device="cpu")),
    ):
        cfg_dict = base_config.to_dict()
        cfg_dict["training"]["observation_mode"] = obs_mode
        env = SheepdogEnvironment(LabConfig.from_dict(cfg_dict))
        env._observation_builder = build_observation_builder(obs_mode)
        env.reset(seed=DEVELOPMENT_EVALUATION_SEEDS[0])
        policy = NeuralPolicy(
            model=mdl,
            model_path=Path("."),
            config=NeuralPolicyConfig(
                hidden_sizes=EXPECTED_PARENT_ARCH,
                observation_size=int(mdl.observation_space.shape[0]),
                observation_mode=obs_mode,
            ),
        )
        trace: list[tuple[Any, ...]] = []
        for _ in range(rollout_steps):
            actions = policy.select_actions(env, deterministic=True)
            snapshot, reward_breakdown = env.step(actions)
            trace.append(
                (
                    tuple(actions),
                    round(float(reward_breakdown.total), 8),
                    tuple((d.position.x, d.position.y) for d in env.dogs),
                    tuple((s.position.x, s.position.y, s.penned) for s in env.sheep),
                )
            )
            if snapshot.terminated:
                break
        rollout_traces[label] = trace

    parent_trace = rollout_traces["parent_54"]
    for arm_id in arm_model_paths:
        if rollout_traces[arm_id] != parent_trace:
            raise AssertionError(f"Bounded deterministic rollout diverged for {arm_id}")

    return {
        "atol": atol,
        "rollout_steps_checked": len(parent_trace),
        "rollout_bit_for_bit_identical": True,
        "arms": arm_reports,
    }


def prepare_gate_experiment_workspace(
    experiment_root: str | Path,
    *,
    parent_checkpoint: str | Path = DEFAULT_PARENT_CHECKPOINT,
    parent_metadata: str | Path = DEFAULT_PARENT_METADATA,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create isolated per-(arm, seed) workspaces with converted 58-input models.

    Never modifies the parent checkpoint and never starts training.
    """
    root = Path(experiment_root)
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Experiment root {root} already exists and is non-empty; pass overwrite=True to replace."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    parent_info = verify_parent_checkpoint(parent_checkpoint, parent_metadata)
    parent_meta_payload = json.loads(Path(parent_metadata).read_text(encoding="utf-8"))

    budget = build_transition_budget(
        policy_transitions_per_arm_seed=PILOT_TRANSITIONS_PER_ARM_SEED,
        rollout_steps=PILOT_ROLLOUT_STEPS,
        checkpoint_count=PILOT_CHECKPOINT_COUNT,
        num_arms=len(GATE_EXPERIMENT_ARMS),
        num_seeds=len(PAIRED_TRAINING_SEEDS),
    )

    arm_entries: dict[str, Any] = {}
    representative_arm_models: dict[str, Path] = {}
    base_config_for_check: LabConfig | None = None

    for arm in GATE_EXPERIMENT_ARMS:
        seed_entries: dict[str, Any] = {}
        for seed in PAIRED_TRAINING_SEEDS:
            ws_dir = root / arm.arm_id / f"seed_{seed}"
            ws_dir.mkdir(parents=True, exist_ok=True)
            arm_cfg = build_gate_arm_config(
                parent_meta_payload,
                arm,
                train_seed=seed,
                output_dir=ws_dir / "training_output",
            )
            if base_config_for_check is None:
                base_config_for_check = arm_cfg
            cfg_path = save_lab_config(arm_cfg, ws_dir / "config.json")
            init_model_path = ws_dir / "initial_model.zip"
            conversion_info = expand_policy_observation_inputs(
                parent_checkpoint,
                init_model_path,
                old_dim=OLD_OBSERVATION_DIM,
                new_dim=NEW_OBSERVATION_DIM,
                learning_rate=EXPECTED_PARENT_LEARNING_RATE,
            )
            if arm.arm_id not in representative_arm_models:
                representative_arm_models[arm.arm_id] = init_model_path
            seed_entries[str(seed)] = {
                "seed": seed,
                "workspace_dir": str(ws_dir),
                "config_path": str(cfg_path),
                "initial_model_path": str(init_model_path),
                "initial_model_sha256": sha256_file(init_model_path),
                "conversion": conversion_info,
                "status": "prepared_not_started",
            }
        arm_entries[arm.arm_id] = {
            "spec": arm.to_dict(),
            "seeds": seed_entries,
        }

    assert base_config_for_check is not None
    equivalence = verify_converted_model_equivalence(
        parent_checkpoint,
        representative_arm_models,
        base_config_for_check,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "prepared_not_started",
        "parent_checkpoint": parent_info,
        "git_provenance": collect_git_provenance(),
        "dependency_versions": collect_dependency_versions(),
        "paired_training_seeds": list(PAIRED_TRAINING_SEEDS),
        "development_evaluation_seeds": list(DEVELOPMENT_EVALUATION_SEEDS),
        "held_out_evaluation_status": HELD_OUT_EVALUATION_STATUS,
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "observation_schema_version": GATE_OBSERVATION_SCHEMA_VERSION,
        "budget": asdict(budget),
        "wall_clock_limits": {
            "per_arm_seed_seconds": MAX_WALL_SECONDS_PER_ARM_SEED,
            "total_seconds": MAX_TOTAL_WALL_SECONDS,
        },
        "diagnostic_telemetry_policy": {
            "enabled_during_training": False,
            "enabled_on_selected_eval_only": True,
            "telemetry_mode": "head_tail",
            "telemetry_head_steps": TELEMETRY_HEAD_STEPS,
            "telemetry_tail_steps": TELEMETRY_TAIL_STEPS,
        },
        "metric_and_decision_spec": build_metric_and_decision_spec(),
        "initial_equivalence_check": equivalence,
        "arms": arm_entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_gate_experiment_workspace(experiment_root: str | Path) -> dict[str, Any]:
    """Verify prepared workspace integrity, config parity, and initial model equivalence."""
    root = Path(experiment_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    parent_info = manifest["parent_checkpoint"]
    if parent_info["sha256"] != EXPECTED_PARENT_SHA256:
        raise ValueError("Manifest parent SHA-256 mismatch")

    checked_models = 0
    control_cfg_norm: dict[str, Any] | None = None

    for arm_id, arm_entry in manifest["arms"].items():
        spec = get_gate_arm_spec(arm_id)
        for seed_str, seed_entry in arm_entry["seeds"].items():
            cfg_path = Path(seed_entry["config_path"])
            model_path = Path(seed_entry["initial_model_path"])
            if not cfg_path.exists() or not model_path.exists():
                raise FileNotFoundError(f"Missing prepared files for {arm_id} seed {seed_str}")
            actual_sha = sha256_file(model_path)
            if actual_sha != seed_entry["initial_model_sha256"]:
                raise ValueError(f"Initial model SHA-256 mismatch for {arm_id} seed {seed_str}")

            cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg_dict["training"]["observation_mode"] != spec.observation_mode:
                raise ValueError(f"Config observation_mode mismatch for {arm_id} seed {seed_str}")
            if cfg_dict["training"]["enable_adaptive_learning"] is not False:
                raise ValueError("enable_adaptive_learning must be False")
            if cfg_dict["training"]["auto_rollback_on_collapse"] is not False:
                raise ValueError("auto_rollback_on_collapse must be False")

            norm = copy.deepcopy(cfg_dict)
            norm["training"].pop("observation_mode", None)
            norm["training"].pop("train_seed", None)
            norm["training"].pop("output_dir", None)
            if control_cfg_norm is None:
                control_cfg_norm = norm
            elif norm != control_cfg_norm:
                raise ValueError(f"Non-treatment config parity violation in {arm_id} seed {seed_str}")

            mdl = MaskablePPO.load(str(model_path), device="cpu")
            if tuple(mdl.observation_space.shape) != (NEW_OBSERVATION_DIM,):
                raise ValueError(f"Expected (58,) obs space for {arm_id} seed {seed_str}")
            checked_models += 1

    return {
        "verified": True,
        "experiment_root": str(root),
        "checked_arm_seeds": checked_models,
        "parent_sha256": parent_info["sha256"],
        "held_out_evaluation_status": manifest["held_out_evaluation_status"],
    }


def run_gate_experiment_arm(
    experiment_root: str | Path,
    arm_id: str,
    seed: int,
    *,
    max_wall_seconds: float = MAX_WALL_SECONDS_PER_ARM_SEED,
) -> dict[str, Any]:
    """Execute one isolated *(arm_id, seed)* pilot run and development evaluation."""
    root = Path(experiment_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    arm_entry = manifest["arms"][arm_id]
    seed_entry = arm_entry["seeds"][str(seed)]
    arm_spec = get_gate_arm_spec(arm_id)
    cfg = LabConfig.from_dict(json.loads(Path(seed_entry["config_path"]).read_text(encoding="utf-8")))
    init_model_path = Path(seed_entry["initial_model_path"])
    ws_dir = Path(seed_entry["workspace_dir"])

    policy_cfg_dict = NeuralPolicyConfig(
        hidden_sizes=EXPECTED_PARENT_ARCH,
        observation_size=NEW_OBSERVATION_DIM,
        observation_mode=arm_spec.observation_mode,
        observation_schema_version=arm_spec.observation_schema_version,
    ).to_dict()
    policy = NeuralPolicy.load(init_model_path, cfg, policy_config=policy_cfg_dict)

    start_time = time.monotonic()
    stopped_reason = "budget_completed"

    def _should_stop() -> bool:
        nonlocal stopped_reason
        if time.monotonic() - start_time >= max_wall_seconds:
            stopped_reason = "wall_clock_limit_reached"
            return True
        for param in policy.model.policy.parameters():
            if not torch.isfinite(param).all():
                stopped_reason = "non_finite_parameters"
                return True
        return False

    trainer = MaskablePPOTrainer(cfg)
    prev_capture = get_global_capture_policy()
    set_global_capture_policy(CapturePolicy(mode="off"))
    try:
        train_result = trainer.train(
            policy=policy,
            total_timesteps=PILOT_TRANSITIONS_PER_ARM_SEED,
            should_stop=_should_stop,
        )
    finally:
        set_global_capture_policy(prev_capture)

    final_model_path = ws_dir / "final_model.zip"
    policy.save(final_model_path)

    result_payload = {
        "arm_id": arm_id,
        "seed": seed,
        "stopped_reason": stopped_reason,
        "elapsed_seconds": time.monotonic() - start_time,
        "final_model_path": str(final_model_path),
        "final_model_sha256": sha256_file(final_model_path),
        "train_summary": getattr(train_result, "__dict__", str(train_result)),
    }
    (ws_dir / "run_result.json").write_text(
        json.dumps(result_payload, indent=2, default=str), encoding="utf-8"
    )
    return result_payload
