"""Tests for the Stage 34 gate-information observation ablation preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.entities import Pen, Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.gate_information_experiment import (
    DEFAULT_PARENT_CHECKPOINT,
    DEFAULT_PARENT_METADATA,
    DEVELOPMENT_EVALUATION_SEEDS,
    EXPECTED_PARENT_SHA256,
    HELD_OUT_EVALUATION_STATUS,
    PAIRED_TRAINING_SEEDS,
    PILOT_CHECKPOINT_COUNT,
    PILOT_ROLLOUT_STEPS,
    PILOT_TRANSITIONS_PER_ARM_SEED,
    build_metric_and_decision_spec,
    compute_fence_stall_duration,
    prepare_gate_experiment_workspace,
    verify_converted_model_equivalence,
    verify_gate_experiment_workspace,
)
from sheepdog.observations import (
    GATE_FEATURE_NAMES,
    GATE_OBSERVATION_SCHEMA_VERSION,
    EmergentObservationBuilder,
    GateAwareObservationBuilder,
    RoleAwareObservationBuilder,
    build_observation_builder,
)
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    expand_policy_observation_inputs,
)
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_stage34_config(observation_mode: str = "guided") -> LabConfig:
    base = LabConfig().to_dict()
    base["environment"]["dogs"] = 3
    base["environment"]["sheep"] = 6
    base["environment"]["curriculum_stage"] = 34
    base["environment"]["max_steps"] = 2320
    base["environment"]["no_progress_window"] = 780
    base["training"]["observation_mode"] = observation_mode
    base["training"]["neural_hidden_sizes"] = [32, 32]
    base["training"]["rollout_steps"] = 64
    base["training"]["batch_size"] = 32
    return LabConfig.from_dict(base)


def test_default_observation_modes_unchanged() -> None:
    """Default TrainingConfig and existing 54/50-dim observation builders remain unchanged."""
    assert TrainingConfig().observation_mode == "guided"
    cfg = _make_stage34_config("guided")
    env = SheepdogEnvironment(cfg)
    env.reset(seed=11)

    guided_obs = RoleAwareObservationBuilder().build(env, 0)
    assert len(guided_obs.values) == 54
    assert len(guided_obs.feature_names) == 54
    assert isinstance(build_observation_builder("guided"), RoleAwareObservationBuilder)
    assert not isinstance(build_observation_builder("guided"), GateAwareObservationBuilder)

    emergent_obs = EmergentObservationBuilder().build(env, 0)
    assert len(emergent_obs.values) == 50
    assert isinstance(build_observation_builder("emergent"), EmergentObservationBuilder)

    with pytest.raises(ValueError, match="Unsupported observation_mode"):
        build_observation_builder("invalid_mode")


@pytest.mark.parametrize(
    ("opening", "expected_normal"),
    [
        ("bottom", (0.0, 1.0)),
        ("top", (0.0, -1.0)),
        ("left", (-1.0, 0.0)),
        ("right", (1.0, 0.0)),
    ],
)
def test_gate_observation_all_four_openings_and_bounds(
    opening: str, expected_normal: tuple[float, float]
) -> None:
    """All 4 gate openings produce exact outward normals, normalized offsets in [-1, 1], and 58 features."""
    cfg_dict = _make_stage34_config("guided_gate_treatment").to_dict()
    cfg_dict["environment"]["pen_opening"] = "bottom"
    cfg = LabConfig.from_dict(cfg_dict)
    env = SheepdogEnvironment(cfg)
    env.reset(seed=23)
    env._pen = Pen(origin=Point(35, 25), width=10, height=10, opening=opening)

    base_obs = RoleAwareObservationBuilder().build(env, 0)
    ctrl_obs = build_observation_builder("guided_gate_control").build(env, 0)
    treat_obs = build_observation_builder("guided_gate_treatment").build(env, 0)

    assert len(ctrl_obs.values) == 58
    assert len(treat_obs.values) == 58
    assert ctrl_obs.feature_names == base_obs.feature_names + GATE_FEATURE_NAMES
    assert treat_obs.feature_names == base_obs.feature_names + GATE_FEATURE_NAMES
    assert ctrl_obs.values[:54] == base_obs.values
    assert treat_obs.values[:54] == base_obs.values
    assert ctrl_obs.values[54:] == (0.0, 0.0, 0.0, 0.0)

    gate_center = env._gate_entrance_center()
    dog_pos = env.dogs[0].position
    expected_dx = max(-1.0, min(1.0, (gate_center.x - dog_pos.x) / (cfg.environment.width - 1)))
    expected_dy = max(-1.0, min(1.0, (gate_center.y - dog_pos.y) / (cfg.environment.height - 1)))

    dx, dy, nx, ny = treat_obs.values[54:]
    assert dx == pytest.approx(expected_dx)
    assert dy == pytest.approx(expected_dy)
    assert (nx, ny) == expected_normal
    assert all(-1.0 <= v <= 1.0 for v in treat_obs.values[54:])
    assert treat_obs.metadata["observation_schema_version"] == GATE_OBSERVATION_SCHEMA_VERSION


def test_rl_adapter_observation_space_sizing() -> None:
    """SheepdogRLAdapter automatically sizes observation_space to (58,) for gate modes and (54,) for guided."""
    for mode, expected_dim in (
        ("guided", 54),
        ("guided_gate_control", 58),
        ("guided_gate_treatment", 58),
    ):
        adapter = SheepdogRLAdapter(_make_stage34_config(mode))
        assert adapter.observation_space.shape == (expected_dim,)
        obs, _info = adapter.reset(seed=11)
        assert obs.shape == (expected_dim,)


def test_synthetic_54_to_58_conversion_equivalence_and_schema_rejection(tmp_path: Path) -> None:
    """Verify bit-for-bit weight preservation, fresh optimizer, equivalence, legal masks, and round-trip rejection."""
    cfg54 = _make_stage34_config("guided")
    vec54 = DummyVecEnv([lambda: SheepdogRLAdapter(cfg54)])
    parent_model = MaskablePPO(
        "MlpPolicy",
        vec54,
        learning_rate=5e-5,
        n_steps=64,
        batch_size=32,
        gamma=0.997,
        gae_lambda=0.98,
        ent_coef=0.0075,
        seed=7,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )
    parent_path = tmp_path / "parent_54.zip"
    parent_model.save(str(parent_path))

    ctrl_path = tmp_path / "ctrl_58.zip"
    treat_path = tmp_path / "treat_58.zip"
    expand_policy_observation_inputs(parent_path, ctrl_path, old_dim=54, new_dim=58, learning_rate=5e-5)
    expand_policy_observation_inputs(parent_path, treat_path, old_dim=54, new_dim=58, learning_rate=5e-5)

    # Verify actor & critic weights, optimizer reset, and rollout buffer shape
    p_loaded = MaskablePPO.load(str(parent_path), device="cpu")
    c_loaded = MaskablePPO.load(str(ctrl_path), device="cpu")
    t_loaded = MaskablePPO.load(str(treat_path), device="cpu")

    assert tuple(c_loaded.observation_space.shape) == (58,)
    assert tuple(t_loaded.observation_space.shape) == (58,)
    assert tuple(c_loaded.rollout_buffer.obs_shape) == (58,)
    assert len(c_loaded.policy.optimizer.state) == 0
    assert len(t_loaded.policy.optimizer.state) == 0

    for arm_m in (c_loaded, t_loaded):
        assert torch.equal(
            arm_m.policy.mlp_extractor.policy_net[0].weight[:, :54],
            p_loaded.policy.mlp_extractor.policy_net[0].weight,
        )
        assert torch.all(arm_m.policy.mlp_extractor.policy_net[0].weight[:, 54:] == 0.0)
        assert torch.equal(
            arm_m.policy.mlp_extractor.value_net[0].weight[:, :54],
            p_loaded.policy.mlp_extractor.value_net[0].weight,
        )
        assert torch.all(arm_m.policy.mlp_extractor.value_net[0].weight[:, 54:] == 0.0)

    # Verify optimizer is bound to the new 58-input parameters
    opt_param_ids = {id(p) for g in c_loaded.policy.optimizer.param_groups for p in g["params"]}
    model_param_ids = {id(p) for p in c_loaded.policy.parameters()}
    assert opt_param_ids == model_param_ids

    # Verify numerical equivalence (logits, masked probs, critic values) and rollout invariance
    eq_report = verify_converted_model_equivalence(
        parent_path,
        {
            "arm_a_control_zero_gate": ctrl_path,
            "arm_b_treatment_gate_info": treat_path,
        },
        cfg54,
    )
    assert eq_report["rollout_bit_for_bit_identical"] is True
    for arm_id in ("arm_a_control_zero_gate", "arm_b_treatment_gate_info"):
        assert eq_report["arms"][arm_id]["max_abs_logit_diff"] <= 1e-5
        assert eq_report["arms"][arm_id]["max_abs_masked_prob_diff"] <= 1e-6
        assert eq_report["arms"][arm_id]["max_abs_value_diff"] <= 1e-5

    # Legal action selection check
    cfg_ctrl = _make_stage34_config("guided_gate_control")
    policy_ctrl = NeuralPolicy.load(
        ctrl_path,
        cfg_ctrl,
        policy_config=NeuralPolicyConfig(
            hidden_sizes=(32, 32),
            observation_size=58,
            observation_mode="guided_gate_control",
            observation_schema_version=GATE_OBSERVATION_SCHEMA_VERSION,
        ).to_dict(),
    )
    env_ctrl = SheepdogEnvironment(cfg_ctrl)
    env_ctrl._observation_builder = build_observation_builder("guided_gate_control")
    env_ctrl.reset(seed=11)
    actions = policy_ctrl.select_actions(env_ctrl, deterministic=True)
    for dog_idx, act in enumerate(actions):
        legal_mask = env_ctrl.action_mask_for_dog(dog_idx, policy_mode="neural_policy")
        assert legal_mask[act] is True

    # Incompatible schema rejections
    with pytest.raises(ValueError):
        NeuralPolicy.load(ctrl_path, _make_stage34_config("guided"))
    with pytest.raises(ValueError, match="Incompatible observation_mode"):
        NeuralPolicy.load(
            ctrl_path,
            _make_stage34_config("guided_gate_treatment"),
            policy_config=NeuralPolicyConfig(
                hidden_sizes=(32, 32),
                observation_size=58,
                observation_mode="guided_gate_control",
                observation_schema_version=GATE_OBSERVATION_SCHEMA_VERSION,
            ).to_dict(),
        )
    with pytest.raises(ValueError, match="Incompatible policy_config observation_size"):
        NeuralPolicy.load(
            ctrl_path,
            cfg_ctrl,
            policy_config=NeuralPolicyConfig(
                hidden_sizes=(32, 32),
                observation_size=54,
            ).to_dict(),
        )


def test_fence_stall_duration_helper() -> None:
    """compute_fence_stall_duration counts fence-adjacent steps lacking penning/route progress."""
    pen = Pen(origin=Point(20, 20), width=10, height=10, opening="bottom")
    # Top closed fence is at y=19, x in [19..30]. Sheep at (24, 18) is adjacent (dy=1).
    frames: list[dict[str, Any]] = [
        {"penned_count": 0, "average_distance_to_pen": 15.0, "sheep": [{"x": 24, "y": 18, "penned": False}]},
        # Adjacent, no route or penning progress -> +1
        {"penned_count": 0, "average_distance_to_pen": 15.0, "sheep": [{"x": 24, "y": 18, "penned": False}]},
        # Adjacent, route progress > 0.1 -> +0
        {"penned_count": 0, "average_distance_to_pen": 14.5, "sheep": [{"x": 24, "y": 18, "penned": False}]},
        # Adjacent, jittered worse route dist -> +1
        {"penned_count": 0, "average_distance_to_pen": 14.7, "sheep": [{"x": 24, "y": 18, "penned": False}]},
        # Not adjacent (far away) -> +0
        {"penned_count": 0, "average_distance_to_pen": 14.7, "sheep": [{"x": 5, "y": 5, "penned": False}]},
        # Adjacent, but a sheep was penned -> +0
        {"penned_count": 1, "average_distance_to_pen": 14.7, "sheep": [{"x": 24, "y": 18, "penned": False}]},
    ]
    assert compute_fence_stall_duration(frames, pen.origin, pen.width, pen.height, pen.opening) == 2


@pytest.mark.skipif(
    not DEFAULT_PARENT_CHECKPOINT.exists(),
    reason="Real Stage 34 parent checkpoint not present in workspace",
)
def test_prepare_and_verify_real_parent_workspace_without_training(tmp_path: Path) -> None:
    """Read-only preparation against the real Stage 34 parent checkpoint verifies equivalence and parity."""
    exp_root = tmp_path / "gate_exp"
    manifest = prepare_gate_experiment_workspace(
        exp_root,
        parent_checkpoint=DEFAULT_PARENT_CHECKPOINT,
        parent_metadata=DEFAULT_PARENT_METADATA,
    )
    assert manifest["status"] == "prepared_not_started"
    assert manifest["parent_checkpoint"]["sha256"] == EXPECTED_PARENT_SHA256
    assert manifest["held_out_evaluation_status"] == HELD_OUT_EVALUATION_STATUS
    assert tuple(manifest["paired_training_seeds"]) == PAIRED_TRAINING_SEEDS
    assert tuple(manifest["development_evaluation_seeds"]) == DEVELOPMENT_EVALUATION_SEEDS
    assert manifest["budget"]["policy_transitions_per_arm_seed"] == PILOT_TRANSITIONS_PER_ARM_SEED
    assert manifest["budget"]["rollout_steps"] == PILOT_ROLLOUT_STEPS
    assert manifest["budget"]["checkpoint_count"] == PILOT_CHECKPOINT_COUNT
    assert manifest["diagnostic_telemetry_policy"]["enabled_during_training"] is False

    eq = manifest["initial_equivalence_check"]
    assert eq["rollout_bit_for_bit_identical"] is True
    for arm_id in ("arm_a_control_zero_gate", "arm_b_treatment_gate_info"):
        assert eq["arms"][arm_id]["max_abs_logit_diff"] <= 1e-5
        assert eq["arms"][arm_id]["max_abs_masked_prob_diff"] <= 1e-6
        assert eq["arms"][arm_id]["max_abs_value_diff"] <= 1e-5

    verification = verify_gate_experiment_workspace(exp_root)
    assert verification["verified"] is True
    assert verification["checked_arm_seeds"] == 6

    spec = build_metric_and_decision_spec()
    assert spec["decision_rules"]["auto_promotion_enabled"] is False
    assert spec["decision_rules"]["held_out_evaluation_status"] == "pending_manifest_audit"
