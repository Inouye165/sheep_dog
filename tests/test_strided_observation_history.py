"""Tests for strided temporal observation history, emergent_gate, loop-flag toggles, and 54->232 transfer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.config import LabConfig, TrainingConfig, validate_observation_history_strides
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.scenarios import AgentLayout, PenLayout, SavedScenario
from sheepdog.observations import (
    EMERGENT_GATE_OBSERVATION_SCHEMA_VERSION,
    GATE_FEATURE_NAMES,
    GATE_OBSERVATION_SCHEMA_VERSION,
    build_observation_builder,
)
from sheepdog.policies.neural import (
    NeuralPolicy,
    NeuralPolicyConfig,
    expand_policy_observation_inputs,
)
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_stage34_config(
    *,
    observation_mode: str = "guided",
    strides: tuple[int, ...] = (0,),
    include_loop_flags: bool = True,
    hidden_sizes: tuple[int, ...] = (32, 32),
) -> LabConfig:
    payload = LabConfig().to_dict()
    payload["environment"]["dogs"] = 3
    payload["environment"]["sheep"] = 6
    payload["environment"]["curriculum_stage"] = 34
    payload["environment"]["max_steps"] = 2320
    payload["environment"]["no_progress_window"] = 780
    payload["training"]["observation_mode"] = observation_mode
    payload["training"]["observation_history_strides"] = list(strides)
    payload["training"]["include_loop_flags_in_observation"] = include_loop_flags
    payload["training"]["neural_hidden_sizes"] = list(hidden_sizes)
    payload["training"]["rollout_steps"] = 64
    payload["training"]["batch_size"] = 32
    payload["training"]["ppo_env_workers"] = 1
    return LabConfig.from_dict(payload)


def test_config_roundtrip_and_stride_validation() -> None:
    """TrainingConfig defaults stay (0,) and True; LabConfig round-trips and validates strides."""
    default_t = TrainingConfig()
    assert default_t.observation_history_strides == (0,)
    assert default_t.include_loop_flags_in_observation is True

    legacy_dict = LabConfig().to_dict()
    legacy_dict["training"].pop("observation_history_strides")
    legacy_dict["training"].pop("include_loop_flags_in_observation")
    restored_legacy = LabConfig.from_dict(legacy_dict)
    assert restored_legacy.training.observation_history_strides == (0,)
    assert restored_legacy.training.include_loop_flags_in_observation is True

    cfg = _make_stage34_config(
        observation_mode="guided_gate_treatment",
        strides=(0, 4, 8, 16),
        include_loop_flags=False,
    )
    roundtripped = LabConfig.from_dict(cfg.to_dict())
    assert roundtripped.training.observation_history_strides == (0, 4, 8, 16)
    assert roundtripped.training.include_loop_flags_in_observation is False

    for bad in [(), (1, 4), (0, -1), (0, 4, 4), (0, 8, 4), (0, True)]:
        with pytest.raises(ValueError):
            validate_observation_history_strides(bad)  # type: ignore[arg-type]


def test_strided_history_buffer_padding_indexing_idempotency_and_reset() -> None:
    """Verify (0, 4, 8, 16) t=0 padding, exact stride indexing, idempotency, and reset clearing."""
    strides = (0, 4, 8, 16)
    cfg = _make_stage34_config(observation_mode="guided_gate_treatment", strides=strides)
    env = SheepdogEnvironment(cfg)
    env.reset(seed=53)

    base_builder = build_observation_builder("guided_gate_treatment")
    base_by_step: dict[int, tuple[float, ...]] = {}
    stacked_by_step: dict[int, tuple[float, ...]] = {}

    for step in range(21):
        assert env.step_count == step
        base_by_step[step] = base_builder.build(env, 0).values
        obs_first = env.build_observation_for_dog(0)
        obs_second = env.build_observation_for_dog(0)
        assert obs_first == obs_second
        assert len(obs_first.values) == 232
        assert obs_first.metadata["observation_history_strides"] == strides
        # Check feature naming: s=0 un-suffixed, s>0 suffixed with @t-{s}
        assert obs_first.feature_names[:58] == base_builder.build(env, 0).feature_names
        assert obs_first.feature_names[58].endswith("@t-4")
        assert obs_first.feature_names[116].endswith("@t-8")
        assert obs_first.feature_names[174].endswith("@t-16")
        stacked_by_step[step] = obs_first.values
        if step < 20:
            env.step(["wait", "wait", "wait"] if step % 2 == 0 else ["up", "right", "down"])

    for check_step in (0, 1, 4, 5, 16, 20):
        expected = (
            base_by_step[max(0, check_step - 0)]
            + base_by_step[max(0, check_step - 4)]
            + base_by_step[max(0, check_step - 8)]
            + base_by_step[max(0, check_step - 16)]
        )
        assert stacked_by_step[check_step] == expected

    # Reset clears history buffer
    env.reset(seed=53)
    assert env._obs_history_by_dog == {}
    obs_after_reset = env.build_observation_for_dog(0)
    assert obs_after_reset.values == base_by_step[0] * 4

    # reset_from_scenario also clears history buffer
    scenario = SavedScenario(
        id="test_reset_clear",
        name="test_reset_clear",
        created_at="2026-11-17T00:00:00Z",
        seed=53,
        seed_offset=cfg.environment.seed_offset,
        width=cfg.environment.width,
        height=cfg.environment.height,
        pen=PenLayout(
            origin_x=env.pen.origin.x,
            origin_y=env.pen.origin.y,
            width=env.pen.width,
            height=env.pen.height,
            opening=env.pen.opening,
        ),
        dogs=tuple(AgentLayout(index=i, x=d.position.x, y=d.position.y) for i, d in enumerate(env.dogs)),
        sheep=tuple(
            AgentLayout(index=i, x=s.position.x, y=s.position.y, personality=s.personality)
            for i, s in enumerate(env.sheep)
        ),
    )
    env.step(["up", "up", "up"])
    env.build_observation_for_dog(0)
    assert len(env._obs_history_by_dog[0]) > 0
    env.reset_from_scenario(scenario)
    assert env._obs_history_by_dog == {}


def test_emergent_gate_and_loop_flag_toggles() -> None:
    """emergent_gate omits TeamStrategy guidance; include_loop_flags=False drops loop flags (54/52 -> 216/208)."""
    for include_loop_flags, expected_base, expected_stacked in [
        (True, 54, 216),
        (False, 52, 208),
    ]:
        cfg_base = _make_stage34_config(
            observation_mode="emergent_gate",
            strides=(0,),
            include_loop_flags=include_loop_flags,
        )
        env_base = SheepdogEnvironment(cfg_base)
        env_base.reset(seed=11)
        obs_base = env_base.build_observation_for_dog(0)
        assert len(obs_base.values) == expected_base
        assert len(obs_base.feature_names) == expected_base
        assert obs_base.metadata["observation_schema_version"] == EMERGENT_GATE_OBSERVATION_SCHEMA_VERSION

        # Zero TeamStrategy role/target guidance
        assert not any(name.startswith("role_") for name in obs_base.feature_names)
        assert "target_x" not in obs_base.feature_names
        assert "target_y" not in obs_base.feature_names
        assert "distance_to_target" not in obs_base.feature_names
        for gname in GATE_FEATURE_NAMES:
            assert gname in obs_base.feature_names

        if include_loop_flags:
            assert "revisits_recent_position" in obs_base.feature_names
            assert "two_position_loop" in obs_base.feature_names
        else:
            assert "revisits_recent_position" not in obs_base.feature_names
            assert "two_position_loop" not in obs_base.feature_names

        cfg_stacked = _make_stage34_config(
            observation_mode="emergent_gate",
            strides=(0, 4, 8, 16),
            include_loop_flags=include_loop_flags,
        )
        adapter = SheepdogRLAdapter(cfg_stacked)
        assert adapter.observation_space.shape == (expected_stacked,)
        obs_arr, _ = adapter.reset(seed=11)
        assert obs_arr.shape == (expected_stacked,)


def test_neural_policy_strict_schema_and_54_to_232_transfer(tmp_path: Path) -> None:
    """Verify strict schema rejection and bit-exact 54->232 zero-padded transfer + 20-step equivalence."""
    cfg_54 = _make_stage34_config(observation_mode="guided", strides=(0,))
    cfg_58 = _make_stage34_config(observation_mode="guided_gate_treatment", strides=(0,))
    cfg_54_emergent_gate = _make_stage34_config(observation_mode="emergent_gate", strides=(0,))
    cfg_232 = _make_stage34_config(
        observation_mode="guided_gate_treatment",
        strides=(0, 4, 8, 16),
    )

    # Train zero steps on synthetic 54-input model
    adapter_54 = SheepdogRLAdapter(cfg_54)
    vec_54 = DummyVecEnv([lambda: adapter_54])
    model_54 = MaskablePPO(
        "MlpPolicy",
        vec_54,
        n_steps=64,
        batch_size=32,
        seed=7,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )
    path_54 = tmp_path / "model_54.zip"
    model_54.save(str(path_54))
    pconfig_54 = NeuralPolicyConfig(
        hidden_sizes=(32, 32),
        observation_size=54,
        observation_mode="guided",
        observation_schema_version=None,
    ).to_dict()

    # Transfer 54 -> 232
    path_232 = tmp_path / "model_232.zip"
    transfer_info = expand_policy_observation_inputs(
        path_54,
        path_232,
        old_dim=54,
        new_dim=232,
        learning_rate=1e-4,
        observation_mode="guided_gate_treatment",
        observation_history_strides=(0, 4, 8, 16),
        include_loop_flags_in_observation=True,
    )
    assert transfer_info["new_dim"] == 232
    assert transfer_info["policy_config"]["observation_schema_version"] == (
        f"{GATE_OBSERVATION_SCHEMA_VERSION}|strides=0,4,8,16"
    )

    # Strict rejections without explicit expansion
    with pytest.raises(ValueError, match=r"(Incompatible|Observation spaces do not match)"):
        NeuralPolicy.load(path_54, cfg_232, policy_config=pconfig_54)
    with pytest.raises(ValueError, match=r"(Incompatible|Observation spaces do not match)"):
        NeuralPolicy.load(path_54, cfg_54_emergent_gate, policy_config=pconfig_54)
    with pytest.raises(ValueError, match=r"(Incompatible|Observation spaces do not match)"):
        NeuralPolicy.load(path_232, cfg_54, policy_config=transfer_info["policy_config"])
    with pytest.raises(ValueError, match=r"(Incompatible|Observation spaces do not match)"):
        NeuralPolicy.load(path_232, cfg_58, policy_config=transfer_info["policy_config"])

    # Also test 58 -> 232 rejection
    path_58 = tmp_path / "model_58.zip"
    expand_policy_observation_inputs(
        path_54,
        path_58,
        old_dim=54,
        new_dim=58,
        observation_mode="guided_gate_treatment",
    )
    with pytest.raises(ValueError, match=r"(Incompatible|Observation spaces do not match)"):
        NeuralPolicy.load(path_58, cfg_232)

    # Valid load into 232-input config
    loaded_232 = NeuralPolicy.load(path_232, cfg_232, policy_config=transfer_info["policy_config"])
    assert loaded_232.config.observation_size == 232
    assert loaded_232.config.observation_history_strides == (0, 4, 8, 16)

    # Numerical logit/value equivalence
    import numpy as np

    src_model = MaskablePPO.load(str(path_54), device="cpu")
    dst_model = MaskablePPO.load(str(path_232), device="cpu")
    obs_54 = np.random.default_rng(53).uniform(-1.0, 1.0, size=(16, 54)).astype(np.float32)
    obs_232 = np.zeros((16, 232), dtype=np.float32)
    obs_232[:, :54] = obs_54
    with torch.no_grad():
        dist_54 = src_model.policy.get_distribution(torch.as_tensor(obs_54))
        dist_232 = dst_model.policy.get_distribution(torch.as_tensor(obs_232))
        val_54 = src_model.policy.predict_values(torch.as_tensor(obs_54))
        val_232 = dst_model.policy.predict_values(torch.as_tensor(obs_232))
        max_abs_logit_diff = float(
            torch.max(torch.abs(dist_54.distribution.logits - dist_232.distribution.logits)).item()
        )
        max_abs_value_diff = float(torch.max(torch.abs(val_54 - val_232)).item())
    assert max_abs_logit_diff <= 1e-5
    assert max_abs_value_diff <= 1e-5

    # Exact 20-step deterministic trajectory equivalence
    env_54 = SheepdogEnvironment(cfg_54)
    env_54.reset(seed=53)
    pol_54 = NeuralPolicy.load(path_54, cfg_54, policy_config=pconfig_54)

    env_232 = SheepdogEnvironment(cfg_232)
    env_232.reset(seed=53)

    for _ in range(20):
        actions_54 = pol_54.select_actions(env_54, deterministic=True)
        actions_232 = loaded_232.select_actions(env_232, deterministic=True)
        assert actions_54 == actions_232
        snap_54, rew_54 = env_54.step(actions_54)
        snap_232, rew_232 = env_232.step(actions_232)
        assert snap_54.to_dict() == snap_232.to_dict()
        assert rew_54.to_dict() == rew_232.to_dict()
        if snap_54.terminated:
            break


def test_strided_gate_validation_run_smoke(tmp_path: Path) -> None:
    """Smoke-test run() end-to-end on a synthetic 54->232 workspace in tmp_path."""
    import json
    import shutil

    from scripts.run_strided_gate_validation import (
        DEFAULT_FAILURE_WEIGHTS,
        STACKED_DIM,
        STRIDES,
        run,
    )

    from sheepdog.atomic_io import atomic_write_json
    from sheepdog.gate_information_preparation import ARCH, ARMS, SEEDS
    from sheepdog.recovery_hyperparam_experiment import sha256_file

    root = tmp_path / "strided_smoke"
    parent_dir = root / "parent"
    parent_dir.mkdir(parents=True)
    frozen = parent_dir / "synthetic_parent.zip"

    cfg_54 = _make_stage34_config(
        observation_mode="guided",
        strides=(0,),
        hidden_sizes=ARCH,
    )
    vec_54 = DummyVecEnv([lambda: SheepdogRLAdapter(cfg_54)])
    try:
        model_54 = MaskablePPO(
            "MlpPolicy",
            vec_54,
            n_steps=64,
            batch_size=32,
            seed=7,
            policy_kwargs={"net_arch": list(ARCH)},
            verbose=0,
        )
        model_54.save(str(frozen))
    finally:
        vec_54.close()

    # Expand once per arm, then copy to each seed workspace so verify() passes quickly.
    arm_base_models: dict[str, tuple[Path, dict]] = {}
    for arm, mode in ARMS.items():
        base_model = root / f"_base_{arm}.zip"
        conversion = expand_policy_observation_inputs(
            frozen,
            base_model,
            old_dim=54,
            new_dim=STACKED_DIM,
            learning_rate=1e-4,
            observation_mode=mode,
            observation_history_strides=STRIDES,
            include_loop_flags_in_observation=True,
        )
        arm_base_models[arm] = (base_model, conversion)

    entries: dict[str, dict] = {}
    for seed in SEEDS:
        for arm, mode in ARMS.items():
            ws = root / arm / f"seed_{seed}"
            ws.mkdir(parents=True)
            cfg_payload = _make_stage34_config(
                observation_mode=mode,
                strides=STRIDES,
                hidden_sizes=ARCH,
            ).to_dict()
            cfg_payload["environment"]["max_steps"] = 20
            cfg_payload["environment"]["no_progress_window"] = 20
            cfg_payload["training"]["train_seed"] = seed
            cfg_payload["training"]["output_dir"] = str(ws / "training_output")
            cfg_payload["training"]["backup_dir"] = str(ws / "backups")
            cfg_payload["training"]["web_export_dir"] = str(ws / "web")
            cfg_payload["training"]["backup_enabled"] = False
            cfg_payload["training"]["failure_directed_training_enabled"] = True
            cfg_payload["training"]["failure_directed_target_ratio"] = 0.25
            cfg_path = ws / "config.json"
            atomic_write_json(cfg_path, cfg_payload)

            model_path = ws / "initial_model.zip"
            base_model, conversion = arm_base_models[arm]
            shutil.copy2(base_model, model_path)
            entries[f"{arm}/seed_{seed}"] = {
                "arm": arm,
                "seed": seed,
                "observation_mode": mode,
                "observation_history_strides": list(STRIDES),
                "observation_size": STACKED_DIM,
                "workspace": str(ws),
                "config_path": str(cfg_path),
                "config_sha256": sha256_file(cfg_path),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "conversion": conversion,
            }

    manifest = {
        "protocol": "stage34_strided_gate_validation_v1",
        "status": "prepared_verified_not_run",
        "strides": list(STRIDES),
        "stacked_observation_size": STACKED_DIM,
        "parent": {
            "model_path": str(frozen),
            "sha256": sha256_file(frozen),
        },
        "entries": entries,
    }
    atomic_write_json(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="multiple of single-env rollout size"):
        run(
            root,
            transitions_per_arm_seed=65,
            arms=("arm_b_treatment_gate_info",),
            seeds=(101,),
            eval_seeds=(11, 23),
        )

    summary = run(
        root,
        transitions_per_arm_seed=64,
        max_arm_seconds=120.0,
        arms=("arm_b_treatment_gate_info",),
        seeds=(101,),
        target_ratio=0.5,
        eval_seeds=(11, 23),
    )
    assert summary["status"] == "completed"
    assert summary["failure_directed_target_ratio"] == 0.5
    assert summary["failure_directed_weights"] == DEFAULT_FAILURE_WEIGHTS
    assert list(summary["results"].keys()) == ["arm_b_treatment_gate_info/seed_101"]

    res = summary["results"]["arm_b_treatment_gate_info/seed_101"]
    assert res["transitions"] == 64
    assert res["world_steps"] > 0
    assert res["complete_training_budget"] is True
    assert res["episode_count"] == 2
    assert len(res["per_seed"]) == 2
    assert all(
        set(item) >= {"seed", "success", "sheep_penned", "steps", "stop_reason", "reward", "replay_path"}
        for item in res["per_seed"]
    )
    assert Path(res["final_model_path"]).is_file()
    assert len(res["sampling_usage"]) == 1
    assert json.loads((root / "run_results.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_clean_sheet_emergent_gate_208_rollout_and_neural_policy_roundtrip(tmp_path: Path) -> None:
    """Verify emergent_gate + (0,4,8,16) + include_loop_flags=False gives 208 inputs, trains 1 rollout, and loads."""
    from sheepdog.curriculum import apply_training_profile
    from sheepdog.policies.neural import compute_observation_schema_version

    profiled = apply_training_profile(
        LabConfig(),
        enable_instinct_rewards=False,
        curriculum_stage=1,
    )
    payload = profiled.to_dict()
    payload["training"]["observation_mode"] = "emergent_gate"
    payload["training"]["observation_history_strides"] = [0, 4, 8, 16]
    payload["training"]["include_loop_flags_in_observation"] = False
    payload["training"]["neural_hidden_sizes"] = [32, 32]
    payload["training"]["rollout_steps"] = 64
    payload["training"]["batch_size"] = 32
    payload["training"]["ppo_env_workers"] = 1
    payload["training"]["output_dir"] = str(tmp_path)
    payload["training"]["web_export_dir"] = str(tmp_path / "web")
    payload["training"]["backup_enabled"] = False
    cfg = LabConfig.from_dict(payload)

    venv = DummyVecEnv([lambda: SheepdogRLAdapter(cfg)])
    try:
        assert venv.observation_space.shape == (208,)
        obs, _ = venv.envs[0].reset(seed=101)
        assert obs.shape == (208,)

        model = MaskablePPO(
            "MlpPolicy",
            venv,
            n_steps=64,
            batch_size=32,
            seed=101,
            policy_kwargs={"net_arch": [32, 32]},
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=64)
        ckpt_path = tmp_path / "clean_sheet_208.zip"
        model.save(str(ckpt_path))
    finally:
        venv.close()

    schema_version = compute_observation_schema_version("emergent_gate", (0, 4, 8, 16), False)
    pconfig = NeuralPolicyConfig(
        hidden_sizes=(32, 32),
        observation_size=208,
        env_workers=1,
        observation_mode="emergent_gate",
        observation_schema_version=schema_version,
        observation_history_strides=(0, 4, 8, 16),
        include_loop_flags_in_observation=False,
    ).to_dict()

    loaded = NeuralPolicy.load(ckpt_path, cfg, policy_config=pconfig)
    assert loaded.config.observation_size == 208
    assert loaded.config.observation_mode == "emergent_gate"
    assert loaded.config.observation_history_strides == (0, 4, 8, 16)
    assert loaded.config.include_loop_flags_in_observation is False

    env = SheepdogEnvironment(cfg)
    env.reset(seed=11)
    actions = loaded.select_actions(env, deterministic=True)
    assert len(actions) == cfg.environment.dogs

