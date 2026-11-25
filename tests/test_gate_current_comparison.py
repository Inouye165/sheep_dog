"""Real evaluator wiring and isolation checks for the bounded gate comparison."""
import json
import time
from dataclasses import replace

import pytest
import torch
from sb3_contrib import MaskablePPO
from scripts.run_gate_current_comparison import (
    AuditedPolicy,
    evaluate,
    isolated_config,
    preflight,
    run,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.observations import EmergentObservationBuilder, RoleAwareObservationBuilder
from sheepdog.training.rl_env import SheepdogRLAdapter


def config(tmp_path, mode):
    base = LabConfig()
    return replace(base, environment=replace(base.environment, max_steps=1),
                   training=replace(base.training, observation_mode=mode,
                                    output_dir=str(tmp_path / "training"),
                                    backup_dir=str(tmp_path / "backups"),
                                    web_export_dir=str(tmp_path / "web"),
                                    backup_enabled=False))


@pytest.mark.parametrize("mode,builder", [("guided", RoleAwareObservationBuilder),
                                          ("emergent", EmergentObservationBuilder)])
def test_existing_observations_unchanged(tmp_path, mode, builder):
    env = SheepdogEnvironment(config(tmp_path, mode))
    env.reset(seed=53)
    assert env.build_observation_for_dog(0) == builder().build(env, 0)


def test_unknown_explicit_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported observation_mode"):
        SheepdogEnvironment(config(tmp_path, "unsupported"))


@pytest.mark.parametrize("mode", ["guided", "guided_gate_control", "guided_gate_treatment"])
def test_real_evaluator_receives_correct_observations(tmp_path, mode):
    torch.set_num_threads(1)
    cfg = config(tmp_path, mode)
    vec = DummyVecEnv([lambda: SheepdogRLAdapter(cfg)])
    try:
        model = MaskablePPO("MlpPolicy", vec, n_steps=8, batch_size=4,
                            policy_kwargs={"net_arch": [256, 256, 128]}, seed=101)
        path = tmp_path / "model.zip"
        model.save(path)
        assert preflight(path, cfg)["dimensions"] == [54 if mode == "guided" else 58]
        result = evaluate(model, cfg, tmp_path / "evaluation", time.monotonic() + 30, path)
        assert result["audited_episodes"] == 10
        assert result["audited_observation_dimensions"] == [54 if mode == "guided" else 58]
        assert len(result["records"]) == 10
        assert result["observation_mode"] == mode
    finally:
        vec.close()


def test_mismatch_rejected_before_policy_action(tmp_path):
    cfg = config(tmp_path, "guided_gate_treatment")
    vec = DummyVecEnv([lambda: SheepdogRLAdapter(cfg)])
    try:
        model = MaskablePPO("MlpPolicy", vec, n_steps=8, batch_size=4)
        policy = AuditedPolicy(model, tmp_path / "unused.zip", cfg)
        env = SheepdogEnvironment(config(tmp_path, "guided"))
        env.reset(seed=53)
        with pytest.raises(ValueError, match="observation size"):
            policy.select_actions(env)
    finally:
        vec.close()


def test_output_isolation_and_config_preservation(tmp_path):
    base = LabConfig().to_dict()
    a = isolated_config(base, tmp_path / "a").to_dict()
    b = isolated_config(base, tmp_path / "b").to_dict()
    assert a["environment"] == b["environment"] == base["environment"]
    assert a["rewards"] == b["rewards"] == base["rewards"]
    for field in ("output_dir", "backup_dir", "web_export_dir"):
        a["training"].pop(field)
        b["training"].pop(field)
    assert a == b
    with pytest.raises(ValueError, match="new artifacts/experiments"):
        run(tmp_path)
    assert not (tmp_path / "manifest.json").exists()
    json.dumps(base)  # the input remains plain serializable config
