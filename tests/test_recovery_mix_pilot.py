"""Safety and pairing checks for the isolated recovery-mix pilot."""

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.run_recovery_mix_pilot import (
    PARENT,
    TRANSITIONS,
    LimitCallback,
    make_config,
    run,
    verify_parent,
    wilson,
)

from sheepdog.config import LabConfig


def metadata():
    cfg = LabConfig().to_dict()
    cfg["environment"].update(dogs=3, sheep=6, max_steps=2320, curriculum_stage=34)
    return {
        "environment_config": cfg["environment"], "reward_config": cfg["rewards"],
        "policy_config": {"observation_size": 54}, "curriculum_stage": 34,
        "policy_state_path": str(PARENT),
    }


def test_arms_only_change_ratio_and_isolated_output_paths(tmp_path):
    parent = metadata()
    base = LabConfig().to_dict()
    original = copy.deepcopy(base)
    a = make_config(base, parent, tmp_path / "a", 101, .25).to_dict()
    b = make_config(base, parent, tmp_path / "b", 101, .60).to_dict()
    assert base == original
    assert a["environment"] == b["environment"] == parent["environment_config"]
    assert a["rewards"] == b["rewards"] == parent["reward_config"]
    for key in ("output_dir", "backup_dir", "web_export_dir", "failure_directed_target_ratio"):
        a["training"].pop(key)
        b["training"].pop(key)
    assert a == b


def test_rejects_wrong_task_and_wrong_parent():
    parent = metadata()
    parent["environment_config"]["sheep"] = 5
    with pytest.raises(ValueError, match="six sheep"):
        make_config(LabConfig().to_dict(), parent, Path("unused"), 101, .25)
    parent = metadata()
    parent["policy_state_path"] = "different.zip"
    with pytest.raises(ValueError, match="different parent"):
        verify_parent(parent, {"n_steps": 2048, "policy_kwargs": {"net_arch": [256, 256, 128]}})


def test_callback_counts_actual_world_steps_and_stops_at_deadline():
    callback = LimitCallback(time.monotonic() + 10)
    callback.locals = {"infos": [{"team_step_completed": False}]}
    assert callback._on_step()
    callback.locals = {"infos": [{"team_step_completed": True, "episode": {"success": False}}]}
    assert callback._on_step()
    assert callback.transitions == 2
    assert callback.world_steps == 1
    assert callback.episodes == [{"success": False}]
    callback.deadline = time.monotonic() - 1
    assert not callback._on_step()


def test_callback_cannot_continue_past_transition_budget():
    callback = LimitCallback(time.monotonic() + 10)
    callback.transitions = TRANSITIONS
    callback.locals = {"infos": [{"team_step_completed": False}]}
    assert not callback._on_step()


def test_wilson_bounds_and_empty_denominator():
    assert wilson(0, 0) is None
    low, high = wilson(6, 10)
    assert 0 < low < .6 < high < 1


def test_output_cannot_target_active_artifacts(tmp_path):
    with pytest.raises(ValueError, match="new artifacts/experiments"):
        run(tmp_path)
    assert not (tmp_path / "parent.zip").exists()


def test_evaluation_records_success_only_steps_and_model_identity(tmp_path, monkeypatch):
    import scripts.run_recovery_mix_pilot as pilot

    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"test model identity")
    captured = {}

    class FakeEvaluator:
        def __init__(self, config, root):
            pass

        def evaluate(self, policy, seeds, **kwargs):
            captured.update(kwargs)
            captured["seeds"] = seeds
            records = [SimpleNamespace(success=True, steps=20),
                       SimpleNamespace(success=False, steps=2320)]
            return SimpleNamespace(records=records, to_dict=lambda: {}), None, None

    monkeypatch.setattr(pilot, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(pilot, "NeuralPolicy", lambda *a, **kw: object())
    result = pilot.evaluate(object(), LabConfig(), tmp_path, time.monotonic() + 10, model_path)
    assert result["mean_steps_among_successes"] == 20
    assert result["steps_among_successes"] == [20]
    assert result["model_sha256"] == pilot.digest(model_path)
    assert captured["seeds"] == pilot.DEV_SEEDS
    assert captured["deterministic"] is True
    assert captured["capture_step_telemetry"] is False
    assert captured["should_stop"]() is False
