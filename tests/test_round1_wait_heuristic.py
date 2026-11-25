"""Unit tests for removing hidden wait heuristic & TeamStrategy calls in neural_policy mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig


def _make_config(
    *,
    policy_mode: str = "neural_policy",
    observation_mode: str = "guided",
    dogs: int = 1,
    sheep: int = 1,
    curriculum_stage: int = 1,
) -> LabConfig:
    payload = LabConfig().to_dict()
    payload["environment"]["dogs"] = dogs
    payload["environment"]["sheep"] = sheep
    payload["environment"]["curriculum_stage"] = curriculum_stage
    payload["policy"]["policy_mode"] = policy_mode
    payload["training"]["observation_mode"] = observation_mode
    return LabConfig.from_dict(payload)


@pytest.mark.parametrize("observation_mode", ["guided", "emergent_gate"])
def test_neural_policy_waiting_has_no_wait_penalty_and_keeps_time_penalty(
    observation_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In neural_policy mode, waiting never triggers _tactically_valid_wait or wait_penalty."""
    cfg = _make_config(
        policy_mode="neural_policy",
        observation_mode=observation_mode,
        dogs=2,
        sheep=2,
    )
    env = SheepdogEnvironment(cfg)
    env.reset(seed=7)

    called = False

    def _forbid_tactical_wait(*args: object, **kwargs: object) -> bool:
        nonlocal called
        called = True
        raise AssertionError("_tactically_valid_wait must not be called in neural_policy mode")

    monkeypatch.setattr(env, "_tactically_valid_wait", _forbid_tactical_wait)

    wait_actions = ["wait"] * env.dog_count
    assert env._waited_without_reason(wait_actions) is False
    _, breakdown = env.step(wait_actions)

    assert called is False
    assert breakdown.wait_penalty == 0.0
    assert breakdown.time_penalty < 0.0


def test_heuristic_mode_waiting_still_checks_tactical_validity_and_applies_wait_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In heuristic mode, _waited_without_reason calls _tactically_valid_wait and penalizes invalid waits."""
    cfg = _make_config(policy_mode="heuristic", observation_mode="guided", dogs=1, sheep=1)
    env = SheepdogEnvironment(cfg)
    env.reset(seed=11)

    calls: list[int] = []
    orig_tactically_valid_wait = env._tactically_valid_wait

    def _spy_tactical_wait(dog: object, *args: object, **kwargs: object) -> bool:
        calls.append(getattr(dog, "index", -1))
        # Ensure at least one check returns False if the spawn happened to be valid,
        # while still exercising the real _tactically_valid_wait implementation.
        orig_tactically_valid_wait(dog, *args, **kwargs)  # type: ignore[arg-type]
        return False

    monkeypatch.setattr(env, "_tactically_valid_wait", _spy_tactical_wait)

    assert env._waited_without_reason(["wait"]) is True
    assert calls == [0]

    _, breakdown = env.step(["wait"])
    assert len(calls) == 2
    assert breakdown.wait_penalty < 0.0
    assert breakdown.time_penalty < 0.0


def test_emergent_gate_neural_policy_skips_team_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In emergent_gate + neural_policy mode, TeamStrategy.assign_roles and prepare_policy_step are skipped."""
    cfg = _make_config(
        policy_mode="neural_policy",
        observation_mode="emergent_gate",
        dogs=3,
        sheep=6,
        curriculum_stage=4,
    )
    env = SheepdogEnvironment(cfg)
    env.reset(seed=23)

    assert env._uses_team_strategy() is False

    def _forbid_assign_roles(*args: object, **kwargs: object) -> object:
        raise AssertionError("TeamStrategy.assign_roles must not be called in emergent_gate + neural_policy")

    def _forbid_prepare_policy_step(*args: object, **kwargs: object) -> None:
        raise AssertionError("prepare_policy_step must not be called in emergent_gate + neural_policy")

    monkeypatch.setattr(env._team_strategy, "assign_roles", _forbid_assign_roles)
    monkeypatch.setattr(env, "prepare_policy_step", _forbid_prepare_policy_step)

    for dog_index in range(env.dog_count):
        obs = env.build_observation_for_dog(dog_index)
        assert len(obs.values) > 0
        mask = env.action_mask_for_dog(dog_index)
        assert mask["wait"] is True

    class _DummyModel:
        def predict(
            self,
            observation: np.ndarray,
            action_masks: np.ndarray,
            deterministic: bool = True,
        ) -> tuple[int, None]:
            del observation, deterministic
            wait_idx = ACTION_ORDER.index("wait")
            assert bool(action_masks[wait_idx])
            return wait_idx, None

    policy = NeuralPolicy(
        model=_DummyModel(),  # type: ignore[arg-type]
        model_path=Path("dummy.zip"),
        config=NeuralPolicyConfig(hidden_sizes=(64, 64), observation_size=len(obs.values)),
    )
    actions = policy.select_actions(env, deterministic=True)
    assert actions == ["wait", "wait", "wait"]

    _, breakdown = env.step(actions)
    assert breakdown.wait_penalty == 0.0
    assert breakdown.time_penalty < 0.0
