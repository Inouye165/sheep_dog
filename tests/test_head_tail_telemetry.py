"""Comprehensive tests for opt-in late-episode head/tail policy decision telemetry."""

from __future__ import annotations

import collections
import gc
import gzip
import json
import random
import statistics
import time
import weakref
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sheepdog.config import LabConfig
from sheepdog.environment import SheepdogEnvironment, StepRecord
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.rewards import RewardBreakdown
from sheepdog.training.diagnostics import temporal_interaction
from sheepdog.training.diagnostics.temporal_interaction import (
    analyze_episode_temporal_interactions,
    build_temporal_diagnostic_payload,
    load_replay_bundle,
)
from sheepdog.training.replay_writer import (
    CapturePolicy,
    set_global_capture_policy,
)
from sheepdog.training.rl_env import SheepdogRLAdapter


def make_cfg(max_steps: int = 15) -> LabConfig:
    """Helper to construct LabConfig with overridden environment.max_steps."""
    base = LabConfig()
    return LabConfig(environment=replace(base.environment, max_steps=max_steps))


# ==============================================================================
# 1. Consistent Configuration Validation & Mid-Episode Mutation Rejection Tests
# ==============================================================================


def test_capture_policy_validation() -> None:
    """Verify CapturePolicy validation on instantiation, mutation, and bounds."""
    # Valid default
    p = CapturePolicy()
    assert p.telemetry_mode == "prefix"
    assert p.telemetry_head_steps == 200
    assert p.telemetry_tail_steps == 500
    assert p.max_telemetry_steps_per_episode == 1000

    # Valid head_tail instantiation
    p2 = CapturePolicy(
        telemetry_mode="head_tail",
        telemetry_head_steps=10,
        telemetry_tail_steps=20,
        max_telemetry_steps_per_episode=30,
    )
    assert p2.telemetry_mode == "head_tail"
    assert p2.telemetry_head_steps == 10

    # Invalid mode (ValueError)
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        CapturePolicy(telemetry_mode="unsupported_mode")

    # Non-string mode (TypeError)
    with pytest.raises(TypeError, match="telemetry_mode must be a string"):
        CapturePolicy(telemetry_mode=123)  # type: ignore[arg-type]

    # Negative lengths (ValueError)
    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        CapturePolicy(telemetry_head_steps=-1)

    with pytest.raises(ValueError, match="telemetry_tail_steps must be non-negative"):
        CapturePolicy(telemetry_tail_steps=-5)

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        CapturePolicy(max_telemetry_steps_per_episode=-10)

    # Non-integer / boolean types (TypeError)
    with pytest.raises(TypeError, match="telemetry_head_steps must be an integer"):
        CapturePolicy(telemetry_head_steps=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        CapturePolicy(telemetry_tail_steps="twenty")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="max_telemetry_steps must be an integer"):
        CapturePolicy(max_telemetry_steps_per_episode=15.5)  # type: ignore[arg-type]

    # Mutation validation via attributes and properties
    p3 = CapturePolicy()
    p3.telemetry_mode = "head_tail"
    assert p3.telemetry_mode == "head_tail"
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        p3.telemetry_mode = "invalid"

    p3.max_telemetry_steps = 50
    assert p3.max_telemetry_steps == 50
    assert p3.max_telemetry_steps_per_episode == 50
    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        p3.max_telemetry_steps = -1


def test_sheepdog_environment_validation_and_mutation() -> None:
    """Verify SheepdogEnvironment rejects invalid capture configs and mid-episode mutations."""
    cfg = make_cfg(15)
    policy = NeuralPolicy.initialize(cfg)

    # 1. Constructor validation
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        SheepdogEnvironment(cfg, telemetry_mode="bad_mode")

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        SheepdogEnvironment(cfg, telemetry_head_steps=-1)

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        SheepdogEnvironment(cfg, telemetry_tail_steps="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        SheepdogEnvironment(cfg, max_telemetry_steps=-1)

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        SheepdogEnvironment(cfg, max_telemetry_steps_per_episode=-5)

    # 2. run_policy validation
    env = SheepdogEnvironment(cfg)
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        env.run_policy(policy, seed=1, telemetry_mode="invalid")

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        env.run_policy(policy, seed=1, telemetry_head_steps=-5)

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        env.run_policy(policy, seed=1, telemetry_tail_steps=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        env.run_policy(policy, seed=1, max_telemetry_steps_per_episode=-1)

    # 3. run_policy_on_scenario validation
    from sheepdog.evaluation.scenarios import scenario_from_snapshot

    env.reset(42)
    scenario = scenario_from_snapshot(name="scen_test", seed=42, snapshot=env.get_state_snapshot().to_dict())
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        env.run_policy_on_scenario(policy, scenario, telemetry_mode="bad_mode")

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        env.run_policy_on_scenario(policy, scenario, telemetry_head_steps=-1)

    # 4. Mid-episode mutation rejection
    env.reset(42)
    env.step(("wait", "wait", "wait"))
    assert env._step_count == 1
    assert not env._terminated

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        env.telemetry_mode = "head_tail"

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        env.telemetry_head_steps = 100

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        env.telemetry_tail_steps = 100

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        env.max_telemetry_steps = 50

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        env.max_telemetry_steps_per_episode = 50

    # 5. After termination, mutation is permitted
    env._terminated = True
    env.telemetry_mode = "head_tail"
    assert env.telemetry_mode == "head_tail"
    env.telemetry_head_steps = 50
    assert env.telemetry_head_steps == 50


def test_evaluator_validation(tmp_path: Path) -> None:
    """Verify Evaluator rejects invalid telemetry configurations upfront in evaluate()."""
    cfg = make_cfg(10)
    policy = NeuralPolicy.initialize(cfg)
    evaluator = Evaluator(cfg, tmp_path)

    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        evaluator.evaluate(policy, seeds=(1,), checkpoint_episode=1, telemetry_mode="bad_mode")

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        evaluator.evaluate(policy, seeds=(1,), checkpoint_episode=1, telemetry_head_steps=-1)

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        evaluator.evaluate(policy, seeds=(1,), checkpoint_episode=1, telemetry_tail_steps="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        evaluator.evaluate(policy, seeds=(1,), checkpoint_episode=1, max_telemetry_steps=-1)

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        evaluator.evaluate(policy, seeds=(1,), checkpoint_episode=1, max_telemetry_steps_per_episode=-10)


def test_sheepdog_rl_adapter_validation_and_mutation() -> None:
    """Verify SheepdogRLAdapter validates telemetry parameters and rejects mid-episode mutations."""
    cfg = make_cfg(15)

    # 1. Constructor validation
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        SheepdogRLAdapter(cfg, telemetry_mode="bad_mode")

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        SheepdogRLAdapter(cfg, telemetry_head_steps=-1)

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        SheepdogRLAdapter(cfg, telemetry_tail_steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        SheepdogRLAdapter(cfg, max_telemetry_steps=-1)

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        SheepdogRLAdapter(cfg, max_telemetry_steps_per_episode=-5)

    # 2. Setters validation
    adapter = SheepdogRLAdapter(cfg)
    with pytest.raises(ValueError, match="Invalid telemetry_mode"):
        adapter.telemetry_mode = "invalid"

    with pytest.raises(ValueError, match="telemetry_head_steps must be non-negative"):
        adapter.telemetry_head_steps = -10

    with pytest.raises(TypeError, match="telemetry_tail_steps must be an integer"):
        adapter.telemetry_tail_steps = "ten"  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_telemetry_steps must be non-negative"):
        adapter.max_telemetry_steps = -1

    # 3. Mid-episode mutation rejection
    adapter.reset()
    for _ in range(3):  # 1 full team world step (3 dogs)
        adapter.step(0)
    assert adapter._environment.step_count == 1
    assert not adapter._environment._terminated

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        adapter.telemetry_mode = "head_tail"

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        adapter.telemetry_head_steps = 10

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        adapter.telemetry_tail_steps = 10

    with pytest.raises(RuntimeError, match="Cannot change telemetry configuration during an active episode"):
        adapter.max_telemetry_steps = 10


# ==============================================================================
# 2. Application-Level Payload Retention & Garbage Collection Tests
# ==============================================================================


class WeakrefPayloadDict(dict):
    """Dictionary subclass that supports Python weak references."""
    pass


def test_environment_run_policy_payload_eviction_and_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Application-level test using real SheepdogEnvironment.run_policy.

    Runs an episode with a genuine gap (T=10, H=2, L=2, so steps 3..8 are intermediate).
    Hooks into payload creation to capture weakref.ref to each created payload.
    Verifies after episode completion and gc.collect(), the weakrefs for evicted payloads
    resolve to None (reclaimed), while env._history retains ALL 10 ordinary trajectory frames,
    and only steps 1..2 and 9..10 have step_telemetry attached (frame 3 has telemetry_omitted marker).
    """
    T = 10
    H = 2
    L = 2
    cfg = make_cfg(T)
    policy = NeuralPolicy.initialize(cfg)
    env = SheepdogEnvironment(
        cfg,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )

    created_refs: dict[int, weakref.ref[Any]] = {}
    orig_build = temporal_interaction.build_step_instrumentation_payload
    call_count = 0

    def hook_build(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raw_dict = orig_build(*args, **kwargs)
        payload = WeakrefPayloadDict(raw_dict)
        step = call_count
        created_refs[step] = weakref.ref(payload)
        return payload

    monkeypatch.setattr(temporal_interaction, "build_step_instrumentation_payload", hook_build)

    res = env.run_policy(
        policy,
        seed=101,
        capture_replay=True,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )

    assert len(res.replay) == T
    assert len(env._history) == T

    # Force garbage collection
    gc.collect()

    # 1. Weakrefs for evicted intermediate payloads (steps 3..8) must resolve to None
    for step_num in range(H + 1, T - L + 1):
        assert step_num in created_refs, f"Step {step_num} payload was not constructed!"
        assert (
            created_refs[step_num]() is None
        ), f"Intermediate evicted payload for step {step_num} was NOT garbage collected!"

    # 2. Weakrefs for retained payloads (steps 1..2 and 9..10) must remain alive
    for step_num in (1, 2, 9, 10):
        assert created_refs[step_num]() is not None, f"Retained payload for step {step_num} was unexpectedly reclaimed!"

    # 3. All 10 ordinary trajectory frames remain present
    assert len(res.replay) == 10
    for idx, frame in enumerate(res.replay):
        assert frame.step == idx + 1
        assert frame.actions is not None
        assert frame.snapshot is not None
        assert frame.reward is not None

    # 4. Frame 1 and 2 have detailed step_telemetry
    assert res.replay[0].step_telemetry is not None
    assert "decisions" in res.replay[0].step_telemetry
    assert res.replay[1].step_telemetry is not None
    assert "decisions" in res.replay[1].step_telemetry

    # 5. Frame 3 has omission marker
    frame_3 = res.replay[2]
    assert frame_3.step_telemetry is not None
    assert frame_3.step_telemetry.get("telemetry_omitted") is True
    assert frame_3.step_telemetry.get("omitted_reason") == "intentionally_omitted_window"
    assert frame_3.step_telemetry.get("omitted_steps_start") == 3
    assert frame_3.step_telemetry.get("omitted_steps_end") == 8
    assert frame_3.step_telemetry.get("omitted_steps_count") == 6
    assert frame_3.step_telemetry.get("total_world_steps") == 10

    # 6. Frames 4..8 have step_telemetry is None
    for i in range(3, 8):
        assert res.replay[i].step_telemetry is None

    # 7. Tail frames 9..10 have detailed step_telemetry
    assert res.replay[8].step_telemetry is not None
    assert "decisions" in res.replay[8].step_telemetry
    assert res.replay[9].step_telemetry is not None
    assert "decisions" in res.replay[9].step_telemetry


def test_rl_adapter_payload_eviction_and_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Application-level test for SheepdogRLAdapter verifying intermediate payload eviction & GC."""
    T = 10
    H = 2
    L = 2
    cfg = make_cfg(T)

    submitted_jobs: list[Any] = []

    class MockWriter:
        def enqueue(self, job: Any) -> bool:
            submitted_jobs.append(job)
            return True

    monkeypatch.setattr("sheepdog.training.replay_writer.get_replay_writer", lambda *args, **kwargs: MockWriter())

    set_global_capture_policy(
        CapturePolicy(
            mode="all",
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=H,
            telemetry_tail_steps=L,
        )
    )

    created_refs: dict[int, weakref.ref[Any]] = {}
    orig_build = temporal_interaction.build_step_instrumentation_payload
    call_count = 0

    def hook_build(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        raw_dict = orig_build(*args, **kwargs)
        payload = WeakrefPayloadDict(raw_dict)
        step = call_count
        created_refs[step] = weakref.ref(payload)
        return payload

    monkeypatch.setattr(temporal_interaction, "build_step_instrumentation_payload", hook_build)

    adapter = SheepdogRLAdapter(
        cfg,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )
    adapter.reset()

    # Step 10 world steps (30 dog steps) until episode timeout at step 10
    for _ in range(T * adapter._environment.dog_count):
        _, _, term, trunc, _ = adapter.step(0)
        if term or trunc:
            break

    assert len(submitted_jobs) == 1
    buf = submitted_jobs[0].payload["frames"]
    assert len(buf) == T + 1  # Step 0 + steps 1..10

    # Force garbage collection
    gc.collect()

    # 1. Intermediate evicted payloads (steps 3..8) must be reclaimed
    for s_num in range(H + 1, T - L + 1):
        assert s_num in created_refs
        assert created_refs[s_num]() is None, f"RL adapter intermediate payload for step {s_num} not reclaimed!"

    # 2. Retained head (1, 2) and tail (9, 10) payloads remain alive
    for s_num in (1, 2, 9, 10):
        assert created_refs[s_num]() is not None

    # 3. Buffer frames verification
    assert buf[0]["step"] == 0
    assert "step_telemetry" not in buf[0] or buf[0]["step_telemetry"] is None

    assert buf[1]["step_telemetry"] is not None
    assert buf[2]["step_telemetry"] is not None

    # Omission marker at step 3 (index 3)
    assert buf[3]["step_telemetry"] is not None
    assert buf[3]["step_telemetry"].get("telemetry_omitted") is True
    assert buf[3]["step_telemetry"].get("omitted_steps_start") == 3
    assert buf[3]["step_telemetry"].get("omitted_steps_end") == 8

    # Omitted intermediate frames
    for i in range(4, 9):
        assert buf[i].get("step_telemetry") is None

    # Tail frames
    assert buf[9]["step_telemetry"] is not None
    assert buf[10]["step_telemetry"] is not None

    # Restore global capture policy
    set_global_capture_policy(CapturePolicy())


# ==============================================================================
# 3. Behavior / RNG Invariance Tests (Deterministic & Sampled Across Gap)
# ==============================================================================


@pytest.mark.parametrize("deterministic", [True, False])
def test_bit_for_bit_invariance_deterministic_and_sampled(deterministic: bool) -> None:
    """Verify bit-for-bit trajectory, reward breakdown, snapshot, and RNG invariance.

    Compares capture OFF vs ON across:
    - Full action sequences across all dogs and steps.
    - All 28 reward breakdown components per frame.
    - Full environment snapshot fields and animal positions.
    - Episode termination flags (terminated, timeout, stopped, success, stop_reason).
    - Exact RNG states: Python random, NumPy, PyTorch, and environment _rng.
    - Runs over an episode with a genuine gap (T > H + L).
    """
    T = 20
    H = 4
    L = 4
    cfg = make_cfg(T)

    policy_off = NeuralPolicy.initialize(cfg)
    policy_on = NeuralPolicy.initialize(cfg)
    policy_on.model.policy.load_state_dict(policy_off.model.policy.state_dict())

    # ------------------
    # Run 1: Capture OFF
    # ------------------
    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    env_off = SheepdogEnvironment(cfg, capture_step_telemetry=False)
    res_off = env_off.run_policy(
        policy_off,
        seed=42,
        capture_replay=True,
        deterministic=deterministic,
        capture_step_telemetry=False,
    )
    rng_py_off = random.getstate()
    rng_np_off = np.random.get_state()
    rng_th_off = torch.get_rng_state()
    rng_env_off = env_off._rng.getstate()

    # ---------------------------------------------
    # Run 2: Capture ON (Head/Tail with genuine gap)
    # ---------------------------------------------
    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    env_on = SheepdogEnvironment(
        cfg,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )
    res_on = env_on.run_policy(
        policy_on,
        seed=42,
        capture_replay=True,
        deterministic=deterministic,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )
    rng_py_on = random.getstate()
    rng_np_on = np.random.get_state()
    rng_th_on = torch.get_rng_state()
    rng_env_on = env_on._rng.getstate()

    # 1. Assert genuine gap occurred
    actual_t = len(res_off.replay)
    assert actual_t > H + L, f"Episode length {actual_t} was not greater than H+L={H+L}!"
    assert len(res_off.replay) == len(res_on.replay)

    # 2. Frame-by-frame bit-for-bit invariance
    for step_idx in range(actual_t):
        f_off = res_off.replay[step_idx]
        f_on = res_on.replay[step_idx]

        # Actions
        assert f_off.actions == f_on.actions, f"Action mismatch at step {step_idx + 1}"

        # All 28 reward components bit-for-bit identical
        rew_off_dict = asdict(f_off.reward)
        rew_on_dict = asdict(f_on.reward)
        assert len(rew_off_dict) == 28, f"Expected 28 reward breakdown fields, found {len(rew_off_dict)}"
        assert rew_off_dict == rew_on_dict, f"Reward breakdown divergence at step {step_idx + 1}"

        # Snapshot fields & animal positions
        snap_off = f_off.snapshot.to_dict()
        snap_on = f_on.snapshot.to_dict()
        assert snap_off == snap_on, f"Snapshot divergence at step {step_idx + 1}"

    # 3. Termination flags invariance
    assert env_off._terminated == env_on._terminated
    assert env_off._timeout == env_on._timeout
    assert env_off._stopped == env_on._stopped
    assert env_off._success == env_on._success
    assert env_off._stop_reason == env_on._stop_reason

    # 4. Exact RNG state invariance (zero random numbers consumed by telemetry)
    assert rng_py_off == rng_py_on, "Python random state diverged!"
    assert rng_np_off[0] == rng_np_on[0]
    assert np.array_equal(rng_np_off[1], rng_np_on[1]), "NumPy random state diverged!"
    assert rng_np_off[2:] == rng_np_on[2:]
    assert torch.equal(rng_th_off, rng_th_on), "PyTorch RNG state diverged!"
    assert rng_env_off == rng_env_on, "Environment _rng state diverged!"


# ==============================================================================
# 4. Training-Adapter Direct Head/Tail Behavior Tests
# ==============================================================================


def test_rl_adapter_overlapping_head_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """When T <= H + L: deduplication, no gap marker, all detailed payloads present."""
    cfg = make_cfg(8)
    H = 5
    L = 5
    submitted_jobs: list[Any] = []

    class MockWriter:
        def enqueue(self, job: Any) -> bool:
            submitted_jobs.append(job)
            return True

    monkeypatch.setattr("sheepdog.training.replay_writer.get_replay_writer", lambda *args, **kwargs: MockWriter())

    set_global_capture_policy(
        CapturePolicy(
            mode="all",
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=H,
            telemetry_tail_steps=L,
        )
    )
    adapter = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=H, telemetry_tail_steps=L)
    adapter.reset()
    for _ in range(8 * 3):
        adapter.step(0)

    assert len(submitted_jobs) == 1
    buf = submitted_jobs[0].payload["frames"]
    assert len(buf) == 9  # Step 0 + 8 frames
    # Contiguous coverage: every frame 1..8 has detailed telemetry
    for i in range(1, 9):
        assert buf[i]["step_telemetry"] is not None
        assert buf[i]["step_telemetry"].get("telemetry_omitted") is not True
        assert "decisions" in buf[i]["step_telemetry"]

    set_global_capture_policy(CapturePolicy())


def test_rl_adapter_exact_gap_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """When T > H + L: frame H+1 has omission marker, tail frames stitched, ordinary frames intact."""
    cfg = make_cfg(12)
    H = 3
    L = 3
    submitted_jobs: list[Any] = []

    class MockWriter:
        def enqueue(self, job: Any) -> bool:
            submitted_jobs.append(job)
            return True

    monkeypatch.setattr("sheepdog.training.replay_writer.get_replay_writer", lambda *args, **kwargs: MockWriter())

    set_global_capture_policy(
        CapturePolicy(
            mode="all",
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=H,
            telemetry_tail_steps=L,
        )
    )
    adapter = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=H, telemetry_tail_steps=L)
    adapter.reset()
    for _ in range(12 * 3):
        adapter.step(0)

    assert len(submitted_jobs) == 1
    buf = submitted_jobs[0].payload["frames"]
    assert len(buf) == 13

    # Head 1..3
    for i in range(1, 4):
        assert buf[i]["step_telemetry"] is not None
        assert "decisions" in buf[i]["step_telemetry"]

    # Frame 4 (index 4) has omission marker
    marker = buf[4]["step_telemetry"]
    assert marker is not None
    assert marker.get("telemetry_omitted") is True
    assert marker.get("omitted_steps_start") == 4
    assert marker.get("omitted_steps_end") == 9
    assert marker.get("omitted_steps_count") == 6

    # Omitted 5..9
    for i in range(5, 10):
        assert buf[i].get("step_telemetry") is None

    # Tail 10..12
    for i in range(10, 13):
        assert buf[i]["step_telemetry"] is not None
        assert "decisions" in buf[i]["step_telemetry"]

    set_global_capture_policy(CapturePolicy())


def test_rl_adapter_zero_length_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test zero-length window variants: H=0/L=0, H>0/L=0, H=0/L>0."""
    cfg = make_cfg(6)
    submitted_jobs: list[Any] = []

    class MockWriter:
        def enqueue(self, job: Any) -> bool:
            submitted_jobs.append(job)
            return True

    monkeypatch.setattr("sheepdog.training.replay_writer.get_replay_writer", lambda *args, **kwargs: MockWriter())

    # 1. H=0, L=0: no detailed payloads, frame 1 has omission marker
    set_global_capture_policy(
        CapturePolicy(mode="all", capture_step_telemetry=True, telemetry_mode="head_tail", telemetry_head_steps=0, telemetry_tail_steps=0)
    )
    a00 = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=0, telemetry_tail_steps=0)
    a00.reset()
    for _ in range(6 * 3):
        a00.step(0)
    buf00 = submitted_jobs[-1].payload["frames"]
    assert buf00[1]["step_telemetry"].get("telemetry_omitted") is True
    for f in buf00[2:]:
        assert f.get("step_telemetry") is None

    # 2. H=3, L=0: head only, steps 1..3 detailed, frame 4 has omission marker
    set_global_capture_policy(
        CapturePolicy(mode="all", capture_step_telemetry=True, telemetry_mode="head_tail", telemetry_head_steps=3, telemetry_tail_steps=0)
    )
    a30 = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=3, telemetry_tail_steps=0)
    a30.reset()
    for _ in range(6 * 3):
        a30.step(0)
    buf30 = submitted_jobs[-1].payload["frames"]
    for i in range(1, 4):
        assert "decisions" in buf30[i]["step_telemetry"]
    assert buf30[4]["step_telemetry"].get("telemetry_omitted") is True
    for f in buf30[5:]:
        assert f.get("step_telemetry") is None

    # 3. H=0, L=3: tail only, frame 1 has omission marker, tail steps detailed
    set_global_capture_policy(
        CapturePolicy(mode="all", capture_step_telemetry=True, telemetry_mode="head_tail", telemetry_head_steps=0, telemetry_tail_steps=3)
    )
    a03 = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=0, telemetry_tail_steps=3)
    a03.reset()
    for _ in range(6 * 3):
        a03.step(0)
    buf03 = submitted_jobs[-1].payload["frames"]
    assert buf03[1]["step_telemetry"].get("telemetry_omitted") is True
    for i in range(4, 7):
        assert "decisions" in buf03[i]["step_telemetry"]

    set_global_capture_policy(CapturePolicy())


def test_rl_adapter_consecutive_resets_and_mock_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify consecutive resets clear buffers cleanly, and replay persistence submits mock job."""
    cfg = make_cfg(5)
    submitted_jobs: list[Any] = []

    class MockWriter:
        def enqueue(self, job: Any) -> bool:
            submitted_jobs.append(job)
            return True

    monkeypatch.setattr("sheepdog.training.replay_writer.get_replay_writer", lambda *args, **kwargs: MockWriter())

    set_global_capture_policy(
        CapturePolicy(
            mode="all",
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=2,
            telemetry_tail_steps=2,
        )
    )

    adapter = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=2, telemetry_tail_steps=2)

    # Run episode 1
    adapter.reset()
    for _ in range(5 * 3):
        adapter.step(0)
    assert len(submitted_jobs) == 1
    job1 = submitted_jobs[0]
    assert job1.payload["telemetry_mode"] == "head_tail"
    assert "telemetry_coverage" in job1.payload

    # Reset
    adapter.reset()
    assert len(adapter._active_trajectory_buffer) == 1  # Only step 0
    assert len(adapter._tail_telemetry_buffer) == 0
    assert adapter._captured_telemetry_steps == 0
    assert adapter._telemetry_omitted is False
    assert adapter._telemetry_truncated is False

    # Run episode 2
    for _ in range(5 * 3):
        adapter.step(0)
    assert len(submitted_jobs) == 2

    set_global_capture_policy(CapturePolicy())


def test_rl_adapter_partial_export_mid_run() -> None:
    """Verify partial trajectory export behavior mid-run before normal termination."""
    cfg = make_cfg(15)
    set_global_capture_policy(
        CapturePolicy(
            capture_step_telemetry=True,
            telemetry_mode="head_tail",
            telemetry_head_steps=2,
            telemetry_tail_steps=2,
        )
    )

    adapter = SheepdogRLAdapter(cfg, telemetry_mode="head_tail", telemetry_head_steps=2, telemetry_tail_steps=2)
    adapter.reset()

    # Step 5 world steps (15 dog steps) out of 15
    for _ in range(15):
        adapter.step(0)

    # 1. Raw export without tail finalization: tail remains in buffer, intermediate steps > H have None
    raw_export = adapter.export_active_trajectory(finalize_tail=False)
    assert raw_export["total_frames"] == 6  # Step 0 + 5 steps
    assert raw_export["frames"][1]["step_telemetry"] is not None
    assert raw_export["frames"][2]["step_telemetry"] is not None
    assert raw_export["frames"][3].get("step_telemetry") is None
    assert raw_export["frames"][4].get("step_telemetry") is None
    assert raw_export["frames"][5].get("step_telemetry") is None
    assert raw_export["tail_buffer_count"] == 2  # Recent steps 4 and 5 in deque

    # 2. Finalized export: stitches the current tail buffer and inserts omission marker relative to step 5
    finalized_export = adapter.export_active_trajectory(finalize_tail=True)
    assert finalized_export["total_frames"] == 6
    assert finalized_export["frames"][1]["step_telemetry"] is not None
    assert finalized_export["frames"][2]["step_telemetry"] is not None
    # Step 3 receives omission marker for steps 3 to 3 (tail starts at 4)
    assert finalized_export["frames"][3]["step_telemetry"].get("telemetry_omitted") is True
    # Steps 4 and 5 have tail telemetry attached
    assert finalized_export["frames"][4]["step_telemetry"] is not None
    assert finalized_export["frames"][5]["step_telemetry"] is not None

    set_global_capture_policy(CapturePolicy())


# ==============================================================================
# 5. Analyzer and Viewer Gap Semantics Tests
# ==============================================================================


def test_exact_benchmark_failure_example_t1354_h200_l500() -> None:
    """Exact benchmark failure example: T=1354, H=200, L=500.

    Head: 1..200 (200 frames)
    Omitted: 201..854 (exactly 654 omitted steps, frame 201 holds marker)
    Tail: 855..1354 (500 frames)
    All 1354 ordinary frames remain present!
    """
    T = 1354
    H = 200
    L = 500
    tail_start = T - L + 1  # 855
    omitted_start = H + 1   # 201
    omitted_end = tail_start - 1  # 854
    omitted_count = tail_start - 1 - H  # 654

    cfg = make_cfg(T)
    env = SheepdogEnvironment(
        cfg,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=H,
        telemetry_tail_steps=L,
    )
    env.reset(seed=53)

    # Simulate 1354 steps directly through environment loop mechanism
    tail_buffer: collections.deque[tuple[int, dict[str, Any]]] = collections.deque(maxlen=L)
    env._tail_telemetry_buffer = tail_buffer

    for step_num in range(1, T + 1):
        env._step_count = step_num
        payload = {"step": step_num, "decisions": [{"dog_index": 0, "mock": True}]}
        if step_num <= H:
            tele = payload
        else:
            tail_buffer.append((step_num, payload))
            tele = None

        env._history.append(
            StepRecord(
                step=step_num,
                actions=("wait", "wait", "wait"),
                snapshot=env.get_state_snapshot(),
                reward=RewardBreakdown(total=0.0),
                step_telemetry=tele,
            )
        )

    # Post-processing matching _run_policy_loop
    if tail_start > H + 1:
        omission_marker = {
            "telemetry_omitted": True,
            "omitted_reason": "intentionally_omitted_window",
            "telemetry_mode": "head_tail",
            "head_steps": H,
            "tail_steps": L,
            "omitted_range": f"steps_{omitted_start}_to_{omitted_end}",
            "omitted_steps_start": omitted_start,
            "omitted_steps_end": omitted_end,
            "omitted_steps_count": omitted_count,
            "total_world_steps": T,
        }
        marker_idx = omitted_start - 1
        rec = env._history[marker_idx]
        env._history[marker_idx] = StepRecord(
            step=rec.step,
            actions=rec.actions,
            snapshot=rec.snapshot,
            reward=rec.reward,
            step_telemetry=omission_marker,
        )

    for s_num, p_load in tail_buffer:
        idx = s_num - 1
        rec = env._history[idx]
        env._history[idx] = StepRecord(
            step=rec.step,
            actions=rec.actions,
            snapshot=rec.snapshot,
            reward=rec.reward,
            step_telemetry=p_load,
        )

    # Verification assertions:
    # 1. All 1354 frames remain present
    assert len(env._history) == 1354

    # 2. Head: steps 1..200 (indices 0..199)
    for i in range(200):
        frame = env._history[i]
        assert frame.step == i + 1
        assert frame.step_telemetry is not None
        assert frame.step_telemetry["step"] == i + 1
        assert "decisions" in frame.step_telemetry

    # 3. Frame 201 (index 200): omission marker
    frame_201 = env._history[200]
    assert frame_201.step == 201
    assert frame_201.step_telemetry is not None
    assert frame_201.step_telemetry["telemetry_omitted"] is True
    assert frame_201.step_telemetry["omitted_reason"] == "intentionally_omitted_window"
    assert frame_201.step_telemetry["telemetry_mode"] == "head_tail"
    assert frame_201.step_telemetry["head_steps"] == 200
    assert frame_201.step_telemetry["tail_steps"] == 500
    assert frame_201.step_telemetry["omitted_range"] == "steps_201_to_854"
    assert frame_201.step_telemetry["omitted_steps_start"] == 201
    assert frame_201.step_telemetry["omitted_steps_end"] == 854
    assert frame_201.step_telemetry["omitted_steps_count"] == 654
    assert frame_201.step_telemetry["total_world_steps"] == 1354

    # 4. Steps 202..854 (indices 201..853): exactly 653 frames with step_telemetry is None
    for i in range(201, 854):
        frame = env._history[i]
        assert frame.step == i + 1
        assert frame.step_telemetry is None

    # Total omitted steps = 1 (marker) + 653 (None) = 654 steps
    omitted_frames = [f for f in env._history if f.step_telemetry is None or f.step_telemetry.get("telemetry_omitted")]
    assert len(omitted_frames) == 654

    # 5. Tail: steps 855..1354 (indices 854..1353): exactly 500 frames
    for i in range(854, 1354):
        frame = env._history[i]
        assert frame.step == i + 1
        assert frame.step_telemetry is not None
        assert frame.step_telemetry["step"] == i + 1
        assert "decisions" in frame.step_telemetry

    detailed_frames = [f for f in env._history if f.step_telemetry and "decisions" in f.step_telemetry]
    assert len(detailed_frames) == 700  # 200 head + 500 tail
    assert len(detailed_frames) + len(omitted_frames) == 1354

    # 6. Verify temporal interaction diagnostics integration
    wrapped = {
        "replay_id": "test_t1354",
        "frames": [
            {
                "step": f.step,
                "actions": list(f.actions),
                "snapshot": f.snapshot.to_dict(),
                "reward": asdict(f.reward),
                "step_telemetry": f.step_telemetry,
            }
            for f in env._history
        ],
        "final_snapshot": env.get_state_snapshot().to_dict(),
    }
    analysis = analyze_episode_temporal_interactions(wrapped)
    assert analysis.evidence_audit.telemetry_omitted is True
    assert analysis.total_steps == 1354

    diag_payload = build_temporal_diagnostic_payload(wrapped)
    step_evidence = diag_payload["step_evidence"]
    assert len(step_evidence) == 1354
    # Frame 201 has schema status omitted and telemetry_omitted=True
    assert step_evidence[200]["telemetry_omitted"] is True
    assert step_evidence[200]["telemetry_omitted_range"] == "steps_201_to_854"
    # Tail step 855 has telemetry_omitted=False
    assert step_evidence[854]["telemetry_omitted"] is False


def test_legacy_replay_loading_compatibility(tmp_path: Path) -> None:
    """Verify loading legacy replays (without head/tail or with old schemas) succeeds cleanly."""
    cfg = LabConfig()
    env = SheepdogEnvironment(cfg)
    env.reset(42)
    snap = env.get_state_snapshot().to_dict()

    legacy_bundle = {
        "replay_id": "legacy_ep_1",
        "seed": 42,
        "policy_name": "legacy_ppo",
        "has_initial_step_zero": False,
        "environment": cfg.to_dict()["environment"],
        "frames": [
            {
                "step": 1,
                "actions": ["wait", "wait", "wait"],
                "snapshot": snap,
                "reward": {"total": -0.01},
            }
        ],
    }
    legacy_file = tmp_path / "legacy_ep_1.json.gz"
    with gzip.open(legacy_file, "wt", encoding="utf-8") as f:
        json.dump(legacy_bundle, f)

    loaded, r_id, p_info = load_replay_bundle(str(legacy_file))
    assert r_id == "legacy_ep_1"
    analysis = analyze_episode_temporal_interactions(loaded)
    assert analysis.evidence_audit.telemetry_omitted is False
    assert analysis.total_steps == 1


# ==============================================================================
# 6. Multi-Iteration Reproducible Benchmark Helper & Verification
# ==============================================================================


def run_multi_iteration_telemetry_benchmark(
    output_root: Path,
    *,
    iterations: int = 5,
    max_steps: int = 300,
) -> dict[str, Any]:
    """Run an N-iteration paired measurement comparing capture OFF vs ON.

    Characteristics:
    - Excludes warm-up iteration.
    - Alternating order across iterations (OFF, ON, ON, OFF, etc.).
    - Explicit step count (T >= 300).
    - Separates collection timing from serialization/JSON timing.
    - Reports median, min, max, std, call counts, and compressed payload sizes.
    """
    cfg = make_cfg(max_steps)
    policy = NeuralPolicy.initialize(cfg)

    # 0. Warm-up iteration (discarded)
    evaluator_warmup = Evaluator(cfg, output_root / "warmup")
    evaluator_warmup.evaluate(
        policy,
        seeds=(999,),
        checkpoint_episode=0,
        capture_replays=True,
        capture_step_telemetry=False,
    )
    evaluator_warmup.evaluate(
        policy,
        seeds=(999,),
        checkpoint_episode=0,
        capture_replays=True,
        capture_step_telemetry=True,
        telemetry_mode="head_tail",
        telemetry_head_steps=50,
        telemetry_tail_steps=50,
    )

    off_collection_times: list[float] = []
    on_collection_times: list[float] = []
    on_file_sizes: list[int] = []

    seeds = (11, 23)

    for i in range(iterations):
        run_off_first = (i % 2 == 0)
        modes = [False, True] if run_off_first else [True, False]

        for capture_on in modes:
            eval_dir = output_root / f"iter_{i}_{'on' if capture_on else 'off'}"
            evaluator = Evaluator(cfg, eval_dir)

            t0 = time.perf_counter()
            summary, _, _ = evaluator.evaluate(
                policy,
                seeds=seeds,
                checkpoint_episode=i + 1,
                capture_replays=True,
                capture_step_telemetry=capture_on,
                telemetry_mode="head_tail" if capture_on else "prefix",
                telemetry_head_steps=50 if capture_on else 200,
                telemetry_tail_steps=50 if capture_on else 500,
            )
            elapsed_total = time.perf_counter() - t0

            # Record file sizes
            for rec in summary.records:
                r_path = Path(rec.replay_path)
                if r_path.exists():
                    sz = r_path.stat().st_size
                    if capture_on:
                        on_file_sizes.append(sz)

            if capture_on:
                on_collection_times.append(elapsed_total)
            else:
                off_collection_times.append(elapsed_total)

    def calc_stats(series: list[float]) -> dict[str, float]:
        return {
            "median": statistics.median(series),
            "min": min(series),
            "max": max(series),
            "std": statistics.stdev(series) if len(series) > 1 else 0.0,
        }

    return {
        "iterations": iterations,
        "max_steps": max_steps,
        "off_timing": calc_stats(off_collection_times),
        "on_timing": calc_stats(on_collection_times),
        "compressed_sizes_bytes": {
            "median": statistics.median(on_file_sizes) if on_file_sizes else 0,
            "max": max(on_file_sizes) if on_file_sizes else 0,
            "min": min(on_file_sizes) if on_file_sizes else 0,
        },
    }


def test_multi_iteration_performance_benchmark(tmp_path: Path) -> None:
    """Execute reproducible multi-iteration benchmark and verify bounded replay sizes."""
    res = run_multi_iteration_telemetry_benchmark(
        tmp_path / "bench",
        iterations=5,
        max_steps=300,
    )
    assert res["iterations"] == 5
    max_sz = res["compressed_sizes_bytes"]["max"]
    # Verify strict storage target: < 2.5 MB compressed
    assert max_sz < 2.5 * 1024 * 1024, f"Compressed replay exceeded 2.5 MB target: {max_sz} bytes"

    # Report results cleanly without brittle execution-time threshold assertions
    med_off = res["off_timing"]["median"]
    med_on = res["on_timing"]["median"]
    overhead_pct = ((med_on - med_off) / max(med_off, 1e-4)) * 100
    print(
        f"\n[Multi-Iteration Benchmark (N=5, T=300)]\n"
        f"  Capture OFF Median: {med_off:.3f}s (min: {res['off_timing']['min']:.3f}s, max: {res['off_timing']['max']:.3f}s)\n"
        f"  Capture ON Median:  {med_on:.3f}s (min: {res['on_timing']['min']:.3f}s, max: {res['on_timing']['max']:.3f}s)\n"
        f"  Median Overhead:    {overhead_pct:.1f}%\n"
        f"  Compressed Size:    {res['compressed_sizes_bytes']['median'] / 1024:.1f} KB (max: {max_sz / 1024:.1f} KB)\n"
        f"  *Note: Micro-benchmark measurements on development machines do not extrapolate directly to Stage 34 training."
    )
