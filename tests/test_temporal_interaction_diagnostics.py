"""Focused unit tests for offline temporal dog-sheep interaction diagnostics and step instrumentation."""

from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from sheepdog.config import LabConfig
from sheepdog.entities import DogRole, DogState, Pen, Point, SheepState
from sheepdog.environment import SheepdogEnvironment
from sheepdog.policies.heuristic import HeuristicExpertPolicy
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.diagnostics.temporal_interaction import (
    _compute_discounted_return,
    analyze_episode_temporal_interactions,
    audit_replay_evidence,
    build_step_instrumentation_payload,
    compare_failure_and_recovery_episodes,
    format_temporal_diagnostic_report,
)
from sheepdog.training.replay_writer import (
    CapturePolicy,
    get_global_capture_policy,
    set_global_capture_policy,
)
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_frame(
    step: int,
    *,
    dogs: list[tuple[int, int, str, str]],
    sheep: list[tuple[int, int, bool, str]],
    pen_origin: tuple[int, int] = (10, 2),
    pen_size: tuple[int, int] = (8, 6),
    pen_opening: str = "bottom",
    width: int = 35,
    height: int = 25,
    avg_dist: float = 12.0,
    flock_spread: float = 3.0,
    no_progress_steps: int = 0,
    reward_total: float = -0.01,
    progress_to_pen: float = 0.0,
    status: str = "running",
    debug: dict[str, Any] | None = None,
    step_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ox, oy = pen_origin
    pw, ph = pen_size
    fence_cells: list[list[int]] = []
    for x in range(ox, ox + pw):
        for y in range(oy, oy + ph):
            on_border = x in (ox, ox + pw - 1) or y in (oy, oy + ph - 1)
            if not on_border:
                continue
            if pen_opening == "bottom" and y == oy + ph - 1 and (ox + 2 <= x <= ox + pw - 3):
                continue
            fence_cells.append([x, y])

    penned_cnt = sum(1 for _, _, penned, _ in sheep if penned)
    actions = [d[3] for d in dogs] if step > 0 else []
    frame: dict[str, Any] = {
        "step": step,
        "actions": actions,
        "snapshot": {
            "step": step,
            "simulated_seconds": step * 0.25,
            "grid_width": width,
            "grid_height": height,
            "field_width": width,
            "field_height": height,
            "dogs": [
                {
                    "index": idx,
                    "x": x,
                    "y": y,
                    "role": role,
                    "penned": False,
                    "last_action": act,
                    "personality": None,
                    "color": None,
                    "packed": False,
                }
                for idx, (x, y, role, act) in enumerate(dogs)
            ],
            "sheep": [
                {
                    "index": idx,
                    "x": x,
                    "y": y,
                    "role": None,
                    "penned": penned,
                    "last_action": "wait",
                    "personality": pers,
                    "color": "#ffffff",
                    "packed": False,
                }
                for idx, (x, y, penned, pers) in enumerate(sheep)
            ],
            "pen": {
                "origin": {"x": ox, "y": oy},
                "width": pw,
                "height": ph,
                "opening": pen_opening,
            },
            "fence_cells": fence_cells,
            "penned_count": penned_cnt,
            "average_distance_to_pen": avg_dist,
            "flock_spread": flock_spread,
            "no_progress_steps": no_progress_steps,
            "terminated": status in {"success", "timeout", "stopped"},
            "timeout": status == "timeout",
            "stopped": status == "stopped",
            "success": status == "success",
            "status": status,
            "debug": debug or {},
        },
        "reward": {
            "progress_to_pen": progress_to_pen,
            "total": reward_total,
        },
    }
    if step_telemetry is not None:
        frame["step_telemetry"] = step_telemetry
    return frame


def test_temporal_alignment_and_missing_data_audit() -> None:
    """Verify step alignment when step 0 is absent vs present and verify explicit missing-data reporting."""
    frames = []
    for step in range(1, 18):
        dx = step % 2
        frames.append(
            _make_frame(
                step,
                dogs=[
                    (12 + dx, 14, "rear_pressure", "right" if dx else "left"),
                    (10, 14, "left_flanker", "wait"),
                    (16, 14, "right_flanker", "wait"),
                ],
                sheep=[(14, 10, False, "obedient")] + [(12, 4, True, "obedient")] * 5,
                avg_dist=6.0,
                no_progress_steps=step,
                status="stopped" if step == 17 else "running",
            )
        )
    bundle_no_step0 = {
        "replay_id": "eval_no_step0",
        "seed": 42,
        "policy_name": "neural_policy",
        "environment": {"dogs": 3, "sheep": 6, "width": 35, "height": 25},
        "final_snapshot": frames[-1]["snapshot"],
        "stats": {"stop_reason": "no-progress", "oscillation_detected": False},
        "frames": frames,
    }

    analysis = analyze_episode_temporal_interactions(
        bundle_no_step0, min_interval_steps=10, window_size=10
    )
    assert not analysis.evidence_audit.has_initial_step_zero
    assert not analysis.evidence_audit.has_recorded_debug_targets
    assert not analysis.evidence_audit.has_recorded_step_telemetry
    assert not analysis.evidence_audit.has_recorded_action_probabilities
    assert any("step 0" in item for item in analysis.evidence_audit.unavailable_fields)
    assert any("probabilities" in item for item in analysis.evidence_audit.unavailable_fields)
    assert len(analysis.intervals) == 1

    iv = analysis.intervals[0]
    assert iv.start_step == 1
    assert iv.end_step == 17
    assert iv.unpenned_sheep_count == 1
    assert iv.dog_summaries[0].guidance_source == "reconstructed_team_strategy"
    assert iv.dog_summaries[0].legal_action_mask_source == "historical_approximate_no_carry"
    assert iv.dog_summaries[0].action_probabilities_available is False


def test_distinguishes_productive_repositioning_from_oscillation_and_stationary() -> None:
    """Verify productive flanking repositioning is distinguished from oscillation loops and stationary holds."""
    # Case A: Oscillation loop (dogs bounce left/right on gate side of sheep pinned on closed fence)
    # Net reward is negative (-0.01 per step), so positive progress steps do NOT claim reward farming.
    osc_frames = []
    for step in range(0, 20):
        bounce = step % 2
        osc_frames.append(
            _make_frame(
                step,
                dogs=[
                    (19 + bounce, 6, "rear_pressure", "right" if bounce else "left"),
                    (19, 8 - bounce, "left_flanker", "up" if bounce else "down"),
                    (20, 7 + bounce, "right_flanker", "down" if bounce else "up"),
                ],
                sheep=[(19, 5, False, "bold")] + [(12, 4, True, "obedient")] * 5,
                avg_dist=9.0,
                no_progress_steps=step,
                progress_to_pen=0.5 if bounce else -0.5,
                reward_total=-0.01,
                status="stopped" if step == 19 else "running",
            )
        )
    osc_analysis = analyze_episode_temporal_interactions(
        {"replay_id": "osc_case", "seed": 1, "frames": osc_frames},
        min_interval_steps=12,
        window_size=12,
    )
    assert len(osc_analysis.intervals) == 1
    assert osc_analysis.intervals[0].classification == "oscillation_loop"
    assert len(osc_analysis.intervals[0].reward_summary.cycles) > 0
    assert osc_analysis.intervals[0].reward_summary.has_net_negative_cycle_with_positive_steps is True
    assert osc_analysis.intervals[0].reward_summary.cycles[0].cycle_classification == "net_negative_cycle_with_positive_steps"
    assert "exploitable_positive_cycle" not in osc_analysis.intervals[0].candidate_mechanisms

    # Case B: Stationary hold (dogs wait continuously, zero net progress)
    hold_frames = [
        _make_frame(
            step,
            dogs=[
                (8, 18, "rear_pressure", "wait"),
                (14, 18, "left_flanker", "wait"),
                (20, 18, "right_flanker", "wait"),
            ],
            sheep=[(14, 12, False, "stubborn"), (15, 12, False, "obedient")]
            + [(12, 4, True, "obedient")] * 4,
            avg_dist=8.0,
            no_progress_steps=step,
        )
        for step in range(0, 18)
    ]
    hold_analysis = analyze_episode_temporal_interactions(
        {"replay_id": "hold_case", "seed": 2, "frames": hold_frames},
        min_interval_steps=12,
        window_size=12,
    )
    assert len(hold_analysis.intervals) == 1
    assert hold_analysis.intervals[0].classification == "stationary_hold"

    # Case C: Productive repositioning (dogs sweep wide around sheep from y=9 to y=17 during steps 0..14
    # while sheep route distance pauses at 10.0, then in aftermath steps 15..22 sheep advance into the pen)
    rep_frames = []
    for step in range(0, 15):
        rep_frames.append(
            _make_frame(
                step,
                dogs=[
                    (14, 8 + step, "rear_pressure", "down"),
                    (9, 8 + step, "left_flanker", "down"),
                    (19, 8 + step, "right_flanker", "down"),
                ],
                sheep=[(14, 12, False, "obedient")] + [(12, 4, True, "obedient")] * 5,
                avg_dist=10.0,
                no_progress_steps=step,
            )
        )
    for step in range(15, 23):
        prog = step - 14
        is_penned = step >= 21
        rep_frames.append(
            _make_frame(
                step,
                dogs=[
                    (14, 22 - prog, "rear_pressure", "up"),
                    (11, 20 - prog, "left_flanker", "up"),
                    (17, 20 - prog, "right_flanker", "up"),
                ],
                sheep=[(14, max(5, 12 - prog), is_penned, "obedient")]
                + [(12, 4, True, "obedient")] * 5,
                avg_dist=0.0 if is_penned else max(1.0, 10.0 - prog * 1.5),
                no_progress_steps=0,
                status="success" if step == 22 else "running",
            )
        )
    rep_analysis = analyze_episode_temporal_interactions(
        {"replay_id": "rep_case", "seed": 1, "frames": rep_frames},
        min_interval_steps=12,
        window_size=12,
    )
    assert len(rep_analysis.intervals) >= 1
    assert rep_analysis.intervals[0].classification == "productive_repositioning"
    assert rep_analysis.intervals[0].aftermath.sheep_penned_delta == 1

    comparison = compare_failure_and_recovery_episodes(osc_analysis, rep_analysis)
    assert comparison.shared_seed == 1
    assert "productive_repositioning" in comparison.recovery_interval_classifications
    report_md = format_temporal_diagnostic_report([osc_analysis, rep_analysis], [comparison])
    assert "osc_case" in report_md
    assert "rep_case" in report_md
    assert "productive_repositioning" in report_md


@pytest.mark.parametrize("unpenned_count", [1, 2, 3, 4, 5, 6])
def test_covers_all_unpenned_sheep_counts_and_two_sheep_blind_spot(unpenned_count: int) -> None:
    """Verify diagnostic covers any number of unpenned sheep (1..6) and detects the 2-sheep collector blind spot."""
    frames = []
    for step in range(0, 16):
        dx = step % 2
        sheep_list: list[tuple[int, int, bool, str]] = []
        for s_i in range(6):
            if s_i < unpenned_count:
                # For 2 unpenned sheep, place one on left exterior of pen (8, 5) and one on right exterior (19, 5)
                if unpenned_count == 2:
                    sx = 8 if s_i == 0 else 19
                    sy = 5
                else:
                    sx = 6 + s_i * 3
                    sy = 14
                sheep_list.append((sx, sy, False, "obedient"))
            else:
                sheep_list.append((13, 4, True, "obedient"))
        frames.append(
            _make_frame(
                step,
                dogs=[
                    (8 + dx, 18, "rear_pressure", "right" if dx else "left"),
                    (14 + dx, 18, "left_flanker", "right" if dx else "left"),
                    (20 + dx, 18, "right_flanker", "right" if dx else "left"),
                ],
                sheep=sheep_list,
                avg_dist=11.0,
                flock_spread=5.0 if unpenned_count == 2 else 3.5,
                no_progress_steps=step,
            )
        )

    analysis = analyze_episode_temporal_interactions(
        {"replay_id": f"unpenned_{unpenned_count}", "seed": 10 + unpenned_count, "frames": frames},
        min_interval_steps=12,
        window_size=12,
    )
    assert analysis.unpenned_count_coverage == (unpenned_count,)
    assert len(analysis.intervals) == 1
    assert analysis.intervals[0].unpenned_sheep_count == unpenned_count
    assert len(analysis.intervals[0].sheep_summaries) == unpenned_count
    if unpenned_count == 2:
        assert analysis.intervals[0].collector_blind_spot_steps == 16
        assert "two_sheep_collector_blind_spot" in analysis.intervals[0].candidate_mechanisms


def test_opt_in_step_instrumentation_does_not_alter_seeded_outcomes() -> None:
    """Verify that enabling capture_step_telemetry does not alter seeded simulation outcomes."""
    prev_policy = get_global_capture_policy()
    try:
        base_config = LabConfig()
        config = replace(
            base_config,
            environment=replace(base_config.environment, max_steps=25),
        )

        # Run 1: default capture_step_telemetry=False
        set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))
        adapter_default = SheepdogRLAdapter(config)
        obs_def, _ = adapter_default.reset(seed=77)
        rewards_def = []
        for step_idx in range(15 * adapter_default._environment.dog_count):
            obs_def, rew, term, trunc, _ = adapter_default.step(step_idx % 5)
            rewards_def.append(rew)
            if term or trunc:
                break
        frames_def = list(adapter_default._active_trajectory_buffer)

        # Run 2: opt-in capture_step_telemetry=True
        set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=True))
        adapter_inst = SheepdogRLAdapter(config)
        obs_inst, _ = adapter_inst.reset(seed=77)
        rewards_inst = []
        for step_idx in range(15 * adapter_inst._environment.dog_count):
            obs_inst, rew, term, trunc, _ = adapter_inst.step(step_idx % 5)
            rewards_inst.append(rew)
            if term or trunc:
                break
        frames_inst = list(adapter_inst._active_trajectory_buffer)

        assert (obs_def == obs_inst).all()
        assert rewards_def == rewards_inst
        assert len(frames_def) == len(frames_inst)
        for f_d, f_i in zip(frames_def, frames_inst, strict=True):
            assert f_d["step"] == f_i["step"]
            assert f_d["actions"] == f_i["actions"]
            assert f_d["snapshot"] == f_i["snapshot"]
            assert f_d["reward"] == f_i["reward"]
            if f_i["step"] > 0:
                assert "step_telemetry" not in f_d
                assert "step_telemetry" in f_i
                assert len(f_i["step_telemetry"]["pre_step_roles"]) == config.environment.dogs
                assert len(f_i["step_telemetry"]["legal_action_masks"]) == config.environment.dogs

        # Verify temporal analyzer recognizes recorded_step_telemetry when present
        inst_analysis = analyze_episode_temporal_interactions(
            {"replay_id": "inst_run", "seed": 77, "frames": frames_inst},
            min_interval_steps=6,
            window_size=6,
        )
        assert inst_analysis.evidence_audit.has_recorded_step_telemetry is True

        # Also check direct environment determinism with build_step_instrumentation_payload
        env_a = SheepdogEnvironment(config)
        res_a = env_a.run_policy(HeuristicExpertPolicy(), seed=19, capture_replay=True)

        env_b = SheepdogEnvironment(config)
        env_b.reset(seed=19)
        policy = HeuristicExpertPolicy()
        while not env_b.get_state_snapshot().terminated:
            actions = policy.select_actions(env_b, deterministic=True)
            _ = build_step_instrumentation_payload(env_b, raw_actions=actions)
            env_b.step(actions, capture_replay=True)
        assert env_a.get_state_snapshot().to_dict() == env_b.get_state_snapshot().to_dict()
        assert res_a.stats.reward_total == env_b._stats.reward_total
    finally:
        set_global_capture_policy(prev_policy)


def test_neural_policy_decision_telemetry_capture() -> None:
    """Verify that NeuralPolicy with capture_step_telemetry captures exact decision fields in a single forward pass."""
    from sheepdog.environment import ACTION_ORDER
    from sheepdog.policies.neural import NeuralPolicy

    config = LabConfig()
    policy = NeuralPolicy.initialize(config)
    env = SheepdogEnvironment(config, capture_step_telemetry=True)
    env.reset(seed=42)

    actions = policy.select_actions(env, deterministic=True)
    assert policy.latest_decision_telemetry is not None
    assert len(policy.latest_decision_telemetry) == config.environment.dogs

    for d_idx, dec in enumerate(policy.latest_decision_telemetry):
        assert dec["dog_index"] == d_idx
        assert dec["world_step"] == 0
        assert "observation" in dec
        assert len(dec["observation"]) > 0
        assert dec["decision_action_mask"] is not None
        assert len(dec["decision_action_mask"]) == len(ACTION_ORDER)
        assert dec["mask_type"] == "actual_decision_time"
        assert dec["selected_action"] in ACTION_ORDER
        assert dec["executed_action"] in ACTION_ORDER
        assert dec["action_probabilities"] is not None
        assert len(dec["action_probabilities"]) == len(ACTION_ORDER)
        assert abs(sum(dec["action_probabilities"].values()) - 1.0) < 1e-4
        assert dec["logits"] is not None
        assert dec["value_estimate"] is not None
        assert dec["decision_mode"] == "deterministic"

    # Step the environment with step_telemetry payload and verify it is attached to StepRecord
    telemetry = build_step_instrumentation_payload(env, decisions=policy.latest_decision_telemetry)
    env.step(actions, capture_replay=True, step_telemetry=telemetry)
    rec = env.history[-1]
    assert rec.step_telemetry is not None
    assert "decisions" in rec.step_telemetry
    assert len(rec.step_telemetry["decisions"]) == config.environment.dogs
    assert rec.step_telemetry["decisions"][0]["mask_type"] == "actual_decision_time"

    # Also test via run_policy loop with capture_step_telemetry=True
    env2 = SheepdogEnvironment(config)
    res2 = env2.run_policy(policy, seed=42, capture_replay=True, capture_step_telemetry=True)
    assert len(res2.replay) > 1
    assert res2.replay[1].step_telemetry is not None
    assert "decisions" in res2.replay[1].step_telemetry


def test_rl_adapter_illegal_action_replacement_and_decision_tracking() -> None:
    """Verify that SheepdogRLAdapter tracks raw selected actions vs executed actions and replacement reasons."""
    from sheepdog.entities import Point
    from sheepdog.environment import ACTION_ORDER

    prev_policy = get_global_capture_policy()
    try:
        set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=True))
        config = LabConfig()
        adapter = SheepdogRLAdapter(config)
        adapter.reset(seed=123)

        # Position dog 0 at boundary (0, 0) where 'up' and 'left' are illegal movements
        adapter._environment._dogs[0].position = Point(0, 0)
        mask = adapter.action_masks()
        illegal_action = None
        legal_action = None
        for a_idx, legal in enumerate(mask):
            if not legal and illegal_action is None:
                illegal_action = a_idx
            if legal and legal_action is None:
                legal_action = a_idx

        assert illegal_action is not None, "Expected at least one illegal action at boundary"
        assert legal_action is not None, "Expected at least one legal action"

        # 1. Step with illegal action: should be replaced by wait with replacement reason
        adapter.step(illegal_action)
        assert len(adapter._pending_decisions) == 1
        dec_illegal = adapter._pending_decisions[0]
        assert dec_illegal["dog_index"] == 0
        assert dec_illegal["selected_action"] == ACTION_ORDER[illegal_action]
        assert dec_illegal["executed_action"] == "wait"
        assert dec_illegal["replacement_reason"] == "action_masked_illegal"
        assert dec_illegal["decision_action_mask"][ACTION_ORDER[illegal_action]] is False
        assert dec_illegal["decision_action_mask"][ACTION_ORDER[legal_action]] is True
        assert dec_illegal["action_probabilities"] is None  # honestly unavailable during RL rollout
        assert dec_illegal["logits"] is None
        assert dec_illegal["value_estimate"] is None
        assert dec_illegal["decision_mode"] == "sampled"

        # 2. Step with legal action for dog 1: selected and executed should match
        adapter.step(legal_action)
        assert len(adapter._pending_decisions) == 2
        dec_legal = adapter._pending_decisions[1]
        assert dec_legal["dog_index"] == 1
        assert dec_legal["selected_action"] == ACTION_ORDER[legal_action]
        assert dec_legal["executed_action"] == ACTION_ORDER[legal_action]
        assert dec_legal["replacement_reason"] is None
    finally:
        set_global_capture_policy(prev_policy)


def test_historical_reconstruction_uncertainty_labels() -> None:
    """Verify that missing initial step 0 or missing configuration produces explicit uncertainty labels."""
    # 1. Missing step 0
    frames_no_step0 = [
        _make_frame(
            step,
            dogs=[(10, 10, "rear_pressure", "wait")] * 3,
            sheep=[(15, 15, False, "obedient")] * 6,
            no_progress_steps=step,
        )
        for step in range(2, 12)
    ]
    bundle_no_step0 = {"replay_id": "gap_rep", "seed": 42, "frames": frames_no_step0}
    analysis = analyze_episode_temporal_interactions(bundle_no_step0, min_interval_steps=4, window_size=4)
    assert len(analysis.intervals) == 1
    iv = analysis.intervals[0]
    assert iv.dog_summaries[0].guidance_certainty == "uncertain_reconstruction_missing_step0"
    assert any("uncertain_reconstruction_missing_step0" in s for s in iv.dog_summaries[0].evidence_limitations)
    assert iv.dog_summaries[0].legal_action_mask_source == "historical_approximate_no_carry"

    # 2. Missing config
    frames_all = [
        _make_frame(
            step,
            dogs=[(10, 10, "rear_pressure", "wait")] * 3,
            sheep=[(15, 15, False, "obedient")] * 6,
            no_progress_steps=step,
        )
        for step in range(0, 10)
    ]
    bundle_no_cfg = {"replay_id": "no_cfg", "seed": 42, "frames": frames_all}
    analysis_no_cfg = analyze_episode_temporal_interactions(bundle_no_cfg, min_interval_steps=4, window_size=4)
    assert analysis_no_cfg.intervals[0].dog_summaries[0].guidance_certainty == "uncertain_reconstruction_missing_config"

    # 3. Frame gaps (discontinuous steps with config present)
    frames_gap = [
        _make_frame(
            step if step < 5 else step + 5,
            dogs=[(10, 10, "rear_pressure", "wait")] * 3,
            sheep=[(15, 15, False, "obedient")] * 6,
            no_progress_steps=step,
        )
        for step in range(0, 10)
    ]
    from dataclasses import asdict
    env_dict = asdict(LabConfig().environment)
    bundle_gap = {"replay_id": "disc_rep", "seed": 42, "environment": env_dict, "frames": frames_gap}
    analysis_gap = analyze_episode_temporal_interactions(bundle_gap, min_interval_steps=4, window_size=4)
    assert len(analysis_gap.intervals) >= 1
    iv_gap = analysis_gap.intervals[0]
    assert iv_gap.dog_summaries[0].guidance_certainty == "uncertain_reconstruction_discontinuous_steps"


def test_reward_cycle_accounting_classifications() -> None:
    """Verify that candidate positive return cycles, net-negative cycles with positive steps, and incomplete cycles are correctly distinguished."""
    # 1. Candidate position recurrence positive return: repeating loop with net positive return without lasting progress
    pos_cycle_frames = []
    for step in range(0, 16):
        bounce = step % 2
        pos_cycle_frames.append(
            _make_frame(
                step,
                dogs=[
                    (15 + bounce, 10, "rear_pressure", "right" if bounce else "left"),
                    (10, 10, "left_flanker", "wait"),
                    (20, 10, "right_flanker", "wait"),
                ],
                sheep=[(15, 6, False, "obedient")] + [(12, 4, True, "obedient")] * 5,
                progress_to_pen=0.0,
                reward_total=0.25 if bounce else 0.10,  # positive every step
            )
        )
    analysis_pos = analyze_episode_temporal_interactions(
        {
            "replay_id": "pos_cycle",
            "seed": 1,
            "gamma": 0.99,
            "discount_timebase": "joint_world_step",
            "frames": pos_cycle_frames,
        },
        min_interval_steps=8,
        window_size=8,
    )
    assert len(analysis_pos.intervals) == 1
    r_sum = analysis_pos.intervals[0].reward_summary
    assert r_sum.has_candidate_positive_return_cycle is True
    assert r_sum.has_exploitable_positive_cycle is True
    assert (
        "candidate_position_recurrence_positive_return"
        in analysis_pos.intervals[0].candidate_mechanisms
    )
    assert r_sum.cycles[0].discounting_status == "aligned_and_computed_joint_world_step"
    assert r_sum.cycles[0].discounted_return is not None
    assert r_sum.cycles[0].discount_factor == 0.99
    assert r_sum.cycles[0].matched_state == "joint_grid_positions"
    assert "no_progress_watermark" in r_sum.cycles[0].unmatched_or_unavailable_state

    # 2. Historical data with gamma but missing discount_timebase
    analysis_historical = analyze_episode_temporal_interactions(
        {"replay_id": "historical_pos", "seed": 1, "gamma": 0.99, "frames": pos_cycle_frames},
        min_interval_steps=8,
        window_size=8,
    )
    r_hist = analysis_historical.intervals[0].reward_summary
    assert (
        r_hist.cycles[0].discounting_status
        == "discount_timebase_unknown_historical_alignment_unavailable"
    )
    assert r_hist.cycles[0].discounted_return is None

    # 3. Incomplete cycle / discount factor unavailable
    analysis_no_gamma = analyze_episode_temporal_interactions(
        {"replay_id": "no_gamma", "seed": 1, "frames": pos_cycle_frames},
        min_interval_steps=8,
        window_size=8,
    )
    assert (
        analysis_no_gamma.intervals[0].reward_summary.cycles[0].discounting_status
        == "discount_factor_unavailable"
    )
    assert analysis_no_gamma.intervals[0].reward_summary.cycles[0].discounted_return is None


def test_passive_capture_rng_and_state_invariance() -> None:
    """Verify that enabling capture produces identical actions, observations, rewards, next states,

    terminations, and RNG progressions across Python, NumPy, PyTorch, and environment RNGs
    for both deterministic and sampled inference.
    """
    for deterministic in (True, False):

        def _run_trajectory(capture_enabled: bool, seed: int = 777, is_det: bool = deterministic) -> dict[str, Any]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            cfg = LabConfig()
            env = SheepdogEnvironment(cfg)
            env.capture_step_telemetry = capture_enabled
            env.reset(seed=seed)

            policy = NeuralPolicy.initialize(cfg)

            actions_taken = []
            rewards_received = []
            dog_positions = []
            sheep_positions = []
            terminations = []
            python_rng_states = []
            torch_rng_states = []

            for _ in range(8):
                acts = policy.select_actions(env, deterministic=is_det)
                actions_taken.append(acts)
                snapshot, reward = env.step(acts)
                rewards_received.append(reward)
                terminations.append(snapshot.terminated)
                dog_positions.append([(d.position.x, d.position.y) for d in env._dogs])
                sheep_positions.append([(s.position.x, s.position.y) for s in env._sheep])
                python_rng_states.append(random.getstate())
                torch_rng_states.append(torch.get_rng_state())

            return {
                "actions": actions_taken,
                "rewards": rewards_received,
                "dogs": dog_positions,
                "sheep": sheep_positions,
                "terminations": terminations,
                "py_rng": python_rng_states,
                "torch_rng": torch_rng_states,
            }

        res_off = _run_trajectory(capture_enabled=False)
        res_on = _run_trajectory(capture_enabled=True)

        assert (
            res_off["actions"] == res_on["actions"]
        ), f"Action mismatch in deterministic={deterministic}"
        assert res_off["rewards"] == res_on["rewards"]
        assert res_off["dogs"] == res_on["dogs"]
        assert res_off["sheep"] == res_on["sheep"]
        assert res_off["terminations"] == res_on["terminations"]
        assert res_off["py_rng"] == res_on["py_rng"]
        for t_off, t_on in zip(res_off["torch_rng"], res_on["torch_rng"], strict=True):
            assert torch.equal(t_off, t_on)


def test_passive_capture_injected_telemetry_failure_no_reselection() -> None:
    """Verify that an injected exception in telemetry extraction does not cause reselection

    and leaves the normal selected action unchanged while recording capture_error.
    """
    cfg = LabConfig()
    env = SheepdogEnvironment(cfg)
    env.capture_step_telemetry = True
    env.reset(seed=123)

    policy = NeuralPolicy.initialize(cfg)

    # First observe the normal selected action with capture=False
    env.capture_step_telemetry = False
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    normal_actions = policy.select_actions(env, deterministic=True)

    # Reset environment and RNGs to same state
    env.reset(seed=123)
    env.capture_step_telemetry = True
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    call_count = 0
    original_predict = policy.model.predict

    def _predict_spy(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_predict(*args, **kwargs)

    with (
        patch.object(policy.model, "predict", side_effect=_predict_spy),
        patch.object(
            policy.model.policy,
            "predict_values",
            side_effect=RuntimeError("Simulated GPU value prediction failure"),
        ),
    ):
        actions = policy.select_actions(env, deterministic=True)

    # Model predict must have been called exactly once per dog (3 dogs, no second predict due to failure)
    assert call_count == len(env._dogs)
    # Action must be identical to the action without failure
    assert actions == normal_actions
    # Telemetry decisions should record capture_error
    assert policy.latest_decision_telemetry is not None
    assert len(policy.latest_decision_telemetry) == len(actions)
    for record in policy.latest_decision_telemetry:
        assert record.get("capture_error") is not None
        assert "Simulated GPU value prediction failure" in record["capture_error"]


def test_discount_timebase_hand_calculated_sequential_and_joint() -> None:
    """Verify hand-calculated discounted returns for sequential shared-dog and joint world-step transitions."""
    st0 = MagicMock()
    st0.step = 1
    st0.reward_dict = {"total": 1.0}

    st1 = MagicMock()
    st1.step = 2
    st1.reward_dict = {"total": 2.0}

    cycle_steps = [st0, st1]

    # 1. Sequential shared-dog:
    # delta_tau for step 0: 0 * 3 + 2 = 2. weight = 0.9^2 = 0.81
    # delta_tau for step 1: 1 * 3 + 2 = 5. weight = 0.9^5 = 0.59049
    # expected = 1.0 * 0.81 + 2.0 * 0.59049 = 0.81 + 1.18098 = 1.99098
    ret_seq, status_seq = _compute_discounted_return(
        cycle_steps, discount_factor=0.9, discount_timebase="sequential_shared_dog", dog_count=3
    )
    assert status_seq == "aligned_and_computed_sequential_shared_dog"
    assert ret_seq is not None
    assert pytest.approx(ret_seq, rel=1e-5) == 1.99098

    # 2. Joint world-step:
    # weight for step 0: 0.9^0 = 1.0
    # weight for step 1: 0.9^1 = 0.9
    # expected = 1.0 * 1.0 + 2.0 * 0.9 = 2.8
    ret_joint, status_joint = _compute_discounted_return(
        cycle_steps, discount_factor=0.9, discount_timebase="joint_world_step", dog_count=3
    )
    assert status_joint == "aligned_and_computed_joint_world_step"
    assert ret_joint is not None
    assert pytest.approx(ret_joint, rel=1e-5) == 2.8


def test_discount_timebase_missing_and_frame_gaps() -> None:
    """Verify that missing timebase or frame gaps mark discounted return unavailable with explicit status."""
    st0 = MagicMock()
    st0.step = 1
    st0.reward_dict = {"total": 1.0}

    st1 = MagicMock()
    st1.step = 2
    st1.reward_dict = {"total": 2.0}

    # Missing discount_timebase:
    ret_no_tb, status_no_tb = _compute_discounted_return(
        [st0, st1], discount_factor=0.99, discount_timebase=None, dog_count=3
    )
    assert ret_no_tb is None
    assert status_no_tb == "discount_timebase_unknown_historical_alignment_unavailable"

    # Frame gaps (e.g. cycle steps are discontinuous):
    ret_gap, status_gap = _compute_discounted_return(
        [st0, st1],
        discount_factor=0.99,
        discount_timebase="joint_world_step",
        dog_count=3,
        cycle_is_contiguous=False,
    )
    assert ret_gap is None
    assert status_gap == "broken_transition_alignment_frame_gaps"


def test_historical_reconstruction_strategy_version_and_config_validation() -> None:
    """Verify explicit validation of strategy_version and required configuration fields."""
    env_full = asdict(LabConfig().environment)
    frames = [
        _make_frame(
            step,
            dogs=[(10, 10, "rear_pressure", "wait")] * 3,
            sheep=[(15, 15, False, "obedient")] * 6,
        )
        for step in range(0, 10)
    ]

    # 1. Missing strategy_version
    bundle_missing_ver = {
        "replay_id": "missing_ver",
        "environment": dict(env_full),
        "frames": frames,
    }
    audit_missing_ver = audit_replay_evidence(
        bundle_missing_ver,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        strategy_version=None,
    )
    assert (
        audit_missing_ver.guidance_certainty
        == "estimated_reconstruction_missing_strategy_version"
    )
    assert audit_missing_ver.strategy_version_status == "missing_strategy_version"

    # 2. Strategy version mismatch
    audit_mismatch = audit_replay_evidence(
        bundle_missing_ver,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        strategy_version="0.5.0",
    )
    assert (
        audit_mismatch.guidance_certainty
        == "estimated_reconstruction_strategy_version_mismatch"
    )
    assert audit_mismatch.strategy_version_status == "incompatible_strategy_version"

    # 3. Missing required config fields (e.g. gate_approach_distance)
    incomplete_env = {k: v for k, v in env_full.items() if k != "gate_approach_distance"}
    bundle_missing_cfg = {
        "replay_id": "missing_cfg",
        "environment": incomplete_env,
        "strategy_version": TeamStrategy.VERSION,
        "frames": frames,
    }
    audit_missing_cfg = audit_replay_evidence(
        bundle_missing_cfg,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        strategy_version=TeamStrategy.VERSION,
    )
    assert audit_missing_cfg.guidance_certainty == "uncertain_reconstruction_missing_config"
    assert "gate_approach_distance" in audit_missing_cfg.missing_required_config

    # 4. Reconstruction when initial strategy state is unrecorded: estimated reconstruction
    bundle_unrecorded_state = {
        "replay_id": "estimated_rep",
        "environment": dict(env_full),
        "strategy_version": TeamStrategy.VERSION,
        "frames": frames,
    }
    audit_estimated = audit_replay_evidence(
        bundle_unrecorded_state,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        has_initial_strategy_state=False,
        strategy_version=TeamStrategy.VERSION,
    )
    assert audit_estimated.guidance_certainty == "estimated_reconstruction_unrecorded_initial_strategy_state"
    assert audit_estimated.strategy_version_status == "compatible_strategy_version"
    assert len(audit_estimated.missing_required_config) == 0
    assert any("initial strategy internal state" in s for s in audit_estimated.unavailable_fields)

    # 5. Reconstructed guidance is conservatively classified as estimated, never exact
    bundle_reconstructed = {
        "replay_id": "reconstructed_rep",
        "environment": dict(env_full),
        "strategy_version": TeamStrategy.VERSION,
        "frames": frames,
    }
    audit_reconstructed = audit_replay_evidence(
        bundle_reconstructed,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        strategy_version=TeamStrategy.VERSION,
    )
    assert audit_reconstructed.guidance_certainty == "estimated_reconstruction_unrecorded_initial_strategy_state"
    assert audit_reconstructed.guidance_certainty != "exact_deterministic_reconstruction"
    assert audit_reconstructed.strategy_version_status == "compatible_strategy_version"
    assert len(audit_reconstructed.missing_required_config) == 0
    assert any("initial strategy internal state" in s for s in audit_reconstructed.unavailable_fields)


def test_interval_specific_partial_runtime_telemetry() -> None:
    """Verify that an interval with telemetry on only a subset of steps is labeled partial_runtime_telemetry."""
    env_full = asdict(LabConfig().environment)
    frames = []
    for step in range(0, 16):
        tel = None
        if step < 6:
            tel = {
                "pre_step_roles": [
                    {"dog_index": 0, "current_role": "rear_pressure", "target": {"x": 10, "y": 10}},
                    {"dog_index": 1, "current_role": "left_flanker", "target": {"x": 8, "y": 10}},
                    {"dog_index": 2, "current_role": "right_flanker", "target": {"x": 12, "y": 10}},
                ]
            }
        frames.append(
            _make_frame(
                step,
                dogs=[(10, 10, "rear_pressure", "wait")] * 3,
                sheep=[(15, 15, False, "obedient")] * 6,
                step_telemetry=tel,
            )
        )
    bundle = {
        "replay_id": "partial_tel_rep",
        "environment": env_full,
        "strategy_version": TeamStrategy.VERSION,
        "frames": frames,
    }
    analysis = analyze_episode_temporal_interactions(bundle, min_interval_steps=8, window_size=8)
    assert len(analysis.intervals) >= 1
    iv = analysis.intervals[0]
    assert iv.guidance_certainty == "partial_runtime_telemetry"
    assert any("partial_runtime_telemetry" in lim for lim in iv.evidence_limitations)


def test_candidate_position_recurrence_with_divergent_hidden_state() -> None:
    """Verify that matching joint grid positions across recurrence conceal divergent hidden state."""
    frames = []
    for step in range(0, 16):
        bounce = step % 4
        # Position is identical at step 0 and step 4, but no_progress_steps increases monotonically
        frames.append(
            _make_frame(
                step,
                dogs=[(10 + (bounce % 2), 10, "rear_pressure", "wait")] * 3,
                sheep=[(15, 15, False, "obedient")] * 6,
                no_progress_steps=step * 2,  # divergent internal hidden state
                reward_total=0.05,
            )
        )
    bundle = {
        "replay_id": "hidden_divergence",
        "seed": 42,
        "gamma": 0.99,
        "discount_timebase": "joint_world_step",
        "frames": frames,
    }
    analysis = analyze_episode_temporal_interactions(bundle, min_interval_steps=8, window_size=8)
    assert len(analysis.intervals) >= 1
    iv = analysis.intervals[0]
    assert iv.reward_summary.has_candidate_positive_return_cycle is True
    cycle = iv.reward_summary.cycles[0]
    assert cycle.cycle_classification == "candidate_position_recurrence_positive_return"
    assert cycle.matched_state == "joint_grid_positions"
    assert "no_progress_watermark" in cycle.unmatched_or_unavailable_state
    # Confirm that start and end no_progress_steps inside the interval differ despite position match
    assert iv.no_progress_steps_start != iv.no_progress_steps_end


def test_historical_reconstruction_omitted_strategy_settings() -> None:
    """Verify that omitting newly identified TeamStrategy settings (stickiness, blocker) flags missing config."""
    env_full = asdict(LabConfig().environment)
    frames = [
        _make_frame(step, dogs=[(10, 10, "rear_pressure", "wait")] * 3, sheep=[(15, 15, False, "obedient")] * 6)
        for step in range(5)
    ]
    omitted_fields = [
        "blocker_min_remaining_dogs",
        "role_minimum_hold_steps",
        "role_stickiness_distance",
        "flank_role_stickiness_bonus",
        "blocker_role_stickiness_bonus",
        "role_stickiness_bonus",
    ]
    for omitted_field in omitted_fields:
        incomplete_env = {k: v for k, v in env_full.items() if k != omitted_field}
        bundle = {
            "replay_id": f"missing_{omitted_field}",
            "environment": incomplete_env,
            "strategy_version": TeamStrategy.VERSION,
            "frames": frames,
        }
        audit = audit_replay_evidence(
            bundle,
            has_initial_step_zero=True,
            has_contiguous_frames=True,
            has_recorded_debug_targets=False,
            has_recorded_step_telemetry=False,
            has_recorded_action_probabilities=False,
            strategy_version=TeamStrategy.VERSION,
        )
        assert audit.guidance_certainty == "uncertain_reconstruction_missing_config"
        assert omitted_field in audit.missing_required_config


def test_historical_reconstruction_frame_gaps() -> None:
    """Verify that discontinuous step frames prevent exact reconstruction claims."""
    env_full = asdict(LabConfig().environment)
    # Frame steps 0, 1, 3 (missing step 2)
    frames = [
        _make_frame(0, dogs=[(10, 10, "rear_pressure", "wait")] * 3, sheep=[(15, 15, False, "obedient")] * 6),
        _make_frame(1, dogs=[(10, 10, "rear_pressure", "wait")] * 3, sheep=[(15, 15, False, "obedient")] * 6),
        _make_frame(3, dogs=[(10, 10, "rear_pressure", "wait")] * 3, sheep=[(15, 15, False, "obedient")] * 6),
    ]
    bundle = {
        "replay_id": "frame_gap_rep",
        "environment": env_full,
        "strategy_version": TeamStrategy.VERSION,
        "frames": frames,
    }
    audit = audit_replay_evidence(
        bundle,
        has_initial_step_zero=True,
        has_contiguous_frames=False,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        strategy_version=TeamStrategy.VERSION,
    )
    assert audit.guidance_certainty == "uncertain_reconstruction_discontinuous_steps"
    assert audit.has_contiguous_frames is False


def test_discount_provenance_unsupported_trainer_and_gamma_mismatch() -> None:
    """Verify that unsupported trainers and checkpoint/config gamma mismatches produce explicit status without aligned return."""
    env_full = asdict(LabConfig().environment)
    frames = [
        _make_frame(
            step,
            dogs=[(10, 10, "rear_pressure", "wait")] * 3,
            sheep=[(15, 15, False, "obedient")] * 6,
            reward_total=1.0,
        )
        for step in range(6)
    ]

    # 1. Unsupported trainer (e.g. hill_climb, baseline)
    bundle_unsupported = {
        "replay_id": "unsupported_trainer",
        "trainer_type": "hill_climb",
        "environment": env_full,
        "gamma": 0.99,
        "frames": frames,
    }
    audit_unsupported = audit_replay_evidence(
        bundle_unsupported,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        discount_factor=0.99,
        discount_timebase=None,
        trainer_type="hill_climb",
    )
    assert audit_unsupported.discount_timebase_status == "unsupported_trainer_discount_unavailable"

    disc_val, disc_status = _compute_discounted_return(
        [], 0.99, "sequential_shared_dog", 3, trainer_type="hill_climb"
    )
    assert disc_val is None
    assert disc_status == "unsupported_trainer_discount_unavailable"

    # 2. Checkpoint vs evaluation gamma mismatch
    bundle_mismatch = {
        "replay_id": "gamma_mismatch",
        "trainer_type": "maskable_ppo",
        "checkpoint_gamma": 0.95,
        "evaluation_gamma": 0.99,
        "environment": env_full,
        "frames": frames,
    }
    audit_mismatch = audit_replay_evidence(
        bundle_mismatch,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=False,
        has_recorded_action_probabilities=False,
        discount_factor=0.95,
        discount_timebase="sequential_shared_dog",
        trainer_type="maskable_ppo",
        checkpoint_gamma=0.95,
        evaluation_gamma=0.99,
        gamma_mismatch=True,
    )
    assert audit_mismatch.discount_timebase_status == "configuration_checkpoint_gamma_mismatch"
    assert audit_mismatch.gamma_mismatch is True

    disc_val_mismatch, disc_status_mismatch = _compute_discounted_return(
        [], 0.95, "sequential_shared_dog", 3, gamma_mismatch=True
    )
    assert disc_val_mismatch is None
    assert disc_status_mismatch == "configuration_checkpoint_gamma_mismatch"

    # 3. Analyze episode with mismatch: undiscounted total reward is preserved while discounted is None
    analysis = analyze_episode_temporal_interactions(bundle_mismatch, min_interval_steps=4, window_size=4)
    assert len(analysis.intervals) >= 1
    iv = analysis.intervals[0]
    assert iv.reward_summary.total_reward > 0.0
    for cycle in iv.reward_summary.cycles:
        assert cycle.discounted_return is None
        assert cycle.discounting_status == "configuration_checkpoint_gamma_mismatch"
        assert cycle.total_reward > 0.0  # Undiscounted measurement is preserved


def test_telemetry_semantics_and_bounded_overhead() -> None:
    """Verify telemetry semantics (recomputed diagnostics, provenance tags, bounded storage, and measurable forward passes)."""
    import json
    import time

    cfg = LabConfig()
    policy = NeuralPolicy.initialize(cfg)
    env = SheepdogEnvironment(cfg)
    env.reset(seed=123)

    # 1. Action selection with capture OFF
    env.capture_step_telemetry = False
    actions_off = policy.select_actions(env, deterministic=True)
    assert policy.latest_decision_telemetry is None
    assert len(actions_off) == env.dog_count

    # 2. Action selection with capture ON
    env.capture_step_telemetry = True
    actions_on = policy.select_actions(env, deterministic=True)
    assert actions_on == actions_off
    assert policy.latest_decision_telemetry is not None
    assert len(policy.latest_decision_telemetry) == env.dog_count

    # 3. Verify diagnostic provenance tags and full 9-action space
    for rec in policy.latest_decision_telemetry:
        assert rec["action_probabilities_source"] == "recomputed_diagnostic_forward_pass"
        assert rec["value_estimate_source"] == "recomputed_diagnostic_forward_pass"
        assert rec["capture_error"] is None
        assert rec["selected_action"] in rec["action_probabilities"]
        assert isinstance(rec["value_estimate"], float)
        # Action space has exactly 9 discrete actions: wait + 4 step moves + 4 sprint moves
        assert len(rec["decision_action_mask"]) == 9
        assert len(rec["recomputed_physical_mask"]) == 9
        assert len(rec["action_probabilities"]) == 9
        assert len(rec["logits"]) == 9

        # 4. Verify bounded payload size: serialized JSON per decision must be < 2.5 KB
        serialized = json.dumps(rec)
        assert len(serialized) < 2500, f"Telemetry record size {len(serialized)} bytes exceeds bounded limit of 2500"

    # 5. Measure runtime overhead across multiple iterations as a benchmark characteristic
    durations_off: list[float] = []
    durations_on: list[float] = []
    env.capture_step_telemetry = False
    for _ in range(5):
        t0 = time.perf_counter()
        policy.select_actions(env, deterministic=True)
        durations_off.append(time.perf_counter() - t0)

    env.capture_step_telemetry = True
    for _ in range(5):
        t0 = time.perf_counter()
        policy.select_actions(env, deterministic=True)
        durations_on.append(time.perf_counter() - t0)

    mean_off_ms = (sum(durations_off) / len(durations_off)) * 1000.0
    mean_on_ms = (sum(durations_on) / len(durations_on)) * 1000.0
    overhead_ms = max(0.0, mean_on_ms - mean_off_ms)
    # Generous bound to avoid brittle single-run CI failures while ensuring latency remains bounded
    assert overhead_ms < 150.0, f"Mean overhead {overhead_ms:.2f}ms exceeded bounded expectation"


def test_reconstruction_divergence_from_unrecorded_initial_strategy_state() -> None:
    """Verify that identical positions and config produce divergent targets when initial strategy state is unrecorded."""
    cfg = LabConfig()
    env_cfg = cfg.environment
    width, height = env_cfg.width, env_cfg.height
    pen = Pen(Point(10, 5), width=8, height=8, opening="bottom")

    strat_a = TeamStrategy(width, height, env_cfg)
    strat_b = TeamStrategy(width, height, env_cfg)

    # Strategy A has sticky_pen_dy = 1, Strategy B has sticky_pen_dy = -1
    strat_a._sticky_pen_dy = 1
    strat_b._sticky_pen_dy = -1

    # Place sheep directly at pen center where approach_target == pen.center
    sheep = [SheepState(index=0, position=pen.center, penned=False, personality="obedient")]
    dogs = [
        DogState(index=0, position=Point(10, 10), current_role=DogRole.REAR_PRESSURE),
        DogState(index=1, position=Point(12, 10), current_role=DogRole.LEFT_FLANKER),
        DogState(index=2, position=Point(16, 10), current_role=DogRole.RIGHT_FLANKER),
    ]

    assign_a, _ = strat_a.assign_roles(dogs, sheep, pen)
    assign_b, _ = strat_b.assign_roles(dogs, sheep, pen)

    # Flanker targets differ because lateral offset depends on _sticky_pen_dy
    target_a_d1 = (assign_a[1].target.x, assign_a[1].target.y)
    target_b_d1 = (assign_b[1].target.x, assign_b[1].target.y)
    assert target_a_d1 != target_b_d1, (
        f"Expected flanker targets to diverge due to unrecorded initial sticky state, but got identical {target_a_d1}"
    )


def test_capture_limit_exhaustion_invariance() -> None:
    """Verify that exhausting the per-episode telemetry budget stops capture without altering transitions or RNG state."""
    cfg = LabConfig()
    budget = 3

    # Run 1: Telemetry enabled with max_telemetry_steps = 3
    env_cap = SheepdogEnvironment(cfg, capture_step_telemetry=True)
    env_cap.max_telemetry_steps = budget
    policy_cap = NeuralPolicy.initialize(cfg)

    val_calls_cap = 0
    orig_val_cap = policy_cap.model.policy.predict_values

    def count_val_cap(*args: Any, **kwargs: Any) -> Any:
        nonlocal val_calls_cap
        val_calls_cap += 1
        return orig_val_cap(*args, **kwargs)

    policy_cap.model.policy.predict_values = count_val_cap

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res_cap = env_cap.run_policy(
        policy_cap,
        seed=999,
        capture_replay=True,
        deterministic=True,
        capture_step_telemetry=True,
        max_telemetry_steps=budget,
    )
    assert val_calls_cap == budget * env_cap.dog_count, (
        f"Diagnostic forward passes continued after budget: expected {budget * env_cap.dog_count}, got {val_calls_cap}"
    )
    rng_after_cap = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )

    # Run 2: Telemetry disabled
    env_nocap = SheepdogEnvironment(cfg, capture_step_telemetry=False)
    policy_nocap = NeuralPolicy.initialize(cfg)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    res_nocap = env_nocap.run_policy(
        policy_nocap,
        seed=999,
        capture_replay=True,
        deterministic=True,
        capture_step_telemetry=False,
    )
    rng_after_nocap = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )

    # 1. Assert bit-for-bit invariant outcomes and RNG progression between capture ON (truncated) and capture OFF
    assert len(res_cap.replay) == len(res_nocap.replay)
    for s in range(len(res_cap.replay)):
        rec_cap = res_cap.replay[s]
        rec_nocap = res_nocap.replay[s]
        assert rec_cap.actions == rec_nocap.actions
        assert rec_cap.reward.total == rec_nocap.reward.total
        assert rec_cap.snapshot.penned_count == rec_nocap.snapshot.penned_count
        assert rec_cap.snapshot.dogs == rec_nocap.snapshot.dogs
        assert rec_cap.snapshot.sheep == rec_nocap.snapshot.sheep

    assert rng_after_cap[0] == rng_after_nocap[0]
    assert torch.equal(rng_after_cap[2], rng_after_nocap[2])

    # 2. Verify telemetry truncation behavior on replay records
    for s in range(budget):
        assert res_cap.replay[s].step_telemetry is not None
        assert "decisions" in res_cap.replay[s].step_telemetry
    assert res_cap.replay[budget].step_telemetry is not None
    assert res_cap.replay[budget].step_telemetry.get("telemetry_truncated") is True
    assert res_cap.replay[budget].step_telemetry.get("truncation_reason") == "per_episode_telemetry_budget_exhausted"
    for s in range(budget + 1, len(res_cap.replay)):
        assert res_cap.replay[s].step_telemetry is None

    # 3. Verify temporal interaction diagnostic correctly flags partial coverage
    bundle = {
        "replay_id": "budget_trunc_rep",
        "environment": cfg.to_dict()["environment"],
        "strategy_version": TeamStrategy.VERSION,
        "telemetry_truncated": True,
        "telemetry_limit_steps": budget,
        "frames": [r.to_dict() for r in res_cap.replay],
    }
    audit = audit_replay_evidence(
        bundle,
        has_initial_step_zero=True,
        has_contiguous_frames=True,
        has_recorded_debug_targets=False,
        has_recorded_step_telemetry=True,
        has_recorded_action_probabilities=False,
        telemetry_truncated=True,
        telemetry_limit_steps=budget,
        telemetry_truncation_step=budget,
        strategy_version=TeamStrategy.VERSION,
    )
    assert audit.guidance_certainty == "partial_runtime_telemetry"
    assert audit.telemetry_truncated is True
    assert any("truncated at step 3" in s for s in audit.unavailable_fields)

    analysis = analyze_episode_temporal_interactions(bundle, min_interval_steps=4, window_size=4)
    assert len(analysis.intervals) >= 1
    assert any(
        iv.guidance_certainty in {"partial_runtime_telemetry", "mixed_recorded_and_reconstructed_telemetry"}
        for iv in analysis.intervals
    )


def test_diagnostic_calls_stop_at_budget_and_reset_safely() -> None:
    """Verify that optional diagnostic distribution/value forward passes stop at budget, support budget zero, reset across consecutive episodes, and restore on exceptions."""
    cfg = LabConfig()
    env = SheepdogEnvironment(cfg)
    policy = NeuralPolicy.initialize(cfg)

    val_calls = 0
    orig_val = policy.model.policy.predict_values

    def counting_predict_values(*args: Any, **kwargs: Any) -> Any:
        nonlocal val_calls
        val_calls += 1
        return orig_val(*args, **kwargs)

    policy.model.policy.predict_values = counting_predict_values

    # 1. Configured budget (budget = 2, 3 dogs -> exactly 6 diagnostic calls, zero on remaining steps)
    val_calls = 0
    res_b2 = env.run_policy(
        policy,
        seed=101,
        capture_replay=True,
        capture_step_telemetry=True,
        max_telemetry_steps=2,
    )
    assert len(res_b2.replay) > 2
    assert val_calls == 6, f"Expected exactly 6 diagnostic calls for budget=2, got {val_calls}"
    assert res_b2.replay[0].step_telemetry is not None
    assert res_b2.replay[1].step_telemetry is not None
    assert res_b2.replay[2].step_telemetry is not None
    assert res_b2.replay[2].step_telemetry.get("telemetry_truncated") is True
    for rec in res_b2.replay[3:]:
        assert rec.step_telemetry is None
    # Verify environment capture flag was restored
    assert env.capture_step_telemetry is True

    # 2. Budget zero (budget = 0 -> exactly 0 diagnostic calls, truncation marker at step 0)
    val_calls = 0
    res_b0 = env.run_policy(
        policy,
        seed=101,
        capture_replay=True,
        capture_step_telemetry=True,
        max_telemetry_steps=0,
    )
    assert len(res_b0.replay) > 0
    assert val_calls == 0, f"Expected 0 diagnostic calls for budget=0, got {val_calls}"
    assert res_b0.replay[0].step_telemetry is not None
    assert res_b0.replay[0].step_telemetry.get("telemetry_truncated") is True
    for rec in res_b0.replay[1:]:
        assert rec.step_telemetry is None
    assert env.capture_step_telemetry is True

    # 3. Consecutive episodes (budget = 1 each -> exactly 3 calls in ep1 and 3 calls in ep2)
    val_calls = 0
    res_ep1 = env.run_policy(
        policy,
        seed=201,
        capture_replay=True,
        capture_step_telemetry=True,
        max_telemetry_steps=1,
    )
    assert len(res_ep1.replay) > 1
    assert val_calls == 3
    assert env.capture_step_telemetry is True

    val_calls = 0
    res_ep2 = env.run_policy(
        policy,
        seed=202,
        capture_replay=True,
        capture_step_telemetry=True,
        max_telemetry_steps=1,
    )
    assert len(res_ep2.replay) > 1
    assert val_calls == 3
    assert env.capture_step_telemetry is True

    # 4. Exception safety: configuration is restored even if policy raises
    class FailingPolicy:
        def select_actions(self, environment: Any, deterministic: bool = True) -> list[str]:
            raise RuntimeError("simulated_policy_failure")

    env.capture_step_telemetry = True
    with pytest.raises(RuntimeError, match="simulated_policy_failure"):
        env.run_policy(
            FailingPolicy(),  # type: ignore[arg-type]
            seed=301,
            capture_step_telemetry=True,
            max_telemetry_steps=1,
        )
    assert env.capture_step_telemetry is True


def test_build_temporal_diagnostic_payload_and_observation_schema_labels() -> None:
    """Verify build_temporal_diagnostic_payload synchronizes step evidence, observation schemas, and plain-language explanations."""
    from sheepdog.training.diagnostics.temporal_interaction import (
        build_temporal_diagnostic_payload,
        label_dog_observation_fields,
    )

    # 1. Verify observation schema labeling across 54 (role_aware), 52 (emergent), 69 (hierarchical), unverified, and None
    obs_54 = label_dog_observation_fields([float(i) * 0.1 for i in range(54)])
    assert obs_54["available"] is True
    assert obs_54["schema_status"] == "verified"
    assert "54" in obs_54["schema_name"]
    assert len(obs_54["fields"]) == 54
    assert obs_54["fields"][0]["key"] == "own_x"

    obs_52 = label_dog_observation_fields([0.0] * 52)
    assert obs_52["available"] is True
    assert obs_52["schema_status"] == "verified"
    assert "52" in obs_52["schema_name"]
    assert len(obs_52["fields"]) == 52

    obs_69 = label_dog_observation_fields([0.0] * 69)
    assert obs_69["available"] is True
    assert obs_69["schema_status"] == "verified"
    assert "69" in obs_69["schema_name"]
    assert len(obs_69["fields"]) == 69

    obs_unverified = label_dog_observation_fields([1.0, 2.0, 3.0])
    assert obs_unverified["available"] is False
    assert obs_unverified["schema_verified"] is False
    assert obs_unverified["schema_status"] == "unverified_schema"
    assert "3" in obs_unverified["reason"]

    obs_none = label_dog_observation_fields(None)
    assert obs_none["available"] is False
    assert obs_none["schema_status"] == "not_recorded"
    assert obs_none["fields"] == []

    # 2. Build a replay with missing step 0, a frame gap between step 5 and step 7, and recorded step telemetry on step 1
    frames = []
    for step in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]:
        bounce = step % 2
        step_tel = None
        if step == 1:
            step_tel = {
                "world_step": 1,
                "pre_step_roles": ["rear_pressure", "left_flanker", "right_flanker"],
                "pre_step_targets": [{"x": 14, "y": 14}, {"x": 10, "y": 12}, {"x": 18, "y": 12}],
                "legal_action_masks": [
                    {"wait": True, "up": True, "down": True, "left": False, "right": True},
                    {"wait": True, "up": True, "down": True, "left": True, "right": True},
                    {"wait": True, "up": True, "down": True, "left": True, "right": True},
                ],
                "decisions": [
                    {
                        "dog_index": 0,
                        "selected_action": "left",
                        "executed_action": "wait",
                        "replacement_reason": "action_masked_illegal",
                        "mask_type": "actual_decision_time",
                        "decision_action_mask": {
                            "wait": True,
                            "up": True,
                            "down": True,
                            "left": False,
                            "right": True,
                        },
                        "action_probabilities": {
                            "wait": 0.6,
                            "up": 0.1,
                            "down": 0.1,
                            "left": 0.0,
                            "right": 0.2,
                        },
                        "action_probabilities_source": "recomputed_diagnostic_forward_pass",
                        "value_estimate": 1.25,
                        "value_estimate_source": "recomputed_diagnostic_forward_pass",
                        "observation": [0.5] * 54,
                    }
                ],
            }
        frames.append(
            _make_frame(
                step,
                dogs=[
                    (14 + bounce, 14, "rear_pressure", "right" if bounce else "left"),
                    (10, 12, "left_flanker", "wait"),
                    (18, 12, "right_flanker", "wait"),
                ],
                sheep=[(14, 10, False, "obedient"), (15, 10, False, "stubborn")]
                + [(12, 4, True, "obedient")] * 4,
                avg_dist=7.0,
                no_progress_steps=step,
                step_telemetry=step_tel,
            )
        )

    bundle = {
        "replay_id": "ui_payload_test",
        "seed": 41,
        "frames": frames,
    }
    payload = build_temporal_diagnostic_payload(
        bundle, min_interval_steps=4, window_size=4, include_replay=True
    )
    assert payload["analysis"]["replay_id"] == "ui_payload_test"
    assert len(payload["analysis"]["intervals"]) >= 1
    first_iv = payload["analysis"]["intervals"][0]
    assert "readable_label" in first_iv
    assert "plain_language" in first_iv
    assert "what_was_observed" in first_iv["plain_language"]
    assert "candidate_explanation" in first_iv["plain_language"]
    assert "controlled_experiment_required" in first_iv["plain_language"]

    step_ev = payload["step_evidence"]
    assert len(step_ev) == len(frames)
    # Step 1 has no prior step (missing step 0)
    assert step_ev[0]["step"] == 1
    assert step_ev[0]["has_prior_contiguous_step"] is False
    assert step_ev[0]["dogs"][0]["pre_step_position"] is None
    assert step_ev[0]["dogs"][0]["decision"]["selected_action"] == "left"
    assert step_ev[0]["dogs"][0]["decision"]["executed_action"] == "wait"
    assert step_ev[0]["dogs"][0]["decision"]["replacement_reason"] == "action_masked_illegal"
    assert (
        step_ev[0]["dogs"][0]["decision"]["action_probabilities_source"]
        == "recomputed_diagnostic_forward_pass"
    )
    assert (
        step_ev[0]["dogs"][0]["decision"]["value_estimate_source"]
        == "recomputed_diagnostic_forward_pass"
    )
    assert step_ev[0]["dogs"][0]["decision"]["observation_labeled"]["available"] is True
    assert (
        "54"
        in step_ev[0]["dogs"][0]["decision"]["observation_labeled"]["schema_name"]
    )

    # Step 2 has contiguous prior step 1
    assert step_ev[1]["step"] == 2
    assert step_ev[1]["has_prior_contiguous_step"] is True
    assert step_ev[1]["dogs"][0]["pre_step_position"] is not None

    # Step 7 follows step 5 (frame gap) -> has_prior_contiguous_step must be False
    step_7 = next(item for item in step_ev if item["step"] == 7)
    assert step_7["has_prior_contiguous_step"] is False
    assert step_7["dogs"][0]["pre_step_position"] is None
    assert step_7["dogs"][0]["displacement"] is None


def test_temporal_diagnostics_server_routes_security_and_caching(tmp_path: Any, monkeypatch: Any) -> None:
    """Verify server read-only temporal diagnostic routes reject path traversal, handle missing/corrupt files, and cache analysis."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler

    import sheepdog.server as server_mod
    from sheepdog.server import ThreadingHTTPServer, TrainingRequestHandler

    replays_dir = tmp_path / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    valid_frames = [
        _make_frame(
            step,
            dogs=[
                (12 + (step % 2), 14, "rear_pressure", "right" if step % 2 else "left"),
                (10, 14, "left_flanker", "wait"),
                (16, 14, "right_flanker", "wait"),
            ],
            sheep=[(14, 10, False, "obedient")] + [(12, 4, True, "obedient")] * 5,
            avg_dist=6.0,
            no_progress_steps=step,
        )
        for step in range(1, 16)
    ]
    valid_replay_file = replays_dir / "valid-episode-001.json"
    valid_replay_file.write_text(
        json.dumps({"replay_id": "valid-episode-001", "seed": 11, "frames": valid_frames}),
        encoding="utf-8",
    )

    corrupt_replay_file = replays_dir / "corrupt-episode.json"
    corrupt_replay_file.write_text("{not valid json", encoding="utf-8")

    empty_frames_file = replays_dir / "empty-frames.json"
    empty_frames_file.write_text(json.dumps({"replay_id": "empty-frames", "frames": []}), encoding="utf-8")

    monkeypatch.setattr(server_mod, "_approved_replay_dirs", lambda: [replays_dir.resolve()])
    server_mod._TEMPORAL_DIAGNOSTICS_CACHE.clear()

    def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    with patch.object(BaseHTTPRequestHandler, "log_message"):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = httpd.server_port
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. Invalid replay IDs / path traversal attempts must be rejected with 400
            for bad_id in ["foo..bar", ".hidden", "bad$id", "a:b"]:
                status_code, _ = _get(port, f"/api/replays/{bad_id}/temporal-analysis")
                assert status_code == 400, f"Expected 400 for {bad_id}, got {status_code}"

            assert server_mod._validate_replay_id("../etc/passwd") is None
            assert server_mod._validate_replay_id("..\\windows\\win.ini") is None
            assert server_mod._validate_replay_id(".hidden") is None
            assert server_mod._validate_replay_id("valid-episode-001.json") == "valid-episode-001"

            # 2. Missing replay returns 404
            status_404, _ = _get(port, "/api/replays/nonexistent-replay-999/temporal-analysis")
            assert status_404 == 404

            # 3. Corrupt JSON or empty frames returns 422
            status_corrupt, _ = _get(port, "/api/replays/corrupt-episode/temporal-analysis")
            assert status_corrupt == 422

            status_empty, _ = _get(port, "/api/replays/empty-frames/temporal-analysis")
            assert status_empty == 422

            # 4. Valid replay returns 200 and caches results on repeated calls
            import sheepdog.training.diagnostics.temporal_interaction as ti_mod

            call_counter = {"count": 0}
            orig_builder = ti_mod.build_temporal_diagnostic_payload

            def counting_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
                call_counter["count"] += 1
                return orig_builder(*args, **kwargs)

            monkeypatch.setattr(ti_mod, "build_temporal_diagnostic_payload", counting_builder)

            status_ok1, data1 = _get(
                port, "/api/replays/valid-episode-001/temporal-analysis?include_replay=true"
            )
            assert status_ok1 == 200, f"Unexpected {status_ok1}: {data1}"
            assert data1["analysis"]["replay_id"] == "valid-episode-001"
            assert len(data1["step_evidence"]) == 15
            assert call_counter["count"] == 1

            # Second request for the same unchanged file must hit the cache (call_counter remains 1)
            status_ok2, _ = _get(
                port, "/api/replays/valid-episode-001/temporal-analysis?include_replay=true"
            )
            assert status_ok2 == 200
            assert call_counter["count"] == 1

            # Examples endpoint works
            status_ex, data_ex = _get(port, "/api/temporal-diagnostics/examples")
            assert status_ex == 200
            assert "examples" in data_ex

            # 5. Read-only preview mode health, training status (without TrainingManager), and POST mutation rejection
            monkeypatch.setattr(server_mod, "_READONLY_PREVIEW_MODE", True)
            status_health, data_health = _get(port, "/api/health")
            assert status_health == 200
            assert data_health["ready"] is True
            assert data_health["read_only_preview"] is True
            assert data_health["mode"] == "read_only_preview"

            status_ready, data_ready = _get(port, "/api/ready")
            assert status_ready == 200
            assert data_ready["read_only_preview"] is True

            status_ts, data_ts = _get(port, "/api/training/status")
            assert status_ts == 200
            assert data_ts["read_only_preview"] is True
            assert data_ts["mode"] == "read_only_preview"
            assert data_ts["phase"] == "read_only_preview"
            assert data_ts["running"] is False
            assert data_ts["curriculum_stage"] is None

            post_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/training/start",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(post_req, timeout=5) as resp:
                    post_code = resp.status
                    post_body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                post_code = err.code
                post_body = json.loads(err.read().decode("utf-8"))
            assert post_code == 403
            assert post_body["code"] == "READ_ONLY_PREVIEW"
            assert post_body["read_only_preview"] is True
        finally:
            httpd.shutdown()
            httpd.server_close()

