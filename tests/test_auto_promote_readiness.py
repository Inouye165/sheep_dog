"""Automated unit tests for Sheepdog Auto-Promotion Readiness Logic."""

import json
from pathlib import Path
import pytest
from sheepdog.server import (
    _get_success_threshold,
    _auto_promote_gate_defaults,
    compute_promotion_gate_snapshot,
)


def _make_eval_payload(
    episode: int,
    stage: int,
    success_rate: float,
    failed_seeds: list[int] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    all_seeds = seeds or [11, 23, 37, 41, 53, 59, 61, 67, 71, 73]
    failures = set(failed_seeds or [])
    records = []
    for s in all_seeds:
        records.append({
            "seed": s,
            "success": s not in failures,
            "timeout": False,
            "stopped": False,
            "steps": 200,
        })
    return {
        "checkpoint_episode": episode,
        "checkpoint_id": f"chk_{episode:06d}",
        "curriculum_stage": stage,
        "policy_version": 1,
        "success_rate": success_rate,
        "average_reward": 200.0,
        "timeout_rate": 0.0,
        "evaluation_seeds": all_seeds,
        "evaluation_seed_count": len(all_seeds),
        "records": records,
    }


def _setup_eval_dir(tmp_path: Path, evals: list[dict]):
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    for ev in evals:
        ep = ev["checkpoint_episode"]
        with open(eval_dir / f"evaluation-checkpoint-{ep:06d}.json", "w") as f:
            json.dump(ev, f)
        with open(chk_dir / f"checkpoint-{ep:06d}.json", "w") as f:
            json.dump(ev, f)


def test_stage_thresholds():
    """Verify Stage 1 defaults to 80%, Stage 2-9 default to 90%, and curriculum overrides are respected."""
    assert _get_success_threshold(1) == 0.80
    assert _get_success_threshold(2) == 0.90
    assert _get_success_threshold(4) == 0.90
    assert _get_success_threshold(10) == 0.70  # Explicit curriculum config override for stage 10


def test_insufficient_history_collecting_evidence(tmp_path: Path):
    """Fewer than 6 formal evaluations cannot promote and report COLLECTING EVIDENCE."""
    evals = [
        _make_eval_payload(10, stage=4, success_rate=1.0),
        _make_eval_payload(20, stage=4, success_rate=1.0),
        _make_eval_payload(30, stage=4, success_rate=1.0),
        _make_eval_payload(40, stage=4, success_rate=1.0),
        _make_eval_payload(50, stage=4, success_rate=1.0),
    ]
    _setup_eval_dir(tmp_path, evals)

    res = compute_promotion_gate_snapshot(tmp_path, target_ep=50)
    assert res["ready"] is False
    assert res["decision"] == "pending"
    assert res["status_text"] == "COLLECTING EVIDENCE"
    assert res["formal_evaluations_available"] == 5
    assert res["formal_evaluations_required"] == 6
    assert "at least 6 required" in res["blocking_reasons"][0]


def test_six_evaluations_minimum_boundary(tmp_path: Path):
    """At 6 evaluations: 5/6 (83.3% >= 75%) passes; 4/6 (66.7% < 75%) is blocked."""
    # Case A: 5/6 qualifying with average >= 90%
    evals_pass = [
        _make_eval_payload(10, stage=4, success_rate=0.9),
        _make_eval_payload(20, stage=4, success_rate=1.0),
        _make_eval_payload(30, stage=4, success_rate=0.9),
        _make_eval_payload(40, stage=4, success_rate=1.0),
        _make_eval_payload(50, stage=4, success_rate=0.8),  # 1 non-qualifying
        _make_eval_payload(60, stage=4, success_rate=1.0),
    ]
    _setup_eval_dir(tmp_path, evals_pass)
    res_pass = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert res_pass["ready"] is True
    assert res_pass["decision"] == "promote_ready"
    assert res_pass["status_text"] == "READY TO PROMOTE"
    assert res_pass["qualified_evaluations"] == 5
    assert res_pass["qualified_evaluations_required"] == 5  # ceil(6 * 0.75) = 5

    # Case B: 4/6 qualifying (blocked)
    tmp_b = tmp_path / "case_b"
    evals_block = [
        _make_eval_payload(10, stage=4, success_rate=0.9),
        _make_eval_payload(20, stage=4, success_rate=0.8),  # non-qualifying
        _make_eval_payload(30, stage=4, success_rate=0.9),
        _make_eval_payload(40, stage=4, success_rate=0.8),  # non-qualifying
        _make_eval_payload(50, stage=4, success_rate=0.9),
        _make_eval_payload(60, stage=4, success_rate=1.0),
    ]
    _setup_eval_dir(tmp_b, evals_block)
    res_block = compute_promotion_gate_snapshot(tmp_b, target_ep=60)
    assert res_block["ready"] is False
    assert res_block["decision"] == "blocked"
    assert res_block["status_text"] == "NOT READY"
    assert res_block["qualified_evaluations"] == 4
    assert res_block["qualified_evaluations_required"] == 5


def test_eight_evaluation_window_consistency(tmp_path: Path):
    """At 8 evaluations: 6/8 (75%) passes; 5/8 (62.5%) is blocked."""
    evals_pass = [
        _make_eval_payload(10, stage=4, success_rate=0.9),
        _make_eval_payload(20, stage=4, success_rate=0.9),
        _make_eval_payload(30, stage=4, success_rate=0.8),  # non-qualifying
        _make_eval_payload(40, stage=4, success_rate=0.9),
        _make_eval_payload(50, stage=4, success_rate=1.0),
        _make_eval_payload(60, stage=4, success_rate=0.8),  # non-qualifying
        _make_eval_payload(70, stage=4, success_rate=1.0),
        _make_eval_payload(80, stage=4, success_rate=1.0),
    ]
    _setup_eval_dir(tmp_path, evals_pass)
    res_pass = compute_promotion_gate_snapshot(tmp_path, target_ep=80)
    assert res_pass["ready"] is True
    assert res_pass["qualified_evaluations"] == 6
    assert res_pass["qualified_evaluations_required"] == 6  # ceil(8 * 0.75) = 6


def test_window_average_success_guard(tmp_path: Path):
    """Even if 6 of 8 qualify, if extreme low scores drag the window average below 90%, it blocks."""
    evals = [
        _make_eval_payload(10, stage=4, success_rate=0.9),
        _make_eval_payload(20, stage=4, success_rate=0.9),
        _make_eval_payload(30, stage=4, success_rate=0.0),  # extreme failure
        _make_eval_payload(40, stage=4, success_rate=0.9),
        _make_eval_payload(50, stage=4, success_rate=0.9),
        _make_eval_payload(60, stage=4, success_rate=0.0),  # extreme failure
        _make_eval_payload(70, stage=4, success_rate=0.9),
        _make_eval_payload(80, stage=4, success_rate=0.9),
    ]
    _setup_eval_dir(tmp_path, evals)
    res = compute_promotion_gate_snapshot(tmp_path, target_ep=80)
    assert res["ready"] is False
    assert res["qualified_evaluations"] == 6
    assert res["recent_average_success"] < 0.90
    assert any("Recent evaluation average is" in reason for reason in res["blocking_reasons"])


def test_persistent_seed_failure_blocks_promotion(tmp_path: Path):
    """Seed failing in 3 consecutive formal evaluations blocks auto-promotion."""
    # Seed 71 fails in episodes 40, 50, 60 (3 in a row)
    evals = [
        _make_eval_payload(10, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(20, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(30, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(40, stage=4, success_rate=0.9, failed_seeds=[71]),
        _make_eval_payload(50, stage=4, success_rate=0.9, failed_seeds=[71]),
        _make_eval_payload(60, stage=4, success_rate=0.9, failed_seeds=[71]),
    ]
    _setup_eval_dir(tmp_path, evals)
    res = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert res["ready"] is False
    assert res["persistent_seed_failure"] is True
    assert res["blocking_seed"] == 71
    assert res["blocking_seed_consecutive_failures"] == 3
    assert any("Seed 71 failed in 3 consecutive formal evaluations." in r for r in res["blocking_reasons"])


def test_non_consecutive_seed_failures_do_not_block(tmp_path: Path):
    """FAIL, PASS, FAIL or only 2 failures do not trigger the persistent seed failure block."""
    # Seed 71 fails in ep 30, passes in ep 40, fails in ep 50
    evals = [
        _make_eval_payload(10, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(20, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(30, stage=4, success_rate=0.9, failed_seeds=[71]),
        _make_eval_payload(40, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(50, stage=4, success_rate=0.9, failed_seeds=[71]),
        _make_eval_payload(60, stage=4, success_rate=1.0, failed_seeds=[]),
    ]
    _setup_eval_dir(tmp_path, evals)
    res = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert res["ready"] is True
    assert res["persistent_seed_failure"] is False
    assert res["blocking_seed"] is None


def test_different_seeds_failing_do_not_combine(tmp_path: Path):
    """Failures on 3 different seeds across evaluations do not trigger a single-seed blocker."""
    evals = [
        _make_eval_payload(10, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(20, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(30, stage=4, success_rate=1.0, failed_seeds=[]),
        _make_eval_payload(40, stage=4, success_rate=0.9, failed_seeds=[11]),
        _make_eval_payload(50, stage=4, success_rate=0.9, failed_seeds=[23]),
        _make_eval_payload(60, stage=4, success_rate=0.9, failed_seeds=[37]),
    ]
    _setup_eval_dir(tmp_path, evals)
    res = compute_promotion_gate_snapshot(tmp_path, target_ep=60)
    assert res["ready"] is True
    assert res["persistent_seed_failure"] is False


def test_history_window_truncation_to_latest_eight(tmp_path: Path):
    """When > 8 evaluations exist, only the most recent 8 are evaluated."""
    # First 2 evaluations were terrible and had persistent seed failures, but are outside the 8-window
    evals = [
        _make_eval_payload(10, stage=4, success_rate=0.0, failed_seeds=[11, 23]),
        _make_eval_payload(20, stage=4, success_rate=0.0, failed_seeds=[11, 23]),
        # Recent 8 evaluations are all passing:
        _make_eval_payload(30, stage=4, success_rate=1.0),
        _make_eval_payload(40, stage=4, success_rate=1.0),
        _make_eval_payload(50, stage=4, success_rate=1.0),
        _make_eval_payload(60, stage=4, success_rate=1.0),
        _make_eval_payload(70, stage=4, success_rate=1.0),
        _make_eval_payload(80, stage=4, success_rate=1.0),
        _make_eval_payload(90, stage=4, success_rate=1.0),
        _make_eval_payload(100, stage=4, success_rate=1.0),
    ]
    _setup_eval_dir(tmp_path, evals)
    res = compute_promotion_gate_snapshot(tmp_path, target_ep=100)
    assert res["ready"] is True
    assert res["formal_evaluations_available"] == 8
    assert res["recent_average_success"] == 1.0
    assert res["persistent_seed_failure"] is False
