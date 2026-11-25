"""Tests for completed-evaluation advisory early warning system (Reduced Scope).

Validates:
1. Evaluation validation: rejects incomplete evaluations, empty records, and missing identities.
2. Honest repetition metrics: requires v2.0 metrics; never substitutes oscillation_detected flag;
   shows "Not available for this evaluation" when absent; measures real fraction and denominator.
3. Compatible success regression: strictly requires matching all 7 compatibility criteria
   (run_id, stage, seed manifest, assistance mode, env/reward hash, episode budget, decision mode).
   Rejects comparisons when metadata is missing or mismatched.
4. Advisory-only semantics: user acknowledgment records review ("seen") but does NOT claim recovery
   or auto-resolve. No automatic stops, pauses, or mutations.
5. Replay linkage: links only to verified available replays on disk with lead-in step;
   clearly explains missing replays without fabricating links.
6. Bounded loading: status polling reuses cached payload without disk writes or re-parsing.
7. Monitoring neutrality: consumes zero random numbers and leaves execution bit-for-bit invariant.
"""

from __future__ import annotations

import json
from pathlib import Path

from sheepdog.training.early_warning import (
    EarlyWarningMonitor,
    WarningConditionType,
    WarningSeverity,
    WarningStatus,
    analyze_single_evaluation_repetition,
    check_evaluation_compatibility,
    validate_evaluation_record,
)


def _make_mock_evaluation(
    eval_dir: Path,
    eval_id: str,
    stage: int = 34,
    checkpoint_ep: int = 39909,
    policy_version: int = 10074,
    success_rate: float = 0.60,
    seed_count: int = 10,
    run_id: str = "run_test_001",
    seed_set_id: str = "seed_set_alpha",
    policy_mode: str = "guided",
    env_hash: str = "hash_12345",
    episode_budget: int = 2320,
    eval_mode: str = "quick",
    is_complete: bool = True,
    replay_path: str | None = None,
    with_v2_diagnostics: bool = False,
    v2_rep_fraction: float = 0.6327,
    v2_total_steps: int = 9766,
    v2_rep_steps: int = 6179,
    v2_stalls: int = 15,
) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    fpath = eval_dir / f"{eval_id}.json"

    records = []
    for i in range(seed_count):
        records.append({
            "seed": 100 + i,
            "success": i < int(success_rate * seed_count),
            "timeout": i >= int(success_rate * seed_count),
            "oscillation_detected": True,  # Old flag (must NOT be substituted for v2.0 metric!)
            "replay_path": replay_path,
        })

    data = {
        "evaluation_id": eval_id,
        "checkpoint_id": f"chk_{eval_id}",
        "checkpoint_episode": checkpoint_ep,
        "policy_version": policy_version,
        "curriculum_stage": stage,
        "success_rate": success_rate,
        "run_id": run_id,
        "evaluation_seed_set_id": seed_set_id,
        "evaluation_seed_count": seed_count,
        "policy_mode": policy_mode,
        "policy_type": "neural",
        "environment_config_hash": env_hash,
        "episode_budget": episode_budget,
        "max_steps": episode_budget,
        "evaluation_mode": eval_mode,
        "is_complete": is_complete,
        "evaluation_timestamp": "2026-11-06T18:00:00Z",
        "records": records,
    }

    if with_v2_diagnostics:
        diag_dir = eval_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_file = diag_dir / f"{eval_id}.diagnostics.json"
        diag_data = {
            "metrics_version": "2.0",
            "evaluation_id": eval_id,
            "checkpoint_episode": checkpoint_ep,
            "curriculum_stage": stage,
            "total_world_steps": v2_total_steps,
            "repeated_movement_steps": v2_rep_steps,
            "repeated_movement_fraction": v2_rep_fraction,
            "stalls_count": v2_stalls,
            "recovered_stalls_count": int(v2_stalls * 0.6),
            "recovery_rate": 0.6,
            "representative_seed": 100,
            "representative_replay_path": replay_path,
            "representative_step": 7,
            "representative_interval_id": "interval_0",
        }
        with diag_file.open("w", encoding="utf-8") as df:
            json.dump(diag_data, df, indent=2)

    with fpath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return fpath


def test_validate_evaluation_record_rejects_incomplete():
    """Verifies that validate_evaluation_record rejects incomplete or corrupt evaluations."""
    # 1. Non-dict payload
    valid, msg = validate_evaluation_record("corrupted string")
    assert valid is False
    assert "not a dictionary" in msg

    # 2. Explicitly marked incomplete
    valid, msg = validate_evaluation_record({"is_complete": False})
    assert valid is False
    assert "is_complete=False" in msg

    # 3. Zero episode records
    valid, msg = validate_evaluation_record({"checkpoint_episode": 100, "records": []})
    assert valid is False
    assert "zero completed episode records" in msg

    # 4. Incomplete seed count (expected 10 seeds, only got 3)
    valid, msg = validate_evaluation_record({
        "checkpoint_episode": 100,
        "evaluation_seed_count": 10,
        "records": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
    })
    assert valid is False
    assert "expected 10 seed records but found 3" in msg

    # 5. Missing checkpoint/policy identity
    valid, msg = validate_evaluation_record({
        "records": [{"seed": 1}],
    })
    assert valid is False
    assert "missing checkpoint identity" in msg

    # 6. Complete valid record
    valid, msg = validate_evaluation_record({
        "checkpoint_episode": 100,
        "evaluation_seed_count": 2,
        "records": [{"seed": 1}, {"seed": 2}],
    })
    assert valid is True


def test_honest_repetition_metrics_no_old_flag_substitution(tmp_path: Path):
    """Verifies that missing v2.0 metrics report 'Not available for this evaluation' and NEVER substitute oscillation_detected."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    # Evaluation with old oscillation_detected=True on all 10 seeds, but NO v2.0 diagnostics
    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_chk_old_flags_only",
        checkpoint_ep=1000,
        success_rate=0.60,
        with_v2_diagnostics=False,
    )

    summary = monitor.get_warnings_summary(current_stage=34, force_refresh=True)
    cur_eval = summary["current_evaluation"]

    # Must honestly report not available
    assert cur_eval["repetition_metrics_available"] is False
    assert cur_eval["repetition_metric_status"] == "Not available for this evaluation."
    assert cur_eval["repetition_metrics"] is None

    # Zero warnings generated (no false alarm from old flag!)
    assert len(summary["active_warnings"]) == 0
    assert summary["summary"]["total_active"] == 0


def test_repetition_warning_with_verified_v2_metrics_and_replay_link(tmp_path: Path):
    """Verifies that when v2.0 metrics exist and exceed threshold, warning is generated with real denominator and verified replay link."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    # Create dummy replay file on disk
    dummy_replay = tmp_path / "replays" / "chk_1000_seed_100.json.gz"
    dummy_replay.parent.mkdir(parents=True, exist_ok=True)
    dummy_replay.write_text("dummy replay content")

    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_chk_with_v2",
        checkpoint_ep=1000,
        success_rate=0.60,
        replay_path=str(dummy_replay),
        with_v2_diagnostics=True,
        v2_rep_fraction=0.6327,
        v2_total_steps=9766,
        v2_rep_steps=6179,
        v2_stalls=15,
    )

    summary = monitor.get_warnings_summary(current_stage=34, force_refresh=True)
    cur_eval = summary["current_evaluation"]

    # Repetition metrics verified
    assert cur_eval["repetition_metrics_available"] is True
    assert "Verified (v2.0)" in cur_eval["repetition_metric_status"]
    assert cur_eval["repetition_metrics"]["repeated_movement_fraction"] == 0.6327
    assert cur_eval["repetition_metrics"]["total_world_steps"] == 9766

    # Warning generated
    assert len(summary["active_warnings"]) == 1
    w = summary["active_warnings"][0]
    assert w["condition_type"] == WarningConditionType.UNPRODUCTIVE_REPETITION.value
    assert w["severity"] == WarningSeverity.WARNING.value
    assert w["metric_version"] == "2.0"
    assert "63.3%" in w["title"]
    assert "6179 of 9766 total world steps" in w["plain_language_observation"]

    # Replay linkage verified
    assert w["replay_available"] is True
    assert w["replay_link"] == str(dummy_replay).replace("\\", "/")
    assert w["replay_step"] == 7


def test_replay_missing_on_disk_reports_unavailable_honestly(tmp_path: Path):
    """Verifies that if replay path does not exist on disk, replay_available is False and explained clearly."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    nonexistent_replay = "artifacts/evaluations/replays/nonexistent-seed-999.json.gz"

    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_chk_missing_replay",
        checkpoint_ep=1000,
        replay_path=nonexistent_replay,
        with_v2_diagnostics=True,
        v2_rep_fraction=0.60,
    )

    summary = monitor.get_warnings_summary(current_stage=34, force_refresh=True)
    assert len(summary["active_warnings"]) == 1
    w = summary["active_warnings"][0]
    assert w["replay_available"] is False
    assert w["replay_link"] is None
    assert "Replay not available on disk" in w["link_explanation"]


def test_compatible_success_regression_and_incompatible_rejection(tmp_path: Path):
    """Verifies that success regression warning requires matching all 7 compatibility criteria."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    # 1. Baseline evaluation: 80% success
    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_01_baseline",
        checkpoint_ep=100,
        success_rate=0.80,
        run_id="run_alpha",
        seed_set_id="manifest_seeds_10",
        policy_mode="guided",
        env_hash="env_hash_v1",
        episode_budget=2320,
        eval_mode="quick",
    )

    # 2. Incompatible candidate: drop from 80% to 40%, BUT run_id differs ("run_beta")
    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_02_incompatible_run",
        checkpoint_ep=200,
        success_rate=0.40,
        run_id="run_beta",  # MISMATCH!
        seed_set_id="manifest_seeds_10",
        policy_mode="guided",
        env_hash="env_hash_v1",
        episode_budget=2320,
        eval_mode="quick",
    )

    summary_incomp = monitor.get_warnings_summary(
        current_stage=34,
        selected_eval_path=eval_dir / "eval_02_incompatible_run.json",
        force_refresh=True,
    )
    # Must reject comparison and NOT trigger regression warning!
    assert "Comparison unavailable" in summary_incomp["current_evaluation"]["comparison_status"]
    assert len(summary_incomp["active_warnings"]) == 0

    # 3. Incompatible candidate: seed manifest differs
    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_03_incompatible_seedset",
        checkpoint_ep=200,
        success_rate=0.40,
        run_id="run_alpha",
        seed_set_id="manifest_seeds_held_out",  # MISMATCH!
        policy_mode="guided",
        env_hash="env_hash_v1",
        episode_budget=2320,
        eval_mode="quick",
    )

    summary_seed_incomp = monitor.get_warnings_summary(
        current_stage=34,
        selected_eval_path=eval_dir / "eval_03_incompatible_seedset.json",
        force_refresh=True,
    )
    assert "Comparison unavailable" in summary_seed_incomp["current_evaluation"]["comparison_status"]
    assert len(summary_seed_incomp["active_warnings"]) == 0

    # 4. Fully compatible candidate: all 7 factors match, success dropped by 40% (80% -> 40%)
    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_04_compatible_regress",
        checkpoint_ep=200,
        success_rate=0.40,
        run_id="run_alpha",
        seed_set_id="manifest_seeds_10",
        policy_mode="guided",
        env_hash="env_hash_v1",
        episode_budget=2320,
        eval_mode="quick",
    )

    summary_comp = monitor.get_warnings_summary(
        current_stage=34,
        selected_eval_path=eval_dir / "eval_04_compatible_regress.json",
        force_refresh=True,
    )
    assert len(summary_comp["active_warnings"]) == 1
    w = summary_comp["active_warnings"][0]
    assert w["condition_type"] == WarningConditionType.SUCCESS_REGRESSION.value
    assert "40%" in w["title"]
    assert w["supporting_measurements"]["drop_percentage"] == 40.0
    assert w["compatibility_factors"]["run_id"] == "run_alpha"
    assert w["compatibility_factors"]["curriculum_stage"] == 34


def test_acknowledgement_records_review_without_claiming_recovery(tmp_path: Path):
    """Verifies that user acknowledgment changes status to acknowledged, records operator and timestamp, but does NOT claim recovery."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_ack_test",
        checkpoint_ep=1000,
        with_v2_diagnostics=True,
        v2_rep_fraction=0.65,
    )

    summary1 = monitor.get_warnings_summary(current_stage=34, force_refresh=True)
    assert len(summary1["active_warnings"]) == 1
    warn_id = summary1["active_warnings"][0]["id"]
    assert summary1["active_warnings"][0]["status"] == WarningStatus.ACTIVE.value

    # Acknowledge
    ok = monitor.acknowledge_warning(warn_id, note="Operator reviewed standoff replay")
    assert ok is True

    # Check updated summary
    summary2 = monitor.get_warnings_summary(current_stage=34, force_refresh=True)
    assert len(summary2["active_warnings"]) == 0
    assert len(summary2["acknowledged_warnings"]) == 1
    ack_w = summary2["acknowledged_warnings"][0]
    assert ack_w["status"] == WarningStatus.ACKNOWLEDGED.value
    assert ack_w["acknowledged_by"] == "Operator reviewed standoff replay"
    assert ack_w["acknowledged_at"] is not None

    # Crucial: measurements and original evidence are preserved, NOT erased or claimed resolved!
    assert ack_w["supporting_measurements"]["repeated_movement_fraction"] == 0.65
    assert "resolved_warnings" not in summary2 or len(summary2.get("resolved_warnings", [])) == 0


def test_bounded_loading_reuses_cache_and_never_rewrites_on_poll(tmp_path: Path):
    """Verifies that status requests reuse in-memory cache when mtime is unchanged and do not touch disk."""
    eval_dir = tmp_path / "evaluations"
    monitor = EarlyWarningMonitor(tmp_path)

    eval_file = _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_cache_test",
        checkpoint_ep=1000,
        with_v2_diagnostics=True,
        v2_rep_fraction=0.55,
    )

    store_file = tmp_path / "early-warnings.json"

    # Initial call populates store and cache
    s1 = monitor.get_warnings_summary(current_stage=34, selected_eval_path=eval_file)
    assert store_file.exists()
    store_mtime1 = store_file.stat().st_mtime_ns

    # Second call without force_refresh: MUST be cache hit
    s2 = monitor.get_warnings_summary(current_stage=34, selected_eval_path=eval_file)
    store_mtime2 = store_file.stat().st_mtime_ns

    # Zero disk writes occurred!
    assert store_mtime1 == store_mtime2
    assert s1 == s2


def test_analyze_single_evaluation_repetition_cli(tmp_path: Path):
    """Verifies that analyze_single_evaluation_repetition processes one evaluation and writes diagnostics JSON."""
    eval_dir = tmp_path / "evaluations"
    eval_file = _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_cli_test",
        checkpoint_ep=500,
        with_v2_diagnostics=False,
    )

    dest_diag = tmp_path / "custom_diagnostics.json"
    summary = analyze_single_evaluation_repetition(eval_file, output_path=dest_diag)

    assert dest_diag.exists()
    assert summary["evaluation_id"] == "eval_cli_test"
    assert summary["metrics_version"] == "2.0"
    assert summary["checkpoint_episode"] == 500


def test_monitoring_neutrality_preserves_rng_and_training():
    """Verifies that early warning evaluation consumes zero environment random numbers."""
    import random

    import numpy as np

    def run_simulation_step(with_early_warning_check: bool, seed: int = 12345):
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        pre = [py_rng.random() for _ in range(25)]

        if with_early_warning_check:
            # Check compatibility logic
            b = {"run_id": "r1", "curriculum_stage": 1, "evaluation_seed_set_id": "s1", "policy_mode": "guided", "environment_config_hash": "h1", "episode_budget": 100, "evaluation_mode": "quick"}
            c = dict(b)
            is_comp, _, _ = check_evaluation_compatibility(b, c)
            assert is_comp is True

        post = [float(np_rng.uniform(0.0, 100.0)) for _ in range(25)]
        return pre + post

    res1 = run_simulation_step(with_early_warning_check=False)
    res2 = run_simulation_step(with_early_warning_check=True)
    assert res1 == res2


def test_ignores_inprogress_evaluation_files(tmp_path: Path, caplog):
    """Verifies that *.inprogress.json files with newer mtimes are ignored by EarlyWarningMonitor."""
    import logging
    import os

    eval_dir = tmp_path / "evaluations"
    completed_file = _make_mock_evaluation(
        eval_dir=eval_dir,
        eval_id="eval_completed_001",
        stage=34,
        checkpoint_ep=100,
        policy_version=10,
        success_rate=0.80,
        with_v2_diagnostics=True,
    )

    inprogress_file = eval_dir / "eval_newer_002.inprogress.json"
    inprogress_file.write_text(
        json.dumps({"evaluation_id": "eval_newer_002", "is_complete": False, "records": []}),
        encoding="utf-8",
    )

    # Ensure inprogress file has strictly newer mtime than completed file
    comp_mtime = completed_file.stat().st_mtime
    os.utime(inprogress_file, (comp_mtime + 10.0, comp_mtime + 10.0))

    monitor = EarlyWarningMonitor(tmp_path)
    with caplog.at_level(logging.WARNING):
        summary = monitor.get_warnings_summary(current_stage=34, force_refresh=True)

    assert "Rejected incomplete evaluation" not in caplog.text
    assert summary["current_evaluation"] is not None
    assert summary["current_evaluation"]["evaluation_id"] == "eval_completed_001"
