"""Comprehensive unit tests for the deterministic training diagnostics engine."""


from sheepdog.training.diagnostics.config import DeterministicDiagnosticsConfig
from sheepdog.training.diagnostics.engine import DeterministicDiagnosticsEngine
from sheepdog.training.diagnostics.seed_tracker import build_seed_matrix_report
from sheepdog.training.diagnostics.signatures import (
    classify_failure_signature,
    extract_failure_candidate_causes,
)


def _make_mock_record(
    seed: int,
    success: bool,
    steps: int = 300,
    sheep_penned: int = 5,
    total_sheep: int | None = None,
    corner_stuck: bool = False,
    final_zone: str = "center",
    min_distance_to_pen: float = 0.0,
    no_progress_steps: int = 0,
    timeout: bool = False,
    stopped: bool = False,
    stop_reason: str = "",
    gate_fail_steps: int = 0,
    wall_time_pct: float = 0.0,
    oscillation: bool = False,
    flock_spread: float = 2.0,
    role_switches: int = 2,
) -> dict:
    rec = {
        "seed": seed,
        "success": success,
        "steps": steps,
        "sheep_penned": sheep_penned,
        "corner_stuck_at_end": corner_stuck,
        "final_sheep_zone": final_zone,
        "initial_sheep_zone": final_zone,
        "min_sheep_distance_to_pen": min_distance_to_pen if not success else 0.0,
        "final_sheep_distance_to_pen": min_distance_to_pen if not success else 0.0,
        "no_progress_steps": no_progress_steps,
        "timeout": timeout or (not success and not stopped),
        "stopped": stopped,
        "stop_reason": stop_reason,
        "gate_corridor_failure_steps": gate_fail_steps,
        "wall_time_pct": wall_time_pct,
        "oscillation_detected": oscillation,
        "final_flock_spread": flock_spread,
        "role_switches": role_switches,
    }
    if total_sheep is not None:
        rec["total_sheep"] = total_sheep
    return rec


def _make_mock_evaluation(
    checkpoint_ep: int,
    stage: int,
    records: list[dict],
    eval_mode: str = "confidence",
) -> dict:
    succ_count = sum(1 for r in records if r.get("success"))
    total = len(records)
    rate = succ_count / total if total > 0 else 0.0
    return {
        "checkpoint_episode": checkpoint_ep,
        "curriculum_stage": stage,
        "evaluation_mode": eval_mode,
        "success_rate": rate,
        "records": records,
    }


def test_classify_failure_signatures_and_uncertainty():
    # 1. Success returns none
    rec_succ = _make_mock_record(101, success=True)
    assert classify_failure_signature(rec_succ) == "none"

    # 2. Insufficient telemetry
    assert classify_failure_signature({"seed": 101, "success": False}) == "insufficient_telemetry"

    # 3. Corner Entrapment
    rec_corner = _make_mock_record(102, success=False, corner_stuck=True, final_zone="top_left")
    assert classify_failure_signature(rec_corner) == "corner_entrapment_top_left"

    # 4. Gate mouth obstruction
    rec_gate = _make_mock_record(103, success=False, gate_fail_steps=25)
    assert classify_failure_signature(rec_gate) == "pen_mouth_obstruction"

    # 5. Wall stall
    rec_wall = _make_mock_record(104, success=False, wall_time_pct=0.55, final_zone="top_wall")
    assert classify_failure_signature(rec_wall) == "wall_stall_top_wall"

    # 6. Action Oscillation
    rec_osc = _make_mock_record(105, success=False, oscillation=True)
    assert classify_failure_signature(rec_osc) == "action_oscillation"

    # 7. Multiple candidate causes
    rec_multi = _make_mock_record(106, success=False, corner_stuck=True, final_zone="top_left", gate_fail_steps=30, oscillation=True)
    assert classify_failure_signature(rec_multi) == "multiple_candidate_causes"
    candidates = extract_failure_candidate_causes(rec_multi)
    assert len(candidates) >= 2
    assert "corner_entrapment_top_left" in candidates
    assert "pen_mouth_obstruction" in candidates

    # 8. Proximity-based Near vs Open timeout
    rec_near = _make_mock_record(107, success=False, timeout=True, min_distance_to_pen=2.0)
    assert classify_failure_signature(rec_near) == "near_pen_timeout"

    rec_open = _make_mock_record(108, success=False, timeout=True, min_distance_to_pen=25.0, sheep_penned=0)
    assert classify_failure_signature(rec_open) == "open_field_timeout"


def test_insufficient_history_safeguards():
    engine = DeterministicDiagnosticsEngine(
        DeterministicDiagnosticsConfig(window_checkpoints_max=5, min_unique_checkpoints_required=3, min_episodes_for_trend=15)
    )

    # Empty evals
    report_empty = engine.analyze_stage_window([], curriculum_stage=1)
    assert report_empty.success_trend == "insufficient_history"
    assert report_empty.step_efficiency_trend == "insufficient_history"
    assert report_empty.unique_checkpoint_count == 0
    assert report_empty.evaluation_run_count == 0
    assert report_empty.episode_sample_count == 0
    assert report_empty.data_adequacy_level == "insufficient"
    assert report_empty.requires_investigation is False

    # 2 checkpoints with 5 samples each = 10 samples (below min 15 and below 3 unique checkpoints)
    recs = [_make_mock_record(100 + s, success=True) for s in range(5)]
    evals = [_make_mock_evaluation(10, 1, recs), _make_mock_evaluation(20, 1, recs)]
    report_short = engine.analyze_stage_window(evals, curriculum_stage=1)
    assert report_short.success_trend == "insufficient_history"
    assert report_short.unique_checkpoint_count == 2
    assert report_short.episode_sample_count == 10


def test_quick_plus_confidence_duplicate_checkpoint_handling():
    # Simulate 4 unique training checkpoints (10, 20, 30, 40), each having both a quick and confidence run (8 runs total)
    evals = []
    for cp in [10, 20, 30, 40]:
        recs_quick = [_make_mock_record(100, success=False, corner_stuck=True, final_zone="top_left")]
        recs_conf = [_make_mock_record(100, success=False, corner_stuck=True, final_zone="top_left")]
        for s in range(1, 10):
            recs_quick.append(_make_mock_record(100 + s, success=True))
            recs_conf.append(_make_mock_record(100 + s, success=True))

        evals.append(_make_mock_evaluation(cp, stage=7, records=recs_quick, eval_mode="quick"))
        evals.append(_make_mock_evaluation(cp, stage=7, records=recs_conf, eval_mode="confidence"))

    engine = DeterministicDiagnosticsEngine()
    report = engine.analyze_stage_window(evals, curriculum_stage=7)

    assert report.unique_checkpoint_count == 4
    assert report.evaluation_run_count == 8
    assert report.episode_sample_count == 80
    assert report.unique_checkpoints == (10, 20, 30, 40)

    s100 = report.seed_matrix.seed_summaries[100]
    assert s100.unique_checkpoints_tested == 4
    assert s100.current_failure_streak == 4  # 4 unique checkpoints
    assert s100.failed_evaluation_runs == 8
    assert s100.total_evaluation_runs == 8
    assert s100.current_persistence_severity == "strong_persistence"
    assert s100.is_stable_signature_failure is True


def test_quick_vs_confidence_mixed_and_single_mode_semantics():
    evals = [
        # Checkpoint 10: Quick pass, Conf pass -> pass
        _make_mock_evaluation(10, 1, [_make_mock_record(101, True)], eval_mode="quick"),
        _make_mock_evaluation(10, 1, [_make_mock_record(101, True)], eval_mode="confidence"),
        # Checkpoint 20: Quick fail, Conf fail -> fail
        _make_mock_evaluation(20, 1, [_make_mock_record(101, False)], eval_mode="quick"),
        _make_mock_evaluation(20, 1, [_make_mock_record(101, False)], eval_mode="confidence"),
        # Checkpoint 30: Quick pass, Conf fail -> mixed
        _make_mock_evaluation(30, 1, [_make_mock_record(101, True)], eval_mode="quick"),
        _make_mock_evaluation(30, 1, [_make_mock_record(101, False)], eval_mode="confidence"),
        # Checkpoint 40: Only confidence present -> pass
        _make_mock_evaluation(40, 1, [_make_mock_record(101, True)], eval_mode="confidence"),
    ]

    report = build_seed_matrix_report(evals, 1)
    s101 = report.seed_summaries[101]
    history = {cpo.checkpoint_episode: cpo for cpo in s101.checkpoint_history}

    assert history[10].checkpoint_status == "pass"
    assert history[20].checkpoint_status == "fail"
    assert history[30].checkpoint_status == "mixed"
    assert history[40].checkpoint_status == "pass"
    assert s101.current_failure_streak == 0  # Passed at CP 40


def test_streak_severity_transitions():
    cfg = DeterministicDiagnosticsConfig(
        watch_checkpoint_streak_threshold=2,
        persistent_candidate_streak_threshold=3,
        strong_persistence_streak_threshold=4,
    )

    evals = [
        _make_mock_evaluation(10, 1, [_make_mock_record(101, False)]),
        _make_mock_evaluation(20, 1, [_make_mock_record(101, False)]),
        _make_mock_evaluation(30, 1, [_make_mock_record(101, False)]),
        _make_mock_evaluation(40, 1, [_make_mock_record(101, False)]),
    ]

    res_1 = build_seed_matrix_report(evals[:1], 1, config=cfg).seed_summaries[101]
    assert res_1.current_persistence_severity == "normal"

    res_2 = build_seed_matrix_report(evals[:2], 1, config=cfg).seed_summaries[101]
    assert res_2.current_persistence_severity == "watch"

    res_3 = build_seed_matrix_report(evals[:3], 1, config=cfg).seed_summaries[101]
    assert res_3.current_persistence_severity == "persistent_candidate"

    res_4 = build_seed_matrix_report(evals[:4], 1, config=cfg).seed_summaries[101]
    assert res_4.current_persistence_severity == "strong_persistence"


def test_repeated_seed_signature_stability():
    # Stable signature
    evals_stable = [
        _make_mock_evaluation(10, 1, [_make_mock_record(103, False, corner_stuck=True, final_zone="top_left")]),
        _make_mock_evaluation(20, 1, [_make_mock_record(103, False, corner_stuck=True, final_zone="top_left")]),
        _make_mock_evaluation(30, 1, [_make_mock_record(103, False, corner_stuck=True, final_zone="top_left")]),
    ]
    report_stable = build_seed_matrix_report(evals_stable, 1)
    s103_stable = report_stable.seed_summaries[103]
    assert s103_stable.is_stable_signature_failure is True
    assert s103_stable.consecutive_signature_checkpoint_streak == 3

    # Varying signatures
    evals_varying = [
        _make_mock_evaluation(10, 1, [_make_mock_record(104, False, corner_stuck=True, final_zone="top_left")]),
        _make_mock_evaluation(20, 1, [_make_mock_record(104, False, gate_fail_steps=30)]),
        _make_mock_evaluation(30, 1, [_make_mock_record(104, False, oscillation=True)]),
    ]
    report_varying = build_seed_matrix_report(evals_varying, 1)
    s104_varying = report_varying.seed_summaries[104]
    assert s104_varying.is_stable_signature_failure is False
    assert s104_varying.consecutive_signature_checkpoint_streak == 1


def test_current_vs_historical_persistence_and_recovery():
    # Seed 41 pattern: Fails checkpoints 10, 20, 30, 40 (streak 4), then passes at checkpoint 50
    evals = [
        _make_mock_evaluation(10, 7, [_make_mock_record(41, False)]),
        _make_mock_evaluation(20, 7, [_make_mock_record(41, False)]),
        _make_mock_evaluation(30, 7, [_make_mock_record(41, False)]),
        _make_mock_evaluation(40, 7, [_make_mock_record(41, False)]),
        _make_mock_evaluation(50, 7, [_make_mock_record(41, True)]),
    ]

    report = build_seed_matrix_report(evals, 7)
    s41 = report.seed_summaries[41]

    assert s41.current_failure_streak == 0
    assert s41.current_persistence_severity == "normal"
    assert s41.max_failure_streak == 4
    assert s41.historical_persistence_severity == "strong_persistence"
    assert s41.recently_recovered is True
    assert 41 in report.recently_recovered_seeds


def test_performance_band_vs_directional_trend():
    engine = DeterministicDiagnosticsEngine()

    # Case A: Near Mastery with Stable trend (90%, 90%, 90%, 90%)
    evals_near_mastery = []
    for i, cp in enumerate([10, 20, 30, 40]):
        fail_seed_idx = i
        recs = [_make_mock_record(100 + s, success=(s != fail_seed_idx)) for s in range(10)]
        evals_near_mastery.append(_make_mock_evaluation(cp, 4, recs))
    rep_a = engine.analyze_stage_window(evals_near_mastery, 4)
    assert rep_a.performance_band == "near_mastery"
    assert rep_a.success_trend == "stable"
    assert rep_a.requires_investigation is False

    # Case B: High Performance with Regressing trend (90%, 80%, 70%)
    evals_regr = []
    for cp, rate in [(10, 9), (20, 8), (30, 7)]:
        recs = [_make_mock_record(100 + s, success=(s < rate)) for s in range(10)]
        evals_regr.append(_make_mock_evaluation(cp, 5, recs))
    rep_b = engine.analyze_stage_window(evals_regr, 5)
    assert rep_b.performance_band == "high"
    assert rep_b.success_trend == "regressing"
    assert rep_b.requires_investigation is True


def test_improving_failures_while_headline_success_is_flat_with_deltas():
    engine = DeterministicDiagnosticsEngine(
        DeterministicDiagnosticsConfig(window_checkpoints_max=4, min_unique_checkpoints_required=3, min_episodes_for_trend=15)
    )
    evals = []
    # Success rate is flat at 50% across 3 checkpoints, but failure proximity is improving (6.0 -> 3.5 -> 1.5) and penned 2 -> 3 -> 4
    for i, (min_d, penned_val) in enumerate([(6.0, 2), (3.5, 3), (1.5, 4)]):
        records = []
        for s in range(10):
            is_succ = s < 5
            records.append(_make_mock_record(100 + s, success=is_succ, sheep_penned=penned_val if not is_succ else 5, total_sheep=5, min_distance_to_pen=min_d))
        evals.append(_make_mock_evaluation((i + 1) * 10, stage=9, records=records))

    report = engine.analyze_stage_window(evals, curriculum_stage=9)
    assert report.performance_band == "moderate"
    assert report.success_trend == "stable"
    assert report.failure_progress.failure_progress_trend == "improving"
    assert report.failure_progress.first_checkpoint_avg_penned == 2.0
    assert report.failure_progress.latest_checkpoint_avg_penned == 4.0
    assert report.failure_progress.penned_delta == 2.0
    assert report.failure_progress.first_checkpoint_avg_min_dist == 6.0
    assert report.failure_progress.latest_checkpoint_avg_min_dist == 1.5
    assert report.failure_progress.min_dist_delta == -4.5
    assert report.failure_progress.improving_proximity is True
    assert report.failure_progress.failure_progress_score >= 0.70


def test_data_integrity_validation_and_malformed_telemetry():
    engine = DeterministicDiagnosticsEngine()
    evals = []

    # In Stage 6 (3 sheep total): records with 2 penned should result in ratio 2/3 = 0.667 (NOT 175%)
    # Also test an impossible record with 10 sheep penned and a NaN distance
    recs = [
        _make_mock_record(101, False, sheep_penned=2, min_distance_to_pen=10.0),
        _make_mock_record(102, False, sheep_penned=10, min_distance_to_pen=float("nan")),
    ]
    for s in range(8):
        recs.append(_make_mock_record(103 + s, True))

    for cp in [10, 20, 30]:
        evals.append(_make_mock_evaluation(cp, stage=6, records=recs))

    report = engine.analyze_stage_window(evals, curriculum_stage=6)

    # Validate that penned ratio is strictly <= 1.0
    assert report.failure_progress.penned_ratio_at_failure <= 1.0
    # Clamping and warnings detected
    assert len(report.failure_progress.data_quality_warnings) > 0
    assert any("exceeded total_sheep" in w for w in report.failure_progress.data_quality_warnings)


def test_spatial_corner_disparity_detection():
    engine = DeterministicDiagnosticsEngine()
    evals = []
    for i in range(3):
        records = [
            _make_mock_record(101, False, corner_stuck=True, final_zone="top_left"),
            _make_mock_record(102, False, corner_stuck=True, final_zone="top_left"),
        ]
        for s in range(8):
            records.append(_make_mock_record(200 + s, True, final_zone="center"))
        evals.append(_make_mock_evaluation((i + 1) * 10, stage=7, records=records))

    report = engine.analyze_stage_window(evals, curriculum_stage=7)
    assert report.spatial_bottlenecks["critical_bottleneck"] is True
    assert "Severe corner entrapment gap" in str(report.spatial_bottlenecks["bottleneck_reason"])
    assert any(f.finding_type == "spatial_disparity" for f in report.findings)


def test_structured_findings_generation():
    engine = DeterministicDiagnosticsEngine()
    evals = []
    for i in range(4):
        records = []
        for s in range(10):
            seed_id = 100 + s
            if seed_id in (103, 104):
                records.append(_make_mock_record(seed_id, False, corner_stuck=True, final_zone="top_left"))
            else:
                records.append(_make_mock_record(seed_id, True))
        evals.append(_make_mock_evaluation((i + 1) * 10, stage=7, records=records))

    report = engine.analyze_stage_window(evals, curriculum_stage=7)
    assert len(report.findings) >= 2
    types = {f.finding_type for f in report.findings}
    assert "dominant_failure_mode" in types
    assert "persistent_seed_failure" in types

    dom_finding = next(f for f in report.findings if f.finding_type == "dominant_failure_mode")
    assert dom_finding.target == "corner_entrapment_top_left"
    assert dom_finding.support.affected_failures == 8
    assert dom_finding.support.total_failures == 8
    assert dom_finding.evidence_level == "strong"


def test_data_adequacy_score_and_levels():
    engine = DeterministicDiagnosticsEngine()

    # Case A: 8 unique checkpoints, 16 runs, 160 samples -> strong
    evals_strong = []
    for cp in range(8):
        recs = [_make_mock_record(100 + s, True) for s in range(10)]
        evals_strong.append(_make_mock_evaluation(cp * 10, 4, recs, eval_mode="quick"))
        evals_strong.append(_make_mock_evaluation(cp * 10, 4, recs, eval_mode="confidence"))

    rep_strong = engine.analyze_stage_window(evals_strong, 4)
    assert rep_strong.data_adequacy_score == 1.0
    assert rep_strong.data_adequacy_level == "strong"


def test_to_dict_serialization():
    engine = DeterministicDiagnosticsEngine(
        DeterministicDiagnosticsConfig(window_checkpoints_max=3, min_unique_checkpoints_required=2, min_episodes_for_trend=4)
    )
    records = [_make_mock_record(101, True), _make_mock_record(102, False, corner_stuck=True)]
    evals = [
        _make_mock_evaluation(10, 1, records),
        _make_mock_evaluation(20, 1, records),
    ]
    report = engine.analyze_stage_window(evals, curriculum_stage=1)
    as_dict = report.to_dict()

    assert isinstance(as_dict, dict)
    assert as_dict["stage"] == 1
    assert "unique_checkpoint_count" in as_dict
    assert "evaluation_run_count" in as_dict
    assert "episode_sample_count" in as_dict
    assert "performance_band" in as_dict
    assert "success_trend" in as_dict
    assert "data_adequacy_score" in as_dict
    assert "data_adequacy_level" in as_dict
