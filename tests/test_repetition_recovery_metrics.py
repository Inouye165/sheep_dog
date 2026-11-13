"""Unit tests for the production repetition and recovery metrics engine (v2.0).

Tests:
1. Stationary holding vs active oscillation (stationary dog never counts as oscillating).
2. Productive waiting: one dog waits while teammates make progress -> not unproductive repetition.
3. Multi-step loops: period-2 (A-B-A-B), period-3 (A-B-C-A-B-C), period-4 (A-B-C-D-A-B-C-D).
4. Productive repositioning / retreat with flock collection allowed (not penalized).
5. Sustained recovery into lasting progress vs immediate relapse into recurrence.
6. Eventual episode success does NOT credit earlier stalls that relapsed or timed out.
7. Empty-set / missing evidence produces explicit None (never 0.0).
"""

from __future__ import annotations

from typing import Any

from sheepdog.training.diagnostics.repetition_recovery import (
    METRICS_VERSION,
    DogMovementType,
    StallOutcome,
    analyze_repetition_and_recovery,
    classify_dog_movement_pattern,
)


def _make_mock_frame(
    step: int,
    dog_positions: list[tuple[int, int]],
    sheep_positions: list[tuple[int, int]],
    penned_flags: list[bool],
    avg_route_dist: float = 30.0,
    flock_spread: float = 5.0,
) -> dict[str, Any]:
    dogs = [{"x": p[0], "y": p[1], "index": i, "current_role": "chaser"} for i, p in enumerate(dog_positions)]
    sheep = [
        {"x": p[0], "y": p[1], "index": i, "penned": penned_flags[i], "personality": "obedient"}
        for i, p in enumerate(sheep_positions)
    ]
    return {
        "step": step,
        "average_distance_to_pen": avg_route_dist,
        "flock_spread": flock_spread,
        "penned_count": sum(1 for p in penned_flags if p),
        "unpenned_count": sum(1 for p in penned_flags if not p),
        "snapshot": {
            "dogs": dogs,
            "sheep": sheep,
            "average_distance_to_pen": avg_route_dist,
            "flock_spread": flock_spread,
        },
        "reward_dict": {"progress_to_pen": 0.0},
    }


def test_stationary_dog_is_not_classified_as_active_repetition():
    """A stationary dog holding position must not be classified as active repetition."""
    stationary_hist = [(10, 10)] * 8
    # Even though len(set(last_4)) == 1 <= 2, this is stationary holding, NOT active repetition
    pat = classify_dog_movement_pattern(stationary_hist, team_flock_progress=0.0)
    assert pat == DogMovementType.STATIONARY_HOLDING
    assert pat != DogMovementType.ACTIVE_REPETITION


def test_productive_waiting_classified_when_team_advances():
    """A waiting dog is classified as productive_waiting when teammates make progress."""
    waiting_hist = [(15, 20)] * 6
    pat = classify_dog_movement_pattern(waiting_hist, team_flock_progress=1.5)
    assert pat == DogMovementType.PRODUCTIVE_WAITING

    pat_pen = classify_dog_movement_pattern(waiting_hist, team_penned_sheep=True)
    assert pat_pen == DogMovementType.PRODUCTIVE_WAITING


def test_multi_step_periodic_loops_detected():
    """Period 2, 3, and 4 periodic loops are classified as active repetition without team progress."""
    # Period 2: A-B-A-B
    p2 = [(10, 10), (10, 11), (10, 10), (10, 11)]
    assert classify_dog_movement_pattern(p2) == DogMovementType.ACTIVE_REPETITION

    # Period 3: A-B-C-A-B-C
    p3 = [(5, 5), (6, 5), (7, 5), (5, 5), (6, 5), (7, 5)]
    assert classify_dog_movement_pattern(p3) == DogMovementType.ACTIVE_REPETITION

    # Period 4: A-B-C-D-A-B-C-D
    p4 = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1)]
    assert classify_dog_movement_pattern(p4) == DogMovementType.ACTIVE_REPETITION


def test_temporary_jitter_or_exploring_is_not_active_repetition():
    """Transient directional shifts or forward exploration are not active repetition."""
    linear = [(10, 10), (11, 10), (12, 10), (13, 10)]
    assert classify_dog_movement_pattern(linear) == DogMovementType.EXPLORING_OR_ADVANCING

    # 3 unique points in 4 steps (not periodic cycle)
    jitter = [(10, 10), (10, 11), (10, 10), (10, 12)]
    assert classify_dog_movement_pattern(jitter) == DogMovementType.EXPLORING_OR_ADVANCING


def test_productive_repositioning_does_not_count_as_unproductive_repetition():
    """When dog movements reduce route distance or pen sheep, it is not unproductive repetition."""
    # 20 frames where D0 cycles A-B but flock distance drops by 0.5 every step
    frames = []
    for s in range(20):
        d0_pos = (10, 10) if s % 2 == 0 else (10, 11)
        d1_pos = (20 + s, 20)
        d2_pos = (30, 30)
        frames.append(
            _make_mock_frame(
                step=s,
                dog_positions=[d0_pos, d1_pos, d2_pos],
                sheep_positions=[(50 - s, 50)] * 6,
                penned_flags=[False] * 6,
                avg_route_dist=50.0 - (s * 1.5),  # flock advancing rapidly
            )
        )

    bundle = {
        "frames": frames,
        "stats": {"success": False, "steps": 20},
        "environment": {"width": 100, "height": 100, "dogs": 3, "sheep": 6},
        "initial_snapshot": frames[0]["snapshot"],
        "final_snapshot": frames[-1]["snapshot"],
        "strategy_version": "v1.4",
        "discount_timebase": "sequential_shared_dog",
        "discount_factor": 0.99,
    }

    res = analyze_repetition_and_recovery(bundle, dog_count=3)
    assert res.metrics_version == METRICS_VERSION
    # Because flock was advancing, unproductive repetition steps should be 0
    assert res.repeated_movement_steps == 0
    assert res.repeated_movement_fraction == 0.0


def test_true_oscillation_without_progress_measured_accurately():
    """True stationary oscillation without flock progress is measured as repeated movement."""
    frames = []
    for s in range(30):
        d0_pos = (10, 10) if s % 2 == 0 else (10, 11)
        d1_pos = (20, 20) if s % 2 == 0 else (20, 21)
        d2_pos = (5, 5)  # stationary
        frames.append(
            _make_mock_frame(
                step=s,
                dog_positions=[d0_pos, d1_pos, d2_pos],
                sheep_positions=[(40, 40)] * 6,
                penned_flags=[False] * 6,
                avg_route_dist=40.0,  # Zero flock progress
            )
        )

    bundle = {
        "frames": frames,
        "stats": {"success": False, "steps": 30},
        "environment": {"width": 100, "height": 100, "dogs": 3, "sheep": 6},
        "initial_snapshot": frames[0]["snapshot"],
        "final_snapshot": frames[-1]["snapshot"],
        "strategy_version": "v1.4",
        "discount_timebase": "sequential_shared_dog",
        "discount_factor": 0.99,
    }

    res = analyze_repetition_and_recovery(bundle, dog_count=3)
    assert res.repeated_movement_steps >= 20
    assert res.repeated_movement_fraction is not None and res.repeated_movement_fraction >= 0.70


def test_sustained_recovery_vs_relapse_vs_unresolved():
    """Tests distinguishing sustained recovery from relapse and unresolved stalls."""
    from unittest.mock import MagicMock, patch

    def _make_mock_iv(iv_id, start, end, dur, unpenned, cls_name, penned_delta, net_prog):
        iv = MagicMock()
        iv.interval_id = iv_id
        iv.start_step = start
        iv.end_step = end
        iv.duration_steps = dur
        iv.unpenned_sheep_count = unpenned
        iv.classification = cls_name
        iv.aftermath = MagicMock()
        iv.aftermath.sheep_penned_delta = penned_delta
        iv.aftermath.net_route_progress = net_prog
        iv.aftermath.flock_spread_start = 5.0
        iv.aftermath.flock_spread_end = 5.0
        return iv

    # 1. Recovered stall: aftermath has sheep_penned_delta = 1, no subsequent stall
    iv_rec = _make_mock_iv("stall_1", 10, 30, 20, 6, "oscillation_loop", penned_delta=1, net_prog=6.0)

    # 2. Relapsed stall: aftermath route progress positive, but second stall starts at step 90 (within aftermath)
    iv_relapse_1 = _make_mock_iv("stall_2", 60, 80, 20, 5, "counterproductive_pressure", penned_delta=0, net_prog=2.0)
    iv_relapse_recurrent = _make_mock_iv("stall_3", 90, 110, 20, 5, "oscillation_loop", penned_delta=0, net_prog=0.0)

    frames = [_make_mock_frame(s, [(10, 10), (20, 20), (30, 30)], [(50, 50)] * 6, [False] * 6) for s in range(120)]
    bundle = {
        "frames": frames,
        "stats": {"success": False, "steps": 120},
        "environment": {"width": 100, "height": 100, "dogs": 3, "sheep": 6},
        "initial_snapshot": frames[0]["snapshot"],
        "final_snapshot": frames[-1]["snapshot"],
        "strategy_version": "v1.4",
        "discount_timebase": "sequential_shared_dog",
        "discount_factor": 0.99,
    }

    mock_analysis = MagicMock()
    mock_analysis.intervals = (iv_rec, iv_relapse_1, iv_relapse_recurrent)
    mock_analysis.total_steps = 120

    with patch(
        "sheepdog.training.diagnostics.repetition_recovery.analyze_episode_temporal_interactions",
        return_value=mock_analysis,
    ):
        res = analyze_repetition_and_recovery(bundle, dog_count=3)
        assert res.stalls_count == 3
        assert res.recovered_stalls_count == 1
        assert res.relapsed_stalls_count == 1
        assert res.unresolved_stalls_count == 1
        assert res.recovery_rate == round(1 / 3, 4)
        assert res.relapse_rate == round(1 / 3, 4)
        assert res.unresolved_rate == round(1 / 3, 4)

        # Check stall outcome records
        rec_outcomes = {r.interval_id: r.outcome for r in res.stall_records}
        assert rec_outcomes["stall_1"] == StallOutcome.RECOVERED
        assert rec_outcomes["stall_2"] == StallOutcome.RELAPSED
        assert rec_outcomes["stall_3"] == StallOutcome.UNRESOLVED


def test_eventual_episode_success_does_not_manufacture_recovery():
    """Eventual success at step 1000 does NOT mark an unresolved stall at step 50 as recovered."""
    from unittest.mock import MagicMock, patch

    iv_unres = MagicMock()
    iv_unres.interval_id = "early_stall"
    iv_unres.start_step = 30
    iv_unres.end_step = 60
    iv_unres.duration_steps = 30
    iv_unres.unpenned_sheep_count = 6
    iv_unres.classification = "oscillation_loop"
    iv_unres.aftermath = MagicMock()
    iv_unres.aftermath.sheep_penned_delta = 0
    iv_unres.aftermath.net_route_progress = 0.0
    iv_unres.aftermath.flock_spread_start = 5.0
    iv_unres.aftermath.flock_spread_end = 5.0

    frames = [_make_mock_frame(s, [(10, 10), (20, 20), (30, 30)], [(50, 50)] * 6, [False] * 6) for s in range(500)]
    bundle = {
        "frames": frames,
        "stats": {"success": True, "steps": 500},  # Episode succeeded at step 500!
        "environment": {"width": 100, "height": 100, "dogs": 3, "sheep": 6},
        "initial_snapshot": frames[0]["snapshot"],
        "final_snapshot": frames[-1]["snapshot"],
        "strategy_version": "v1.4",
        "discount_timebase": "sequential_shared_dog",
        "discount_factor": 0.99,
    }

    mock_analysis = MagicMock()
    mock_analysis.intervals = (iv_unres,)
    mock_analysis.total_steps = 500

    with patch(
        "sheepdog.training.diagnostics.repetition_recovery.analyze_episode_temporal_interactions",
        return_value=mock_analysis,
    ):
        res = analyze_repetition_and_recovery(bundle, dog_count=3)
        assert res.eventual_episode_success is True
        # The early stall was UNRESOLVED at its aftermath; must NOT be credited as recovered!
        assert res.stalls_count == 1
        assert res.recovered_stalls_count == 0
        assert res.unresolved_stalls_count == 1
        assert res.recovery_rate == 0.0
        assert res.stall_records[0].outcome == StallOutcome.UNRESOLVED
        assert res.stall_records[0].eventual_episode_success is True


def test_missing_stalls_yields_none_recovery_rate():
    """When an episode has 0 stalls, recovery_rate and relapse_rate are None (not 0.0)."""
    frames = [_make_mock_frame(s, [(s, s), (s + 5, s + 5), (s + 10, s + 10)], [(60 - s, 60)] * 6, [True] * 6) for s in range(10)]
    bundle = {
        "frames": frames,
        "stats": {"success": True, "steps": 10},
        "environment": {"width": 100, "height": 100, "dogs": 3, "sheep": 6},
        "initial_snapshot": frames[0]["snapshot"],
        "final_snapshot": frames[-1]["snapshot"],
        "strategy_version": "v1.4",
        "discount_timebase": "sequential_shared_dog",
        "discount_factor": 0.99,
    }
    from unittest.mock import MagicMock, patch
    mock_analysis = MagicMock()
    mock_analysis.intervals = ()
    mock_analysis.total_steps = 10

    with patch(
        "sheepdog.training.diagnostics.repetition_recovery.analyze_episode_temporal_interactions",
        return_value=mock_analysis,
    ):
        res = analyze_repetition_and_recovery(bundle, dog_count=3)
        assert res.stalls_count == 0
        assert res.recovery_rate is None
        assert res.relapse_rate is None
        assert res.unresolved_rate is None
