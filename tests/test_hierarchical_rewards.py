"""Tests for HierarchicalRewardComputer and HierarchicalRewardConfig."""

from __future__ import annotations

from sheepdog.rewards import (
    HierarchicalRewardBreakdown,
    HierarchicalRewardComputer,
    HierarchicalRewardConfig,
    HierarchicalRewardInputs,
)

PEN_X = 40.0
PEN_Y = 30.0


def _default_inputs(**overrides) -> HierarchicalRewardInputs:
    base = dict(
        previous_average_distance_to_pen=20.0,
        current_average_distance_to_pen=18.0,
        previous_flock_spread=6.0,
        current_flock_spread=5.0,
        newly_penned=0,
        success=False,
        timeout=False,
        dog_positions=((PEN_X - 5.0, PEN_Y - 5.0),),
        sheep_positions=((PEN_X - 10.0, PEN_Y - 10.0),),
        pen_gate_x=PEN_X,
        pen_gate_y=PEN_Y,
        flock_centroid=(PEN_X - 10.0, PEN_Y - 10.0),
        flock_to_pen_dx=10.0,
        flock_to_pen_dy=10.0,
        step_fraction=0.5,
    )
    base.update(overrides)
    return HierarchicalRewardInputs(**base)


def _compute(**overrides) -> HierarchicalRewardBreakdown:
    return HierarchicalRewardComputer(HierarchicalRewardConfig()).compute(
        _default_inputs(**overrides)
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_compute_returns_breakdown():
    bd = _compute()
    assert isinstance(bd, HierarchicalRewardBreakdown)


def test_total_equals_sum_of_components():
    bd = _compute()
    manual_total = (
        bd.sheep_closer_to_pen
        + bd.sheep_penned
        + bd.flock_grouped
        + bd.pressure_from_behind
        + bd.dog_spread
        + bd.dog_blocking_escape
        + bd.task_completion
        + bd.speed_bonus
        + bd.scatter_penalty
        + bd.overpressure_penalty
        + bd.gate_blocking_penalty
        + bd.dog_stack_penalty
        + bd.sheep_away_from_pen
        + bd.wandering_penalty
        + bd.timeout_penalty
    )
    assert abs(bd.total - manual_total) < 1e-6


# ---------------------------------------------------------------------------
# Positive signals
# ---------------------------------------------------------------------------


def test_reward_positive_when_flock_approaches_pen():
    bd = _compute(
        previous_average_distance_to_pen=20.0,
        current_average_distance_to_pen=15.0,
    )
    assert bd.sheep_closer_to_pen > 0.0


def test_sheep_penned_positive():
    bd = _compute(newly_penned=2)
    assert bd.sheep_penned > 0.0


def test_task_completion_on_success():
    bd = _compute(success=True)
    assert bd.task_completion > 0.0


def test_no_task_completion_without_success():
    bd = _compute(success=False)
    assert bd.task_completion == 0.0


# ---------------------------------------------------------------------------
# Negative signals
# ---------------------------------------------------------------------------


def test_scatter_penalised_when_spread_increases():
    bd = _compute(
        previous_flock_spread=4.0,
        current_flock_spread=8.0,
    )
    assert bd.scatter_penalty < 0.0


def test_sheep_away_from_pen_penalised():
    bd = _compute(
        previous_average_distance_to_pen=10.0,
        current_average_distance_to_pen=15.0,
    )
    assert bd.sheep_away_from_pen < 0.0


def test_gate_blocking_penalised_when_dog_near_gate():
    """Dog sitting exactly on the pen gate should trigger a penalty."""
    bd = _compute(
        dog_positions=((PEN_X, PEN_Y),),
        sheep_positions=((PEN_X - 30.0, PEN_Y - 30.0),),
    )
    assert bd.gate_blocking_penalty < 0.0


def test_dog_stacking_penalised():
    """Two dogs at the same tile should incur a stacking penalty."""
    bd = _compute(
        dog_positions=((PEN_X - 10.0, PEN_Y - 10.0), (PEN_X - 10.0, PEN_Y - 10.0)),
    )
    assert bd.dog_stack_penalty < 0.0


def test_timeout_penalised():
    bd = _compute(timeout=True)
    assert bd.timeout_penalty < 0.0


def test_no_timeout_penalty_without_timeout():
    bd = _compute(timeout=False)
    assert bd.timeout_penalty == 0.0
