"""Unit tests for AdaptiveStepController."""

from __future__ import annotations

import pytest

from sheepdog.training.adaptive_learning import (
    MAX_ADAPTIVE_STAGES,
    AdaptiveStepController,
    _target_stage_for_success,
)


def test_target_stage_for_success_thresholds():
    assert _target_stage_for_success(0.0) == 1
    assert _target_stage_for_success(0.59) == 1
    assert _target_stage_for_success(0.60) == 2
    assert _target_stage_for_success(0.74) == 2
    assert _target_stage_for_success(0.75) == 3
    assert _target_stage_for_success(0.89) == 3
    assert _target_stage_for_success(0.90) == 4
    assert _target_stage_for_success(1.00) == 4


def test_initial_state_is_stage_one_no_modification():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        base_mutation_scale=0.08,
        base_entropy_coef=0.016,
        initial_curriculum_stage=1,
    )
    state = controller.get_state()
    assert state.stage == 1
    assert state.max_stages == MAX_ADAPTIVE_STAGES == 4
    assert state.multiplier == 1.0
    assert "Stage 1 of 4" in state.label
    assert "No modification" in state.label
    assert state.effective_learning_rate == pytest.approx(1e-4)
    assert state.effective_mutation_scale == pytest.approx(0.08)
    assert state.effective_entropy_coef == pytest.approx(0.016)

    # Check to_dict serialization
    state_dict = state.to_dict()
    assert state_dict["effective_entropy_coef"] == pytest.approx(0.016)
    assert state_dict["stage"] == 1


def test_debounce_protects_against_single_fluke():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        debounce_required_hits=2,
        initial_curriculum_stage=1,
    )
    # 1. Start with 0.50 (stage 1)
    s1 = controller.update(0.50, current_curriculum_stage=1)
    assert s1.stage == 1

    # 2. Single 0.90 hit should NOT jump straight to stage 4 because debounce required = 2
    s2 = controller.update(0.90, current_curriculum_stage=1)
    assert s2.stage == 1  # Still stage 1 due to debounce + EMA smoothing

    # 3. Second 0.90 hit confirms improvement, advances conservatively to stage 2
    s3 = controller.update(0.90, current_curriculum_stage=1)
    assert s3.stage >= 2


def test_stage_progression_with_sustained_high_success():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        base_mutation_scale=0.08,
        base_entropy_coef=0.016,
        debounce_required_hits=2,
        initial_curriculum_stage=1,
    )

    # Sustained 70% success advances to Stage 2
    controller.update(0.70, 1)
    state = controller.update(0.70, 1)
    assert state.stage == 2
    assert state.multiplier == 0.80
    assert "Stage 2 of 4" in state.label
    assert state.effective_learning_rate == pytest.approx(8e-5)
    assert state.effective_mutation_scale == pytest.approx(0.064)
    assert state.effective_entropy_coef == pytest.approx(0.016 * 0.80)

    # Sustained 80% success advances to Stage 3
    controller.update(0.80, 1)
    controller.update(0.80, 1)
    state = controller.update(0.80, 1)
    assert state.stage == 3
    assert state.multiplier == 0.65
    assert "Stage 3 of 4" in state.label
    assert state.effective_entropy_coef == pytest.approx(0.016 * 0.65)

    # Sustained 90% success advances to Stage 4
    controller.update(0.95, 1)
    controller.update(0.95, 1)
    controller.update(0.95, 1)
    state = controller.update(0.95, 1)
    assert state.stage == 4
    assert state.multiplier == 0.50
    assert "Stage 4 of 4" in state.label
    assert state.effective_learning_rate == pytest.approx(5e-5)
    assert state.effective_mutation_scale == pytest.approx(0.04)
    assert state.effective_entropy_coef == pytest.approx(0.016 * 0.50)


def test_curriculum_stage_reset_preserves_plasticity():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        base_mutation_scale=0.08,
        debounce_required_hits=1,
        initial_curriculum_stage=6,
    )
    # Reach Stage 4 in Curriculum Stage 6
    controller.update(0.90, 6)
    controller.update(0.90, 6)
    controller.update(0.90, 6)
    state = controller.get_state()
    assert state.stage == 4
    assert state.multiplier == 0.50

    # Auto-promotion occurs to Curriculum Stage 7!
    new_state = controller.update(0.20, current_curriculum_stage=7)
    assert new_state.stage == 1
    assert new_state.multiplier == 1.0
    assert "Stage 1 of 4" in new_state.label
    assert "No modification" in new_state.label
    assert new_state.curriculum_stage == 7
    assert new_state.effective_learning_rate == pytest.approx(1e-4)


def test_restore_preserves_same_curriculum_adaptive_progress():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        base_mutation_scale=0.08,
        base_entropy_coef=0.01,
        initial_curriculum_stage=8,
    )

    state = controller.restore(
        {
            "stage": 2,
            "curriculum_stage": 8,
            "ema_success_rate": 0.6281,
            "consecutive_hits": 1,
        },
        current_curriculum_stage=8,
    )

    assert state.stage == 2
    assert state.curriculum_stage == 8
    assert state.ema_success_rate == pytest.approx(0.6281)
    assert state.consecutive_hits == 1
    assert state.effective_learning_rate == pytest.approx(8e-5)
    assert state.effective_entropy_coef == pytest.approx(0.008)


def test_restore_resets_progress_when_curriculum_changes():
    controller = AdaptiveStepController(initial_curriculum_stage=9)

    state = controller.restore(
        {
            "stage": 4,
            "curriculum_stage": 8,
            "ema_success_rate": 0.95,
            "consecutive_hits": 1,
        },
        current_curriculum_stage=9,
    )

    assert state.stage == 1
    assert state.curriculum_stage == 9
    assert state.ema_success_rate == 0.0
    assert state.consecutive_hits == 0


def test_graceful_asymmetric_recovery_on_performance_drop():
    controller = AdaptiveStepController(
        base_learning_rate=1e-4,
        debounce_required_hits=1,
        initial_curriculum_stage=1,
    )
    # Move to stage 3
    controller.update(0.80, 1)
    controller.update(0.80, 1)
    assert controller.get_state().stage == 3

    # Performance drops to 30%
    state_after_drop = controller.update(0.30, 1)
    # Asymmetric recovery steps down at most 1 stage per check rather than jumping instantly
    assert state_after_drop.stage == 2

    # Continued low performance steps down to stage 1
    state_after_drop_2 = controller.update(0.30, 1)
    assert state_after_drop_2.stage == 1
    assert state_after_drop_2.multiplier == 1.0
