"""Centralized configuration for deterministic training diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeterministicDiagnosticsConfig:
    """Centralized diagnostic thresholds and evaluation window constants.

    Note: This is strictly diagnostic observation configuration and does NOT
    modify RL training, environment dynamics, rewards, or PPO hyperparameters.
    """

    window_checkpoints_max: int = 8
    min_unique_checkpoints_required: int = 3
    min_episodes_for_trend: int = 15

    # Performance Band thresholds (Absolute performance level)
    near_mastery_threshold: float = 0.90
    high_performance_threshold: float = 0.75
    moderate_performance_threshold: float = 0.50

    # Success Trend thresholds (Direction of change over unique checkpoints)
    improving_slope_threshold: float = 0.02
    regressing_slope_threshold: float = -0.04
    plateau_max_slope: float = 0.02
    plateau_variance_threshold: float = 0.02

    # Step efficiency thresholds
    step_reduction_threshold_pct: float = 0.05
    step_reduction_min_steps: float = 5.0
    step_regression_threshold_pct: float = 0.10
    step_regression_min_steps: float = 10.0

    # Seed failure streak thresholds (Over consecutive unique training checkpoints)
    watch_checkpoint_streak_threshold: int = 2
    persistent_candidate_streak_threshold: int = 3
    strong_persistence_streak_threshold: int = 4

    # Seed outlier thresholds
    outlier_min_checkpoints: int = 3
    outlier_max_pass_rate: float = 0.25
    outlier_min_stage_pass_rate: float = 0.50

    # Spatial bottleneck thresholds
    spatial_disparity_threshold: float = 0.35
    wall_stall_time_pct_threshold: float = 0.40
    gate_corridor_fail_steps_threshold: int = 20

    # Cliff detection
    cliff_all_time_min_max: float = 0.50
    cliff_latest_max: float = 0.15
