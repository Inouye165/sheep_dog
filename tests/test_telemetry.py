"""Unit tests for CurriculumTelemetryManager W&B logging and step monotonicity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from sheepdog.training.telemetry import CurriculumTelemetryManager


def test_initialize_wandb_without_reinit_boolean(tmp_path: Path) -> None:
    """Test that initialize_wandb calls wandb.init without the deprecated reinit=True argument."""
    manager = CurriculumTelemetryManager(output_dir=tmp_path)

    mock_wandb = MagicMock()
    mock_wandb.run = None
    mock_wandb.init = MagicMock()

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        manager.initialize_wandb(project_name="test_project", config_dict={"lr": 0.001})

        assert manager._wandb_initialized is True
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            config={"lr": 0.001},
        )


def test_telemetry_wandb_step_monotonicity(tmp_path: Path) -> None:
    """Test that interleaving log_episode and log calls maintains monotonic steps in wandb.log."""
    manager = CurriculumTelemetryManager(output_dir=tmp_path)
    manager._wandb_initialized = True

    logged_steps: list[int] = []

    def capture_log(data: dict, step: int | None = None) -> None:
        if step is not None:
            logged_steps.append(step)

    mock_wandb = MagicMock()
    mock_wandb.log = capture_log

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        # Episode logs (1 to 50)
        for ep in range(1, 51):
            manager.log_episode(
                episode=ep,
                stage=1,
                reward=100.0 + ep,
                penned=4,
                total_sheep=4,
                success=True,
            )

        # Checkpoint evaluation log at episode 50
        manager.log(
            step=50,
            stage=1,
            success_rate=1.0,
            metrics={"average_reward": 125.0},
            hyperparameters={"learning_rate": 0.0003},
            global_episode=50,
        )

        # Episode logs resume (51 to 100)
        for ep in range(51, 101):
            manager.log_episode(
                episode=ep,
                stage=1,
                reward=200.0 + ep,
                penned=4,
                total_sheep=4,
                success=True,
            )

    assert len(logged_steps) == 101
    # Verify steps are strictly monotonically increasing
    for i in range(1, len(logged_steps)):
        assert logged_steps[i] > logged_steps[i - 1], (
            f"Non-monotonic step at index {i}: {logged_steps[i - 1]} -> {logged_steps[i]}"
        )


def test_telemetry_wandb_step_regression_prevention(tmp_path: Path) -> None:
    """Test that if a caller passes a step lower than previously logged step, target_step increments safely."""
    manager = CurriculumTelemetryManager(output_dir=tmp_path)
    manager._wandb_initialized = True

    logged_steps: list[int] = []

    def capture_log(data: dict, step: int | None = None) -> None:
        if step is not None:
            logged_steps.append(step)

    mock_wandb = MagicMock()
    mock_wandb.log = capture_log

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        # Call log with high step (e.g. 5000)
        manager.log_episode(episode=5000, stage=1, reward=100.0, penned=4, total_sheep=4, success=True)

        # Call log with a step lower (e.g. 4000)
        manager.log_episode(episode=4000, stage=1, reward=105.0, penned=4, total_sheep=4, success=True)

    assert logged_steps[0] == 5000
    assert logged_steps[1] == 5001  # Auto-corrected to prevent wandb warning!
