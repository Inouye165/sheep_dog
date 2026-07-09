"""Curriculum telemetry manager for local and cloud (wandb) logging."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


class CurriculumTelemetryManager:
    """Manages training telemetry by appending logs locally and syncing to Weights & Biases."""

    def __init__(self, output_dir: str | Path = "artifacts") -> None:
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / "training_history.json"
        self._wandb_initialized = False

    def initialize_wandb(
        self,
        project_name: str = "sheep_dog_herding",
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Weights & Biases if present on the system."""
        try:
            import wandb
            # Avoid re-initializing if there is an active run
            if wandb.run is None:
                wandb.init(project=project_name, config=config_dict, reinit=True)
            self._wandb_initialized = True
            logger.info("Weights & Biases initialized successfully.")
        except ImportError:
            logger.warning("Weights & Biases (wandb) is not installed. Cloud telemetry is disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")

    def log(
        self,
        step: int,
        stage: int,
        success_rate: float,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
    ) -> None:
        """Log a metrics point to both local JSON and wandb."""
        record = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "step": step,
            "stage": stage,
            "success_rate": success_rate,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
        }

        # 1. Local logging: append to a flat JSON array in training_history.json
        history: list[dict[str, Any]] = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

        history.append(record)
        try:
            atomic_write_json(self.history_path, history)
        except Exception as e:
            logger.error(f"Failed to write telemetry to local history: {e}")

        # 2. Cloud logging (Weights & Biases)
        if self._wandb_initialized:
            try:
                import wandb
                # Flatten the data structure for wandb logging
                wandb_data = {
                    "stage": stage,
                    "success_rate": success_rate,
                    **metrics,
                }
                for k, v in hyperparameters.items():
                    if isinstance(v, (int, float, str, bool)):
                        wandb_data[f"hyperparam/{k}"] = v
                wandb.log(wandb_data, step=step)
            except Exception as e:
                logger.error(f"Failed to log to wandb: {e}")
