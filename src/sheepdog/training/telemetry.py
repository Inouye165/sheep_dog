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
        run_id: str | None = None,
        checkpoint_id: str | None = None,
        evaluation_id: str | None = None,
        global_episode: int | None = None,
        episode_in_stage: int | None = None,
        recorded_at: str | None = None,
    ) -> None:
        """Log a metrics point to both local JSON and wandb."""
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        r_id = run_id or "run_unknown"
        cp_id = checkpoint_id or f"chk_{r_id}_ep_{step}"
        eval_id = evaluation_id or f"eval_{r_id}_ep_{step}"
        glob_ep = global_episode if global_episode is not None else step
        ep_in_stage = episode_in_stage if episode_in_stage is not None else step
        rec_at = recorded_at or now_iso

        record = {
            "timestamp": rec_at,
            "step": step,
            "stage": stage,
            "success_rate": success_rate,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "run_id": r_id,
            "checkpoint_id": cp_id,
            "evaluation_id": eval_id,
            "global_episode": glob_ep,
            "episode_in_stage": ep_in_stage,
            "recorded_at": rec_at,
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
