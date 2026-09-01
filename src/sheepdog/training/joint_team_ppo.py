"""Isolated trainer for true joint team-step MaskablePPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheepdog.checkpoints.store import CheckpointStore
from sheepdog.environment import ACTION_ORDER
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.observations import HERD_DOG_SLOTS
from sheepdog.policies.joint_team import JointTeamPolicy
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.team_rl_env import TeamActionRLEnv


class JointTeamPPOTrainer(MaskablePPOTrainer):
    """Train centralized joint actions without sharing legacy PPO state."""

    MODEL_DIRNAME = "models/joint_team"
    MODEL_PREFIX = "joint-team-ppo"
    POLICY_CLASS = JointTeamPolicy
    TRAINER_TYPE = "joint_maskable_ppo"
    POLICY_MODE = "joint_team_policy"
    REPLAY_MODE = "joint_team_ppo"
    STATE_FILENAME = "joint-team-training-state.json"
    SUMMARY_FILENAME = "joint-team-training-summary.json"

    def __init__(self, config: Any, output_root: str | Path) -> None:
        super().__init__(config, output_root)
        self.checkpoint_store = CheckpointStore(
            self.output_root / "checkpoints" / "joint_team"
        )
        self.evaluator = Evaluator(config, self.output_root / "evaluations" / "joint_team")

    def _stage_model_root(self, active_stage: int) -> Path:
        """Keep all joint model checkpoints under their own directory."""
        stage_root = self.output_root / self.MODEL_DIRNAME
        return stage_root / f"stage{active_stage}" if active_stage >= 9 else stage_root

    def _training_signature(self) -> dict[str, Any]:
        """Return a signature incompatible with sequential PPO checkpoints."""
        signature = super()._training_signature()
        adapter = TeamActionRLEnv(self.config)
        signature.update(
            {
                "architecture": "joint_team_v1",
                "action_sizes": [len(ACTION_ORDER)] * HERD_DOG_SLOTS,
                "observation_size": int(adapter.observation_space.shape[0]),
                "dog_slots": HERD_DOG_SLOTS,
            }
        )
        signature.pop("action_size", None)
        return signature

    def _load_summary_checkpoints(self) -> list[dict[str, Any]]:
        """Load only prior joint-team checkpoint summaries."""
        path = self.output_root / self.SUMMARY_FILENAME
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        checkpoints = payload.get("checkpoints")
        return list(checkpoints) if isinstance(checkpoints, list) else []
