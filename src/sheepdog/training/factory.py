"""Trainer construction helpers."""

from __future__ import annotations

from pathlib import Path

from sheepdog.config import LabConfig
from sheepdog.training.trainer import Trainer


def create_trainer(config: LabConfig, output_root: str | Path) -> Trainer:
    """Create the configured trainer implementation."""

    if config.training.trainer_type == "hill_climb":
        return Trainer(config, output_root)
    if config.training.trainer_type == "maskable_ppo":
        # pylint: disable-next=import-outside-toplevel
        from sheepdog.training.maskable_ppo import MaskablePPOTrainer

        return MaskablePPOTrainer(config, output_root)
    if config.training.trainer_type == "joint_maskable_ppo":
        # pylint: disable-next=import-outside-toplevel
        from sheepdog.training.joint_team_ppo import JointTeamPPOTrainer

        return JointTeamPPOTrainer(config, output_root)
    if config.training.trainer_type == "hierarchical_maskable_ppo":
        # pylint: disable-next=import-outside-toplevel
        from sheepdog.training.hierarchical_trainer import HierarchicalMaskablePPOTrainer

        return HierarchicalMaskablePPOTrainer(config, output_root)
    raise ValueError(f"Unsupported trainer type: {config.training.trainer_type}")
