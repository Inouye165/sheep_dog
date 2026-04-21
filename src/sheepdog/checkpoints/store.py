"""Checkpoint persistence utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Serialized checkpoint metadata."""

    checkpoint_episode: int
    total_training_episodes: int
    policy_name: str
    seed: int
    success_rate: float
    average_completion_steps: float
    timeout_rate: float
    average_sheep_penned: float
    average_reward: float
    environment_config: dict[str, Any]
    reward_config: dict[str, Any]
    policy_weights: dict[str, float] | None = None
    evaluation_replay_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointStore:
    """Write checkpoints to disk in a predictable layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, metadata: CheckpointMetadata) -> Path:
        path = self.root / f"checkpoint-{metadata.checkpoint_episode:06d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2)
        return path
