"""Configuration objects for the simulation, reward model, and training flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Grid, entity, and termination settings."""

    width: int = 80
    height: int = 60
    dogs: int = 3
    sheep: int = 6
    pen_width: int = 10
    pen_height: int = 10
    pen_opening: str = "left"
    max_steps: int = 600
    seconds_per_step: float = 1.0
    dog_vision: int = 16
    sheep_vision: int = 12
    flock_radius: int = 10
    dog_speed: float = 1.0
    sheep_speed: float = 0.75
    no_progress_window: int = 80
    no_progress_distance_delta: float = 0.15
    no_progress_penned_delta: int = 0
    no_progress_timeout_penalty_steps: int = 80
    seed_offset: int = 0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward shaping controls."""

    progress_scale: float = 2.0
    sheep_penned_reward: float = 8.0
    flock_cohesion_scale: float = 0.35
    scatter_penalty_scale: float = 0.65
    time_penalty: float = 0.05
    no_progress_penalty: float = 1.0
    terminal_success_reward: float = 20.0
    terminal_failure_penalty: float = 12.0
    wall_pressure_penalty: float = 0.4
    wait_penalty: float = 0.05


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training and checkpoint schedule settings."""

    episodes: int = 1_000
    checkpoint_episodes: tuple[int, ...] = (0, 5, 10, 25, 50, 100, 500, 1_000)
    evaluation_seeds: tuple[int, ...] = (11, 23, 37, 41, 53)
    train_seed: int = 7
    evaluation_seed: int = 91
    mutation_scale: float = 0.08
    output_dir: str = "artifacts"
    web_export_dir: str = "web/public/generated"


@dataclass(frozen=True, slots=True)
class LabConfig:
    """Bundle of environment, reward, and training settings."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabConfig:
        return cls(
            environment=EnvironmentConfig(**payload["environment"]),
            rewards=RewardConfig(**payload["rewards"]),
            training=TrainingConfig(**payload["training"]),
        )


def default_artifact_root(output_dir: str | Path) -> Path:
    """Return the root path for generated artifacts."""

    return Path(output_dir).resolve()
