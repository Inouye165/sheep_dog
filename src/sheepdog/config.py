"""Configuration objects for the simulation, reward model, and training flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sheepdog.policies.base import PolicyMode


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Grid, entity, and termination settings."""

    width: int = 40
    height: int = 30
    dogs: int = 3
    sheep: int = 6
    pen_width: int = 5
    pen_height: int = 5
    pen_opening: str = "left"
    max_steps: int = 300
    seconds_per_step: float = 1.0
    dog_vision: int = 8
    sheep_vision: int = 6
    flock_radius: int = 5
    dog_speed: int = 3
    sheep_speed: int = 1
    no_progress_window: int = 40
    no_progress_distance_delta: float = 0.15
    no_progress_penned_delta: int = 0
    no_progress_timeout_penalty_steps: int = 40
    seed_offset: int = 0


@dataclass(frozen=True, slots=True)
class InstinctRewardConfig:
    """Weights and toggles for the optional sheepdog-instinct reward shaping.

    These rewards simulate biological herding instincts (pressure-zone,
    safe distance, grouping, drive/fetch, anti-chaos, cooperation). They
    are off by default so existing training behavior stays unchanged.
    """

    enable_instinct_rewards: bool = False
    debug_reward_breakdown: bool = False
    curriculum_stage: int = 0

    pressure_zone_weight: float = 0.6
    safe_pressure_weight: float = 0.4
    grouping_weight: float = 0.3
    target_progress_weight: float = 0.5
    chaos_penalty_weight: float = 0.5
    overpressure_penalty_weight: float = 0.4
    split_flock_penalty_weight: float = 0.3

    safe_pressure_min_distance: float = 2.0
    safe_pressure_max_distance: float = 6.0
    overpressure_distance: float = 1.5
    chaos_inside_flock_distance: float = 1.5
    chaos_scatter_delta: float = 0.5
    split_flock_ratio: float = 1.8


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
    instincts: InstinctRewardConfig = field(default_factory=InstinctRewardConfig)


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
class PolicyConfig:
    """Runtime policy controls for demo, evaluation, and untrained playback."""

    policy_mode: PolicyMode = "instinct_only"
    allow_instinct_target_awareness: bool = False
    handler_target_enabled: bool = False


@dataclass(frozen=True, slots=True)
class LabConfig:
    """Bundle of environment, reward, and training settings."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabConfig:
        rewards_payload = dict(payload["rewards"])
        instincts_payload = rewards_payload.pop("instincts", None)
        policy_payload = payload.get("policy")
        instincts = (
            InstinctRewardConfig(**instincts_payload)
            if isinstance(instincts_payload, dict)
            else InstinctRewardConfig()
        )
        policy = PolicyConfig(**policy_payload) if isinstance(policy_payload, dict) else PolicyConfig()
        return cls(
            environment=EnvironmentConfig(**payload["environment"]),
            rewards=RewardConfig(instincts=instincts, **rewards_payload),
            training=TrainingConfig(**payload["training"]),
            policy=policy,
        )


def default_artifact_root(output_dir: str | Path) -> Path:
    """Return the root path for generated artifacts."""

    return Path(output_dir).resolve()
