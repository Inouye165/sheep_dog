"""Configuration objects for the simulation, reward model, and training flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sheepdog.policies.base import PolicyMode, PolicyType, TrainerType


def resolve_workspace_path(path_str: str | Path) -> Path:
    """Resolve relative workspace path against repository root if available."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return (cwd / p).resolve()
    for parent in cwd.parents:
        if (parent / "pyproject.toml").exists():
            return (parent / p).resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "pyproject.toml").exists():
        return (repo_root / p).resolve()
    return (cwd / p).resolve()



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
    dog_sprint_multiplier: float = 2.0
    sheep_speed: float = 0.75
    # Sheep-to-flock attraction used by sheep movement. Lower values reduce
    # autonomous self-grouping, letting dog pressure matter more.
    sheep_flock_cohesion_weight: float = 0.2
    # Inward boundary recovery bonus weight applied when a sheep candidate step reduces wall contact.
    sheep_inward_recovery_weight: float = 16.0
    # Threshold for consecutive steps on an outer wall to count as a wall stall episode diagnostic.
    wall_stall_window: int = 15
    # When False, flock cohesion is only applied if at least one dog is within
    # sheep vision range, preventing autonomous regrouping across the field.
    sheep_cohere_without_dog_pressure: bool = True
    # Per-personality display colors used by the web replay viewer. The keys
    # must match entries in ``entities.SHEEP_PERSONALITIES``; any sheep with
    # an unknown personality falls back to the ``obedient`` entry. When
    # ``sheep_personality_strength`` is 0.0 all sheep are ``obedient`` and
    # therefore share the same color in the viewer.
    sheep_personality_colors: dict[str, str] = field(
        default_factory=lambda: {
            "obedient": "#f8fafc",
            "pen_fearful": "#f87171",
            "pen_shy": "#93c5fd",
            "escapist": "#f9a8d4",
            "bold": "#fdba74",
        }
    )
    no_progress_window: int = 80
    no_progress_distance_delta: float = 0.15
    no_progress_penned_delta: int = 0
    no_progress_timeout_penalty_steps: int = 80
    role_stickiness_distance: float = 40.0
    role_stickiness_bonus: float = 8.0
    flank_role_stickiness_bonus: float = 9.0
    blocker_role_stickiness_bonus: float = 6.0
    role_minimum_hold_steps: int = 8
    gate_corridor_half_width: float = 2.5
    gate_approach_distance: float = 12.0
    gate_hold_safe_distance: float = 6.0
    gate_progress_epsilon: float = 0.1
    controlled_flock_spread_threshold: float = 4.0
    stalled_control_activation_steps: int = 6
    seed_offset: int = 0
    # Strength of per-sheep personality biases (0.0 disables, ~0.25-0.5 is mild,
    # >1.0 becomes pronounced). Personalities are assigned at episode reset and
    # held fixed for the entire episode. See ``entities.SHEEP_PERSONALITIES``.
    sheep_personality_strength: float = 0.0
    # Optional fixed sheep personality override for all spawned sheep.
    # Empty string keeps mixed personalities based on RNG.
    sheep_personality_override: str = ""
    # Offset added to a dedicated personality RNG (separate from the env RNG
    # that drives positions/jitter). Bump this to reshuffle the personality
    # lineup for the same eval seed without changing positions or dynamics.
    sheep_personality_seed_offset: int = 0
    # Minimum number of dogs that must remain after assigning a blocker.
    # Set to 1 to require at least one herder still pushing the flock;
    # set to 0 to allow blocker even when it would be the only dog (legacy).
    blocker_min_remaining_dogs: int = 1
    # Current active herding curriculum stage (0 disables overrides)
    curriculum_stage: int = 0
    # Weighted sheep-spawn scenario mix, e.g. {"fixed_easy": 0.7,
    # "randomized_flock": 0.3}. Keys must be entries from ``SPAWN_MODES``.
    # An empty dict keeps the legacy stage-keyed spawn behaviour so older
    # checkpoints and replays remain reproducible.
    spawn_mix: dict[str, float] = field(default_factory=dict)
    # Distance bands (in cells from the flock centre) used by the
    # nearby_stray / two_strays and farther_stray spawn modes.
    stray_near_min: float = 8.0
    stray_near_max: float = 12.0
    stray_far_min: float = 18.0
    stray_far_max: float = 28.0
    # Pen placement mode; one of ``PEN_PLACEMENTS``. "corner" keeps the
    # legacy fixed top-right pen so old checkpoints replay unchanged.
    pen_placement: str = "corner"
    # When True, collection signals (flock spread shrinking, the farthest
    # sheep approaching the flock or the pen) count as progress for the
    # no-progress termination check.
    count_collection_progress: bool = False


# Sheep spawn layout modes understood by the environment.
SPAWN_MODES: tuple[str, ...] = (
    "fixed_easy",
    "randomized_flock",
    "nearby_stray",
    "farther_stray",
    "two_strays",
    "split_flock",
    "partial_scattered",
    "scattered_sheep",
    "all_corners",
    "wall_recovery",
)

# Pen placement modes understood by the environment.
PEN_PLACEMENTS: tuple[str, ...] = (
    "corner",
    "same_wall",
    "any_wall",
    "away_from_corner",
    "interior",
    "random",
)


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
    safe_pressure_weight: float = 0.9
    grouping_weight: float = 0.3
    target_progress_weight: float = 1.0
    chaos_penalty_weight: float = 0.5
    overpressure_penalty_weight: float = 0.5
    split_flock_penalty_weight: float = 0.3
    dog_overshoot_penalty_hold: float = 0.5

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
    scatter_penalty_scale: float = 0.2
    time_penalty: float = 0.05
    no_progress_penalty: float = 0.1
    terminal_success_reward: float = 20.0
    terminal_failure_penalty: float = 12.0
    wall_pressure_penalty: float = 0.4
    wait_penalty: float = 0.05
    sprint_cost_scale: float = 0.12
    gate_progress_scale: float = 1.6
    gate_corridor_progress_scale: float = 0.8
    gate_alignment_scale: float = 1.0
    lane_crowding_penalty_scale: float = 0.9
    lane_crowding_activation_distance: float = 14.0
    lane_crowding_forward_distance: float = 6.0
    lane_crowding_lateral_tolerance: float = 1.75
    stalled_control_penalty: float = 0.45
    wrong_hold_penalty: float = 0.8
    instincts: InstinctRewardConfig = field(default_factory=InstinctRewardConfig)
    farthest_sheep_progress_scale: float = 0.0
    stray_ignore_penalty_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training and checkpoint schedule settings."""

    trainer_type: TrainerType = "hill_climb"
    policy_type: PolicyType = "linear"
    episodes: int = 1_000
    checkpoint_episodes: tuple[int, ...] = (0, 5, 10, 25, 50, 100, 500, 1_000)
    checkpoint_timesteps: tuple[int, ...] = ()
    evaluation_seeds: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    train_seed: int = 7
    evaluation_seed: int = 91
    candidate_evaluation_seeds: tuple[int, ...] = (91, 92, 93, 94, 95)
    candidate_pool_size: int = 4
    mutation_scale: float = 0.08
    neural_hidden_sizes: tuple[int, ...] = (128, 128, 128)
    learning_rate: float = 1e-4
    learning_rate_final: float = 3e-5
    rollout_steps: int = 2048
    batch_size: int = 1024
    total_timesteps: int = 500_000
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    # Number of vectorized RL environments used by neural PPO training.
    # >1 enables process-based parallel stepping (SubprocVecEnv).
    ppo_env_workers: int = 8
    quick_evaluation_seed_count: int = 10
    confidence_candidate_success_rate: float = 0.5
    replay_export_on_new_best: bool = True
    replay_export_on_promotion: bool = True
    replay_export_on_final: bool = True
    replay_export_on_failed_diagnostic: bool = True
    replay_export_every_n_checkpoints: int = 0
    runtime_heartbeat_seconds: int = 15
    invalid_action_masking: bool = True
    output_dir: str = "artifacts"
    web_export_dir: str = "web/public/generated"
    backup_enabled: bool = True
    backup_dir: str = "artifacts/backups"
    hourly_backup_interval_seconds: int = 3600
    max_hourly_backups_per_stage: int = 24
    # Scenario-based training: enable to mix difficult starting scenarios
    # with normal random starts for robustness training.
    scenario_training_enabled: bool = False
    scenario_mix: dict[str, float] = field(
        default_factory=lambda: {
            "random": 0.50,
            "scattered_sheep": 0.20,
            "split_flock": 0.15,
            "corner_huddle": 0.15,
        }
    )
    # Observation mode: "guided" (default) includes scripted role labels and
    # strategy targets; "emergent" strips roles/targets so the model must learn
    # herding purely from raw observations and reward signal.
    observation_mode: str = "guided"
    # Weights & Biases telemetry tracking integration toggle.
    wandb_enabled: bool = False
    # Adaptive step-size / learning rate controller toggle.
    enable_adaptive_learning: bool = True


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
        """Serialize this config to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabConfig:
        """Construct a LabConfig from a raw dict (e.g., loaded from JSON)."""
        rewards_payload = dict(payload["rewards"])
        instincts_payload = rewards_payload.pop("instincts", None)
        policy_payload = payload.get("policy")
        training_payload = dict(payload["training"])
        if "checkpoint_episodes" in training_payload:
            training_payload["checkpoint_episodes"] = tuple(training_payload["checkpoint_episodes"])
        if "evaluation_seeds" in training_payload:
            training_payload["evaluation_seeds"] = tuple(training_payload["evaluation_seeds"])
        candidate_seeds = training_payload.get("candidate_evaluation_seeds")
        if candidate_seeds is not None:
            training_payload["candidate_evaluation_seeds"] = tuple(candidate_seeds)
        elif "evaluation_seed" in training_payload:
            training_payload["candidate_evaluation_seeds"] = ()
        hidden_sizes = training_payload.get("neural_hidden_sizes")
        if hidden_sizes is not None:
            training_payload["neural_hidden_sizes"] = tuple(hidden_sizes)
        if "ppo_env_workers" in training_payload:
            training_payload["ppo_env_workers"] = int(training_payload["ppo_env_workers"])
        instincts = (
            InstinctRewardConfig(**instincts_payload)
            if isinstance(instincts_payload, dict)
            else InstinctRewardConfig()
        )
        policy = (
            PolicyConfig(**policy_payload) if isinstance(policy_payload, dict) else PolicyConfig()
        )
        environment_payload = dict(payload["environment"])
        personality_colors = environment_payload.get("sheep_personality_colors")
        if isinstance(personality_colors, dict):
            environment_payload["sheep_personality_colors"] = {
                str(name): str(color) for name, color in personality_colors.items()
            }
        # Backward-compatible: drop any legacy ``sheep_palette`` field so
        # checkpoints written before per-personality colors can still load.
        environment_payload.pop("sheep_palette", None)
        spawn_mix = environment_payload.get("spawn_mix")
        if isinstance(spawn_mix, dict):
            environment_payload["spawn_mix"] = {
                str(name): float(weight) for name, weight in spawn_mix.items()
            }
        return cls(
            environment=EnvironmentConfig(**environment_payload),
            rewards=RewardConfig(instincts=instincts, **rewards_payload),
            training=TrainingConfig(**training_payload),
            policy=policy,
        )


def default_artifact_root(output_dir: str | Path) -> Path:
    """Return the root path for generated artifacts."""

    return Path(output_dir).resolve()
