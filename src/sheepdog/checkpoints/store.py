"""Checkpoint persistence utilities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sheepdog.atomic_io import atomic_write_json


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
    trainer_type: str = "hill_climb"
    policy_type: str = "linear"
    policy_weights: dict[str, float] | None = None
    policy_state_path: str | None = None
    policy_config: dict[str, Any] | None = None
    evaluation_replay_path: str | None = None
    run_id: str | None = None
    checkpoint_id: str | None = None
    environment_episodes_total: int | None = None
    environment_episodes_since_run_start: int | None = None
    parent_run_id: str | None = None
    parent_checkpoint_id: str | None = None
    global_timestep: int | None = None
    observation_schema_hash: str | None = None
    action_space_hash: str | None = None
    reward_schema_version: str | None = None
    env_config_version: str | None = None
    created_timestamp: str | None = None
    deterministic_evaluation: bool | None = None
    evaluation_seeds: list[int] | None = None
    policy_version: int | None = None
    policy_gradient_loss: float | None = None
    value_loss: float | None = None
    entropy_loss: float | None = None
    loss: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None
    explained_variance: float | None = None
    training_scenario_coverage: dict[str, Any] | None = None
    curriculum_stage: int = 1
    evaluation_seed_set_id: str | None = None
    evaluation_seed_count: int | None = None
    environment_config_hash: str | None = None
    evaluation_timestamp: str | None = None
    evaluation_id: str | None = None
    evaluation_mode: str | None = None
    promotion_eligible: bool | None = None
    active_runtime_seconds_total: float | None = None
    training_seconds_total: float | None = None
    evaluation_seconds_total: float | None = None
    wall_clock_elapsed_seconds: float | None = None
    session_id: str | None = None
    promotion_gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_env_config_hash(env_config: dict[str, Any]) -> str:
    import hashlib
    import json
    serialized = json.dumps(env_config, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def compute_seed_set_id(seeds: list[int] | tuple[int, ...]) -> str:
    import hashlib
    import json
    serialized = json.dumps(sorted(list(seeds)))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()



class CheckpointStore:
    """Write checkpoints to disk in a predictable layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, metadata: CheckpointMetadata) -> Path:
        payload = metadata.to_dict()
        legacy_path = self.root / f"checkpoint-{metadata.checkpoint_episode:06d}.json"
        if not metadata.checkpoint_id:
            atomic_write_json(legacy_path, payload)
            return legacy_path

        safe_checkpoint_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", metadata.checkpoint_id)
        checkpoint_path = self.root / f"checkpoint-{safe_checkpoint_id}.json"
        atomic_write_json(checkpoint_path, payload)
        atomic_write_json(legacy_path, payload)
        return checkpoint_path


def get_observation_schema_hash(config: Any) -> str:
    import hashlib
    import json
    from sheepdog.environment import SheepdogEnvironment
    from sheepdog.config import LabConfig

    if isinstance(config, dict):
        # Reconstruct LabConfig from serialized environment/reward configurations
        from dataclasses import replace
        import dataclasses
        lab_config = LabConfig()
        env_dict = config.get("environment_config", {})
        rew_dict = config.get("reward_config", {})

        env_fields = {f.name for f in dataclasses.fields(lab_config.environment)}
        filtered_env = {k: v for k, v in env_dict.items() if k in env_fields}
        new_env = replace(lab_config.environment, **filtered_env)

        rew_fields = {f.name for f in dataclasses.fields(lab_config.rewards)}
        filtered_rew = {k: v for k, v in rew_dict.items() if k in rew_fields and k != "instincts"}
        if "instincts" in rew_dict and isinstance(rew_dict["instincts"], dict):
            instinct_fields = {f.name for f in dataclasses.fields(lab_config.rewards.instincts)}
            filtered_inst = {ik: iv for ik, iv in rew_dict["instincts"].items() if ik in instinct_fields}
            new_instincts = replace(lab_config.rewards.instincts, **filtered_inst)
            filtered_rew["instincts"] = new_instincts

        new_rewards = replace(lab_config.rewards, **filtered_rew)
        config = replace(lab_config, environment=new_env, rewards=new_rewards)

    env = SheepdogEnvironment(config)
    env.reset(seed=42)
    obs = env.build_observation_for_dog(0)
    names = list(obs.feature_names)
    return hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest()


def get_action_space_hash() -> str:
    import hashlib
    import json
    from sheepdog.environment import ACTION_ORDER
    return hashlib.sha256(json.dumps(list(ACTION_ORDER)).encode("utf-8")).hexdigest()


def verify_checkpoint_compatibility(checkpoint_metadata: dict[str, Any], current_config: Any) -> dict[str, Any]:
    from sheepdog.rewards import REWARD_SCHEMA_VERSION
    from sheepdog.environment import ENV_CONFIG_VERSION

    current_obs_hash = get_observation_schema_hash(current_config)
    current_action_hash = get_action_space_hash()
    current_reward_version = REWARD_SCHEMA_VERSION
    current_env_version = ENV_CONFIG_VERSION

    cp_obs_hash = checkpoint_metadata.get("observation_schema_hash")
    cp_action_hash = checkpoint_metadata.get("action_space_hash")
    cp_reward_version = checkpoint_metadata.get("reward_schema_version")
    cp_env_version = checkpoint_metadata.get("env_config_version")

    # Gracefully compute hashes/versions for legacy checkpoints lacking metadata hashes
    if cp_obs_hash is None:
        try:
            cp_obs_hash = get_observation_schema_hash(checkpoint_metadata)
        except Exception:
            pass

    if cp_action_hash is None:
        cp_action_hash = current_action_hash

    if cp_reward_version is None:
        cp_reward_version = "1.0"

    if cp_env_version is None:
        cp_env_version = "1.0"

    errors = []
    if cp_obs_hash != current_obs_hash:
        errors.append(
            f"Observation schema mismatch (checkpoint: {cp_obs_hash or 'legacy'}, current: {current_obs_hash[:8]})"
        )
    if cp_action_hash != current_action_hash:
        errors.append(
            f"Action space mismatch (checkpoint: {cp_action_hash or 'legacy'}, current: {current_action_hash[:8]})"
        )
    if cp_reward_version != current_reward_version:
        errors.append(
            f"Reward schema version mismatch (checkpoint: {cp_reward_version or 'legacy'}, current: {current_reward_version})"
        )
    if cp_env_version != current_env_version:
        errors.append(
            f"Environment config version mismatch (checkpoint: {cp_env_version or 'legacy'}, current: {current_env_version})"
        )

    # Validate critical environment structural parameters
    cp_env = checkpoint_metadata.get("environment_config", {})
    if cp_env:
        cp_dogs = cp_env.get("dogs")
        current_dogs = current_config.environment.dogs
        if cp_dogs is not None and cp_dogs != current_dogs:
            errors.append(f"Environment dogs mismatch (checkpoint: {cp_dogs}, current: {current_dogs})")

        cp_sheep = cp_env.get("sheep")
        current_sheep = current_config.environment.sheep
        if cp_sheep is not None and cp_sheep != current_sheep:
            errors.append(f"Environment sheep mismatch (checkpoint: {cp_sheep}, current: {current_sheep})")

        cp_width = cp_env.get("width")
        current_width = current_config.environment.width
        if cp_width is not None and cp_width != current_width:
            errors.append(f"Environment width mismatch (checkpoint: {cp_width}, current: {current_width})")

        cp_height = cp_env.get("height")
        current_height = current_config.environment.height
        if cp_height is not None and cp_height != current_height:
            errors.append(f"Environment height mismatch (checkpoint: {cp_height}, current: {current_height})")

    return {
        "compatible": len(errors) == 0,
        "errors": errors,
        "current": {
            "observation_schema_hash": current_obs_hash,
            "action_space_hash": current_action_hash,
            "reward_schema_version": current_reward_version,
            "env_config_version": current_env_version,
        },
        "checkpoint": {
            "observation_schema_hash": cp_obs_hash,
            "action_space_hash": cp_action_hash,
            "reward_schema_version": cp_reward_version,
            "env_config_version": cp_env_version,
        },
    }
