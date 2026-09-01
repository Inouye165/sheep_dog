"""Centralized MaskablePPO policy for simultaneous dog-team decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.observations import HERD_DOG_SLOTS
from sheepdog.policies.base import Action
from sheepdog.policies.neural import NeuralPolicy, _resolve_env_workers, tensorboard_available
from sheepdog.training.team_rl_env import (
    TeamActionRLEnv,
    build_team_action_mask,
    build_team_observation,
)

LOGGER = logging.getLogger(__name__)


def _make_team_rl_adapter(config_payload: dict[str, Any]) -> TeamActionRLEnv:
    """Build one team-step adapter from serializable configuration."""
    return TeamActionRLEnv(LabConfig.from_dict(config_payload))


def _build_team_vec_env(config: LabConfig) -> tuple[VecEnv, int, str]:
    """Create vectorized true-team environments for MaskablePPO."""
    workers = _resolve_env_workers(config)
    config_payload = config.to_dict()
    env_fns = [partial(_make_team_rl_adapter, config_payload) for _ in range(workers)]
    if workers == 1:
        LOGGER.info("Joint PPO vector environment: workers=%d backend=dummy", workers)
        return DummyVecEnv(env_fns), workers, "dummy"
    LOGGER.info("Joint PPO vector environment: workers=%d backend=subproc", workers)
    return SubprocVecEnv(env_fns, start_method="spawn"), workers, "subproc"


@dataclass(frozen=True, slots=True)
class JointTeamPolicyConfig:
    """Architecture metadata stored with joint-team checkpoints."""

    hidden_sizes: tuple[int, ...]
    observation_size: int
    action_sizes: tuple[int, ...] = (len(ACTION_ORDER),) * HERD_DOG_SLOTS
    env_workers: int = 1
    architecture: str = "joint_team_v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this policy configuration."""
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "observation_size": self.observation_size,
            "action_sizes": list(self.action_sizes),
            "env_workers": self.env_workers,
            "architecture": self.architecture,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        observation_size: int,
    ) -> JointTeamPolicyConfig:
        """Reconstruct joint policy metadata from persisted state."""
        if not payload:
            return cls(hidden_sizes=(128, 128, 128), observation_size=observation_size)
        return cls(
            hidden_sizes=tuple(
                int(value) for value in payload.get("hidden_sizes", (128, 128, 128))
            ),
            observation_size=int(payload.get("observation_size", observation_size)),
            action_sizes=tuple(int(value) for value in payload.get("action_sizes", ())),
            env_workers=max(1, int(payload.get("env_workers", 1))),
            architecture=str(payload.get("architecture", "")),
        )


class JointTeamPolicy(NeuralPolicy):
    """Select every active dog's action from one centralized observation."""

    name = "joint_team_policy"
    trainer_type = "joint_maskable_ppo"
    policy_type = "neural"

    def __init__(
        self,
        model: MaskablePPO,
        model_path: Path,
        config: JointTeamPolicyConfig,
    ) -> None:
        super().__init__(model=model, model_path=model_path, config=config)  # type: ignore[arg-type]
        self.config = config

    @classmethod
    def initialize(cls, config: LabConfig) -> JointTeamPolicy:
        """Create a fresh joint-team policy and its vector environments."""
        vec_env, env_workers, _backend = _build_team_vec_env(config)
        policy_config = JointTeamPolicyConfig(
            hidden_sizes=tuple(config.training.neural_hidden_sizes),
            observation_size=int(vec_env.observation_space.shape[0]),
            env_workers=env_workers,
        )
        has_tensorboard = tensorboard_available()
        tensorboard_dir = (
            Path(config.training.output_dir) / "tb_logs" / "joint_team"
            if config.training.output_dir and has_tensorboard
            else None
        )
        if tensorboard_dir is not None:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.training.learning_rate,
            n_steps=config.training.rollout_steps,
            batch_size=config.training.batch_size,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_range=config.training.clip_range,
            ent_coef=config.training.entropy_coef,
            vf_coef=config.training.value_coef,
            target_kl=getattr(config.training, "target_kl", None),
            seed=config.training.train_seed,
            policy_kwargs={"net_arch": list(policy_config.hidden_sizes)},
            verbose=0,
            tensorboard_log=str(tensorboard_dir) if tensorboard_dir else None,
        )
        return cls(model=model, model_path=Path(""), config=policy_config)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: LabConfig,
        policy_config: dict[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> JointTeamPolicy:
        """Load a joint checkpoint after validating its architecture metadata."""
        vec_env, _env_workers, _backend = _build_team_vec_env(config)
        observation_size = int(vec_env.observation_space.shape[0])
        loaded_config = JointTeamPolicyConfig.from_dict(policy_config, observation_size)
        expected_hidden_sizes = tuple(config.training.neural_hidden_sizes)
        expected_action_sizes = (len(ACTION_ORDER),) * HERD_DOG_SLOTS
        if loaded_config.architecture != "joint_team_v1":
            raise ValueError("Checkpoint is not a joint-team policy")
        if loaded_config.hidden_sizes != expected_hidden_sizes:
            raise ValueError(
                "Incompatible joint model architecture: checkpoint hidden layers "
                f"{list(loaded_config.hidden_sizes)} do not match target configuration "
                f"{list(expected_hidden_sizes)}"
            )
        if loaded_config.observation_size != observation_size:
            raise ValueError("Checkpoint joint observation size is incompatible")
        if loaded_config.action_sizes != expected_action_sizes:
            raise ValueError("Checkpoint joint action heads are incompatible")

        resolved_path = Path(path)
        model = MaskablePPO.load(str(resolved_path), env=vec_env)
        model.batch_size = config.training.batch_size
        if tensorboard_available() and config.training.output_dir:
            tensorboard_dir = Path(config.training.output_dir) / "tb_logs" / "joint_team"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            model.tensorboard_log = str(tensorboard_dir)
        else:
            model.tensorboard_log = None
        policy = cls(model=model, model_path=resolved_path, config=loaded_config)
        policy.policy_version = policy_version
        return policy

    def select_actions(self, environment: object, deterministic: bool = True) -> list[Action]:
        """Return one action per active dog from one joint model prediction."""
        if hasattr(environment, "prepare_policy_step"):
            environment.prepare_policy_step()
        observation = build_team_observation(environment)  # type: ignore[arg-type]
        action_mask = build_team_action_mask(environment)  # type: ignore[arg-type]
        action_indices, _state = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=deterministic,
        )
        flattened_indices = np.asarray(action_indices, dtype=np.int64).reshape(-1)
        return [
            ACTION_ORDER[int(flattened_indices[dog_index])]
            for dog_index in range(environment.dog_count)  # type: ignore[attr-defined]
        ]
