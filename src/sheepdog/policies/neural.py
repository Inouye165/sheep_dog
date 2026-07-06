"""Small neural-network policy wrapper for the experimental PPO path."""

from __future__ import annotations

import os
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.policies.base import Action
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_rl_adapter(config_payload: dict[str, Any]) -> SheepdogRLAdapter:
    """Build one RL adapter from a serializable config payload."""
    return SheepdogRLAdapter(LabConfig.from_dict(config_payload))


def _resolve_env_workers(config: LabConfig) -> int:
    """Resolve PPO worker count from config and optional env override."""
    # Python 3.13 on Windows can deadlock while worker processes import SB3
    # modules under spawned subprocess startup. Use a single in-process env
    # to keep training progress reliable for interactive server runs.
    if os.name == "nt" and sys.version_info >= (3, 13):
        return 1

    raw_override = os.getenv("SHEEPDOG_PPO_NUM_ENVS")
    if raw_override is not None:
        try:
            return max(1, int(raw_override))
        except ValueError:
            pass
    return max(1, int(config.training.ppo_env_workers))


def _build_vec_env(config: LabConfig) -> tuple[VecEnv, int, str]:
    """Create a vectorized env with the selected backend and worker count."""
    workers = _resolve_env_workers(config)
    config_payload = config.to_dict()
    env_fns = [partial(_make_rl_adapter, config_payload) for _ in range(workers)]
    if workers == 1:
        return DummyVecEnv(env_fns), workers, "dummy"
    # Use spawn to stay Windows-safe when called from threaded server code.
    return SubprocVecEnv(env_fns, start_method="spawn"), workers, "subproc"


@dataclass(frozen=True, slots=True)
class NeuralPolicyConfig:
    """Neural policy metadata stored with checkpoints."""

    hidden_sizes: tuple[int, ...]
    observation_size: int
    action_size: int = len(ACTION_ORDER)
    env_workers: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "env_workers": self.env_workers,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, observation_size: int) -> NeuralPolicyConfig:
        """Reconstruct a NeuralPolicyConfig from persisted state."""
        if not payload:
            return cls(hidden_sizes=(64, 64), observation_size=observation_size)
        hidden_sizes = payload.get("hidden_sizes", (64, 64))
        return cls(
            hidden_sizes=tuple(int(value) for value in hidden_sizes),
            observation_size=int(payload.get("observation_size", observation_size)),
            action_size=int(payload.get("action_size", len(ACTION_ORDER))),
            env_workers=max(1, int(payload.get("env_workers", 1))),
        )


class NeuralPolicy:
    """Shared role-aware neural policy powered by MaskablePPO."""

    name = "neural_policy"
    trainer_type = "maskable_ppo"
    policy_type = "neural"

    def __init__(self, model: MaskablePPO, model_path: Path, config: NeuralPolicyConfig) -> None:
        self.model = model
        self.model_path = Path(model_path)
        self.config = config

    @classmethod
    def initialize(cls, config: LabConfig) -> NeuralPolicy:
        """Create a fresh, untrained neural policy with vectorized environments."""
        vec_env, env_workers, _backend = _build_vec_env(config)
        observation_size = int(vec_env.observation_space.shape[0])
        policy_config = NeuralPolicyConfig(
            hidden_sizes=tuple(config.training.neural_hidden_sizes),
            observation_size=observation_size,
            env_workers=env_workers,
        )
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
            seed=config.training.train_seed,
            policy_kwargs={"net_arch": list(policy_config.hidden_sizes)},
            verbose=0,
        )
        return cls(model=model, model_path=Path(""), config=policy_config)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: LabConfig,
        policy_config: dict[str, Any] | None = None,
    ) -> NeuralPolicy:
        """Load a trained neural policy from a checkpoint file."""
        vec_env, _env_workers, _backend = _build_vec_env(config)
        observation_size = int(vec_env.observation_space.shape[0])
        resolved_path = Path(path)
        model = MaskablePPO.load(str(resolved_path), env=vec_env)
        return cls(
            model=model,
            model_path=resolved_path,
            config=NeuralPolicyConfig.from_dict(policy_config, observation_size),
        )

    def save(self, path: str | Path) -> Path:
        """Save the model to *path* and return the resolved path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target))
        resolved_target = target if target.suffix == ".zip" else target.with_suffix(".zip")
        self.model_path = resolved_target
        return resolved_target

    def select_actions(self, environment: object) -> list[Action]:
        """Return one action per dog based on the neural model's predictions."""
        actions: list[Action] = []
        reserved_positions: set[object] = set()
        if hasattr(environment, "prepare_policy_step"):
            environment.prepare_policy_step()
        for dog_index in range(environment.dog_count):
            observation = np.asarray(
                environment.build_observation_for_dog(dog_index).values,
                dtype=np.float32,
            )
            mask_map = environment.action_mask_for_dog(
                dog_index,
                reserved_positions=reserved_positions,
            )
            action_masks = np.asarray([mask_map[action] for action in ACTION_ORDER], dtype=bool)
            action_index, _state = self.model.predict(
                observation,
                action_masks=action_masks,
                deterministic=True,
            )
            action = ACTION_ORDER[int(action_index)]
            actions.append(action)
            reserved_positions.add(environment.project_dog_action(dog_index, action))
        return actions

    def score_actions(self, observation: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Return masked action scores for inspection-oriented tooling."""

        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observation_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(
            observation_tensor,
            action_masks=action_mask.reshape(1, -1),
        )
        logits = distribution.distribution.logits.detach().cpu().numpy().reshape(-1)
        logits[~action_mask.astype(bool)] = -np.inf
        return logits
