"""Small neural-network policy wrapper for the experimental PPO path."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import uuid
import zipfile

from sheepdog.atomic_io import atomic_replace
from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.policies.base import Action
from sheepdog.training.rl_env import SheepdogRLAdapter

LOGGER = logging.getLogger(__name__)


def tensorboard_available() -> bool:
    """Return whether Stable-Baselines TensorBoard logging can be used."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False
    return SummaryWriter is not None


def _make_rl_adapter(config_payload: dict[str, Any]) -> SheepdogRLAdapter:
    """Build one RL adapter from a serializable config payload."""
    return SheepdogRLAdapter(LabConfig.from_dict(config_payload))


def _resolve_env_workers(config: LabConfig) -> int:
    """Resolve PPO worker count from config and optional env override."""
    raw_override = os.getenv("SHEEPDOG_PPO_NUM_ENVS")
    if raw_override is not None:
        try:
            return max(1, int(raw_override))
        except ValueError:
            pass

    # Windows process spawning under multithreaded server can fail or pipe-break (WinError 232).
    # Use a single in-process env (DummyVecEnv) on Windows to keep server runs stable.
    if os.name == "nt":
        return 1

    return max(1, int(config.training.ppo_env_workers))


def _build_vec_env(config: LabConfig) -> tuple[VecEnv, int, str]:
    """Create a vectorized env with the selected backend and worker count."""
    workers = _resolve_env_workers(config)
    config_payload = config.to_dict()
    env_fns = [partial(_make_rl_adapter, config_payload) for _ in range(workers)]
    if workers == 1:
        LOGGER.info("PPO vector environment: workers=%d backend=dummy", workers)
        return DummyVecEnv(env_fns), workers, "dummy"
    # Use spawn to stay Windows-safe when called from threaded server code.
    LOGGER.info("PPO vector environment: workers=%d backend=subproc", workers)
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
            return cls(hidden_sizes=(128, 128, 128), observation_size=observation_size)
        hidden_sizes = payload.get("hidden_sizes", (128, 128, 128))
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
        self.policy_version: int | None = None

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
        has_tensorboard = tensorboard_available()
        tb_dir = (Path(config.training.output_dir) / "tb_logs") if config.training.output_dir and has_tensorboard else None
        if tb_dir:
            tb_dir.mkdir(parents=True, exist_ok=True)

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
            tensorboard_log=str(tb_dir) if tb_dir else None,
        )
        return cls(model=model, model_path=Path(""), config=policy_config)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: LabConfig,
        policy_config: dict[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> NeuralPolicy:
        """Load a trained neural policy from a checkpoint file."""
        vec_env, _env_workers, _backend = _build_vec_env(config)
        observation_size = int(vec_env.observation_space.shape[0])
        resolved_path = Path(path)
        model = MaskablePPO.load(str(resolved_path), env=vec_env)
        if tensorboard_available() and config.training.output_dir:
            tb_dir = Path(config.training.output_dir) / "tb_logs"
            tb_dir.mkdir(parents=True, exist_ok=True)
            model.tensorboard_log = str(tb_dir)
        else:
            model.tensorboard_log = None
        loaded_pconfig = NeuralPolicyConfig.from_dict(policy_config, observation_size)
        expected_arch = tuple(config.training.neural_hidden_sizes)
        if loaded_pconfig.hidden_sizes != expected_arch:
            raise ValueError(
                f"Incompatible model architecture: checkpoint hidden layers {list(loaded_pconfig.hidden_sizes)} "
                f"do not match target configuration {list(expected_arch)}"
            )
        if hasattr(model, "policy_kwargs") and isinstance(model.policy_kwargs, dict):
            net_arch = model.policy_kwargs.get("net_arch")
            if isinstance(net_arch, list) and tuple(net_arch) != expected_arch:
                raise ValueError(
                    f"Incompatible model architecture: model net_arch {net_arch} "
                    f"does not match target configuration {list(expected_arch)}"
                )
        policy_obj = cls(
            model=model,
            model_path=resolved_path,
            config=loaded_pconfig,
        )
        policy_obj.policy_version = policy_version
        return policy_obj

    def save(self, path: str | Path) -> Path:
        """Save the model to *path* atomically and return the resolved path."""
        target = Path(path)
        resolved_target = target if target.suffix == ".zip" else target.with_suffix(".zip")
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to a temporary location first
        tmp_stem = f"{resolved_target.stem}-{uuid.uuid4().hex[:8]}.tmp"
        tmp_target = resolved_target.parent / tmp_stem
        self.model.save(str(tmp_target))
        tmp_zip = tmp_target.with_suffix(".zip") if not str(tmp_target).endswith(".zip") else tmp_target
        if not tmp_zip.exists() and tmp_target.exists():
            tmp_zip = tmp_target

        if not tmp_zip.exists():
            raise FileNotFoundError(f"Failed to create temporary model file at {tmp_zip}")

        # Validate zip file integrity before replacing target
        try:
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise ValueError(f"Corrupt file in zip archive: {bad_file}")
            MaskablePPO.load(str(tmp_zip), device="cpu")
        except Exception as exc:
            if tmp_zip.exists():
                try:
                    os.remove(tmp_zip)
                except Exception:
                    pass
            raise RuntimeError(f"Model checkpoint zip validation failed: {exc}") from exc

        atomic_replace(tmp_zip, resolved_target)
        self.model_path = resolved_target
        return resolved_target

    def select_actions(self, environment: object, deterministic: bool = True) -> list[Action]:
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
                deterministic=deterministic,
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
