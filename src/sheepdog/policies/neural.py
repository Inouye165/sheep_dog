"""Small neural-network policy wrapper for the experimental PPO path."""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
import zipfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch import nn

from sheepdog.atomic_io import atomic_replace
from sheepdog.config import LabConfig, validate_observation_history_strides
from sheepdog.environment import ACTION_ORDER
from sheepdog.observations import (
    EMERGENT_GATE_OBSERVATION_SCHEMA_VERSION,
    GATE_OBSERVATION_SCHEMA_VERSION,
)
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


def _make_rl_adapter(
    config_payload: dict[str, Any],
    worker_index: int = 0,
    workers: int = 1,
) -> SheepdogRLAdapter:
    """Build one RL adapter from a serializable config payload."""
    if workers > 1 and worker_index >= 0:
        # Prevent OpenMP/PyTorch thread oversubscription across worker processes.
        torch.set_num_threads(1)
        payload = dict(config_payload)
        training_payload = dict(payload.get("training", {}))
        base_train_seed = int(training_payload.get("train_seed", 0))
        training_payload["train_seed"] = base_train_seed + worker_index * 100_000
        payload["training"] = training_payload
        config = LabConfig.from_dict(payload)
    else:
        config = LabConfig.from_dict(config_payload)
    adapter = SheepdogRLAdapter(config)
    adapter.env_index = int(worker_index)
    return adapter


def _resolve_env_workers(config: LabConfig) -> int:
    """Resolve PPO worker count from config and optional env override."""
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
    if workers == 1:
        env_fns = [partial(_make_rl_adapter, config_payload, 0, 1)]
        LOGGER.info("PPO vector environment: workers=%d backend=dummy", workers)
        return DummyVecEnv(env_fns), workers, "dummy"
    env_fns = [
        partial(_make_rl_adapter, config_payload, worker_index, workers)
        for worker_index in range(workers)
    ]
    # Use spawn to stay Windows-safe when called from threaded server code.
    LOGGER.info("PPO vector environment: workers=%d backend=subproc", workers)
    try:
        return SubprocVecEnv(env_fns, start_method="spawn"), workers, "subproc"
    except (OSError, RuntimeError, EOFError, MemoryError, BrokenPipeError) as exc:
        LOGGER.warning(
            "SubprocVecEnv(workers=%d) failed to start (%s); falling back to DummyVecEnv(workers=1).",
            workers,
            exc,
        )
        fallback_fns = [partial(_make_rl_adapter, config_payload, 0, 1)]
        return DummyVecEnv(fallback_fns), 1, "dummy"


def compute_observation_schema_version(
    observation_mode: str,
    strides: tuple[int, ...] = (0,),
    include_loop_flags: bool = True,
) -> str | None:
    """Compute the expected observation schema version for mode, strides, and loop flags."""
    validated_strides = validate_observation_history_strides(strides)
    if observation_mode in {"guided_gate_control", "guided_gate_treatment"}:
        base: str | None = GATE_OBSERVATION_SCHEMA_VERSION
    elif observation_mode == "emergent_gate":
        base = EMERGENT_GATE_OBSERVATION_SCHEMA_VERSION
    else:
        base = None

    if validated_strides == (0,) and include_loop_flags:
        return base

    version = base if base is not None else observation_mode
    if validated_strides != (0,):
        version = f"{version}|strides={','.join(str(s) for s in validated_strides)}"
    if not include_loop_flags:
        version = f"{version}|no_loop_flags"
    return version


@dataclass(frozen=True, slots=True)
class NeuralPolicyConfig:
    """Neural policy metadata stored with checkpoints."""

    hidden_sizes: tuple[int, ...]
    observation_size: int
    action_size: int = len(ACTION_ORDER)
    env_workers: int = 1
    observation_mode: str | None = None
    observation_schema_version: str | None = None
    observation_history_strides: tuple[int, ...] = (0,)
    include_loop_flags_in_observation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        payload: dict[str, Any] = {
            "hidden_sizes": list(self.hidden_sizes),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "env_workers": self.env_workers,
            "observation_history_strides": list(self.observation_history_strides),
            "include_loop_flags_in_observation": self.include_loop_flags_in_observation,
        }
        if self.observation_mode is not None:
            payload["observation_mode"] = self.observation_mode
        if self.observation_schema_version is not None:
            payload["observation_schema_version"] = self.observation_schema_version
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        observation_size: int,
        default_hidden_sizes: tuple[int, ...] | None = None,
    ) -> NeuralPolicyConfig:
        """Reconstruct a NeuralPolicyConfig from persisted state."""
        fallback = default_hidden_sizes if default_hidden_sizes is not None else LabConfig().training.neural_hidden_sizes
        if not payload:
            return cls(hidden_sizes=tuple(fallback), observation_size=observation_size)
        hidden_sizes = payload.get("hidden_sizes", fallback)
        raw_obs_mode = payload.get("observation_mode")
        raw_schema_version = payload.get("observation_schema_version")
        raw_strides = payload.get("observation_history_strides")
        strides = (
            validate_observation_history_strides(list(raw_strides))
            if raw_strides is not None
            else (0,)
        )
        include_loop_flags = bool(payload.get("include_loop_flags_in_observation", True))
        return cls(
            hidden_sizes=tuple(int(value) for value in hidden_sizes),
            observation_size=int(payload.get("observation_size", observation_size)),
            action_size=int(payload.get("action_size", len(ACTION_ORDER))),
            env_workers=max(1, int(payload.get("env_workers", 1))),
            observation_mode=str(raw_obs_mode) if raw_obs_mode is not None else None,
            observation_schema_version=(
                str(raw_schema_version) if raw_schema_version is not None else None
            ),
            observation_history_strides=strides,
            include_loop_flags_in_observation=include_loop_flags,
        )


class NeuralPolicy:
    """Shared role-aware neural policy powered by MaskablePPO."""

    name = "neural_policy"
    trainer_type = "maskable_ppo"
    policy_type = "neural"
    discount_timebase = "sequential_shared_dog"

    def __init__(
        self,
        model: MaskablePPO,
        model_path: Path,
        config: NeuralPolicyConfig,
        *,
        checkpoint_gamma: float | None = None,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path)
        self.config = config
        self.policy_version: int | None = None
        self.latest_decision_telemetry: list[dict[str, Any]] | None = None
        self._closed = False
        if checkpoint_gamma is not None:
            self._checkpoint_gamma: float | None = float(checkpoint_gamma)
        elif hasattr(model, "gamma") and isinstance(model.gamma, int | float):
            self._checkpoint_gamma = float(model.gamma)
        else:
            self._checkpoint_gamma = None

    def close(self) -> None:
        """Close the underlying vectorized environment if open."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        model = getattr(self, "model", None)
        if model is None:
            return
        env = getattr(model, "env", None)
        if env is None and hasattr(model, "get_env"):
            with contextlib.suppress(Exception):
                env = model.get_env()
        if env is not None and hasattr(env, "close"):
            with contextlib.suppress(Exception):
                env.close()

    @property
    def checkpoint_gamma(self) -> float | None:
        """Return the training discount factor gamma stored with the model."""
        if self._checkpoint_gamma is not None:
            return self._checkpoint_gamma
        if hasattr(self.model, "gamma") and isinstance(self.model.gamma, int | float):
            return float(self.model.gamma)
        return None

    def live_training_parameters(self) -> dict[str, Any]:
        """Return a snapshot of the live SB3 model, rollout buffer, and optimizer parameters."""
        model = self.model
        rollout_buffer = getattr(model, "rollout_buffer", None)
        optimizer_lrs: list[float] = []
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            optimizer_lrs = [
                float(group.get("lr", 0.0))
                for group in getattr(model.policy.optimizer, "param_groups", ())
            ]
        return {
            "gamma": float(model.gamma) if hasattr(model, "gamma") and isinstance(model.gamma, int | float) else None,
            "rollout_buffer_gamma": (
                float(rollout_buffer.gamma)
                if rollout_buffer is not None and hasattr(rollout_buffer, "gamma") and isinstance(rollout_buffer.gamma, int | float)
                else None
            ),
            "gae_lambda": (
                float(model.gae_lambda)
                if hasattr(model, "gae_lambda") and isinstance(model.gae_lambda, int | float)
                else None
            ),
            "rollout_buffer_gae_lambda": (
                float(rollout_buffer.gae_lambda)
                if rollout_buffer is not None and hasattr(rollout_buffer, "gae_lambda") and isinstance(rollout_buffer.gae_lambda, int | float)
                else None
            ),
            "ent_coef": (
                float(model.ent_coef)
                if hasattr(model, "ent_coef") and isinstance(model.ent_coef, int | float)
                else None
            ),
            "vf_coef": (
                float(model.vf_coef)
                if hasattr(model, "vf_coef") and isinstance(model.vf_coef, int | float)
                else None
            ),
            "batch_size": (
                int(model.batch_size)
                if hasattr(model, "batch_size") and isinstance(model.batch_size, int | float)
                else None
            ),
            "n_steps": (
                int(model.n_steps)
                if hasattr(model, "n_steps") and isinstance(model.n_steps, int | float)
                else None
            ),
            "target_kl": (
                float(model.target_kl)
                if hasattr(model, "target_kl") and isinstance(model.target_kl, int | float)
                else None
            ),
            "learning_rate": (
                float(model.learning_rate)
                if hasattr(model, "learning_rate") and isinstance(model.learning_rate, int | float)
                else None
            ),
            "optimizer_learning_rates": optimizer_lrs,
            "seed": (
                int(model.seed)
                if hasattr(model, "seed") and isinstance(model.seed, int)
                else None
            ),
            "rollout_buffer_pos": (
                int(rollout_buffer.pos)
                if rollout_buffer is not None and hasattr(rollout_buffer, "pos") and isinstance(rollout_buffer.pos, int)
                else None
            ),
            "rollout_buffer_full": (
                bool(rollout_buffer.full)
                if rollout_buffer is not None and hasattr(rollout_buffer, "full")
                else None
            ),
            "n_updates": (
                int(model._n_updates)
                if hasattr(model, "_n_updates") and isinstance(model._n_updates, int)
                else None
            ),
            "num_timesteps": (
                int(model.num_timesteps)
                if hasattr(model, "num_timesteps") and isinstance(model.num_timesteps, int)
                else None
            ),
        }

    def configure_for_training(
        self,
        config: LabConfig,
        *,
        effective_learning_rate: float | None = None,
        effective_entropy_coef: float | None = None,
        reset_rollout_buffer: bool = True,
        set_seed: bool = False,
    ) -> dict[str, Any]:
        """Apply training hyperparameters explicitly to a loaded or initialized SB3 model.

        Stable-Baselines3 ``MaskablePPO.load()`` restores ``gamma``, ``gae_lambda``,
        ``ent_coef``, ``learning_rate``, ``seed``, and ``rollout_buffer`` parameters
        from the checkpoint archive. When resuming training under controlled
        hyperparameter overrides, both the model attributes and the live
        ``rollout_buffer``/optimizer state must be updated before ``learn()``.
        """
        train_cfg = config.training
        gamma = float(train_cfg.gamma)
        gae_lambda = float(train_cfg.gae_lambda)
        ent_coef = float(
            effective_entropy_coef
            if effective_entropy_coef is not None
            else train_cfg.entropy_coef
        )
        vf_coef = float(train_cfg.value_coef)
        batch_size = int(train_cfg.batch_size)
        target_kl = getattr(train_cfg, "target_kl", None)
        learning_rate = float(
            effective_learning_rate
            if effective_learning_rate is not None
            else train_cfg.learning_rate
        )

        self.model.gamma = gamma
        self.model.gae_lambda = gae_lambda
        self.model.ent_coef = ent_coef
        self.model.vf_coef = vf_coef
        self.model.batch_size = batch_size
        self.model.target_kl = float(target_kl) if target_kl is not None else None
        self.model.learning_rate = learning_rate
        self.model.lr_schedule = lambda _, _lr=learning_rate: _lr

        if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
            for param_group in getattr(self.model.policy.optimizer, "param_groups", ()):
                param_group["lr"] = learning_rate

        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is not None:
            if hasattr(rollout_buffer, "gamma"):
                rollout_buffer.gamma = gamma
            if hasattr(rollout_buffer, "gae_lambda"):
                rollout_buffer.gae_lambda = gae_lambda
            if reset_rollout_buffer and hasattr(rollout_buffer, "reset"):
                rollout_buffer.reset()
        if reset_rollout_buffer and hasattr(self.model, "_last_obs"):
            self.model._last_obs = None

        if set_seed:
            train_seed = int(train_cfg.train_seed)
            self.model.seed = train_seed
            if hasattr(self.model, "set_random_seed"):
                self.model.set_random_seed(train_seed)

        self._checkpoint_gamma = gamma
        return self.live_training_parameters()

    @classmethod
    def initialize(cls, config: LabConfig) -> NeuralPolicy:
        """Create a fresh, untrained neural policy with vectorized environments."""
        torch.set_num_threads(max(1, int(config.training.torch_num_threads)))
        vec_env, env_workers, _backend = _build_vec_env(config)
        observation_size = int(vec_env.observation_space.shape[0])
        obs_mode = config.training.observation_mode
        obs_strides = validate_observation_history_strides(
            config.training.observation_history_strides
        )
        include_loop_flags = bool(config.training.include_loop_flags_in_observation)
        schema_version = compute_observation_schema_version(
            obs_mode,
            obs_strides,
            include_loop_flags,
        )
        policy_config = NeuralPolicyConfig(
            hidden_sizes=tuple(config.training.neural_hidden_sizes),
            observation_size=observation_size,
            env_workers=env_workers,
            observation_mode=obs_mode,
            observation_schema_version=schema_version,
            observation_history_strides=obs_strides,
            include_loop_flags_in_observation=include_loop_flags,
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
            target_kl=getattr(config.training, "target_kl", None),
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
        torch.set_num_threads(max(1, int(config.training.torch_num_threads)))
        vec_env, _env_workers, _backend = _build_vec_env(config)
        try:
            observation_size = int(vec_env.observation_space.shape[0])
            resolved_path = Path(path)
            model = MaskablePPO.load(str(resolved_path), env=vec_env)
            model.batch_size = config.training.batch_size
            if tensorboard_available() and config.training.output_dir:
                tb_dir = Path(config.training.output_dir) / "tb_logs"
                tb_dir.mkdir(parents=True, exist_ok=True)
                model.tensorboard_log = str(tb_dir)
            else:
                model.tensorboard_log = None
            loaded_pconfig = NeuralPolicyConfig.from_dict(
                policy_config,
                observation_size,
                default_hidden_sizes=tuple(config.training.neural_hidden_sizes),
            )
            loaded_pconfig = replace(loaded_pconfig, env_workers=_env_workers)
            model_obs_shape = getattr(model.observation_space, "shape", None)
            if not model_obs_shape or int(model_obs_shape[0]) != observation_size:
                raise ValueError(
                    f"Incompatible observation size: model expects {model_obs_shape}, "
                    f"target configuration expects ({observation_size},)"
                )
            if loaded_pconfig.observation_size != observation_size:
                raise ValueError(
                    f"Incompatible policy_config observation_size: "
                    f"{loaded_pconfig.observation_size} != {observation_size}"
                )
            target_obs_mode = config.training.observation_mode
            target_strides = validate_observation_history_strides(
                config.training.observation_history_strides
            )
            target_loop_flags = bool(config.training.include_loop_flags_in_observation)
            if (
                loaded_pconfig.observation_mode is not None
                and loaded_pconfig.observation_mode != target_obs_mode
            ):
                raise ValueError(
                    f"Incompatible observation_mode: checkpoint policy_config has "
                    f"{loaded_pconfig.observation_mode!r}, target configuration has {target_obs_mode!r}"
                )
            if loaded_pconfig.observation_history_strides != target_strides:
                raise ValueError(
                    f"Incompatible observation_history_strides: checkpoint policy_config has "
                    f"{loaded_pconfig.observation_history_strides!r}, target configuration has {target_strides!r}"
                )
            if loaded_pconfig.include_loop_flags_in_observation != target_loop_flags:
                raise ValueError(
                    f"Incompatible include_loop_flags_in_observation: checkpoint policy_config has "
                    f"{loaded_pconfig.include_loop_flags_in_observation!r}, target configuration has {target_loop_flags!r}"
                )
            expected_schema_version = compute_observation_schema_version(
                target_obs_mode,
                target_strides,
                target_loop_flags,
            )
            if (
                loaded_pconfig.observation_schema_version is not None
                and expected_schema_version is not None
                and loaded_pconfig.observation_schema_version != expected_schema_version
            ):
                raise ValueError(
                    f"Incompatible observation_schema_version: checkpoint policy_config has "
                    f"{loaded_pconfig.observation_schema_version!r}, expected {expected_schema_version!r}"
                )
            if (
                loaded_pconfig.observation_schema_version is not None
                and expected_schema_version is None
            ):
                raise ValueError(
                    f"Incompatible observation_schema_version: checkpoint policy_config has "
                    f"{loaded_pconfig.observation_schema_version!r}, incompatible with {target_obs_mode!r}"
                )
            if (
                policy_config is not None
                and loaded_pconfig.observation_schema_version is None
                and expected_schema_version == EMERGENT_GATE_OBSERVATION_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"Incompatible observation_schema_version: checkpoint policy_config has None, "
                    f"expected {expected_schema_version!r}"
                )
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
        except Exception:
            with contextlib.suppress(Exception):
                vec_env.close()
            raise

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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if tmp_zip.exists():
                with contextlib.suppress(OSError):
                    os.remove(tmp_zip)
            raise RuntimeError(f"Model checkpoint zip validation failed: {exc}") from exc

        atomic_replace(tmp_zip, resolved_target)
        self.model_path = resolved_target
        return resolved_target

    def reload_from(self, path: str | Path) -> bool:
        """Reload policy neural weights from checkpoint file without recreating environments."""
        resolved = Path(path)
        if resolved.suffix != ".zip":
            resolved = resolved.with_suffix(".zip")
        if not resolved.exists():
            return False
        try:
            loaded_model = MaskablePPO.load(str(resolved), env=self.model.get_env())
            if hasattr(self.model, "policy") and hasattr(loaded_model, "policy"):
                self.model.policy.load_state_dict(loaded_model.policy.state_dict())
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    def select_actions(self, environment: object, deterministic: bool = True) -> list[Action]:
        """Return one action per dog based on the neural model's predictions.

        Capture is completely passive: the identical inference path (self.model.predict)
        is executed regardless of whether capture is on or off.

        Telemetric Diagnostics & Equivalence Assumptions:
        - When capture_step_telemetry is True, policy logits, action probabilities, and value
          estimates are collected via a separate read-only diagnostic evaluation
          (model.policy.get_distribution and model.policy.predict_values) inside torch.no_grad().
        - These values are explicitly identified as "recomputed_diagnostic_forward_pass" rather
          than original decision outputs, because Stable-Baselines3 predict() returns only the
          action indices and does not expose internal intermediate tensors.
        - Equivalence assumption: For standard feedforward (MLP) architectures in eval mode,
          recomputed logits and value predictions are bitwise identical to the tensors evaluated
          inside predict(). For stochastic action selection, the action was chosen during the
          predict() call; recomputed probabilities reflect the categorical distribution but
          do not consume any additional random numbers.
        - If extraction fails (e.g. torch OOM or unsupported custom head), the exception is caught,
          recorded in capture_error, and the selected action is strictly preserved without reselection.
        - Telemetry payload size is bounded to fixed action-order dictionaries (9 actions max),
          ensuring storage and memory remain strictly bounded.
        """
        actions: list[Action] = []
        decision_records: list[dict[str, Any]] = []
        reserved_positions: set[object] = set()
        capture_telemetry = bool(getattr(environment, "capture_step_telemetry", False))
        if hasattr(environment, "prepare_policy_step") and (
            not hasattr(environment, "_uses_team_strategy")
            or environment._uses_team_strategy()
        ):
            environment.prepare_policy_step()
        for dog_index in range(environment.dog_count):
            obs_obj = environment.build_observation_for_dog(dog_index)
            observation = np.asarray(
                obs_obj.values,
                dtype=np.float32,
            )
            mask_map = environment.action_mask_for_dog(
                dog_index,
                reserved_positions=reserved_positions,
            )
            action_masks = np.asarray([mask_map[action] for action in ACTION_ORDER], dtype=bool)

            # Normal single-pass action selection: identical whether capture is on or off
            action_index, _state = self.model.predict(
                observation,
                action_masks=action_masks,
                deterministic=deterministic,
            )
            action = ACTION_ORDER[int(action_index)]
            actions.append(action)
            reserved_positions.add(environment.project_dog_action(dog_index, action))

            if capture_telemetry:
                logits_dict: dict[str, float] | None = None
                probs_dict: dict[str, float] | None = None
                value_est: float | None = None
                capture_error: str | None = None

                # Passive telemetry extraction: observe tensors from model.policy without sampling or RNG consumption
                if hasattr(self.model, "policy"):
                    try:
                        import torch

                        with torch.no_grad():
                            obs_tensor, _ = self.model.policy.obs_to_tensor(observation.reshape(1, -1))
                            masks_tensor = torch.as_tensor(
                                action_masks.reshape(1, -1),
                                device=self.model.policy.device,
                            )
                            distribution = self.model.policy.get_distribution(
                                obs_tensor, action_masks=masks_tensor
                            )
                            raw_logits = distribution.distribution.logits.detach().cpu().numpy().reshape(-1)
                            probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
                            logits_dict = {act: float(raw_logits[i]) for i, act in enumerate(ACTION_ORDER)}
                            probs_dict = {act: float(probs[i]) for i, act in enumerate(ACTION_ORDER)}
                            val_tensor = self.model.policy.predict_values(obs_tensor)
                            value_est = float(val_tensor.detach().cpu().numpy().reshape(-1)[0])
                    except Exception as ex:
                        capture_error = f"telemetry_extraction_failed: {type(ex).__name__}: {ex}"
                else:
                    capture_error = "unsupported_model_type_no_policy_attribute"

                physical_mask = environment.action_mask_for_dog(
                    dog_index,
                    policy_mode="neural_policy",
                    reserved_positions=reserved_positions,
                )
                decision_records.append({
                    "dog_index": dog_index,
                    "world_step": getattr(environment, "step_count", 0),
                    "observation": observation.tolist(),
                    "decision_action_mask": {act: bool(mask_map[act]) for act in ACTION_ORDER},
                    "recomputed_physical_mask": {
                        act: bool(physical_mask.get(act, False)) for act in ACTION_ORDER
                    },
                    "selected_action": action,
                    "executed_action": action,
                    "replacement_reason": None,
                    "action_probabilities": probs_dict,
                    "logits": logits_dict,
                    "value_estimate": value_est,
                    "action_probabilities_source": (
                        "recomputed_diagnostic_forward_pass" if probs_dict is not None else None
                    ),
                    "value_estimate_source": (
                        "recomputed_diagnostic_forward_pass" if value_est is not None else None
                    ),
                    "decision_mode": "deterministic" if deterministic else "sampled",
                    "policy_version": self.policy_version,
                    "checkpoint_id": (
                        str(self.model_path.stem)
                        if self.model_path and str(self.model_path) != "."
                        else None
                    ),
                    "mask_type": "actual_decision_time",
                    "capture_error": capture_error,
                })
        self.latest_decision_telemetry = decision_records if capture_telemetry else None
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


def _expand_linear_layer_inputs(
    layer: nn.Linear,
    old_dim: int,
    new_dim: int,
) -> nn.Linear:
    """Return a new Linear layer with *new_dim* inputs, preserving [:old_dim] and zero-padding [old_dim:new_dim]."""
    if layer.in_features != old_dim:
        raise ValueError(
            f"Expected Linear.in_features={old_dim}, got {layer.in_features}"
        )
    expanded = nn.Linear(
        in_features=new_dim,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :old_dim].copy_(layer.weight)
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias)
    return expanded


def _expand_extractor_linear_layers(
    module: nn.Module | None,
    prefix: str,
    old_dim: int,
    new_dim: int,
    expanded_names: list[str],
    seen_ids: set[int],
) -> None:
    """Update *_features_dim* and expand any first-layer Linear modules inside *module*."""
    if module is None or id(module) in seen_ids:
        return
    seen_ids.add(id(module))
    if getattr(module, "_features_dim", None) == old_dim:
        module._features_dim = new_dim
    for attr_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features == old_dim:
            setattr(module, attr_name, _expand_linear_layer_inputs(child, old_dim, new_dim))
            expanded_names.append(f"{prefix}.{attr_name}")
        elif isinstance(child, nn.Sequential):
            for idx, sub in enumerate(child):
                if isinstance(sub, nn.Linear) and sub.in_features == old_dim:
                    child[idx] = _expand_linear_layer_inputs(sub, old_dim, new_dim)
                    expanded_names.append(f"{prefix}.{attr_name}.{idx}")
                    break


def expand_policy_observation_inputs(
    source_path: str | Path,
    output_path: str | Path,
    *,
    old_dim: int = 54,
    new_dim: int = 58,
    learning_rate: float | None = None,
    observation_mode: str | None = None,
    observation_schema_version: str | None = None,
    observation_history_strides: tuple[int, ...] = (0,),
    include_loop_flags_in_observation: bool = True,
) -> dict[str, Any]:
    """Expand a MaskablePPO checkpoint from *old_dim* to *new_dim* inputs and save atomically.

    Preserves all existing weights ``[:, :old_dim]`` and biases bit-for-bit,
    initializes ``[:, old_dim:new_dim]`` to exact ``0.0``, resets the optimizer
    with fresh empty state bound to the new parameters (approximate continuation),
    updates observation spaces and rollout buffer to ``(new_dim,)``, and writes
    atomically to *output_path* without modifying *source_path*.
    """
    if new_dim <= old_dim:
        raise ValueError(f"new_dim ({new_dim}) must be greater than old_dim ({old_dim})")
    resolved_source = Path(source_path).resolve()
    resolved_output = Path(output_path)
    if resolved_output.suffix != ".zip":
        resolved_output = resolved_output.with_suffix(".zip")
    if resolved_source == resolved_output.resolve():
        raise ValueError("output_path must differ from source_path to avoid overwriting parent checkpoint")

    model = MaskablePPO.load(str(resolved_source), device="cpu")
    obs_shape = getattr(model.observation_space, "shape", None)
    if not obs_shape or len(obs_shape) != 1 or int(obs_shape[0]) != old_dim:
        raise ValueError(
            f"Expected source observation_space shape ({old_dim},), got {obs_shape}"
        )

    policy = model.policy
    expanded_layers: list[str] = []
    seen_extractor_ids: set[int] = set()
    for ext_name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        _expand_extractor_linear_layers(
            getattr(policy, ext_name, None),
            ext_name,
            old_dim,
            new_dim,
            expanded_layers,
            seen_extractor_ids,
        )

    mlp_extractor = getattr(policy, "mlp_extractor", None)
    if mlp_extractor is None:
        raise ValueError("Policy has no mlp_extractor to expand")

    for net_name in ("policy_net", "value_net", "shared_net"):
        net = getattr(mlp_extractor, net_name, None)
        if (
            isinstance(net, nn.Sequential)
            and len(net) > 0
            and isinstance(net[0], nn.Linear)
            and net[0].in_features == old_dim
        ):
            net[0] = _expand_linear_layer_inputs(net[0], old_dim, new_dim)
            expanded_layers.append(f"mlp_extractor.{net_name}.0")

    if not expanded_layers:
        raise ValueError(f"No input Linear layers with in_features={old_dim} found to expand")

    low_val = float(np.min(model.observation_space.low))
    high_val = float(np.max(model.observation_space.high))
    new_obs_space = spaces.Box(
        low=low_val,
        high=high_val,
        shape=(new_dim,),
        dtype=np.float32,
    )
    model.observation_space = new_obs_space
    policy.observation_space = new_obs_space

    if getattr(model, "rollout_buffer", None) is not None and getattr(model, "rollout_buffer_class", None) is not None:
        model.rollout_buffer = model.rollout_buffer_class(
            model.n_steps,
            new_obs_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )

    if learning_rate is not None:
        effective_lr = float(learning_rate)
    elif callable(model.learning_rate):
        effective_lr = float(model.learning_rate(1.0))
    else:
        effective_lr = float(model.learning_rate)

    optimizer_kwargs = dict(getattr(policy, "optimizer_kwargs", {}) or {})
    policy.optimizer = policy.optimizer_class(
        policy.parameters(),
        lr=effective_lr,
        **optimizer_kwargs,
    )

    hidden_sizes = tuple(
        int(v)
        for v in (
            model.policy_kwargs.get("net_arch", ())
            if isinstance(getattr(model, "policy_kwargs", None), dict)
            else ()
        )
    )
    validated_strides = validate_observation_history_strides(observation_history_strides)
    include_loop_flags = bool(include_loop_flags_in_observation)
    resolved_schema_version = observation_schema_version
    if resolved_schema_version is None and observation_mode is not None:
        resolved_schema_version = compute_observation_schema_version(
            observation_mode,
            validated_strides,
            include_loop_flags,
        )
    wrapper = NeuralPolicy(
        model=model,
        model_path=resolved_output,
        config=NeuralPolicyConfig(
            hidden_sizes=hidden_sizes,
            observation_size=new_dim,
            action_size=int(model.action_space.n),
            env_workers=int(getattr(model, "n_envs", 1)),
            observation_mode=observation_mode,
            observation_schema_version=resolved_schema_version,
            observation_history_strides=validated_strides,
            include_loop_flags_in_observation=include_loop_flags,
        ),
    )
    saved_path = wrapper.save(resolved_output)
    return {
        "source_path": str(resolved_source),
        "output_path": str(saved_path),
        "old_dim": old_dim,
        "new_dim": new_dim,
        "expanded_layers": expanded_layers,
        "optimizer_reset": True,
        "continuation_mode": "approximate_adam_state_reset",
        "learning_rate": effective_lr,
        "policy_config": wrapper.config.to_dict(),
    }
