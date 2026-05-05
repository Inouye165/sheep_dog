"""Hierarchical shepherd + neural dog policy.

Architecture
------------
ShepherdNeuralDogPolicy
  ├── ScriptedShepherd   (Phase A: scripted; Phase B: replaceable with learned)
  └── NeuralDogPolicy    (trained MaskablePPO model built on JointActionRLEnv)

The shepherd issues a high-level command each step.  That command is baked into
each dog's observation via HierarchicalObservationBuilder.  The neural model
then selects actions from those extended observations.

Training
--------
Dogs are trained separately using ``HierarchicalMaskablePPOTrainer`` (see
training/hierarchical_trainer.py).  The policy loaded here is the *inference*
wrapper used for evaluation, demo, and server playback.

Checkpoint compatibility
------------------------
Neural dog checkpoints saved by the hierarchical trainer store the observation
size under ``policy_config.observation_size``.  The size includes the base
role-aware vector plus shepherd command one-hot (8) plus identity features
(2 + MAX_DOG_SLOTS).  Loading a checkpoint created without these extra features
into this policy will raise a shape mismatch at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sheepdog.config import LabConfig
from sheepdog.environment import ACTION_ORDER
from sheepdog.observations import HierarchicalObservationBuilder
from sheepdog.policies.base import Action
from sheepdog.shepherd import ScriptedShepherd, ShepherdCommand
from sheepdog.training.joint_rl_env import JointActionRLEnv


@dataclass(frozen=True, slots=True)
class HierarchicalNeuralPolicyConfig:
    """Metadata stored with hierarchical neural checkpoints."""

    hidden_sizes: tuple[int, ...]
    observation_size: int
    action_size: int = len(ACTION_ORDER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        observation_size: int,
    ) -> HierarchicalNeuralPolicyConfig:
        if not payload:
            return cls(hidden_sizes=(64, 64), observation_size=observation_size)
        return cls(
            hidden_sizes=tuple(int(v) for v in payload.get("hidden_sizes", (64, 64))),
            observation_size=int(payload.get("observation_size", observation_size)),
            action_size=int(payload.get("action_size", len(ACTION_ORDER))),
        )


class ShepherdNeuralDogPolicy:
    """Inference-time hierarchical policy: scripted shepherd + neural dogs.

    Use ``initialize`` to create an untrained model (for testing / baseline),
    or ``load`` to restore a trained checkpoint.

    Both shepherd and neural model are replaceable for Phase-B extensibility:
      - Pass a custom ``shepherd`` to swap in a learned shepherd.
      - Subclass and override ``_shepherd`` for finer control.
    """

    name = "shepherd_neural_dogs"
    trainer_type = "hierarchical_maskable_ppo"
    policy_type = "neural"

    def __init__(
        self,
        model: Any,  # MaskablePPO – kept as Any to avoid hard import at class definition
        model_path: Path,
        policy_config: HierarchicalNeuralPolicyConfig,
        shepherd: ScriptedShepherd | None = None,
    ) -> None:
        self._model = model
        self.model_path = model_path
        self.policy_config = policy_config
        self._shepherd = shepherd if shepherd is not None else ScriptedShepherd()
        self._obs_builder = HierarchicalObservationBuilder()

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        config: LabConfig,
        shepherd: ScriptedShepherd | None = None,
    ) -> ShepherdNeuralDogPolicy:
        """Create a fresh, untrained hierarchical model."""
        from sb3_contrib import MaskablePPO

        adapter = JointActionRLEnv(config, shepherd=shepherd)
        observation_size = int(adapter.observation_space.shape[0])
        policy_config = HierarchicalNeuralPolicyConfig(
            hidden_sizes=tuple(config.training.neural_hidden_sizes),
            observation_size=observation_size,
        )
        model = MaskablePPO(
            "MlpPolicy",
            adapter,
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
        return cls(
            model=model,
            model_path=Path(""),
            policy_config=policy_config,
            shepherd=shepherd,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: LabConfig,
        policy_config_dict: dict[str, Any] | None = None,
        shepherd: ScriptedShepherd | None = None,
    ) -> ShepherdNeuralDogPolicy:
        """Load a trained hierarchical checkpoint."""
        from sb3_contrib import MaskablePPO

        adapter = JointActionRLEnv(config, shepherd=shepherd)
        observation_size = int(adapter.observation_space.shape[0])
        resolved_path = Path(path)
        model = MaskablePPO.load(str(resolved_path), env=adapter)
        return cls(
            model=model,
            model_path=resolved_path,
            policy_config=HierarchicalNeuralPolicyConfig.from_dict(
                policy_config_dict, observation_size
            ),
            shepherd=shepherd,
        )

    def save(self, path: str | Path) -> Path:
        """Save the underlying model to *path* and return the resolved path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(target))
        return target

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def select_actions(self, environment: object) -> list[Action]:
        """Return one action per dog, conditioned on the shepherd's command."""
        from sheepdog.environment import SheepdogEnvironment

        env = environment  # type: ignore[assignment]
        assert isinstance(env, SheepdogEnvironment)

        # Issue command for this step.
        command: ShepherdCommand = self._shepherd.issue_command(env)

        actions: list[Action] = []
        for dog_index in range(env.dog_count):
            obs_obj = self._obs_builder.build_hierarchical(env, dog_index, command)
            obs = np.asarray(obs_obj.values, dtype=np.float32)[np.newaxis]
            # Build action mask.
            reserved = {
                env.project_dog_action(i, actions[i]) for i in range(len(actions))
            }
            mask_map = env.action_mask_for_dog(dog_index, reserved_positions=reserved)
            mask = np.asarray(
                [mask_map[a] for a in ACTION_ORDER], dtype=bool
            )[np.newaxis]
            action_idx, _ = self._model.predict(
                obs, action_masks=mask, deterministic=True
            )
            action_name: Action = ACTION_ORDER[int(action_idx[0])]
            if not bool(mask[0][int(action_idx[0])]):
                action_name = "wait"
            actions.append(action_name)
        return actions
