"""Policy creation and checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.policies.base import Policy, PolicyMode
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy
from sheepdog.training.trainer import Trainer

POLICY_CHOICES: tuple[str, ...] = (
    "random_untrained",
    "instinct_only",
    "heuristic_expert",
    "trained_policy",
    "neural_policy",
    "shepherd_neural_dogs",
)


def create_policy_from_name(
    policy_name: str,
    *,
    weights_payload: dict[str, float] | None = None,
    config: LabConfig | None = None,
    policy_state_path: str | None = None,
    policy_config: dict[str, Any] | None = None,
) -> Policy:
    """Create a runnable policy by name."""

    if policy_name in {"random_untrained", "random_policy"}:
        return RandomPolicy()
    if policy_name == "heuristic_expert":
        return HeuristicExpertPolicy()
    if policy_name == "instinct_only":
        return InstinctOnlyPolicy()
    if policy_name == "neural_policy":
        from sheepdog.policies.neural import NeuralPolicy

        if config is None:
            raise ValueError("Neural policy creation requires config")
        if policy_state_path:
            return NeuralPolicy.load(policy_state_path, config, policy_config)
        return NeuralPolicy.initialize(config)
    if policy_name == "shepherd_neural_dogs":
        from sheepdog.policies.hierarchical import ShepherdNeuralDogPolicy

        if config is None:
            raise ValueError("Hierarchical policy creation requires config")
        if policy_state_path:
            return ShepherdNeuralDogPolicy.load(
                policy_state_path, config, policy_config_dict=policy_config
            )
        return ShepherdNeuralDogPolicy.initialize(config)
    return TrainableLinearPolicy(PolicyWeights.from_dict(weights_payload))


def load_playable_policy(
    config: LabConfig,
    *,
    checkpoint_episode: int | None = None,
    policy_mode: PolicyMode | None = None,
) -> Policy:
    """Return a runnable policy for replay, demo, or evaluation flows."""

    selected_mode = policy_mode or config.policy.policy_mode
    if selected_mode in {
        "random_untrained",
        "random_policy",
        "heuristic_expert",
        "instinct_only",
    }:
        return create_policy_from_name(selected_mode)

    output_root = Path(config.training.output_dir)
    weights_payload: dict[str, float] | None = None
    policy_state_path: str | None = None
    policy_config: dict[str, Any] | None = None
    if checkpoint_episode is not None:
        checkpoint_path = output_root / "checkpoints" / f"checkpoint-{checkpoint_episode:06d}.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_episode} not found")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        weights_payload = payload.get("policy_weights")
        policy_state_path = payload.get("policy_state_path")
        policy_config = payload.get("policy_config")
        selected_mode = payload.get("policy_name", selected_mode)
    else:
        state_path = output_root / Trainer.STATE_FILENAME
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            weights_payload = payload.get("weights")
            # Prefer the best-performing model over the most recently trained one
            # when replaying — this prevents policy-collapse regressions from
            # replacing the peak-performance model in the viewer.
            policy_state_path = payload.get("best_model_path") or payload.get("policy_state_path")
            policy_config = payload.get("policy_config")
    return create_policy_from_name(
        selected_mode,
        weights_payload=weights_payload,
        config=config,
        policy_state_path=policy_state_path,
        policy_config=policy_config,
    )