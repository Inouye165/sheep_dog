"""Regression tests for the MaskablePPO neural policy experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.benchmark import BenchmarkHarness
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import TrainableLinearPolicy
from sheepdog.server import _load_playable_policy
from sheepdog.training.factory import create_trainer
from sheepdog.training.rl_env import SheepdogRLAdapter


def make_experiment_config(tmp_path: Path, **environment_overrides: int) -> LabConfig:
    environment_payload = {"max_steps": 40, "dogs": 2, "sheep": 2}
    environment_payload.update(environment_overrides)
    environment = EnvironmentConfig(**environment_payload)
    training = TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        episodes=0,
        checkpoint_episodes=(0,),
        evaluation_seeds=(11,),
        candidate_evaluation_seeds=(91,),
        rollout_steps=8,
        batch_size=4,
        total_timesteps=16,
        output_dir=str(tmp_path / "artifacts"),
        web_export_dir=str(tmp_path / "web" / "generated"),
    )
    return LabConfig(environment=environment, rewards=RewardConfig(), training=training)


def test_neural_policy_initializes_and_acts(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    policy = NeuralPolicy.initialize(config)
    adapter = SheepdogRLAdapter(config, fixed_seed_sequence=(11,))
    adapter.reset(seed=11)

    actions = policy.select_actions(adapter._environment)  # pylint: disable=protected-access

    assert len(actions) == config.environment.dogs
    assert set(actions).issubset(
        {
            "up",
            "down",
            "left",
            "right",
            "sprint_up",
            "sprint_down",
            "sprint_left",
            "sprint_right",
            "wait",
        }
    )


def test_rl_adapter_produces_expected_observation_shape_and_masks_invalid_actions(
    tmp_path: Path,
) -> None:
    config = make_experiment_config(tmp_path, dogs=1, sheep=0)
    adapter = SheepdogRLAdapter(config, fixed_seed_sequence=(13,))
    observation, info = adapter.reset(seed=13)
    adapter._environment.dogs[0].position = adapter._environment.dogs[0].position.__class__(0, 0)  # pylint: disable=protected-access

    mask = adapter.action_masks()

    assert observation.shape == adapter.observation_space.shape
    assert info["current_dog_index"] == 0
    assert mask.shape == (9,)
    assert bool(mask[0]) is False
    assert bool(mask[2]) is False
    assert bool(mask[4]) is False
    assert bool(mask[6]) is False


def test_rl_adapter_fixed_seed_sequence_is_deterministic(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    adapter = SheepdogRLAdapter(config, fixed_seed_sequence=(21, 21))

    first_observation, _ = adapter.reset()
    second_observation, _ = adapter.reset()

    assert np.allclose(first_observation, second_observation)


def test_maskable_ppo_trainer_writes_checkpoint_and_model(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)

    trainer = create_trainer(config, config.training.output_dir)
    summary = trainer.train()

    assert summary.checkpoints
    assert Path(summary.final_model_path).exists()
    assert (tmp_path / "artifacts" / "checkpoints" / "checkpoint-000000.json").exists()


def test_maskable_ppo_trainer_emits_progress_updates(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    trainer = create_trainer(config, config.training.output_dir)
    payloads: list[dict[str, object]] = []

    trainer.train(progress_callback=payloads.append)

    phases = {str(payload.get("phase")) for payload in payloads}
    assert "starting" in phases
    assert "learning" in phases
    assert "checkpoint" in phases
    assert "complete" in phases
    assert any(payload.get("checkpoint_episode") == 0 for payload in payloads)


def test_maskable_ppo_trainer_resets_incompatible_saved_action_space(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    artifacts_dir = Path(config.training.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "training-state.json").write_text(
        """
        {
            "total_episodes_trained": 5,
            "total_timesteps": 123,
            "policy_state_path": "old-model.zip",
            "policy_config": {
                "hidden_sizes": [64, 64],
                "observation_size": 10,
                "action_size": 5
            }
        }
        """.strip(),
        encoding="utf-8",
    )

    trainer = create_trainer(config, config.training.output_dir)
    summary = trainer.train()
    state_payload = (artifacts_dir / "training-state.json").read_text(encoding="utf-8")

    assert summary.checkpoints
    assert '"total_episodes_trained": 1' in state_payload
    assert '"total_timesteps": 16' in state_payload


def test_maskable_ppo_trainer_resets_when_reward_signature_changes(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    artifacts_dir = Path(config.training.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "training-state.json").write_text(
        """
        {
            "total_episodes_trained": 7,
            "total_timesteps": 200,
            "policy_state_path": "old-model.zip",
            "policy_config": {
                "hidden_sizes": [64, 64],
                "observation_size": 10,
                "action_size": 9
            },
            "training_signature": {
                "action_size": 9,
                "rewards": {
                    "progress_scale": 999
                },
                "environment": {
                    "dogs": 3,
                    "sheep": 6,
                    "dog_speed": 1,
                    "dog_sprint_multiplier": 2,
                    "sheep_speed": 0.75
                }
            }
        }
        """.strip(),
        encoding="utf-8",
    )

    trainer = create_trainer(config, config.training.output_dir)
    trainer.train()
    state_payload = (artifacts_dir / "training-state.json").read_text(encoding="utf-8")

    assert '"total_episodes_trained": 1' in state_payload
    assert '"total_timesteps": 16' in state_payload


def test_neural_checkpoint_loads_for_playback(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    trainer = create_trainer(config, config.training.output_dir)
    trainer.train()

    policy = _load_playable_policy(config, checkpoint_episode=0, policy_mode="neural_policy")

    assert isinstance(policy, NeuralPolicy)


def test_benchmark_harness_compares_multiple_policy_modes(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    neural_policy = NeuralPolicy.initialize(config)
    harness = BenchmarkHarness(config, tmp_path / "benchmarks")

    results, json_path, csv_path, summary_path = harness.compare(
        [
            ("random", RandomPolicy(seed=0)),
            ("instinct", InstinctOnlyPolicy()),
            ("heuristic", HeuristicExpertPolicy()),
            ("linear", TrainableLinearPolicy()),
            ("neural", neural_policy),
        ],
        seeds=(11,),
    )

    assert len(results) == 5
    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()


def test_evaluator_is_deterministic_for_fixed_seed_neural_policy(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    policy = NeuralPolicy.initialize(config)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    first, _, _ = evaluator.evaluate(policy, (11,), checkpoint_episode=0)
    second, _, _ = evaluator.evaluate(policy, (11,), checkpoint_episode=1)

    assert first.average_reward == second.average_reward
    assert first.success_rate == second.success_rate
