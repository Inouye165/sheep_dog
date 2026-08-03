"""Regression tests for the MaskablePPO neural policy experiment."""

# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.benchmark import BenchmarkHarness
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.neural import NeuralPolicy, _build_vec_env
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import TrainableLinearPolicy
from sheepdog.server import _load_playable_policy
from sheepdog.training.factory import create_trainer
from sheepdog.training.maskable_ppo import finish_wandb_run
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
        ppo_env_workers=1,
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


def test_parallel_ppo_environment_starts_worker_processes(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    config = replace(
        config,
        training=replace(config.training, ppo_env_workers=2),
    )

    vec_env, workers, backend = _build_vec_env(config)
    try:
        observations = vec_env.reset()
    finally:
        vec_env.close()

    expected_workers = 1 if os.name == "nt" and "SHEEPDOG_PPO_NUM_ENVS" not in os.environ else 2
    expected_backend = "dummy" if expected_workers == 1 else "subproc"
    assert workers == expected_workers
    assert backend == expected_backend
    assert observations.shape[0] == expected_workers


def test_loaded_policy_disables_unavailable_tensorboard(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    model = MagicMock()
    model.tensorboard_log = str(tmp_path / "stale-tensorboard-path")
    vec_env = SimpleNamespace(observation_space=SimpleNamespace(shape=(12,)))

    with (
        patch("sheepdog.policies.neural._build_vec_env", return_value=(vec_env, 1, "dummy")),
        patch("sheepdog.policies.neural.MaskablePPO.load", return_value=model),
        patch("sheepdog.policies.neural.tensorboard_available", return_value=False),
    ):
        NeuralPolicy.load(tmp_path / "model.zip", config)

    assert model.tensorboard_log is None


def test_wandb_cleanup_accepts_partial_module() -> None:
    with patch.dict(sys.modules, {"wandb": SimpleNamespace()}):
        finish_wandb_run()


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


def test_maskable_ppo_trainer_compatible_across_curriculum_stages(tmp_path: Path) -> None:
    import json
    from dataclasses import asdict

    from sheepdog.curriculum import apply_training_profile

    config_stage_1 = make_experiment_config(tmp_path)
    config_stage_1 = apply_training_profile(
        config_stage_1,
        curriculum_stage=1,
        enable_instinct_rewards=True,
    )

    training_signature = {
        "action_size": 9,
        "observation_mode": "guided",
        "rewards": asdict(config_stage_1.rewards),
        "environment": {
            "dog_speed": config_stage_1.environment.dog_speed,
            "dog_sprint_multiplier": config_stage_1.environment.dog_sprint_multiplier,
            "sheep_speed": config_stage_1.environment.sheep_speed,
        }
    }

    state_payload = {
        "total_episodes_trained": 10,
        "total_timesteps": 300,
        "policy_state_path": "stage-1-model.zip",
        "policy_config": {
            "hidden_sizes": [128, 128],
            "observation_size": 54,
            "action_size": 9
        },
        "training_signature": training_signature
    }

    artifacts_dir = Path(config_stage_1.training.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "training-state.json").write_text(
        json.dumps(state_payload, indent=2),
        encoding="utf-8",
    )

    config_stage_3 = apply_training_profile(
        make_experiment_config(tmp_path),
        curriculum_stage=3,
        enable_instinct_rewards=True,
    )

    trainer = create_trainer(config_stage_3, config_stage_3.training.output_dir)
    assert trainer.has_compatible_policy_state() is True


def test_maskable_ppo_phase_contexts_and_runtime_tracking(tmp_path: Path) -> None:
    from sheepdog.training.runtime import TrainingRuntimeTracker

    config = make_experiment_config(tmp_path)
    tracker_file = tmp_path / "training-runtime.json"
    tracker = TrainingRuntimeTracker(tracker_file, start_heartbeat_thread=False)
    tracker.start_session("test-run", session_id="test-session")

    trainer = create_trainer(config, config.training.output_dir)
    trainer.runtime_tracker = tracker

    summary = trainer.train()
    tracker.end_session("completed")

    assert len(summary.checkpoints) >= 1
    runtime_summary = tracker.snapshot()
    assert runtime_summary["checkpoint_save_seconds"] >= 0
    assert runtime_summary["evaluation_seconds"] >= 0


def test_quick_vs_confidence_evaluation_mode_selection(tmp_path: Path) -> None:
    from sheepdog.evaluation.evaluator import EvaluationRecord, EvaluationSummary

    config = make_experiment_config(tmp_path)
    config = replace(
        config,
        training=replace(
            config.training,
            total_timesteps=16,
            rollout_steps=8,
            confidence_candidate_success_rate=0.75,
        ),
    )

    trainer = create_trainer(config, config.training.output_dir)

    evaluate_calls: list[str] = []

    def fake_evaluate(policy, seeds, **kwargs):
        mode = kwargs.get("evaluation_mode", "unknown")
        evaluate_calls.append(mode)
        rec = EvaluationRecord(
            seed=int(seeds[0]),
            success=mode == "confidence",
            timeout=mode != "confidence",
            stopped=False,
            steps=10 if mode == "confidence" else 40,
            simulated_seconds=1.0,
            sheep_penned=2 if mode == "confidence" else 0,
            final_sheep_distance_to_pen=1.0,
            final_flock_spread=1.0,
            no_progress_steps=0,
            stop_reason="success" if mode == "confidence" else "timeout",
            spawn_mode="default",
            reward_total=100.0 if mode == "confidence" else -10.0,
            final_farthest_distance_to_pen=1.0,
            final_farthest_distance_to_flock_center=1.0,
            role_switches=0,
            collector_activations=0,
            blocker_activations=0,
            cumulative_gate_progress=0.0,
            controlled_stall_steps=0,
            left_flank_occupancy_steps=0,
            right_flank_occupancy_steps=0,
            gate_corridor_occupancy_peak=0.0,
            gate_corridor_failure_steps=0,
            dog_role_occupancy={},
            reward_breakdown={},
            replay_path="",
        )
        return (
            EvaluationSummary(
                checkpoint_episode=0,
                policy_name="test",
                records=(rec,),
                success_rate=0.8 if mode == "confidence" else 0.2,
                timeout_rate=0.0 if mode == "confidence" else 1.0,
                average_completion_steps=10.0 if mode == "confidence" else 40.0,
                average_completion_seconds=1.0,
                average_sheep_penned=2.0 if mode == "confidence" else 0.0,
                average_reward=100.0 if mode == "confidence" else -10.0,
                average_distance_to_pen=1.0,
                average_flock_spread=1.0,
                evaluation_mode=mode,
                promotion_eligible=mode == "confidence",
            ),
            SimpleNamespace(name="eval.json"),
            SimpleNamespace(name="eval.csv"),
        )

    with patch.object(trainer.evaluator, "evaluate", side_effect=fake_evaluate):
        trainer.train()

    assert "quick" in evaluate_calls


def test_quick_and_confidence_evaluations_keep_distinct_artifacts(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")
    policy = HeuristicExpertPolicy()

    _, quick_json, quick_csv = evaluator.evaluate(
        policy,
        (11,),
        checkpoint_episode=7,
        capture_replays=False,
        evaluation_mode="quick",
        run_id="run-test",
        checkpoint_id="chk-run-test-ts-16",
        policy_version=1,
    )
    _, confidence_json, confidence_csv = evaluator.evaluate(
        policy,
        (11,),
        checkpoint_episode=7,
        capture_replays=False,
        evaluation_mode="confidence",
        run_id="run-test",
        checkpoint_id="chk-run-test-ts-16",
        policy_version=1,
    )

    assert quick_json != confidence_json
    assert quick_csv != confidence_csv
    assert quick_json.exists()
    assert quick_csv.exists()
    assert confidence_json.exists()
    assert confidence_csv.exists()


def test_consecutive_batches_do_not_reuse_terminal_checkpoint_episode(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    config = replace(
        config,
        training=replace(
            config.training,
            checkpoint_episodes=(0, 1),
            total_timesteps=96,
        ),
    )

    first_summary = create_trainer(config, config.training.output_dir).train()
    first_state = json.loads(
        (Path(config.training.output_dir) / "training-state.json").read_text(encoding="utf-8")
    )
    second_summary = create_trainer(config, config.training.output_dir).train()
    second_state = json.loads(
        (Path(config.training.output_dir) / "training-state.json").read_text(encoding="utf-8")
    )

    first_terminal = first_summary.checkpoints[-1]["checkpoint_episode"]
    second_new_checkpoints = [
        checkpoint
        for checkpoint in second_summary.checkpoints
        if checkpoint.get("checkpoint_episode", -1) > first_terminal
    ]

    assert len(second_summary.checkpoints) == len(first_summary.checkpoints) + 2
    assert len(second_new_checkpoints) == 2
    assert second_new_checkpoints[0]["checkpoint_episode"] == first_terminal + 1
    assert first_state["total_environment_episodes"] > 0
    assert second_state["total_environment_episodes"] > first_state["total_environment_episodes"]
    assert all(
        checkpoint["environment_episodes_total"] >= first_state["total_environment_episodes"]
        for checkpoint in second_new_checkpoints
    )
    assert second_new_checkpoints[-1]["environment_episodes_total"] == second_state[
        "total_environment_episodes"
    ]


def test_training_uses_exact_checkpoint_timestep_targets(tmp_path: Path) -> None:
    config = make_experiment_config(tmp_path)
    config = replace(
        config,
        training=replace(
            config.training,
            checkpoint_episodes=(0, 1),
            checkpoint_timesteps=(64, 96),
            total_timesteps=96,
        ),
    )

    summary = create_trainer(config, config.training.output_dir).train()
    state = json.loads(
        (Path(config.training.output_dir) / "training-state.json").read_text(
            encoding="utf-8"
        )
    )

    assert [checkpoint["global_timestep"] for checkpoint in summary.checkpoints] == [64, 96]
    assert state["total_timesteps"] == 96



