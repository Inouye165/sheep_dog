"""Focused tests for parallel evaluation and deduplicated quick/confidence evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sb3_contrib import MaskablePPO

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.training.factory import create_trainer
from sheepdog.training.rl_env import SheepdogRLAdapter


def _make_small_config(
    tmp_path: Path,
    *,
    evaluation_seeds: tuple[int, ...] = (11, 23, 37),
    quick_evaluation_seed_count: int = 1,
    checkpoint_episodes: tuple[int, ...] = (0,),
    total_timesteps: int = 8,
    confidence_candidate_success_rate: float = 1.1,
) -> LabConfig:
    environment = EnvironmentConfig(max_steps=20, dogs=2, sheep=2)
    training = TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        episodes=0,
        checkpoint_episodes=checkpoint_episodes,
        evaluation_seeds=evaluation_seeds,
        quick_evaluation_seed_count=quick_evaluation_seed_count,
        confidence_candidate_success_rate=confidence_candidate_success_rate,
        rollout_steps=8,
        batch_size=4,
        total_timesteps=total_timesteps,
        ppo_env_workers=1,
        backup_enabled=False,
        wandb_enabled=False,
        output_dir=str(tmp_path / "artifacts"),
        web_export_dir=str(tmp_path / "web" / "generated"),
    )
    return LabConfig(environment=environment, rewards=RewardConfig(), training=training)


def _build_saved_neural_policy(config: LabConfig, model_path: Path) -> NeuralPolicy:
    env = SheepdogRLAdapter(config)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=4,
        policy_kwargs={"net_arch": [32, 32]},
        seed=7,
        device="cpu",
    )
    obs_dim = int(env.observation_space.shape[0])
    policy_config = NeuralPolicyConfig(
        hidden_sizes=(32, 32),
        observation_size=obs_dim,
        observation_mode=config.training.observation_mode,
    )
    policy = NeuralPolicy(model=model, model_path=model_path, config=policy_config)
    policy.policy_version = 1
    policy.save(model_path)
    return policy


def test_parallel_evaluation_matches_sequential_results_and_order(tmp_path: Path) -> None:
    config = _make_small_config(tmp_path)
    model_path = tmp_path / "test_neural_policy.zip"
    policy = _build_saved_neural_policy(config, model_path)
    seeds = (37, 11, 23)

    seq_evaluator = Evaluator(config, tmp_path / "eval_seq")
    par_evaluator = Evaluator(config, tmp_path / "eval_par")
    try:
        seq_summary, _, _ = seq_evaluator.evaluate(
            policy,
            seeds,
            checkpoint_episode=1,
            deterministic=True,
            capture_replays=True,
            capture_step_telemetry=True,
            evaluation_mode="confidence",
            record_in_registry=False,
            max_workers=1,
        )
        par_summary, _, _ = par_evaluator.evaluate(
            policy,
            seeds,
            checkpoint_episode=1,
            deterministic=True,
            capture_replays=True,
            capture_step_telemetry=True,
            evaluation_mode="confidence",
            record_in_registry=False,
            max_workers=2,
        )
    finally:
        par_evaluator.close()
        seq_evaluator.close()

    assert [r.seed for r in seq_summary.records] == list(seeds)
    assert [r.seed for r in par_summary.records] == list(seeds)

    comparable_fields = (
        "seed",
        "success",
        "timeout",
        "steps",
        "sheep_penned",
        "reward_total",
    )
    for seq_rec, par_rec in zip(seq_summary.records, par_summary.records, strict=True):
        for field_name in comparable_fields:
            assert getattr(par_rec, field_name) == getattr(seq_rec, field_name)

    assert par_summary.success_rate == pytest.approx(seq_summary.success_rate)
    assert par_summary.average_reward == pytest.approx(seq_summary.average_reward)


def test_trainer_skips_duplicate_quick_eval_when_quick_seeds_equal_all_seeds(
    tmp_path: Path,
) -> None:
    config = _make_small_config(
        tmp_path,
        evaluation_seeds=(11, 23),
        quick_evaluation_seed_count=2,
        checkpoint_episodes=(0,),
        total_timesteps=8,
        confidence_candidate_success_rate=1.1,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    original_evaluate = trainer.evaluator.evaluate
    original_record = trainer.evaluator.record_evaluation_in_registry
    evaluate_modes: list[str] = []

    def spy_evaluate(*args, **kwargs):
        evaluate_modes.append(str(kwargs.get("evaluation_mode", "confidence")))
        return original_evaluate(*args, **kwargs)

    try:
        with (
            patch.object(trainer.evaluator, "evaluate", side_effect=spy_evaluate) as mock_eval,
            patch.object(
                trainer.evaluator,
                "record_evaluation_in_registry",
                wraps=original_record,
            ) as mock_record,
        ):
            trainer.train(progress_callback=events.append)
    finally:
        trainer.evaluator.close()

    assert mock_eval.call_count == 1
    assert evaluate_modes == ["confidence"]
    assert mock_record.call_count == 1

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 1
    timing = checkpoint_events[0]["segment_timing"]
    assert isinstance(timing, dict)
    assert timing["quick_eval_seconds"] == 0.0
    assert timing["confidence_eval_seconds"] >= 0.0


def test_unclosed_evaluator_with_live_pool_exits_subprocess_cleanly(tmp_path: Path) -> None:
    script = (
        "import os\n"
        "from pathlib import Path\n"
        "from sheepdog.config import LabConfig\n"
        "from sheepdog.evaluation.evaluator import Evaluator\n"
        f"ev = Evaluator(LabConfig(), Path({str(tmp_path / 'eval_unclosed')!r}))\n"
        "pool = ev._get_or_create_eval_pool(2, exact_workers=True)\n"
        "pids = [pool.submit(os.getpid).result() for _ in range(2)]\n"
        "assert len(pids) == 2\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
