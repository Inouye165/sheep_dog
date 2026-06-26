"""Regression tests for checkpoint and evaluation export."""

# pylint: disable=missing-function-docstring,missing-class-docstring,import-outside-toplevel
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.factory import create_policy_from_name
from sheepdog.policies.heuristic import HeuristicExpertPolicy, InstinctOnlyPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy
from sheepdog.server import TrainingManager, _build_training_job_config, _load_playable_policy
from sheepdog.training.factory import create_trainer
from sheepdog.training.trainer import CandidateEvaluationSummary, Trainer


class TrainerProbe(Trainer):
    def candidate_evaluation_seeds(self) -> tuple[int, ...]:
        return self._candidate_evaluation_seeds()

    def evaluate_candidate(self, policy: TrainableLinearPolicy) -> CandidateEvaluationSummary:
        return self._evaluate_candidate(policy)


def make_config(output_dir: Path) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(max_steps=30, dogs=2, sheep=3),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11, 13),
            train_seed=7,
            evaluation_seed=9,
            mutation_scale=0.05,
            output_dir=str(output_dir),
            web_export_dir=str(output_dir / "web" / "generated"),
        ),
    )


def test_checkpoint_metadata_is_written(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    trainer = Trainer(config, tmp_path)

    summary = trainer.train()

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint-000000.json"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_episode"] == 0
    assert payload["environment_config"]["dogs"] == 2
    assert payload["reward_config"]["progress_scale"] > 0
    assert summary.checkpoints


def test_evaluation_writes_json_and_csv(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    summary, json_path, csv_path = evaluator.evaluate(
        HeuristicExpertPolicy(), (11, 13), checkpoint_episode=0
    )

    assert json_path.exists()
    assert csv_path.exists()
    assert len(summary.records) == 2
    assert {record.seed for record in summary.records} == {11, 13}


def test_evaluation_summary_includes_success_timeout_and_completion_metrics(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = Evaluator(config, tmp_path / "evaluations")

    summary, _, _ = evaluator.evaluate(HeuristicExpertPolicy(), (11, 13), checkpoint_episode=0)

    assert 0.0 <= summary.success_rate <= 1.0
    assert 0.0 <= summary.timeout_rate <= 1.0
    assert summary.average_completion_steps >= 0
    assert summary.average_completion_seconds >= 0


def test_training_state_persists_across_runs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    Trainer(config, tmp_path).train()
    state_path = tmp_path / Trainer.STATE_FILENAME
    assert state_path.exists()
    first_total = json.loads(state_path.read_text(encoding="utf-8"))["total_episodes_trained"]

    second = Trainer(config, tmp_path)
    assert second.total_episodes_trained == first_total
    second.train()
    second_total = json.loads(state_path.read_text(encoding="utf-8"))["total_episodes_trained"]
    assert second_total >= first_total


def test_policy_weights_load_legacy_state_payload() -> None:
    payload = {
        "nearest_sheep": 1.0,
        "flock_center": 2.0,
        "pen_pressure": 3.0,
        "behind_flock": 4.0,
        "wall_margin": 5.0,
        "wait_bias": -6.0,
    }

    weights = PolicyWeights.from_dict(payload)

    assert weights.nearest_sheep == 1.0
    assert weights.team_formation == PolicyWeights().team_formation
    assert weights.collector_focus == PolicyWeights().collector_focus
    assert weights.rear_behind_flock == PolicyWeights().rear_behind_flock
    assert weights.blocker_gate_control == PolicyWeights().blocker_gate_control


def test_policy_weights_serialize_new_role_specific_fields() -> None:
    weights = PolicyWeights(
        rear_behind_flock=1.7,
        flank_side_control=1.6,
        flank_handedness=0.8,
        collector_stray_focus=1.8,
        blocker_gate_control=1.4,
        blocker_funnel_lane=0.9,
    )

    payload = asdict(weights)
    restored = PolicyWeights.from_dict(payload)

    assert restored.rear_behind_flock == 1.7
    assert restored.flank_side_control == 1.6
    assert restored.flank_handedness == 0.8
    assert restored.collector_stray_focus == 1.8
    assert restored.blocker_gate_control == 1.4
    assert restored.blocker_funnel_lane == 0.9


def test_hill_climber_training_saves_role_aware_weights(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    summary = Trainer(config, tmp_path).train()
    state_path = tmp_path / Trainer.STATE_FILENAME
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert state_path.exists()
    assert "rear_drive" in payload["weights"]
    assert "collector_focus" in payload["weights"]
    assert summary.final_weights.collector_focus == payload["weights"]["collector_focus"]


def test_build_training_job_config_applies_fast_mode_and_curriculum() -> None:
    config = _build_training_job_config(
        4,
        True,
        enable_instinct_rewards=True,
        curriculum_stage=2,
        debug_reward_breakdown=True,
    )

    assert config.training.episodes == 3
    assert config.training.trainer_type == "maskable_ppo"
    assert config.training.policy_type == "neural"
    assert config.policy.policy_mode == "neural_policy"
    assert config.training.evaluation_seeds == (11, 23, 37)
    assert config.training.candidate_evaluation_seeds == TrainingConfig().candidate_evaluation_seeds
    assert config.training.candidate_pool_size == TrainingConfig().candidate_pool_size
    assert config.rewards.instincts.enable_instinct_rewards is True
    assert config.rewards.instincts.curriculum_stage == 2
    assert config.environment.dogs == 1


def test_lab_config_old_training_payload_falls_back_to_single_candidate_seed(
    tmp_path: Path,
) -> None:
    payload = make_config(tmp_path).to_dict()
    del payload["training"]["candidate_evaluation_seeds"]

    config = LabConfig.from_dict(payload)
    trainer = TrainerProbe(config, tmp_path)

    assert trainer.candidate_evaluation_seeds() == (config.training.evaluation_seed,)


def test_candidate_evaluation_uses_multiple_seeds_and_averages_scores(tmp_path: Path) -> None:
    base_config = make_config(tmp_path)
    config = replace(
        base_config,
        training=replace(
            base_config.training,
            candidate_evaluation_seeds=(91, 92, 93),
        ),
    )
    trainer = TrainerProbe(config, tmp_path)
    seen_seeds: list[int] = []

    def fake_result(seed: int) -> SimpleNamespace:
        seen_seeds.append(seed)
        reward_total = float(seed - 90)
        success = seed != 91
        timeout = seed == 91
        stopped = seed == 93
        return SimpleNamespace(
            seed=seed,
            final_snapshot=SimpleNamespace(
                average_distance_to_pen=float(seed) / 10.0,
                flock_spread=float(seed) / 100.0,
            ),
            stats=SimpleNamespace(
                reward_total=reward_total,
                success=success,
                timeout=timeout,
                stopped=stopped,
                sheep_penned=2,
                steps=seed,
            ),
        )

    import sheepdog.training.trainer as trainer_module

    original_environment = trainer_module.SheepdogEnvironment

    class FakeEnvironment:
        def __init__(self, _config: LabConfig) -> None:
            pass

        def run_policy(
            self,
            _policy: TrainableLinearPolicy,
            seed: int,
            capture_replay: bool = False,
        ) -> SimpleNamespace:
            del capture_replay
            return fake_result(seed)

    trainer_module.SheepdogEnvironment = FakeEnvironment
    try:
        summary = trainer.evaluate_candidate(TrainableLinearPolicy())
    finally:
        trainer_module.SheepdogEnvironment = original_environment

    assert seen_seeds == [91, 92, 93]
    assert summary.seeds == (91, 92, 93)
    assert summary.average_reward == 2.0
    assert round(summary.success_rate, 4) == round(2 / 3, 4)
    assert round(summary.timeout_rate, 4) == round(1 / 3, 4)
    assert round(summary.stopped_rate, 4) == round(1 / 3, 4)


def test_candidate_evaluation_summary_score_uses_averages() -> None:
    summary = CandidateEvaluationSummary(
        seeds=(91, 92, 93),
        average_reward=5.0,
        success_rate=0.5,
        timeout_rate=0.25,
        stopped_rate=0.0,
        average_sheep_penned=3.0,
        average_distance_to_pen=8.0,
        average_flock_spread=2.0,
        average_steps=200.0,
    )

    assert summary.score == 4904.34


def test_training_evaluation_keeps_fixed_checkpoint_seeds(tmp_path: Path) -> None:
    base_config = make_config(tmp_path)
    config = replace(
        base_config,
        training=replace(
            base_config.training,
            evaluation_seeds=(11, 13),
            candidate_evaluation_seeds=(91, 92),
        ),
    )
    trainer = Trainer(config, tmp_path)
    captured: list[tuple[int, ...]] = []
    original_evaluate = trainer.evaluator.evaluate

    def wrapped_evaluate(policy: object, seeds: tuple[int, ...], checkpoint_episode: int):
        captured.append(seeds)
        return original_evaluate(policy, seeds, checkpoint_episode)

    trainer.evaluator.evaluate = wrapped_evaluate  # type: ignore[method-assign]

    trainer.train()

    assert captured == [(11, 13)]


def test_load_playable_policy_reads_checkpoint_weights(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), training=replace(make_config(tmp_path).training))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "checkpoint-000003.json").write_text(
        json.dumps(
            {
                "checkpoint_episode": 3,
                "policy_weights": {"rear_drive": 2.5, "collector_focus": 1.9},
            }
        ),
        encoding="utf-8",
    )

    policy = _load_playable_policy(config, checkpoint_episode=3, policy_mode="trained_policy")

    assert isinstance(policy, TrainableLinearPolicy)
    assert policy.weights.rear_drive == 2.5
    assert policy.weights.collector_focus == 1.9


def test_load_playable_policy_uses_random_untrained_mode(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    policy = _load_playable_policy(config, policy_mode="random_untrained")

    assert isinstance(policy, RandomPolicy)


def test_create_trainer_defaults_to_hill_climb(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    trainer = create_trainer(config, tmp_path)

    assert isinstance(trainer, Trainer)


def test_training_config_round_trips_neural_settings(tmp_path: Path) -> None:
    payload = make_config(tmp_path).to_dict()
    payload["training"].update(
        {
            "trainer_type": "maskable_ppo",
            "policy_type": "neural",
            "neural_hidden_sizes": [32, 16],
            "learning_rate": 0.001,
        }
    )

    config = LabConfig.from_dict(payload)

    assert config.training.trainer_type == "maskable_ppo"
    assert config.training.policy_type == "neural"
    assert config.training.neural_hidden_sizes == (32, 16)


def test_create_policy_from_name_preserves_baseline_linear_policy() -> None:
    policy = create_policy_from_name("trained_policy", weights_payload={"rear_drive": 2.5})

    assert isinstance(policy, TrainableLinearPolicy)
    assert policy.weights.rear_drive == 2.5


def test_create_policy_from_name_requires_config_for_neural_policy() -> None:
    try:
        create_policy_from_name("neural_policy")
    except ValueError as exc:
        assert "requires config" in str(exc)
    else:
        raise AssertionError("Expected neural policy creation without config to fail clearly")


def test_load_playable_policy_keeps_random_policy_alias(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    policy = _load_playable_policy(config, policy_mode="random_policy")

    assert isinstance(policy, RandomPolicy)


def test_load_playable_policy_uses_instinct_only_mode(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    policy = _load_playable_policy(config, policy_mode="instinct_only")

    assert isinstance(policy, InstinctOnlyPolicy)


def test_training_manager_live_replay_writes_latest_replay(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "generated"
    config = replace(
        make_config(artifacts),
        training=replace(
            make_config(artifacts).training,
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        ),
    )
    manager = TrainingManager()

    import sheepdog.server as server_module

    original_config = server_module.LabConfig

    class TestConfig:
        def __new__(cls):
            return config

    server_module.LabConfig = TestConfig
    try:
        replay = manager.run_live_replay(11)
    finally:
        server_module.LabConfig = original_config

    assert replay["seed"] == 11
    assert (generated / "latest-replay.json").exists()
