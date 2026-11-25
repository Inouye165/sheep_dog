"""Exercise the continuous controller with bounded, isolated fake batches."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sheepdog import server
from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.curriculum import CURRICULUM_STAGES

_restore_active_run_state = server.TrainingManager.restore_active_run_state


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> server.TrainingManager:
    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(tmp_path),
            web_export_dir=str(tmp_path / "web"),
            backup_dir=str(tmp_path / "backups"),
            backup_enabled=False,
        )
    )
    monkeypatch.setattr(server, "LabConfig", lambda: config)
    monkeypatch.setattr(server.TrainingManager, "_reconcile_web_exports", lambda self: None)
    monkeypatch.setattr(server.TrainingManager, "_run_watchdog", lambda self: None)
    monkeypatch.setattr(server.TrainingManager, "restore_active_run_state", lambda self: None)
    monkeypatch.setattr(server, "ensure_free_disk_space", lambda *args, **kwargs: None)
    return server.TrainingManager()


def test_repeat_then_promote_and_finish_on_mastery(
    manager: server.TrainingManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    highest = max(CURRICULUM_STAGES)
    stages = [highest - 1, highest - 1, highest, highest]
    checkpoints = iter(range(101, 105))
    jobs: list[tuple[int, int]] = []

    def gate(_root: Path, episode: int, **_kwargs: object) -> dict[str, object]:
        ready = episode in (102, 104)
        return {
            "ready": ready,
            "decision": "promote_ready" if ready else "hold",
            "best_success": 1.0 if ready else 0.2,
            "qualified_streak": server.MAX_STAGE_MASTERY_QUALIFIED_STREAK if ready else 0,
            "full_success_hits": server.MAX_STAGE_MASTERY_FULL_SUCCESS_HITS if ready else 0,
        }

    def trainer_for(config: LabConfig, _root: str) -> object:
        stage = config.rewards.instincts.curriculum_stage
        jobs.append((stage, config.training.episodes))
        episode = next(checkpoints)
        seeds = config.training.evaluation_seeds
        trainer = Mock(
            spec=["train", "total_episodes_trained", "_loaded_state", "prefer_latest_checkpoint", "runtime_tracker"]
        )
        trainer.total_episodes_trained = episode - 1
        trainer._loaded_state = {"run_id": "test", "total_environment_episodes": episode * 5}
        trainer.prefer_latest_checkpoint = False
        monkeypatch.setattr(
            manager,
            "get_checkpoint_details",
            lambda _episode: {"curriculum_stage": stage, "evaluation_seeds": list(seeds)},
        )

        def train(*, progress_callback, should_stop):
            assert not should_stop()
            assert trainer.prefer_latest_checkpoint
            progress_callback({
                "phase": "checkpoint",
                "checkpoint_episode": episode,
                "total_episodes_trained": episode,
                "batch_completed_segments": 1,
                "environment_episodes_total": episode * 5 + 5,
                "summary": {
                    "curriculum_stage": stage,
                    "success_rate": 1.0 if episode in (102, 104) else 0.2,
                    "average_reward": 2.0,
                    "evaluation_seed_count": len(seeds),
                    "records": [],
                },
            })

        trainer.train.side_effect = train
        return trainer

    monkeypatch.setattr(server, "create_trainer", trainer_for)
    monkeypatch.setattr(server, "compute_promotion_gate_snapshot", gate)
    monkeypatch.setattr(server, "PLATEAU_STOP_MIN_CHECKPOINTS", 0)
    monkeypatch.setattr(server, "PLATEAU_STOP_NO_IMPROVEMENT_STREAK", 0)
    manager._run_training(7, True, curriculum_stage=highest - 1, auto_promote=True, continuous=True)

    assert jobs == [(stage, 6) for stage in stages]
    assert manager._status["phase"] == "complete"
    assert "sustained mastery" in manager._status["message"]
    assert manager._status["stage_history"] == {str(highest - 1): 10, str(highest): 10}
    assert server._read_training_session_state(tmp_path) is None
    revisions = json.loads((tmp_path / "config-history.json").read_text())["revisions"]
    assert len(revisions) == 2


@pytest.mark.parametrize("state", ["paused", "stopped", "running"])
def test_explicit_resume_preserves_original_batch_at_boundary(
    manager: server.TrainingManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    request = manager._training_request_payload(7, True, curriculum_stage=1, continuous=True)
    server._write_training_session_state(
        tmp_path,
        {"state": state, "training_request": request, "remaining_episodes": 0},
    )
    manager._status = manager._initial_status()
    worker = Mock()
    thread = Mock(return_value=worker)
    monkeypatch.setattr(server.threading, "Thread", thread)

    assert manager._status["resume_available"]
    assert manager._status["phase"] == ("stopped" if state == "running" else state)
    manager._resume_interrupted_session()
    thread.assert_not_called()
    assert server._read_training_session_state(tmp_path)["training_request"] == request

    manager.start(1, continuous=True, resume=True)
    assert thread.call_args.kwargs["args"] == (7, True)
    assert thread.call_args.kwargs["kwargs"]["continuous"]
    assert thread.call_args.kwargs["kwargs"]["resume"]


def test_shutdown_preserves_request_after_worker_already_paused(
    manager: server.TrainingManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sheepdog.training.episode_store as episode_store

    monkeypatch.setattr(episode_store, "get_episode_store", Mock(return_value=Mock()))
    manager._status["continuous"] = True
    manager._status["resume_request"] = manager._training_request_payload(7, True, continuous=True)
    manager.stop()
    marker = server._read_training_session_state(tmp_path)
    assert marker["state"] == "stopped"
    assert marker["training_request"]["episodes"] == 7


def test_restart_recovers_promotion_recorded_before_the_next_stage_marker(
    manager: server.TrainingManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = manager._training_request_payload(7, True, curriculum_stage=37, continuous=True)
    server._write_training_session_state(tmp_path, {
        "state": "running",
        "training_request": previous,
        "status": {"curriculum_stage": 37, "run_id": "same-run"},
    })
    server._append_promotion_history(tmp_path, {
        "event_type": "promotion",
        "run_id": "same-run",
        "from_stage": 37,
        "to_stage": 38,
        "trigger_checkpoint_episode": 101,
    })
    monkeypatch.setattr(manager, "_resolve_precedence_state", lambda _root: {
        "run_id": "same-run",
        "active_curriculum_stage": 38,
        "active_checkpoint_episode": 101,
    })
    _restore_active_run_state(manager)
    assert manager._status["phase"] == "stopped"
    assert manager._status["resume_available"]
    request = manager._status["resume_request"]
    assert request["curriculum_stage"] == 38
    assert request["promote_from_checkpoint_episode"] == 101
    assert request["episodes"] == 7
    assert server._read_training_session_state(tmp_path)["training_request"] == request


@pytest.mark.parametrize("state", ["paused", "stopped"])
def test_active_control_is_not_overwritten_by_next_batch(
    manager: server.TrainingManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    import sheepdog.training.episode_store as episode_store

    monkeypatch.setattr(episode_store, "get_episode_store", Mock(return_value=Mock()))
    launched = 0

    def trainer_for(_config: LabConfig, _root: str) -> object:
        nonlocal launched
        launched += 1
        trainer = Mock(spec=["train", "total_episodes_trained", "_loaded_state", "runtime_tracker"])
        trainer.total_episodes_trained = 0
        trainer._loaded_state = {"run_id": "test"}

        def train(*, progress_callback, should_stop):
            assert not should_stop()
            manager.pause() if state == "paused" else manager.stop()
            # The worker sees the same signal even if a callback reported work.
            progress_callback({"phase": "training", "batch_completed_segments": 1})
            assert should_stop()

        trainer.train.side_effect = train
        return trainer

    monkeypatch.setattr(server, "create_trainer", trainer_for)
    manager._thread = Mock()
    manager._thread.is_alive.return_value = True
    manager._run_training(7, True, curriculum_stage=1, auto_promote=True, continuous=True)
    assert launched == 1
    assert manager._status["phase"] == state
    assert manager._status["resume_request"]["continuous"]
    marker = server._read_training_session_state(tmp_path)
    assert marker["state"] == state
    assert marker["training_request"]["episodes"] == 7
    manager.runtime_tracker.end_session("test")


def test_latest_checkpoint_loads_its_weights_and_optimizer_after_completed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch as th

    from sheepdog.policies.neural import NeuralPolicy
    from sheepdog.training.maskable_ppo import MaskablePPOTrainer

    config = LabConfig(training=TrainingConfig(
        output_dir=str(tmp_path),
        web_export_dir=str(tmp_path / "web"),
        neural_hidden_sizes=(8,),
        ppo_env_workers=1,
        rollout_steps=8,
        batch_size=4,
        checkpoint_episodes=(0,),
        total_timesteps=8,
    ))
    original = NeuralPolicy.initialize(config)
    best = tmp_path / "best.zip"
    latest = tmp_path / "latest.zip"
    original.model.save(str(best))
    original_parameter = next(original.model.policy.parameters())
    with th.no_grad():
        original_parameter.fill_(0.42)
    original.model.policy.optimizer.state[original_parameter] = {
        "step": th.tensor(9.0),
        "exp_avg": th.full_like(original_parameter, 0.25),
        "exp_avg_sq": th.full_like(original_parameter, 0.5),
    }
    original.model.save(str(latest))
    original.model.get_env().close()
    server.atomic_write_json(tmp_path / "training-state.json", {
        "policy_state_path": str(latest),
        "best_model_path": str(best),
        "best_model_curriculum_stage": config.rewards.instincts.curriculum_stage,
        "total_episodes_trained": 2,
        "policy_config": original.config.to_dict(),
        "incomplete_batch": None,
    })

    class LoadingVerified(Exception):
        pass

    load_policy = NeuralPolicy.load

    def verify_loading(path, config, policy_config):
        assert path == latest
        restored = load_policy(path, config, policy_config)
        try:
            parameter = next(restored.model.policy.parameters())
            optimizer_state = restored.model.policy.optimizer.state[parameter]
            assert th.allclose(parameter, th.full_like(parameter, 0.42))
            assert optimizer_state["step"].item() == 9.0
            assert th.allclose(optimizer_state["exp_avg"], th.full_like(parameter, 0.25))
        finally:
            restored.model.get_env().close()
        raise LoadingVerified

    trainer = MaskablePPOTrainer(config, tmp_path)
    trainer.prefer_latest_checkpoint = True
    monkeypatch.setattr(NeuralPolicy, "load", staticmethod(verify_loading))

    import sheepdog.training.episode_store as episode_store

    monkeypatch.setattr(episode_store, "get_episode_store", Mock(return_value=Mock(get_max_episode=Mock(return_value=0))))
    with pytest.raises(LoadingVerified):
        trainer.train()
