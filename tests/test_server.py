"""Tests for the interactive training server."""

# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-lines
from __future__ import annotations

import threading
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog import server as server_module
from sheepdog.server import TrainingManager, TrainingRequestHandler


def test_clear_training_restores_untrained_baseline(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    generated.mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    (artifacts / "training-state.json").write_text("{}", encoding="utf-8")
    (artifacts / "training-summary.json").write_text("{}", encoding="utf-8")
    (generated / "latest-replay.json").write_text("{}", encoding="utf-8")

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        payload, status = manager.clear()

    assert status == 200
    # clear() is async: it returns immediately with "Clearing..." and finishes
    # the baseline export in a background thread.  Poll until the thread is done.
    assert payload["message"] in {
        "Clearing... restoring baseline replay",
        "Training cleared. Baseline replay restored",
    }
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        phase = manager.snapshot().get("phase")
        if phase == "idle":
            break
        time.sleep(0.1)
    assert (artifacts / "training-state.json").exists()
    assert (artifacts / "training-summary.json").exists()
    assert (artifacts / "checkpoints" / "checkpoint-000000.json").exists()
    assert (artifacts / "evaluations" / "evaluation-checkpoint-000000.json").exists()
    index_payload = json.loads((generated / "checkpoint-index.json").read_text(encoding="utf-8"))
    assert index_payload["checkpoints"][0]["checkpoint_episode"] == 0
    assert index_payload["latest"]["checkpoint_episode"] == 0
    assert generated.joinpath("latest-replay.json").exists()

    replay_payload = json.loads((generated / "latest-replay.json").read_text(encoding="utf-8"))
    final_snapshot = replay_payload["final_snapshot"]
    assert replay_payload["policy_name"] == "instinct_only"
    assert replay_payload["trainer_type"] == "baseline"
    assert replay_payload["policy_type"] == "instinct"
    assert replay_payload["replay_mode"] == "baseline"
    assert final_snapshot["grid_width"] == config.environment.width
    assert final_snapshot["grid_height"] == config.environment.height


def test_reset_journey_archives_and_resets_to_stage_one(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    (artifacts / "checkpoints" / "checkpoint-000123.json").write_text("{}", encoding="utf-8")
    (artifacts / "training-summary.json").write_text("{}", encoding="utf-8")
    (generated / "latest-replay.json").write_text("{}", encoding="utf-8")

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        payload, status = manager.reset_journey()

    assert status == 200
    assert payload["phase"] in {"clearing", "idle"}
    assert payload["curriculum_stage"] == 1

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        snap = manager.snapshot()
        if snap.get("phase") == "idle":
            break
        time.sleep(0.1)

    archive_root = artifacts / "archive"
    archived_runs = sorted(path for path in archive_root.glob("journey-*") if path.is_dir())
    assert archived_runs
    latest_archive = archived_runs[-1]
    assert (latest_archive / "checkpoints" / "checkpoint-000123.json").exists()
    assert (latest_archive / "training-summary.json").exists()

    settings_payload = json.loads(
        (artifacts / "training-settings.json").read_text(encoding="utf-8")
    )
    assert settings_payload["curriculum_stage"] == 1


@dataclass(frozen=True, slots=True)
class _FakeStats:
    steps: int = 1
    simulated_seconds: float = 1.0
    sheep_penned: int = 0
    timeout: bool = False
    terminated: bool = True
    success: bool = False
    stopped: bool = False
    stop_reason: str = "complete"
    reward_total: float = 0.0
    no_progress_steps: int = 0
    final_avg_distance_to_pen: float = 0.0
    final_flock_spread: float = 0.0
    role_distribution: dict[str, int] = None  # type: ignore[assignment]
    role_switches: int = 0
    collector_activations: int = 0
    blocker_activations: int = 0
    sheep_split_events: int = 0
    final_reward_breakdown: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_distribution", self.role_distribution or {})
        object.__setattr__(
            self, "final_reward_breakdown", self.final_reward_breakdown or {"total": 0.0}
        )


class _FakeSnapshot:
    def __init__(self, config: LabConfig) -> None:
        self._config = config

    def to_dict(self) -> dict[str, object]:
        return {
            "step": 1,
            "simulated_seconds": 1,
            "grid_width": self._config.environment.width,
            "grid_height": self._config.environment.height,
            "dogs": [
                {"index": index, "x": 1, "y": index + 1}
                for index in range(self._config.environment.dogs)
            ],
            "sheep": [
                {"index": index, "x": 2, "y": index + 2, "penned": False}
                for index in range(self._config.environment.sheep)
            ],
            "pen": {"origin": {"x": 1, "y": 1}, "width": 4, "height": 4, "opening": "left"},
            "penned_count": 0,
            "average_distance_to_pen": 0,
            "flock_spread": 0,
            "no_progress_steps": 0,
            "terminated": True,
            "timeout": False,
            "stopped": False,
            "success": False,
            "status": "complete",
        }


def test_run_live_replay_prefers_latest_trained_linear_artifact(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    checkpoints = artifacts / "checkpoints"
    checkpoints.mkdir(parents=True)
    generated.mkdir(parents=True)

    checkpoint_episode = 4
    checkpoint_payload = {
        "checkpoint_episode": checkpoint_episode,
        "total_training_episodes": checkpoint_episode,
        "policy_name": "trained_policy",
        "trainer_type": "hill_climb",
        "policy_type": "linear",
        "environment_config": {
            "width": 60,
            "height": 45,
            "dogs": 1,
            "sheep": 1,
        },
        "reward_config": {
            "progress_scale": 2.0,
            "sheep_penned_reward": 8.0,
            "flock_cohesion_scale": 0.35,
            "scatter_penalty_scale": 0.65,
            "time_penalty": 0.05,
            "no_progress_penalty": 1.0,
            "terminal_success_reward": 20.0,
            "terminal_failure_penalty": 12.0,
            "wall_pressure_penalty": 0.4,
            "wait_penalty": 0.05,
            "gate_progress_scale": 1.6,
            "gate_corridor_progress_scale": 0.8,
            "gate_alignment_scale": 1.0,
            "stalled_control_penalty": 0.45,
            "wrong_hold_penalty": 0.8,
            "instincts": {
                "enable_instinct_rewards": True,
                "debug_reward_breakdown": False,
                "curriculum_stage": 1,
            },
        },
    }
    (checkpoints / f"checkpoint-{checkpoint_episode:06d}.json").write_text(
        json.dumps(checkpoint_payload),
        encoding="utf-8",
    )
    (artifacts / "training-summary.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_episode": checkpoint_episode,
                        "checkpoint": f"checkpoint-{checkpoint_episode:06d}.json",
                    }
                ],
                "trainer_type": "hill_climb",
                "policy_type": "linear",
                "total_episodes_trained": checkpoint_episode,
            }
        ),
        encoding="utf-8",
    )

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )
    captured: dict[str, object] = {}

    class TestConfig:
        def __new__(cls):
            return config

    def fake_load_playable_policy(
        effective_config: LabConfig,
        *,
        checkpoint_episode: int | None = None,
        policy_mode: str | None = None,
    ) -> object:
        captured["config"] = effective_config
        captured["checkpoint_episode"] = checkpoint_episode
        captured["policy_mode"] = policy_mode
        return SimpleNamespace(name=policy_mode or "trained_policy")

    class FakeEnvironment:
        def __init__(self, effective_config: LabConfig) -> None:
            captured["environment_config"] = effective_config

        def run_policy(self, policy: object, seed: int, capture_replay: bool = False) -> object:
            del capture_replay
            active_config = captured["environment_config"]
            assert isinstance(active_config, LabConfig)
            return SimpleNamespace(
                seed=seed,
                policy_name=getattr(policy, "name", "trained_policy"),
                final_snapshot=_FakeSnapshot(active_config),
                stats=_FakeStats(),
                replay=[],
            )

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server._load_playable_policy", fake_load_playable_policy),
        patch("sheepdog.server.SheepdogEnvironment", FakeEnvironment),
    ):
        manager = TrainingManager()
        payload = manager.run_live_replay(11)

    effective_config = captured["config"]
    assert isinstance(effective_config, LabConfig)
    assert captured["checkpoint_episode"] == checkpoint_episode
    assert captured["policy_mode"] == "trained_policy"
    assert effective_config.environment.dogs == 1
    assert effective_config.environment.sheep == 1
    assert effective_config.rewards.instincts.curriculum_stage == 1
    assert payload["replay_mode"] == "trained_linear"
    assert payload["environment"]["dogs"] == 1
    assert payload["environment"]["sheep"] == 1


def test_run_live_replay_honors_effective_stage_for_baseline(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    generated.mkdir(parents=True)

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )
    captured: dict[str, object] = {}

    class TestConfig:
        def __new__(cls):
            return config

    def fake_load_playable_policy(
        effective_config: LabConfig,
        *,
        checkpoint_episode: int | None = None,
        policy_mode: str | None = None,
    ) -> object:
        del checkpoint_episode
        captured["config"] = effective_config
        return SimpleNamespace(name=policy_mode or "instinct_only")

    class FakeEnvironment:
        def __init__(self, effective_config: LabConfig) -> None:
            captured["environment_config"] = effective_config

        def run_policy(self, policy: object, seed: int, capture_replay: bool = False) -> object:
            del capture_replay
            active_config = captured["environment_config"]
            assert isinstance(active_config, LabConfig)
            return SimpleNamespace(
                seed=seed,
                policy_name=getattr(policy, "name", "instinct_only"),
                final_snapshot=_FakeSnapshot(active_config),
                stats=_FakeStats(),
                replay=[],
            )

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server._load_playable_policy", fake_load_playable_policy),
        patch("sheepdog.server.SheepdogEnvironment", FakeEnvironment),
    ):
        manager = TrainingManager()
        payload = manager.run_live_replay(
            17,
            policy_mode="instinct_only",
            effective_config={
                "enable_instinct_rewards": True,
                "curriculum_stage": 1,
                "debug_reward_breakdown": False,
            },
        )

    effective_config = captured["config"]
    assert isinstance(effective_config, LabConfig)
    assert effective_config.environment.dogs == 1
    assert effective_config.environment.sheep == 1
    assert effective_config.rewards.instincts.curriculum_stage == 1
    assert payload["replay_mode"] == "baseline"
    assert payload["environment"]["dogs"] == 1
    assert payload["environment"]["sheep"] == 1


def test_run_live_replay_applies_environment_overrides(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    generated.mkdir(parents=True)

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )
    captured: dict[str, object] = {}

    class TestConfig:
        def __new__(cls):
            return config

    def fake_load_playable_policy(
        effective_config: LabConfig,
        *,
        checkpoint_episode: int | None = None,
        policy_mode: str | None = None,
    ) -> object:
        del checkpoint_episode
        captured["config"] = effective_config
        return SimpleNamespace(name=policy_mode or "instinct_only")

    class FakeEnvironment:
        def __init__(self, effective_config: LabConfig) -> None:
            captured["environment_config"] = effective_config

        def run_policy(self, policy: object, seed: int, capture_replay: bool = False) -> object:
            del capture_replay
            active_config = captured["environment_config"]
            assert isinstance(active_config, LabConfig)
            return SimpleNamespace(
                seed=seed,
                policy_name=getattr(policy, "name", "instinct_only"),
                final_snapshot=_FakeSnapshot(active_config),
                stats=_FakeStats(),
                replay=[],
            )

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server._load_playable_policy", fake_load_playable_policy),
        patch("sheepdog.server.SheepdogEnvironment", FakeEnvironment),
    ):
        manager = TrainingManager()
        payload = manager.run_live_replay(
            42,
            policy_mode="instinct_only",
            environment_overrides={"sheep_personality_strength": 0.75},
        )

    effective_config = captured["environment_config"]
    assert isinstance(effective_config, LabConfig)
    assert effective_config.environment.sheep_personality_strength == 0.75
    assert payload["environment"]["sheep_personality_strength"] == 0.75


def test_initial_status_prefers_saved_stage_over_history_max(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)
    (artifacts / "training-settings.json").write_text(
        json.dumps(
            {
                "curriculum_stage": 2,
                "enable_instinct_rewards": True,
                "debug_reward_breakdown": False,
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "stage-history.json").write_text(json.dumps({"2": 50, "5": 300}), encoding="utf-8")

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        status = manager.snapshot()

    assert status["curriculum_stage"] == 2
    assert status["stage_history"] == {"2": 50, "5": 300}


def test_start_rejects_stage_below_current_journey_history(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    checkpoints = artifacts / "checkpoints"
    models = artifacts / "models"
    checkpoints.mkdir(parents=True)
    models.mkdir(parents=True)
    generated.mkdir(parents=True)

    model_path = models / "best-model.zip"
    model_path.write_text("model", encoding="utf-8")
    checkpoint = {
        "checkpoint": "checkpoint-000008.json",
        "checkpoint_episode": 8,
        "total_training_episodes": 8,
        "policy_state_path": str(model_path),
        "policy_type": "neural",
        "trainer_type": "maskable_ppo",
        "reward_config": {"instincts": {"curriculum_stage": 8}},
    }
    (artifacts / "training-summary.json").write_text(
        json.dumps({"checkpoints": [checkpoint]}), encoding="utf-8"
    )
    (checkpoints / "checkpoint-000008.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    (artifacts / "stage-history.json").write_text(json.dumps({"2": 50, "8": 122}), encoding="utf-8")

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        with pytest.raises(ValueError, match="below the current journey's highest stage 8"):
            manager.start(requested_episodes=2, fast_mode=True, curriculum_stage=2)


def test_rewind_rehydrates_active_model_state(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    checkpoints = artifacts / "checkpoints"
    models = artifacts / "models"
    checkpoints.mkdir(parents=True)
    models.mkdir(parents=True)
    generated.mkdir(parents=True)

    model_path = models / "best-model.zip"
    model_path.write_text("model", encoding="utf-8")
    checkpoint = {
        "checkpoint": "checkpoint-000008.json",
        "checkpoint_episode": 8,
        "total_training_episodes": 8,
        "policy_state_path": str(model_path),
        "policy_type": "neural",
        "trainer_type": "maskable_ppo",
        "reward_config": {"instincts": {"curriculum_stage": 8}},
    }
    (artifacts / "training-summary.json").write_text(
        json.dumps({"checkpoints": [checkpoint]}), encoding="utf-8"
    )
    (checkpoints / "checkpoint-000008.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    (artifacts / "stage-history.json").write_text(json.dumps({"8": 8}), encoding="utf-8")

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        payload, status = manager.rewind_to_stage(8)

    assert status == 200
    assert payload["curriculum_stage"] == 8
    assert payload["active_model_path"] == str(model_path)


def test_initial_status_loads_paused_training_session(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    session_dir = artifacts / "startup"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (artifacts / "training-settings.json").write_text(
        json.dumps(
            {
                "curriculum_stage": 3,
                "enable_instinct_rewards": True,
                "debug_reward_breakdown": False,
                "auto_promote": True,
            }
        ),
        encoding="utf-8",
    )
    session_payload = {
        "state": "paused",
        "requested_at": "2026-07-01T12:00:00Z",
        "remaining_episodes": 12,
        "training_request": {
            "episodes": 50,
            "fast_mode": True,
            "enable_instinct_rewards": True,
            "curriculum_stage": 3,
            "debug_reward_breakdown": False,
            "auto_promote": True,
            "promote_from_checkpoint_episode": None,
        },
        "status": {
            "message": "Pause requested; waiting for the current checkpoint to finish",
            "requested_episodes": 24,
            "batch_completed_episodes": 12,
            "curriculum_stage": 3,
        },
    }
    (session_dir / "training-session.json").write_text(
        json.dumps(session_payload), encoding="utf-8"
    )

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        manager = TrainingManager()
        status = manager.snapshot()

    assert status["phase"] == "paused"
    assert status["running"] is False
    assert status["resume_available"] is True
    assert status["resume_remaining_episodes"] == 12
    assert status["resume_request"]["episodes"] == 50
    assert status["message"].startswith("Pause requested")


def test_startup_auto_resumes_interrupted_running_session(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    session_dir = artifacts / "startup"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (session_dir / "training-session.json").write_text(
        json.dumps(
            {
                "state": "running",
                "remaining_episodes": 250,
                "training_request": {
                    "episodes": 500,
                    "fast_mode": True,
                    "enable_instinct_rewards": True,
                    "curriculum_stage": 6,
                    "debug_reward_breakdown": False,
                    "auto_promote": True,
                    "promote_from_checkpoint_episode": 249,
                },
            }
        ),
        encoding="utf-8",
    )

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    start_calls: list[dict[str, object]] = []

    def fake_start(
        _self,
        requested_episodes,
        fast_mode,
        *,
        enable_instinct_rewards=None,
        curriculum_stage=None,
        debug_reward_breakdown=None,
        auto_promote=None,
        promote_from_checkpoint_episode=None,
        resume=False,
    ):
        start_calls.append(
            {
                "requested_episodes": requested_episodes,
                "fast_mode": fast_mode,
                "enable_instinct_rewards": enable_instinct_rewards,
                "curriculum_stage": curriculum_stage,
                "debug_reward_breakdown": debug_reward_breakdown,
                "auto_promote": auto_promote,
                "promote_from_checkpoint_episode": promote_from_checkpoint_episode,
                "resume": resume,
            }
        )
        return {"running": True}

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch.object(TrainingManager, "start", autospec=True, side_effect=fake_start),
    ):
        TrainingManager()

    assert len(start_calls) == 1
    assert start_calls[0]["requested_episodes"] == 250
    assert start_calls[0]["fast_mode"] is True
    assert start_calls[0]["curriculum_stage"] == 6
    assert start_calls[0]["enable_instinct_rewards"] is True
    assert start_calls[0]["promote_from_checkpoint_episode"] == 249
    assert start_calls[0]["resume"] is True


def test_startup_auto_resumes_canonical_remaining_timesteps(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    session_dir = artifacts / "startup"
    generated.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (session_dir / "training-session.json").write_text(
        json.dumps(
            {
                "state": "running",
                "remaining_timesteps": 18_432,
                "training_request": {
                    "total_timesteps": 51_200,
                    "curriculum_stage": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    start_calls: list[dict[str, object]] = []

    def fake_start(_self, *args, **kwargs):
        start_calls.append({"args": args, **kwargs})
        return {"running": True}

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch.object(TrainingManager, "start", autospec=True, side_effect=fake_start),
    ):
        TrainingManager()

    assert start_calls == [
        {
            "args": (),
            "total_timesteps": 18_432,
            "enable_instinct_rewards": None,
            "curriculum_stage": 2,
            "debug_reward_breakdown": None,
            "auto_promote": None,
            "promote_from_checkpoint_episode": None,
            "resume": True,
        }
    ]


def test_launcher_resume_request_preserves_resume_semantics() -> None:
    launcher = Path(__file__).parents[1] / "start-app.ps1"
    launcher_text = launcher.read_text(encoding="utf-8")

    assert "$resumeRequest['resume'] = $true" in launcher_text


def test_start_endpoint_returns_json_conflict_when_start_is_rejected() -> None:
    manager = cast(
        TrainingManager,
        SimpleNamespace(
            _status={"phase": "idle"},
            start=lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("Requested stage 8 does not match active stage 2")
            ),
        ),
    )
    original_manager = TrainingRequestHandler.manager
    TrainingRequestHandler.manager = manager
    server = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        body = json.dumps({"episodes": 10, "curriculum_stage": 8}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/training/start",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error_info:
            urllib.request.urlopen(request)

        assert error_info.value.code == 409
        response = json.loads(error_info.value.read().decode("utf-8"))
        assert response == {
            "error": "Requested stage 8 does not match active stage 2",
            "code": "TRAINING_START_REJECTED",
        }
    finally:
        server.shutdown()
        server.server_close()
        TrainingRequestHandler.manager = original_manager


def test_start_endpoint_accepts_canonical_timestep_request() -> None:
    start_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def start(*args, **kwargs):
        start_calls.append((args, kwargs))
        return {"running": True, "requested_timesteps": kwargs["total_timesteps"]}

    manager = cast(
        TrainingManager,
        SimpleNamespace(_status={"phase": "idle"}, start=start),
    )
    original_manager = TrainingRequestHandler.manager
    TrainingRequestHandler.manager = manager
    server = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        body = json.dumps({"total_timesteps": 50_000, "curriculum_stage": 2}).encode(
            "utf-8"
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/training/start",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["requested_timesteps"] == 50_000
        assert start_calls == [
            (
                (),
                {
                    "total_timesteps": 50_000,
                    "enable_instinct_rewards": None,
                    "curriculum_stage": 2,
                    "debug_reward_breakdown": None,
                    "auto_promote": None,
                    "promote_from_checkpoint_episode": None,
                    "evaluation_mode": "quick",
                    "resume": False,
                },
            )
        ]
    finally:
        server.shutdown()
        server.server_close()
        TrainingRequestHandler.manager = original_manager


def test_training_manager_error_handling_on_setup_failure(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)

    config = LabConfig(
        training=TrainingConfig(
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch(
            "sheepdog.server._build_training_job_config",
            side_effect=ValueError("Simulated setup error"),
        ),
    ):
        manager = TrainingManager()
        manager.start(requested_episodes=10, fast_mode=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snap = manager.snapshot()
            if not snap["running"]:
                break
            time.sleep(0.05)

        snap = manager.snapshot()
        assert not snap["running"]
        assert snap["phase"] == "error"
        assert snap["error_type"] == "ValueError"
        assert "Simulated setup error" in str(snap["error"])
        assert "ValueError: Simulated setup error" in str(snap["traceback"])


def test_stage_25_does_not_plateau_stop_before_max_stage(tmp_path: Path) -> None:
    base_config = LabConfig()
    config = replace(
        base_config,
        training=replace(
            base_config.training,
            output_dir=str(tmp_path / "artifacts"),
            web_export_dir=str(tmp_path / "web" / "public" / "generated"),
        ),
    )
    artifacts = Path(config.training.output_dir)
    generated = Path(config.training.web_export_dir)
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)

    class TestConfig:
        def __new__(cls):
            return config

    class FakeTrainer:
        total_episodes_trained = 0

        def train(self, progress_callback=None, should_stop=None):
            assert progress_callback is not None
            assert should_stop is not None
            for checkpoint_episode in range(21):
                progress_callback(
                    {
                        "phase": "checkpoint",
                        "batch_completed_episodes": checkpoint_episode + 1,
                        "current_episode": checkpoint_episode,
                        "total_episodes_trained": checkpoint_episode,
                        "checkpoint_episode": checkpoint_episode,
                        "best_score": 0.0,
                        "summary": {
                            "success_rate": 0.0,
                            "average_reward": -100.0,
                            "timeout_rate": 0.0,
                            "average_sheep_penned": 0.0,
                            "average_completion_steps": 100.0,
                            "average_no_progress_steps": 0.0,
                            "average_distance_to_pen": 0.0,
                            "average_flock_spread": 0.0,
                            "average_farthest_distance_to_pen": 0.0,
                            "average_farthest_distance_to_flock_center": 0.0,
                            "records": [
                                {"seed": 11, "success": False, "replay_path": "replay-11.json"},
                                {"seed": 23, "success": False, "replay_path": "replay-23.json"},
                                {"seed": 37, "success": False, "replay_path": "replay-37.json"},
                            ],
                        },
                        "replay_path": str(artifacts / "replay.json"),
                        "message": f"Checkpoint {checkpoint_episode} exported",
                    }
                )

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=FakeTrainer()),
        patch("sheepdog.server.CurriculumTelemetryManager.initialize_wandb"),
    ):
        manager = TrainingManager()
        manager.start(requested_episodes=21, fast_mode=True, curriculum_stage=25)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snap = manager.snapshot()
            if not snap["running"]:
                break
            time.sleep(0.05)

    snap = manager.snapshot()
    assert snap["auto_promote_gate"]["reason"] == "Promotion criteria not met yet"
    assert snap["auto_promote_gate"]["decision"] == "hold"


def test_start_training_auto_archives_existing_if_not_resuming(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    (artifacts / "checkpoints" / "checkpoint-000123.json").write_text("{}", encoding="utf-8")
    (artifacts / "training-summary.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_episode": 123,
                        "checkpoint": "checkpoint-000123.json",
                        "reward_config": {"instincts": {"curriculum_stage": 2}},
                        "records": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    class FakeTrainer:
        total_episodes_trained = 0

        def train(self, progress_callback=None, should_stop=None):
            assert progress_callback is not None
            assert should_stop is not None

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=FakeTrainer()),
    ):
        manager = TrainingManager()
        manager.start(
            requested_episodes=1,
            fast_mode=True,
            curriculum_stage=3,
        )

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            snap = manager.snapshot()
            if not snap["running"]:
                break
            time.sleep(0.1)

    archive_root = artifacts / "archive"
    archived_runs = sorted(path for path in archive_root.glob("journey-*") if path.is_dir())
    assert archived_runs
    latest_archive = archived_runs[-1]
    assert (latest_archive / "checkpoints" / "checkpoint-000123.json").exists()
    assert (latest_archive / "training-summary.json").exists()


def test_auto_promotion_updates_batch_episodes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)

    config = LabConfig(
        training=TrainingConfig(
            episodes=50,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    configs_seen = []

    class FakeTrainer:
        total_episodes_trained = 0

        def __init__(self, cfg, output_dir):
            self.cfg = cfg
            self.output_dir = output_dir
            configs_seen.append(cfg)

        def train(self, progress_callback=None, should_stop=None):
            assert progress_callback is not None
            assert should_stop is not None
            # When Stage 1 runs, trigger early promotion
            if self.cfg.rewards.instincts.curriculum_stage == 1:
                raise server_module._EarlyPromotionSignal(  # pylint: disable=protected-access
                    checkpoint_episode=10,
                    best_success=1.0,
                    qualified_streak=3,
                    seed_gate_hits=3,
                    full_success_hits=3,
                )
            else:
                # For Stage 2, just raise an error to break the infinite loop once verified
                raise ValueError("Stage 2 reached successfully")

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", side_effect=FakeTrainer),
    ):
        manager = TrainingManager()
        manager.start(
            requested_episodes=50,
            fast_mode=True,
            curriculum_stage=1,
            auto_promote=True,
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            snap = manager.snapshot()
            if not snap["running"]:
                break
            time.sleep(0.05)

    snap = manager.snapshot()
    assert not snap["running"]
    assert snap["phase"] == "error"
    assert "Stage 2 reached successfully" in str(snap["error"])

    assert len(configs_seen) >= 2
    assert configs_seen[-2].rewards.instincts.curriculum_stage == 1
    assert configs_seen[-2].training.episodes == 49

    assert configs_seen[-1].rewards.instincts.curriculum_stage == 2
    assert configs_seen[-1].training.episodes == 74


def test_diagnostics_endpoint_route_integration(tmp_path: Path) -> None:
    # Setup directories
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    generated.mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    (artifacts / "training-state.json").write_text("{}", encoding="utf-8")
    (artifacts / "training-summary.json").write_text("{}", encoding="utf-8")
    (generated / "checkpoint-index.json").write_text(
        '{"checkpoints": [], "latest": null}', encoding="utf-8"
    )

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    with patch("sheepdog.server.LabConfig", TestConfig):
        # We start the server in a background thread on a free port, e.g. 51829
        port = 51829
        server = ThreadingHTTPServer(("127.0.0.1", port), TrainingRequestHandler)

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            # Wait a short moment for the server to bind
            time.sleep(0.5)

            # Query the diagnostics endpoint
            url = f"http://127.0.0.1:{port}/api/training/diagnostics"
            req = urllib.request.Request(url)
            try:
                with urllib.request.urlopen(req) as response:
                    assert response.status == 200
                    data = json.loads(response.read().decode("utf-8"))
                    assert "diagnosticsAvailable" in data
                    assert data["diagnosticsAvailable"] is True
                    assert "snapshot" in data
                    assert data["snapshot"] is not None
                    assert "error" in data
                    assert data["error"] is None
            except urllib.error.HTTPError as err:
                # Under test conditions without check points, a 400 bad request error is raised.
                # The response must still conform to the diagnostics contract.
                assert err.status in (400, 404, 500)
                body = err.read().decode("utf-8")
                data = json.loads(body)
                assert data["diagnosticsAvailable"] is False
                assert data["snapshot"] is None
                assert "error" in data
                assert data["error"] is not None
                assert "code" in data["error"]
                assert "message" in data["error"]
        finally:
            server.shutdown()
            server.server_close()


def test_log_message_suppresses_routine_polling_noise() -> None:
    from http.server import BaseHTTPRequestHandler
    from unittest.mock import MagicMock, patch
    from sheepdog.server import TrainingRequestHandler

    handler = object.__new__(TrainingRequestHandler)

    logged_messages = []
    def mock_super_log(format, *args):
        logged_messages.append(format % args if args else format)

    with patch.object(BaseHTTPRequestHandler, "log_message", side_effect=mock_super_log):
        # 1. Routine 200 OK polling GET logs should be suppressed
        handler.log_message('"GET /api/training/status HTTP/1.1" 200 -')
        handler.log_message('"GET /api/training/history HTTP/1.1" 200 -')
        handler.log_message('"GET /api/health HTTP/1.1" 200 -')
        assert len(logged_messages) == 0

        # 2. Error logs or non-polling logs should still be emitted
        handler.log_message('"GET /api/training/status HTTP/1.1" 500 -')
        handler.log_message('"POST /api/training/start HTTP/1.1" 200 -')
        assert len(logged_messages) == 2
        assert '500' in logged_messages[0]
        assert 'POST /api/training/start' in logged_messages[1]


def test_health_endpoint_routes(tmp_path: Path) -> None:
    """Test that GET /health and GET /api/health return 200 OK with status info and CORS headers."""
    import json
    import urllib.request
    from unittest.mock import MagicMock, patch
    from sheepdog.server import ThreadingHTTPServer, TrainingRequestHandler

    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "evaluations").mkdir(parents=True)
    generated.mkdir(parents=True)
    (generated / "replays").mkdir(parents=True)

    with patch("sheepdog.server.LabConfig") as mock_lab_config:
        mock_cfg = MagicMock()
        mock_cfg.training.output_dir = str(artifacts)
        mock_cfg.training.web_export_dir = str(generated)
        mock_cfg.training.runtime_heartbeat_seconds = 60
        mock_lab_config.return_value = mock_cfg

        server = ThreadingHTTPServer(("127.0.0.1", 0), TrainingRequestHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            for path in ("/health", "/api/health"):
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
                with urllib.request.urlopen(req) as resp:
                    assert resp.status == 200
                    assert resp.headers.get("Access-Control-Allow-Origin") == "*"
                    data = json.loads(resp.read().decode("utf-8"))
                    assert data.get("status") == "ok"
                    assert data.get("service") == "sheepdog-api"
        finally:
            server.shutdown()
            server.server_close()


def test_worker_exception_handling_unexpected_vs_deliberate_shutdown(tmp_path: Path) -> None:
    """Verify worker thread logs/persists unexpected exceptions, but re-raises KeyboardInterrupt and SystemExit after cleanup."""
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)

    config = LabConfig(
        training=TrainingConfig(
            episodes=1,
            output_dir=str(artifacts),
            web_export_dir=str(generated),
        )
    )

    class TestConfig:
        def __new__(cls):
            return config

    # 1. Test unexpected exception (ValueError) is caught & status set to error
    class FailingTrainerValueError:
        total_episodes_trained = 0
        def train(self, progress_callback=None, should_stop=None):
            raise ValueError("Unexpected worker failure")

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=FailingTrainerValueError()),
    ):
        manager = TrainingManager()
        manager._run_training(1, True)
        snap = manager.snapshot()
        assert snap["running"] is False
        assert snap["phase"] == "error"
        assert "Unexpected worker failure" in snap["error"]

    # 2. Test deliberate shutdown exception (KeyboardInterrupt) performs cleanup & re-raises
    class FailingTrainerKeyboardInterrupt:
        total_episodes_trained = 0
        def train(self, progress_callback=None, should_stop=None):
            raise KeyboardInterrupt("Deliberate shutdown signal")

    with (
        patch("sheepdog.server.LabConfig", TestConfig),
        patch("sheepdog.server.create_trainer", return_value=FailingTrainerKeyboardInterrupt()),
    ):
        manager = TrainingManager()
        with pytest.raises(KeyboardInterrupt):
            manager._run_training(1, True)
        snap = manager.snapshot()
        assert snap["running"] is False
        assert snap["phase"] == "error"
        assert "KeyboardInterrupt" in snap["error_type"]




