"""Tests for the interactive training server."""

# pylint: disable=missing-function-docstring,missing-class-docstring
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sheepdog.config import LabConfig, TrainingConfig
from sheepdog.server import TrainingManager


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

    manager = TrainingManager()

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

    manager = TrainingManager()

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

    manager = TrainingManager()

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

    manager = TrainingManager()

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
