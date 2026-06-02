"""Tests for saved scenario definitions and per-scenario evaluation."""

# pylint: disable=missing-function-docstring,protected-access
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.scenario_evaluator import (
    ScenarioResultsStore,
    evaluate_scenario,
    is_strictly_better_scenario_result,
    resolve_checkpoint_episode,
)
from sheepdog.evaluation.scenarios import (
    AgentLayout,
    PenLayout,
    Scenario,
    ScenarioStore,
    scenario_from_snapshot,
)
from sheepdog.policies.heuristic import InstinctOnlyPolicy
from sheepdog.server import TrainingManager, _parse_scenario_action_path


def make_config(output_dir: Path) -> LabConfig:
    return LabConfig(
        environment=EnvironmentConfig(max_steps=40, dogs=2, sheep=2, width=24, height=20),
        rewards=RewardConfig(),
        training=TrainingConfig(
            episodes=0,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(output_dir),
            web_export_dir=str(output_dir / "web" / "generated"),
        ),
    )


def sample_scenario() -> Scenario:
    return Scenario(
        id="sc-test-1",
        name="tight flock",
        created_at="2026-01-01T00:00:00+00:00",
        seed=42,
        width=24,
        height=20,
        dogs=(AgentLayout(0, 2, 10), AgentLayout(1, 4, 10)),
        sheep=(AgentLayout(0, 12, 10, "pen_shy"), AgentLayout(1, 14, 11, "escapist")),
        pen=PenLayout(origin_x=18, origin_y=1, width=5, height=5, opening="left"),
        sheep_personality_strength=0.35,
        description="difficult pen approach",
    )


def test_scenario_store_round_trip(tmp_path: Path) -> None:
    store = ScenarioStore(tmp_path / "scenarios")
    scenario = sample_scenario()
    store.save(scenario)
    loaded = store.get(scenario.id)
    assert loaded is not None
    assert loaded.name == scenario.name
    assert loaded.description == "difficult pen approach"
    assert loaded.sheep[0].personality == "pen_shy"
    payload = json.loads((tmp_path / "scenarios" / "scenarios.json").read_text(encoding="utf-8"))
    assert len(payload["scenarios"]) == 1


def test_scenario_from_snapshot_captures_layout() -> None:
    snapshot = {
        "grid_width": 30,
        "grid_height": 25,
        "dogs": [{"index": 0, "x": 1, "y": 2}],
        "sheep": [{"index": 0, "x": 5, "y": 6, "personality": "bold"}],
        "pen": {"origin": {"x": 20, "y": 3}, "width": 6, "height": 4, "opening": "right"},
    }
    scenario = scenario_from_snapshot(name="capture", seed=7, snapshot=snapshot, description="note")
    assert scenario.width == 30
    assert scenario.dogs[0].x == 1
    assert scenario.sheep[0].personality == "bold"
    assert scenario.pen.opening == "right"
    assert scenario.description == "note"


def test_reset_from_scenario_is_deterministic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    scenario = sample_scenario()
    run_config = LabConfig(
        environment=EnvironmentConfig(
            max_steps=config.environment.max_steps,
            dogs=len(scenario.dogs),
            sheep=len(scenario.sheep),
            width=scenario.width,
            height=scenario.height,
            pen_width=scenario.pen.width,
            pen_height=scenario.pen.height,
            pen_opening=scenario.pen.opening,
        ),
        rewards=config.rewards,
        training=config.training,
    )
    first = SheepdogEnvironment(run_config).reset_from_scenario(scenario)
    second = SheepdogEnvironment(run_config).reset_from_scenario(scenario)
    assert first.dogs[0].x == second.dogs[0].x
    assert first.sheep[0].personality == second.sheep[0].personality == "pen_shy"
    assert first.dogs[0].x == 2
    assert first.sheep[1].x == 14


def test_is_strictly_better_scenario_result_ordering() -> None:
    success = {
        "success": True,
        "sheep_penned": 2,
        "steps": 100,
        "reward_total": 10.0,
    }
    partial = {
        "success": False,
        "sheep_penned": 1,
        "steps": 50,
        "reward_total": 20.0,
    }
    assert is_strictly_better_scenario_result(success, partial)
    assert not is_strictly_better_scenario_result(partial, success)
    faster = {**success, "steps": 80}
    assert is_strictly_better_scenario_result(faster, success)


def test_evaluate_scenario_records_replay_and_best(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = ScenarioStore(tmp_path / "scenarios")
    store.save(sample_scenario())
    policy = InstinctOnlyPolicy()

    result = evaluate_scenario(config, policy, sample_scenario(), checkpoint_episode=3)
    assert result.scenario_id == "sc-test-1"
    assert result.checkpoint_episode == 3
    assert result.replay_path.startswith("/generated/replays/")
    assert (tmp_path / "web" / "generated" / "replays").exists()

    results_store = ScenarioResultsStore(tmp_path / "scenarios")
    best = results_store.best_for_scenario("sc-test-1")
    assert best is not None
    assert best["checkpoint_episode"] == 3
    assert best["replay_path"] == result.replay_path

    index_path = tmp_path / "web" / "generated" / "scenario-index.json"
    assert index_path.exists()


def test_best_per_scenario_updates_only_when_improved(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = ScenarioStore(tmp_path / "scenarios")
    store.save(sample_scenario())
    policy = InstinctOnlyPolicy()
    scenario = sample_scenario()

    first = evaluate_scenario(config, policy, scenario, checkpoint_episode=1)
    results_store = ScenarioResultsStore(tmp_path / "scenarios")
    best_after_first = results_store.best_for_scenario(scenario.id)
    assert best_after_first is not None
    assert best_after_first["checkpoint_episode"] == 1

    results_store.record_run(
        type(first)(
            scenario_id=scenario.id,
            checkpoint_episode=2,
            success=False,
            sheep_penned=0,
            steps=999,
            timeout=True,
            stopped=False,
            reward_total=-50.0,
            replay_path="/generated/replays/worse.json",
        )
    )
    best_after_worse = results_store.best_for_scenario(scenario.id)
    assert best_after_worse is not None
    assert best_after_worse["checkpoint_episode"] == 1
    assert best_after_worse["steps"] == first.steps


def test_resolve_checkpoint_episode_latest_and_specific(tmp_path: Path) -> None:
    make_config(tmp_path)
    output_root = tmp_path
    summary = {
        "checkpoints": [
            {"checkpoint_episode": 10, "success_rate": 0.2},
            {"checkpoint_episode": 20, "success_rate": 0.5},
        ]
    }
    (output_root / "training-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    latest = resolve_checkpoint_episode("latest", output_root=output_root)
    assert latest == 20
    specific = resolve_checkpoint_episode(
        "specific",
        output_root=output_root,
        explicit_episode=10,
    )
    assert specific == 10


def test_parse_scenario_action_path() -> None:
    assert _parse_scenario_action_path("/api/scenarios/abc123/evaluate") == ("abc123", "evaluate")
    assert _parse_scenario_action_path("/api/scenarios/abc123/replay") == ("abc123", "replay")
    assert _parse_scenario_action_path("/api/scenarios") is None


def test_latest_checkpoint_can_differ_from_global_best(tmp_path: Path) -> None:
    output_root = tmp_path
    summary = {
        "checkpoints": [
            {"checkpoint_episode": 1, "success_rate": 0.9, "average_completion_steps": 50},
            {"checkpoint_episode": 2, "success_rate": 0.3, "average_completion_steps": 40},
        ]
    }
    (output_root / "training-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_root / "training-state.json").write_text(
        json.dumps({"best_formal_episode": 1}),
        encoding="utf-8",
    )
    latest = resolve_checkpoint_episode("latest", output_root=output_root)
    best = resolve_checkpoint_episode(
        "global_best",
        output_root=output_root,
        web_export_dir=tmp_path / "web" / "generated",
    )
    assert latest == 2
    assert best == 1
    assert latest != best


def test_server_save_and_evaluate_scenario(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    generated = tmp_path / "web" / "public" / "generated"
    artifacts.mkdir(parents=True)
    generated.mkdir(parents=True)
    checkpoints_dir = artifacts / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (artifacts / "training-summary.json").write_text(
        json.dumps({"checkpoints": [{"checkpoint_episode": 5}]}),
        encoding="utf-8",
    )
    (checkpoints_dir / "checkpoint-000005.json").write_text(
        json.dumps(
            {
                "checkpoint_episode": 5,
                "total_training_episodes": 5,
                "policy_name": "instinct_only",
                "trainer_type": "baseline",
                "policy_type": "instinct",
                "environment_config": asdict(
                    EnvironmentConfig(max_steps=40, dogs=1, sheep=1, width=24, height=20)
                ),
                "reward_config": asdict(RewardConfig()),
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

    monkeypatch.setattr("sheepdog.server.LabConfig", TestConfig)

    manager = TrainingManager()
    scenario = manager.save_scenario(
        {
            "name": "api scenario",
            "seed": 9,
            "snapshot": {
                "grid_width": 24,
                "grid_height": 20,
                "dogs": [{"index": 0, "x": 1, "y": 2}],
                "sheep": [{"index": 0, "x": 10, "y": 10, "personality": "obedient"}],
                "pen": {"origin": {"x": 18, "y": 1}, "width": 5, "height": 5, "opening": "left"},
            },
            "description": "from api",
        }
    )
    scenario_id = scenario["id"]
    bundle = manager.list_scenarios()
    assert len(bundle["scenarios"]) == 1

    result = manager.evaluate_scenario_by_id(
        scenario_id,
        {"checkpoint_mode": "latest"},
    )
    assert result["checkpoint_episode"] == 5
    assert result["result"]["scenario_id"] == scenario_id
    assert result["result"]["replay_path"]

    replay = manager.replay_scenario_by_id(
        scenario_id,
        {"checkpoint_mode": "specific", "checkpoint_episode": 5},
    )
    assert replay["scenario_id"] == scenario_id
    assert replay["frames"]
