"""Focused tests verifying telemetry enrichment, database persistence, and replay logging."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.diagnostics.temporal_interaction import (
    analyze_episode_temporal_interactions,
    load_replay_bundle,
)
from sheepdog.training.episode_store import EpisodeStore, get_episode_store
from sheepdog.training.maskable_ppo import _TrainingProgressCallback
from sheepdog.training.telemetry import CurriculumTelemetryManager


def test_maskable_ppo_emit_includes_spatial_metrics_and_setup() -> None:
    """_emit for episode_complete extracts spatial metrics, pen_zone, spawn_mode, and corner_stuck_at_end."""
    emitted_payloads: list[dict[str, Any]] = []

    def mock_emit(payload: dict[str, Any]) -> None:
        emitted_payloads.append(payload)

    callback = _TrainingProgressCallback(
        emit=mock_emit,
        should_stop=None,
        report_interval=10,
        total_timesteps=1000,
        starting_total_episodes=0,
        batch_total_episodes=10,
        batch_total_timesteps=1000,
        completed_timesteps=0,
        completed_segments=0,
        segment_index=0,
        policy_version=1,
        starting_total=0,
    )

    test_spatial_metrics = {
        "corner_count": 3,
        "flock_spread": 4.12,
        "wall_distance": 2.5,
    }
    ep_info = {
        "r": 25.5,
        "l": 40,
        "success": True,
        "penned": 6,
        "total_sheep": 6,
        "status": "SUCCESS",
        "seed": 1042,
        "field_setup": "3d/6s | Pen: right",
        "replay_available": 1,
        "replay_id": "test_replay_001",
        "replay_path": "/tmp/replay.json.gz",
        "replay_source": "training-diagnostic",
        "capture_reason": "success",
        "capture_status": "queued",
        "reward_breakdown": {"penning": 20.0},
        "pen_zone": "right_center",
        "spawn_mode": "corner_split",
        "initial_sheep_zone": "top_left",
        "final_sheep_zone": "right_center",
        "corner_time_pct": 0.125,
        "wall_time_pct": 0.250,
        "corner_stuck_at_end": False,
        "spatial_metrics": test_spatial_metrics,
    }

    mock_model = type("MockModel", (), {"num_timesteps": 40, "get_env": lambda self: None})()
    callback.model = mock_model
    callback.num_timesteps = 40
    callback.locals = {"dones": [True], "infos": [{"episode": ep_info}]}
    callback._on_step()

    ep_payloads = [p for p in emitted_payloads if p.get("phase") == "episode_complete"]
    assert len(ep_payloads) == 1
    p = ep_payloads[0]

    assert p["pen_zone"] == "right_center"
    assert p["spawn_mode"] == "corner_split"
    assert p["initial_sheep_zone"] == "top_left"
    assert p["final_sheep_zone"] == "right_center"
    assert p["corner_time_pct"] == 0.125
    assert p["wall_time_pct"] == 0.250
    assert p["corner_stuck_at_end"] is False
    assert p["spatial_metrics"] == test_spatial_metrics


def test_episode_store_saves_non_null_spatial_and_session_fields(tmp_path: Path) -> None:
    """EpisodeStore persists spawn_mode, pen_zone, spatial_metrics, corner_stuck_at_end, and session_id."""
    db_path = tmp_path / "training-telemetry.sqlite"
    store = EpisodeStore(db_path)

    test_spatial = {
        "corner_time_pct": 0.22,
        "wall_time_pct": 0.44,
        "wall_stalls": 2,
    }
    store.add_episode(
        {
            "episode": 1,
            "stage": 2,
            "reward": 35.5,
            "penned": 6,
            "total_sheep": 6,
            "success": True,
            "status": "SUCCESS",
            "seed": 2024,
            "run_id": "run_enrichment_test",
            "session_id": "session_20261108_abcd12",
            "global_timestep": 120,
            "policy_version": 3,
            "length": 55,
            "checkpoint_id": "chk_001",
            "spawn_mode": "wall_trapped",
            "pen_zone": "top_right",
            "initial_sheep_zone": "bottom_left",
            "final_sheep_zone": "top_right",
            "corner_time_pct": 0.22,
            "wall_time_pct": 0.44,
            "corner_stuck_at_end": True,
            "spatial_metrics": test_spatial,
        }
    )
    store.flush()

    # Verify directly from SQLite table
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT spawn_mode, pen_zone, initial_sheep_zone, final_sheep_zone,
               corner_time_pct, wall_time_pct, corner_stuck_at_end,
               spatial_metrics, session_id, run_id, global_environment_episode
        FROM training_episodes
        WHERE global_environment_episode = 1
        """
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["spawn_mode"] == "wall_trapped"
    assert row["pen_zone"] == "top_right"
    assert row["initial_sheep_zone"] == "bottom_left"
    assert row["final_sheep_zone"] == "top_right"
    assert abs(row["corner_time_pct"] - 0.22) < 1e-4
    assert abs(row["wall_time_pct"] - 0.44) < 1e-4
    assert row["corner_stuck_at_end"] == 1
    assert row["session_id"] == "session_20261108_abcd12"
    assert row["run_id"] == "run_enrichment_test"
    assert row["spatial_metrics"] is not None

    # Verify CurriculumTelemetryManager.log_episode also passes through and flushes
    telemetry_dir = tmp_path / "telemetry_mgr"
    telemetry_mgr = CurriculumTelemetryManager(telemetry_dir)
    telemetry_mgr.log_episode(
        episode=2,
        stage=3,
        reward=50.0,
        penned=6,
        total_sheep=6,
        success=True,
        status="SUCCESS",
        seed=3033,
        run_id="run_mgr_test",
        session_id="session_mgr_xyz",
        spawn_mode="stray_split",
        pen_zone="bottom_center",
        corner_stuck_at_end=False,
        spatial_metrics={"stray_distance": 3.4},
    )
    mgr_store = get_episode_store(telemetry_dir / "training-telemetry.sqlite")
    mgr_store.flush()

    mgr_episodes = mgr_store.get_episodes(run_id="run_mgr_test")["episodes"]
    assert len(mgr_episodes) == 1
    mgr_ep = mgr_episodes[0]
    assert mgr_ep["session_id"] == "session_mgr_xyz"
    assert mgr_ep["spawn_mode"] == "stray_split"
    assert mgr_ep["pen_zone"] == "bottom_center"
    assert mgr_ep["corner_stuck_at_end"] is False
    assert mgr_ep["spatial_metrics"] == {"stray_distance": 3.4}

    store.close()
    mgr_store.close()


def test_evaluator_produces_replay_with_step0_decisions_and_strategy(tmp_path: Path) -> None:
    """evaluator.evaluate with capture_step_telemetry produces step 0 frame, decisions, and strategy."""
    config = LabConfig()
    eval_output = tmp_path / "evaluation"
    evaluator = Evaluator(config, output_root=eval_output)
    policy = NeuralPolicy.initialize(config)

    summary, json_path, _ = evaluator.evaluate(
        policy,
        seeds=(42,),
        checkpoint_episode=10,
        capture_replays=True,
        capture_step_telemetry=True,
        evaluation_mode="confidence",
    )

    assert json_path.exists()
    assert len(summary.records) == 1
    record = summary.records[0]
    assert record.replay_path != ""

    # Load via temporal_interaction bundle loader
    bundle, replay_id, _ = load_replay_bundle(record.replay_path)

    # 1. Top-level initial_snapshot and step 0 frame presence
    assert "initial_snapshot" in bundle
    assert isinstance(bundle["initial_snapshot"], dict)
    assert bundle["initial_snapshot"]["step"] == 0
    assert "frames" in bundle
    assert len(bundle["frames"]) > 1

    first_frame = bundle["frames"][0]
    assert first_frame["step"] == 0
    assert first_frame["snapshot"] == bundle["initial_snapshot"]

    # Verify temporal analysis detects initial step 0 evidence
    analysis = analyze_episode_temporal_interactions(bundle)
    assert analysis.evidence_audit.has_initial_step_zero is True

    # 2. Decision telemetry: action_probabilities, logits, value_estimate, 54-dim observation
    frames_with_telemetry = [f for f in bundle["frames"] if f.get("step_telemetry")]
    assert len(frames_with_telemetry) > 0

    first_active_step = frames_with_telemetry[0]
    telemetry = first_active_step["step_telemetry"]
    assert "decisions" in telemetry
    decisions = telemetry["decisions"]
    assert len(decisions) == config.environment.dogs

    for dec in decisions:
        assert "action_probabilities" in dec
        assert isinstance(dec["action_probabilities"], dict)
        assert len(dec["action_probabilities"]) == 9  # 9 movement & wait actions
        assert abs(sum(dec["action_probabilities"].values()) - 1.0) < 1e-4

        assert "logits" in dec
        assert isinstance(dec["logits"], dict)
        assert len(dec["logits"]) == 9

        assert "value_estimate" in dec
        assert isinstance(dec["value_estimate"], float)

        assert "observation" in dec
        assert isinstance(dec["observation"], list)
        assert len(dec["observation"]) == 54  # 54-dimensional observation vector

    # 3. Strategy telemetry: sticky_pen_dx and sticky_pen_dy
    assert "strategy" in telemetry
    strategy = telemetry["strategy"]
    assert "sticky_pen_dx" in strategy
    assert "sticky_pen_dy" in strategy
    assert strategy["sticky_pen_dx"] in (-1, 0, 1)
    assert strategy["sticky_pen_dy"] in (-1, 0, 1)
