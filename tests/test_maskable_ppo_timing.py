"""Focused unit tests for per-segment wall-clock timing in MaskablePPOTrainer."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.training.factory import create_trainer

EXPECTED_TIMING_KEYS = {
    "segment_steps",
    "learn_seconds",
    "transitions_per_sec",
    "quick_eval_seconds",
    "confidence_eval_seconds",
    "save_and_export_seconds",
    "segment_total_seconds",
    "eval_overhead_pct",
    "ppo_env_workers",
}


def _make_timing_config(
    tmp_path: Path,
    *,
    checkpoint_episodes: tuple[int, ...] = (0, 1),
    total_timesteps: int = 16,
    rollout_steps: int = 8,
    batch_size: int = 4,
    ppo_env_workers: int = 1,
    confidence_candidate_success_rate: float = 1.1,
) -> LabConfig:
    environment = EnvironmentConfig(max_steps=20, dogs=2, sheep=2)
    training = TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        episodes=0,
        checkpoint_episodes=checkpoint_episodes,
        evaluation_seeds=(11, 23),
        quick_evaluation_seed_count=1,
        confidence_candidate_success_rate=confidence_candidate_success_rate,
        rollout_steps=rollout_steps,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        ppo_env_workers=ppo_env_workers,
        backup_enabled=False,
        wandb_enabled=False,
        output_dir=str(tmp_path / "artifacts"),
        web_export_dir=str(tmp_path / "web" / "generated"),
    )
    return LabConfig(environment=environment, rewards=RewardConfig(), training=training)


def _assert_valid_timing_dict(timing: dict[str, object], *, expected_workers: int = 1) -> None:
    assert EXPECTED_TIMING_KEYS.issubset(timing.keys())
    assert isinstance(timing["segment_steps"], int) and timing["segment_steps"] > 0
    assert isinstance(timing["ppo_env_workers"], int) and timing["ppo_env_workers"] == expected_workers
    for key in (
        "learn_seconds",
        "quick_eval_seconds",
        "confidence_eval_seconds",
        "save_and_export_seconds",
        "segment_total_seconds",
        "eval_overhead_pct",
    ):
        value = timing[key]
        assert isinstance(value, (int, float))
        assert value >= 0.0

    learn_s = float(timing["learn_seconds"])
    tps = timing["transitions_per_sec"]
    if learn_s > 0:
        assert isinstance(tps, float)
        assert tps == pytest.approx(float(timing["segment_steps"]) / learn_s)
    else:
        assert tps is None


def test_maskable_ppo_segment_timing_persisted_and_emitted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_timing_config(
        tmp_path,
        checkpoint_episodes=(0, 1),
        total_timesteps=16,
        confidence_candidate_success_rate=1.1,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    trainer.train(progress_callback=events.append)

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 2

    # First segment runs quick + confidence (since quick_seeds is a strict subset of evaluation_seeds)
    first_event_timing = checkpoint_events[0]["segment_timing"]
    assert isinstance(first_event_timing, dict)
    _assert_valid_timing_dict(first_event_timing, expected_workers=1)
    assert first_event_timing["quick_eval_seconds"] >= 0.0
    assert first_event_timing["confidence_eval_seconds"] >= 0.0

    # Final segment always runs confidence evaluation
    second_event_timing = checkpoint_events[1]["segment_timing"]
    assert isinstance(second_event_timing, dict)
    _assert_valid_timing_dict(second_event_timing, expected_workers=1)
    assert second_event_timing["confidence_eval_seconds"] >= 0.0

    # Persisted in training-summary.json
    summary_path = Path(config.training.output_dir) / "training-summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_checkpoints = summary_data["checkpoints"]
    assert len(summary_checkpoints) == 2
    _assert_valid_timing_dict(summary_checkpoints[0]["segment_timing"], expected_workers=1)
    _assert_valid_timing_dict(summary_checkpoints[1]["segment_timing"], expected_workers=1)

    # Persisted in training-state.json
    state_path = Path(config.training.output_dir) / "training-state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert "last_segment_timing" in state_data
    _assert_valid_timing_dict(state_data["last_segment_timing"], expected_workers=1)

    # Single-line stdout summary printed
    captured = capsys.readouterr()
    timing_lines = [line for line in captured.out.splitlines() if line.startswith("[Sheepdog] Timing |")]
    assert len(timing_lines) == 2
    assert "eval_overhead=" in timing_lines[0]


def test_segment_timing_transitions_per_sec_and_overhead_math(tmp_path: Path) -> None:
    config = _make_timing_config(
        tmp_path,
        checkpoint_episodes=(0,),
        total_timesteps=16,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    # Provide deterministic perf_counter ticks across the single segment:
    # segment_start=10.0, learn_start=10.5, learn_end=12.5 (learn=2.0s),
    # model_save_start=13.0, model_save_end=13.25 (0.25s),
    # quick_eval_start=13.5, quick_eval_end=14.0 (0.5s),
    # conf_eval_start=14.0, conf_eval_end=14.5 (0.5s),
    # post_eval_save_start=15.0, pre_json_save=15.25, post_export=16.0
    ticks = iter([10.0, 10.5, 12.5, 13.0, 13.25, 13.5, 14.0, 14.0, 14.5, 15.0, 15.25, 15.25, 16.0, 16.0])
    last_tick = 16.0

    def fake_perf_counter() -> float:
        nonlocal last_tick
        with contextlib.suppress(StopIteration):
            last_tick = next(ticks)
        return last_tick

    with patch("sheepdog.training.maskable_ppo.time.perf_counter", side_effect=fake_perf_counter):
        trainer.train(progress_callback=events.append)

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 1
    timing = checkpoint_events[0]["segment_timing"]
    assert isinstance(timing, dict)
    assert timing["segment_steps"] == 16
    assert timing["learn_seconds"] == pytest.approx(2.0)
    assert timing["transitions_per_sec"] == pytest.approx(8.0)
    assert timing["quick_eval_seconds"] == pytest.approx(0.5)
    assert timing["confidence_eval_seconds"] == pytest.approx(0.5)
    assert timing["save_and_export_seconds"] == pytest.approx(1.25)
    assert timing["segment_total_seconds"] == pytest.approx(6.0)
    assert timing["eval_overhead_pct"] == pytest.approx(100.0 * 1.0 / 6.0)


def test_segment_timing_accounts_for_full_sb3_rollout_exceeding_requested_steps(
    tmp_path: Path,
) -> None:
    config = _make_timing_config(
        tmp_path,
        checkpoint_episodes=(0,),
        total_timesteps=10,
        rollout_steps=16,
        batch_size=4,
        ppo_env_workers=1,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    summary = trainer.train(progress_callback=events.append)

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 1
    timing = checkpoint_events[0]["segment_timing"]
    assert isinstance(timing, dict)
    assert timing["requested_segment_steps"] == 10
    assert timing["actual_segment_steps"] == 16
    assert timing["segment_steps"] == 16
    learn_seconds = float(timing["learn_seconds"])
    assert learn_seconds > 0
    assert timing["transitions_per_sec"] == pytest.approx(16.0 / learn_seconds)

    assert len(summary.checkpoints) == 1
    assert summary.checkpoints[0]["global_timestep"] == 16
    assert checkpoint_events[0]["total_timesteps"] == 16
    assert checkpoint_events[0]["current_global_timestep"] == 16

    state_path = Path(config.training.output_dir) / "training-state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_data["total_timesteps"] == 16
    assert state_data["last_segment_timing"]["requested_segment_steps"] == 10
    assert state_data["last_segment_timing"]["actual_segment_steps"] == 16
    assert state_data["last_segment_timing"]["segment_steps"] == 16


def test_multi_segment_total_timesteps_accumulates_actual_sb3_transitions(
    tmp_path: Path,
) -> None:
    config = _make_timing_config(
        tmp_path,
        checkpoint_episodes=(0, 1),
        total_timesteps=20,
        rollout_steps=16,
        batch_size=4,
        ppo_env_workers=1,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    summary = trainer.train(progress_callback=events.append)

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 2
    assert len(summary.checkpoints) == 2

    # Each segment requests 10 steps, but SB3 collects a full 16-step rollout per segment.
    for idx, event in enumerate(checkpoint_events, start=1):
        timing = event["segment_timing"]
        assert isinstance(timing, dict)
        assert timing["requested_segment_steps"] == 10
        assert timing["actual_segment_steps"] == 16
        assert timing["segment_steps"] == 16
        assert event["total_timesteps"] == 16 * idx
        assert event["current_global_timestep"] == 16 * idx
        assert summary.checkpoints[idx - 1]["global_timestep"] == 16 * idx

    state_path = Path(config.training.output_dir) / "training-state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_data["total_timesteps"] == 32

    complete_events = [e for e in events if e.get("phase") == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0]["current_global_timestep"] == 32


def test_incomplete_batch_resume_preserves_schedule_with_full_sb3_rollout_accounting(
    tmp_path: Path,
) -> None:
    config = _make_timing_config(
        tmp_path,
        checkpoint_episodes=(0, 1, 2),
        total_timesteps=30,
        rollout_steps=16,
        batch_size=4,
        ppo_env_workers=1,
    )

    # Run 1: stop immediately after the first checkpoint is emitted.
    stop_flag = False
    run1_events: list[dict[str, object]] = []

    def on_run1_event(event: dict[str, object]) -> None:
        nonlocal stop_flag
        run1_events.append(event)
        if event.get("phase") == "checkpoint":
            stop_flag = True

    trainer1 = create_trainer(config, config.training.output_dir)
    summary1 = trainer1.train(
        progress_callback=on_run1_event,
        should_stop=lambda: stop_flag,
    )

    run1_checkpoints = [e for e in run1_events if e.get("phase") == "checkpoint"]
    assert len(run1_checkpoints) == 1
    assert run1_checkpoints[0]["checkpoint_episode"] == 0
    assert run1_checkpoints[0]["total_timesteps"] == 16
    assert len(summary1.checkpoints) == 1
    assert summary1.checkpoints[0]["checkpoint_episode"] == 0
    assert summary1.checkpoints[0]["global_timestep"] == 16

    state_path = Path(config.training.output_dir) / "training-state.json"
    state_after_run1 = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_after_run1["total_timesteps"] == 16
    assert state_after_run1["incomplete_batch"] == {
        "batch_completed_segments": 1,
        "batch_total_segments": 3,
        "batch_steps_per_segment": 10,
        "batch_timestep_targets": [10, 20, 30],
        "batch_starting_checkpoint_episode": 0,
    }

    # Run 2 (Resume): same config & output_dir, run to completion without should_stop.
    run2_events: list[dict[str, object]] = []
    trainer2 = create_trainer(config, config.training.output_dir)
    summary2 = trainer2.train(progress_callback=run2_events.append)

    run2_checkpoints = [e for e in run2_events if e.get("phase") == "checkpoint"]
    assert len(run2_checkpoints) == 2
    assert [e["checkpoint_episode"] for e in run2_checkpoints] == [1, 2]
    assert [e["total_timesteps"] for e in run2_checkpoints] == [32, 48]
    assert [e["current_global_timestep"] for e in run2_checkpoints] == [32, 48]

    assert [cp["checkpoint_episode"] for cp in summary2.checkpoints] == [0, 1, 2]
    assert [cp["global_timestep"] for cp in summary2.checkpoints] == [16, 32, 48]

    state_after_run2 = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_after_run2["total_timesteps"] == 48
    assert state_after_run2["total_episodes_trained"] == 2
    assert state_after_run2["incomplete_batch"] is None
