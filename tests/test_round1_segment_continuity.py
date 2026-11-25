"""Tests for Change 1: episode continuity across PPO segments and rollout alignment."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from sb3_contrib import MaskablePPO

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.training.factory import create_trainer


def _make_config(
    tmp_path: Path,
    *,
    checkpoint_episodes: tuple[int, ...],
    total_timesteps: int,
    rollout_steps: int,
    batch_size: int = 8,
    ppo_env_workers: int = 1,
    max_steps: int = 40,
    no_progress_window: int = 100,
    dogs: int = 3,
    sheep: int = 6,
) -> LabConfig:
    environment = EnvironmentConfig(
        max_steps=max_steps,
        no_progress_window=no_progress_window,
        dogs=dogs,
        sheep=sheep,
    )
    training = TrainingConfig(
        trainer_type="maskable_ppo",
        policy_type="neural",
        episodes=0,
        checkpoint_episodes=checkpoint_episodes,
        evaluation_seeds=(11, 23),
        quick_evaluation_seed_count=1,
        confidence_candidate_success_rate=1.1,
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


def test_episodes_continue_across_segment_boundary_without_double_counting(
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        checkpoint_episodes=(0, 1),
        total_timesteps=128,
        rollout_steps=64,
        batch_size=16,
        ppo_env_workers=1,
        max_steps=40,
        no_progress_window=100,
        dogs=3,
        sheep=6,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []
    learn_calls: list[dict[str, object]] = []

    orig_learn = MaskablePPO.learn

    def spy_learn(self, *args, **kwargs):
        learn_calls.append(
            {
                "reset_num_timesteps": kwargs.get("reset_num_timesteps"),
                "total_timesteps": kwargs.get("total_timesteps"),
            }
        )
        return orig_learn(self, *args, **kwargs)

    with patch.object(MaskablePPO, "learn", new=spy_learn):
        summary = trainer.train(progress_callback=events.append)

    # Segment 1 resets num_timesteps; Segment 2 preserves it across the boundary.
    assert learn_calls == [
        {"reset_num_timesteps": True, "total_timesteps": 64},
        {"reset_num_timesteps": False, "total_timesteps": 64},
    ]

    # Episodes are not cut off at the 64-step segment boundary (64 // 3 = 21 world steps);
    # the episode continues across the boundary and completes with world steps > 21.
    episode_events = [e for e in events if e.get("phase") == "episode_complete"]
    assert len(episode_events) >= 1
    assert any(int(e["length"]) > 21 for e in episode_events)

    # Timesteps accumulate to 64 and 128 without double-counting.
    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 2
    assert len(summary.checkpoints) == 2
    assert checkpoint_events[0]["current_global_timestep"] == 64
    assert checkpoint_events[1]["current_global_timestep"] == 128
    assert summary.checkpoints[0]["global_timestep"] == 64
    assert summary.checkpoints[1]["global_timestep"] == 128

    state_path = Path(config.training.output_dir) / "training-state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_data["total_timesteps"] == 128


def test_unaligned_requested_segment_steps_aligns_to_rollout_quantum(
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        checkpoint_episodes=(0,),
        total_timesteps=10,
        rollout_steps=16,
        batch_size=4,
        ppo_env_workers=1,
        max_steps=20,
        no_progress_window=40,
        dogs=2,
        sheep=2,
    )
    trainer = create_trainer(config, config.training.output_dir)
    events: list[dict[str, object]] = []

    trainer.train(progress_callback=events.append)

    checkpoint_events = [e for e in events if e.get("phase") == "checkpoint"]
    assert len(checkpoint_events) == 1
    timing = checkpoint_events[0]["segment_timing"]
    assert isinstance(timing, dict)
    assert timing["requested_segment_steps"] == 10
    assert timing["aligned_segment_steps"] == 16
    assert timing["actual_segment_steps"] == 16
