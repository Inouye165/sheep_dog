"""Focused tests for SheepdogEnvironment hot-loop invariance and web asset export."""

from __future__ import annotations

import os
from pathlib import Path

from sheepdog.config import EnvironmentConfig, LabConfig, RewardConfig, TrainingConfig
from sheepdog.environment import ACTION_ORDER, SheepdogEnvironment
from sheepdog.evaluation.evaluator import EvaluationSummary
from sheepdog.policies.base import PolicyMode
from sheepdog.training.factory import create_trainer
from sheepdog.training.rl_env import SheepdogRLAdapter


def test_environment_hotloop_helpers_invariance() -> None:
    cfg = LabConfig(environment=EnvironmentConfig(max_steps=40, dogs=3, sheep=6))
    env = SheepdogEnvironment(cfg)
    env.reset(seed=123)

    actions_seq = ["up", "right", "wait"]
    for step_idx in range(30):
        # Verify project_dog_action with explicit blocked_positions matches default
        for dog_index in range(env.dog_count):
            blocked = env._blocked_positions_for_dog(dog_index)
            for action in ACTION_ORDER:
                assert env.project_dog_action(dog_index, action) == env.project_dog_action(
                    dog_index, action, blocked_positions=blocked
                )

            # Verify action_mask_for_dog across policy modes
            for mode in (None, "neural_policy", "heuristic", "instinct_only"):
                pm: PolicyMode | None = mode  # type: ignore[assignment]
                mask = env.action_mask_for_dog(dog_index, policy_mode=pm)
                assert isinstance(mask, dict)
                assert set(mask.keys()) == set(ACTION_ORDER)
                if mode in (None, "neural_policy"):
                    assert mask["wait"] is True

            # Verify _action_score with explicit blocked_positions matches default
            for action in ("wait", "up", "down", "left", "right"):
                assert env._action_score(dog_index, action) == env._action_score(
                    dog_index, action, blocked_positions=blocked
                )

        step_actions = [
            actions_seq[(step_idx + d) % len(actions_seq)] for d in range(env.dog_count)
        ]
        # _touched_wall_this_step and _waited_without_reason must run cleanly
        _ = env._touched_wall_this_step(step_actions)  # type: ignore[arg-type]
        _ = env._waited_without_reason(step_actions)  # type: ignore[arg-type]

        snap, reward = env.step(step_actions)
        assert isinstance(reward.total, float)
        if snap.terminated:
            break


def test_rl_adapter_deterministic_seeded_rollout() -> None:
    cfg = LabConfig(environment=EnvironmentConfig(max_steps=30, dogs=3, sheep=6))
    a1 = SheepdogRLAdapter(cfg)
    a2 = SheepdogRLAdapter(cfg)
    obs1, _ = a1.reset(seed=77)
    obs2, _ = a2.reset(seed=77)
    assert (obs1 == obs2).all()

    for i in range(60):
        m1 = a1.action_masks()
        m2 = a2.action_masks()
        assert (m1 == m2).all()
        valid = [idx for idx, ok in enumerate(m1) if ok]
        act = valid[i % len(valid)]
        o1, r1, t1, tr1, _ = a1.step(act)
        o2, r2, t2, tr2, _ = a2.step(act)
        assert (o1 == o2).all()
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
        if t1 or tr1:
            break


def test_export_web_assets_skips_matching_size_and_updates_diff_size(tmp_path: Path) -> None:
    config = LabConfig(
        environment=EnvironmentConfig(max_steps=10, dogs=2, sheep=2),
        rewards=RewardConfig(),
        training=TrainingConfig(
            trainer_type="hill_climb",
            policy_type="linear",
            episodes=1,
            checkpoint_episodes=(0,),
            evaluation_seeds=(11,),
            output_dir=str(tmp_path / "artifacts"),
            web_export_dir=str(tmp_path / "web" / "generated"),
        ),
    )
    trainer = create_trainer(config, config.training.output_dir)
    web_dir = tmp_path / "web" / "generated"
    replays_dir = tmp_path / "artifacts" / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    src_replay = replays_dir / "ep_0001_seed_11.json"
    src_replay.write_text('{"steps": [1, 2, 3]}', encoding="utf-8")

    cp_path = tmp_path / "artifacts" / "checkpoints" / "checkpoint-000000.json"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text("{}", encoding="utf-8")

    summary = EvaluationSummary(
        checkpoint_episode=0,
        policy_name="heuristic",
        records=(),
        success_rate=0.0,
        timeout_rate=1.0,
        average_completion_steps=10.0,
        average_completion_seconds=5.0,
        average_sheep_penned=0.0,
        average_reward=-1.0,
    )
    cp_payload = {
        "checkpoint_episode": 0,
        "records": [{"seed": 11, "replay_path": str(src_replay)}],
    }

    # First export copies the replay
    trainer._export_web_assets(web_dir, [cp_payload], summary, src_replay, cp_path)
    target_replay = web_dir / "replays" / src_replay.name
    assert target_replay.exists()
    assert target_replay.read_text(encoding="utf-8") == '{"steps": [1, 2, 3]}'

    # Set a sentinel mtime in the past on target_replay; second export with same size must skip copy
    sentinel_ns = 1_600_000_000_000_000_000
    os.utime(target_replay, ns=(sentinel_ns, sentinel_ns))
    trainer._export_web_assets(web_dir, [cp_payload], summary, src_replay, cp_path)
    assert target_replay.stat().st_mtime_ns == sentinel_ns

    # Now change source size; export must overwrite target_replay
    src_replay.write_text('{"steps": [1, 2, 3, 4, 5]}', encoding="utf-8")
    trainer._export_web_assets(web_dir, [cp_payload], summary, src_replay, cp_path)
    assert target_replay.read_text(encoding="utf-8") == '{"steps": [1, 2, 3, 4, 5]}'
    assert target_replay.stat().st_mtime_ns != sentinel_ns
