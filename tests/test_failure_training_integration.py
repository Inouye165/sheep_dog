"""Worker exposure and reset provenance checks without policy training."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sheepdog.config import LabConfig
from sheepdog.training.maskable_ppo import collect_worker_scenario_usage
from sheepdog.training.rl_env import SheepdogRLAdapter
from sheepdog.training.scenario_sampler import STANDARD_EVALUATION_SEEDS


def config(tmp_path, seed=500):
    base = LabConfig()
    return replace(base, environment=replace(base.environment, width=144, height=104,
        pen_width=14, pen_height=14, pen_placement="random"), training=replace(
        base.training, train_seed=seed, output_dir=str(tmp_path / "artifacts"),
        backup_dir=str(tmp_path / "backups"), web_export_dir=str(tmp_path / "web"),
        backup_enabled=False, failure_directed_training_enabled=True,
        scenario_training_enabled=False, failure_directed_target_ratio=0.6,
    ))


def test_actual_workers_sample_and_counts_are_cumulative(tmp_path):
    vec = DummyVecEnv([
        lambda: SheepdogRLAdapter(config(tmp_path / "a", 500)),
        lambda: SheepdogRLAdapter(config(tmp_path / "b", 700)),
    ])
    try:
        weights = {"wall_stray_extraction": 1.0, "high_pen_standoff": 1.0}
        vec.env_method("update_failure_weights", weights)
        for _ in range(60):
            vec.reset()
        first = collect_worker_scenario_usage(vec)
        assert first["status"] == "available"
        assert first["total_episodes"] == 120
        assert 40 < first["targeted_episodes"] < 100
        assert first["normal_episodes"] > 0
        assert all(w["active_failure_weights"] == weights for w in first["workers"])
        assert collect_worker_scenario_usage(vec) == first  # reads never reset counts
        vec.reset()
        second = collect_worker_scenario_usage(vec)
        assert second["total_episodes"] == 122
        assert sum(second["scenario_counts"].values()) == 122
        for env in vec.envs:
            assert len(env._environment.dogs) == 3
            assert len(env._environment.sheep) == 6
    finally:
        vec.close()


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_reset_seed_matches_actual_start_and_excludes_evaluation(tmp_path, ratio):
    cfg = config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, failure_directed_target_ratio=ratio))
    a, b = SheepdogRLAdapter(cfg), SheepdogRLAdapter(cfg)
    try:
        for env in (a, b):
            env.update_failure_weights({"wall_stray_extraction": 1.0})
        for raw_seed in (11, 53, 73, 901, 902):
            obs_a, info_a = a.reset(seed=raw_seed)
            obs_b, info_b = b.reset(seed=raw_seed)
            actual = a._environment._seed
            assert actual == info_a["latest_seed"] == info_b["latest_seed"]
            assert actual not in STANDARD_EVALUATION_SEEDS
            assert actual == raw_seed + 100_000 if raw_seed in STANDARD_EVALUATION_SEEDS else actual == raw_seed
            np.testing.assert_array_equal(obs_a, obs_b)
            assert a._environment.get_state_snapshot() == b._environment.get_state_snapshot()
            assert bool(a._latest_scenario_id) == bool(ratio)
    finally:
        a.close()
        b.close()


def test_fixed_seed_sequence_is_used_by_scenario_builder(tmp_path):
    cfg = config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, failure_directed_target_ratio=1.0))
    env = SheepdogRLAdapter(cfg, fixed_seed_sequence=(800, 801))
    try:
        env.update_failure_weights({"wall_stray_extraction": 1.0})
        assert env.reset()[1]["latest_seed"] == 800
        assert env._environment._seed == 800
        assert env.reset()[1]["latest_seed"] == 801
        assert env._environment._seed == 801
    finally:
        env.close()


def test_worker_query_failure_is_unavailable(caplog):
    def broken(_):
        raise RuntimeError("worker disconnected")
    result = collect_worker_scenario_usage(SimpleNamespace(env_method=broken, num_envs=2))
    assert result["status"] == "unavailable"
    assert result["total_episodes"] is None
    assert "worker disconnected" in result["error"]
    assert "worker disconnected" in caplog.text


def test_partial_worker_response_is_unavailable():
    vec = SimpleNamespace(env_method=lambda _: [], num_envs=2)
    assert collect_worker_scenario_usage(vec)["status"] == "unavailable"


@pytest.mark.parametrize("opening", ["top", "bottom", "left", "right"])
def test_wall_stray_initial_layout_is_legal_for_all_gate_orientations(tmp_path, opening):
    from sheepdog.entities import Pen, Point
    from sheepdog.training.training_scenarios import create_wall_stray_extraction_scenario
    cfg = config(tmp_path)
    env_cfg = replace(cfg.environment, pen_opening=opening)
    for seed in range(800, 820):
        scenario = create_wall_stray_extraction_scenario(seed=seed, config=env_cfg)
        pen = Pen(Point(scenario.pen.origin_x, scenario.pen.origin_y),
                  scenario.pen.width, scenario.pen.height, opening)
        points = [Point(a.x, a.y) for a in (*scenario.sheep, *scenario.dogs)]
        assert len(points) == len(set(points)) == 9
        assert not set(points) & pen.fence_cells()
        assert all(0 <= p.x < 144 and 0 <= p.y < 104 for p in points)
        assert scenario.seed == seed
