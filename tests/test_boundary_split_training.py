"""Boundary-split initialization and geometry-directed targeting regressions."""
from dataclasses import replace
from math import hypot, nan

import pytest

from sheepdog.config import EnvironmentConfig, LabConfig, TrainingConfig
from sheepdog.entities import Pen, Point
from sheepdog.environment import SheepdogEnvironment
from sheepdog.training.scenario_sampler import ScenarioSampler
from sheepdog.training.training_scenarios import (
    SCENARIO_BUILDERS,
    boundary_split_geometry,
    create_boundary_split_flock_scenario,
    get_scenario_builder,
)


def config():
    return EnvironmentConfig(width=144, height=104, dogs=3, sheep=6,
                             pen_width=14, pen_height=14, pen_placement="random",
                             sheep_vision=12, flock_radius=10)


def geometry():
    # An anonymized geometric family, rather than a recorded seed/layout.
    return [[61, 49], [62, 50], [60, 50], [63, 48], [61, 51], [29, 2]]


@pytest.mark.parametrize("wall", ["top", "bottom", "left", "right"])
def test_signature_is_orientation_independent(wall):
    cfg = config()
    points = geometry()
    if wall == "bottom":
        points = [[x, cfg.height - 1 - y] for x, y in points]
    elif wall in ("left", "right"):
        points = [[y, x] for x, y in points]
        if wall == "right":
            points = [[cfg.width - 1 - x, y] for x, y in points]
    assert boundary_split_geometry(points, cfg)


def test_actual_geometry_precedes_historical_seed_but_success_never_targets():
    sampler = ScenarioSampler(TrainingConfig(), config())
    result = sampler.update_from_evaluation([
        {"seed": 73, "success": False, "sheep_penned": 0,
         "final_sheep_positions": geometry()},
        {"seed": 700_777, "success": True, "sheep_penned": 0,
         "final_sheep_positions": geometry()},
    ])
    assert result == {"boundary_split_flock": 1.0}
    assert get_scenario_builder("boundary_split_flock") is create_boundary_split_flock_scenario
    assert sampler.update_from_evaluation([
        {"seed": 700_888, "success": False, "sheep_penned": 0,
         "final_sheep_positions": geometry()}
    ]) == {"boundary_split_flock": 1.0}


@pytest.mark.parametrize("bad", [None, [], [[nan, 2]] * 6, [[1, 2]] * 6,
                                  [[1, 2, 3]] * 6, [[-1, 2]] * 6,
                                  [[True, 2]] * 6, "not positions"])
def test_malformed_geometry_is_unclassified_and_falls_back(bad):
    cfg = config()
    assert not boundary_split_geometry(bad, cfg)
    sampler = ScenarioSampler(TrainingConfig(), cfg)
    assert sampler.update_from_evaluation([
        {"seed": 73, "success": False, "sheep_penned": 0,
         "final_sheep_positions": bad}
    ]) == {"high_pen_standoff": 1.0}


def test_compact_flock_nearby_stray_and_partial_completion_do_not_match():
    cfg = config()
    compact = geometry()
    compact[-1] = [62, 52]
    assert not boundary_split_geometry(compact, cfg)
    nearby = geometry()
    nearby[-1] = [61, 33]
    assert not boundary_split_geometry(nearby, cfg)
    sampler = ScenarioSampler(TrainingConfig(), cfg)
    assert sampler.update_from_evaluation([
        {"seed": 59, "success": False, "sheep_penned": 1,
         "final_sheep_positions": geometry()}
    ]) == {"wall_stray_extraction": 1.0}


def test_eighty_procedural_layouts_are_legal_diverse_and_match_family():
    cfg = config()
    walls, openings, distinct = set(), set(), set()
    representatives = {}
    for seed in range(700_100, 700_180):
        scenario = create_boundary_split_flock_scenario(seed=seed, config=cfg)
        assert scenario == create_boundary_split_flock_scenario(seed=seed, config=cfg)
        assert (len(scenario.dogs), len(scenario.sheep)) == (3, 6)
        assert scenario.seed == seed
        pen = Pen(Point(scenario.pen.origin_x, scenario.pen.origin_y),
                  scenario.pen.width, scenario.pen.height, scenario.pen.opening)
        sheep = [Point(s.x, s.y) for s in scenario.sheep]
        dogs = [Point(d.x, d.y) for d in scenario.dogs]
        assert boundary_split_geometry([(p.x, p.y) for p in sheep], cfg)
        assert len(set(sheep + dogs)) == 9
        assert all(0 <= p.x < cfg.width and 0 <= p.y < cfg.height
                   and p not in pen.fence_cells() and not pen.contains(p) for p in sheep + dogs)
        center = (sum(p.x for p in sheep[:-1]) / 5, sum(p.y for p in sheep[:-1]) / 5)
        assert hypot(center[0] - pen.center.x, center[1] - pen.center.y) >= cfg.width * .25
        assert all(min(d.distance_to(s) for s in sheep[:-1]) > cfg.sheep_vision + 2 for d in dogs)
        stray = sheep[-1]
        wall = min({"top": stray.y, "bottom": cfg.height - 1 - stray.y,
                    "left": stray.x, "right": cfg.width - 1 - stray.x},
                   key=lambda side: {"top": stray.y, "bottom": cfg.height - 1 - stray.y,
                                     "left": stray.x, "right": cfg.width - 1 - stray.x}[side])
        walls.add(wall)
        openings.add(pen.opening)
        distinct.add(tuple(sheep + dogs))
        representatives[wall] = scenario
    assert walls == openings == {"top", "bottom", "left", "right"}
    assert len(distinct) == 80
    env = SheepdogEnvironment(replace(LabConfig(), environment=cfg))
    for scenario in representatives.values():
        state = env.reset_from_scenario(scenario)
        assert state.penned_count == 0
        assert not state.success


def test_generic_counts_and_impossible_configuration():
    cfg = replace(config(), dogs=2, sheep=4)
    scenario = create_boundary_split_flock_scenario(seed=700_001, config=cfg)
    assert len(scenario.dogs) == 2 and len(scenario.sheep) == 4
    with pytest.raises(ValueError, match="sufficiently large field"):
        create_boundary_split_flock_scenario(seed=700_001, config=replace(cfg, width=20, height=20))


def test_normal_random_scenario_stage1_never_places_on_fence_pen_overlap_or_oob():
    from sheepdog.training.training_scenarios import create_normal_random_scenario

    stage1_cfg = EnvironmentConfig(
        width=60,
        height=45,
        dogs=1,
        sheep=1,
        pen_width=12,
        pen_height=12,
        pen_opening="bottom",
    )
    env = SheepdogEnvironment(replace(LabConfig(), environment=stage1_cfg))
    for seed in range(100):
        scenario = create_normal_random_scenario(seed=seed, config=stage1_cfg)
        pen = Pen(
            Point(scenario.pen.origin_x, scenario.pen.origin_y),
            scenario.pen.width,
            scenario.pen.height,
            scenario.pen.opening,
        )
        fence = pen.fence_cells()
        points = [Point(a.x, a.y) for a in (*scenario.sheep, *scenario.dogs)]
        assert len(points) == len(set(points)) == 2
        assert all(
            0 <= p.x < stage1_cfg.width
            and 0 <= p.y < stage1_cfg.height
            and p not in fence
            and not pen.contains(p)
            for p in points
        )
        env.reset_from_scenario(scenario)


def test_scenario_sampler_skips_boundary_split_on_stage1():
    stage1_cfg = EnvironmentConfig(
        width=60,
        height=45,
        dogs=1,
        sheep=1,
        pen_width=12,
        pen_height=12,
        pen_opening="bottom",
    )
    train_cfg = TrainingConfig(
        failure_directed_training_enabled=True,
        failure_directed_target_ratio=1.0,
        scenario_training_enabled=False,
    )
    sampler = ScenarioSampler(train_cfg, stage1_cfg)
    sampler.set_failure_weights({"boundary_split_flock": 1.0})
    for episode_index in range(20):
        selection = sampler.sample(episode_index)
        assert selection.scenario_type == "normal"
        assert selection.scenario is None


def test_all_applicable_scenario_builders_stage1_legal_and_clean_reset():
    stage1_cfg = EnvironmentConfig(
        width=60,
        height=45,
        dogs=1,
        sheep=1,
        pen_width=12,
        pen_height=12,
        pen_opening="bottom",
    )
    env = SheepdogEnvironment(replace(LabConfig(), environment=stage1_cfg))
    seeds = (*range(200), 352)
    for name, builder in SCENARIO_BUILDERS.items():
        if name == "boundary_split_flock":
            continue
        for seed in seeds:
            scenario = builder(seed=seed, config=stage1_cfg)
            pen = Pen(
                Point(scenario.pen.origin_x, scenario.pen.origin_y),
                scenario.pen.width,
                scenario.pen.height,
                scenario.pen.opening,
            )
            fence = pen.fence_cells()
            points = [Point(a.x, a.y) for a in (*scenario.sheep, *scenario.dogs)]
            assert len(points) == len(set(points)), f"{name} seed={seed} overlap"
            assert all(
                0 <= p.x < stage1_cfg.width
                and 0 <= p.y < stage1_cfg.height
                and p not in fence
                and not pen.contains(p)
                for p in points
            ), f"{name} seed={seed} illegal placement: {points}"
            state = env.reset_from_scenario(scenario)
            assert state.penned_count == 0, f"{name} seed={seed} penned at reset"
            assert not state.success, f"{name} seed={seed} succeeded at reset"
