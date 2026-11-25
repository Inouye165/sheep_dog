"""Tests for descriptive fidelity metrics and bounded audit isolation."""
from dataclasses import replace

import pytest
from scripts.audit_seed59_recovery import (
    FAMILIES,
    coverage_match,
    geometry,
    run,
    sample_scenarios,
)

from sheepdog.config import LabConfig
from sheepdog.training.scenario_sampler import STANDARD_EVALUATION_SEEDS


def snapshot():
    return {
        "grid_width": 144, "grid_height": 104,
        "pen": {"origin": {"x": 128, "y": 3}, "width": 14, "height": 14, "opening": "left"},
        "dogs": [{"x": 0, "y": 2}, {"x": 10, "y": 31}, {"x": 2, "y": 22}],
        "sheep": [{"index": i, "x": x, "y": y, "penned": False}
                  for i, (x, y) in enumerate(((49, 46), (48, 48), (47, 47), (50, 47), (46, 48), (10, 2)))],
    }


def test_identifies_wall_stray_and_distant_compact_main_flock():
    g = geometry(snapshot())
    assert g["stray_index"] == 5
    assert g["stray_wall_margin"] == 2
    assert g["nearest_dog_stray_distance"] == 10
    assert g["main_radius"] < 3
    assert g["stray_main_separation"] > 50
    assert g["main_pen_center_distance"] > 80
    assert coverage_match(g, 144)
    assert coverage_match(g, 144, relaxed=True)


def test_near_pen_flock_does_not_match_remote_main_flock_failure():
    g = geometry(snapshot())
    g["main_pen_center_distance"] = 12
    assert not coverage_match(g, 144)
    assert not coverage_match(g, 144, relaxed=True)


def test_large_split_clusters_do_not_qualify_as_compact_five_plus_one():
    g = geometry(snapshot())
    g["main_radius"] = 20
    assert not coverage_match(g, 144)
    assert not coverage_match(g, 144, relaxed=True)


def test_penned_or_different_size_is_explicitly_not_same_coverage_case():
    s = snapshot()
    s["sheep"][0]["penned"] = True
    with pytest.raises(ValueError, match="six unpenned"):
        geometry(s)


def test_scenario_sampling_is_bounded_seeded_and_evaluation_disjoint():
    base = LabConfig()
    cfg = replace(base, environment=replace(base.environment, width=144, height=104,
                  pen_width=14, pen_height=14, pen_placement="random"))
    first, second = sample_scenarios(cfg), sample_scenarios(cfg)
    assert first == second
    assert set(first) == set(FAMILIES)
    assert sum(v["count"] for v in first.values()) == 300
    for family in first.values():
        assert len(family["layouts"]) == 60
        assert {r["opening"] for r in family["layouts"]} == {"top", "bottom", "left", "right"}
        assert all(r["seed"] not in STANDARD_EVALUATION_SEEDS for r in family["layouts"])


def test_audit_cannot_write_into_active_artifacts(tmp_path):
    with pytest.raises(ValueError, match="new artifacts/experiments"):
        run(tmp_path)
    assert list(tmp_path.iterdir()) == []
