"""Bounded factual audit of the development-seed-59 recovery failure.

This does not train or change actions. Scenario samples describe initial-layout
coverage only; matching these descriptive thresholds is not proof of recovery.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.entities import Pen, Point
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.recovery_hyperparam_experiment import collect_dependency_versions
from sheepdog.training.diagnostics.repetition_recovery import analyze_repetition_and_recovery
from sheepdog.training.diagnostics.temporal_interaction import ROLE_AWARE_OBSERVATION_SCHEMA
from sheepdog.training.scenario_sampler import STANDARD_EVALUATION_SEEDS
from sheepdog.training.training_scenarios import get_scenario_builder

RUN = Path("artifacts/experiments/gate_current_run_20261113_174700")
PARENT_SHA = "26aecd34c7d1723feb4ae2f44a03fbddc16c494b2855a6c26216b31703c54dd2"
FAMILIES = ("wall_stray_extraction", "isolated_stray", "split_flock",
            "corner_huddle", "high_pen_standoff")


def sha(path: Path) -> str:
    """Return the exact file-content identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replay(path: Path) -> dict:
    """Read existing plain or compressed replay data without mutation."""
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def center(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return a Euclidean arithmetic centroid."""
    return tuple(sum(p[i] for p in points) / len(points) for i in (0, 1))


def geometry(snapshot: dict) -> dict:
    """Use Euclidean geometry, explicitly not pen-navigation route distance."""
    sheep = [s for s in snapshot["sheep"] if not s.get("penned", False)]
    dogs = [(d["x"], d["y"]) for d in snapshot["dogs"]]
    if len(sheep) != 6 or len(dogs) != 3:
        raise ValueError("Audit metrics require three dogs and six unpenned sheep")
    points = [(s["x"], s["y"]) for s in sheep]
    isolated = max(range(6), key=lambda i: math.dist(points[i], center(points[:i] + points[i + 1:])))
    stray = points[isolated]
    main = points[:isolated] + points[isolated + 1:]
    main_center = center(main)
    pen = snapshot["pen"]
    origin = pen["origin"]
    pen_center = (origin["x"] + pen["width"] / 2, origin["y"] + pen["height"] / 2)
    width, height = snapshot["grid_width"], snapshot["grid_height"]
    return {
        "stray_index": sheep[isolated]["index"], "stray": list(stray),
        "main_center": list(main_center),
        "main_radius": max(math.dist(p, main_center) for p in main),
        "stray_main_separation": math.dist(stray, main_center),
        "stray_wall_margin": min(stray[0], width - 1 - stray[0], stray[1], height - 1 - stray[1]),
        "main_pen_center_distance": math.dist(main_center, pen_center),
        "nearest_dog_stray_distance": min(math.dist(p, stray) for p in dogs),
        "nearest_dog_main_center_distance": min(math.dist(p, main_center) for p in dogs),
        "dogs": [list(p) for p in dogs], "sheep": [list(p) for p in points],
        "pen_opening": pen["opening"],
    }


def coverage_match(g: dict, width: int, *, relaxed: bool = False) -> bool:
    """Descriptive five-plus-one wall/remote-flock coverage predicate."""
    margin, separation, radius, pen_fraction = (5, 20, 12, .20) if relaxed else (3, 24, 8, .25)
    return (g["stray_wall_margin"] <= margin and g["stray_main_separation"] >= separation
            and g["main_radius"] <= radius and g["main_pen_center_distance"] >= width * pen_fraction)


def scenario_snapshot(scenario) -> tuple[dict, int]:
    """Describe a generated initial layout and count illegal/duplicate cells."""
    p = scenario.pen
    pen = Pen(Point(p.origin_x, p.origin_y), p.width, p.height, p.opening)
    sheep = [{"index": s.index, "x": s.x, "y": s.y, "penned": pen.contains(Point(s.x, s.y))}
             for s in scenario.sheep]
    positions = [Point(a.x, a.y) for a in (*scenario.sheep, *scenario.dogs)]
    illegal = sum(not (0 <= q.x < scenario.width and 0 <= q.y < scenario.height)
                  or q in pen.fence_cells() for q in positions)
    illegal += len(positions) - len(set(positions))
    return {"grid_width": scenario.width, "grid_height": scenario.height,
            "sheep": sheep, "dogs": [d.to_dict() for d in scenario.dogs],
            "pen": {"origin": {"x": p.origin_x, "y": p.origin_y},
                    "width": p.width, "height": p.height, "opening": p.opening}}, illegal


def sample_scenarios(config: LabConfig) -> dict:
    """Exactly 300 procedural layouts; never use development/held-out seeds."""
    reports = {}
    for family in FAMILIES:
        rows = []
        for index in range(60):
            seed = 500_001 + index
            if seed in STANDARD_EVALUATION_SEEDS:
                raise ValueError("Scenario audit seed collides with evaluation")
            env = replace(config.environment, pen_opening=("top", "bottom", "left", "right")[index % 4])
            scenario = get_scenario_builder(family)(seed=seed, config=env)
            snapshot, illegal = scenario_snapshot(scenario)
            penned = sum(s["penned"] for s in snapshot["sheep"])
            g = geometry(snapshot) if penned == 0 else None
            rows.append({"seed": seed, "opening": env.pen_opening, "already_penned": penned,
                         "illegal_or_overlap_positions": illegal, "geometry": g,
                         "strict_match": bool(g and coverage_match(g, env.width)),
                         "relaxed_match": bool(g and coverage_match(g, env.width, relaxed=True))})
        reports[family] = {"layouts": rows, "count": len(rows),
                           "strict_matches": sum(r["strict_match"] for r in rows),
                           "relaxed_matches": sum(r["relaxed_match"] for r in rows),
                           "layouts_with_illegal_or_overlap": sum(r["illegal_or_overlap_positions"] > 0 for r in rows),
                           "layouts_with_already_penned": sum(r["already_penned"] > 0 for r in rows)}
    return reports


def trajectory_summary(bundle: dict, vision: float) -> dict:
    """Measure saved state geometry, rewards and actual displacement."""
    frames = bundle["frames"]
    initial = bundle["initial_snapshot"]
    checkpoints = [{"step": 0, "geometry": geometry(initial)}]
    for step in (1, 24, 50, 100, 200, 500, 1000, 2320):
        f = next(f for f in frames if f["step"] == step)
        checkpoints.append({"step": step, "geometry": geometry(f["snapshot"]),
                            "no_progress_steps": f["snapshot"]["no_progress_steps"]})
    late = frames[-500:]
    rewards = {key: sum(f["reward"][key] for f in frames) for key in frames[0]["reward"]}
    dog_metrics = {}
    snapshots = [initial] + [f["snapshot"] for f in frames]
    for dog in range(3):
        rows = [f["snapshot"]["dogs"][dog] for f in late]
        displacements = [math.dist((snapshots[i]["dogs"][dog]["x"], snapshots[i]["dogs"][dog]["y"]),
                                   (snapshots[i + 1]["dogs"][dog]["x"], snapshots[i + 1]["dogs"][dog]["y"]))
                         for i in range(len(frames) - 500, len(frames))]
        dog_metrics[str(dog)] = {"unique_positions_last500": len({(d["x"], d["y"]) for d in rows}),
                                 "stationary_last500": sum(v == 0 for v in displacements),
                                 "actual_distance_travelled_last500": sum(displacements),
                                 "action_counts_last500": dict(Counter(f["actions"][dog] for f in late))}
    late_geometry = [geometry(f["snapshot"]) for f in late]
    return {"actual_world_steps": frames[-1]["step"], "sampled_states": checkpoints,
            "all_reward_components_sum": rewards,
            "dog_metrics": dog_metrics,
            "last500_main_center_outside_all_dog_vision_steps": sum(
                g["nearest_dog_main_center_distance"] > vision for g in late_geometry),
            "last500_stray_within_at_least_one_dog_vision_steps": sum(
                g["nearest_dog_stray_distance"] <= vision for g in late_geometry),
            "last500_stray_positions": dict(Counter(str(tuple(g["stray"])) for g in late_geometry)),
            "last500_no_progress_zero_steps": sum(f["snapshot"]["no_progress_steps"] == 0 for f in late),
            "strict_geometry_match_steps": sum(coverage_match(geometry(f["snapshot"]), initial["grid_width"]) for f in frames),
            "best_recorded_route_distance": min(s["average_distance_to_pen"] for s in snapshots),
            "initial_route_distance": initial["average_distance_to_pen"]}


def decision_summary(bundle: dict) -> dict:
    """Summarize recorded tail decisions without treating entropy as action randomness."""
    indices = {name: index for index, (name, _, _) in enumerate(ROLE_AWARE_OBSERVATION_SCHEMA)}
    tail = bundle["frames"][-500:]
    reports = {}
    for dog in range(3):
        rows = [f["step_telemetry"]["decisions"][dog] for f in tail]
        if any(len(row["observation"]) != len(ROLE_AWARE_OBSERVATION_SCHEMA) for row in rows):
            raise ValueError("Decision summary requires the verified guided 54-feature schema")
        entropy = [-sum(p * math.log(p) for p in row["action_probabilities"].values() if p > 0)
                   for row in rows]
        targets = [f["step_telemetry"]["pre_step_roles"][dog]["target"] for f in tail]
        reports[str(dog)] = {
            "decisions": len(rows), "action_replacements": sum(row["replacement_reason"] is not None for row in rows),
            "entropy_nats_min": min(entropy), "entropy_nats_max": max(entropy),
            "entropy_nats_mean": sum(entropy) / len(entropy),
            "revisit_feature_nonzero": sum(bool(row["observation"][indices["revisits_recent_position"]]) for row in rows),
            "two_position_loop_feature_nonzero": sum(bool(row["observation"][indices["two_position_loop"]]) for row in rows),
            "recorded_pre_step_target_counts": dict(Counter(str((t["x"], t["y"])) for t in targets)),
            "selected_action_counts": dict(Counter(row["selected_action"] for row in rows)),
            "probability_source_counts": dict(Counter(row["action_probabilities_source"] for row in rows)),
            "decision_modes": dict(Counter(row["decision_mode"] for row in rows)),
        }
    return {"actual_episode_tail_steps": 500, "recorded_dog_decisions": 1500,
            "guidance_source": "recorded_pre_step_roles_not_reconstruction", "dogs": reports,
            "limit": "Actor entropy is recomputed diagnostic probability entropy; execution is deterministic. No unchosen move is established as successful."}


def run(root: Path, capture: bool = False) -> dict:
    """Write an isolated audit, optionally capturing one passive episode."""
    root = root.resolve()
    if root.parent != Path("artifacts/experiments").resolve() or not root.name.startswith("seed59_audit_"):
        raise ValueError("Use a new artifacts/experiments/seed59_audit_<timestamp> directory")
    if root.exists():
        raise FileExistsError("No audit overwrites")
    root.mkdir(parents=True)
    source = json.loads((RUN / "manifest.json").read_text())
    record = next(r for r in source["baseline"]["records"] if r["seed"] == 59)
    replay = Path(record["replay_path"])
    bundle = load_replay(replay)
    parent = RUN / "prepared" / source["parent"]["model"]
    if sha(parent) != PARENT_SHA:
        raise ValueError("Wrong parent identity")
    cfg = LabConfig.from_dict(json.loads((RUN / "baseline_config.json").read_text()))
    config_data = cfg.to_dict()
    config_data["training"].update(output_dir=str(root / "capture_output"),
                                    backup_dir=str(root / "backups"), web_export_dir=str(root / "web"))
    cfg = LabConfig.from_dict(config_data)
    atomic_write_json(root / "config.json", cfg.to_dict())
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    (root / "dirty.patch").write_bytes(patch)
    report = {"status": "offline_audit", "assistance": "guided; existing assistance unchanged",
              "baseline": {"parent_sha256": PARENT_SHA, "seed": 59, "success": False,
                           "penned": 0, "steps": 2320, "source_replay": str(replay), "replay_sha256": sha(replay)},
              "hypothesis": "Current recovery starts underrepresent a wall stray plus distant unpenned main flock and displaced dogs",
              "trajectory": trajectory_summary(bundle, cfg.environment.sheep_vision),
              "scenario_samples": sample_scenarios(cfg),
              "recovery": analyze_repetition_and_recovery(replay).to_dict(),
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "versions": collect_dependency_versions(), "capture": None,
              "limits": {"sampled_layouts": 300, "maximum_new_policy_episodes": int(capture),
                         "maximum_new_world_steps": 2320 if capture else 0, "capture_seconds": 120},
              "held_out": "unused", "training_transitions": 0}
    if capture:
        torch.set_num_threads(1)
        start = time.monotonic()
        model = MaskablePPO.load(str(parent), device="cpu")
        policy = NeuralPolicy(model, parent, NeuralPolicyConfig(hidden_sizes=(256, 256, 128),
                               observation_size=54, action_size=9, observation_mode="guided"))
        summary, _, _ = Evaluator(cfg, root / "capture").evaluate(
            policy, (59,), 0, capture_replays=True, capture_step_telemetry=True,
            telemetry_mode="head_tail", telemetry_head_steps=200, telemetry_tail_steps=500,
            evaluation_mode="development_seed59_passive_audit", checkpoint_id=source["parent"]["checkpoint_id"],
            should_stop=lambda: time.monotonic() - start >= 120)
        captured = load_replay(Path(summary.records[0].replay_path))
        def trace(b):
            return [(f["step"], f["actions"], f["snapshot"], f["reward"]) for f in b["frames"]]
        if trace(captured) != trace(bundle):
            raise AssertionError("Passive capture differs from saved baseline trajectory")
        report["capture"] = {"elapsed_seconds": time.monotonic() - start,
                             "world_steps": summary.records[0].steps,
                             "exact_saved_actions_snapshots_rewards_match": True,
                             "replay_path": summary.records[0].replay_path,
                             "telemetry_coverage": captured["telemetry_coverage"]}
        report["decision_summary"] = decision_summary(captured)
    report["parent_unchanged"] = sha(parent) == PARENT_SHA
    report["audit_source_sha256"] = sha(Path(__file__))
    atomic_write_json(root / "audit.json", report)
    manifest = {"status": "completed", "audit": "audit.json", "training_transitions": 0,
                "files_sha256": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}}
    atomic_write_json(root / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "root": str(root),
                      "capture": report["capture"],
                      "scenario_coverage": {f: {k: v for k, v in r.items() if k != "layouts"}
                                            for f, r in report["scenario_samples"].items()}}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-decisions", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.capture_decisions)
