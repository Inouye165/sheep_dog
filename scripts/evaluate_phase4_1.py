"""Evaluate Phase 4.1 (Threshold 150) on Pilot Checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy


def run_eval(
    model_path: Path,
    output_root: Path,
    run_id: str,
    chk_id: str,
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73),
) -> dict[str, Any]:
    """Run deterministic evaluation on specified seeds."""
    print(f"\n--- Evaluating {model_path.name} on seeds {seeds} ---")
    cfg = apply_curriculum_stage(LabConfig(), 34)
    policy = NeuralPolicy.load(str(model_path), config=cfg)
    evaluator = Evaluator(cfg, output_root=output_root)

    summary, json_path, _ = evaluator.evaluate(
        policy=policy,
        seeds=seeds,
        checkpoint_episode=40400,
        deterministic=True,
        capture_replays=True,
        capture_step_telemetry=True,
        evaluation_mode="confidence",
        curriculum_stage=34,
        run_id=run_id,
        checkpoint_id=chk_id,
    )
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "summary": summary,
        "json_path": json_path,
        "records": {r["seed"]: r for r in data.get("records", [])},
    }


def main() -> None:
    """Execute comparative 10-seed evaluation for Phase 4.1."""
    seeds = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    pilot_model = Path("artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip")
    output_dir = Path("artifacts/experiments/stage34_recovery_pilot_v1/evaluations_phase4_1")

    res = run_eval(pilot_model, output_dir, "phase4_1_pilot", "pilot_chk00057344", seeds)

    # Historical Phase 3 (threshold=None) & Phase 4 (threshold=60)
    phase3 = {
        11: {"success": False, "penned": 0, "steps": 2320, "reward": -6492.31},
        23: {"success": True, "penned": 6, "steps": 236, "reward": 325.47},
        37: {"success": True, "penned": 6, "steps": 226, "reward": 521.84},
        41: {"success": True, "penned": 6, "steps": 201, "reward": 282.50},
        53: {"success": True, "penned": 6, "steps": 309, "reward": 293.18},
        59: {"success": True, "penned": 6, "steps": 336, "reward": 176.98},
        61: {"success": True, "penned": 6, "steps": 257, "reward": 351.78},
        67: {"success": True, "penned": 6, "steps": 552, "reward": -186.56},
        71: {"success": True, "penned": 6, "steps": 325, "reward": 315.20},
        73: {"success": False, "penned": 0, "steps": 2320, "reward": -4020.96},
    }
    phase4 = {
        11: {"success": True, "penned": 6, "steps": 793, "reward": -744.93},
        23: {"success": True, "penned": 6, "steps": 221, "reward": 341.38},
        37: {"success": True, "penned": 6, "steps": 268, "reward": 477.56},
        41: {"success": True, "penned": 6, "steps": 322, "reward": 141.05},
        53: {"success": True, "penned": 6, "steps": 371, "reward": 225.07},
        59: {"success": True, "penned": 6, "steps": 472, "reward": -1.55},
        61: {"success": True, "penned": 6, "steps": 412, "reward": 179.80},
        67: {"success": True, "penned": 6, "steps": 501, "reward": -127.33},
        71: {"success": True, "penned": 6, "steps": 462, "reward": 160.01},
        73: {"success": False, "penned": 0, "steps": 2320, "reward": -3835.43},
    }

    print("\n" + "=" * 125)
    print("COMPARATIVE EVALUATION: PHASE 3 (No decay) vs PHASE 4 (Decay @ 60) vs PHASE 4.1 (Decay @ 150)")
    print(f"{'Seed':<6} | {'Ph3 Succ':<8} {'Ph4 Succ':<8} {'Ph4.1 Succ':<10} | {'Ph3 Steps':<9} {'Ph4 Steps':<9} {'Ph4.1 Steps':<11} | {'Ph3 Rew':<10} {'Ph4 Rew':<10} {'Ph4.1 Rew':<11}")
    print("-" * 125)

    for s in seeds:
        p3 = phase3[s]
        p4 = phase4[s]
        p4_1 = res["records"][s]

        print(
            f"{s:<6} | {str(p3['success']):<8} {str(p4['success']):<8} {str(p4_1['success']):<10} | "
            f"{p3['steps']:<9} {p4['steps']:<9} {p4_1['steps']:<11} | "
            f"{p3['reward']:<10.2f} {p4['reward']:<10.2f} {p4_1['reward_total']:<11.2f}"
        )

    print("-" * 125)
    p3_sr = sum(1 for p in phase3.values() if p["success"]) / len(seeds)
    p4_sr = sum(1 for p in phase4.values() if p["success"]) / len(seeds)
    p4_1_sr = sum(1 for p in res["records"].values() if p["success"]) / len(seeds)

    p3_rew = sum(p["reward"] for p in phase3.values()) / len(seeds)
    p4_rew = sum(p["reward"] for p in phase4.values()) / len(seeds)
    p4_1_rew = sum(p["reward_total"] for p in res["records"].values()) / len(seeds)

    p3_steps = sum(p["steps"] for p in phase3.values() if p["success"]) / sum(1 for p in phase3.values() if p["success"])
    p4_steps = sum(p["steps"] for p in phase4.values() if p["success"]) / sum(1 for p in phase4.values() if p["success"])
    p4_1_steps = sum(p["steps"] for p in res["records"].values() if p["success"]) / max(1, sum(1 for p in res["records"].values() if p["success"]))

    print(f"Phase 3 (Pilot No-Decay):   Success Rate = {p3_sr:.1%} | Avg Steps (Succ) = {p3_steps:.1f} | Avg Reward = {p3_rew:.2f}")
    print(f"Phase 4 (Pilot Decay @ 60):  Success Rate = {p4_sr:.1%} | Avg Steps (Succ) = {p4_steps:.1f} | Avg Reward = {p4_rew:.2f}")
    print(f"Phase 4.1 (Pilot Decay @ 150): Success Rate = {p4_1_sr:.1%} | Avg Steps (Succ) = {p4_1_steps:.1f} | Avg Reward = {p4_1_rew:.2f}")
    print("=" * 125)


if __name__ == "__main__":
    main()
