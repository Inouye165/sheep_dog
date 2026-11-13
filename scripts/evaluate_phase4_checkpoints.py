"""Evaluate Phase 4 Role-Lock Stagnation Breaker on Pilot and Parent Checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.training.diagnostics.temporal_interaction import load_replay_bundle


def run_eval(
    model_path: Path,
    output_root: Path,
    run_id: str,
    chk_id: str,
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73),
) -> dict[str, Any]:
    """Run deterministic evaluation with full telemetry."""
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


def analyze_replay_roles(rep_path: Path) -> dict[str, Any]:
    """Inspect replay frames for role hold steps and role switches."""
    payload, _, _ = load_replay_bundle(rep_path)
    frames = payload.get("frames", [])
    stats = payload.get("stats", {})

    max_holds_by_dog: dict[str, int] = {"0": 0, "1": 0, "2": 0}
    role_hold_samples: list[dict[str, Any]] = []

    for f_idx, f in enumerate(frames):
        st = f.get("step_telemetry") or {}
        strat = st.get("strategy", {})
        holds = strat.get("role_hold_steps")
        if holds:
            for k, v in holds.items():
                k_str = str(k)
                if k_str in max_holds_by_dog:
                    max_holds_by_dog[k_str] = max(max_holds_by_dog[k_str], v)

        if f_idx in (0, 50, 100, 250, 500, 1000, 1500, 2000, len(frames) - 1):
            role_hold_samples.append({
                "step": f.get("step"),
                "holds": holds,
            })

    return {
        "success": stats.get("success"),
        "steps": stats.get("steps"),
        "sheep_penned": stats.get("sheep_penned"),
        "reward_total": stats.get("reward_total"),
        "role_switches": stats.get("role_switches"),
        "collector_activations": stats.get("collector_activations"),
        "max_holds": max_holds_by_dog,
        "samples": role_hold_samples,
    }


def main() -> None:
    """Run evaluations and comparative analysis for Phase 4."""
    seeds = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    pilot_model = Path("artifacts/experiments/stage34_recovery_pilot_v1/run/models/maskable-ppo-00057344.zip")
    parent_model = Path("artifacts/checkpoints/stage34/maskable-ppo-stage34-336663972.zip")

    # 1. Evaluate Phase 3 Pilot Checkpoint with Phase 4 strategy
    pilot_eval_dir = Path("artifacts/experiments/stage34_recovery_pilot_v1/evaluations_phase4")
    pilot_res = run_eval(pilot_model, pilot_eval_dir, "phase4_pilot", "pilot_chk00057344", seeds)

    # 2. Evaluate Parent Checkpoint with Phase 4 strategy
    parent_eval_dir = Path("artifacts/evaluations_phase4_parent")
    parent_res = run_eval(parent_model, parent_eval_dir, "phase4_parent", "parent_chk40398", seeds)

    # Historical Phase 3 Pilot Results:
    phase3_pilot = {
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

    print("\n" + "=" * 105)
    print("COMPARATIVE EVALUATION: PHASE 3 PILOT vs PHASE 4 PILOT vs PHASE 4 PARENT")
    print(f"{'Seed':<6} | {'Ph3 Succ':<9} {'Ph4Pil Succ':<12} {'Ph4Par Succ':<12} | {'Ph3 Pen':<8} {'Ph4Pil Pen':<11} {'Ph4Par Pen':<11} | {'Ph3 Rew':<10} {'Ph4Pil Rew':<11} {'Ph4Par Rew':<11}")
    print("-" * 105)

    for s in seeds:
        p3 = phase3_pilot[s]
        p4_pil = pilot_res["records"][s]
        p4_par = parent_res["records"][s]

        print(
            f"{s:<6} | {str(p3['success']):<9} {str(p4_pil['success']):<12} {str(p4_par['success']):<12} | "
            f"{p3['penned']:<8} {p4_pil['sheep_penned']:<11} {p4_par['sheep_penned']:<11} | "
            f"{p3['reward']:<10.2f} {p4_pil['reward_total']:<11.2f} {p4_par['reward_total']:<11.2f}"
        )

    print("-" * 105)
    p3_sr = sum(1 for p in phase3_pilot.values() if p["success"]) / len(phase3_pilot)
    p4_pil_sr = sum(1 for p in pilot_res["records"].values() if p["success"]) / len(seeds)
    p4_par_sr = sum(1 for p in parent_res["records"].values() if p["success"]) / len(seeds)

    p3_rew = sum(p["reward"] for p in phase3_pilot.values()) / len(phase3_pilot)
    p4_pil_rew = sum(p["reward_total"] for p in pilot_res["records"].values()) / len(seeds)
    p4_par_rew = sum(p["reward_total"] for p in parent_res["records"].values()) / len(seeds)

    print(f"Phase 3 Pilot:  Success Rate = {p3_sr:.1%} | Avg Reward = {p3_rew:.2f}")
    print(f"Phase 4 Pilot:  Success Rate = {p4_pil_sr:.1%} | Avg Reward = {p4_pil_rew:.2f} (Delta: {p4_pil_rew - p3_rew:+.2f})")
    print(f"Phase 4 Parent: Success Rate = {p4_par_sr:.1%} | Avg Reward = {p4_par_rew:.2f}")
    print("=" * 105)

    # 3. Replay Deep Dive on Seed 73, Seed 11, Seed 59
    print("\n" + "=" * 80)
    print("REPLAY ROLE-HOLD ANALYSIS: SEEDS 73, 11, 59")
    print("=" * 80)

    for target_seed in (73, 11, 59):
        print(f"\n--- SEED {target_seed} (Phase 4 Pilot Replay) ---")
        rep_path = pilot_eval_dir / "replays" / f"checkpoint-040400-seed-{target_seed:06d}.json.gz"
        if rep_path.exists():
            analysis = analyze_replay_roles(rep_path)
            print(f"Outcome:       Success={analysis['success']}, Steps={analysis['steps']}, Penned={analysis['sheep_penned']}/6, Reward={analysis['reward_total']:.2f}")
            print(f"Role Switches: {analysis['role_switches']} (vs Ph3 Seed 73: 2)")
            print(f"Collector Act: {analysis['collector_activations']}")
            print(f"Max Role Hold: {analysis['max_holds']} (vs Ph3 Seed 73: 1000)")
            print("Hold Progression:")
            for sample in analysis["samples"]:
                print(f"  Step {sample['step']:4d} | Holds={sample['holds']}")


if __name__ == "__main__":
    main()
