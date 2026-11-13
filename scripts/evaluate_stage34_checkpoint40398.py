"""Deterministic evaluation of Stage 34 checkpoint 40398 across 10 evaluation seeds."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy


def main() -> None:
    """Run deterministic 10-seed evaluation and print comparative report."""
    seeds = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    checkpoint_episode = 40398
    model_path = Path("artifacts/checkpoints/stage34/maskable-ppo-stage34-336663972.zip")

    print(f"Loading Stage 34 config and model from {model_path}...")
    cfg = apply_curriculum_stage(LabConfig(), 34)
    policy = NeuralPolicy.load(str(model_path), config=cfg)

    evaluator = Evaluator(cfg, output_root=Path("artifacts/evaluations"))

    summary_json_path = Path("artifacts/evaluations/eval_maskable-ppo-stage34-336663972_confidence_ff85663fc223.json")
    if not summary_json_path.exists():
        print(f"Running deterministic evaluation on seeds {seeds}...")
        summary, summary_json_path, summary_csv_path = evaluator.evaluate(
            policy=policy,
            seeds=seeds,
            checkpoint_episode=checkpoint_episode,
            deterministic=True,
            capture_replays=True,
            capture_step_telemetry=True,
            evaluation_mode="confidence",
            curriculum_stage=34,
            run_id="eval_phase2_blindspot_fix",
            checkpoint_id="maskable-ppo-stage34-336663972",
        )

    with summary_json_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Success Rate: {eval_data.get('success_rate', 0):.2%}")
    print(f"Average Steps: {eval_data.get('mean_steps', 0):.1f}")
    print(f"Average Penned: {eval_data.get('average_sheep_penned', 0):.2f}/6")
    print(f"Average Reward: {eval_data.get('average_reward_total', 0):.2f}")
    print("=" * 80)

    # Historical pre-fix baseline for checkpoint 40398:
    pre_fix_40398 = {
        11: {"success": True, "penned": 6, "steps": 643, "reward": -1135.48},
        23: {"success": True, "penned": 6, "steps": 281, "reward": 56.55},
        37: {"success": True, "penned": 6, "steps": 506, "reward": -436.97},
        41: {"success": True, "penned": 6, "steps": 263, "reward": -66.95},
        53: {"success": True, "penned": 6, "steps": 464, "reward": -363.31},
        59: {"success": False, "penned": 0, "steps": 2320, "reward": -8618.09},
        61: {"success": True, "penned": 6, "steps": 258, "reward": 319.26},
        67: {"success": True, "penned": 6, "steps": 351, "reward": -250.89},
        71: {"success": True, "penned": 6, "steps": 369, "reward": -64.81},
        73: {"success": False, "penned": 0, "steps": 2320, "reward": -10334.56},
    }

    # Historical failure replay for seed 59 at checkpoint 40199:
    # checkpoint-040199-seed-000059.json.gz:
    # success=False, penned=4, steps=2320, reward=-6158.17

    print("\nPER-SEED COMPARISON (Pre-fix vs Post-fix):")
    print(f"{'Seed':<6} | {'Pre Success':<12} {'Post Success':<13} | {'Pre Penned':<11} {'Post Penned':<12} | {'Pre Steps':<10} {'Post Steps':<11} | {'Pre Reward':<12} {'Post Reward':<12}")
    print("-" * 95)

    with summary_json_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    post_results = {r["seed"]: r for r in eval_data.get("records", [])}

    regressions = []
    improvements = []

    for s in seeds:
        pre = pre_fix_40398[s]
        post = post_results[s]

        pre_succ = str(pre["success"])
        post_succ = str(post["success"])
        pre_pen = str(pre["penned"])
        post_pen = str(post["sheep_penned"])
        pre_step = str(pre["steps"])
        post_step = str(post["steps"])
        pre_rew = f"{pre['reward']:.2f}"
        post_rew = f"{post['reward_total']:.2f}"

        status_marker = ""
        if pre["success"] and not post["success"]:
            status_marker = " [REGRESSION!]"
            regressions.append(s)
        elif not pre["success"] and post["success"]:
            status_marker = " [IMPROVEMENT!]"
            improvements.append(s)
        elif post["reward_total"] > pre["reward"] + 100:
            status_marker = " [REWARD +]"
        elif post["reward_total"] < pre["reward"] - 100:
            status_marker = " [REWARD -]"

        print(f"{s:<6} | {pre_succ:<12} {post_succ:<13} | {pre_pen:<11} {post_pen:<12} | {pre_step:<10} {post_step:<11} | {pre_rew:<12} {post_rew:<12}{status_marker}")

    print("-" * 95)
    print(f"Total Regressions: {len(regressions)} {regressions}")
    print(f"Total Improvements: {len(improvements)} {improvements}")

    # Inspect Seed 59 replay in depth:
    rep_59_path = Path("artifacts/evaluations/replays/checkpoint-040398-seed-000059.json.gz")
    if rep_59_path.exists():
        with gzip.open(rep_59_path, "rt", encoding="utf-8") as f:
            rep_59 = json.load(f)
        s59_stats = rep_59["stats"]
        print("\nSEED 59 DETAILED ANALYSIS (Post-fix Checkpoint 40398):")
        print(f"Success: {s59_stats['success']}")
        print(f"Stop Reason: {s59_stats['stop_reason']}")
        print(f"Steps: {s59_stats['steps']}")
        print(f"Sheep Penned: {s59_stats['sheep_penned']}/6")
        print(f"Reward Total: {s59_stats['reward_total']:.2f} (vs Pre-fix 40398: -8618.09, vs 40199: -6158.17)")
        print(f"Collector Activations: {s59_stats.get('collector_activations')}")
        print(f"Role Switches: {s59_stats.get('role_switches')}")
        print(f"No Progress Steps: {s59_stats.get('no_progress_steps')}")
        print(f"Role Distribution: {s59_stats.get('role_distribution')}")
        rb = s59_stats.get("final_reward_breakdown", {})
        print("Top Reward Penalties / Credits:")
        for k, v in sorted(rb.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
            print(f"  - {k}: {v:.2f}")


if __name__ == "__main__":
    main()
