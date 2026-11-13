"""Analyze replays from Stage 34 Recovery Pilot (Phase 3)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from sheepdog.training.diagnostics.temporal_interaction import load_replay_bundle


def compute_entropy(probs: dict[str, float] | list[float]) -> float:
    """Compute Shannon entropy from probability distribution."""
    values = probs.values() if isinstance(probs, dict) else probs
    return -sum(p * math.log(max(p, 1e-12)) for p in values if p > 0)


def analyze_seed(target_seed: int, replay_dir: Path) -> dict[str, Any]:
    """Analyze a single evaluation seed replay."""
    rep_path = replay_dir / f"checkpoint-040399-seed-{target_seed:06d}.json.gz"
    assert rep_path.exists(), f"Replay {rep_path} not found"

    payload, _label, _ = load_replay_bundle(rep_path)
    frames = payload.get("frames", [])
    stats = payload.get("stats", {})
    init_snap = payload.get("initial_snapshot")

    has_step_zero = len(frames) > 0 and frames[0].get("step") == 0

    entropies_by_dog: dict[int, list[float]] = {0: [], 1: [], 2: []}
    values_by_dog: dict[int, list[float]] = {0: [], 1: [], 2: []}
    actions_by_dog: dict[int, dict[str, int]] = {0: {}, 1: {}, 2: {}}
    role_holds_by_step: list[dict[str, Any]] = []

    for f_idx, frame in enumerate(frames):
        st = frame.get("step_telemetry") or {}
        decisions = st.get("decisions", [])
        strat = st.get("strategy", {})

        for d in decisions:
            dog_idx = d.get("dog_index", 0)
            v = d.get("value_estimate")
            if v is not None:
                values_by_dog[dog_idx].append(v)
            probs = d.get("action_probabilities")
            if probs:
                entropies_by_dog[dog_idx].append(compute_entropy(probs))
            act = d.get("selected_action")
            if act:
                actions_by_dog[dog_idx][act] = actions_by_dog[dog_idx].get(act, 0) + 1

        if f_idx % 250 == 0 or f_idx in (0, len(frames) - 1):
            role_holds_by_step.append({
                "step": frame.get("step"),
                "holds": strat.get("role_hold_steps"),
                "sticky_dx": strat.get("sticky_pen_dx"),
                "sticky_dy": strat.get("sticky_pen_dy"),
                "v_est": [d.get("value_estimate") for d in decisions if d.get("value_estimate") is not None],
                "entropy": [
                    round(compute_entropy(d.get("action_probabilities", {})), 3)
                    for d in decisions
                    if d.get("action_probabilities")
                ],
            })

    # Find spatial trajectory highlights
    dog_positions_start = frames[0].get("snapshot", {}).get("dogs", []) if frames else []
    dog_positions_end = frames[-1].get("snapshot", {}).get("dogs", []) if frames else []
    sheep_positions_start = frames[0].get("snapshot", {}).get("sheep", []) if frames else []
    sheep_positions_end = frames[-1].get("snapshot", {}).get("sheep", []) if frames else []

    all_entropies = [e for d_list in entropies_by_dog.values() for e in d_list]
    all_values = [v for d_list in values_by_dog.values() for v in d_list]

    return {
        "seed": target_seed,
        "success": stats.get("success"),
        "stop_reason": stats.get("stop_reason"),
        "sheep_penned": stats.get("sheep_penned"),
        "steps": stats.get("steps"),
        "reward_total": stats.get("reward_total"),
        "has_step_zero": has_step_zero,
        "has_top_level_initial_snapshot": init_snap is not None,
        "total_frames": len(frames),
        "mean_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else 0.0,
        "min_entropy": min(all_entropies) if all_entropies else 0.0,
        "max_entropy": max(all_entropies) if all_entropies else 0.0,
        "mean_value": sum(all_values) / len(all_values) if all_values else 0.0,
        "min_value": min(all_values) if all_values else 0.0,
        "max_value": max(all_values) if all_values else 0.0,
        "start_values": [v_list[0] for v_list in values_by_dog.values() if v_list],
        "end_values": [v_list[-1] for v_list in values_by_dog.values() if v_list],
        "actions_by_dog": actions_by_dog,
        "sample_steps": role_holds_by_step,
        "pen_position": stats.get("pen_position"),
        "final_dog_positions": stats.get("final_dog_positions") or dog_positions_end,
        "final_sheep_positions": stats.get("final_sheep_positions") or sheep_positions_end,
        "initial_dog_positions": dog_positions_start,
        "initial_sheep_positions": sheep_positions_start,
        "role_distribution": stats.get("role_distribution"),
        "role_switches": stats.get("role_switches"),
        "collector_activations": stats.get("collector_activations"),
        "no_progress_steps": stats.get("no_progress_steps"),
        "reward_breakdown": stats.get("final_reward_breakdown", {}),
    }


def main() -> None:
    """Analyze evaluation seeds 59, 73, and 11."""
    replay_dir = Path("artifacts/experiments/stage34_recovery_pilot_v1/evaluations/replays")

    for seed in (59, 73, 11):
        res = analyze_seed(seed, replay_dir)
        print("=" * 80)
        print(f"DEEP REPLAY ANALYSIS: SEED {seed}")
        print("=" * 80)
        print(f"Outcome:       Success={res['success']}, Stop Reason={res['stop_reason']}, Steps={res['steps']}, Penned={res['sheep_penned']}/6")
        print(f"Total Reward:  {res['reward_total']:.2f}")
        print(f"Step 0 Frame:  Present={res['has_step_zero']}, Top-level Initial Snapshot Present={res['has_top_level_initial_snapshot']}, Frames={res['total_frames']}")
        print(f"Role Switches: {res['role_switches']}, Collector Activations: {res['collector_activations']}, No Progress Steps: {res['no_progress_steps']}")
        print(f"Entropy:       Mean={res['mean_entropy']:.3f}, Min={res['min_entropy']:.3f}, Max={res['max_entropy']:.3f} (Max theoretical for 9 acts: {math.log(9):.3f})")
        print(f"Value V(s):    Start={res['start_values']}, End={res['end_values']}, Range=[{res['min_value']:.2f}, {res['max_value']:.2f}]")
        print(f"Pen Position:  {res['pen_position']}")
        print("\nSampled Progression:")
        for s in res["sample_steps"]:
            print(f"  Step {s['step']:4d} | V(s)={[round(v, 2) for v in s['v_est']]} | Entropy={s['entropy']} | Role Holds={s['holds']} | Sticky=({s['sticky_dx']}, {s['sticky_dy']})")

        print("\nActions by Dog (Top 3 per dog):")
        for dog_i, acts in res["actions_by_dog"].items():
            top = sorted(acts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(f"  Dog {dog_i}: {top}")

        rb = res["reward_breakdown"]
        print("\nTop Reward Breakdown Components:")
        for k, v in sorted(rb.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
            print(f"  - {k}: {v:.2f}")
        print()


if __name__ == "__main__":
    main()
