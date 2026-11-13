"""Execute Stage 34 Recovery Pilot (Phase 3: stage34_recovery_entropy_gamma_v1)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_curriculum_stage
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy
from sheepdog.server import compute_promotion_gate_snapshot
from sheepdog.training.diagnostics.temporal_interaction import load_replay_bundle
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.replay_writer import (
    CapturePolicy,
    get_global_capture_policy,
    set_global_capture_policy,
)


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_entropy(probs: dict[str, float] | list[float]) -> float:
    """Compute Shannon entropy from probability distribution."""
    values = probs.values() if isinstance(probs, dict) else probs
    return -sum(p * math.log(max(p, 1e-12)) for p in values if p > 0)


def main() -> None:
    """Run isolated bounded pilot, formal evaluation, and deep telemetry diagnostics."""
    experiment_root = Path("artifacts/experiments/stage34_recovery_pilot_v1")
    parent_model_path = Path("artifacts/checkpoints/stage34/maskable-ppo-stage34-336663972.zip")

    print("=" * 80)
    print("PHASE 3: STAGE 34 RECOVERY PILOT (stage34_recovery_entropy_gamma_v1)")
    print(f"Parent Checkpoint: {parent_model_path}")
    print(f"Target Directory: {experiment_root}")
    print("=" * 80)

    # 1. Freeze Parent Model Copy and Verify Immutability
    frozen_dir = experiment_root / "parent"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_parent_copy = frozen_dir / parent_model_path.name
    if not frozen_parent_copy.exists():
        print(f"Freezing copy of parent model to {frozen_parent_copy}...")
        shutil.copy2(parent_model_path, frozen_parent_copy)

    parent_sha = sha256_file(parent_model_path)
    frozen_sha = sha256_file(frozen_parent_copy)
    print(f"Parent Model SHA-256: {parent_sha}")
    print(f"Frozen Copy  SHA-256: {frozen_sha}")
    assert parent_sha == frozen_sha, "Parent SHA and Frozen Copy SHA must match!"

    # 2. Configure Isolated Run
    run_dir = experiment_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = apply_curriculum_stage(LabConfig(), 34)
    cfg_dict = copy.deepcopy(base_cfg.to_dict())

    train_dict = cfg_dict["training"]
    # Pilot Hyperparameters:
    train_dict["entropy_coef"] = 0.02
    train_dict["gamma"] = 0.997
    train_dict["train_seed"] = 101
    train_dict["learning_rate"] = 5e-5
    train_dict["learning_rate_final"] = 5e-5
    train_dict["enable_adaptive_learning"] = False
    train_dict["failure_directed_training_enabled"] = False
    train_dict["scenario_training_enabled"] = False
    train_dict["auto_rollback_on_collapse"] = False
    train_dict["wandb_enabled"] = False
    train_dict["output_dir"] = str(run_dir)

    # Bounded budget: 20,480 policy transitions (1 checkpoint update)
    budget_transitions = 20_480
    rollout_steps = 2_048
    train_dict["rollout_steps"] = rollout_steps
    train_dict["total_timesteps"] = budget_transitions
    train_dict["checkpoint_timesteps"] = (budget_transitions,)
    train_dict["checkpoint_episodes"] = (1,)
    train_dict["episodes"] = 1
    train_dict["evaluation_seeds"] = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    train_dict["quick_evaluation_seed_count"] = 10

    pilot_cfg = LabConfig.from_dict(cfg_dict)

    # 3. Seed Isolated training-state.json
    state = {
        "trainer_type": "maskable_ppo",
        "policy_type": "neural",
        "policy_state_path": str(frozen_parent_copy.resolve()),
        "best_model_path": str(frozen_parent_copy.resolve()),
        "best_model_curriculum_stage": 34,
        "best_success_rate": 0.8,
        "best_average_reward": -857.31,
        "best_completion_steps": 798.6,
        "total_episodes_trained": 40398,
        "total_environment_episodes": 40398,
        "total_timesteps": 36864,
        "policy_version": 10276,
        "run_id": "run_stage34_recovery_pilot_v1",
        "parent_run_id": "run_20260817_063453_91f4",
        "parent_checkpoint_id": "maskable-ppo-stage34-336663972",
        "training_signature": {"action_size": 9, "observation_mode": "guided"},
        "policy_config": {
            "hidden_sizes": [256, 256, 128],
            "observation_size": 54,
            "action_size": 9,
            "env_workers": 1,
        },
    }
    (run_dir / "training-state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Save effective config
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    (run_dir / "effective-training-config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    print("\nStarting MaskablePPO training run (20,480 policy transitions)...")
    prev_capture = get_global_capture_policy()
    set_global_capture_policy(CapturePolicy(mode="off"))

    trainer = MaskablePPOTrainer(pilot_cfg, output_root=run_dir)
    assert trainer.has_compatible_policy_state(), "Policy state must be compatible!"

    def _progress(payload: dict[str, Any]) -> None:
        phase = payload.get("phase")
        msg = payload.get("message", "")
        if phase in ("learning", "checkpoint", "checkpoint_saved", "episode_complete"):
            print(f"  [{phase}] {msg}")

    try:
        _ = trainer.train(progress_callback=_progress)
        print("Training completed successfully!")
    finally:
        set_global_capture_policy(prev_capture)

    # 4. Verify Parent Immutability Post-Run
    post_parent_sha = sha256_file(parent_model_path)
    assert post_parent_sha == parent_sha, "CRITICAL: Parent model was modified during training!"
    print(f"\nParent checkpoint integrity verified: {post_parent_sha} (Unchanged)")

    # 5. Locate Post-Pilot Checkpoint Model
    stage_model_dir = run_dir / "checkpoints" / "stage34"
    saved_models = sorted(stage_model_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not saved_models:
        # Fallback to models/ directory if saved there
        saved_models = sorted((run_dir / "models").glob("*.zip"), key=lambda p: p.stat().st_mtime)

    assert saved_models, "No trained model zip found in checkpoint directories!"
    trained_model_path = saved_models[-1]
    print(f"\nTrained Pilot Model: {trained_model_path} ({trained_model_path.stat().st_size} bytes)")

    # 6. Formal Post-Pilot Evaluation on 10 Seeds
    eval_seeds = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
    eval_output_dir = experiment_root / "evaluations"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning formal post-pilot evaluation on seeds {eval_seeds} with telemetry...")
    eval_policy = NeuralPolicy.load(str(trained_model_path), config=pilot_cfg)
    evaluator = Evaluator(pilot_cfg, output_root=eval_output_dir)

    eval_summary, eval_json_path, _ = evaluator.evaluate(
        policy=eval_policy,
        seeds=eval_seeds,
        checkpoint_episode=40399,
        deterministic=True,
        capture_replays=True,
        capture_step_telemetry=True,
        evaluation_mode="confidence",
        curriculum_stage=34,
        run_id="run_stage34_recovery_pilot_v1",
        checkpoint_id=trained_model_path.stem,
    )

    with eval_json_path.open("r", encoding="utf-8") as f:
        post_eval_data = json.load(f)

    post_results = {r["seed"]: r for r in post_eval_data.get("records", [])}

    # 7. Compute Promotion Gate Snapshot
    print("\nComputing promotion gate snapshot...")
    gate_snapshot = compute_promotion_gate_snapshot(
        output_root=eval_output_dir,
        target_ep=40399,
        journey="standard",
    )
    print("Promotion Gate Snapshot:")
    print(f"  Ready: {gate_snapshot.get('ready')}")
    print(f"  Decision: {gate_snapshot.get('decision')}")
    print(f"  Status: {gate_snapshot.get('status_text')}")
    print(f"  Reason: {gate_snapshot.get('reason')}")
    print(f"  Blocking Reasons: {gate_snapshot.get('blocking_reasons')}")

    # 8. Comparative Table: Pre-Pilot (Phase 2 Static) vs Post-Pilot (Phase 3)
    pre_pilot = {
        11: {"success": True, "penned": 6, "steps": 643, "reward": -423.00},
        23: {"success": True, "penned": 6, "steps": 224, "reward": 306.26},
        37: {"success": True, "penned": 6, "steps": 439, "reward": 189.61},
        41: {"success": True, "penned": 6, "steps": 363, "reward": -141.73},
        53: {"success": True, "penned": 6, "steps": 734, "reward": -772.45},
        59: {"success": False, "penned": 0, "steps": 2320, "reward": -3808.72},
        61: {"success": True, "penned": 6, "steps": 258, "reward": 415.47},
        67: {"success": True, "penned": 6, "steps": 340, "reward": 135.79},
        71: {"success": True, "penned": 6, "steps": 345, "reward": 260.71},
        73: {"success": False, "penned": 0, "steps": 2320, "reward": -4735.06},
    }

    print("\n" + "=" * 95)
    print("COMPARATIVE EVALUATION: PHASE 2 (Pre-pilot) vs PHASE 3 (Post-pilot)")
    print(f"{'Seed':<6} | {'Pre Succ':<9} {'Post Succ':<10} | {'Pre Pen':<8} {'Post Pen':<9} | {'Pre Steps':<9} {'Post Steps':<10} | {'Pre Reward':<11} {'Post Reward':<11} | {'Status'}")
    print("-" * 95)

    regressions = []
    improvements = []

    for s in eval_seeds:
        pre = pre_pilot[s]
        post = post_results[s]

        pre_s = str(pre["success"])
        post_s = str(post["success"])
        pre_p = str(pre["penned"])
        post_p = str(post["sheep_penned"])
        pre_st = str(pre["steps"])
        post_st = str(post["steps"])
        pre_r = f"{pre['reward']:.2f}"
        post_r = f"{post['reward_total']:.2f}"

        marker = ""
        if pre["success"] and not post["success"]:
            marker = "REGRESSION!"
            regressions.append(s)
        elif not pre["success"] and post["success"]:
            marker = "IMPROVEMENT!"
            improvements.append(s)
        elif post["reward_total"] > pre["reward"] + 50:
            marker = "REWARD +"
        elif post["reward_total"] < pre["reward"] - 50:
            marker = "REWARD -"

        print(f"{s:<6} | {pre_s:<9} {post_s:<10} | {pre_p:<8} {post_p:<9} | {pre_st:<9} {post_st:<10} | {pre_r:<11} {post_r:<11} | {marker}")

    print("-" * 95)
    pre_avg_rew = sum(p["reward"] for p in pre_pilot.values()) / len(pre_pilot)
    post_avg_rew = sum(p["reward_total"] for p in post_results.values()) / len(post_results)
    pre_sr = sum(1 for p in pre_pilot.values() if p["success"]) / len(pre_pilot)
    post_sr = sum(1 for p in post_results.values() if p["success"]) / len(post_results)
    print(f"Pre-pilot Success Rate:  {pre_sr:.1%}  | Avg Reward: {pre_avg_rew:.2f}")
    print(f"Post-pilot Success Rate: {post_sr:.1%}  | Avg Reward: {post_avg_rew:.2f} (Delta: {post_avg_rew - pre_avg_rew:+.2f})")
    print(f"Total Regressions: {len(regressions)} {regressions}")
    print(f"Total Improvements: {len(improvements)} {improvements}")
    print("=" * 95)

    # 9. Deep Replay Telemetry Analysis on Seed 59 & Seed 73
    replay_dir = eval_output_dir / "replays"
    print("\n" + "=" * 80)
    print("DEEP REPLAY TELEMETRY ANALYSIS (Seeds 59 & 73)")
    print("=" * 80)

    for target_seed in (59, 73):
        rep_path = replay_dir / f"checkpoint-040399-seed-{target_seed:06d}.json.gz"
        if not rep_path.exists():
            print(f"Warning: Replay file {rep_path} does not exist!")
            continue

        print(f"\n--- SEED {target_seed} TELEMETRY DEEP DIVE ---")
        bundle = load_replay_bundle(rep_path)
        payload = bundle.payload
        frames = payload.get("frames", [])
        stats = payload.get("stats", {})

        print(f"Replay File: {rep_path.name}")
        print(f"Step 0 Synthesized/Present: {bundle.has_initial_step_zero} (Total frames: {len(frames)})")
        print(f"Outcome: Success={stats.get('success')}, Stop Reason={stats.get('stop_reason')}, Penned={stats.get('sheep_penned')}/6, Steps={stats.get('steps')}")
        print(f"Total Reward: {stats.get('reward_total'):.2f}")

        # Analyze decisions telemetry across time:
        entropies = []
        values = []
        action_counts = {}

        sampled_steps = [0, 50, 200, 500, 1000, 1500, 2000, min(2319, len(frames) - 1)]

        for f_idx in sampled_steps:
            if f_idx >= len(frames):
                continue
            frame = frames[f_idx]
            st = frame.get("step_telemetry") or {}
            decisions = st.get("decisions", [])
            strat = st.get("strategy", {})

            v_est = [d.get("value_estimate") for d in decisions if d.get("value_estimate") is not None]
            ent_est = [
                compute_entropy(d.get("action_probabilities", {}))
                for d in decisions
                if d.get("action_probabilities")
            ]
            holds = strat.get("role_hold_steps", {})

            print(f"  Step {f_idx:4d} | V(s)={v_est[:3]} | Entropy={[f'{e:.3f}' for e in ent_est[:3]]} | Role Holds={holds}")

        # Full episode aggregate statistics:
        for frame in frames:
            st = frame.get("step_telemetry") or {}
            for d in st.get("decisions", []):
                v = d.get("value_estimate")
                if v is not None:
                    values.append(v)
                probs = d.get("action_probabilities")
                if probs:
                    entropies.append(compute_entropy(probs))
                act = d.get("selected_action")
                if act:
                    action_counts[act] = action_counts.get(act, 0) + 1

        if entropies:
            print(f"Mean Decision Entropy: {sum(entropies)/len(entropies):.4f} (Max possible for 9 actions: {math.log(9):.4f})")
            print(f"Min / Max Entropy:     {min(entropies):.4f} / {max(entropies):.4f}")
        if values:
            print(f"Value Estimate V(s):   Start={values[0]:.2f}, Mid={values[len(values)//2]:.2f}, End={values[-1]:.2f}, Min={min(values):.2f}, Max={max(values):.2f}")
        print(f"Top Executed Actions:  {sorted(action_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]}")

        final_st = frames[-1].get("step_telemetry") or {}
        final_strat = final_st.get("strategy", {})
        print(f"Final Strategy Axes:   sticky_pen_dx={final_strat.get('sticky_pen_dx')}, sticky_pen_dy={final_strat.get('sticky_pen_dy')}")
        print(f"Final Role Holds:      {final_strat.get('role_hold_steps')}")
        print(f"Final Dog Positions:   {stats.get('final_dog_positions')}")
        print(f"Final Sheep Positions: {stats.get('final_sheep_positions')}")
        print(f"Pen Position:          {stats.get('pen_position')}")


if __name__ == "__main__":
    main()
