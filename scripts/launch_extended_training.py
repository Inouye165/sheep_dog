"""Launch Extended Stage 34 Continuous Training with Phase 4.1 Telemetry and Strategy."""

from __future__ import annotations

import argparse
import copy
import signal
import sys
from pathlib import Path
from typing import Any

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.training.maskable_ppo import MaskablePPOTrainer
from sheepdog.training.replay_writer import (
    CapturePolicy,
    get_global_capture_policy,
    set_global_capture_policy,
)

DEFAULT_SEEDS = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments for extended training."""
    parser = argparse.ArgumentParser(
        description="Launch extended Stage 34 training with recovery hyperparams and authentic telemetry."
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=50,
        help="Number of checkpoint segments to train (default: 50).",
    )
    parser.add_argument(
        "--timesteps-per-checkpoint",
        type=int,
        default=20_480,
        help="Policy transition steps per checkpoint (default: 20480, which is 10 PPO rollout updates).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.015,
        help="PPO policy entropy coefficient (default: 0.015).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.997,
        help="PPO discount factor (default: 0.997).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=34,
        help="Curriculum stage (default: 34).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts"),
        help="Experiment / training output root directory (default: artifacts).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a 1-checkpoint minimal test run (128 steps, 1 eval seed) to verify end-to-end pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify configuration and checkpoint compatibility without training.",
    )
    return parser


def main() -> None:
    """Launch extended Stage 34 training run."""
    parser = parse_args()
    args = parser.parse_args()

    checkpoints = 1 if args.test_run else args.checkpoints
    timesteps_per_checkpoint = 128 if args.test_run else args.timesteps_per_checkpoint
    total_timesteps = checkpoints * timesteps_per_checkpoint
    eval_seeds = (11,) if args.test_run else DEFAULT_SEEDS

    print("=" * 80)
    print("STAGE 34 EXTENDED TRAINING LAUNCHER")
    print(f"Curriculum Stage:        {args.curriculum_stage}")
    print(f"Checkpoints:             {checkpoints}")
    print(f"Timesteps / Checkpoint:  {timesteps_per_checkpoint}")
    print(f"Total Timesteps:         {total_timesteps}")
    print(f"Entropy Coef:            {args.entropy_coef}")
    print(f"Gamma:                   {args.gamma}")
    print(f"Learning Rate:           {args.learning_rate}")
    print(f"Evaluation Seeds:        {eval_seeds}")
    print(f"Output Root:             {args.output_root}")
    print(f"Test Run Mode:           {args.test_run}")
    print("=" * 80)

    # 1. Build LabConfig with Stage 34 and hyperparams
    base_cfg = apply_training_profile(LabConfig(), curriculum_stage=args.curriculum_stage)
    cfg_dict = copy.deepcopy(base_cfg.to_dict())
    if "rewards" in cfg_dict and "instincts" in cfg_dict["rewards"]:
        cfg_dict["rewards"]["instincts"]["curriculum_stage"] = args.curriculum_stage

    train_dict = cfg_dict["training"]
    train_dict["entropy_coef"] = args.entropy_coef
    train_dict["gamma"] = args.gamma
    train_dict["learning_rate"] = args.learning_rate
    train_dict["learning_rate_final"] = args.learning_rate
    train_dict["total_timesteps"] = total_timesteps
    train_dict["rollout_steps"] = min(2048, timesteps_per_checkpoint)
    train_dict["batch_size"] = min(1024, timesteps_per_checkpoint)

    # Checkpoint targets
    checkpoint_timesteps = tuple(
        timesteps_per_checkpoint * i for i in range(1, checkpoints + 1)
    )
    train_dict["checkpoint_timesteps"] = checkpoint_timesteps
    train_dict["checkpoint_episodes"] = tuple(range(1, checkpoints + 1))
    train_dict["episodes"] = checkpoints
    train_dict["evaluation_seeds"] = eval_seeds
    train_dict["quick_evaluation_seed_count"] = len(eval_seeds)
    train_dict["confidence_candidate_success_rate"] = 0.0  # Always run confidence evaluation
    train_dict["enable_adaptive_learning"] = False  # Lock entropy & LR to targeted values
    train_dict["output_dir"] = str(args.output_root)

    cfg = LabConfig.from_dict(cfg_dict)

    # 2. Instantiate Trainer & Check Compatibility
    trainer = MaskablePPOTrainer(cfg, output_root=args.output_root)
    if not trainer.has_compatible_policy_state():
        print("ERROR: Incompatible policy state detected in output root!")
        sys.exit(1)

    print("Trainer initialized successfully. Compatible policy state confirmed.")
    loaded_state = trainer._loaded_state
    print(f"Resuming from policy state: {loaded_state.get('policy_state_path')}")
    print(f"Starting total episodes:    {loaded_state.get('total_episodes_trained', 0)}")
    print(f"Starting total timesteps:   {loaded_state.get('total_timesteps', 0)}")

    if args.dry_run:
        print("\nDry run requested. Verification complete, exiting without training.")
        return

    # Graceful stop handler
    stop_requested = False

    def _sig_handler(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"\nReceived signal {signum}. Requesting graceful stop after current segment...")
        stop_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    def _progress(payload: dict[str, Any]) -> None:
        phase = payload.get("phase")
        msg = payload.get("message", "")
        if phase in (
            "learning",
            "checkpoint",
            "checkpoint_saved",
            "evaluating",
            "evaluation_complete",
            "episode_complete",
            "confidence_evaluation",
        ):
            print(f"  [{phase}] {msg}", flush=True)

    prev_capture = get_global_capture_policy()
    set_global_capture_policy(CapturePolicy(mode="off"))

    print("\n--- Starting Extended Stage 34 Training ---")
    try:
        summary = trainer.train(
            progress_callback=_progress,
            should_stop=lambda: stop_requested,
        )
        print("\n" + "=" * 80)
        print("TRAINING RUN COMPLETED")
        print(f"Final Model Path: {summary.final_model_path}")
        print(f"Checkpoints Saved: {len(summary.checkpoints)}")
        print("=" * 80)
    finally:
        set_global_capture_policy(prev_capture)


if __name__ == "__main__":
    main()
