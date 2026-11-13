"""CLI runner for the controlled Stage 34 recovery hyperparameter experiment (Arms A, B, C)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sheepdog.recovery_hyperparam_experiment import (  # noqa: E402
    DEFAULT_LATEST_PARENT_CHECKPOINT,
    DEFAULT_PAIRED_TRAINING_SEEDS,
    EXPERIMENT_ARMS,
    PROPOSED_PILOT_CHECKPOINT_COUNT,
    PROPOSED_PILOT_ROLLOUT_STEPS,
    PROPOSED_PILOT_TRANSITIONS_PER_ARM_SEED,
    build_transition_budget,
    evaluate_experiment_promotion_guardrails,
    prepare_experiment_workspace,
    run_controlled_arm_seed,
    sha256_file,
    verify_loaded_arm_models,
)

DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments/recovery_hyperparam_prep_v2")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, verify, or execute the controlled three-dog recovery "
            "hyperparameter experiment (Arm A control, Arm B entropy=0.02, Arm C gamma=0.997)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser(
        "prepare",
        help="Prepare isolated experiment workspace, configs, and manifests without starting training.",
    )
    prep.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Isolated directory for experiment manifests, configs, and arm outputs.",
    )
    prep.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=DEFAULT_LATEST_PARENT_CHECKPOINT,
        help="Parent Stage 34 checkpoint .zip to freeze (never overwritten).",
    )
    prep.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_PAIRED_TRAINING_SEEDS),
        help="Paired independent training seeds across all arms.",
    )
    prep.add_argument(
        "--transitions-per-seed",
        type=int,
        default=PROPOSED_PILOT_TRANSITIONS_PER_ARM_SEED,
        help="Additional policy transitions per arm and seed.",
    )
    prep.add_argument(
        "--checkpoints",
        type=int,
        default=PROPOSED_PILOT_CHECKPOINT_COUNT,
        help="Number of checkpoints per arm and seed.",
    )
    prep.add_argument(
        "--rollout-steps",
        type=int,
        default=PROPOSED_PILOT_ROLLOUT_STEPS,
        help="PPO rollout buffer steps per update.",
    )

    ver = subparsers.add_parser(
        "verify",
        help="Verify parent checkpoint immutability, single-parameter diffs, and loaded SB3 trainer settings.",
    )
    ver.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Prepared experiment root directory.",
    )

    run_one = subparsers.add_parser(
        "run-arm",
        help="Run one isolated (arm, seed) configuration after pilot approval.",
    )
    run_one.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Prepared experiment root directory.",
    )
    run_one.add_argument(
        "--arm-id",
        choices=[arm.arm_id for arm in EXPERIMENT_ARMS],
        required=True,
        help="Arm identifier to run.",
    )
    run_one.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Paired training seed to run.",
    )
    run_one.add_argument(
        "--evaluate-dev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate development recovery suite at each checkpoint boundary (default: enabled; use --no-evaluate-dev to disable).",
    )
    run_one.add_argument(
        "--evaluate-held-out",
        action="store_true",
        help="Also evaluate final checkpoint on held-out suite (reserve for release decision).",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Aggregate completed arm/seed runs and evaluate promotion guardrails.",
    )
    compare.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Prepared experiment root directory.",
    )
    compare.add_argument(
        "--suite",
        choices=["development", "held_out"],
        default="development",
        help="Evaluation suite to aggregate.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the requested CLI subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        seeds = tuple(int(s) for s in args.seeds)
        budget = build_transition_budget(
            policy_transitions_per_arm_seed=int(args.transitions_per_seed),
            checkpoint_count=int(args.checkpoints),
            rollout_steps=int(args.rollout_steps),
            num_seeds=len(seeds),
        )
        manifest = prepare_experiment_workspace(
            args.experiment_root,
            parent_checkpoint_path=args.parent_checkpoint,
            paired_seeds=seeds,
            budget=budget,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "experiment_root": manifest["experiment_root"],
                    "parent_checkpoint_sha256": manifest["parent_checkpoint"][
                        "checkpoint_sha256"
                    ],
                    "paired_training_seeds": manifest["paired_training_seeds"],
                    "budget": manifest["pilot_proposal"]["budget"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "verify":
        report = verify_loaded_arm_models(args.experiment_root)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run-arm":
        result = run_controlled_arm_seed(
            args.experiment_root,
            arm_id=args.arm_id,
            train_seed=int(args.seed),
            evaluate_development=bool(args.evaluate_dev),
            evaluate_held_out=bool(args.evaluate_held_out),
        )
        print(
            json.dumps(
                {
                    "arm_id": result["arm_id"],
                    "train_seed": result["train_seed"],
                    "completed": result["completed"],
                    "trained_policy_transitions": result["trained_policy_transitions"],
                    "trained_world_steps": result["trained_world_steps"],
                    "live_parameter_verification_passed": result[
                        "live_parameter_verification_passed"
                    ],
                    "final_checkpoint_path": result["final_checkpoint_path"],
                    "final_checkpoint_sha256": result["final_checkpoint_sha256"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "compare":
        root = Path(args.experiment_root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected_seeds = tuple(int(s) for s in manifest["paired_training_seeds"])
        expected_transitions = int(
            manifest.get("budget", {}).get("policy_transitions_per_arm_seed")
            or manifest["pilot_proposal"]["budget"]["policy_transitions_per_arm_seed"]
        )
        parent_sha_before = str(manifest["parent_checkpoint"]["checkpoint_sha256"])
        frozen_parent = Path(manifest["frozen_parent_copy_path"])
        parent_sha_after = sha256_file(frozen_parent)

        arm_seed_results: dict[str, dict[int, dict]] = {}
        for arm_id, arm_info in manifest["arms"].items():
            arm_seed_results[arm_id] = {}
            for seed_str, seed_info in arm_info["seeds"].items():
                result_file = Path(seed_info["output_dir"]) / "arm_seed_result.json"
                if result_file.exists():
                    arm_seed_results[arm_id][int(seed_str)] = json.loads(
                        result_file.read_text(encoding="utf-8")
                    )

        comparison = evaluate_experiment_promotion_guardrails(
            parent_sha256_before=parent_sha_before,
            parent_sha256_after=parent_sha_after,
            expected_seeds=expected_seeds,
            expected_transitions_per_arm_seed=expected_transitions,
            arm_seed_results=arm_seed_results,
            suite=args.suite,
            decision_mode="deterministic",
        )
        out_path = root / f"comparison_{args.suite}.json"
        out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(json.dumps(comparison, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
