"""CLI for preparing, verifying, and running the Stage 34 gate-information ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sheepdog.gate_information_experiment import (
    DEFAULT_PARENT_CHECKPOINT,
    DEFAULT_PARENT_METADATA,
    MAX_WALL_SECONDS_PER_ARM_SEED,
    prepare_gate_experiment_workspace,
    run_gate_experiment_arm,
    verify_gate_experiment_workspace,
)

DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments/gate_info_prep_v1")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Stage 34 gate-information observation ablation experiment CLI."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="Verify parent checkpoint, create isolated arm-seed workspaces, and convert weights (54->58).",
    )
    p_prep.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p_prep.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    p_prep.add_argument("--parent-metadata", type=Path, default=DEFAULT_PARENT_METADATA)
    p_prep.add_argument("--overwrite", action="store_true")
    p_prep.add_argument("--json", action="store_true")

    p_ver = sub.add_parser(
        "verify",
        help="Verify prepared experiment workspace integrity, config parity, and 58-input models.",
    )
    p_ver.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p_ver.add_argument("--json", action="store_true")

    p_run = sub.add_parser(
        "run-arm",
        help="Run one isolated (arm, seed) pilot training job from a prepared workspace.",
    )
    p_run.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p_run.add_argument("--arm", required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument(
        "--max-wall-seconds",
        type=float,
        default=MAX_WALL_SECONDS_PER_ARM_SEED,
    )
    p_run.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate-information experiment CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    payload: dict[str, Any]
    if args.command == "prepare":
        payload = prepare_gate_experiment_workspace(
            args.experiment_root,
            parent_checkpoint=args.parent_checkpoint,
            parent_metadata=args.parent_metadata,
            overwrite=args.overwrite,
        )
    elif args.command == "verify":
        payload = verify_gate_experiment_workspace(args.experiment_root)
    elif args.command == "run-arm":
        payload = run_gate_experiment_arm(
            args.experiment_root,
            arm_id=args.arm,
            seed=args.seed,
            max_wall_seconds=args.max_wall_seconds,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        print(f"Completed '{args.command}' on {args.experiment_root}")
        if args.command == "prepare":
            eq = payload["initial_equivalence_check"]
            print(f"Parent SHA-256: {payload['parent_checkpoint']['sha256']}")
            print(f"Rollout bit-for-bit identical: {eq['rollout_bit_for_bit_identical']}")
            for arm_id, arm_eq in eq["arms"].items():
                print(
                    f"  {arm_id}: params={arm_eq['parameter_count']}, "
                    f"max_logit_diff={arm_eq['max_abs_logit_diff']:.3e}, "
                    f"max_prob_diff={arm_eq['max_abs_masked_prob_diff']:.3e}, "
                    f"max_val_diff={arm_eq['max_abs_value_diff']:.3e}"
                )
        elif args.command == "verify":
            print(
                f"Verified {payload['checked_arm_seeds']} arm-seed workspaces "
                f"(held_out={payload['held_out_evaluation_status']})"
            )
        elif args.command == "run-arm":
            print(
                f"Finished {payload['arm_id']} seed={payload['seed']} "
                f"({payload['stopped_reason']}, {payload['elapsed_seconds']:.1f}s)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
