"""Command-line entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.heuristic import HeuristicPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.policies.trainable import PolicyWeights, TrainableLinearPolicy
from sheepdog.training.trainer import Trainer


def _load_config(path: str | None) -> LabConfig:
    if not path:
        return LabConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        return LabConfig.from_dict(json.load(handle))


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-instincts",
        action="store_true",
        help="Enable instinct reward shaping for this run.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=None,
        help="Apply a curriculum stage before training, evaluation, or demo export.",
    )
    parser.add_argument(
        "--debug-reward-breakdown",
        action="store_true",
        help="Emit debug-friendly replay metadata for reward and pressure diagnostics.",
    )


def _apply_runtime_profile(args: argparse.Namespace, config: LabConfig) -> LabConfig:
    try:
        return apply_training_profile(
            config,
            enable_instinct_rewards=True if args.enable_instincts else None,
            curriculum_stage=args.curriculum_stage,
            debug_reward_breakdown=True if args.debug_reward_breakdown else None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def train_command() -> None:
    parser = argparse.ArgumentParser(description="Train sheepdog checkpoint policy.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Override the artifact root.")
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    if args.output_dir:
        config = LabConfig(
            environment=config.environment,
            rewards=config.rewards,
            training=replace_training_output(config, args.output_dir),
        )
    Trainer(config, config.training.output_dir).train()


def evaluate_command() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint policy.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--policy", choices=["heuristic", "random", "trained"], default="heuristic")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    seeds = tuple(args.seeds or config.training.evaluation_seeds)
    policy = _policy_from_name(args.policy)
    evaluator = Evaluator(config, Path(config.training.output_dir) / "evaluations")
    evaluator.evaluate(policy, seeds, checkpoint_episode=0)


def export_demo_command() -> None:
    parser = argparse.ArgumentParser(description="Export a playable demo replay for the UI.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--seed", type=int, default=11, help="Replay seed.")
    parser.add_argument("--policy", choices=["heuristic", "random", "trained"], default="heuristic")
    parser.add_argument("--output", default=None, help="Replay output file path.")
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    policy = _policy_from_name(args.policy)
    result = SheepdogEnvironment(config).run_policy(policy, seed=args.seed, capture_replay=True)
    output = Path(args.output or Path(config.training.web_export_dir) / "latest-replay.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": result.seed,
                "policy_name": result.policy_name,
                "final_snapshot": result.final_snapshot.to_dict(),
                "stats": asdict(result.stats),
                "frames": [frame.to_dict() for frame in result.replay],
            },
            handle,
            indent=2,
        )


def _policy_from_name(policy_name: str):
    if policy_name == "heuristic":
        return HeuristicPolicy()
    if policy_name == "random":
        return RandomPolicy()
    return TrainableLinearPolicy(PolicyWeights())


def replace_training_output(config: LabConfig, output_dir: str):
    return replace(
        config.training,
        output_dir=output_dir,
    )


def main() -> None:
    if len(sys.argv) <= 1:
        train_command()
        return

    command = sys.argv[1]
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        if command == "train":
            train_command()
            return
        if command == "evaluate":
            evaluate_command()
            return
        if command in {"export-demo", "export_demo"}:
            export_demo_command()
            return
        raise SystemExit(f"Unknown command: {command}")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
