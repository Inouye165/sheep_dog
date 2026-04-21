"""Command-line entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sheepdog.config import LabConfig
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


def train_command() -> None:
    parser = argparse.ArgumentParser(description="Train sheepdog checkpoint policy.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Override the artifact root.")
    args = parser.parse_args()
    config = _load_config(args.config)
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
    args = parser.parse_args()
    config = _load_config(args.config)
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
    args = parser.parse_args()
    config = _load_config(args.config)
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
    return config.training.__class__(
        episodes=config.training.episodes,
        checkpoint_episodes=config.training.checkpoint_episodes,
        evaluation_seeds=config.training.evaluation_seeds,
        train_seed=config.training.train_seed,
        evaluation_seed=config.training.evaluation_seed,
        mutation_scale=config.training.mutation_scale,
        output_dir=output_dir,
        web_export_dir=config.training.web_export_dir,
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
