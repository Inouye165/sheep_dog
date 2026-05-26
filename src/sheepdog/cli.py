"""Command-line entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from sheepdog.config import LabConfig
from sheepdog.curriculum import apply_training_profile
from sheepdog.environment import SheepdogEnvironment
from sheepdog.evaluation.benchmark import BenchmarkHarness
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.factory import POLICY_CHOICES, create_policy_from_name
from sheepdog.policies.heuristic import HeuristicExpertPolicy
from sheepdog.policies.random_policy import RandomPolicy
from sheepdog.server import _load_playable_policy
from sheepdog.training.factory import create_trainer

POLICY_CHOICES = list(POLICY_CHOICES)
TRAINER_CHOICES = ["hill_climb", "maskable_ppo", "hierarchical_maskable_ppo"]
POLICY_TYPE_CHOICES = ["linear", "neural"]


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
    """Run the CLI train command: parse args and start training."""
    parser = argparse.ArgumentParser(description="Train sheepdog checkpoint policy.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Override the artifact root.")
    parser.add_argument("--trainer-type", choices=TRAINER_CHOICES, default=None)
    parser.add_argument("--policy-type", choices=POLICY_TYPE_CHOICES, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    config = _apply_training_overrides(args, config)
    if args.output_dir:
        config = LabConfig(
            environment=config.environment,
            rewards=config.rewards,
            training=replace_training_output(config, args.output_dir),
        )
    create_trainer(config, config.training.output_dir).train()


def evaluate_command() -> None:
    """Run the CLI evaluate command: run evaluation on a checkpoint policy."""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint policy.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="instinct_only")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    seeds = tuple(args.seeds or config.training.evaluation_seeds)
    policy = create_policy_from_name(args.policy)
    evaluator = Evaluator(config, Path(config.training.output_dir) / "evaluations")
    evaluator.evaluate(policy, seeds, checkpoint_episode=0)


def export_demo_command() -> None:
    """Run the CLI export-demo command: write a replay JSON for the web UI."""
    parser = argparse.ArgumentParser(description="Export a playable demo replay for the UI.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--seed", type=int, default=11, help="Replay seed.")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="instinct_only")
    parser.add_argument("--output", default=None, help="Replay output file path.")
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    policy = create_policy_from_name(args.policy)
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


def benchmark_command() -> None:
    """Run the CLI benchmark command: compare policy variants on fixed seeds."""
    parser = argparse.ArgumentParser(description="Benchmark baseline and PPO policy variants.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Benchmark artifact directory.")
    parser.add_argument(
        "--linear-output-dir",
        default=None,
        help="Artifact root for linear checkpoints.",
    )
    parser.add_argument(
        "--neural-output-dir",
        default=None,
        help="Artifact root for neural checkpoints.",
    )
    parser.add_argument("--linear-checkpoint", type=int, default=None)
    parser.add_argument("--neural-checkpoint", type=int, default=None)
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    seeds = tuple(args.seeds or config.training.evaluation_seeds)
    benchmark_root = Path(args.output_dir or Path(config.training.output_dir) / "benchmarks")
    harness = BenchmarkHarness(config, benchmark_root)

    linear_config = config
    if args.linear_output_dir:
        linear_config = replace(
            config,
            training=replace(config.training, output_dir=args.linear_output_dir),
        )
    neural_config = config
    if args.neural_output_dir:
        neural_config = replace(
            config,
            training=replace(config.training, output_dir=args.neural_output_dir),
        )

    entries = [
        ("random_baseline", RandomPolicy(seed=0)),
        ("instinct_baseline", create_policy_from_name("instinct_only")),
        ("heuristic_expert", HeuristicExpertPolicy()),
        (
            "hill_climb_linear",
            _load_playable_policy(
                linear_config,
                checkpoint_episode=args.linear_checkpoint,
                policy_mode="trained_policy",
            ),
        ),
        (
            "maskable_ppo_neural",
            _load_playable_policy(
                neural_config,
                checkpoint_episode=args.neural_checkpoint,
                policy_mode="neural_policy",
            ),
        ),
    ]
    _results, json_path, csv_path, summary_path = harness.compare(entries, seeds)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


def _apply_training_overrides(args: argparse.Namespace, config: LabConfig) -> LabConfig:
    """Apply any trainer/policy/timestep overrides from CLI args to config."""
    training = config.training
    if getattr(args, "trainer_type", None):
        training = replace(training, trainer_type=args.trainer_type)
    if getattr(args, "policy_type", None):
        training = replace(training, policy_type=args.policy_type)
    if getattr(args, "total_timesteps", None) is not None:
        training = replace(training, total_timesteps=int(args.total_timesteps))
    if training is config.training:
        return config
    return replace(config, training=training)


def replace_training_output(config: LabConfig, output_dir: str):
    """Return a copy of config with the training output_dir replaced."""
    return replace(
        config.training,
        output_dir=output_dir,
    )


def train_hierarchical_command() -> None:
    """Train the hierarchical shepherd + neural dog policy."""
    parser = argparse.ArgumentParser(
        description="Train the hierarchical shepherd + neural dog policy."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--output-dir", default=None, help="Override the artifact root.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    config = _apply_training_overrides(args, config)
    # Force hierarchical trainer and neural policy type.
    config = replace(
        config,
        training=replace(
            config.training,
            trainer_type="hierarchical_maskable_ppo",
            policy_type="neural",
        ),
    )
    output_dir = args.output_dir or config.training.output_dir
    if args.output_dir:
        config = replace(
            config,
            training=replace(config.training, output_dir=output_dir),
        )
    # pylint: disable-next=import-outside-toplevel
    from sheepdog.training.hierarchical_trainer import HierarchicalMaskablePPOTrainer

    trainer = HierarchicalMaskablePPOTrainer(config, output_dir)
    result = trainer.train()
    print(
        json.dumps({"status": "done", "final_model_path": result.get("final_model_path")}, indent=2)
    )


def herding_eval_command() -> None:
    """Generate a proof-of-learning evaluation report comparing all policy types."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare random, scripted baseline, and hierarchical neural dog policies. "
            "Outputs JSON + Markdown to reports/."
        )
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--hierarchical-model",
        default=None,
        help="Path to a trained ShepherdNeuralDogPolicy .zip checkpoint.",
    )
    parser.add_argument(
        "--hierarchical-checkpoint",
        type=int,
        default=None,
        help="Checkpoint episode label for the hierarchical model (default 0).",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=int,
        default=None,
        help="Baseline checkpoint episode to include (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write herding_eval_latest.{json,md} (default: reports/).",
    )
    _add_profile_args(parser)
    args = parser.parse_args()
    config = _apply_runtime_profile(args, _load_config(args.config))
    seeds = tuple(args.seeds or config.training.evaluation_seeds)
    # pylint: disable-next=import-outside-toplevel
    from sheepdog.evaluation.benchmark import run_herding_eval_report

    json_path, md_path = run_herding_eval_report(
        config,
        args.output_dir,
        seeds=seeds,
        hierarchical_model_path=args.hierarchical_model,
        hierarchical_checkpoint_episode=args.hierarchical_checkpoint,
        baseline_checkpoint_episode=args.baseline_checkpoint,
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


def main() -> None:
    """Entry point: dispatch to the appropriate sub-command based on argv."""
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
        if command in {"train-hierarchical", "train_hierarchical"}:
            train_hierarchical_command()
            return
        if command == "evaluate":
            evaluate_command()
            return
        if command in {"herding-eval", "herding_eval"}:
            herding_eval_command()
            return
        if command == "benchmark":
            benchmark_command()
            return
        if command in {"export-demo", "export_demo"}:
            export_demo_command()
            return
        raise SystemExit(f"Unknown command: {command}")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
