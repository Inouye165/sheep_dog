"""Evaluate guided TeamStrategy variants on development seeds (ASSISTED/guided; no training).

Evaluation only: existing checkpoints see TeamStrategy-shifted guided observations.
Usage: python -m scripts.eval_team_strategy_variant --out <dir> --model-name pv_10486 \
    --model <zip> --variant all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from sheepdog.atomic_io import atomic_write_json
from sheepdog.config import LabConfig
from sheepdog.evaluation.evaluator import Evaluator
from sheepdog.policies.neural import NeuralPolicy, NeuralPolicyConfig
from sheepdog.team_strategy import TeamStrategy
from sheepdog.training.replay_writer import CapturePolicy, set_global_capture_policy

DEV_SEEDS = (11, 23, 37, 41, 53, 59, 61, 67, 71, 73)
VARIANTS = {
    "baseline": (False, False, False),
    "main_group": (True, False, False),
    "collector_yield": (False, True, False),
    "corner_clearance": (False, False, True),
    "all": (True, True, True),
}


def load_config(path: Path, root: Path) -> LabConfig:
    """Load a frozen config, redirecting only output paths into the experiment dir."""
    data = json.loads(path.read_text())
    data["training"].update(output_dir=str(root / "training_output"),
                            backup_dir=str(root / "backups"), web_export_dir=str(root / "web"))
    return LabConfig.from_dict(data)


def main() -> None:
    """Run one model/variant development evaluation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    set_global_capture_policy(CapturePolicy(mode="off", capture_step_telemetry=False))
    main_group, collector_yield, corner = VARIANTS[args.variant]
    TeamStrategy.DEFAULT_MAIN_GROUP_TARGETS = main_group
    TeamStrategy.DEFAULT_COLLECTOR_YIELD = collector_yield
    TeamStrategy.DEFAULT_CORNER_CLEARANCE = corner
    root = args.out / f"{args.model_name}__{args.variant}"
    root.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config, root)
    model = MaskablePPO.load(str(args.model), device="cpu")
    size = int(model.observation_space.shape[0])
    policy = NeuralPolicy(model, args.model, NeuralPolicyConfig(
        hidden_sizes=(256, 256, 128), observation_size=size, action_size=9,
        observation_mode=cfg.training.observation_mode))
    start = time.monotonic()
    summary, _, _ = Evaluator(cfg, root).evaluate(
        policy, DEV_SEEDS, checkpoint_episode=0, deterministic=True,
        capture_replays=True, capture_step_telemetry=False,
        evaluation_mode="development_team_strategy_variant", curriculum_stage=34,
        run_id=root.name, checkpoint_id=args.model_name)
    data = summary.to_dict()
    data.update(variant=args.variant, model_name=args.model_name,
                team_strategy_version=TeamStrategy.VERSION, options=dict(
                    main_group_targets=main_group, collector_yield=collector_yield,
                    corner_clearance=corner),
                assistance="guided TeamStrategy targets in observations (assisted)",
                observation_size=size, elapsed_seconds=time.monotonic() - start)
    atomic_write_json(root / "result.json", data)
    print(json.dumps({"variant": args.variant, "model": args.model_name,
                      "success_rate": data.get("success_rate")}))


if __name__ == "__main__":
    main()
