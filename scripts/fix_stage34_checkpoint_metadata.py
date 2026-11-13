"""Correct Stage 34 checkpoint metadata in training-summary, checkpoint-index, and training-state."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data to file via temporary file replacement."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def fix_training_summary(path: Path) -> int:
    """Update curriculum_stage to 34 for checkpoints >= 40399 in training-summary.json."""
    if not path.exists():
        print(f"{path} not found.")
        return 0

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    checkpoints = data.get("checkpoints", [])
    for chk in checkpoints:
        ep = chk.get("checkpoint_episode", 0)
        if ep >= 40399 and chk.get("curriculum_stage") != 34:
            chk["curriculum_stage"] = 34
            updated_count += 1

    atomic_write_json(path, data)
    print(f"Updated {updated_count} checkpoints in {path}")
    return updated_count


def fix_checkpoint_index(path: Path) -> int:
    """Update curriculum_stage to 34 for checkpoints >= 40399 and latest in checkpoint-index.json."""
    if not path.exists():
        print(f"{path} not found.")
        return 0

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    checkpoints = data.get("checkpoints", [])
    for chk in checkpoints:
        ep = chk.get("checkpoint_episode", 0)
        if ep >= 40399 and chk.get("curriculum_stage") != 34:
            chk["curriculum_stage"] = 34
            updated_count += 1

    latest = data.get("latest")
    if isinstance(latest, dict):
        ep = latest.get("checkpoint_episode", 0)
        if ep >= 40399 and latest.get("curriculum_stage") != 34:
            latest["curriculum_stage"] = 34
            print(f"Updated latest object in {path} to curriculum_stage 34")

    atomic_write_json(path, data)
    print(f"Updated {updated_count} checkpoints in {path}")
    return updated_count


def fix_evaluations(eval_dir: Path) -> int:
    """Update curriculum_stage to 34 in recent evaluation json files."""
    if not eval_dir.exists():
        return 0

    pattern = str(eval_dir / "*ts_337*.json")
    updated_files = 0
    for file_path_str in glob.glob(pattern):
        p = Path(file_path_str)
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            ep = data.get("checkpoint_episode", 0)
            if ep >= 40399 and data.get("curriculum_stage") != 34:
                data["curriculum_stage"] = 34
                atomic_write_json(p, data)
                updated_files += 1
        except Exception as exc:
            print(f"Warning: Failed to update {p}: {exc}")

    print(f"Updated {updated_files} evaluation json files in {eval_dir}")
    return updated_files


def fix_training_state(path: Path) -> None:
    """Ensure training-state.json points to latest checkpoint 40449 and curriculum_stage 34."""
    if not path.exists():
        print(f"{path} not found.")
        return

    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    latest_model = "artifacts/models/maskable-ppo-337687972.zip"
    state["total_episodes_trained"] = 40449
    state["total_timesteps"] = 337687972
    state["policy_state_path"] = latest_model
    state["best_model_curriculum_stage"] = 34

    if "adaptive_step_state" in state and isinstance(state["adaptive_step_state"], dict):
        state["adaptive_step_state"]["curriculum_stage"] = 34

    atomic_write_json(path, state)
    print(f"Updated {path} pointing to {latest_model} (episode 40449, ts 337687972, stage 34).")


def main() -> None:
    """Execute metadata fix for Stage 34 extended checkpoints."""
    print("=" * 80)
    print("FIXING STAGE 34 EXTENDED CHECKPOINT METADATA")
    print("=" * 80)

    summary_path = Path("artifacts/training-summary.json")
    fix_training_summary(summary_path)

    web_index_path = Path("web/public/generated/checkpoint-index.json")
    fix_checkpoint_index(web_index_path)

    art_index_path = Path("artifacts/checkpoint-index.json")
    if art_index_path.exists():
        fix_checkpoint_index(art_index_path)

    eval_dir = Path("artifacts/evaluations")
    fix_evaluations(eval_dir)

    state_path = Path("artifacts/training-state.json")
    fix_training_state(state_path)

    print("=" * 80)
    print("METADATA CORRECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
