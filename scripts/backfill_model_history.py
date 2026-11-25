#!/usr/bin/env python3
"""CLI script to backfill model-pass and per-stage history into SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from sheepdog.tracking.model_history import (  # noqa: E402
    MODEL_HISTORY_DB_FILENAME,
    ModelHistoryStore,
)


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}s"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def print_report(store: ModelHistoryStore) -> None:
    """Print pass summary and per-stage breakdown from the model history store."""
    passes = store.list_passes(limit=100)
    if not passes:
        print("No model passes found in model-history.sqlite.")
        return

    print(f"\n=== Model History Report ({len(passes)} pass(es)) ===")
    for p in passes:
        print(
            f"\nPass: {p['pass_id']} | Label: {p.get('label') or '-'} | "
            f"Type: {p['pass_type']} | Stages: {p['stage_spec']} | Status: {p['status']}"
        )
        print(
            f"  Run ID: {p.get('run_id') or '-'} | Parent: {p.get('parent_pass_id') or '-'} | "
            f"Assistance: {p.get('assistance_mode') or '-'} | Source: {p['source']}"
        )
        print(
            f"  Completed: {p['completed_stages_count']}/{len(p['planned_stages'])} | "
            f"Active: {_fmt_seconds(p['total_active_seconds'])} | "
            f"Train: {_fmt_seconds(p['total_training_seconds'])} | "
            f"Eval: {_fmt_seconds(p['total_evaluation_seconds'])} | "
            f"Wall: {_fmt_seconds(p.get('total_wall_seconds'))} | "
            f"Timesteps: {_fmt_int(p['total_policy_timesteps'])}"
        )
        print(
            f"  Resulting Checkpoint: {p.get('resulting_checkpoint_id') or '-'} "
            f"(sha256: {(p.get('resulting_checkpoint_sha256') or '-')[:12]})"
        )

        stage_runs = p.get("stage_runs") or []
        if not stage_runs:
            continue
        header = (
            f"  {'Seq':>3}  {'Stg':>3}  {'Status':<9}  {'Train':>8}  {'Eval':>7}  "
            f"{'Active':>8}  {'Wall':>8}  {'Timesteps':>11}  {'SPS':>7}  {'FinalSR':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sr in stage_runs:
            sps = f"{sr['steps_per_second']:.1f}" if sr.get("steps_per_second") is not None else "-"
            print(
                f"  {sr['sequence_order']:>3}  {sr['stage']:>3}  {sr['status']:<9}  "
                f"{_fmt_seconds(sr.get('training_seconds')):>8}  "
                f"{_fmt_seconds(sr.get('evaluation_seconds')):>7}  "
                f"{_fmt_seconds(sr.get('active_seconds')):>8}  "
                f"{_fmt_seconds(sr.get('wall_seconds')):>8}  "
                f"{_fmt_int(sr.get('policy_timesteps')):>11}  "
                f"{sps:>7}  "
                f"{_fmt_rate(sr.get('final_success_rate')):>7}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run CLI to backfill artifact runs into model-history.sqlite and optionally report."""
    parser = argparse.ArgumentParser(
        description="Backfill Sheepdog model-lineage and per-stage history into SQLite."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Artifacts root directory (default: artifacts)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print pass summary and per-stage table from model-history.sqlite",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    db_path = output_dir / MODEL_HISTORY_DB_FILENAME
    store = ModelHistoryStore(db_path)
    summary = store.backfill_from_artifacts(output_dir)
    print(
        f"Backfill complete ({db_path}): "
        f"imported_passes={summary['imported_passes']}, "
        f"skipped_passes={summary['skipped_passes']}, "
        f"imported_stage_runs={summary['imported_stage_runs']}"
    )
    if args.report:
        print_report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
