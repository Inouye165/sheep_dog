#!/usr/bin/env python3
"""CLI script to generate a provenance and history report for a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if needed
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from sheepdog.tracking.registry import MigrationRequiredError, TrackingRegistry  # noqa: E402
from sheepdog.tracking.report import format_checkpoint_history_report  # noqa: E402


def main() -> int:
    """Run CLI to generate checkpoint provenance report or list tracked checkpoints."""
    parser = argparse.ArgumentParser(
        description="Generate a 5-question provenance and history report for a sheepdog checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint identifier or record ID (e.g. checkpoint-040400-phase4_1, checkpoint-040398)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="artifacts/training-telemetry.sqlite",
        help="Path to SQLite database (default: artifacts/training-telemetry.sqlite)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoints currently registered in the database",
    )
    args = parser.parse_args()

    try:
        registry = TrackingRegistry(args.db, read_only=True)
    except FileNotFoundError as err:
        print(f"\nError: {err}", file=sys.stderr)
        return 1
    except MigrationRequiredError as err:
        print(f"\nError: {err}", file=sys.stderr)
        return 1

    if args.list:
        with registry._get_connection() as conn:
            rows = conn.execute(
                "SELECT checkpoint_record_id, checkpoint_id, experiment_id, checkpoint_label_episode, created_at FROM registry_checkpoints ORDER BY global_environment_episode ASC, ROWID ASC;"
            ).fetchall()
            print(f"\nRegistered Checkpoints in {args.db} ({len(rows)} total):")
            print(f"{'Record ID':<35} {'Checkpoint ID':<30} {'Episode':<8} {'Experiment ID':<35}")
            print("-" * 110)
            for r in rows:
                ep_str = str(r["checkpoint_label_episode"]) if r["checkpoint_label_episode"] is not None else "-"
                print(f"{r['checkpoint_record_id']:<35} {r['checkpoint_id']:<30} {ep_str:<8} {r['experiment_id'] or '-':<35}")
            print("-" * 110 + "\n")
        return 0

    if not args.checkpoint:
        parser.print_help()
        print("\nError: Please specify --checkpoint <id> or use --list.")
        return 1

    lineage = registry.get_history_for_checkpoint(args.checkpoint)
    report_text = format_checkpoint_history_report(lineage)
    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
