"""Reproducible script to generate the durable temporal dog-sheep interaction diagnostic report.

Reads real evaluation replays from artifacts/evaluations/replays/, generates offline
temporal interaction analyses, constructs paired failure vs recovery comparisons,
explicitly identifies missing historical sources, and writes docs/temporal_interaction_analysis_report.md.

Usage:
    python scripts/generate_temporal_diagnostic_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path so script can be run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sheepdog.training.diagnostics.temporal_interaction import (
    analyze_episode_temporal_interactions,
    compare_failure_and_recovery_episodes,
    format_temporal_diagnostic_report,
)

EVAL_REPLAYS_DIR = Path("artifacts/evaluations/replays")
OUTPUT_REPORT_PATH = Path("docs/temporal_interaction_analysis_report.md")

TARGET_RECORDINGS = [
    "checkpoint-039909-seed-000061",
    "checkpoint-039909-seed-000041",
    "checkpoint-039911-seed-000041",
    "checkpoint-039903-seed-000053",
    "checkpoint-039897-seed-000023",
    "checkpoint-039909-seed-000073",
    "checkpoint-039911-seed-000073",
    "checkpoint-039911-seed-000071",
    "checkpoint-039910-seed-000071",
]

HISTORICAL_MISSING_RECORDINGS = [
    "diag_stage34_ep139_seed146_t1793983025",
]

PAIRED_COMPARISONS = [
    ("checkpoint-039909-seed-000041", "checkpoint-039911-seed-000041"),
    ("checkpoint-039909-seed-000073", "checkpoint-039911-seed-000073"),
    ("checkpoint-039911-seed-000071", "checkpoint-039910-seed-000071"),
]


def resolve_replay_path(name: str) -> Path | None:
    """Find a replay on disk matching the given stem name."""
    for ext in (".json.gz", ".json"):
        cand = EVAL_REPLAYS_DIR / f"{name}{ext}"
        if cand.exists():
            return cand
        cand_alt = Path("artifacts/replays") / f"{name}{ext}"
        if cand_alt.exists():
            return cand_alt
    return None


def main() -> None:
    """Run temporal diagnostic analysis on real evaluation replays and write markdown report."""
    analyses = []
    analyses_by_id = {}
    missing_sources = []

    for name in TARGET_RECORDINGS:
        path = resolve_replay_path(name)
        if path is None:
            print(f"[WARN] Replay '{name}' not found on disk. Adding to missing sources.")
            missing_sources.append(name)
            continue
        print(f"Analyzing {name} ({path})...")
        analysis = analyze_episode_temporal_interactions(path)
        analyses.append(analysis)
        analyses_by_id[name] = analysis

    for name in HISTORICAL_MISSING_RECORDINGS:
        path = resolve_replay_path(name)
        if path is None:
            missing_sources.append(name)
        else:
            print(f"Analyzing historical {name} ({path})...")
            analysis = analyze_episode_temporal_interactions(path)
            analyses.append(analysis)
            analyses_by_id[name] = analysis

    comparisons = []
    for fail_name, rec_name in PAIRED_COMPARISONS:
        if fail_name in analyses_by_id and rec_name in analyses_by_id:
            comp = compare_failure_and_recovery_episodes(
                analyses_by_id[fail_name],
                analyses_by_id[rec_name],
            )
            comparisons.append(comp)

    report_content = format_temporal_diagnostic_report(
        analyses,
        comparisons=comparisons,
        unavailable_sources=missing_sources,
    )

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"Report generated successfully: {OUTPUT_REPORT_PATH} ({len(report_content)} characters)")


if __name__ == "__main__":
    main()
