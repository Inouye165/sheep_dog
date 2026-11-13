#!/usr/bin/env python3
"""CLI script to import historical November 8-9 trials into the tracking registry."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path if needed
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from sheepdog.tracking.importer import (  # noqa: E402
    import_historical_trials,
    repair_historical_registry,
)
from sheepdog.tracking.schema import verify_migration_safety  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_historical_trials")


def main() -> int:
    """Run CLI to safely migrate and import historical trials into SQLite."""
    parser = argparse.ArgumentParser(description="Import historical trials into Sheepdog tracking registry.")
    parser.add_argument(
        "--db",
        type=str,
        default="artifacts/training-telemetry.sqlite",
        help="Path to SQLite database (default: artifacts/training-telemetry.sqlite)",
    )
    parser.add_argument(
        "--skip-safety-check",
        action="store_true",
        help="Skip isolated migration safety verification before running import",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Run idempotent repair migration on existing historical rows without full reimport",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.repair:
        if not db_path.exists():
            logger.error("Database does not exist: %s", db_path)
            return 1
        logger.info("Running idempotent repair on historical records in %s...", db_path)
        try:
            repair_counts = repair_historical_registry(db_path)
            print("\n=== Historical Registry Repair Succeeded ===")
            for table, count in repair_counts.items():
                print(f"- {table:<25}: {count}")
            print("============================================\n")
            return 0
        except Exception as exc:
            logger.exception("Historical registry repair failed: %s", exc)
            return 1

    db_path = Path(args.db)
    if not args.skip_safety_check and db_path.exists():
        logger.info("Running pre-migration safety check on isolated copy of %s...", db_path)
        try:
            verify_migration_safety(db_path)
            logger.info("Migration safety verification passed.")
        except Exception as exc:
            logger.error("Migration safety verification FAILED: %s", exc)
            return 1

    logger.info("Importing historical trials into %s...", db_path)
    try:
        counts = import_historical_trials(db_path, repo_root=repo_root)
        print("\n=== Historical Trials Import Succeeded ===")
        for entity, count in counts.items():
            print(f"- {entity.capitalize():<15}: {count}")
        print("==========================================\n")
        return 0
    except Exception as exc:
        logger.exception("Historical trials import failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
