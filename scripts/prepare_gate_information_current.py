"""Prepare/verify the explicit current-parent gate-information experiment; never train."""
import argparse
import json
from pathlib import Path

from sheepdog.gate_information_preparation import prepare, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if not all((args.parent, args.metadata, args.sha256)):
            parser.error("prepare requires --parent, --metadata and --sha256")
        result = prepare(args.root, args.parent, args.metadata, args.sha256, args.baseline)
        print(json.dumps({"status": result["status"], "workspaces": len(result["entries"])}))
    else:
        print(json.dumps(verify(args.root)))
