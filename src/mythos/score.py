"""CLI for scoring a submission against solution JSON."""

from __future__ import annotations

import argparse
import json
import sys

from mythos.arc import ArcValidationError
from mythos.metrics import score_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an ARC submission JSON file.")
    parser.add_argument("--pred", required=True, help="Path to submission JSON.")
    parser.add_argument("--solutions", required=True, help="Path to solution JSON.")
    args = parser.parse_args(argv)

    try:
        result = score_files(args.pred, args.solutions)
    except ArcValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
