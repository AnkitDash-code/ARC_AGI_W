"""CLI for ARC challenge validation."""

from __future__ import annotations

import argparse
import sys

from mythos.arc import ArcValidationError, load_challenges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an ARC challenges.json file.")
    parser.add_argument("path", help="Path to ARC-style challenges JSON.")
    args = parser.parse_args(argv)

    try:
        tasks = load_challenges(args.path)
    except ArcValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    train_count = sum(len(task.train) for task in tasks.values())
    test_count = sum(len(task.test) for task in tasks.values())
    print(f"OK: {len(tasks)} tasks, {train_count} train examples, {test_count} test items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
