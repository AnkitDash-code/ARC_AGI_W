"""Confirm the latest competition submission was accepted and scored by Kaggle.

Polls `kaggle competitions submissions -c <slug> --csv` (the officially
supported way to read submission status/score) until the newest row leaves
the pending/running state, then reports its public score.

Usage:
    python check_submission_score.py --max-seconds 3600
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

from common import COMPETITION_SLUG, run_kaggle

PENDING_STATES = {"pending", "running", "submitted"}
FAILURE_STATES = {"error", "failed"}


def latest_submission_row(competition: str) -> dict | None:
    result = run_kaggle(["competitions", "submissions", competition, "--csv"])
    if result.returncode != 0:
        print(f"[score] submissions lookup failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if not rows:
        return None
    return rows[0]  # Kaggle lists submissions newest-first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--report", default=str(Path(__file__).resolve().parents[2] / "runs" / "kaggle_score_report.json"))
    args = parser.parse_args(argv)

    started = time.monotonic()
    row = None
    while True:
        elapsed = time.monotonic() - started
        row = latest_submission_row(args.competition)
        if row is None:
            print("[score] no submissions found yet for this competition.")
        else:
            status = str(row.get("status", "")).strip().lower()
            print(f"[score] +{int(elapsed)}s: status={status!r} row={row}")
            if status not in PENDING_STATES:
                break
        if elapsed > args.max_seconds:
            print(f"[score] gave up after {elapsed:.0f}s without a terminal status.", file=sys.stderr)
            return 2
        time.sleep(args.poll_seconds)

    import json

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    status = str(row.get("status", "")).strip().lower()
    score = row.get("publicScore") or row.get("public_score") or row.get("score")
    print(f"[score] final status = {status}")
    print(f"[score] public score = {score}")
    print(f"[score] full row written to {report_path}")

    if status in FAILURE_STATES or not score:
        print("[score] submission did not score successfully -- check kaggle.com for the error message.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
