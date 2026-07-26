"""Top-level Kaggle CI/CD orchestrator: push, watch, (optionally) submit, verify score.

This wires together the other scripts in this folder in the order you'd run
them by hand:

  1. kaggle_api_watch.py   -- push the kernel, poll status via the official
                               API until it finishes, verify a schema-valid
                               submission.json was produced and no fatal
                               errors are in the log. This is the gate.
  2. kaggle_submit_click.py -- (only with --auto-submit) click "Submit to
                               Competition" via DrissionPage, since Kaggle
                               has no API for this action on Code
                               Competitions.
  3. check_submission_score.py -- poll the official submissions API until
                               Kaggle reports a public score.

Stops and prints a clear failure summary at the first step that doesn't
pass rather than plowing ahead -- this is meant to hand you (or the agent
loop) a specific, actionable error, not to silently retry or auto-patch
code.

Usage:
    python run_ci.py --push --max-seconds 32400
    python run_ci.py --push --max-seconds 32400 --auto-submit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import kaggle_api_watch
import kaggle_submit_click
import check_submission_score
from common import COMPETITION_SLUG, KAGGLE_USERNAME, KERNEL_SLUG

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=KAGGLE_USERNAME)
    parser.add_argument("--slug", default=KERNEL_SLUG)
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--push", action="store_true", help="Push the notebook before watching")
    parser.add_argument("--max-seconds", type=int, default=32400, help="Run-watch budget (default 9h)")
    parser.add_argument("--auto-submit", action="store_true", help="Auto-click Submit to Competition after a successful run (requires Chrome debug port + DrissionPage)")
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--score-max-seconds", type=int, default=3600)
    args = parser.parse_args(argv)

    print("=" * 70)
    print("STEP 1/3: watch the Kaggle run via the official API")
    print("=" * 70)
    watch_code = kaggle_api_watch.main(
        [
            "--username", args.username,
            "--slug", args.slug,
            *(["--push"] if args.push else []),
            "--max-seconds", str(args.max_seconds),
        ]
    )
    if watch_code != 0:
        print("\nRUN FAILED OR INCONCLUSIVE. See runs/kaggle_ci_report.json for the fatal log lines.")
        print(f"Inspect manually at: https://www.kaggle.com/code/{args.username}/{args.slug}")
        return watch_code

    if not args.auto_submit:
        print("\nRun succeeded and produced a valid submission.json.")
        print(f"--auto-submit was not set; finish the submission yourself at "
              f"https://www.kaggle.com/code/{args.username}/{args.slug}, "
              "then re-run check_submission_score.py to confirm scoring.")
        return 0

    print("\n" + "=" * 70)
    print("STEP 2/3: click Submit to Competition")
    print("=" * 70)
    submit_code = kaggle_submit_click.main(
        ["--username", args.username, "--slug", args.slug, "--debug-port", str(args.debug_port)]
    )
    if submit_code != 0:
        print("\nCould not confirm the submit click. Check the saved screenshots in "
              "runs/kaggle_ci_screenshots/ and finish the submission manually.")
        return submit_code

    print("\n" + "=" * 70)
    print("STEP 3/3: confirm Kaggle scored the submission")
    print("=" * 70)
    score_code = check_submission_score.main(
        ["--competition", args.competition, "--max-seconds", str(args.score_max_seconds)]
    )
    if score_code != 0:
        print("\nSubmission was sent but scoring did not confirm successfully. "
              f"Check https://www.kaggle.com/competitions/{args.competition}/submissions manually.")
    else:
        print("\nAll clear: run succeeded, submitted, and scored by Kaggle.")
    return score_code


if __name__ == "__main__":
    raise SystemExit(main())
