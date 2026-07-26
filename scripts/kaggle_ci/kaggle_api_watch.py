"""Robust, API-only Kaggle kernel watcher -- the reliability backbone of the CI loop.

Polls `kaggle kernels status` until the run reaches a terminal state (or a
wall-clock budget is exceeded), then pulls the run's output via
`kaggle kernels output` and verifies:

  1. a schema-valid submission.json was actually produced, and
  2. the log contains no fatal error markers (a Traceback, CUDA OOM, a
     Kaggle-reported time-limit kill, etc).

This only talks to the officially supported `kaggle` CLI -- no Chrome, no
undocumented endpoints -- so its verdict is authoritative even if the
DrissionPage live-tail in kaggle_live_monitor.py silently stops working.

Usage:
    python kaggle_api_watch.py --push --max-seconds 32400 --report run_report.json

Exit codes: 0 success, 1 fatal error in the run, 2 timeout, 3 could not
determine (inspect the report/log by hand on kaggle.com).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from common import (
    COMPETITION_SLUG,
    KAGGLE_USERNAME,
    KERNEL_SLUG,
    RunVerdict,
    classify_line,
    kernel_ref,
    run_kaggle,
    validate_submission_shape,
)

ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STATUSES = {"complete", "error", "cancelled", "cancelacknowledged"}


def push(kernel_dir: Path) -> bool:
    print(f"[watch] pushing kernel from {kernel_dir} ...")
    result = run_kaggle(["kernels", "push", "-p", str(kernel_dir)])
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"[watch] push failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def poll_status(ref: str, poll_seconds: int, max_seconds: int) -> tuple[str, float]:
    """Poll until a terminal status or the time budget runs out."""
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            return "timeout", elapsed

        result = run_kaggle(["kernels", "status", ref])
        if result.returncode != 0:
            print(f"[watch] status check failed: {result.stderr.strip()}", file=sys.stderr)
        else:
            text = result.stdout.strip()
            print(f"[watch] +{int(elapsed)}s: {text}")
            lowered = text.lower()
            for candidate in TERMINAL_STATUSES:
                if candidate in lowered:
                    return candidate, elapsed
        time.sleep(poll_seconds)


def _iter_log_text_lines_from_text(raw: str) -> Iterator[str]:
    """Yield printed-text lines from kernel log JSON (from `kaggle kernels logs`).

    The log is a JSON array of {"stream_name", "time", "data"} entries (one
    stdout/stderr write per entry), not plain text -- confirmed against a
    real run's output. Parse that structure and yield each entry's text so
    pattern matching (which anchors on line starts like "Wrote submission:")
    actually sees the printed text instead of the JSON envelope around it.
    Falls back to raw line-splitting if it isn't in that shape.
    """
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = None

    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("data"), str):
                yield from entry["data"].splitlines()
        return

    yield from raw.splitlines()


def pull_and_classify(ref: str, out_dir: Path) -> RunVerdict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # `kernels logs` streams just the text log (works even on huge model-output
    # kernels and doesn't need to transfer gigabytes of committed checkpoint
    # files the way `kernels output` does -- that download failed twice with
    # IncompleteRead on a ~1.5GB transfer). `kernels output` is still used
    # below, but filtered to just submission.json via --file-pattern.
    logs_result = run_kaggle(["kernels", "logs", ref])
    if logs_result.returncode != 0:
        print(f"[watch] kernels logs failed: {logs_result.stderr.strip()}", file=sys.stderr)

    result = run_kaggle(["kernels", "output", ref, "-p", str(out_dir), "-o", "--file-pattern", r"submission\.json"])
    if result.returncode != 0:
        print(f"[watch] kernels output failed: {result.stderr.strip()}", file=sys.stderr)

    fatal_lines: list[str] = []
    warning_lines: list[str] = []
    success_markers: list[str] = []

    log_sources: list[Iterator[str]] = []
    scanned_log_names: list[str] = []
    if logs_result.returncode == 0 and logs_result.stdout.strip():
        log_sources.append(_iter_log_text_lines_from_text(logs_result.stdout))
        scanned_log_names.append("kernels logs (stdout)")
    for log_path in sorted(out_dir.glob("*.log")) + sorted(out_dir.glob("*.txt")):
        log_sources.append(_iter_log_text_lines_from_text(log_path.read_text(encoding="utf-8", errors="replace")))
        scanned_log_names.append(log_path.name)

    for lines in log_sources:
        for text_line in lines:
            kind = classify_line(text_line)
            if kind == "fatal":
                fatal_lines.append(text_line.strip())
            elif kind == "warning":
                warning_lines.append(text_line.strip())
            elif kind == "success":
                success_markers.append(text_line.strip())

    submission_candidates = sorted(out_dir.glob("submission.json")) + sorted(out_dir.glob("*.json"))
    submission_ok = False
    submission_detail = "no submission.json found in kernel output"
    for candidate in submission_candidates:
        ok, detail = validate_submission_shape(candidate)
        submission_ok, submission_detail = ok, detail
        if ok:
            break

    if fatal_lines:
        status = "fatal_error"
    elif submission_ok:
        status = "success"
    else:
        status = "unknown"

    return RunVerdict(
        status=status,
        fatal_lines=fatal_lines,
        warning_lines=warning_lines,
        success_markers_seen=success_markers,
        submission_check=submission_detail,
        detail=f"log sources scanned: {scanned_log_names}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", default=str(ROOT), help="Folder containing kernel-metadata.json")
    parser.add_argument("--username", default=KAGGLE_USERNAME)
    parser.add_argument("--slug", default=KERNEL_SLUG)
    parser.add_argument("--push", action="store_true", help="Push the kernel before watching")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=32400, help="Wall-clock budget (default 9h, a common Kaggle GPU session cap)")
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "kaggle_ci_output"))
    parser.add_argument("--report", default=str(ROOT / "runs" / "kaggle_ci_report.json"))
    args = parser.parse_args(argv)

    ref = kernel_ref(args.username, args.slug)
    report_path = Path(args.report)

    if args.push and not push(Path(args.kernel_dir)):
        RunVerdict(status="unknown", detail="push failed").write(report_path)
        return 3

    status, elapsed = poll_status(ref, args.poll_seconds, args.max_seconds)

    if status == "timeout":
        verdict = RunVerdict(
            status="timeout",
            elapsed_seconds=elapsed,
            detail=(
                f"no terminal status after {elapsed:.0f}s (budget {args.max_seconds}s). "
                f"Check https://www.kaggle.com/code/{ref} manually -- it may still be running, "
                "or Kaggle's own session limit may have killed it without updating status yet."
            ),
        )
        verdict.write(report_path)
        print(f"[watch] TIMEOUT: {verdict.detail}")
        return 2

    verdict = pull_and_classify(ref, Path(args.out_dir))
    verdict.elapsed_seconds = elapsed
    verdict.write(report_path)

    if status == "error":
        verdict.status = "fatal_error"
        verdict.detail = f"kaggle reported kernel status=error. {verdict.detail}"

    print(f"[watch] kaggle status = {status}")
    print(f"[watch] verdict = {verdict.status}")
    print(f"[watch] submission check: {verdict.submission_check}")
    if verdict.fatal_lines:
        print("[watch] fatal lines found in log:")
        for line in verdict.fatal_lines[:20]:
            print(f"  {line}")
    if verdict.warning_lines:
        print(f"[watch] {len(verdict.warning_lines)} non-fatal per-task fallback warning(s) (this is expected/OK)")
    print(f"[watch] full report written to {report_path}")

    if verdict.status == "success":
        return 0
    if verdict.status == "fatal_error":
        return 1
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
