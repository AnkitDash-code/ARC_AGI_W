"""Live-tail a running Kaggle kernel's log via Chrome DevTools Protocol (DrissionPage).

This is a best-effort VISIBILITY layer, not the reliability backbone --
kaggle_api_watch.py (official `kaggle` CLI only) is the authoritative
success/failure verdict. Use this script alongside it when you want to see
output streaming in near-real-time and abort early on a fatal error instead
of waiting out the full poll interval.

Why this is inherently fragile: it intercepts an internal, undocumented
Kaggle network endpoint via CDP. The endpoint name/response shape is not a
public contract and can change without notice. If interception stops
producing lines, this script will say so explicitly (rather than silently
hanging) and you should fall back to kaggle_api_watch.py or the Kaggle
website.

Prerequisites:
    pip install DrissionPage
    Launch Chrome with a debugging port and sign in to Kaggle in it first:
        chrome.exe --remote-debugging-port=9222

Usage:
    python kaggle_live_monitor.py --username ankitdash24 --slug project-mythos-pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from common import KAGGLE_USERNAME, KERNEL_SLUG, NONFATAL_PREFIX, classify_line

DEFAULT_LISTEN_TARGET = "kernels.KernelsService/GetKernelViewerLog"


def _extract_log_lines(raw_body: Any) -> list[str] | None:
    """Best-effort extraction of new log lines from an unknown/undocumented response shape.

    Tries the shapes we've seen or can reasonably expect from Kaggle's
    internal log-viewer API. Returns None if nothing recognizable was found
    so the caller can dump the raw body for inspection instead of guessing.
    """
    if isinstance(raw_body, str):
        try:
            raw_body = json.loads(raw_body)
        except json.JSONDecodeError:
            return None

    if isinstance(raw_body, dict):
        # Flat shape: {"log": ["line1", "line2", ...]}
        if isinstance(raw_body.get("log"), list):
            return [str(item) for item in raw_body["log"]]
        # Nested shape: {"log": {"log": [...]}}
        nested = raw_body.get("log")
        if isinstance(nested, dict) and isinstance(nested.get("log"), list):
            return [str(item) for item in nested["log"]]
        # Message-object shape: {"log": [{"message": "..."}]} or {"messages": [...]}
        for key in ("log", "messages", "entries"):
            value = raw_body.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                lines = []
                for entry in value:
                    text = entry.get("message") or entry.get("text") or entry.get("line")
                    if text is not None:
                        lines.append(str(text))
                if lines:
                    return lines
    return None


def monitor(
    username: str,
    slug: str,
    debug_port: int,
    listen_target: str,
    debug_dump: Path | None,
    log_out: Path | None,
) -> int:
    try:
        from DrissionPage import ChromiumOptions, ChromiumPage
    except ImportError:
        print("DrissionPage is not installed. Run: pip install DrissionPage", file=sys.stderr)
        return 3

    co = ChromiumOptions()
    co.set_local_port(debug_port)
    try:
        page = ChromiumPage(co)
    except Exception as exc:  # noqa: BLE001 - report connection failure clearly
        print(
            f"Could not connect to Chrome on port {debug_port}: {exc!r}\n"
            f"Ensure Chrome is running with --remote-debugging-port={debug_port} "
            "and you're already signed in to Kaggle in that browser.",
            file=sys.stderr,
        )
        return 3

    kernel_url = f"https://www.kaggle.com/code/{username}/{slug}/log"
    page.listen.start(listen_target)
    print(f"Navigating to {kernel_url}")
    page.get(kernel_url)
    print(f"Listening for packets matching '{listen_target}' ...")
    print("-" * 60)

    log_handle = log_out.open("a", encoding="utf-8") if log_out else None
    last_read_index = 0
    consecutive_unparsed = 0
    fatal_seen = False

    try:
        while True:
            packet = page.listen.wait(timeout=30)
            if not packet:
                print("[monitor] no packet in 30s (page idle or interception stalled)")
                continue
            if not packet.response:
                continue

            lines = _extract_log_lines(packet.response.body)
            if lines is None:
                consecutive_unparsed += 1
                if debug_dump:
                    debug_dump.parent.mkdir(parents=True, exist_ok=True)
                    with debug_dump.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"raw_body": packet.response.body}, default=str) + "\n")
                if consecutive_unparsed in (1, 5, 20):
                    print(
                        f"[monitor] WARNING: {consecutive_unparsed} packet(s) matched the listen target "
                        "but had an unrecognized shape. Response shape may have changed; "
                        f"raw bodies are being dumped to {debug_dump or '(pass --debug-dump to capture them)'}."
                    )
                continue
            consecutive_unparsed = 0

            if len(lines) <= last_read_index:
                continue
            new_lines = lines[last_read_index:]
            last_read_index = len(lines)

            for line in new_lines:
                sys.stdout.write(f"[KAGGLE-LIVE] {line}\n")
                sys.stdout.flush()
                if log_handle:
                    log_handle.write(line + "\n")
                    log_handle.flush()

                kind = classify_line(line)
                if kind == "fatal":
                    fatal_seen = True
                    sys.stderr.write(f"\nFATAL ERROR DETECTED: {line}\nHALTING STREAM.\n")
                    return 1
                if kind == "success" and "submission_tasks" in line:
                    print("\n[monitor] submission validated in-notebook -- run looks successful.")
                    return 0
    except KeyboardInterrupt:
        print("\nStopped listening (Ctrl-C). This does not stop the Kaggle run itself.")
        return 130
    finally:
        if log_handle:
            log_handle.close()

    return 1 if fatal_seen else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=KAGGLE_USERNAME)
    parser.add_argument("--slug", default=KERNEL_SLUG)
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--listen-target", default=DEFAULT_LISTEN_TARGET)
    parser.add_argument("--debug-dump", default=None, help="Path to append unrecognized raw packet bodies to")
    parser.add_argument("--log-out", default=None, help="Path to append plain-text log lines to")
    args = parser.parse_args(argv)

    return monitor(
        args.username,
        args.slug,
        args.debug_port,
        args.listen_target,
        Path(args.debug_dump) if args.debug_dump else None,
        Path(args.log_out) if args.log_out else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
