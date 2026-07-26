"""Shared helpers for the Kaggle CI/CD scripts (push, monitor, verify).

Only depends on the standard library and the official `kaggle` CLI being on
PATH. Nothing here talks to Chrome/DrissionPage -- that lives in
kaggle_live_monitor.py, which is a best-effort visibility layer on top of
this reliability backbone.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Any

KAGGLE_USERNAME = "ankitdash24"
KERNEL_SLUG = "project-mythos-pipeline"
COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"

# Lines starting with this prefix come from our own per-task fault-isolation
# fallback (see src/mythos/kaggle_run.py::solve_with_fallback and the matching
# notebook cell) -- they mean one task degraded to a baseline prediction, not
# that the run crashed. Never classify these as fatal.
NONFATAL_PREFIX = "WARNING: "

FATAL_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"CUDA out of memory"),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"Kernel (crashed|died|Restarting)", re.IGNORECASE),
    re.compile(r"No space left on device"),
    re.compile(r"Segmentation fault"),
    re.compile(r"exceeded the (session|notebook) time limit", re.IGNORECASE),
    re.compile(r"^ERROR: "),  # kaggle_run.py main()'s outer failure path
]

SUCCESS_LINE_PATTERNS = [
    re.compile(r"^Wrote submission:"),
    re.compile(r"^submission_tasks ="),
]


def classify_line(line: str) -> str:
    """Return 'fatal', 'warning', 'success', or 'info' for one log line."""
    stripped = line.strip()
    if not stripped:
        return "info"
    if stripped.startswith(NONFATAL_PREFIX):
        return "warning"
    for pattern in FATAL_PATTERNS:
        if pattern.search(stripped):
            return "fatal"
    for pattern in SUCCESS_LINE_PATTERNS:
        if pattern.match(stripped):
            return "success"
    return "info"


def kernel_ref(username: str = KAGGLE_USERNAME, slug: str = KERNEL_SLUG) -> str:
    return f"{username}/{slug}"


def run_kaggle(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Run the official `kaggle` CLI and return the completed process (never raises on non-zero).

    Forces UTF-8 for both the child process's own stdout/stderr encoding and
    our decoding of it. Without this, on Windows the kaggle CLI's own rich-
    table output (box-drawing characters) crashes mid-write when its stdout
    is piped rather than a real console, truncating whatever it was
    downloading (observed: `kernels output` silently wrote a 0-byte log file
    after the crash cut off the download).
    """
    import os

    cmd = ["kaggle", *args]
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=child_env, **kwargs)


def validate_submission_shape(path: Path) -> tuple[bool, str]:
    """Schema-check a downloaded submission.json without importing mythos.

    Mirrors mythos.submission.validate_submission_data's shape requirements:
    a non-empty object keyed by task id, each value a non-empty list of
    {attempt_1, attempt_2} grids.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{path.name} is not readable/valid JSON: {exc}"
    if not isinstance(data, dict) or not data:
        return False, f"{path.name} must be a non-empty object keyed by task id"
    for task_id, outputs in data.items():
        if not isinstance(outputs, list) or not outputs:
            return False, f"{path.name}: task {task_id!r} has no test outputs"
        for index, item in enumerate(outputs):
            if not isinstance(item, dict) or "attempt_1" not in item or "attempt_2" not in item:
                return False, f"{path.name}: task {task_id!r}[{index}] missing attempt_1/attempt_2"
    return True, f"{path.name}: {len(data)} tasks, schema OK"


@dataclasses.dataclass
class RunVerdict:
    status: str  # "success", "fatal_error", "timeout", "unknown"
    fatal_lines: list[str] = dataclasses.field(default_factory=list)
    warning_lines: list[str] = dataclasses.field(default_factory=list)
    success_markers_seen: list[str] = dataclasses.field(default_factory=list)
    submission_check: str = ""
    elapsed_seconds: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
