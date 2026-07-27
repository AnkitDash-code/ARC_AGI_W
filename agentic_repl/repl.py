"""Sandboxed execution of LLM-generated solve(grid) candidates.

Runs candidates in a subprocess (multiprocessing, spawn context) rather than
using signal.alarm: signal.alarm doesn't exist on Windows (this repo is
developed on Windows, deployed on Kaggle/Linux), and a subprocess is also the
only way to actually kill code that's genuinely hung.

The worker process is persistent, not spawned fresh per call: a fresh spawn
has to re-import this whole module (and mythos/agentic_repl's dependency
chain) before it can run anything, and that reimport cost -- not the
candidate code itself -- can exceed a short per-candidate timeout on its own
(measured in practice: ~2s of pure process-startup overhead when invoked
from a plain script, versus negligible overhead from inside an
already-running test process). Reusing one worker across many calls pays
that cost once; a fresh `solve` namespace is still built per call (see
_build_namespace), so candidates never see state left behind by a previous
one. If a candidate hangs, its worker is killed and a replacement is spawned
lazily on the next call -- this trades a small amount of inter-candidate
process isolation (they share one OS process across a run, not one each) for
throughput that's necessary given real per-task wall-clock budgets; this is
fine here since candidates are LLM-generated ARC solve() attempts, not
adversarial input.

Not thread-safe: callers (solver.py, augment_vote.py) invoke run_candidate
serially, one call at a time.

exec() runs against a restricted namespace: a small builtins allowlist plus
the DSL functions from agentic_repl.dsl, and nothing else -- no imports, no
file/network/process access.
"""

from __future__ import annotations

import atexit
import builtins
from dataclasses import dataclass
import multiprocessing as mp
import queue as queue_module
import time
import traceback

from mythos.arc import ArcValidationError, Grid, validate_grid

from agentic_repl.dsl import PUBLIC_NAMESPACE

DEFAULT_TIMEOUT_SECONDS = 2.0
_WORKER_STARTUP_TIMEOUT_SECONDS = 60.0
_WORKER_READY = "__agentic_repl_worker_ready__"

_ALLOWED_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max",
    "min", "next", "range", "reversed", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip",
    "Exception", "IndexError", "KeyError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
)


@dataclass
class ExecutionResult:
    ok: bool
    output: Grid | None
    error: str | None
    elapsed_seconds: float


def _build_namespace() -> dict[str, object]:
    safe_builtins = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}
    namespace: dict[str, object] = {"__builtins__": safe_builtins}
    namespace.update(PUBLIC_NAMESPACE)
    return namespace


def _execute(code: str, input_grid: Grid) -> tuple[bool, Grid | None, str | None]:
    """Runs inside the worker process. Never raises -- always returns a report."""

    namespace = _build_namespace()
    try:
        exec(compile(code, "<agentic_repl_candidate>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - report every failure mode, don't crash the worker
        return False, None, f"candidate code failed to compile/exec:\n{traceback.format_exc()}"

    solve_fn = namespace.get("solve")
    if not callable(solve_fn):
        return False, None, "candidate code must define a top-level `solve(grid)` function"

    try:
        output = solve_fn([row[:] for row in input_grid])
    except Exception:  # noqa: BLE001
        return False, None, f"solve(grid) raised:\n{traceback.format_exc()}"

    try:
        validated = validate_grid(output, field="solve() output")
    except ArcValidationError as exc:
        return False, None, f"solve(grid) returned an invalid grid: {exc}"

    return True, validated, None


def _worker_loop(task_queue: "mp.Queue[object]", result_queue: "mp.Queue[object]") -> None:
    result_queue.put(_WORKER_READY)
    while True:
        message = task_queue.get()
        if message is None:  # shutdown sentinel
            return
        code, input_grid = message
        result_queue.put(_execute(code, input_grid))


_worker_process: "mp.process.BaseProcess | None" = None
_worker_task_queue: "mp.Queue[object] | None" = None
_worker_result_queue: "mp.Queue[object] | None" = None


def _spawn_worker() -> tuple["mp.process.BaseProcess", "mp.Queue[object]", "mp.Queue[object]"]:
    ctx = mp.get_context("spawn")
    task_queue: "mp.Queue[object]" = ctx.Queue()
    result_queue: "mp.Queue[object]" = ctx.Queue()
    process = ctx.Process(target=_worker_loop, args=(task_queue, result_queue), daemon=True)
    process.start()
    try:
        ready = result_queue.get(timeout=_WORKER_STARTUP_TIMEOUT_SECONDS)
    except queue_module.Empty as exc:
        process.kill()
        raise RuntimeError(
            f"agentic_repl worker failed to start within {_WORKER_STARTUP_TIMEOUT_SECONDS}s"
        ) from exc
    if ready != _WORKER_READY:
        process.kill()
        raise RuntimeError("agentic_repl worker sent an unexpected startup message")
    return process, task_queue, result_queue


def _kill_worker() -> None:
    global _worker_process, _worker_task_queue, _worker_result_queue
    if _worker_process is not None and _worker_process.is_alive():
        _worker_process.terminate()
        _worker_process.join(1.0)
        if _worker_process.is_alive():
            _worker_process.kill()
            _worker_process.join()
    _worker_process = None
    _worker_task_queue = None
    _worker_result_queue = None


atexit.register(_kill_worker)


def run_candidate(
    code: str,
    input_grid: Grid,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Execute `code`'s solve(grid) against input_grid under a strict timeout.

    Always returns an ExecutionResult -- never raises -- so callers (the
    solver's train-pair verification loop, the augmentation/voting stage)
    can treat every candidate uniformly, whether it's a syntax error, a
    runtime crash, a shape mismatch, or a genuine timeout.
    """

    global _worker_process, _worker_task_queue, _worker_result_queue
    if _worker_process is None or not _worker_process.is_alive():
        _worker_process, _worker_task_queue, _worker_result_queue = _spawn_worker()

    start = time.perf_counter()
    _worker_task_queue.put((code, input_grid))
    try:
        ok, output, error = _worker_result_queue.get(timeout=timeout_s)
    except queue_module.Empty:
        elapsed = time.perf_counter() - start
        _kill_worker()  # the worker may be genuinely hung; discard it, respawn lazily next call
        return ExecutionResult(
            ok=False, output=None, error=f"execution exceeded {timeout_s}s timeout", elapsed_seconds=elapsed
        )

    elapsed = time.perf_counter() - start
    return ExecutionResult(ok=ok, output=output, error=error, elapsed_seconds=elapsed)
