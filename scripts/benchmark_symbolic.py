"""Local, zero-Kaggle-cost benchmark for SymbolicSolver against real ARC-AGI-2 data.

Usage:
    python scripts/benchmark_symbolic.py --challenges <path> --solutions <path> [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mythos.arc import attach_solutions, load_challenges, load_solutions  # noqa: E402
from mythos.solvers.base import SolverError  # noqa: E402
from mythos.solvers.symbolic import SymbolicSolver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-task-timeout", type=float, default=5.0)
    args = parser.parse_args()

    challenges = load_challenges(args.challenges)
    solutions = load_solutions(args.solutions)
    tasks = attach_solutions(challenges, solutions)
    task_list = list(tasks.values())
    if args.limit:
        task_list = task_list[: args.limit]

    solver = SymbolicSolver()

    fired = 0
    fired_slow: list[str] = []
    exact_tasks = 0  # every test item in the task matched
    exact_items = 0
    total_items = 0
    errors: list[str] = []
    start = time.perf_counter()

    for index, task in enumerate(task_list):
        task_start = time.perf_counter()
        try:
            prediction = solver.solve(task)
        except SolverError:
            continue
        except Exception as exc:  # noqa: BLE001 - report, don't crash the benchmark
            errors.append(f"{task.id}: {exc!r}")
            continue
        elapsed = time.perf_counter() - task_start
        if elapsed > args.per_task_timeout:
            fired_slow.append(f"{task.id}: {elapsed:.2f}s")

        fired += 1
        truths = solutions[task.id]
        task_exact = True
        for output, truth in zip(prediction.outputs, truths):
            total_items += 1
            item_exact = output.attempt_1 == truth or output.attempt_2 == truth
            if item_exact:
                exact_items += 1
            else:
                task_exact = False
        if task_exact:
            exact_tasks += 1

        if (index + 1) % 100 == 0:
            print(f"...{index + 1}/{len(task_list)} tasks processed, {fired} fired so far", file=sys.stderr)

    total_time = time.perf_counter() - start
    print(f"tasks_total = {len(task_list)}")
    print(f"fired = {fired} ({100 * fired / len(task_list):.1f}%)")
    print(f"exact_tasks_of_fired = {exact_tasks} ({100 * exact_tasks / fired:.1f}% precision)" if fired else "exact_tasks_of_fired = 0")
    print(f"exact_items_of_fired = {exact_items}/{total_items}")
    print(f"total_time_seconds = {total_time:.1f}")
    print(f"avg_seconds_per_task = {total_time / len(task_list):.3f}")
    if fired_slow:
        print(f"slow_tasks (>{args.per_task_timeout}s): {len(fired_slow)}")
        for entry in fired_slow[:10]:
            print(f"  {entry}")
    if errors:
        print(f"errors: {len(errors)}")
        for entry in errors[:10]:
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
