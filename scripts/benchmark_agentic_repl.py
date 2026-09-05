"""Local, zero-Kaggle-cost benchmark for AgenticReplSolver against real ARC-AGI-2 data.

Mirrors scripts/benchmark_symbolic.py's shape and "never crash the whole run"
philosophy. Defaults to --llm-client stub (a deterministic FakeLLMClient that
always proposes the identity function) so this runs with no GPU, no network,
and no staged model -- useful for confirming the prompt-build -> sandboxed
verify -> augment/vote -> submission-shaped-output pipeline doesn't crash.
Pass --llm-client llamacpp (with MYTHOS_AGENTIC_MODEL_PATH set) to benchmark
against the real staged model instead.

Usage:
    python scripts/benchmark_agentic_repl.py --challenges <path> --solutions <path> [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mythos.arc import attach_solutions, load_challenges, load_solutions  # noqa: E402
from mythos.solvers.base import SolverError  # noqa: E402

from agentic_repl.llm.client import FakeLLMClient  # noqa: E402
from agentic_repl.solver import AgenticReplSolver  # noqa: E402

_IDENTITY_CODE = "```python\ndef solve(grid):\n    return grid\n```"


class _CountingLLMClient:
    """Wraps an LLMClient and counts generate() calls, so the benchmark can
    report total LLM calls -- one of the two cost numbers (with wall-clock
    time) Step 2's simplify-pass cost/benefit measurement needs."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.call_count = 0

    def generate(self, prompt: str, *, n: int, temperature: float = 0.7):
        self.call_count += 1
        return self._inner.generate(prompt, n=n, temperature=temperature)


def _make_solver(
    llm_client_name: str,
    num_candidates: int,
    refinement_rounds: int,
    timeout_s: float,
    simplify_rounds: int,
) -> tuple[AgenticReplSolver, _CountingLLMClient]:
    if llm_client_name == "stub":
        client = FakeLLMClient(responses=[_IDENTITY_CODE])
    elif llm_client_name == "llamacpp":
        from agentic_repl.llm.client import LlamaCppClient

        client = LlamaCppClient()
    else:
        raise ValueError(f"unknown --llm-client: {llm_client_name}")
    counting_client = _CountingLLMClient(client)
    solver = AgenticReplSolver(
        counting_client,
        num_candidates=num_candidates,
        refinement_rounds=refinement_rounds,
        simplify_rounds=simplify_rounds,
        timeout_s=timeout_s,
    )
    return solver, counting_client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-task-timeout", type=float, default=30.0)
    parser.add_argument("--candidate-timeout", type=float, default=2.0, help="Per-candidate sandbox exec timeout.")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--simplify-rounds", type=int, default=1)
    parser.add_argument("--llm-client", choices=["stub", "llamacpp"], default="stub")
    args = parser.parse_args()

    challenges = load_challenges(args.challenges)
    solutions = load_solutions(args.solutions)
    tasks = attach_solutions(challenges, solutions)
    task_list = list(tasks.values())
    if args.limit:
        task_list = task_list[: args.limit]

    solver, counting_client = _make_solver(
        args.llm_client, args.num_candidates, args.refinement_rounds, args.candidate_timeout, args.simplify_rounds
    )

    fired = 0
    fired_slow: list[str] = []
    exact_tasks = 0
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
    print(f"total_llm_calls = {counting_client.call_count}")
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
