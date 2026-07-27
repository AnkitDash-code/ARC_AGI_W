"""Agentic program-synthesis solver: LLM-generated solve(grid) candidates,
verified against every train pair in a sandboxed REPL, refined on failure,
and voted across augmented views before producing a two-attempt prediction.

Mirrors mythos.solvers.symbolic.SymbolicSolver's "prove correctness on train
or don't fire" contract (src/mythos/solvers/symbolic.py): this solver raises
SolverError rather than ever guessing, so it's safe to slot into the same
solver-factory fallback chain as the existing solvers.
"""

from __future__ import annotations

import os

from mythos.arc import ArcTask, Grid, grid_equal
from mythos.solvers.base import SolverError, make_prediction
from mythos.submission import Prediction

from agentic_repl.augment_vote import vote_predictions
from agentic_repl.dsl.catalog import build_catalog_text
from agentic_repl.llm.client import LLMClient
from agentic_repl.llm.prompts import build_initial_prompt, build_refinement_prompt, extract_code_block
from agentic_repl.repl import ExecutionResult, run_candidate

DEFAULT_NUM_CANDIDATES = 4
DEFAULT_REFINEMENT_ROUNDS = 2
DEFAULT_TIMEOUT_SECONDS = 2.0


def _debug_enabled() -> bool:
    return os.environ.get("MYTHOS_AGENTIC_DEBUG") == "1"


def _debug_log(label: str, text: str, *, limit: int = 800) -> None:
    if not _debug_enabled():
        return
    truncated = text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"
    print(f"[agentic_repl debug] {label}:\n{truncated}")


def _failure_report(train_index: int, result: ExecutionResult, expected: Grid) -> str:
    if not result.ok or result.output is None:
        return f"Example {train_index + 1}: {result.error}"
    got_rows, got_cols = len(result.output), len(result.output[0])
    want_rows, want_cols = len(expected), len(expected[0])
    if (got_rows, got_cols) != (want_rows, want_cols):
        return (
            f"Example {train_index + 1}: got a {got_rows}x{got_cols} grid, "
            f"expected {want_rows}x{want_cols}."
        )
    return f"Example {train_index + 1}: output shape matched but cell values did not."


def _verify_on_train(code: str, task: ArcTask, *, timeout_s: float) -> tuple[bool, str | None]:
    for index, example in enumerate(task.train):
        assert example.output is not None  # train examples always have outputs
        result = run_candidate(code, example.input, timeout_s=timeout_s)
        if not result.ok or result.output is None or not grid_equal(result.output, example.output):
            return False, _failure_report(index, result, example.output)
    return True, None


class AgenticReplSolver:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        refinement_rounds: int = DEFAULT_REFINEMENT_ROUNDS,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._llm_client = llm_client
        self._num_candidates = num_candidates
        self._refinement_rounds = refinement_rounds
        self._timeout_s = timeout_s
        self._dsl_catalog = build_catalog_text()

    def solve(self, task: ArcTask) -> Prediction:
        verified_codes = self._search_verified_programs(task)
        if not verified_codes:
            raise SolverError(f"{task.id}: no agentic-REPL candidate verified against all train pairs")

        attempts = vote_predictions(
            verified_codes, task, run_candidate=run_candidate, timeout_s=self._timeout_s
        )
        return make_prediction(task, attempts)

    def _search_verified_programs(self, task: ArcTask) -> list[str]:
        prompt = build_initial_prompt(task, self._dsl_catalog)
        _debug_log(f"{task.id} initial prompt", prompt, limit=1500)
        raw_completions = self._llm_client.generate(prompt, n=self._num_candidates)
        candidates = []
        for index, completion in enumerate(raw_completions):
            _debug_log(f"{task.id} candidate {index} raw completion", completion)
            code = extract_code_block(completion)
            _debug_log(f"{task.id} candidate {index} extracted code", code)
            candidates.append(code)

        verified: list[str] = []
        for code in candidates:
            verified_code = self._refine_until_verified(code, task)
            if verified_code is not None:
                verified.append(verified_code)
        return verified

    def _refine_until_verified(self, code: str, task: ArcTask) -> str | None:
        current = code
        attempts_left = self._refinement_rounds + 1
        while attempts_left > 0:
            ok, failure = _verify_on_train(current, task, timeout_s=self._timeout_s)
            _debug_log(f"{task.id} verify result", f"ok={ok} failure={failure}")
            if ok:
                return current
            attempts_left -= 1
            if attempts_left == 0:
                return None
            refinement_prompt = build_refinement_prompt(current, failure or "unknown failure")
            completions = self._llm_client.generate(refinement_prompt, n=1)
            if not completions:
                return None
            _debug_log(f"{task.id} refinement raw completion", completions[0])
            current = extract_code_block(completions[0])
        return None
