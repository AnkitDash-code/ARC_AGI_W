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
from agentic_repl.complexity import program_complexity
from agentic_repl.dsl.catalog import build_catalog_text
from agentic_repl.llm.client import LLMClient
from agentic_repl.llm.prompts import (
    build_initial_prompt,
    build_refinement_prompt,
    build_simplify_prompt,
    extract_code_block,
)
from agentic_repl.repl import ExecutionResult, run_candidate

DEFAULT_NUM_CANDIDATES = 4
DEFAULT_REFINEMENT_ROUNDS = 2
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_SIMPLIFY_ROUNDS = 1


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
        simplify_rounds: int = DEFAULT_SIMPLIFY_ROUNDS,
    ) -> None:
        self._llm_client = llm_client
        self._num_candidates = num_candidates
        self._refinement_rounds = refinement_rounds
        self._timeout_s = timeout_s
        self._simplify_rounds = simplify_rounds
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
                verified.append(self._simplify_if_possible(verified_code, task))
        return verified

    def _simplify_if_possible(self, verified_code: str, task: ArcTask) -> str:
        """Ask for a shorter equivalent of an already-verified candidate and
        keep whichever version is actually shorter, by AST node count
        (agentic_repl.complexity.program_complexity) -- an MDL/Occam's-razor
        safeguard against a candidate that satisfies train pairs by encoding
        incidental detail rather than the general rule.

        This is a separate, additional bounded step on top of an
        already-verified candidate -- it does not consume any of the
        refinement_rounds budget, and per this project's rule that a
        "simplify" step must never replace a verified candidate with an
        unverified one, an unverified simplification is always discarded,
        never surfaced.
        """

        if self._simplify_rounds <= 0:
            return verified_code
        for _ in range(self._simplify_rounds):
            prompt = build_simplify_prompt(task, self._dsl_catalog, verified_code)
            completions = self._llm_client.generate(prompt, n=1)
            if not completions:
                continue
            candidate = extract_code_block(completions[0])
            ok, _ = _verify_on_train(candidate, task, timeout_s=self._timeout_s)
            if not ok:
                continue  # discard: never surface an unverified simplification
            try:
                if program_complexity(candidate) < program_complexity(verified_code):
                    verified_code = candidate
            except SyntaxError:
                continue  # discard: shouldn't happen for verified code, but never trust blindly
        return verified_code

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
            refinement_prompt = build_refinement_prompt(
                task, self._dsl_catalog, current, failure or "unknown failure"
            )
            completions = self._llm_client.generate(refinement_prompt, n=1)
            if not completions:
                return None
            _debug_log(f"{task.id} refinement raw completion", completions[0])
            current = extract_code_block(completions[0])
        return None
