"""Tests for the agentic-REPL solver: sandbox safety + the end-to-end verified loop.

Uses FakeLLMClient exclusively -- no network, no GPU, no real model -- so
this suite runs identically on a dev machine and in CI, and never needs the
staged Kaggle-Dataset model from agentic_repl/models/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mythos.arc import load_challenges
from mythos.solvers.base import SolverError

from agentic_repl.llm.client import FakeLLMClient
from agentic_repl.repl import run_candidate
from agentic_repl.solver import AgenticReplSolver

TOY_CHALLENGES = Path(__file__).resolve().parent.parent / "data" / "toy" / "challenges.json"

IDENTITY_CODE = "def solve(grid):\n    return grid\n"
MIRROR_CODE = "def solve(grid):\n    return [row[::-1] for row in grid]\n"
INFINITE_LOOP_CODE = "def solve(grid):\n    while True:\n        pass\n"
SYNTAX_ERROR_CODE = "def solve(grid)\n    return grid\n"
INVALID_GRID_CODE = "def solve(grid):\n    return [[0, 1], [2]]\n"  # ragged rows: not a valid grid
IMPORT_OS_CODE = "def solve(grid):\n    import os\n    return grid\n"


def _load_task(task_id: str):
    tasks = load_challenges(TOY_CHALLENGES)
    return tasks[task_id]


def _fenced(code: str) -> str:
    return f"```python\n{code}```"


def test_run_candidate_kills_infinite_loop():
    task = _load_task("toy_identity")
    result = run_candidate(INFINITE_LOOP_CODE, task.train[0].input, timeout_s=0.5)
    assert result.ok is False
    assert "timeout" in (result.error or "").lower()


def test_run_candidate_reports_syntax_error_without_raising():
    task = _load_task("toy_identity")
    result = run_candidate(SYNTAX_ERROR_CODE, task.train[0].input, timeout_s=2.0)
    assert result.ok is False
    assert result.error


def test_run_candidate_reports_invalid_grid_without_raising():
    task = _load_task("toy_identity")
    result = run_candidate(INVALID_GRID_CODE, task.train[0].input, timeout_s=2.0)
    assert result.ok is False
    assert result.error


def test_run_candidate_blocks_imports():
    task = _load_task("toy_identity")
    result = run_candidate(IMPORT_OS_CODE, task.train[0].input, timeout_s=2.0)
    assert result.ok is False
    assert result.error


def test_run_candidate_succeeds_on_correct_code():
    task = _load_task("toy_identity")
    result = run_candidate(IDENTITY_CODE, task.train[0].input, timeout_s=2.0)
    assert result.ok is True
    assert result.output == task.train[0].input


def test_solver_end_to_end_with_correct_candidate():
    task = _load_task("toy_identity")
    client = FakeLLMClient(responses=[_fenced(IDENTITY_CODE)])
    solver = AgenticReplSolver(client, num_candidates=1, refinement_rounds=0, timeout_s=2.0)

    prediction = solver.solve(task)

    assert prediction.task_id == "toy_identity"
    assert len(prediction.outputs) == len(task.test)
    assert prediction.outputs[0].attempt_1 == task.test[0].input


def test_solver_refines_a_failing_candidate_into_a_passing_one():
    task = _load_task("toy_mirror")
    client = FakeLLMClient(
        responses=[
            _fenced(IDENTITY_CODE),  # wrong on the train pair (not a mirror)
            _fenced(MIRROR_CODE),  # corrected after feedback
        ]
    )
    solver = AgenticReplSolver(client, num_candidates=1, refinement_rounds=1, timeout_s=2.0)

    prediction = solver.solve(task)

    expected = [row[::-1] for row in task.test[0].input]
    assert prediction.outputs[0].attempt_1 == expected


def test_solver_raises_solver_error_when_nothing_verifies():
    task = _load_task("toy_mirror")
    client = FakeLLMClient(responses=[_fenced(IDENTITY_CODE)])
    solver = AgenticReplSolver(client, num_candidates=1, refinement_rounds=0, timeout_s=2.0)

    with pytest.raises(SolverError):
        solver.solve(task)
