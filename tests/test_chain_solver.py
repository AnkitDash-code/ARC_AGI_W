"""Tests for ChainSolver: ordered fallback, first success wins."""

from __future__ import annotations

import pytest

from mythos.solvers.base import SolverError
from mythos.solvers.chain import ChainSolver

class _DummyTask:
    """Minimal stand-in with the .id attribute ChainSolver's error message uses."""

    id = "dummy_task"


_DUMMY_TASK = _DummyTask()


class _CountingSolver:
    """Wraps another stub solver, recording how many times solve() was called."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def solve(self, task):
        self.calls += 1
        return self.inner.solve(task)


class _AlwaysSucceeds:
    def __init__(self, result: str = "prediction") -> None:
        self.result = result

    def solve(self, task):
        return self.result


class _AlwaysRaises:
    def solve(self, task):
        raise SolverError("stub always raises")


def test_chain_short_circuits_on_first_success():
    first = _CountingSolver(_AlwaysSucceeds("first"))
    second = _CountingSolver(_AlwaysSucceeds("second"))
    chain = ChainSolver([first, second])

    result = chain.solve(_DUMMY_TASK)

    assert result == "first"
    assert second.calls == 0


def test_chain_falls_through_on_solver_error():
    first = _CountingSolver(_AlwaysRaises())
    second = _CountingSolver(_AlwaysSucceeds("second"))
    chain = ChainSolver([first, second])

    result = chain.solve(_DUMMY_TASK)

    assert result == "second"


def test_chain_always_returns_when_last_solver_never_raises():
    first = _CountingSolver(_AlwaysRaises())
    second = _CountingSolver(_AlwaysRaises())
    last = _CountingSolver(_AlwaysSucceeds("fixture-like"))  # mimics FixtureSolver
    chain = ChainSolver([first, second, last])

    result = chain.solve(_DUMMY_TASK)

    assert result == "fixture-like"


def test_chain_raises_if_every_solver_fails():
    chain = ChainSolver([_AlwaysRaises(), _AlwaysRaises()])

    with pytest.raises(SolverError):
        chain.solve(_DUMMY_TASK)


def test_chain_rejects_empty_solver_list():
    with pytest.raises(ValueError):
        ChainSolver([])


def test_make_solver_chain_degrades_without_llama_cpp_installed():
    """make_solver("chain") is configs/base.json's default -- it must stay
    constructible on a machine without llama-cpp-python installed (this dev
    environment, CI) instead of crashing because AgenticReplSolver's real
    LlamaCppClient loads a GGUF model eagerly in its own __init__."""

    from mythos.solvers.factory import make_solver
    from mythos.solvers.fixture import FixtureSolver
    from mythos.solvers.pipeline import PlannedPipelineSolver
    from mythos.solvers.symbolic import SymbolicSolver

    solver = make_solver("chain")

    assert isinstance(solver, ChainSolver)
    kinds = [type(inner) for inner in solver._solvers]
    assert kinds[0] is SymbolicSolver
    assert PlannedPipelineSolver in kinds
    assert FixtureSolver in kinds
    # PlannedPipelineSolver never raises SolverError in this repo's default
    # config, so it must be last -- otherwise FixtureSolver's verified
    # guesses would never get a turn (see factory.py's "chain" branch).
    assert kinds[-1] is PlannedPipelineSolver


def test_make_solver_chain_solves_toy_task_end_to_end():
    from pathlib import Path

    from mythos.arc import load_challenges
    from mythos.solvers.factory import make_solver

    toy_path = Path(__file__).resolve().parent.parent / "data" / "toy" / "challenges.json"
    task = load_challenges(toy_path)["toy_identity"]

    solver = make_solver("chain")
    prediction = solver.solve(task)

    assert prediction.task_id == "toy_identity"
    assert len(prediction.outputs) == len(task.test)
