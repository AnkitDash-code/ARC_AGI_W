"""Ordered fallback chain: try each solver in order, first success wins."""

from __future__ import annotations

from mythos.arc import ArcTask
from mythos.solvers.base import SolverError
from mythos.submission import Prediction


class ChainSolver:
    """Tries each solver in order; returns the first one that doesn't raise
    SolverError. The last solver in the chain should be one that never raises
    (e.g. FixtureSolver) so solve() always returns a valid prediction."""

    def __init__(self, solvers: list) -> None:
        if not solvers:
            raise ValueError("ChainSolver needs at least one solver")
        self._solvers = solvers

    def solve(self, task: ArcTask) -> Prediction:
        last_error: SolverError | None = None
        for solver in self._solvers:
            try:
                return solver.solve(task)
            except SolverError as exc:
                last_error = exc
                continue
        # Should be unreachable if the chain ends in a never-raising solver.
        raise SolverError(
            f"{task.id}: every solver in the chain raised; last error: {last_error}"
        )
