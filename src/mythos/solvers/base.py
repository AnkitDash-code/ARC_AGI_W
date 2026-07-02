"""Common solver types."""

from __future__ import annotations

from typing import Protocol

from mythos.arc import ArcTask, Grid
from mythos.submission import Prediction, TestPrediction


class SolverError(RuntimeError):
    """Raised when a solver cannot produce a prediction."""


class Solver(Protocol):
    def solve(self, task: ArcTask) -> Prediction:
        """Return a two-attempt prediction for every test item in a task."""


def make_prediction(task: ArcTask, attempts: list[tuple[Grid, Grid]]) -> Prediction:
    if len(attempts) != len(task.test):
        raise SolverError(
            f"{task.id}: expected {len(task.test)} test predictions, got {len(attempts)}"
        )
    return Prediction(
        task_id=task.id,
        outputs=tuple(TestPrediction(attempt_1=a1, attempt_2=a2) for a1, a2 in attempts),
    )
