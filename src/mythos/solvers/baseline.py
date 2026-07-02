"""Guaranteed-output baseline solver for smoke runs and Kaggle plumbing tests."""

from __future__ import annotations

from collections import Counter

from mythos.arc import ArcTask, Grid, copy_grid
from mythos.solvers.base import SolverError, make_prediction
from mythos.solvers.fixture import FixtureSolver
from mythos.submission import Prediction


class BaselineSolver:
    """Try simple fixture rules, then fall back to valid low-skill predictions."""

    def __init__(self) -> None:
        self.fixture_solver = FixtureSolver()

    def solve(self, task: ArcTask) -> Prediction:
        try:
            return self.fixture_solver.solve(task)
        except SolverError:
            attempts = [
                (copy_grid(example.input), _blank_output_for_task(task, example.input))
                for example in task.test
            ]
            return make_prediction(task, attempts)


def _blank_output_for_task(task: ArcTask, input_grid: Grid) -> Grid:
    height, width = _fallback_shape(task, input_grid)
    color = _dominant_output_color(task)
    return [[color for _ in range(width)] for _ in range(height)]


def _fallback_shape(task: ArcTask, input_grid: Grid) -> tuple[int, int]:
    output_shapes = {
        (len(example.output), len(example.output[0]))
        for example in task.train
        if example.output is not None
    }
    if len(output_shapes) == 1:
        return next(iter(output_shapes))
    return len(input_grid), len(input_grid[0])


def _dominant_output_color(task: ArcTask) -> int:
    counts: Counter[int] = Counter()
    for example in task.train:
        if example.output is None:
            continue
        for row in example.output:
            counts.update(row)
    if not counts:
        return 0
    return counts.most_common(1)[0][0]
