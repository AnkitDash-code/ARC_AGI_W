"""Scoring helpers for ARC-style submissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple

from mythos.arc import ArcValidationError, Grid, SolutionMap, grid_equal, load_solutions
from mythos.submission import SubmissionMap, TestPrediction, load_submission


@dataclass(frozen=True)
class ScoreResult:
    total_items: int
    exact_matches: int
    exact_attempt_1: int
    exact_attempt_2: int
    total_cells: int
    matched_cells: int
    extra_predictions: int

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.total_items if self.total_items else 0.0

    @property
    def cell_accuracy(self) -> float:
        return self.matched_cells / self.total_cells if self.total_cells else 0.0

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data["exact_accuracy"] = self.exact_accuracy
        data["cell_accuracy"] = self.cell_accuracy
        return data


def _cell_count(grid: Grid) -> int:
    return sum(len(row) for row in grid)


def _cell_matches(prediction: Grid, truth: Grid) -> Tuple[int, int]:
    total = _cell_count(truth)
    if len(prediction) != len(truth) or len(prediction[0]) != len(truth[0]):
        return 0, total
    matches = 0
    for pred_row, truth_row in zip(prediction, truth):
        for pred_cell, truth_cell in zip(pred_row, truth_row):
            if pred_cell == truth_cell:
                matches += 1
    return matches, total


def score_submission_data(predictions: SubmissionMap, solutions: SolutionMap) -> ScoreResult:
    total_items = 0
    exact_matches = 0
    exact_attempt_1 = 0
    exact_attempt_2 = 0
    matched_cells = 0
    total_cells = 0

    extra_predictions = len(set(predictions) - set(solutions))

    for task_id, truth_outputs in solutions.items():
        if task_id not in predictions:
            raise ArcValidationError(f"submission is missing task {task_id}")
        task_predictions = predictions[task_id]
        if len(task_predictions) != len(truth_outputs):
            raise ArcValidationError(
                f"{task_id} has {len(task_predictions)} predictions but "
                f"{len(truth_outputs)} solution outputs"
            )

        for prediction, truth in zip(task_predictions, truth_outputs):
            total_items += 1
            attempt_1_exact = grid_equal(prediction.attempt_1, truth)
            attempt_2_exact = grid_equal(prediction.attempt_2, truth)
            exact_attempt_1 += int(attempt_1_exact)
            exact_attempt_2 += int(attempt_2_exact)
            exact_matches += int(attempt_1_exact or attempt_2_exact)

            cells_1, cells_total = _cell_matches(prediction.attempt_1, truth)
            cells_2, _ = _cell_matches(prediction.attempt_2, truth)
            matched_cells += max(cells_1, cells_2)
            total_cells += cells_total

    return ScoreResult(
        total_items=total_items,
        exact_matches=exact_matches,
        exact_attempt_1=exact_attempt_1,
        exact_attempt_2=exact_attempt_2,
        total_cells=total_cells,
        matched_cells=matched_cells,
        extra_predictions=extra_predictions,
    )


def score_files(prediction_path: str, solution_path: str) -> ScoreResult:
    return score_submission_data(load_submission(prediction_path), load_solutions(solution_path))
