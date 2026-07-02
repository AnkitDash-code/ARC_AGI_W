from __future__ import annotations

from mythos.arc import Grid
from mythos.metrics import score_submission_data
from mythos.submission import TestPrediction


def test_scoring_uses_best_of_two_attempts_for_exact_and_cell_accuracy() -> None:
    truth: Grid = [[1, 2], [3, 4]]
    predictions = {
        "task": (
            TestPrediction(
                attempt_1=[[1, 2], [0, 0]],
                attempt_2=[[1, 2], [3, 4]],
            ),
        )
    }
    solutions = {"task": (truth,)}

    result = score_submission_data(predictions, solutions)

    assert result.total_items == 1
    assert result.exact_matches == 1
    assert result.exact_attempt_1 == 0
    assert result.exact_attempt_2 == 1
    assert result.cell_accuracy == 1.0
