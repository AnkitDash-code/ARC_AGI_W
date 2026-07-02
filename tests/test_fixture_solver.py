from __future__ import annotations

from pathlib import Path

from mythos.arc import load_challenges
from mythos.metrics import score_files
from mythos.solvers.fixture import FixtureSolver
from mythos.submission import load_submission, write_submission


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_solver_round_trips_submission_and_scores_exact(tmp_path: Path) -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    solver = FixtureSolver()
    predictions = [solver.solve(task) for task in tasks.values()]

    submission_path = tmp_path / "submission.json"
    write_submission(predictions, submission_path)
    submission = load_submission(submission_path)

    assert set(submission) == set(tasks)
    for task_id, outputs in submission.items():
        assert len(outputs) == len(tasks[task_id].test)
        for output in outputs:
            assert output.attempt_1
            assert output.attempt_2

    result = score_files(
        str(submission_path),
        str(ROOT / "data" / "toy" / "solutions.json"),
    )
    assert result.total_items == 5
    assert result.exact_matches == 5
    assert result.exact_accuracy == 1.0
    assert result.cell_accuracy == 1.0
