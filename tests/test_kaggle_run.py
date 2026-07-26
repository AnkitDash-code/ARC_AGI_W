from __future__ import annotations

import shutil
from pathlib import Path

from mythos.arc import ArcExample, ArcTask
from mythos.kaggle_run import main, resolve_challenge_path, solve_with_fallback
from mythos.solvers.baseline import BaselineSolver
from mythos.submission import load_submission


ROOT = Path(__file__).resolve().parents[1]


def _toy_task() -> ArcTask:
    example = ArcExample(input=[[1, 2], [3, 4]], output=[[1, 2], [3, 4]])
    return ArcTask(id="toy", train=(example,), test=(ArcExample(input=[[5, 6], [7, 8]]),))


def test_resolves_hyphenated_test_challenge_name(tmp_path: Path) -> None:
    challenge_path = tmp_path / "arc-agi_test-challenges.json"
    shutil.copy(ROOT / "data" / "toy" / "challenges.json", challenge_path)

    assert resolve_challenge_path(tmp_path, "test") == challenge_path


class _AlwaysFailsSolver:
    def solve(self, task: ArcTask):
        raise RuntimeError("boom")


def test_solve_with_fallback_recovers_from_primary_solver_failure() -> None:
    task = _toy_task()
    prediction = solve_with_fallback(_AlwaysFailsSolver(), BaselineSolver(), task)

    assert prediction.task_id == "toy"
    assert len(prediction.outputs) == len(task.test)


def test_solve_with_fallback_recovers_when_fallback_also_fails() -> None:
    task = _toy_task()
    prediction = solve_with_fallback(_AlwaysFailsSolver(), _AlwaysFailsSolver(), task)

    assert prediction.task_id == "toy"
    assert len(prediction.outputs) == len(task.test)
    assert prediction.outputs[0].attempt_2 == [[0]]


def test_kaggle_runner_writes_submission_from_explicit_challenges(tmp_path: Path) -> None:
    output_path = tmp_path / "submission.json"

    exit_code = main(
        [
            "--challenges",
            str(ROOT / "data" / "toy" / "challenges.json"),
            "--solver",
            "baseline",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    submission = load_submission(output_path)
    assert len(submission) == 5
