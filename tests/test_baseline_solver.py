from __future__ import annotations

from mythos.arc import ArcExample, ArcTask
from mythos.solvers.baseline import BaselineSolver


def test_baseline_solver_falls_back_when_fixture_rules_do_not_match() -> None:
    task = ArcTask(
        id="unknown",
        train=(
            ArcExample(input=[[1, 2], [3, 4]], output=[[9]]),
        ),
        test=(
            ArcExample(input=[[5, 6], [7, 8]]),
        ),
    )

    prediction = BaselineSolver().solve(task)

    assert prediction.task_id == "unknown"
    assert prediction.outputs[0].attempt_1 == [[5, 6], [7, 8]]
    assert prediction.outputs[0].attempt_2 == [[9]]
