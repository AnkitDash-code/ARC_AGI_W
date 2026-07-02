from __future__ import annotations

from pathlib import Path

from mythos.arc import load_challenges
from mythos.pipeline import PLAN_STAGE_ORDER
from mythos.solvers.pipeline import PlannedPipelineSolver


ROOT = Path(__file__).resolve().parents[1]


def test_planned_pipeline_runs_master_plan_stage_order() -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    solver = PlannedPipelineSolver()

    prediction = solver.solve(tasks["toy_identity"])

    assert prediction.task_id == "toy_identity"
    assert solver.last_trace is not None
    assert solver.last_trace.stage_names == list(PLAN_STAGE_ORDER)
    assert all(prediction.outputs)


def test_planned_pipeline_keeps_valid_fallback_for_unknown_rules() -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")
    solver = PlannedPipelineSolver()

    predictions = [solver.solve(task) for task in tasks.values()]

    assert len(predictions) == len(tasks)
    assert set(solver.traces) == set(tasks)
