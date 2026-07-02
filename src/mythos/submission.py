"""Prediction and submission JSON helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from mythos.arc import ArcValidationError, Grid, validate_grid


@dataclass(frozen=True)
class TestPrediction:
    __test__ = False

    attempt_1: Grid
    attempt_2: Grid


@dataclass(frozen=True)
class Prediction:
    task_id: str
    outputs: Tuple[TestPrediction, ...]


SubmissionMap = Dict[str, Tuple[TestPrediction, ...]]


def prediction_to_json(prediction: Prediction) -> List[dict[str, Grid]]:
    return [
        {"attempt_1": item.attempt_1, "attempt_2": item.attempt_2}
        for item in prediction.outputs
    ]


def predictions_to_submission(predictions: Iterable[Prediction]) -> dict[str, List[dict[str, Grid]]]:
    submission: dict[str, List[dict[str, Grid]]] = {}
    for prediction in predictions:
        if prediction.task_id in submission:
            raise ArcValidationError(f"duplicate prediction for task {prediction.task_id}")
        submission[prediction.task_id] = prediction_to_json(prediction)
    if not submission:
        raise ArcValidationError("submission must contain at least one prediction")
    return submission


def validate_submission_data(data: Any) -> SubmissionMap:
    if not isinstance(data, Mapping) or not data:
        raise ArcValidationError("submission must be a non-empty object keyed by task id")
    validated: SubmissionMap = {}
    for task_id, raw_outputs in data.items():
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ArcValidationError(f"{task_id} must contain a non-empty list of test outputs")
        outputs: List[TestPrediction] = []
        for index, raw_output in enumerate(raw_outputs):
            if not isinstance(raw_output, Mapping):
                raise ArcValidationError(f"{task_id}[{index}] must be an object")
            if "attempt_1" not in raw_output or "attempt_2" not in raw_output:
                raise ArcValidationError(f"{task_id}[{index}] must contain attempt_1 and attempt_2")
            outputs.append(
                TestPrediction(
                    attempt_1=validate_grid(raw_output["attempt_1"], field=f"{task_id}[{index}].attempt_1"),
                    attempt_2=validate_grid(raw_output["attempt_2"], field=f"{task_id}[{index}].attempt_2"),
                )
            )
        validated[str(task_id)] = tuple(outputs)
    return validated


def write_submission(predictions: Iterable[Prediction], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = predictions_to_submission(predictions)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_submission(path: str | Path) -> SubmissionMap:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArcValidationError(f"{input_path} is not valid JSON: {exc}") from exc
    return validate_submission_data(raw)
