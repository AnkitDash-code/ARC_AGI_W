"""ARC JSON loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

Grid = List[List[int]]
SolutionMap = Dict[str, Tuple[Grid, ...]]

MAX_GRID_ROWS = 30
MAX_GRID_COLS = 30
MIN_CELL_VALUE = 0
MAX_CELL_VALUE = 9


class ArcValidationError(ValueError):
    """Raised when ARC-style data is malformed."""


@dataclass(frozen=True)
class ArcExample:
    input: Grid
    output: Optional[Grid] = None


@dataclass(frozen=True)
class ArcTask:
    id: str
    train: Tuple[ArcExample, ...]
    test: Tuple[ArcExample, ...]


def _json_load(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArcValidationError(f"{file_path} is not valid JSON: {exc}") from exc


def validate_grid(value: Any, *, field: str = "grid") -> Grid:
    """Validate and copy an ARC grid."""

    if not isinstance(value, list) or not value:
        raise ArcValidationError(f"{field} must be a non-empty list of rows")
    if len(value) > MAX_GRID_ROWS:
        raise ArcValidationError(f"{field} has {len(value)} rows; max is {MAX_GRID_ROWS}")

    rows: Grid = []
    expected_width: int | None = None
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ArcValidationError(f"{field}[{row_idx}] must be a non-empty list")
        if expected_width is None:
            expected_width = len(row)
            if expected_width > MAX_GRID_COLS:
                raise ArcValidationError(
                    f"{field} has {expected_width} columns; max is {MAX_GRID_COLS}"
                )
        elif len(row) != expected_width:
            raise ArcValidationError(
                f"{field} must be rectangular; row 0 has {expected_width} columns "
                f"but row {row_idx} has {len(row)}"
            )

        copied_row: List[int] = []
        for col_idx, cell in enumerate(row):
            if isinstance(cell, bool) or not isinstance(cell, int):
                raise ArcValidationError(f"{field}[{row_idx}][{col_idx}] must be an integer")
            if cell < MIN_CELL_VALUE or cell > MAX_CELL_VALUE:
                raise ArcValidationError(
                    f"{field}[{row_idx}][{col_idx}]={cell}; expected 0..9"
                )
            copied_row.append(cell)
        rows.append(copied_row)
    return rows


def _parse_example(raw: Any, *, task_id: str, split: str, index: int, require_output: bool) -> ArcExample:
    if not isinstance(raw, Mapping):
        raise ArcValidationError(f"{task_id}.{split}[{index}] must be an object")
    if "input" not in raw:
        raise ArcValidationError(f"{task_id}.{split}[{index}] is missing input")
    output = raw.get("output")
    if require_output and output is None:
        raise ArcValidationError(f"{task_id}.{split}[{index}] is missing output")
    return ArcExample(
        input=validate_grid(raw["input"], field=f"{task_id}.{split}[{index}].input"),
        output=validate_grid(output, field=f"{task_id}.{split}[{index}].output")
        if output is not None
        else None,
    )


def _parse_examples(
    raw_examples: Any, *, task_id: str, split: str, require_output: bool
) -> Tuple[ArcExample, ...]:
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ArcValidationError(f"{task_id}.{split} must be a non-empty list")
    return tuple(
        _parse_example(
            raw_example,
            task_id=task_id,
            split=split,
            index=index,
            require_output=require_output,
        )
        for index, raw_example in enumerate(raw_examples)
    )


def parse_task(task_id: str, raw_task: Any) -> ArcTask:
    if not isinstance(raw_task, Mapping):
        raise ArcValidationError(f"{task_id} must be an object")
    if "train" not in raw_task or "test" not in raw_task:
        raise ArcValidationError(f"{task_id} must contain train and test splits")
    return ArcTask(
        id=task_id,
        train=_parse_examples(raw_task["train"], task_id=task_id, split="train", require_output=True),
        test=_parse_examples(raw_task["test"], task_id=task_id, split="test", require_output=False),
    )


def load_challenges(path: str | Path) -> Dict[str, ArcTask]:
    """Load an ARC challenge JSON file keyed by task id."""

    raw = _json_load(path)
    if not isinstance(raw, Mapping) or not raw:
        raise ArcValidationError("challenge file must be a non-empty object keyed by task id")
    return {str(task_id): parse_task(str(task_id), raw_task) for task_id, raw_task in raw.items()}


def grid_shape(grid: Grid) -> Tuple[int, int]:
    return len(grid), len(grid[0])


def grid_equal(left: Grid, right: Grid) -> bool:
    return left == right


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _looks_like_grid(value: Any) -> bool:
    try:
        validate_grid(value)
    except ArcValidationError:
        return False
    return True


def _solution_grids_from_value(task_id: str, value: Any) -> Tuple[Grid, ...]:
    if isinstance(value, Mapping):
        if "test" in value:
            return tuple(
                validate_grid(item["output"], field=f"{task_id}.test[{idx}].output")
                for idx, item in enumerate(value["test"])
                if isinstance(item, Mapping) and "output" in item
            )
        if "output" in value:
            return (validate_grid(value["output"], field=f"{task_id}.output"),)
    if _looks_like_grid(value):
        return (validate_grid(value, field=f"{task_id}.output"),)
    if isinstance(value, list):
        grids: List[Grid] = []
        for idx, item in enumerate(value):
            if isinstance(item, Mapping) and "output" in item:
                grids.append(validate_grid(item["output"], field=f"{task_id}[{idx}].output"))
            elif _looks_like_grid(item):
                grids.append(validate_grid(item, field=f"{task_id}[{idx}]"))
            else:
                raise ArcValidationError(f"{task_id}[{idx}] is not a solution grid")
        if grids:
            return tuple(grids)
    raise ArcValidationError(f"{task_id} does not contain solution outputs")


def load_solutions(path: str | Path) -> SolutionMap:
    raw = _json_load(path)
    if not isinstance(raw, Mapping) or not raw:
        raise ArcValidationError("solution file must be a non-empty object keyed by task id")
    return {
        str(task_id): _solution_grids_from_value(str(task_id), value)
        for task_id, value in raw.items()
    }


def attach_solutions(tasks: Mapping[str, ArcTask], solutions: SolutionMap) -> Dict[str, ArcTask]:
    """Return tasks with test outputs filled from a solution map."""

    attached: Dict[str, ArcTask] = {}
    for task_id, task in tasks.items():
        if task_id not in solutions:
            raise ArcValidationError(f"missing solutions for task {task_id}")
        if len(solutions[task_id]) != len(task.test):
            raise ArcValidationError(
                f"{task_id} has {len(task.test)} test items but "
                f"{len(solutions[task_id])} solution outputs"
            )
        attached[task_id] = ArcTask(
            id=task.id,
            train=task.train,
            test=tuple(
                ArcExample(input=example.input, output=solutions[task_id][index])
                for index, example in enumerate(task.test)
            ),
        )
    return attached


def require_test_outputs(tasks: Iterable[ArcTask]) -> None:
    for task in tasks:
        for index, example in enumerate(task.test):
            if example.output is None:
                raise ArcValidationError(f"{task.id}.test[{index}] is missing output")
