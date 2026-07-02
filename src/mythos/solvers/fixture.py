"""Small deterministic solver for the committed toy fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from mythos.arc import ArcTask, Grid, copy_grid, grid_equal
from mythos.solvers.base import SolverError, make_prediction
from mythos.submission import Prediction

Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class Candidate:
    name: str
    transform: Transform


class FixtureSolver:
    """Infer one simple transformation from train examples and apply it to tests."""

    def solve(self, task: ArcTask) -> Prediction:
        candidates = _matching_candidates(task)
        if not candidates:
            raise SolverError(f"{task.id}: no fixture transformation matched train examples")

        primary = candidates[0].transform
        secondary = candidates[1].transform if len(candidates) > 1 else primary
        attempts = [(primary(example.input), secondary(example.input)) for example in task.test]
        return make_prediction(task, attempts)


def _matching_candidates(task: ArcTask) -> List[Candidate]:
    candidates = _base_candidates()
    candidates.extend(_translation_candidates(task))
    recolor = _recolor_candidate(task)
    if recolor is not None:
        candidates.append(recolor)

    matched: List[Candidate] = []
    seen_outputs: set[str] = set()
    for candidate in candidates:
        if _fits(task, candidate.transform):
            signature = _candidate_signature(task, candidate.transform)
            if signature not in seen_outputs:
                matched.append(candidate)
                seen_outputs.add(signature)
    return matched


def _candidate_signature(task: ArcTask, transform: Transform) -> str:
    return repr([transform(example.input) for example in task.test])


def _fits(task: ArcTask, transform: Transform) -> bool:
    for example in task.train:
        if example.output is None:
            return False
        try:
            predicted = transform(example.input)
        except ValueError:
            return False
        if not grid_equal(predicted, example.output):
            return False
    return True


def _base_candidates() -> List[Candidate]:
    return [
        Candidate("identity", copy_grid),
        Candidate("mirror_horizontal", _mirror_horizontal),
        Candidate("mirror_vertical", _mirror_vertical),
        Candidate("rotate_clockwise", _rotate_clockwise),
        Candidate("rotate_180", lambda grid: _rotate_clockwise(_rotate_clockwise(grid))),
        Candidate("rotate_counterclockwise", _rotate_counterclockwise),
    ]


def _mirror_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _mirror_vertical(grid: Grid) -> Grid:
    return [row[:] for row in reversed(grid)]


def _rotate_clockwise(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def _rotate_counterclockwise(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def _recolor_candidate(task: ArcTask) -> Optional[Candidate]:
    mapping: dict[int, int] = {}
    for example in task.train:
        if example.output is None:
            return None
        if len(example.input) != len(example.output) or len(example.input[0]) != len(example.output[0]):
            return None
        for in_row, out_row in zip(example.input, example.output):
            for in_cell, out_cell in zip(in_row, out_row):
                previous = mapping.setdefault(in_cell, out_cell)
                if previous != out_cell:
                    return None

    def transform(grid: Grid) -> Grid:
        return [[mapping.get(cell, cell) for cell in row] for row in grid]

    return Candidate("recolor", transform)


def _translation_candidates(task: ArcTask) -> List[Candidate]:
    offsets: Optional[set[tuple[int, int]]] = None
    for example in task.train:
        if example.output is None:
            return []
        example_offsets = set(_valid_translation_offsets(example.input, example.output))
        offsets = example_offsets if offsets is None else offsets & example_offsets
    return [
        Candidate(f"translate_{dr}_{dc}", _translate_transform(dr, dc))
        for dr, dc in sorted(offsets or set())
        if dr != 0 or dc != 0
    ]


def _valid_translation_offsets(source: Grid, target: Grid) -> Iterable[tuple[int, int]]:
    if len(source) != len(target) or len(source[0]) != len(target[0]):
        return []

    source_cells = _foreground_cells(source)
    target_cells = _foreground_cells(target)
    if len(source_cells) != len(target_cells):
        return []
    if not source_cells and not target_cells:
        return [(0, 0)]

    offsets = []
    first_r, first_c, first_value = source_cells[0]
    for target_r, target_c, target_value in target_cells:
        if target_value != first_value:
            continue
        dr = target_r - first_r
        dc = target_c - first_c
        try:
            translated = _translate_grid(source, dr, dc)
        except ValueError:
            continue
        if translated == target:
            offsets.append((dr, dc))
    return offsets


def _foreground_cells(grid: Grid) -> List[tuple[int, int, int]]:
    return [
        (row_idx, col_idx, cell)
        for row_idx, row in enumerate(grid)
        for col_idx, cell in enumerate(row)
        if cell != 0
    ]


def _translate_transform(dr: int, dc: int) -> Transform:
    def transform(grid: Grid) -> Grid:
        return _translate_grid(grid, dr, dc)

    return transform


def _translate_grid(grid: Grid, dr: int, dc: int) -> Grid:
    height = len(grid)
    width = len(grid[0])
    translated = [[0 for _ in range(width)] for _ in range(height)]
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == 0:
                continue
            next_r = row_idx + dr
            next_c = col_idx + dc
            if next_r < 0 or next_r >= height or next_c < 0 or next_c >= width:
                raise ValueError("translation moves cell out of bounds")
            translated[next_r][next_c] = cell
    return translated
