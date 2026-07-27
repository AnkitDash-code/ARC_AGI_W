"""Verified symbolic solver: symmetry repair, then rigid-transform search.

Every candidate this solver produces is checked for exact agreement against
*every* train pair before it is ever applied to a test input -- when this
solver returns a prediction, it is either provably correct on the train
demonstrations or it doesn't fire at all (raises SolverError, letting the
caller's fallback chain move on to a different solver). This trades recall
for precision: it will never guess.
"""

from __future__ import annotations

from mythos.arc import ArcTask, Grid, grid_equal
from mythos.object_ops import ALL_OBJECT_TRANSFORM_FINDERS
from mythos.objects import ArcObject
from mythos.solvers.base import SolverError, make_prediction
from mythos.solvers.fixture import FixtureSolver
from mythos.submission import Prediction
from mythos.symmetry import (
    crop,
    find_occlusion_color_candidates,
    hole_bbox,
    hole_cells_for_color,
    repair_grid,
)


class SymbolicSolver:
    """Object/symmetry-aware verified solver; falls back to rigid transforms."""

    def __init__(self) -> None:
        self.fixture_solver = FixtureSolver()

    def solve(self, task: ArcTask) -> Prediction:
        prediction = _try_symmetry_repair(task)
        if prediction is not None:
            return prediction
        prediction = _try_object_transforms(task)
        if prediction is not None:
            return prediction
        return self.fixture_solver.solve(task)


def _try_object_transforms(task: ArcTask) -> Prediction | None:
    for find_transform in ALL_OBJECT_TRANSFORM_FINDERS:
        transform = find_transform(task)
        if transform is None:
            continue
        try:
            attempts = [(transform(example.input), transform(example.input)) for example in task.test]
        except Exception:  # noqa: BLE001 - any failure applying to test just means this candidate doesn't fire
            continue
        try:
            return make_prediction(task, attempts)
        except SolverError:
            continue
    return None


def _try_symmetry_repair(task: ArcTask) -> Prediction | None:
    if any(example.output is None for example in task.train):
        return None

    # Occlusion marker colors are almost always literally consistent across
    # a task's own examples (it's one procedurally-generated task instance),
    # so try each color that looks like a plausible occlusion block in *any*
    # train example, as that same literal color everywhere -- more robust
    # than assuming a stable per-example rank ordering.
    candidate_colors: set[int] = set()
    for example in task.train:
        candidate_colors.update(find_occlusion_color_candidates(example.input))

    for color in candidate_colors:
        variant = _verify_occlusion_color(task, color)
        if variant is not None:
            prediction = _apply_occlusion_plan(task, color, variant)
            if prediction is not None:
                return prediction
    return None


def _verify_occlusion_color(task: ArcTask, color: int) -> str | None:
    """Return the verified output variant ('full' or 'crop') for this color, or None."""

    variant: str | None = None
    for example in task.train:
        hole_cells = hole_cells_for_color(example.input, color)
        if not hole_cells:
            return None
        repaired = repair_grid(example.input, hole_cells)
        if repaired is None:
            return None

        example_variant = _matches_variant(repaired, hole_cells, example.output)
        if example_variant is None:
            return None
        if variant is None:
            variant = example_variant
        elif variant != example_variant:
            return None
    return variant


def _matches_variant(repaired: Grid, hole_cells: set, output: Grid) -> str | None:
    if grid_equal(repaired, output):
        return "full"
    top, left, height, width = hole_bbox(hole_cells)
    if len(output) == height and len(output[0]) == width and grid_equal(crop(repaired, top, left, height, width), output):
        return "crop"
    return None


def _apply_occlusion_plan(task: ArcTask, color: int, variant: str) -> Prediction | None:
    attempts: list[tuple[Grid, Grid]] = []
    for example in task.test:
        hole_cells = hole_cells_for_color(example.input, color)
        if not hole_cells:
            return None
        repaired = repair_grid(example.input, hole_cells)
        if repaired is None:
            return None
        if variant == "full":
            grid = repaired
        else:
            top, left, height, width = hole_bbox(hole_cells)
            grid = crop(repaired, top, left, height, width)
        attempts.append((grid, grid))
    try:
        return make_prediction(task, attempts)
    except SolverError:
        return None


__all__ = ["SymbolicSolver", "ArcObject"]
