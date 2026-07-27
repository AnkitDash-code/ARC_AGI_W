import pytest

from mythos.arc import ArcExample, ArcTask
from mythos.solvers.base import SolverError
from mythos.solvers.symbolic import SymbolicSolver


def _mirror_symmetric_grid() -> list[list[int]]:
    # Values kept out of 8/9 so those remain free to use as occlusion markers
    # in tests without colliding with legitimate grid content.
    half = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 6, 5],
        [5, 6, 7],
        [6, 5, 4],
        [3, 2, 1],
    ]
    return [row + row[::-1] for row in half]


def _occlude(grid: list[list[int]], hole: set[tuple[int, int]], color: int) -> list[list[int]]:
    occluded = [row[:] for row in grid]
    for r, c in hole:
        occluded[r][c] = color
    return occluded


def test_symbolic_solver_repairs_full_grid_variant() -> None:
    grid = _mirror_symmetric_grid()
    hole = {(0, 0), (0, 1), (1, 0), (1, 1)}
    occluded_input = _occlude(grid, hole, 9)

    # Two train examples with the same occlusion color, different hole
    # positions, so the recolor-substitution candidates in FixtureSolver
    # can't accidentally take priority.
    hole2 = {(4, 4), (4, 5), (5, 4), (5, 5)}
    occluded_input2 = _occlude(grid, hole2, 9)

    task = ArcTask(
        id="t",
        train=(
            ArcExample(input=occluded_input, output=grid),
            ArcExample(input=occluded_input2, output=grid),
        ),
        test=(ArcExample(input=occluded_input),),
    )

    solver = SymbolicSolver()
    prediction = solver.solve(task)
    assert prediction.outputs[0].attempt_1 == grid


def test_symbolic_solver_repairs_crop_variant() -> None:
    grid = _mirror_symmetric_grid()
    hole = {(0, 0), (0, 1), (1, 0), (1, 1)}
    patch = [[grid[r][c] for c in range(2)] for r in range(2)]
    occluded_input = _occlude(grid, hole, 9)

    hole2 = {(4, 4), (4, 5), (5, 4), (5, 5)}
    patch2 = [[grid[r][c] for c in range(4, 6)] for r in range(4, 6)]
    occluded_input2 = _occlude(grid, hole2, 9)

    task = ArcTask(
        id="t",
        train=(
            ArcExample(input=occluded_input, output=patch),
            ArcExample(input=occluded_input2, output=patch2),
        ),
        test=(ArcExample(input=occluded_input),),
    )

    solver = SymbolicSolver()
    prediction = solver.solve(task)
    assert prediction.outputs[0].attempt_1 == patch


def test_symbolic_solver_falls_through_to_fixture_solver() -> None:
    grid = [[1, 2], [3, 4]]
    mirrored = [row[::-1] for row in grid]
    task = ArcTask(
        id="t",
        train=(ArcExample(input=grid, output=mirrored),),
        test=(ArcExample(input=grid),),
    )
    solver = SymbolicSolver()
    prediction = solver.solve(task)
    assert prediction.outputs[0].attempt_1 == mirrored


def test_symbolic_solver_raises_when_nothing_matches() -> None:
    task = ArcTask(
        id="t",
        train=(ArcExample(input=[[1, 2]], output=[[3, 4, 5]]),),
        test=(ArcExample(input=[[1, 2]]),),
    )
    solver = SymbolicSolver()
    with pytest.raises(SolverError):
        solver.solve(task)
