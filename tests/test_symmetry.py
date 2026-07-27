from mythos.symmetry import (
    crop,
    find_occlusion_color_candidates,
    hole_bbox,
    hole_cells_for_color,
    repair_grid,
)


def _mirror_symmetric_grid() -> list[list[int]]:
    # Horizontally mirror-symmetric (about the true center) 6x6 grid.
    half = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1],
    ]
    return [row + row[::-1] for row in half]


def test_repair_grid_fills_hole_using_mirror_symmetry() -> None:
    grid = _mirror_symmetric_grid()
    hole = {(0, 0), (0, 1), (1, 0)}
    occluded = [row[:] for row in grid]
    for r, c in hole:
        occluded[r][c] = 8

    repaired = repair_grid(occluded, hole)

    assert repaired is not None
    for r, c in hole:
        assert repaired[r][c] == grid[r][c]


def test_repair_grid_returns_none_when_no_symmetry_covers_the_hole() -> None:
    grid = [[i + j for j in range(6)] for i in range(6)]  # no symmetry at all
    hole = {(0, 0)}
    assert repair_grid(grid, hole) is None


def test_find_occlusion_color_candidates_finds_solid_rectangle() -> None:
    grid = [
        [1, 2, 3, 4],
        [5, 8, 8, 6],
        [7, 8, 8, 9],
        [1, 2, 3, 4],
    ]
    candidates = find_occlusion_color_candidates(grid)
    assert 8 in candidates


def test_find_occlusion_color_candidates_ignores_scattered_color() -> None:
    grid = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    # color 1 is scattered (each cell its own 1x1 component below the area floor)
    assert find_occlusion_color_candidates(grid) == []


def test_hole_bbox_and_crop() -> None:
    hole = {(2, 3), (2, 4), (3, 3), (3, 4)}
    assert hole_bbox(hole) == (2, 3, 2, 2)
    grid = [[r * 10 + c for c in range(6)] for r in range(6)]
    assert crop(grid, 2, 3, 2, 2) == [[23, 24], [33, 34]]


def test_hole_cells_for_color() -> None:
    grid = [[0, 8, 0], [8, 8, 0]]
    assert hole_cells_for_color(grid, 8) == {(0, 1), (1, 0), (1, 1)}
