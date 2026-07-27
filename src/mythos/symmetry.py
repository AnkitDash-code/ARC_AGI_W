"""Symmetry-repair primitives: recover an occluded region from grid symmetry.

Targets a common ARC-AGI-2 task family (e.g. the diagnosed task 0934a4d8):
a mosaic-like grid has one or more axis-aligned rectangular blocks painted
over with a single "occlusion" color, and the true content underneath must
be recovered from the grid's own mirror/rotational/periodic symmetries. The
train examples then either want the full repaired grid back, or just the
recovered patch cropped to the occlusion's bounding box.

This is deliberately verification-driven, not heuristic-driven: every
candidate (which color is the occlusion, which symmetries hold) is checked
for exact consistency against the parts of the grid that are NOT occluded,
so a wrong guess self-rejects instead of producing a plausible-looking but
wrong answer.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from mythos.arc import Grid
from mythos.objects import segment_objects

Cell = tuple[int, int]
SymmetryFn = Callable[[Cell], Cell]

_MIN_OCCLUSION_AREA = 4
_MAX_OCCLUSION_CANDIDATES = 6
_MAX_BASE_SYMMETRIES_TO_COMPOSE = 24


def find_occlusion_color_candidates(grid: Grid) -> list[int]:
    """Colors whose every same-color component is a filled rectangle, ranked by area.

    A real occlusion block is normally a single solid rectangle (or a few of
    them, all the same fill color); this rejects colors that just happen to
    be common but scattered (real image content), and trivial single-pixel
    "rectangles" via _MIN_OCCLUSION_AREA.
    """

    objects = segment_objects(grid, background=None, connectivity=4, univalued=True)
    by_color: dict[int, list] = {}
    for obj in objects:
        by_color.setdefault(obj.dominant_color, []).append(obj)

    candidates: list[tuple[int, int]] = []
    for color, objs in by_color.items():
        total_area = 0
        max_component_area = 0
        all_solid_rectangles = True
        for obj in objs:
            _, _, height, width = obj.bbox
            if obj.size != height * width:
                all_solid_rectangles = False
                break
            total_area += obj.size
            max_component_area = max(max_component_area, obj.size)
        # Gate on the largest single component, not the summed area: many
        # scattered same-color single pixels (e.g. a checkerboard) would
        # otherwise trivially clear a summed floor without being one
        # genuine occlusion block.
        if all_solid_rectangles and max_component_area >= _MIN_OCCLUSION_AREA:
            candidates.append((color, total_area))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [color for color, _ in candidates[:_MAX_OCCLUSION_CANDIDATES]]


def _grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _candidate_symmetries(grid: Grid, hole_cells: set[Cell], min_overlap: int) -> list[SymmetryFn]:
    """Generate and pre-verify symmetry candidates.

    The symmetry axis is *not* assumed to be the geometric center -- a grid
    can be a window onto a larger symmetric pattern, so the true axis can
    sit anywhere. Every axis position is tried and only those consistent
    with at least `min_overlap` known (non-hole) cell pairs are kept, which
    both finds off-center axes and rejects coincidental few-cell matches.
    """

    height, width = _grid_shape(grid)

    row_axes = [
        axis for axis in range(2 * height - 1)
        if verify_symmetry(grid, hole_cells, _row_mirror(axis), min_overlap=min_overlap)
    ]
    col_axes = [
        axis for axis in range(2 * width - 1)
        if verify_symmetry(grid, hole_cells, _col_mirror(axis), min_overlap=min_overlap)
    ]

    symmetries: list[SymmetryFn] = [_row_mirror(axis) for axis in row_axes]
    symmetries.extend(_col_mirror(axis) for axis in col_axes)
    symmetries.extend(_rotate_180(row_axis, col_axis) for row_axis in row_axes for col_axis in col_axes)

    if height == width:
        symmetries.append(lambda cell: (cell[1], cell[0]))  # transpose
        symmetries.append(lambda cell: (width - 1 - cell[1], height - 1 - cell[0]))  # anti-transpose

    for period in range(1, width):
        symmetries.append(_make_periodic(0, period, height, width))
        symmetries.append(_make_periodic(0, -period, height, width))
    for period in range(1, height):
        symmetries.append(_make_periodic(period, 0, height, width))
        symmetries.append(_make_periodic(-period, 0, height, width))

    verified = [fn for fn in symmetries if verify_symmetry(grid, hole_cells, fn, min_overlap=min_overlap)]

    # Closure under composition: if f and g are each independently verified
    # symmetries of this grid, g(f(x)) is transitively guaranteed consistent
    # wherever both hold -- not a heuristic, just equality chaining. This is
    # what discovers compound symmetries (glide reflections, mirror axes
    # shifted by a verified translation period) without hand-listing every
    # possible symmetry family up front. Skipped when the base list is
    # already large (a highly-symmetric/background-heavy grid), since that
    # already gives plenty of coverage and O(n^2) composition would be the
    # dominant cost for no real benefit.
    if 0 < len(verified) <= _MAX_BASE_SYMMETRIES_TO_COMPOSE:
        composed = [_compose(outer, inner) for inner in verified for outer in verified]
        verified.extend(fn for fn in composed if verify_symmetry(grid, hole_cells, fn, min_overlap=min_overlap))

    return verified


def _compose(outer: SymmetryFn, inner: SymmetryFn) -> SymmetryFn:
    return lambda cell: outer(inner(cell))


def _row_mirror(axis_sum: int) -> SymmetryFn:
    return lambda cell: (axis_sum - cell[0], cell[1])


def _col_mirror(axis_sum: int) -> SymmetryFn:
    return lambda cell: (cell[0], axis_sum - cell[1])


def _rotate_180(row_axis: int, col_axis: int) -> SymmetryFn:
    return lambda cell: (row_axis - cell[0], col_axis - cell[1])


def _make_periodic(dr: int, dc: int, height: int, width: int) -> SymmetryFn:
    def shift(cell: Cell) -> Cell:
        return (cell[0] + dr, cell[1] + dc)

    return shift


def _in_bounds(cell: Cell, height: int, width: int) -> bool:
    r, c = cell
    return 0 <= r < height and 0 <= c < width


def verify_symmetry(grid: Grid, hole_cells: set[Cell], symmetry_fn: SymmetryFn, *, min_overlap: int = 1) -> bool:
    height, width = _grid_shape(grid)
    checked = 0
    for r in range(height):
        for c in range(width):
            if (r, c) in hole_cells:
                continue
            mapped = symmetry_fn((r, c))
            if not _in_bounds(mapped, height, width) or mapped in hole_cells:
                continue
            checked += 1
            if grid[r][c] != grid[mapped[0]][mapped[1]]:
                return False
    return checked >= min_overlap


def _default_min_overlap(height: int, width: int) -> int:
    return max(8, (height * width) // 10)


def repair_grid(grid: Grid, hole_cells: set[Cell]) -> Grid | None:
    """Fill hole_cells using every symmetry that verifies against the rest of the grid.

    Iterates to a fixed point since one symmetry's target may itself be an
    unresolved hole cell that a *different* symmetry (or the same one, after
    another cell resolves) can fill.
    """

    if not hole_cells:
        return [row[:] for row in grid]

    height, width = _grid_shape(grid)
    min_overlap = _default_min_overlap(height, width)
    verified = _candidate_symmetries(grid, hole_cells, min_overlap)
    if not verified:
        return None

    repaired = [row[:] for row in grid]
    remaining = set(hole_cells)
    progressed = True
    while remaining and progressed:
        progressed = False
        for cell in list(remaining):
            for fn in verified:
                mapped = fn(cell)
                if not _in_bounds(mapped, height, width) or mapped in remaining:
                    continue
                repaired[cell[0]][cell[1]] = repaired[mapped[0]][mapped[1]]
                remaining.discard(cell)
                progressed = True
                break

    return None if remaining else repaired


def hole_bbox(hole_cells: set[Cell]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in hole_cells]
    cols = [c for _, c in hole_cells]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    return top, left, bottom - top + 1, right - left + 1


def crop(grid: Grid, top: int, left: int, height: int, width: int) -> Grid:
    return [row[left : left + width] for row in grid[top : top + height]]


def hole_cells_for_color(grid: Grid, color: int) -> set[Cell]:
    return {(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == color}
