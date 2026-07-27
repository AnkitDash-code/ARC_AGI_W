from mythos.arc import ArcExample, ArcTask
from mythos.object_ops import (
    find_crop_to_selected_object_transform,
    find_fill_enclosed_regions_transform,
    find_fragment_to_slot_transform,
)

_DONOR_SHAPE = frozenset({(0, 0), (1, 0), (1, 1)})  # an L tromino


def _rotate90(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rotated = {(c, -r) for r, c in cells}
    min_r = min(r for r, _ in rotated)
    min_c = min(c for _, c in rotated)
    return frozenset((r - min_r, c - min_c) for r, c in rotated)


def _blank(height: int, width: int) -> list[list[int]]:
    return [[0] * width for _ in range(height)]


def _paint(grid: list[list[int]], cells: frozenset[tuple[int, int]], origin: tuple[int, int], color: int) -> None:
    top, left = origin
    for r, c in cells:
        grid[top + r][left + c] = color


def _build_example(donor_origin, slot_origin, *, donor_color=5, slot_color=8, keep_donor=True):
    grid = _blank(10, 10)
    _paint(grid, _DONOR_SHAPE, donor_origin, donor_color)
    slot_shape = _rotate90(_DONOR_SHAPE)
    _paint(grid, slot_shape, slot_origin, slot_color)

    output = [row[:] for row in grid]
    # rotate90(_DONOR_SHAPE) applied again matches slot_shape's own cells one-to-one in order below
    rotated_donor_cells = _rotate90(_DONOR_SHAPE)
    top, left = slot_origin
    for r, c in rotated_donor_cells:
        output[top + r][left + c] = donor_color
    if not keep_donor:
        for r, c in _DONOR_SHAPE:
            output[donor_origin[0] + r][donor_origin[1] + c] = 0

    return grid, output


def test_fragment_to_slot_pastes_rotated_donor_and_removes_it() -> None:
    grid1, output1 = _build_example((1, 1), (6, 6), keep_donor=False)
    grid2, output2 = _build_example((2, 5), (7, 1), keep_donor=False)
    task = ArcTask(
        id="t",
        train=(
            ArcExample(input=grid1, output=output1),
            ArcExample(input=grid2, output=output2),
        ),
        test=(ArcExample(input=grid1),),
    )
    transform = find_fragment_to_slot_transform(task)
    assert transform is not None
    assert transform(grid1) == output1


def test_fragment_to_slot_can_keep_donor_in_place() -> None:
    grid1, output1 = _build_example((1, 1), (6, 6), keep_donor=True)
    grid2, output2 = _build_example((0, 4), (7, 0), keep_donor=True)
    task = ArcTask(
        id="t",
        train=(
            ArcExample(input=grid1, output=output1),
            ArcExample(input=grid2, output=output2),
        ),
        test=(ArcExample(input=grid1),),
    )
    transform = find_fragment_to_slot_transform(task)
    assert transform is not None
    assert transform(grid1) == output1


def test_crop_to_largest_object() -> None:
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
    ]
    task = ArcTask(
        id="t",
        train=(ArcExample(input=grid, output=[[1], [1]]),),
        test=(ArcExample(input=grid),),
    )
    transform = find_crop_to_selected_object_transform(task)
    assert transform is not None
    assert transform(grid) == [[1], [1]]


def test_fill_enclosed_regions_with_fixed_color() -> None:
    grid = [
        [3, 3, 3],
        [3, 0, 3],
        [3, 3, 3],
    ]
    output = [
        [3, 3, 3],
        [3, 4, 3],
        [3, 3, 3],
    ]
    task = ArcTask(
        id="t",
        train=(ArcExample(input=grid, output=output),),
        test=(ArcExample(input=grid),),
    )
    transform = find_fill_enclosed_regions_transform(task)
    assert transform is not None
    assert transform(grid) == output


def test_fill_enclosed_regions_ignores_border_touching_background() -> None:
    grid = [
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 3],
    ]
    # top-left 0 touches the border, so it must stay untouched even though
    # a fixed fill-color rule would otherwise want to paint every 0.
    output = [
        [0, 3, 3],
        [3, 4, 3],
        [3, 3, 3],
    ]
    task = ArcTask(
        id="t",
        train=(ArcExample(input=grid, output=output),),
        test=(ArcExample(input=grid),),
    )
    transform = find_fill_enclosed_regions_transform(task)
    assert transform is not None
    assert transform(grid) == output
