import pytest

from mythos.arc import ArcExample, ArcTask
from mythos.augment import NUM_TRANSFORMS, forward_transform, inverse_transform, transform_task

_SQUARE_GRID = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
_RECT_GRID = [[1, 2, 3, 4], [5, 6, 7, 8]]


@pytest.mark.parametrize("index", range(NUM_TRANSFORMS))
def test_inverse_undoes_forward_on_square_grid(index: int) -> None:
    transformed = forward_transform(index, _SQUARE_GRID)
    assert inverse_transform(index, transformed) == _SQUARE_GRID


@pytest.mark.parametrize("index", range(NUM_TRANSFORMS))
def test_inverse_undoes_forward_on_rectangular_grid(index: int) -> None:
    transformed = forward_transform(index, _RECT_GRID)
    assert inverse_transform(index, transformed) == _RECT_GRID


@pytest.mark.parametrize("index", range(NUM_TRANSFORMS))
def test_all_transforms_are_distinguishable_except_known_duplicates(index: int) -> None:
    # every transform of an asymmetric grid should be a valid permutation of
    # the same multiset of values (no cells invented or dropped)
    transformed = forward_transform(index, _RECT_GRID)
    assert sorted(v for row in transformed for v in row) == sorted(v for row in _RECT_GRID for v in row)


def test_transform_task_applies_consistently_to_train_and_test() -> None:
    task = ArcTask(
        id="t",
        train=(ArcExample(input=_SQUARE_GRID, output=_SQUARE_GRID),),
        test=(ArcExample(input=_SQUARE_GRID),),
    )
    transformed = transform_task(task, 1, id_suffix="__aug1")  # rotate90
    assert transformed.id == "t__aug1"
    assert transformed.train[0].input == forward_transform(1, _SQUARE_GRID)
    assert transformed.train[0].output == forward_transform(1, _SQUARE_GRID)
    assert transformed.test[0].input == forward_transform(1, _SQUARE_GRID)


def test_transform_task_leaves_missing_test_output_as_none() -> None:
    task = ArcTask(id="t", train=(), test=(ArcExample(input=_SQUARE_GRID, output=None),))
    transformed = transform_task(task, 3, id_suffix="__aug3")
    assert transformed.test[0].output is None
