"""Grid geometric augmentation: the 8 dihedral-group transforms and their inverses.

Used for inference-time ensembling: solving a task in several different
orientations and voting across the un-augmented predictions squeezes more
accuracy out of an already-adapted model, independent of TTT quality.
Transforming demo pairs and the test input *together* (not just the test
input alone) keeps any orientation-dependent rule internally consistent
within its own augmented frame -- TTT refits on the transformed demo pairs,
so it learns the same relative rule, just viewed from a different frame.
"""

from __future__ import annotations

from typing import Callable

from mythos.arc import ArcExample, ArcTask, Grid

GridTransform = Callable[[Grid], Grid]


def _identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def _rotate180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def _rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def _mirror_horizontal(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def _mirror_vertical(grid: Grid) -> Grid:
    return [row[:] for row in grid[::-1]]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def _anti_transpose(grid: Grid) -> Grid:
    return _rotate180(_transpose(grid))


# index -> (name, forward, inverse). rotate90/rotate270 invert each other;
# every other transform is its own inverse (an involution).
_TRANSFORMS: tuple[tuple[str, GridTransform, GridTransform], ...] = (
    ("identity", _identity, _identity),
    ("rotate90", _rotate90, _rotate270),
    ("rotate180", _rotate180, _rotate180),
    ("rotate270", _rotate270, _rotate90),
    ("mirror_horizontal", _mirror_horizontal, _mirror_horizontal),
    ("mirror_vertical", _mirror_vertical, _mirror_vertical),
    ("transpose", _transpose, _transpose),
    ("anti_transpose", _anti_transpose, _anti_transpose),
)

NUM_TRANSFORMS = len(_TRANSFORMS)


def transform_name(index: int) -> str:
    return _TRANSFORMS[index][0]


def forward_transform(index: int, grid: Grid) -> Grid:
    return _TRANSFORMS[index][1](grid)


def inverse_transform(index: int, grid: Grid) -> Grid:
    return _TRANSFORMS[index][2](grid)


def transform_task(task: ArcTask, index: int, *, id_suffix: str) -> ArcTask:
    """Apply one dihedral transform to every grid in a task (demos and test).

    Both the input and output of every train example are transformed the
    same way, so any transformation rule the task encodes remains valid
    (and re-learnable via TTT) within the transformed frame.
    """

    forward = _TRANSFORMS[index][1]

    def transform_example(example: ArcExample) -> ArcExample:
        return ArcExample(
            input=forward(example.input),
            output=forward(example.output) if example.output is not None else None,
        )

    return ArcTask(
        id=f"{task.id}{id_suffix}",
        train=tuple(transform_example(example) for example in task.train),
        test=tuple(transform_example(example) for example in task.test),
    )
