"""Regression tests for HRM output-shape inference.

`output_shape_hint` used to fall back to the *input* grid's shape whenever
train examples disagreed on output shape (e.g. crop/symmetry-repair tasks
where the output size is the bounding box of an occluded region and varies
per example). That made exact-match scoring impossible for that whole task
family regardless of prediction quality, since `hrm_sequence_to_grid` would
then decode a full 30x30 grid instead of the small true-sized patch.
"""

from mythos.arc import ArcExample, ArcTask
from mythos.features import ARC_MAX_SIZE, hrm_sequence_to_grid, output_shape_hint


def _task_with_train_outputs(*shapes: tuple[int, int]) -> ArcTask:
    train = tuple(
        ArcExample(input=[[0] * 30 for _ in range(30)], output=[[0] * width for _ in range(height)])
        for height, width in shapes
    )
    test = (ArcExample(input=[[0] * 30 for _ in range(30)]),)
    return ArcTask(id="t", train=train, test=test)


def test_output_shape_hint_returns_shape_when_train_examples_agree() -> None:
    task = _task_with_train_outputs((4, 4), (4, 4))
    assert output_shape_hint(task, task.train[0].input) == (4, 4)


def test_output_shape_hint_returns_none_when_train_shapes_disagree() -> None:
    task = _task_with_train_outputs((9, 4), (4, 5), (3, 7), (4, 4))
    assert output_shape_hint(task, task.train[0].input) is None


def test_hrm_sequence_to_grid_uses_eos_markers_when_shape_hint_is_none() -> None:
    matrix = [[0] * ARC_MAX_SIZE for _ in range(ARC_MAX_SIZE)]
    for row in range(9):
        matrix[row][3] = 1  # EOS column marker at width=3
    for col in range(3):
        matrix[9][col] = 1  # EOS row marker at height=9
    for row in range(9):
        for col in range(3):
            matrix[row][col] = 5  # color token (5 - 2 = color 3)
    sequence = tuple(token for row in matrix for token in row)

    grid = hrm_sequence_to_grid(sequence, shape_hint=None)

    assert len(grid) == 9
    assert len(grid[0]) == 3


def test_hrm_sequence_to_grid_ignores_early_noise_and_finds_the_real_boundary() -> None:
    # Reproduces the v44 regression: real HRM output has an orthogonal EOS
    # marker baked into every in-grid row/column (one stray 1 each), plus
    # occasional prediction noise. A "first row/col with >=2 EOS tokens"
    # heuristic latches onto row/col 0 here and collapses to a 1x1 grid;
    # the genuine boundary (height=9, width=3) carries far more EOS tokens
    # and must win instead.
    matrix = [[0] * ARC_MAX_SIZE for _ in range(ARC_MAX_SIZE)]
    for row in range(9):
        matrix[row][3] = 1  # legitimate orthogonal EOS-column marker
    for col in range(3):
        matrix[9][col] = 1  # genuine EOS row boundary (score=3)
    matrix[0][5] = 1  # single unit of noise on row 0 -> row_scores[0] == 2
    matrix[7][8] = 1  # single unit of noise on col 8 -> col_scores[8] == 1 (still below threshold)
    matrix[2][1] = 1  # noise pushes row 2's score to 2 as well (still below row 9's genuine score of 3)

    grid = hrm_sequence_to_grid(tuple(token for row in matrix for token in row), shape_hint=None)

    assert len(grid) == 9
    assert len(grid[0]) == 3


def test_hrm_sequence_to_grid_falls_back_to_full_size_without_eos_markers() -> None:
    sequence = tuple([2] * (ARC_MAX_SIZE * ARC_MAX_SIZE))  # no EOS(=1) tokens anywhere

    grid = hrm_sequence_to_grid(sequence, shape_hint=None)

    assert len(grid) == ARC_MAX_SIZE
    assert len(grid[0]) == ARC_MAX_SIZE
