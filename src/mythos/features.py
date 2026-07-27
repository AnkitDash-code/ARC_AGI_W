"""Deterministic ARC feature encoders shared by training and pipeline stages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Iterator, Sequence

from mythos.arc import ArcTask, Grid

ARC_MAX_SIZE = 30
ARC_NUM_COLORS = 10
DEFAULT_JEPA_FEATURE_DIM = 1280
DEFAULT_HRM_FEATURE_DIM = 768
DEFAULT_RULE_DIM = 4


@dataclass(frozen=True)
class GridPair:
    task_id: str
    index: int
    input: Grid
    output: Grid


def iter_train_grid_pairs(tasks: Iterable[ArcTask]) -> Iterator[GridPair]:
    """Yield supervised input/output pairs from ARC train examples."""

    for task in tasks:
        for index, example in enumerate(task.train):
            if example.output is None:
                continue
            yield GridPair(
                task_id=task.id,
                index=index,
                input=example.input,
                output=example.output,
            )


def iter_all_supervised_grid_pairs(tasks: Iterable[ArcTask]) -> Iterator[GridPair]:
    """Yield train pairs plus test pairs whose labels are attached."""

    for task in tasks:
        yield from iter_train_grid_pairs((task,))
        for index, example in enumerate(task.test):
            if example.output is None:
                continue
            yield GridPair(
                task_id=task.id,
                index=index,
                input=example.input,
                output=example.output,
            )


def grid_to_feature_vector(grid: Grid, dim: int = DEFAULT_JEPA_FEATURE_DIM) -> tuple[float, ...]:
    """Encode a grid as a fixed-length numeric vector.

    This is intentionally deterministic and dependency-free. On Kaggle it acts
    as the local ARC grid-to-token bridge when the external I-JEPA model cannot
    be invoked directly, and as a stable target for projection/world-model smoke
    training.
    """

    if dim <= 0:
        raise ValueError("feature dimension must be positive")

    values = [0.0 for _ in range(dim)]
    height = len(grid)
    width = len(grid[0])
    area = float(height * width)
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)

    def add(index: int, value: float) -> None:
        if 0 <= index < dim:
            values[index] += float(value)

    add(0, height / ARC_MAX_SIZE)
    add(1, width / ARC_MAX_SIZE)
    add(2, area / float(ARC_MAX_SIZE * ARC_MAX_SIZE))
    add(3, sum(1 for cell in flat if cell != 0) / area)
    add(4, sum(flat) / (9.0 * area))

    for color in range(ARC_NUM_COLORS):
        add(8 + color, counts.get(color, 0) / area)

    row_offset = 32
    for row_index in range(ARC_MAX_SIZE):
        if row_index < height:
            row = grid[row_index]
            row_area = float(width)
            add(row_offset + row_index, sum(row) / (9.0 * row_area))
            add(row_offset + ARC_MAX_SIZE + row_index, sum(1 for cell in row if cell != 0) / row_area)

    col_offset = row_offset + 2 * ARC_MAX_SIZE
    for col_index in range(ARC_MAX_SIZE):
        if col_index < width:
            column = [grid[row_index][col_index] for row_index in range(height)]
            col_area = float(height)
            add(col_offset + col_index, sum(column) / (9.0 * col_area))
            add(col_offset + ARC_MAX_SIZE + col_index, sum(1 for cell in column if cell != 0) / col_area)

    flat_offset = col_offset + 2 * ARC_MAX_SIZE
    hash_offset = max(flat_offset, int(dim * 0.875))
    flat_capacity = max(0, min(hash_offset, dim) - flat_offset)
    for row_index in range(ARC_MAX_SIZE):
        for col_index in range(ARC_MAX_SIZE):
            flat_index = row_index * ARC_MAX_SIZE + col_index
            if flat_index >= flat_capacity:
                break
            target_index = flat_offset + flat_index
            if row_index < height and col_index < width:
                add(target_index, (grid[row_index][col_index] + 1) / 10.0)
            else:
                add(target_index, 0.0)

    hash_dim = dim - hash_offset
    if hash_dim > 0:
        for row_index, row in enumerate(grid):
            for col_index, color in enumerate(row):
                base = row_index * 1009 + col_index * 917 + color * 613
                magnitude = ((color + 1) / 10.0) * (1.0 + (row_index + col_index) / 60.0)
                for salt in range(4):
                    slot = hash_offset + ((base + salt * 193) % hash_dim)
                    sign = 1.0 if ((base + salt * 389) % 2 == 0) else -1.0
                    add(slot, sign * magnitude)

    return _l2_normalize(values)


def task_rule_vector(task: ArcTask, dim: int = DEFAULT_RULE_DIM) -> tuple[float, ...]:
    """Summarize the demonstrated transformation as a fixed-length rule vector."""

    if dim <= 0:
        raise ValueError("rule dimension must be positive")

    shape_deltas: list[tuple[int, int]] = []
    density_deltas: list[float] = []
    color_overlaps: list[float] = []
    color_shift_votes: list[float] = []

    for example in task.train:
        if example.output is None:
            continue
        in_h, in_w = len(example.input), len(example.input[0])
        out_h, out_w = len(example.output), len(example.output[0])
        shape_deltas.append((out_h - in_h, out_w - in_w))
        density_deltas.append(_density(example.output) - _density(example.input))
        color_overlaps.append(_color_jaccard(example.input, example.output))
        color_shift_votes.append(_dominant_color(example.output) - _dominant_color(example.input))

    base = [
        _average(delta[0] for delta in shape_deltas) / ARC_MAX_SIZE,
        _average(delta[1] for delta in shape_deltas) / ARC_MAX_SIZE,
        _average(density_deltas),
        _average(color_overlaps),
        _average(color_shift_votes) / 9.0,
    ]
    if dim <= len(base):
        return tuple(round(value, 6) for value in base[:dim])

    values = [0.0 for _ in range(dim)]
    for index, value in enumerate(base):
        values[index] = value
    for pair_index, example in enumerate(task.train):
        if example.output is None:
            continue
        for grid_index, grid in enumerate((example.input, example.output)):
            for color, count in Counter(cell for row in grid for cell in row).items():
                slot = len(base) + ((pair_index * 97 + grid_index * 31 + color * 17) % (dim - len(base)))
                values[slot] += count / 900.0
    return tuple(round(value, 6) for value in values)


def grid_to_hrm_sequence(grid: Grid, *, max_size: int = ARC_MAX_SIZE) -> tuple[int, ...]:
    """Encode a grid using HRM ARC token conventions: PAD=0, EOS=1, colors=2..11."""

    height = len(grid)
    width = len(grid[0])
    if height > max_size or width > max_size:
        raise ValueError(f"grid shape {height}x{width} exceeds {max_size}x{max_size}")
    tokens = [[0 for _ in range(max_size)] for _ in range(max_size)]
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            tokens[row_index][col_index] = color + 2
    if height < max_size:
        for col_index in range(width):
            tokens[height][col_index] = 1
    if width < max_size:
        for row_index in range(height):
            tokens[row_index][width] = 1
    return tuple(token for row in tokens for token in row)


def hrm_sequence_to_grid(
    sequence: Sequence[int],
    *,
    shape_hint: tuple[int, int] | None = None,
    max_size: int = ARC_MAX_SIZE,
) -> Grid:
    """Decode a HRM ARC token sequence into a valid 0..9 grid."""

    if len(sequence) != max_size * max_size:
        raise ValueError(f"expected {max_size * max_size} HRM tokens, got {len(sequence)}")
    matrix = [
        [int(sequence[row * max_size + col]) for col in range(max_size)]
        for row in range(max_size)
    ]
    if shape_hint is None:
        shape_hint = _infer_shape_from_eos(matrix, max_size=max_size)
    height = min(max(1, shape_hint[0]), max_size)
    width = min(max(1, shape_hint[1]), max_size)
    return [
        [_token_to_color(matrix[row][col]) for col in range(width)]
        for row in range(height)
    ]


def output_shape_hint(task: ArcTask, input_grid: Grid) -> tuple[int, int] | None:
    """Choose an output shape only when train examples agree on one.

    When train output shapes disagree (common for crop/symmetry-repair
    tasks, where the output size is the bounding box of some occluded
    region and varies per example), there is no safe static guess -- return
    None so the caller falls back to the model's own predicted EOS boundary
    markers (see `_infer_shape_from_eos`) instead of forcing the full input
    grid's shape, which is virtually never correct for this task family and
    makes exact-match scoring impossible regardless of prediction quality.
    """

    shapes = {
        (len(example.output), len(example.output[0]))
        for example in task.train
        if example.output is not None
    }
    if len(shapes) == 1:
        return next(iter(shapes))
    return None


def embedding_cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def max_pairwise_cosine(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    max_value = -1.0
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            max_value = max(max_value, embedding_cosine_similarity(left, right))
    return max_value


def _infer_shape_from_eos(matrix: Sequence[Sequence[int]], *, max_size: int) -> tuple[int, int]:
    row_scores = [
        sum(1 for value in matrix[row_index] if value == 1)
        for row_index in range(max_size)
    ]
    col_scores = [
        sum(1 for row_index in range(max_size) if matrix[row_index][col_index] == 1)
        for col_index in range(max_size)
    ]
    height = _best_boundary_index(row_scores, max_size)
    width = _best_boundary_index(col_scores, max_size)
    return max(1, height), max(1, width)


def _best_boundary_index(scores: Sequence[int], default: int, *, min_score: int = 2) -> int:
    """Pick the index most likely to be the EOS boundary marker.

    Per `grid_to_hrm_sequence`, every in-grid row/column already carries
    exactly one legitimate stray EOS(=1) token from the *orthogonal*
    boundary marker (the EOS column marks every content row; the EOS row
    marks every content column). A low fixed threshold is therefore
    trivially false-triggered by that baseline plus a single unit of
    prediction noise on some earlier row/column. The genuine boundary
    instead carries a full run of EOS tokens -- one per in-grid cell along
    that axis -- so picking the argmax (not the first index crossing a low
    threshold) is far more robust to noisy raw predictions.
    """

    best_index = default
    best_score = min_score - 1
    for index, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= min_score else default


def _token_to_color(token: int) -> int:
    if token <= 1:
        return 0
    return min(max(token - 2, 0), 9)


def _density(grid: Grid) -> float:
    flat = [cell for row in grid for cell in row]
    return sum(1 for cell in flat if cell != 0) / float(len(flat))


def _dominant_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def _color_jaccard(left: Grid, right: Grid) -> float:
    left_colors = {cell for row in left for cell in row}
    right_colors = {cell for row in right for cell in row}
    union = left_colors | right_colors
    if not union:
        return 1.0
    return len(left_colors & right_colors) / len(union)


def _average(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(round(value, 6) for value in values)
    return tuple(round(value / norm, 6) for value in values)
