"""Augmentation + majority voting: turn a pool of verified programs into the
two grids for attempt_1/attempt_2.

Every verified program already reproduces every train pair exactly (see
solver.py's _verify_on_train) -- this stage isn't re-checking correctness,
it's a robustness/agreement signal, mirroring mythos.augment's existing
inference-time-ensembling rationale (see src/mythos/augment.py's module
docstring) applied to program *outputs* instead of model logits. There is no
majority-voting/candidate-ranking logic anywhere in mythos.score, so it has
to live here, before make_prediction ever sees the two chosen grids.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from mythos.arc import ArcTask, Grid
from mythos.augment import NUM_TRANSFORMS, forward_transform, inverse_transform
from mythos.solvers.base import SolverError

from agentic_repl.repl import ExecutionResult

RunCandidate = Callable[..., ExecutionResult]
GridKey = tuple[tuple[int, ...], ...]


def _grid_key(grid: Grid) -> GridKey:
    return tuple(tuple(row) for row in grid)


def _predict_one(
    code: str,
    input_grid: Grid,
    *,
    run_candidate: RunCandidate,
    timeout_s: float,
) -> Grid | None:
    """Run one verified program across all D4 views of input_grid and self-vote.

    Reverse-transforming each augmented view's output back to the original
    frame and voting across them catches programs that are only *coincidentally*
    correct on train in one orientation -- a genuine rule agrees with itself
    across every view once un-rotated; a lucky one usually doesn't.
    """

    votes: Counter[GridKey] = Counter()
    grids_by_key: dict[GridKey, Grid] = {}
    for transform_index in range(NUM_TRANSFORMS):
        augmented_input = forward_transform(transform_index, input_grid)
        result = run_candidate(code, augmented_input, timeout_s=timeout_s)
        if not result.ok or result.output is None:
            continue
        try:
            restored = inverse_transform(transform_index, result.output)
        except Exception:  # noqa: BLE001 - a malformed candidate output shouldn't crash voting
            continue
        key = _grid_key(restored)
        votes[key] += 1
        grids_by_key[key] = restored

    if not votes:
        return None
    best_key, _ = votes.most_common(1)[0]
    return grids_by_key[best_key]


def vote_predictions(
    verified_codes: list[str],
    task: ArcTask,
    *,
    run_candidate: RunCandidate,
    timeout_s: float,
) -> list[tuple[Grid, Grid]]:
    """Return one (attempt_1, attempt_2) pair per task.test item.

    For each test item, every verified program casts one (augmentation
    self-voted) prediction; those per-program predictions are then
    majority-voted across programs. The top two distinct results become
    attempt_1/attempt_2 (attempt_2 repeats attempt_1 if only one distinct
    grid was produced at all).
    """

    attempts: list[tuple[Grid, Grid]] = []
    for test_example in task.test:
        program_votes: Counter[GridKey] = Counter()
        grids_by_key: dict[GridKey, Grid] = {}
        for code in verified_codes:
            predicted = _predict_one(code, test_example.input, run_candidate=run_candidate, timeout_s=timeout_s)
            if predicted is None:
                continue
            key = _grid_key(predicted)
            program_votes[key] += 1
            grids_by_key[key] = predicted

        ranked = program_votes.most_common(2)
        if not ranked:
            raise SolverError(f"{task.id}: no verified program produced output for a test item")
        attempt_1 = grids_by_key[ranked[0][0]]
        attempt_2 = grids_by_key[ranked[1][0]] if len(ranked) > 1 else attempt_1
        attempts.append((attempt_1, attempt_2))
    return attempts
