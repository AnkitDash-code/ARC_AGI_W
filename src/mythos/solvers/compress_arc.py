"""Integration wrapper around the vendored CompressARC solver.

CompressARC (see third_party/compress_arc/NOTICE.md for provenance) trains
a small, randomly-initialized network from scratch on each task's own demo
pairs via a compression (MDL/VAE) objective -- unlike HRM, it needs no
pretrained checkpoint at all. This file is our own glue code around the
vendored, unmodified upstream implementation.

Note: `arc_compressor.py` calls `torch.set_default_device('cuda')` at
*import time*, unconditionally -- there is no CPU fallback, and importing
it changes torch's global default device for the rest of the process.
Run this solver in an isolated subprocess when combining it with any other
PyTorch-using code (e.g. the HRM solver) in the same pipeline, the same way
HRM's own dataset builder is already run out-of-process for isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

from mythos.arc import ArcTask
from mythos.solvers.base import SolverError, make_prediction
from mythos.submission import Prediction

_THIRD_PARTY_DIR = Path(__file__).resolve().parents[3] / "third_party" / "compress_arc"


def _ensure_on_path() -> None:
    path_str = str(_THIRD_PARTY_DIR)
    if _THIRD_PARTY_DIR.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclass(frozen=True)
class CompressARCConfig:
    steps: int = 2000
    # Wall-clock cutoff per task; solve_task-style early exit keeps whatever
    # the best-so-far tracked solution is. None = no limit (run all steps).
    time_limit_seconds: float | None = None


class CompressARCSolver:
    """Per-task from-scratch compression-based solver (no pretraining)."""

    def __init__(self, config: CompressARCConfig | None = None) -> None:
        self.config = config or CompressARCConfig()

    def solve(self, task: ArcTask) -> Prediction:
        _ensure_on_path()
        import torch  # local: only needed once the vendored deps are on sys.path
        if not torch.cuda.is_available():
            raise SolverError("CompressARC requires CUDA: arc_compressor.py hard-sets the default device at import time")
        import arc_compressor  # import-time side effect: torch.set_default_device('cuda') for the whole process
        import preprocessing
        import solution_selection
        import train as compress_arc_train

        problem = {
            "train": [{"input": example.input, "output": example.output} for example in task.train],
            "test": [{"input": example.input} for example in task.test],
        }
        torch.manual_seed(0)
        try:
            compress_task = preprocessing.Task(task.id, problem, None)
        except Exception as exc:  # noqa: BLE001 - a malformed/unsupported task must not abort the run
            raise SolverError(f"{task.id}: CompressARC preprocessing failed: {exc!r}") from exc

        model = arc_compressor.ARCCompressor(compress_task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        logger = solution_selection.Logger(compress_task)
        logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(compress_task.n_test))
        logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for _ in range(compress_task.n_test))

        deadline = time.time() + self.config.time_limit_seconds if self.config.time_limit_seconds else None
        for step in range(self.config.steps):
            compress_arc_train.take_step(compress_task, model, optimizer, step, logger)
            if deadline is not None and time.time() > deadline:
                break

        attempts: list[tuple] = []
        for example_index in range(compress_task.n_test):
            attempt_1 = [list(row) for row in logger.solution_most_frequent[example_index]]
            attempt_2 = [list(row) for row in logger.solution_second_most_frequent[example_index]]
            attempts.append((attempt_1, attempt_2))
        return make_prediction(task, attempts)
