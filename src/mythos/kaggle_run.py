"""Kaggle-oriented runner for producing /kaggle/working/submission.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from mythos.arc import ArcTask, ArcValidationError, copy_grid, load_challenges
from mythos.metrics import score_files
from mythos.solvers.base import SolverError, make_prediction
from mythos.solvers.baseline import BaselineSolver
from mythos.solvers.factory import make_solver
from mythos.solvers.hrm import HRMEnvironment, HRMInferenceRunner, HRMTTTRunner, TTTConfig
from mythos.submission import Prediction, write_submission

DEFAULT_KAGGLE_DATA_DIR = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
DEFAULT_KAGGLE_OUTPUT = Path("/kaggle/working/submission.json")

CHALLENGE_FILES = {
    "training": ("arc-agi_training-challenges.json", "arc-agi_training_challenges.json"),
    "evaluation": ("arc-agi_evaluation-challenges.json", "arc-agi_evaluation_challenges.json"),
    "test": ("arc-agi_test-challenges.json", "arc-agi_test_challenges.json"),
}

SOLUTION_FILES = {
    "training": ("arc-agi_training-solutions.json", "arc-agi_training_solutions.json"),
    "evaluation": ("arc-agi_evaluation-solutions.json", "arc-agi_evaluation_solutions.json"),
}


def resolve_challenge_path(data_dir: str | Path, split: str) -> Path:
    return _first_existing(Path(data_dir), CHALLENGE_FILES[split])


def resolve_solution_path(data_dir: str | Path, split: str) -> Path | None:
    candidates = SOLUTION_FILES.get(split)
    if not candidates:
        return None
    try:
        return _first_existing(Path(data_dir), candidates)
    except FileNotFoundError:
        return None


def _first_existing(data_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    joined = ", ".join(names)
    raise FileNotFoundError(f"none of these files exist in {data_dir}: {joined}")


def solve_with_fallback(solver, fallback_solver, task: ArcTask) -> Prediction:
    """Solve one task, degrading to guaranteed-output fallbacks on any failure.

    A Kaggle rerun must always write a submission.json even if some tasks
    crash their primary solver (model errors, OOM, malformed grids, etc.);
    letting one bad task abort the whole loop would zero out every other
    already-solved task too.
    """
    try:
        return solver.solve(task)
    except Exception as exc:  # noqa: BLE001 - any solver failure must not abort the run
        print(f"WARNING: {task.id} failed with {solver.__class__.__name__}: {exc!r}; using baseline fallback", file=sys.stderr)
    try:
        return fallback_solver.solve(task)
    except Exception as exc:  # noqa: BLE001 - last-resort guarantee of a valid prediction
        print(f"WARNING: {task.id} baseline fallback also failed: {exc!r}; using trivial prediction", file=sys.stderr)
    attempts = [(copy_grid(example.input), [[0]]) for example in task.test]
    return make_prediction(task, attempts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Mythos against Kaggle ARC-AGI data.")
    parser.add_argument("--data-dir", default=str(DEFAULT_KAGGLE_DATA_DIR))
    parser.add_argument("--split", choices=sorted(CHALLENGE_FILES), default="test")
    parser.add_argument("--challenges", help="Explicit challenge JSON path; overrides --data-dir/--split.")
    parser.add_argument("--solutions", help="Explicit solutions JSON path for local scoring.")
    parser.add_argument("--solver", choices=["pipeline", "baseline", "fixture", "hrm"], default="pipeline")
    parser.add_argument("--model-mode", choices=["fallback", "strict"], default=None)
    parser.add_argument("--out", default=str(DEFAULT_KAGGLE_OUTPUT))
    parser.add_argument("--score", action="store_true", help="Score output when solutions are available.")
    args = parser.parse_args(argv)

    try:
        challenge_path = Path(args.challenges) if args.challenges else resolve_challenge_path(args.data_dir, args.split)
        solution_path = Path(args.solutions) if args.solutions else resolve_solution_path(args.data_dir, args.split)

        tasks = load_challenges(challenge_path)
        solver = make_solver(args.solver, model_mode=args.model_mode)
        fallback_solver = solver if isinstance(solver, BaselineSolver) else BaselineSolver()
        if args.solver == "hrm":
            try:
                env = HRMEnvironment.from_env()
                env.validate(require_cuda=True)
                if os.environ.get("MYTHOS_ENABLE_TTT") == "1":
                    runner = HRMTTTRunner(
                        env,
                        ttt=TTTConfig(
                            rank=int(os.environ.get("MYTHOS_TTT_RANK", "16")),
                            steps=int(os.environ.get("MYTHOS_TTT_STEPS", "20")),
                            lr=float(os.environ.get("MYTHOS_TTT_LR", "1e-3")),
                            batch_size=int(os.environ.get("MYTHOS_TTT_BATCH_SIZE", "2")),
                            genie_weight=float(os.environ.get("MYTHOS_TTT_GENIE_WEIGHT", "0.1")),
                        ),
                        num_aug=int(os.environ.get("MYTHOS_TTT_NUM_AUG", "0")),
                    )
                else:
                    runner = HRMInferenceRunner(env)
                predictions = runner.solve_tasks(list(tasks.values()))
            except Exception as exc:  # noqa: BLE001 - HRM batch failure must not abort the run
                print(f"WARNING: HRM batch run failed: {exc!r}; using baseline fallback for all tasks", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                if getattr(exc, "stdout", None):
                    print("--- subprocess stdout (tail) ---", file=sys.stderr)
                    print(exc.stdout[-4000:], file=sys.stderr)
                if getattr(exc, "stderr", None):
                    print("--- subprocess stderr (tail) ---", file=sys.stderr)
                    print(exc.stderr[-4000:], file=sys.stderr)
                predictions = [fallback_solver.solve(task) for task in tasks.values()]
        else:
            predictions = [solve_with_fallback(solver, fallback_solver, task) for task in tasks.values()]
        write_submission(predictions, args.out)

        summary: dict[str, object] = {
            "challenge_path": str(challenge_path),
            "output_path": str(args.out),
            "solver": args.solver,
            "model_mode": args.model_mode or "fallback",
            "tasks": len(tasks),
        }
        if hasattr(solver, "pipeline"):
            summary["models"] = solver.pipeline.model_registry.summary()
        if args.score and solution_path is not None:
            summary["score"] = score_files(args.out, str(solution_path)).to_dict()
        elif args.score:
            summary["score"] = "skipped: no solutions file for this split"
    except (ArcValidationError, SolverError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
