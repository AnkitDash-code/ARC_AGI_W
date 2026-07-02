"""CLI for running a solver and writing submission JSON."""

from __future__ import annotations

import argparse
import sys

from mythos.arc import ArcValidationError, load_challenges
from mythos.solvers.base import SolverError
from mythos.solvers.factory import make_solver
from mythos.submission import write_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Mythos solver.")
    parser.add_argument("--solver", choices=["pipeline", "baseline", "fixture", "hrm"], default="pipeline")
    parser.add_argument("--model-mode", choices=["fallback", "strict"], default=None)
    parser.add_argument("--challenges", required=True, help="Path to ARC-style challenges JSON.")
    parser.add_argument("--out", required=True, help="Output submission JSON path.")
    args = parser.parse_args(argv)

    try:
        tasks = load_challenges(args.challenges)
        solver = make_solver(args.solver, model_mode=args.model_mode)
        predictions = [solver.solve(task) for task in tasks.values()]
        write_submission(predictions, args.out)
    except (ArcValidationError, SolverError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(predictions)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
