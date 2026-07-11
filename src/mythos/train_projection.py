"""CLI for training the JEPA-to-HRM projection checkpoint."""

from __future__ import annotations

import argparse
import json
import sys

from mythos.arc import ArcValidationError, attach_solutions, load_challenges, load_solutions
from mythos.training import JepaProjectionConfig, train_jepa_projection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Mythos I-JEPA projection checkpoint.")
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=1280)
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--device")
    parser.add_argument("--include-test-solutions", action="store_true")
    args = parser.parse_args(argv)

    try:
        tasks = load_challenges(args.challenges)
        if args.solutions:
            tasks = attach_solutions(tasks, load_solutions(args.solutions))
        result = train_jepa_projection(
            tasks.values(),
            checkpoint_path=args.out,
            config=JepaProjectionConfig(input_dim=args.input_dim, output_dim=args.output_dim),
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            include_test_solutions=args.include_test_solutions,
        )
    except (ArcValidationError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
