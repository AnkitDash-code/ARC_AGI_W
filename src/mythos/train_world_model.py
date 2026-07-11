"""CLI for training the Mythos world-model transition checkpoint."""

from __future__ import annotations

import argparse
import json
import sys

from mythos.arc import ArcValidationError, attach_solutions, load_challenges, load_solutions
from mythos.training import WorldModelConfig, train_world_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Mythos world-model checkpoint.")
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--z-dim", type=int, default=768)
    parser.add_argument("--rule-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--device")
    parser.add_argument("--include-test-solutions", action="store_true")
    args = parser.parse_args(argv)

    try:
        tasks = load_challenges(args.challenges)
        if args.solutions:
            tasks = attach_solutions(tasks, load_solutions(args.solutions))
        result = train_world_model(
            tasks.values(),
            checkpoint_path=args.out,
            config=WorldModelConfig(
                z_dim=args.z_dim,
                rule_dim=args.rule_dim,
                hidden_dim=args.hidden_dim,
            ),
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
