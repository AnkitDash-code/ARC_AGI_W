"""CLI for validating LoRA-only test-time-training mechanics."""

from __future__ import annotations

import argparse
import json
import sys

from mythos.training import run_ttt_lora_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Mythos LoRA/TTT smoke test.")
    parser.add_argument("--out")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    try:
        result = run_ttt_lora_smoke(
            rank=args.rank,
            steps=args.steps,
            dim=args.dim,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            checkpoint_path=args.out,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
