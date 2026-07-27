"""Combine per-task files from a cloned arcprize/ARC-AGI-2 repo into the
combined challenges.json + solutions.json format mythos.arc.load_challenges/
load_solutions expect (matching the toy fixtures under data/toy/).

The public arcprize/ARC-AGI-2 repo's data/{training,evaluation}/*.json files
each already include test outputs (unlike a real competition challenges.json,
which strips them) -- this script splits each task into the two files.

Usage:
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git <src>
    python scripts/convert_arc_agi_2_data.py --src <src> --split training --out-dir <out>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(src_dir: Path, split: str, out_dir: Path) -> tuple[int, int]:
    split_dir = src_dir / "data" / split
    if not split_dir.is_dir():
        raise SystemExit(f"{split_dir} does not exist -- check --src points at a cloned ARC-AGI-2 repo")

    challenges: dict[str, dict] = {}
    solutions: dict[str, list] = {}
    for task_file in sorted(split_dir.glob("*.json")):
        task_id = task_file.stem
        raw = json.loads(task_file.read_text(encoding="utf-8"))
        challenges[task_id] = {
            "train": raw["train"],
            "test": [{"input": item["input"]} for item in raw["test"]],
        }
        solutions[task_id] = [item["output"] for item in raw["test"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    challenges_path = out_dir / f"{split}_challenges.json"
    solutions_path = out_dir / f"{split}_solutions.json"
    challenges_path.write_text(json.dumps(challenges), encoding="utf-8")
    solutions_path.write_text(json.dumps(solutions), encoding="utf-8")
    return len(challenges), 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to a cloned arcprize/ARC-AGI-2 repo.")
    parser.add_argument("--split", choices=["training", "evaluation"], required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    count, _ = convert(Path(args.src), args.split, Path(args.out_dir))
    print(f"Converted {count} {args.split} tasks -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
