"""Top-level module help for `python -m mythos`."""

from __future__ import annotations


def main() -> int:
    print(
        "Project Mythos commands:\n"
        "  python -m mythos.validate data/toy/challenges.json\n"
        "  python -m mythos.solve --solver fixture --challenges data/toy/challenges.json --out runs/submission.json\n"
        "  python -m mythos.score --pred runs/submission.json --solutions data/toy/solutions.json\n"
        "  python -m mythos.hrm_smoke --task data/toy/challenges.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
