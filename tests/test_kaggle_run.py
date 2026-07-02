from __future__ import annotations

import shutil
from pathlib import Path

from mythos.kaggle_run import main, resolve_challenge_path
from mythos.submission import load_submission


ROOT = Path(__file__).resolve().parents[1]


def test_resolves_hyphenated_test_challenge_name(tmp_path: Path) -> None:
    challenge_path = tmp_path / "arc-agi_test-challenges.json"
    shutil.copy(ROOT / "data" / "toy" / "challenges.json", challenge_path)

    assert resolve_challenge_path(tmp_path, "test") == challenge_path


def test_kaggle_runner_writes_submission_from_explicit_challenges(tmp_path: Path) -> None:
    output_path = tmp_path / "submission.json"

    exit_code = main(
        [
            "--challenges",
            str(ROOT / "data" / "toy" / "challenges.json"),
            "--solver",
            "baseline",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    submission = load_submission(output_path)
    assert len(submission) == 5
