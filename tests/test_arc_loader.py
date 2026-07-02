from __future__ import annotations

import json
from pathlib import Path

import pytest

from mythos.arc import ArcValidationError, load_challenges


ROOT = Path(__file__).resolve().parents[1]


def test_load_toy_challenges() -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")

    assert set(tasks) == {
        "toy_identity",
        "toy_recolor",
        "toy_mirror",
        "toy_rotate",
        "toy_translate",
    }
    assert len(tasks["toy_identity"].train) == 1
    assert len(tasks["toy_identity"].test) == 1


def test_rejects_grid_that_is_too_large(tmp_path: Path) -> None:
    bad = {
        "bad": {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0] for _ in range(31)]}],
        }
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ArcValidationError, match="max is 30"):
        load_challenges(path)


def test_rejects_invalid_cell_value(tmp_path: Path) -> None:
    bad = {
        "bad": {
            "train": [{"input": [[10]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ArcValidationError, match="expected 0..9"):
        load_challenges(path)
