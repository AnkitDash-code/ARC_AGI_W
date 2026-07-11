from __future__ import annotations

from pathlib import Path

from mythos.arc import load_challenges
from mythos.text_reasoning import format_arc_rule_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_arc_rule_prompt_contains_train_and_test_grids() -> None:
    tasks = load_challenges(ROOT / "data" / "toy" / "challenges.json")

    prompt = format_arc_rule_prompt(tasks["toy_identity"])

    assert "Train 0 input" in prompt
    assert "Train 0 output" in prompt
    assert "Test 0 input" in prompt
    assert prompt.rstrip().endswith("Rule:")
