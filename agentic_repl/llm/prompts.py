"""Prompt templates for the agentic-REPL code-generation loop."""

from __future__ import annotations

from mythos.arc import ArcTask, Grid


def render_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_initial_prompt(task: ArcTask, dsl_catalog: str) -> str:
    examples = []
    for index, example in enumerate(task.train):
        assert example.output is not None  # train examples always have outputs
        examples.append(
            f"Example {index + 1} input:\n{render_grid(example.input)}\n\n"
            f"Example {index + 1} output:\n{render_grid(example.output)}\n"
        )
    examples_text = "\n".join(examples)
    return (
        "You are solving an ARC-AGI grid transformation puzzle. Write a Python "
        "function `solve(grid)` that takes a grid (a list of lists of ints, 0-9) "
        "and returns the transformed grid, reproducing the rule shown by these "
        "training examples exactly.\n\n"
        f"{dsl_catalog}\n"
        f"{examples_text}\n"
        "Respond with ONLY a Python code block defining `solve(grid)`. Do not "
        "import anything -- the DSL primitives above are already in scope.\n"
    )


def build_refinement_prompt(
    task: ArcTask, dsl_catalog: str, previous_code: str, failure_report: str
) -> str:
    """Re-includes the full task (not just the failure text).

    Each LLMClient.generate() call is a fresh, stateless completion, not a
    multi-turn conversation -- without re-rendering the actual train
    examples here, the model has no way to see what it got wrong beyond the
    failure summary, and can't "remember" example dimensions/content from
    the now-discarded initial-prompt turn. Confirmed as a real bug via a
    real benchmark run: refinement rounds kept regenerating the same class
    of error (e.g. wrong output shape) instead of converging.
    """

    examples = []
    for index, example in enumerate(task.train):
        assert example.output is not None
        examples.append(
            f"Example {index + 1} input:\n{render_grid(example.input)}\n\n"
            f"Example {index + 1} output:\n{render_grid(example.output)}\n"
        )
    examples_text = "\n".join(examples)
    return (
        "You are solving an ARC-AGI grid transformation puzzle. Your previous "
        "solve(grid) candidate did not reproduce every training example exactly. "
        "Fix it.\n\n"
        f"{dsl_catalog}\n"
        f"{examples_text}\n"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"Failure report:\n{failure_report}\n\n"
        "Respond with ONLY a corrected Python code block defining `solve(grid)`.\n"
    )


def extract_code_block(completion: str) -> str:
    """Pull the first ```-fenced code block out of a completion, else return as-is."""

    if "```" not in completion:
        return completion.strip()
    fence_parts = completion.split("```")
    if len(fence_parts) < 2:
        return completion.strip()
    block = fence_parts[1]
    if block.lstrip().lower().startswith("python"):
        block = block.lstrip()[len("python"):]
    return block.strip()
