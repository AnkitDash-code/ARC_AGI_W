"""Prompt templates for the agentic-REPL code-generation loop."""

from __future__ import annotations

from mythos.arc import ArcTask, Grid


def render_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


# Concrete usage examples for the DSL primitives, not just their signatures.
# Added after a real benchmark run (v56, 100 ARC-AGI-2 training tasks)
# showed the model reaching only for generic per-pixel loops even when a
# task was exactly the kind mythos.solvers.symbolic's hand-written transform
# finders already solve (object selection, symmetry repair) -- the catalog's
# signatures alone weren't enough to make the model reach for them instead.
_DSL_USAGE_EXAMPLES = """\
Example uses of the DSL primitives above (adapt the pattern, don't copy verbatim):

# Pattern: output is one selected object, cropped to its bounding box.
def solve(grid):
    objects = segment_objects(grid, background=0, connectivity=4, univalued=True)
    largest = max(objects, key=lambda obj: obj.size)
    return crop_to_object(grid, largest, background=0)

# Pattern: a rectangular region was painted over with one color; recover it
# from the grid's own mirror/rotational/periodic symmetry.
def solve(grid):
    occlusion_color = find_occlusion_color_candidates(grid)[0]
    holes = hole_cells_for_color(grid, occlusion_color)
    repaired = repair_grid(grid, holes)
    top, left, height, width = hole_bbox(holes)
    return crop(repaired, top, left, height, width)
"""


def build_dsl_reference(dsl_catalog: str) -> str:
    return f"{dsl_catalog}\n{_DSL_USAGE_EXAMPLES}"


def render_train_examples(task: ArcTask) -> str:
    """Shared by every prompt builder that needs the task's own train pairs
    rendered as text (initial, refinement, simplify) -- kept in one place so
    a formatting change never has to be made in more than one prompt."""

    examples = []
    for index, example in enumerate(task.train):
        assert example.output is not None  # train examples always have outputs
        examples.append(
            f"Example {index + 1} input:\n{render_grid(example.input)}\n\n"
            f"Example {index + 1} output:\n{render_grid(example.output)}\n"
        )
    return "\n".join(examples)


def build_initial_prompt(task: ArcTask, dsl_catalog: str) -> str:
    examples_text = render_train_examples(task)
    return (
        "You are solving an ARC-AGI grid transformation puzzle. Write a Python "
        "function `solve(grid)` that takes a grid (a list of lists of ints, 0-9) "
        "and returns the transformed grid, reproducing the rule shown by these "
        "training examples exactly. Prefer the DSL primitives below over "
        "hand-written pixel loops when a task looks like object selection, "
        "cropping, or symmetry repair -- they handle edge cases a from-scratch "
        "loop usually misses.\n\n"
        f"{build_dsl_reference(dsl_catalog)}\n"
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

    examples_text = render_train_examples(task)
    return (
        "You are solving an ARC-AGI grid transformation puzzle. Your previous "
        "solve(grid) candidate did not reproduce every training example exactly. "
        "Fix it -- consider whether one of the DSL primitives below handles this "
        "more robustly than a hand-written pixel loop.\n\n"
        f"{build_dsl_reference(dsl_catalog)}\n"
        f"{examples_text}\n"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"Failure report:\n{failure_report}\n\n"
        "Respond with ONLY a corrected Python code block defining `solve(grid)`.\n"
    )


def build_simplify_prompt(task: ArcTask, dsl_catalog: str, verified_code: str) -> str:
    """Ask the LLM for a simpler equivalent of an already-verified solve(grid).

    Mirrors build_refinement_prompt's structure (task examples + dsl_catalog +
    existing code), but frames the ask as simplification rather than bug-fixing:
    the existing code is already correct, and the model should look for a
    shorter/cleaner formulation using the same DSL, without changing behavior.
    """

    examples_text = render_train_examples(task)
    return (
        "You are solving an ARC-AGI grid transformation puzzle. The following "
        "solve(grid) candidate already reproduces every training example "
        "exactly -- it is correct. Do not change its behavior. Look only for a "
        "simpler, shorter equivalent: prefer the DSL primitives below over "
        "hand-written pixel loops, remove unnecessary intermediate steps, and "
        "avoid special-casing that a more general use of a primitive would "
        "cover. If the existing code is already about as simple as it can be, "
        "you may return it unchanged.\n\n"
        f"{build_dsl_reference(dsl_catalog)}\n"
        f"{examples_text}\n"
        f"Current (correct) code:\n```python\n{verified_code}\n```\n\n"
        "Respond with ONLY a Python code block defining `solve(grid)`.\n"
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
