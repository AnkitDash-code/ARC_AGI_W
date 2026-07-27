"""Builds the LLM-facing DSL reference text by introspecting the real callables.

Generating this from inspect.signature()/inspect.getdoc() over
agentic_repl.dsl.DSL_FUNCTIONS/DSL_CLASSES (rather than hand-writing a
separate description) means the prompt catalog can never drift from what
the sandboxed exec() namespace in agentic_repl.repl actually exposes.
"""

from __future__ import annotations

import inspect

from agentic_repl.dsl import DSL_CLASSES, DSL_FUNCTIONS


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _function_entry(fn: object) -> str:
    signature = inspect.signature(fn)  # type: ignore[arg-type]
    summary = _first_line(inspect.getdoc(fn))
    header = f"{fn.__name__}{signature}"  # type: ignore[attr-defined]
    return f"{header}\n    {summary}" if summary else header


def _class_entry(cls: type) -> str:
    lines = [f"class {cls.__name__}:"]
    class_summary = _first_line(inspect.getdoc(cls))
    if class_summary:
        lines.append(f"    {class_summary}")
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or not isinstance(member, property):
            continue
        summary = _first_line(inspect.getdoc(member))
        lines.append(f"    .{name}" + (f"  -- {summary}" if summary else ""))
    return "\n".join(lines)


def build_catalog_text() -> str:
    """Return the DSL reference text to inject into the code-generation prompt."""

    sections = ["Available DSL primitives (already imported into scope, call directly):", ""]
    for cls in DSL_CLASSES:
        sections.append(_class_entry(cls))
        sections.append("")
    for fn in DSL_FUNCTIONS:
        sections.append(_function_entry(fn))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"
