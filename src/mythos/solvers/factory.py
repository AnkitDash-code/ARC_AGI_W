"""Solver factory shared by CLIs."""

from __future__ import annotations

import os

from mythos.solvers.baseline import BaselineSolver
from mythos.solvers.fixture import FixtureSolver
from mythos.solvers.hrm import HRMSolver
from mythos.solvers.pipeline import PlannedPipelineSolver
from mythos.solvers.symbolic import SymbolicSolver


def make_solver(name: str, *, model_mode: str | None = None):
    selected_mode = model_mode or os.environ.get("MYTHOS_MODEL_MODE", "fallback")
    if selected_mode not in {"fallback", "strict"}:
        raise ValueError(f"unknown model mode: {selected_mode}")
    strict_models = selected_mode == "strict"
    if name == "pipeline":
        return PlannedPipelineSolver(strict_models=strict_models)
    if name == "baseline":
        return BaselineSolver()
    if name == "fixture":
        return FixtureSolver()
    if name == "symbolic":
        return SymbolicSolver()
    if name == "hrm":
        return HRMSolver()
    if name == "agentic_repl":
        # Local import: agentic_repl/ is a separate top-level package (kept
        # apart from the shelved neural path here in src/mythos), and its
        # real LLM backend needs llama-cpp-python -- selecting any other
        # solver should never require that dependency to be installed.
        from agentic_repl.llm.client import LlamaCppClient
        from agentic_repl.solver import AgenticReplSolver

        return AgenticReplSolver(LlamaCppClient())
    raise ValueError(f"unknown solver: {name}")
