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
    raise ValueError(f"unknown solver: {name}")
