"""Solver factory shared by CLIs."""

from __future__ import annotations

import os

from mythos.solvers.baseline import BaselineSolver
from mythos.solvers.chain import ChainSolver
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
    if name == "chain":
        # Local import for the same reason as "agentic_repl" above: selecting
        # "chain" pulls in llama-cpp-python transitively, but every other
        # solver name must stay usable without it installed.
        from agentic_repl.llm.client import LlamaCppClient
        from agentic_repl.solver import AgenticReplSolver

        # Order: SymbolicSolver and AgenticReplSolver both only ever return a
        # prediction that's been verified to exactly reproduce every train
        # pair (SolverError otherwise), so they run first -- whichever fires
        # first is a free, provably-correct-on-the-demos win. FixtureSolver
        # is the same kind of verified-or-raise solver (a smaller rigid-
        # transform search), so it comes next. PlannedPipelineSolver runs
        # last deliberately, not third as a naive reading of "wire in the
        # better system" might suggest: in this repo's default config (no
        # MYTHOS_ENABLE_REAL_HRM/MYTHOS_ENABLE_TTT/etc. set), every stage in
        # PlannedPipeline's pipeline.py falls back to BaselineSolver's
        # unverified trivial guess *unconditionally* -- strict_models only
        # changes behavior when a model is enabled but then fails, never
        # when it's simply disabled -- so PlannedPipelineSolver never raises
        # SolverError and would otherwise always short-circuit the chain
        # before FixtureSolver's better, verified guess ever got a turn.
        # Placing it last instead makes it exactly the guaranteed-output
        # terminal fallback ChainSolver's docstring calls for, while still
        # giving it a real chance to contribute genuine model predictions
        # ahead of nothing (i.e. it's still in the chain) on a real Kaggle
        # run where those models are actually loaded.
        solvers: list = [SymbolicSolver()]
        try:
            # LlamaCppClient() loads the real GGUF model eagerly in its own
            # __init__ (see agentic_repl/llm/client.py) and raises
            # RuntimeError if llama-cpp-python isn't installed or no model
            # path is configured -- both true outside a Kaggle session with
            # the model staged. "chain" is configs/base.json's default
            # solver, so it must degrade gracefully in that case (dev
            # machines, CI) rather than making every caller unable to
            # construct the default solver at all; the real agentic_repl
            # step is simply skipped from the chain when its dependency
            # isn't available; every solver still in the chain only ever
            # returns a train-pair-verified prediction, so nothing about
            # correctness is weakened by leaving it out.
            solvers.append(AgenticReplSolver(LlamaCppClient()))
        except RuntimeError:
            pass
        solvers.append(FixtureSolver())
        solvers.append(PlannedPipelineSolver(strict_models=strict_models))
        return ChainSolver(solvers)
    raise ValueError(f"unknown solver: {name}")
