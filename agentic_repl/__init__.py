"""Agentic program-synthesis & REPL-refinement ARC solver.

Kept as a top-level package, separate from the shelved src/mythos neural
path (HRM/TTT/LoRA/CompressARC), but built on top of mythos's data model
(ArcTask/Grid) and solver contract (mythos.solvers.base.Solver) and reusing
mythos's verified grid-primitive modules (objects/object_ops/symmetry/
augment) as the DSL surface offered to a code-generating LLM.
"""
