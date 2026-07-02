"""Solver wrapper for the plan-aligned Project Mythos pipeline."""

from __future__ import annotations

from mythos.arc import ArcTask
from mythos.models import ModelRegistry
from mythos.pipeline import PipelineTrace, PlannedPipeline
from mythos.submission import Prediction


class PlannedPipelineSolver:
    """Solver that runs every task through the master-plan stage boundaries."""

    def __init__(
        self,
        pipeline: PlannedPipeline | None = None,
        *,
        model_registry: ModelRegistry | None = None,
        strict_models: bool = False,
    ) -> None:
        self.pipeline = pipeline or PlannedPipeline(
            model_registry=model_registry,
            strict_models=strict_models,
        )
        self.traces: dict[str, PipelineTrace] = {}
        self.last_trace: PipelineTrace | None = None

    def solve(self, task: ArcTask) -> Prediction:
        result = self.pipeline.run(task)
        self.traces[task.id] = result.trace
        self.last_trace = result.trace
        return result.prediction
