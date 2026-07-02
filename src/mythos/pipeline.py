"""Plan-aligned Project Mythos inference pipeline.

The real research components are still adapters here. The important point for
the base implementation is that data flows through the same stage boundaries as
the master plan, so each placeholder has an obvious replacement point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Tuple

from mythos.arc import ArcTask, Grid
from mythos.models import ModelRegistry
from mythos.solvers.baseline import BaselineSolver
from mythos.submission import Prediction, prediction_to_json, validate_submission_data

PLAN_STAGE_ORDER = (
    "ingest",
    "encode_jepa",
    "plan_hrm_text",
    "simulate_world_model",
    "adapt_ttt_lora",
    "execute_hrm_l_module",
    "decode_output",
)


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class GridEmbedding:
    source: str
    shape: tuple[int, int]
    vector: tuple[float, ...]


@dataclass(frozen=True)
class RuleVector:
    source: str
    description: str
    vector: tuple[float, ...]


@dataclass
class PipelineTrace:
    task_id: str
    stages: list[StageRecord] = field(default_factory=list)

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def add(self, name: str, status: str, detail: str) -> None:
        self.stages.append(StageRecord(name=name, status=status, detail=detail))

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "stages": [
                {"name": stage.name, "status": stage.status, "detail": stage.detail}
                for stage in self.stages
            ],
        }


@dataclass(frozen=True)
class PipelineResult:
    prediction: Prediction
    trace: PipelineTrace


@dataclass
class PipelineState:
    task: ArcTask
    trace: PipelineTrace
    embeddings: tuple[GridEmbedding, ...] = ()
    rule_vector: RuleVector | None = None
    prediction: Prediction | None = None


class PlannedPipeline:
    """Runs the ARC task through the Project Mythos master-plan stages."""

    def __init__(
        self,
        executor: BaselineSolver | None = None,
        model_registry: ModelRegistry | None = None,
        *,
        strict_models: bool = False,
    ) -> None:
        self.executor = executor or BaselineSolver()
        self.model_registry = model_registry or ModelRegistry.from_env(strict=strict_models)

    def run(self, task: ArcTask) -> PipelineResult:
        state = PipelineState(task=task, trace=PipelineTrace(task_id=task.id))

        self._ingest(state)
        self._encode_jepa(state)
        self._plan_hrm_text(state)
        self._simulate_world_model(state)
        self._adapt_ttt_lora(state)
        self._execute_hrm_l_module(state)
        self._decode_output(state)

        assert state.prediction is not None
        return PipelineResult(prediction=state.prediction, trace=state.trace)

    def _ingest(self, state: PipelineState) -> None:
        train_count = len(state.task.train)
        test_count = len(state.task.test)
        state.trace.add(
            "ingest",
            "ok",
            f"loaded task with {train_count} train pairs and {test_count} test inputs",
        )

    def _encode_jepa(self, state: PipelineState) -> None:
        model = self.model_registry.get("jepa")
        embeddings: list[GridEmbedding] = []
        for split, grids in _iter_task_grids(state.task):
            for index, grid in enumerate(grids):
                embeddings.append(_encode_grid(f"{split}[{index}]", grid))
        state.embeddings = tuple(embeddings)
        status = "model_loaded" if model.loaded else "fallback"
        detail = (
            f"{model.describe()}; deterministic grid-feature adapter produced "
            f"{len(embeddings)} embeddings"
        )
        state.trace.add(
            "encode_jepa",
            status,
            detail,
        )

    def _plan_hrm_text(self, state: PipelineState) -> None:
        model = self.model_registry.get("hrm_text")
        state.rule_vector = _make_rule_vector(state.task)
        status = "model_loaded" if model.loaded else "fallback"
        state.trace.add(
            "plan_hrm_text",
            status,
            f"{model.describe()}; rule vector derived from train-pair deltas",
        )

    def _simulate_world_model(self, state: PipelineState) -> None:
        if state.rule_vector is None:
            raise RuntimeError("rule vector must exist before world-model simulation")
        model = self.model_registry.get("world_model")
        status = "model_loaded" if model.loaded else "fallback"
        state.trace.add(
            "simulate_world_model",
            status,
            f"{model.describe()}; rollout hook passed rule vector through",
        )

    def _adapt_ttt_lora(self, state: PipelineState) -> None:
        model = self.model_registry.get("ttt_lora")
        status = "model_loaded" if model.loaded else "fallback"
        state.trace.add(
            "adapt_ttt_lora",
            status,
            f"{model.describe()}; per-task adapter update hook completed",
        )

    def _execute_hrm_l_module(self, state: PipelineState) -> None:
        model = self.model_registry.get("hrm_l_module")
        state.prediction = self.executor.solve(state.task)
        status = "model_loaded_fallback_executor" if model.loaded else "fallback"
        detail = (
            f"{model.describe()}; baseline executor produced valid output because "
            "direct HRM inference is not implemented in this adapter yet"
        )
        state.trace.add(
            "execute_hrm_l_module",
            status,
            detail,
        )

    def _decode_output(self, state: PipelineState) -> None:
        if state.prediction is None:
            raise RuntimeError("prediction must exist before decode/output")
        validate_submission_data({state.prediction.task_id: prediction_to_json(state.prediction)})
        state.trace.add(
            "decode_output",
            "ok",
            f"validated {len(state.prediction.outputs)} two-attempt test predictions",
        )


def _iter_task_grids(task: ArcTask) -> Iterable[tuple[str, tuple[Grid, ...]]]:
    yield "train_input", tuple(example.input for example in task.train)
    yield "train_output", tuple(example.output for example in task.train if example.output is not None)
    yield "test_input", tuple(example.input for example in task.test)


def _encode_grid(source: str, grid: Grid) -> GridEmbedding:
    height = len(grid)
    width = len(grid[0])
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    dominant_color = counts.most_common(1)[0][0]
    nonzero = sum(1 for cell in flat if cell != 0)
    total = len(flat)
    vector = (
        height / 30.0,
        width / 30.0,
        dominant_color / 9.0,
        nonzero / total,
        sum(flat) / (9.0 * total),
    )
    return GridEmbedding(source=source, shape=(height, width), vector=_rounded(vector))


def _make_rule_vector(task: ArcTask) -> RuleVector:
    shape_deltas: list[tuple[int, int]] = []
    nonzero_deltas: list[int] = []
    color_jaccards: list[float] = []

    for example in task.train:
        if example.output is None:
            continue
        in_h, in_w = len(example.input), len(example.input[0])
        out_h, out_w = len(example.output), len(example.output[0])
        shape_deltas.append((out_h - in_h, out_w - in_w))
        nonzero_deltas.append(_nonzero_count(example.output) - _nonzero_count(example.input))
        color_jaccards.append(_color_jaccard(example.input, example.output))

    avg_shape_delta_h = _average(delta[0] for delta in shape_deltas)
    avg_shape_delta_w = _average(delta[1] for delta in shape_deltas)
    avg_nonzero_delta = _average(nonzero_deltas)
    avg_color_overlap = _average(color_jaccards)
    vector = _rounded(
        (
            avg_shape_delta_h / 30.0,
            avg_shape_delta_w / 30.0,
            avg_nonzero_delta / 900.0,
            avg_color_overlap,
        )
    )
    return RuleVector(
        source="deterministic_train_pair_delta",
        description="shape, density, and color-overlap summary from demonstration pairs",
        vector=vector,
    )


def _nonzero_count(grid: Grid) -> int:
    return sum(1 for row in grid for cell in row if cell != 0)


def _color_jaccard(left: Grid, right: Grid) -> float:
    left_colors = {cell for row in left for cell in row}
    right_colors = {cell for row in right for cell in row}
    union = left_colors | right_colors
    if not union:
        return 1.0
    return len(left_colors & right_colors) / len(union)


def _average(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _rounded(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(round(value, 6) for value in values)
