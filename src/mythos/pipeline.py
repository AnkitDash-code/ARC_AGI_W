"""Plan-aligned Project Mythos inference pipeline.

The real research components are still adapters here. The important point for
the base implementation is that data flows through the same stage boundaries as
the master plan, so each placeholder has an obvious replacement point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Iterable, Tuple

from mythos.arc import ArcTask, Grid, copy_grid
from mythos.features import (
    DEFAULT_HRM_FEATURE_DIM,
    DEFAULT_JEPA_FEATURE_DIM,
    embedding_cosine_similarity,
    grid_to_feature_vector,
    task_rule_vector,
)
from mythos.jepa_encoder import JepaEncodingError, JepaImageEncoder
from mythos.models import ModelRegistry
from mythos.solvers.baseline import BaselineSolver
from mythos.solvers.base import SolverError
from mythos.solvers.hrm import HRMSolver
from mythos.submission import Prediction, TestPrediction, prediction_to_json, validate_submission_data
from mythos.text_reasoning import HRMTextError, generate_hrm_text_rule
from mythos.training import load_projection_checkpoint, load_world_model_checkpoint

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
    world_rollouts: tuple[GridEmbedding, ...] = ()
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
        self.strict_models = strict_models
        self._jepa_encoder = None
        self._projection_model = None
        self._world_model = None

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
        jepa_model = self.model_registry.get("jepa")
        projection = self.model_registry.get("jepa_projection")
        embeddings: list[GridEmbedding] = []
        projection_detail = "projection checkpoint not configured"
        projection_loaded = False
        real_jepa_enabled = os.environ.get("MYTHOS_ENABLE_REAL_JEPA", "0") == "1"
        real_jepa_loaded = False
        try:
            jepa_encoder = None
            if real_jepa_enabled and jepa_model.loaded and jepa_model.checkpoint_path is not None:
                jepa_encoder = self._get_jepa_encoder(jepa_model.checkpoint_path)
                real_jepa_loaded = True
            if projection.loaded and projection.checkpoint_path is not None:
                projection_model = self._get_projection_model(projection.checkpoint_path)
                projection_loaded = True
                projection_detail = f"projection={projection.checkpoint_path}"
                for split, grids in _iter_task_grids(state.task):
                    for index, grid in enumerate(grids):
                        embeddings.append(
                            _encode_jepa_grid(
                                f"{split}[{index}]",
                                grid,
                                jepa_encoder=jepa_encoder,
                                projection_model=projection_model,
                            )
                        )
            else:
                for split, grids in _iter_task_grids(state.task):
                    for index, grid in enumerate(grids):
                        embeddings.append(
                            _encode_jepa_grid(
                                f"{split}[{index}]",
                                grid,
                                jepa_encoder=jepa_encoder,
                                projection_model=None,
                            )
                        )
        except Exception as exc:
            if self.strict_models:
                raise
            projection_detail = f"projection failed: {exc}; deterministic ARC-feature fallback used"
            embeddings = [
                _feature_grid(f"{split}[{index}]", grid, dim=DEFAULT_JEPA_FEATURE_DIM)
                for split, grids in _iter_task_grids(state.task)
                for index, grid in enumerate(grids)
            ]
        state.embeddings = tuple(embeddings)
        if real_jepa_loaded and projection_loaded:
            status = "jepa_forward_projection_loaded"
        elif real_jepa_loaded:
            status = "jepa_forward"
        elif projection_loaded:
            status = "projection_loaded"
        else:
            status = "fallback"

        if real_jepa_loaded:
            detail = (
                f"{jepa_model.describe()}; {projection.describe()}; "
                f"MYTHOS_ENABLE_REAL_JEPA=1; transformers I-JEPA forward ran; "
                f"{projection_detail}; produced {len(embeddings)} embeddings"
            )
        elif projection_loaded:
            detail = (
                f"{jepa_model.describe()}; {projection.describe()}; {projection_detail}; "
                "no I-JEPA forward pass is run without MYTHOS_ENABLE_REAL_JEPA=1; "
                f"projected {len(embeddings)} deterministic ARC-feature embeddings"
            )
        else:
            detail = (
                f"{jepa_model.describe()}; {projection.describe()}; {projection_detail}; "
                f"MYTHOS_ENABLE_REAL_JEPA={int(real_jepa_enabled)}; no I-JEPA forward pass ran; produced "
                f"{len(embeddings)} deterministic ARC-feature embeddings"
            )
        state.trace.add(
            "encode_jepa",
            status,
            detail,
        )

    def _plan_hrm_text(self, state: PipelineState) -> None:
        model = self.model_registry.get("hrm_text")
        enabled = os.environ.get("MYTHOS_ENABLE_HRM_TEXT", "0") == "1"
        state.rule_vector = _make_rule_vector(state.task)
        status = "fallback"
        detail = (
            f"{model.describe()}; MYTHOS_ENABLE_HRM_TEXT={int(enabled)}; "
            "deterministic fallback rule vector derived from train-pair deltas"
        )
        if enabled and model.loaded and model.checkpoint_path is not None:
            try:
                result = generate_hrm_text_rule(state.task, model.checkpoint_path)
                state.rule_vector = RuleVector(
                    source="hrm_text_forward",
                    description=result.description,
                    vector=result.vector,
                )
                status = "model_forward"
                detail = (
                    f"{model.describe()}; HRM-Text forward/generation ran from "
                    f"{result.model_root}; rule={result.description[:160]!r}"
                )
            except HRMTextError as exc:
                if self.strict_models:
                    raise
                status = "fallback"
                detail = (
                    f"{model.describe()}; HRM-Text forward failed: {exc}; "
                    "deterministic fallback rule vector used"
                )
        elif model.loaded and not enabled:
            status = "disabled_checkpoint_loaded"
        state.trace.add(
            "plan_hrm_text",
            status,
            detail,
        )

    def _simulate_world_model(self, state: PipelineState) -> None:
        if state.rule_vector is None:
            raise RuntimeError("rule vector must exist before world-model simulation")
        model = self.model_registry.get("world_model")
        status = "model_loaded" if model.loaded else "fallback"
        detail = f"{model.describe()}; no world-model checkpoint, so no rollout guidance was produced"
        if model.loaded and model.checkpoint_path is not None:
            try:
                world_model = self._get_world_model(model.checkpoint_path)
                state.world_rollouts = tuple(_simulate_rollouts(state.task, world_model))
                detail = (
                    f"{model.describe()}; generated {len(state.world_rollouts)} "
                    "test-input latent rollouts"
                )
            except Exception as exc:
                if self.strict_models:
                    raise
                status = "fallback"
                detail = f"{model.describe()}; world-model rollout failed: {exc}"
        state.trace.add(
            "simulate_world_model",
            status,
            detail,
        )

    def _adapt_ttt_lora(self, state: PipelineState) -> None:
        model = self.model_registry.get("ttt_lora")
        hrm = self.model_registry.get("hrm_l_module")
        enabled = os.environ.get("MYTHOS_ENABLE_TTT", "0") == "1"
        status = "model_loaded" if model.loaded else "fallback"
        if enabled and hrm.loaded:
            detail = (
                f"{model.describe()}; MYTHOS_ENABLE_TTT=1, but live LoRA-on-HRM "
                "is not connected because the external HRM train-step hook is not "
                "wrapped by this pipeline yet"
            )
            status = "not_connected"
        else:
            detail = (
                f"{model.describe()}; per-task LoRA update skipped "
                f"(MYTHOS_ENABLE_TTT={int(enabled)}, hrm_loaded={hrm.loaded})"
            )
        state.trace.add(
            "adapt_ttt_lora",
            status,
            detail,
        )

    def _execute_hrm_l_module(self, state: PipelineState) -> None:
        model = self.model_registry.get("hrm_l_module")
        enable_real_hrm = os.environ.get("MYTHOS_ENABLE_REAL_HRM", "0") == "1"
        status = "model_loaded_fallback_executor" if model.loaded else "fallback"
        if enable_real_hrm:
            try:
                state.prediction = HRMSolver().solve(state.task)
                status = "model_loaded"
                detail = f"{model.describe()}; external HRM prediction path returned output"
            except SolverError as exc:
                if self.strict_models:
                    raise
                state.prediction = self.executor.solve(state.task)
                status = "fallback"
                detail = f"{model.describe()}; HRM execution failed: {exc}; baseline executor used"
        else:
            state.prediction = self.executor.solve(state.task)
            detail = (
                f"{model.describe()}; baseline executor produced valid output "
                "because MYTHOS_ENABLE_REAL_HRM is not set"
            )
        if state.prediction is not None and state.world_rollouts:
            state.prediction, guided_count = _apply_world_rollout_attempts(
                state.task,
                state.prediction,
                state.world_rollouts,
            )
            detail += f"; world-model rollout guidance replaced attempt_2 for {guided_count} test item(s)"
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

    def _get_projection_model(self, checkpoint_path):  # type: ignore[no-untyped-def]
        if self._projection_model is None:
            self._projection_model = load_projection_checkpoint(checkpoint_path)
        return self._projection_model

    def _get_jepa_encoder(self, checkpoint_path):  # type: ignore[no-untyped-def]
        if self._jepa_encoder is None:
            self._jepa_encoder = JepaImageEncoder.from_path(checkpoint_path)
        return self._jepa_encoder

    def _get_world_model(self, checkpoint_path):  # type: ignore[no-untyped-def]
        if self._world_model is None:
            self._world_model = load_world_model_checkpoint(checkpoint_path)
        return self._world_model


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


def _feature_grid(source: str, grid: Grid, *, dim: int) -> GridEmbedding:
    return GridEmbedding(
        source=source,
        shape=(len(grid), len(grid[0])),
        vector=grid_to_feature_vector(grid, dim),
    )


def _encode_jepa_grid(
    source: str,
    grid: Grid,
    *,
    jepa_encoder,
    projection_model,
) -> GridEmbedding:  # type: ignore[no-untyped-def]
    if jepa_encoder is not None:
        vector = jepa_encoder.encode_grid(grid)
        vector_source = "i_jepa_forward"
    else:
        feature_dim = projection_model.config.input_dim if projection_model is not None else DEFAULT_JEPA_FEATURE_DIM
        vector = grid_to_feature_vector(grid, feature_dim)
        vector_source = "deterministic_arc_features"

    if projection_model is not None:
        vector = _project_vector(vector, projection_model)
        vector_source += "_projected"

    return GridEmbedding(
        source=f"{source}.{vector_source}",
        shape=(len(grid), len(grid[0])),
        vector=_rounded(float(value) for value in vector),
    )


def _project_grid(source: str, grid: Grid, projection_model) -> GridEmbedding:  # type: ignore[no-untyped-def]
    vector = grid_to_feature_vector(grid, projection_model.config.input_dim)
    return GridEmbedding(
        source=source,
        shape=(len(grid), len(grid[0])),
        vector=_project_vector(vector, projection_model),
    )


def _project_vector(vector: tuple[float, ...], projection_model) -> tuple[float, ...]:  # type: ignore[no-untyped-def]
    import torch

    if len(vector) != projection_model.config.input_dim:
        raise JepaEncodingError(
            f"projection expects {projection_model.config.input_dim} features, got {len(vector)}"
        )
    device = next(projection_model.parameters()).device
    tensor = torch.tensor([vector], dtype=torch.float32, device=device)
    with torch.no_grad():
        projected = projection_model(tensor)[0].detach().cpu().tolist()
    return _rounded(float(value) for value in projected)


def _simulate_rollouts(task: ArcTask, world_model) -> Iterable[GridEmbedding]:  # type: ignore[no-untyped-def]
    import torch

    device = next(world_model.parameters()).device
    rule = torch.tensor(
        [task_rule_vector(task, world_model.config.rule_dim)],
        dtype=torch.float32,
        device=device,
    )
    for index, example in enumerate(task.test):
        z_input = torch.tensor(
            [grid_to_feature_vector(example.input, world_model.config.z_dim)],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            rollout = world_model(z_input, rule)[0].detach().cpu().tolist()
        yield GridEmbedding(
            source=f"test_input[{index}].world_rollout",
            shape=(len(example.input), len(example.input[0])),
            vector=_rounded(float(value) for value in rollout),
        )


def _apply_world_rollout_attempts(
    task: ArcTask,
    prediction: Prediction,
    rollouts: tuple[GridEmbedding, ...],
) -> tuple[Prediction, int]:
    train_outputs = [
        (grid_to_feature_vector(example.output, len(rollouts[0].vector)), example.output)
        for example in task.train
        if example.output is not None
    ]
    if not train_outputs:
        return prediction, 0

    guided_outputs: list[TestPrediction] = []
    guided_count = 0
    for index, item in enumerate(prediction.outputs):
        if index >= len(rollouts):
            guided_outputs.append(item)
            continue
        rollout = rollouts[index]
        nearest_grid = max(
            train_outputs,
            key=lambda candidate: embedding_cosine_similarity(rollout.vector, candidate[0]),
        )[1]
        guided_outputs.append(
            TestPrediction(
                attempt_1=item.attempt_1,
                attempt_2=copy_grid(nearest_grid),
            )
        )
        guided_count += 1
    return Prediction(task_id=prediction.task_id, outputs=tuple(guided_outputs)), guided_count


def _make_rule_vector(task: ArcTask) -> RuleVector:
    vector = task_rule_vector(task)
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
