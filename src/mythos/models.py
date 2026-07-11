"""External model loading for the Project Mythos pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import sys
from typing import Any

from mythos.solvers.base import SolverError


class ModelLoadError(SolverError):
    """Raised when a configured external model cannot be loaded."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    checkpoint_env: str
    repo_env: str | None = None
    module_names: tuple[str, ...] = ()


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="jepa",
        label="JEPA encoder",
        checkpoint_env="IJEPA_CHECKPOINT_PATH",
    ),
    ModelSpec(
        key="jepa_projection",
        label="I-JEPA projection",
        checkpoint_env="IJEPA_PROJECTION_CHECKPOINT_PATH",
    ),
    ModelSpec(
        key="hrm_text",
        label="HRM-Text H-module",
        checkpoint_env="HRM_TEXT_CHECKPOINT_PATH",
    ),
    ModelSpec(
        key="world_model",
        label="World model",
        checkpoint_env="WORLD_MODEL_CHECKPOINT_PATH",
    ),
    ModelSpec(
        key="ttt_lora",
        label="TTT LoRA adapters",
        checkpoint_env="TTT_LORA_CHECKPOINT_PATH",
    ),
    ModelSpec(
        key="hrm_l_module",
        label="HRM 27M L-module",
        repo_env="HRM_REPO_DIR",
        checkpoint_env="HRM_CHECKPOINT_PATH",
        module_names=("pretrain", "evaluate"),
    ),
)


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    repo_dir: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint: Any = None
    modules: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.checkpoint is not None

    def describe(self) -> str:
        if not self.loaded:
            if self.error:
                return f"{self.spec.label}: load failed: {self.error}"
            return f"{self.spec.label}: not configured"
        parts = [f"{self.spec.label}: checkpoint={self.checkpoint_path}"]
        if self.repo_dir is not None:
            parts.append(f"repo={self.repo_dir}")
        if self.modules:
            parts.append(f"modules={','.join(sorted(self.modules))}")
        return "; ".join(parts)


class ModelRegistry:
    """Loads and stores external models keyed by planned pipeline component."""

    def __init__(self, models: dict[str, LoadedModel], *, strict: bool) -> None:
        self.models = models
        self.strict = strict

    @classmethod
    def from_env(cls, *, strict: bool = False) -> "ModelRegistry":
        models: dict[str, LoadedModel] = {}
        for spec in MODEL_SPECS:
            try:
                models[spec.key] = _load_model_from_env(spec, strict=strict)
            except ModelLoadError as exc:
                if strict:
                    raise
                models[spec.key] = LoadedModel(spec=spec, error=str(exc))
        return cls(models=models, strict=strict)

    def get(self, key: str) -> LoadedModel:
        return self.models[key]

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "key": key,
                "label": model.spec.label,
                "loaded": model.loaded,
                "repo_dir": str(model.repo_dir) if model.repo_dir is not None else None,
                "checkpoint_path": str(model.checkpoint_path) if model.checkpoint_path is not None else None,
                "modules": sorted(model.modules),
                "error": model.error,
            }
            for key, model in self.models.items()
        ]


def _load_model_from_env(spec: ModelSpec, *, strict: bool) -> LoadedModel:
    repo_value = os.environ.get(spec.repo_env) if spec.repo_env is not None else None
    checkpoint_value = os.environ.get(spec.checkpoint_env)

    if not repo_value and not checkpoint_value:
        if strict:
            missing = spec.checkpoint_env
            if spec.repo_env is not None:
                missing = f"{spec.repo_env} and {spec.checkpoint_env}"
            raise ModelLoadError(f"{missing} are required for strict model loading")
        return LoadedModel(spec=spec)

    if spec.repo_env is not None and not repo_value:
        raise ModelLoadError(f"{spec.repo_env} is required when loading {spec.label}")
    if not checkpoint_value:
        raise ModelLoadError(f"{spec.checkpoint_env} is required when loading {spec.label}")

    repo_dir = Path(repo_value) if repo_value else None
    checkpoint_path = Path(checkpoint_value)

    if repo_dir is not None:
        if not repo_dir.exists():
            raise ModelLoadError(f"{spec.repo_env} does not exist: {repo_dir}")
        _add_repo_to_path(repo_dir)

    if not checkpoint_path.exists():
        raise ModelLoadError(f"{spec.checkpoint_env} does not exist: {checkpoint_path}")

    modules = _import_modules(spec)
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    return LoadedModel(
        spec=spec,
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        modules=modules,
    )


def _add_repo_to_path(repo_dir: Path) -> None:
    repo = str(repo_dir.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _import_modules(spec: ModelSpec) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for module_name in spec.module_names:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            raise ModelLoadError(
                f"failed to import {module_name!r} for {spec.label}: {exc}"
            ) from exc
    return modules


def _load_torch_checkpoint(checkpoint_path: Path) -> Any:
    if checkpoint_path.is_dir() or checkpoint_path.suffix.lower() == ".safetensors":
        return {
            "path": str(checkpoint_path),
            "format": checkpoint_path.suffix.lower().lstrip(".") or "directory",
            "lazy": True,
        }

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise ModelLoadError("PyTorch is required to load model checkpoints") from exc

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
