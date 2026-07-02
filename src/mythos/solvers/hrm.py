"""External HRM adapter and environment checks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

from mythos.arc import ArcTask
from mythos.solvers.base import SolverError
from mythos.submission import Prediction


class HRMEnvironmentError(SolverError):
    """Raised when the external HRM runtime is not ready."""


@dataclass(frozen=True)
class HRMEnvironment:
    repo_dir: Path
    checkpoint_path: Path

    @classmethod
    def from_env(cls) -> "HRMEnvironment":
        repo_value = os.environ.get("HRM_REPO_DIR")
        checkpoint_value = os.environ.get("HRM_CHECKPOINT_PATH")
        if not repo_value:
            raise HRMEnvironmentError("HRM_REPO_DIR is required for HRM execution")
        if not checkpoint_value:
            raise HRMEnvironmentError("HRM_CHECKPOINT_PATH is required for HRM execution")
        return cls(repo_dir=Path(repo_value), checkpoint_path=Path(checkpoint_value))

    def validate(self, *, require_cuda: bool = True) -> None:
        if not self.repo_dir.exists():
            raise HRMEnvironmentError(f"HRM_REPO_DIR does not exist: {self.repo_dir}")
        if not (self.repo_dir / "evaluate.py").exists():
            raise HRMEnvironmentError(f"HRM checkout is missing evaluate.py: {self.repo_dir}")
        if not (self.repo_dir / "dataset" / "build_arc_dataset.py").exists():
            raise HRMEnvironmentError(
                f"HRM checkout is missing dataset/build_arc_dataset.py: {self.repo_dir}"
            )
        if not self.checkpoint_path.exists():
            raise HRMEnvironmentError(f"HRM_CHECKPOINT_PATH does not exist: {self.checkpoint_path}")

        torch = self._import_torch()
        if require_cuda and not torch.cuda.is_available():
            raise HRMEnvironmentError("HRM execution requires CUDA; torch.cuda.is_available() is false")

    def import_modules(self) -> dict[str, Any]:
        self._add_repo_to_path()
        modules = {}
        for module_name in ("pretrain", "evaluate"):
            try:
                modules[module_name] = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - depends on external HRM deps.
                raise HRMEnvironmentError(
                    f"failed to import HRM module {module_name!r} from {self.repo_dir}: {exc}"
                ) from exc
        return modules

    def load_checkpoint(self) -> Any:
        torch = self._import_torch()
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return torch.load(
                self.checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            try:
                return torch.load(self.checkpoint_path, map_location=map_location)
            except Exception as exc:  # pragma: no cover - depends on checkpoint format.
                raise HRMEnvironmentError(
                    f"failed to load HRM checkpoint {self.checkpoint_path}: {exc}"
                ) from exc
        except Exception as exc:  # pragma: no cover - depends on checkpoint format.
            raise HRMEnvironmentError(f"failed to load HRM checkpoint {self.checkpoint_path}: {exc}") from exc

    def _add_repo_to_path(self) -> None:
        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    @staticmethod
    def _import_torch() -> Any:
        try:
            return importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - torch is optional locally.
            raise HRMEnvironmentError("PyTorch is required for HRM execution") from exc


class HRMSolver:
    """Placeholder real-model solver that fails early unless HRM is configured."""

    def __init__(self, env: HRMEnvironment | None = None) -> None:
        self.env = env

    def solve(self, task: ArcTask) -> Prediction:
        env = self.env or HRMEnvironment.from_env()
        env.validate(require_cuda=True)
        raise HRMEnvironmentError(
            f"{task.id}: HRM environment is valid, but direct prediction wiring is not implemented yet. "
            "Use `python -m mythos.hrm_smoke` to validate the external runtime and dataset path."
        )
