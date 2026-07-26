"""External HRM adapter and environment checks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from mythos.arc import ArcTask
from mythos.features import hrm_sequence_to_grid, output_shape_hint
from mythos.hrm_dataset import build_hrm_dataset, default_run_dir, prepare_hrm_raw_dataset
from mythos.solvers.base import SolverError, make_prediction
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
    """External HRM solver for CUDA/Kaggle smoke inference."""

    def __init__(self, env: HRMEnvironment | None = None) -> None:
        self.env = env

    def solve(self, task: ArcTask) -> Prediction:
        env = self.env or HRMEnvironment.from_env()
        env.validate(require_cuda=True)
        runner = HRMInferenceRunner(env)
        return runner.solve_task(task)


@dataclass(frozen=True)
class HRMInferenceRunner:
    env: HRMEnvironment
    num_aug: int = 0

    def solve_task(self, task: ArcTask) -> Prediction:
        run_dir = default_run_dir() / "hrm_inference" / task.id
        raw_dir = prepare_hrm_raw_dataset(
            (task,),
            run_dir / "raw" / "ARC-AGI-2" / "data",
            allow_dummy_test_outputs=True,
        )
        dataset_dir = run_dir / "data" / "arc-2-one-task"
        build_hrm_dataset(
            hrm_repo_dir=self.env.repo_dir,
            raw_data_dir=raw_dir,
            output_dir=dataset_dir,
            num_aug=self.num_aug,
        )
        prediction_tokens = self._run_external_evaluate(dataset_dir, run_dir / "outputs")
        return self._tokens_to_prediction(task, prediction_tokens)

    def solve_tasks(self, tasks: list[ArcTask] | tuple[ArcTask, ...]) -> list[Prediction]:
        task_list = list(tasks)
        if not task_list:
            return []
        run_dir = default_run_dir() / "hrm_inference_batch"
        raw_dir = prepare_hrm_raw_dataset(
            task_list,
            run_dir / "raw" / "ARC-AGI-2" / "data",
            allow_dummy_test_outputs=True,
        )
        dataset_dir = run_dir / "data" / "arc-2-batch"
        build_hrm_dataset(
            hrm_repo_dir=self.env.repo_dir,
            raw_data_dir=raw_dir,
            output_dir=dataset_dir,
            num_aug=self.num_aug,
        )
        prediction_tokens = self._run_external_evaluate(dataset_dir, run_dir / "outputs")
        predictions: list[Prediction] = []
        cursor = 0
        for task in task_list:
            count = len(task.test)
            predictions.append(self._tokens_to_prediction(task, prediction_tokens[cursor : cursor + count]))
            cursor += count
        if cursor > len(prediction_tokens):
            raise HRMEnvironmentError(
                f"HRM returned {len(prediction_tokens)} predictions for {cursor} requested test inputs"
            )
        return predictions

    def _tokens_to_prediction(
        self,
        task: ArcTask,
        prediction_tokens: list[tuple[list[int], list[int]]],
    ) -> Prediction:
        if len(prediction_tokens) < len(task.test):
            raise HRMEnvironmentError(
                f"{task.id}: HRM returned {len(prediction_tokens)} predictions for "
                f"{len(task.test)} test inputs"
            )

        attempts = []
        for index, example in enumerate(task.test):
            shape_hint = output_shape_hint(task, example.input)
            top1, top2 = prediction_tokens[index]
            attempts.append(
                (
                    hrm_sequence_to_grid(top1, shape_hint=shape_hint),
                    hrm_sequence_to_grid(top2, shape_hint=shape_hint),
                )
            )
        return make_prediction(task, attempts)

    def _run_external_evaluate(self, dataset_dir: Path, output_dir: Path) -> list[tuple[list[int], list[int]]]:
        torch = HRMEnvironment._import_torch()
        modules = self.env.import_modules()
        pretrain = modules["pretrain"]

        output_dir.mkdir(parents=True, exist_ok=True)
        config = self._load_eval_config(dataset_dir=dataset_dir, output_dir=output_dir)
        train_loader, train_metadata = pretrain.create_dataloader(
            config,
            "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=0,
            world_size=1,
        )
        eval_loader, eval_metadata = pretrain.create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=0,
            world_size=1,
        )
        del train_loader

        train_state = pretrain.init_train_state(config, train_metadata, world_size=1)
        checkpoint = torch.load(self.env.checkpoint_path, map_location="cuda", weights_only=False)
        _load_hrm_checkpoint_best_effort(train_state.model, checkpoint)
        train_state.step = 0
        train_state.model.eval()
        pretrain.evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1)
        return _load_decoded_hrm_predictions(output_dir)

    def _load_eval_config(self, *, dataset_dir: Path, output_dir: Path):  # type: ignore[no-untyped-def]
        try:
            import yaml
            from pretrain import PretrainConfig
        except Exception as exc:  # pragma: no cover - depends on external HRM deps.
            raise HRMEnvironmentError(f"failed to import HRM config dependencies: {exc}") from exc

        config_path = self.env.checkpoint_path.parent / "all_config.yaml"
        if not config_path.exists():
            raise HRMEnvironmentError(f"HRM checkpoint directory is missing all_config.yaml: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = PretrainConfig(**yaml.safe_load(handle))
        config.data_path = str(dataset_dir)
        config.checkpoint_path = str(output_dir)
        config.eval_save_outputs = ["inputs", "puzzle_identifiers", "logits"]
        if "HRM_GLOBAL_BATCH_SIZE" in os.environ:
            config.global_batch_size = int(os.environ["HRM_GLOBAL_BATCH_SIZE"])
        return config


def _load_hrm_checkpoint_best_effort(model: Any, checkpoint: dict[str, Any]) -> None:
    """Load the pretrained checkpoint, keeping randomly-initialized weights for any
    key whose shape doesn't match.

    HRM's `puzzle_emb` is a per-puzzle lookup table sized to the exact puzzle
    identifier vocabulary of whatever dataset it was trained on (the public
    checkpoint: 1,045,829 entries). A dataset built from a different task set
    (ours: 240 tasks) gets fresh, unrelated identifier indices, so this table
    can never meaningfully transfer -- there is no "fix" for that mismatch,
    only whether to keep evaluating with it randomly initialized (this) or
    fail outright. Every other weight (attention/MLP layers, token/H/L init,
    LM head) is the real pretrained model and does transfer correctly.
    """
    model_state = model.state_dict()
    filtered: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in checkpoint.items():
        candidates = (key, f"_orig_mod.{key}", key.removeprefix("_orig_mod."))
        matched_key = next((name for name in candidates if name in model_state), None)
        if matched_key is None:
            skipped.append(f"{key}: not present in model")
            continue
        if tuple(model_state[matched_key].shape) != tuple(value.shape):
            skipped.append(
                f"{matched_key}: checkpoint={tuple(value.shape)} model={tuple(model_state[matched_key].shape)}"
            )
            continue
        filtered[matched_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False, assign=True)
    print(f"HRM checkpoint: loaded {len(filtered)}/{len(model_state)} weight tensors from the pretrained checkpoint")
    if skipped:
        print(f"HRM checkpoint: kept randomly-initialized (shape/name mismatch) for {len(skipped)} key(s): {skipped}")
    if missing:
        print(f"HRM checkpoint: {len(missing)} model key(s) had no checkpoint match: {list(missing)}")
    if unexpected:
        print(f"HRM checkpoint: {len(unexpected)} checkpoint key(s) were unused: {list(unexpected)}")


def _load_decoded_hrm_predictions(output_dir: Path) -> list[tuple[list[int], list[int]]]:
    torch = HRMEnvironment._import_torch()
    pred_files = sorted(output_dir.glob("step_*_all_preds.*"))
    if not pred_files:
        raise HRMEnvironmentError(f"HRM evaluation wrote no prediction files in {output_dir}")

    raw = torch.load(pred_files[0], map_location="cpu", weights_only=False)
    if "logits" not in raw:
        raise HRMEnvironmentError(
            f"HRM prediction file {pred_files[0]} is missing logits; keys={sorted(raw)}"
        )
    logits = raw["logits"]
    topk = logits.topk(k=2, dim=-1).indices
    decoded: list[tuple[list[int], list[int]]] = []
    for row in range(topk.shape[0]):
        top1 = topk[row, :, 0].to(dtype=torch.int64).tolist()
        top2 = topk[row, :, 1].to(dtype=torch.int64).tolist()
        decoded.append((top1, top2))
    return decoded
