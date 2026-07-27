"""External HRM adapter and environment checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from mythos.arc import ArcTask, Grid
from mythos.augment import inverse_transform, transform_task
from mythos.features import ARC_MAX_SIZE, hrm_sequence_to_grid, output_shape_hint
from mythos.hrm_dataset import build_hrm_dataset, default_run_dir, prepare_hrm_raw_dataset
from mythos.losses import genie_background_consistency_loss
from mythos.lora import inject_lora_adapters, lora_parameters
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
        config = _load_hrm_config(self.env.checkpoint_path, dataset_dir=dataset_dir, output_dir=output_dir)
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


@dataclass(frozen=True)
class TTTConfig:
    rank: int = 16
    steps: int = 20
    lr: float = 1e-3
    grad_clip_norm: float = 1.0
    # A step whose (pre-clip) LoRA gradient norm exceeds this is treated as an
    # explosion: skipped and rolled back rather than clipped-and-applied.
    explosion_grad_norm: float = 20.0
    # Must be fixed and consistent across every task's forward passes: the
    # puzzle embedding's sparse-update buffer (local_weights) is allocated
    # once at model-init time sized to whatever global_batch_size was used
    # then, and cannot accept a different batch size later -- confirmed by a
    # real run: "expand: attempting to expand a dimension of length 4 -> 32"
    # once the sizing config (32) and a per-task batch (4) diverged. 2 is the
    # ARC-guaranteed minimum train-pair count for any task, so it's never
    # dropped as an incomplete batch regardless of augmentation settings.
    batch_size: int = 2
    # Weight for the Genie-style background-consistency auxiliary loss (see
    # mythos.losses.genie_background_consistency_loss): penalizes the model
    # for changing background cells during TTT, the master plan's named fix
    # for TTT "hallucinating" -- objects vanishing, backgrounds recoloring --
    # under pure demo-pair supervision. 0 disables it.
    genie_weight: float = 0.1
    # Dihedral transform indices (see mythos.augment) to ensemble across at
    # inference: each gets its own from-scratch TTT fit (demo pairs and test
    # input transformed together, so any orientation-dependent rule stays
    # internally consistent) and the un-augmented predictions are voted
    # over. (0,) = identity only, i.e. ensembling disabled -- the original
    # single-view behavior. Costs roughly len(ensemble_transforms)x the TTT
    # compute per task, so keep this short.
    ensemble_transforms: tuple[int, ...] = (0,)

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "steps": self.steps,
            "lr": self.lr,
            "grad_clip_norm": self.grad_clip_norm,
            "explosion_grad_norm": self.explosion_grad_norm,
            "batch_size": self.batch_size,
            "genie_weight": self.genie_weight,
            "ensemble_transforms": list(self.ensemble_transforms),
        }


class HRMTTTRunner:
    """Per-task test-time training on top of the loaded HRM checkpoint.

    Loads the model once and injects LoRA adapters once (the backbone is
    frozen at injection time and never receives gradients). For each task:
    resets the LoRA adapters to their initial no-op state, runs `ttt.steps`
    gradient-descent steps against only that task's own train pairs, then
    runs inference with the now-adapted model before moving to the next task.

    The model's puzzle-identifier embedding table is sized once from a
    dataset build over *all* tasks (matching the batched eval-only path) so
    it's safely oversized for any single task's tiny per-task dataset build,
    whose own identifier indices will always fit inside it.
    """

    def __init__(self, env: HRMEnvironment, ttt: TTTConfig | None = None, num_aug: int = 0) -> None:
        self.env = env
        self.ttt = ttt or TTTConfig()
        self.num_aug = num_aug
        self._pretrain: Any = None
        self._train_state: Any = None
        self._lora_report: Any = None
        self._lora_init_snapshot: dict[str, Any] = {}
        self._backbone_verified = False

    def solve_tasks(self, tasks: list[ArcTask] | tuple[ArcTask, ...]) -> list[Prediction]:
        task_list = list(tasks)
        if not task_list:
            return []

        self._ensure_model_loaded(task_list)

        predictions: list[Prediction] = []
        for task in task_list:
            predictions.append(self._solve_one_task_ensembled(task))
        return predictions

    def _solve_one_task_ensembled(self, task: ArcTask) -> Prediction:
        transform_indices = self.ttt.ensemble_transforms or (0,)
        if tuple(transform_indices) == (0,):
            return self._solve_one_task_with_ttt(task)  # unchanged single-view path

        # Per test item, every un-augmented candidate grid seen across views
        # (both attempts from every view all count as votes).
        candidates_per_item: list[list[Grid]] = [[] for _ in task.test]
        for index in transform_indices:
            view_task = task if index == 0 else transform_task(task, index, id_suffix=f"__aug{index}")
            try:
                prediction = self._solve_one_task_with_ttt(view_task)
            except Exception as exc:  # noqa: BLE001 - one bad view must not sink the whole ensemble
                print(f"TTT: {task.id}: ensemble view {index} failed: {exc!r}; skipping this view")
                continue
            for item_index, output in enumerate(prediction.outputs):
                candidates_per_item[item_index].append(inverse_transform(index, output.attempt_1))
                candidates_per_item[item_index].append(inverse_transform(index, output.attempt_2))

        attempts: list[tuple[Grid, Grid]] = []
        for candidates in candidates_per_item:
            attempts.append(_top_two_by_vote(candidates))
        return make_prediction(task, attempts)

    def _ensure_model_loaded(self, task_list: list[ArcTask]) -> None:
        if self._train_state is not None:
            return
        torch = HRMEnvironment._import_torch()
        modules = self.env.import_modules()
        self._pretrain = modules["pretrain"]

        # Dataset build over every task purely to size the model's architecture
        # (vocab size, puzzle-identifier count) the same way the working
        # eval-only batched path already does -- not used for training or eval.
        sizing_run_dir = default_run_dir() / "hrm_ttt_sizing"
        raw_dir = prepare_hrm_raw_dataset(
            task_list, sizing_run_dir / "raw" / "ARC-AGI-2" / "data", allow_dummy_test_outputs=True
        )
        sizing_dataset_dir = sizing_run_dir / "data" / "arc-2-sizing"
        build_hrm_dataset(
            hrm_repo_dir=self.env.repo_dir, raw_data_dir=raw_dir, output_dir=sizing_dataset_dir, num_aug=self.num_aug
        )
        sizing_config = _load_hrm_config(
            self.env.checkpoint_path, dataset_dir=sizing_dataset_dir, output_dir=sizing_run_dir / "outputs"
        )
        # Must match what every per-task TTT call below uses (self.ttt.batch_size),
        # not the eval-only path's larger default -- see TTTConfig.batch_size.
        sizing_config.global_batch_size = self.ttt.batch_size
        sizing_loader, sizing_metadata = self._pretrain.create_dataloader(
            sizing_config, "train", test_set_mode=False, epochs_per_iter=1,
            global_batch_size=sizing_config.global_batch_size, rank=0, world_size=1,
        )
        del sizing_loader

        train_state = self._pretrain.init_train_state(sizing_config, sizing_metadata, world_size=1)
        checkpoint = torch.load(self.env.checkpoint_path, map_location="cuda", weights_only=False)
        _load_hrm_checkpoint_best_effort(train_state.model, checkpoint)
        train_state.step = 0

        report = inject_lora_adapters(
            train_state.model,
            rank=self.ttt.rank,
            # Attention-only adapters cap how much the model's actual per-task
            # behavior can change; the MLP layers (gate_up_proj/down_proj) are
            # roughly half the model's parameters and are where most of the
            # per-token transformation logic lives. Widening the LoRA target
            # set gives more real capacity to adapt, instead of just pushing
            # rank/LR higher on a narrower slice of the model (confirmed
            # unstable: v36's rank=64 attention-only run diverged).
            target_patterns=("self_attn", "attn", "qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
            freeze_backbone=True,
        )
        print(
            f"TTT: injected LoRA (rank={self.ttt.rank}) into {len(report.injected_modules)} module(s); "
            f"{report.trainable_parameters} trainable / {report.frozen_parameters} frozen parameters"
        )
        self._lora_report = report
        self._lora_init_snapshot = {
            name: parameter.detach().clone()
            for name, parameter in train_state.model.named_parameters()
            if "lora_" in name
        }
        self._train_state = train_state

    def _reset_lora(self) -> None:
        torch = HRMEnvironment._import_torch()
        with torch.no_grad():
            for name, parameter in self._train_state.model.named_parameters():
                snapshot = self._lora_init_snapshot.get(name)
                if snapshot is not None:
                    parameter.copy_(snapshot)

    def _solve_one_task_with_ttt(self, task: ArcTask) -> Prediction:
        from mythos.lora import changed_frozen_parameters, snapshot_frozen_parameters

        torch = HRMEnvironment._import_torch()
        pretrain = self._pretrain
        train_state = self._train_state

        run_dir = default_run_dir() / "hrm_ttt" / task.id
        raw_dir = prepare_hrm_raw_dataset(
            (task,), run_dir / "raw" / "ARC-AGI-2" / "data", allow_dummy_test_outputs=True
        )
        dataset_dir = run_dir / "data" / "arc-2-ttt"
        build_hrm_dataset(
            hrm_repo_dir=self.env.repo_dir, raw_data_dir=raw_dir, output_dir=dataset_dir, num_aug=self.num_aug
        )
        output_dir = run_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = _load_hrm_config(self.env.checkpoint_path, dataset_dir=dataset_dir, output_dir=output_dir)
        # Must equal the sizing config's batch size (see TTTConfig.batch_size):
        # the puzzle embedding's sparse-update buffer is allocated once at
        # model-init time and cannot accept a different batch size per task.
        original_batch_size = config.global_batch_size
        config.global_batch_size = self.ttt.batch_size
        print(f"TTT: {task.id}: global_batch_size {original_batch_size} -> {config.global_batch_size}")

        train_loader, _train_metadata = pretrain.create_dataloader(
            config, "train", test_set_mode=False, epochs_per_iter=1,
            global_batch_size=config.global_batch_size, rank=0, world_size=1,
        )
        eval_loader, eval_metadata = pretrain.create_dataloader(
            config, "test", test_set_mode=True, epochs_per_iter=1,
            global_batch_size=config.global_batch_size, rank=0, world_size=1,
        )

        self._reset_lora()
        lora_params = lora_parameters(train_state.model)
        optimizer = torch.optim.AdamW(lora_params, lr=self.ttt.lr)

        backbone_snapshot = None
        if not self._backbone_verified:
            backbone_snapshot = snapshot_frozen_parameters(train_state.model)

        # Diagnostic: lora_b is zero-initialized (a fresh LoRA adapter is a
        # mathematical no-op), so any nonzero value after training proves
        # gradients actually flowed and the optimizer actually updated
        # something, independent of whether the eval predictions changed.
        lora_b_before = sum(p.detach().abs().sum().item() for name, p in train_state.model.named_parameters() if name.endswith("lora_b"))

        train_state.model.train()
        carry = None
        skipped_steps = 0
        first_loss = None
        last_loss = None
        step_index = 0
        genie_enabled = self.ttt.genie_weight > 0
        genie_applied = 0
        return_keys = ["logits"] if genie_enabled else []
        for _ in range(self.ttt.steps):
            for _set_name, batch, global_batch_size in train_loader:
                batch = {key: (value.to("cuda") if hasattr(value, "to") else value) for key, value in batch.items()}
                if carry is None:
                    # initial_carry() creates some internal state tensors (e.g. the
                    # `halted` flag) without an explicit device argument, relying on
                    # the ambient default-device context to land them on CUDA --
                    # confirmed both by HRM's own evaluate() doing the same thing
                    # and by a real run failing with "Unhandled FakeTensor Device
                    # Propagation for aten.where.self, found two different devices
                    # cpu, cuda:0" without this wrapper.
                    with torch.device("cuda"):
                        carry = train_state.model.initial_carry(batch)
                carry, primary_loss, _metrics, preds, _all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                loss = primary_loss
                # Only "logits" needs to come from the model's return_keys -- the
                # input grid is already in hand as batch["inputs"] (what we're
                # feeding in, not something the model needs to report back).
                # Requiring all_finish (the model's own halt state) first made
                # this never fire in a real run: 0/50 steps across every task,
                # since a single training forward call rarely reaches full ACT
                # convergence within the step budget. logits are populated on
                # every call regardless of halt state, so use those directly --
                # a consistency signal on the model's current best guess is
                # still useful even before it's fully converged.
                if genie_enabled:
                    try:
                        genie_term = _batch_genie_loss(preds, batch.get("inputs"))
                        if genie_term is not None:
                            loss = primary_loss + self.ttt.genie_weight * genie_term
                            genie_applied += 1
                    except Exception as exc:  # noqa: BLE001 - auxiliary loss must never abort real TTT
                        print(f"TTT: {task.id}: disabling Genie loss after a failure: {exc!r}")
                        genie_enabled = False
                        return_keys = []
                loss_value = float(loss.detach())
                if first_loss is None:
                    first_loss = loss_value
                last_loss = loss_value
                optimizer.zero_grad(set_to_none=True)
                (loss / max(1, global_batch_size)).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, self.ttt.explosion_grad_norm)
                if not torch.isfinite(grad_norm) or grad_norm >= self.ttt.explosion_grad_norm:
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(lora_params, self.ttt.grad_clip_norm)
                optimizer.step()
                step_index += 1
        if skipped_steps:
            print(f"TTT: {task.id}: rolled back {skipped_steps}/{self.ttt.steps} step(s) on gradient explosion")
        if self.ttt.genie_weight > 0:
            print(f"TTT: {task.id}: Genie consistency loss applied on {genie_applied}/{step_index} step(s)")

        lora_b_after = sum(p.detach().abs().sum().item() for name, p in train_state.model.named_parameters() if name.endswith("lora_b"))
        print(
            f"TTT: {task.id}: loss {first_loss!r} -> {last_loss!r} over {step_index} applied step(s); "
            f"sum(|lora_b|) {lora_b_before:.6f} -> {lora_b_after:.6f}"
        )

        if backbone_snapshot is not None:
            changed = changed_frozen_parameters(train_state.model, backbone_snapshot)
            if changed:
                print(f"TTT WARNING: {len(changed)} frozen backbone parameter(s) changed during TTT: {changed[:5]}")
            else:
                print("TTT: verified frozen backbone parameters are unchanged after a TTT run")
            self._backbone_verified = True

        train_state.model.eval()
        pretrain.evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1)
        tokens = _load_decoded_hrm_predictions(output_dir)
        return HRMInferenceRunner(self.env)._tokens_to_prediction(task, tokens)


def _top_two_by_vote(candidates: list[Grid]) -> tuple[Grid, Grid]:
    """Pick the two most-common grids among candidates (majority vote across ensemble views)."""

    if not candidates:
        return [[0]], [[0]]
    counts = Counter(tuple(tuple(row) for row in grid) for grid in candidates)
    ranked = [[list(row) for row in flat] for flat, _ in counts.most_common(2)]
    if len(ranked) == 1:
        ranked.append(ranked[0])
    return ranked[0], ranked[1]


def _load_hrm_config(checkpoint_path: Path, *, dataset_dir: Path, output_dir: Path):  # type: ignore[no-untyped-def]
    try:
        import yaml
        from pretrain import PretrainConfig
    except Exception as exc:  # pragma: no cover - depends on external HRM deps.
        raise HRMEnvironmentError(f"failed to import HRM config dependencies: {exc}") from exc

    config_path = checkpoint_path.parent / "all_config.yaml"
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


def _batch_genie_loss(preds: Any, inputs_batch: Any) -> Any:
    """Average Genie background-consistency loss across a training batch.

    Decodes each example's own input tokens (from the batch fed to the model,
    not something the model needs to report back) to a grid -- no need to
    match against the original un-augmented ArcTask, since the preserve mask
    is derived from the decoded grid's own dominant color -- and penalizes
    the model's predicted logits for changing cells that should stay
    background. Returns None (rather than raising) when preds doesn't
    contain what's needed, so the caller's own try/except only has to guard
    against genuine failures.
    """
    if not preds or "logits" not in preds or inputs_batch is None:
        return None
    logits_batch = preds["logits"]
    if logits_batch is None or logits_batch.shape[0] == 0:
        return None
    # logits comes back as [batch, 900, vocab_size] -- a flat HRM token
    # sequence, not a [H, W, 10] grid (confirmed by a real run: "logits must
    # have rank 3 or rank 4"). Reshape to the 30x30 canvas, then slice out
    # just the 10 color-token channels (HRM's vocab is PAD=0, EOS=1,
    # colors=2..11 -- see grid_to_hrm_sequence/hrm_sequence_to_grid, the same
    # scheme already used to decode this model's own predicted output
    # tokens) since genie_background_consistency_loss compares against plain
    # 0-9 color indices.
    per_example_losses = []
    for example_index in range(logits_batch.shape[0]):
        input_tokens = inputs_batch[example_index].detach().cpu().tolist()
        decoded_input = hrm_sequence_to_grid(input_tokens)
        example_logits = logits_batch[example_index].reshape(ARC_MAX_SIZE, ARC_MAX_SIZE, -1)[:, :, 2:12]
        per_example_losses.append(genie_background_consistency_loss(example_logits, decoded_input))
    if not per_example_losses:
        return None
    return sum(per_example_losses) / len(per_example_losses)


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
