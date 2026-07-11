"""Training helpers for Project Mythos planned stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Iterable

from mythos.arc import ArcTask
from mythos.features import (
    DEFAULT_HRM_FEATURE_DIM,
    DEFAULT_JEPA_FEATURE_DIM,
    DEFAULT_RULE_DIM,
    grid_to_feature_vector,
    iter_all_supervised_grid_pairs,
    iter_train_grid_pairs,
    max_pairwise_cosine,
    task_rule_vector,
)
from mythos.lora import (
    changed_frozen_parameters,
    inject_lora_adapters,
    lora_parameters,
    save_lora_checkpoint,
    snapshot_frozen_parameters,
)


try:  # Keep imports cheap for CLI validation paths that do not train.
    from torch import nn as _OPTIONAL_NN
except Exception:  # pragma: no cover - depends on optional torch install.
    _OPTIONAL_NN = None


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("PyTorch is required for Mythos training helpers") from exc
    return torch


def _nn():
    try:
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("PyTorch is required for Mythos training helpers") from exc
    return nn


@dataclass(frozen=True)
class JepaProjectionConfig:
    input_dim: int = DEFAULT_JEPA_FEATURE_DIM
    output_dim: int = DEFAULT_HRM_FEATURE_DIM
    seed: int = 7


@dataclass(frozen=True)
class WorldModelConfig:
    z_dim: int = DEFAULT_HRM_FEATURE_DIM
    rule_dim: int = DEFAULT_RULE_DIM
    hidden_dim: int = 3072
    seed: int = 11


@dataclass(frozen=True)
class TrainingResult:
    stage: str
    checkpoint_path: str | None
    steps: int
    examples: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    extra: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "checkpoint_path": self.checkpoint_path,
            "steps": self.steps,
            "examples": self.examples,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "elapsed_seconds": self.elapsed_seconds,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class TTTSmokeResult:
    steps: int
    rank: int
    injected_modules: tuple[str, ...]
    initial_loss: float
    final_loss: float
    first_backward_seconds: float
    frozen_parameter_changes: tuple[str, ...]
    checkpoint_path: str | None

    @property
    def backbone_frozen(self) -> bool:
        return not self.frozen_parameter_changes

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "rank": self.rank,
            "injected_modules": list(self.injected_modules),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "first_backward_seconds": self.first_backward_seconds,
            "frozen_parameter_changes": list(self.frozen_parameter_changes),
            "backbone_frozen": self.backbone_frozen,
            "checkpoint_path": self.checkpoint_path,
        }


_BASE_MODULE = _OPTIONAL_NN.Module if _OPTIONAL_NN is not None else object


class JepaProjection(_BASE_MODULE):
    """Projection layer from I-JEPA/ARC features into the HRM feature width."""

    def __init__(self, config: JepaProjectionConfig) -> None:
        nn = _nn()
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim, elementwise_affine=False)
        self.proj = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        return self.proj(self.norm(inputs))


class WorldModelMLP(_BASE_MODULE):
    """Two-layer transition model f(z_input, v_rule) -> z_output."""

    def __init__(self, config: WorldModelConfig) -> None:
        nn = _nn()
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.z_dim + config.rule_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.z_dim),
        )

    def forward(self, z_input, rule):  # type: ignore[no-untyped-def]
        torch = _torch()
        return self.net(torch.cat([z_input, rule], dim=-1))


def train_jepa_projection(
    tasks: Iterable[ArcTask],
    *,
    checkpoint_path: str | Path | None = None,
    config: JepaProjectionConfig | None = None,
    steps: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    include_test_solutions: bool = False,
) -> TrainingResult:
    """Train only the ARC-to-HRM projection checkpoint."""

    torch = _torch()
    nn = _nn()
    cfg = config or JepaProjectionConfig()
    torch.manual_seed(cfg.seed)
    started = time.perf_counter()

    pairs = list(
        iter_all_supervised_grid_pairs(tasks) if include_test_solutions else iter_train_grid_pairs(tasks)
    )
    if not pairs:
        raise ValueError("no supervised ARC grid pairs available for projection training")

    selected_device = _select_device(device)
    x = torch.tensor(
        [grid_to_feature_vector(pair.input, cfg.input_dim) for pair in pairs],
        dtype=torch.float32,
        device=selected_device,
    )
    y = torch.tensor(
        [grid_to_feature_vector(pair.output, cfg.output_dim) for pair in pairs],
        dtype=torch.float32,
        device=selected_device,
    )

    model = JepaProjection(cfg).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(model(x), y).detach().cpu())
    for _ in range(max(0, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_fn(model(x), y).detach().cpu())

    output = None
    if checkpoint_path is not None:
        output = save_projection_checkpoint(model, checkpoint_path)

    diversity = max_pairwise_cosine(
        [grid_to_feature_vector(pair.input, cfg.input_dim) for pair in pairs[: min(10, len(pairs))]]
    )
    return TrainingResult(
        stage="jepa_projection",
        checkpoint_path=str(output) if output is not None else None,
        steps=steps,
        examples=len(pairs),
        initial_loss=round(initial_loss, 8),
        final_loss=round(final_loss, 8),
        elapsed_seconds=round(time.perf_counter() - started, 3),
        extra={
            "config": asdict(cfg),
            "device": selected_device,
            "max_pairwise_input_cosine": round(diversity, 6),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    )


def save_projection_checkpoint(model: JepaProjection, path: str | Path) -> Path:
    torch = _torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "mythos_jepa_projection",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_projection_checkpoint(path: str | Path, *, device: str | None = None) -> JepaProjection:
    torch = _torch()
    selected_device = _select_device(device)
    raw = torch.load(path, map_location=selected_device, weights_only=False)
    config = JepaProjectionConfig(**raw["config"])
    model = JepaProjection(config).to(selected_device)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model


def train_world_model(
    tasks: Iterable[ArcTask],
    *,
    checkpoint_path: str | Path | None = None,
    config: WorldModelConfig | None = None,
    steps: int = 300,
    lr: float = 1e-3,
    device: str | None = None,
    include_test_solutions: bool = False,
) -> TrainingResult:
    """Train the two-layer transition world model from ARC before/after pairs."""

    torch = _torch()
    nn = _nn()
    cfg = config or WorldModelConfig()
    torch.manual_seed(cfg.seed)
    started = time.perf_counter()

    task_list = list(tasks)
    pairs = list(
        iter_all_supervised_grid_pairs(task_list) if include_test_solutions else iter_train_grid_pairs(task_list)
    )
    task_by_id = {task.id: task for task in task_list}
    if not pairs:
        raise ValueError("no supervised ARC grid pairs available for world-model training")

    selected_device = _select_device(device)
    z_input = torch.tensor(
        [grid_to_feature_vector(pair.input, cfg.z_dim) for pair in pairs],
        dtype=torch.float32,
        device=selected_device,
    )
    rules = torch.tensor(
        [task_rule_vector(task_by_id[pair.task_id], cfg.rule_dim) for pair in pairs],
        dtype=torch.float32,
        device=selected_device,
    )
    z_target = torch.tensor(
        [grid_to_feature_vector(pair.output, cfg.z_dim) for pair in pairs],
        dtype=torch.float32,
        device=selected_device,
    )

    model = WorldModelMLP(cfg).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial_loss = float(loss_fn(model(z_input, rules), z_target).detach().cpu())
    for _ in range(max(0, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(z_input, rules), z_target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_fn(model(z_input, rules), z_target).detach().cpu())

    output = None
    if checkpoint_path is not None:
        output = save_world_model_checkpoint(model, checkpoint_path)

    return TrainingResult(
        stage="world_model",
        checkpoint_path=str(output) if output is not None else None,
        steps=steps,
        examples=len(pairs),
        initial_loss=round(initial_loss, 8),
        final_loss=round(final_loss, 8),
        elapsed_seconds=round(time.perf_counter() - started, 3),
        extra={
            "config": asdict(cfg),
            "device": selected_device,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    )


def save_world_model_checkpoint(model: WorldModelMLP, path: str | Path) -> Path:
    torch = _torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "mythos_world_model",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def load_world_model_checkpoint(path: str | Path, *, device: str | None = None) -> WorldModelMLP:
    torch = _torch()
    selected_device = _select_device(device)
    raw = torch.load(path, map_location=selected_device, weights_only=False)
    config = WorldModelConfig(**raw["config"])
    model = WorldModelMLP(config).to(selected_device)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model


def run_ttt_lora_smoke(
    *,
    rank: int = 16,
    steps: int = 50,
    dim: int = 32,
    batch_size: int = 8,
    lr: float = 1e-2,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> TTTSmokeResult:
    """Run a small LoRA-only optimization to validate the TTT mechanics."""

    torch = _torch()
    nn = _nn()
    selected_device = _select_device(device)
    torch.manual_seed(23)

    class TinyAttentionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_q = nn.Linear(dim, dim)
            self.attention_out = nn.Linear(dim, dim)
            self.mlp = nn.Sequential(nn.GELU(), nn.Linear(dim, dim))

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.mlp(self.attention_out(torch.tanh(self.attention_q(x))))

    model = TinyAttentionModel().to(selected_device)
    report = inject_lora_adapters(
        model,
        rank=rank,
        target_patterns=("attention",),
        fallback_to_all_linear=False,
        freeze_backbone=True,
    )
    snapshot = snapshot_frozen_parameters(model)
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=lr)
    loss_fn = nn.MSELoss()
    inputs = torch.randn(batch_size, dim, device=selected_device)
    targets = torch.flip(inputs, dims=(-1,))

    with torch.no_grad():
        initial_loss = float(loss_fn(model(inputs), targets).detach().cpu())

    first_backward_seconds = 0.0
    for step_index in range(max(0, steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        started = time.perf_counter()
        loss.backward()
        if step_index == 0:
            first_backward_seconds = time.perf_counter() - started
        optimizer.step()

    with torch.no_grad():
        final_loss = float(loss_fn(model(inputs), targets).detach().cpu())
    changed = changed_frozen_parameters(model, snapshot)

    output = None
    if checkpoint_path is not None:
        output = save_lora_checkpoint(
            model,
            checkpoint_path,
            metadata={
                "rank": rank,
                "steps": steps,
                "dim": dim,
                "batch_size": batch_size,
                "smoke": True,
            },
        )

    return TTTSmokeResult(
        steps=steps,
        rank=rank,
        injected_modules=report.injected_modules,
        initial_loss=round(initial_loss, 8),
        final_loss=round(final_loss, 8),
        first_backward_seconds=round(first_backward_seconds, 6),
        frozen_parameter_changes=changed,
        checkpoint_path=str(output) if output is not None else None,
    )


def adaptive_ttt_should_stop(
    losses: Iterable[float],
    *,
    min_delta: float = 1e-4,
    patience: int = 5,
) -> bool:
    """Return True when recent TTT loss improvements have flattened."""

    collected = list(losses)
    if len(collected) <= patience:
        return False
    recent = collected[-(patience + 1) :]
    improvements = [recent[index] - recent[index + 1] for index in range(len(recent) - 1)]
    return all(improvement < min_delta for improvement in improvements)


def _select_device(device: str | None) -> str:
    if device:
        return device
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"
