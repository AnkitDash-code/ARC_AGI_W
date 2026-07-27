"""Minimal LoRA utilities for HRM/TTT adapter experiments."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Iterable, Sequence


try:  # Keep this module importable until a LoRA function is actually used.
    from torch import nn as _OPTIONAL_NN
except Exception:  # pragma: no cover - depends on optional torch install.
    _OPTIONAL_NN = None

DEFAULT_LORA_TARGET_PATTERNS = (
    "attn",
    "attention",
    "qkv_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
)


@dataclass(frozen=True)
class LoRAInjectionReport:
    injected_modules: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int

    def to_dict(self) -> dict[str, object]:
        return {
            "injected_modules": list(self.injected_modules),
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
        }


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("PyTorch is required for LoRA utilities") from exc
    return torch


def _nn():
    try:
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError("PyTorch is required for LoRA utilities") from exc
    return nn


_BASE_MODULE = _OPTIONAL_NN.Module if _OPTIONAL_NN is not None else object


class LoRALinear(_BASE_MODULE):
    """Wrap a Linear layer with trainable low-rank adapter weights."""

    def __init__(
        self,
        base_layer,
        *,
        rank: int = 16,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        nn = _nn()
        torch = _torch()
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Match the base layer's device/dtype: creating these as plain CPU/fp32
        # tensors breaks torch.compile'd models (HRM runs compiled on CUDA in
        # bfloat16) -- confirmed by a real run: Dynamo's tracer rejected the
        # LoRA matmul with "Unhandled FakeTensor Device Propagation ... found
        # two different devices cuda:0, cpu".
        base_device = base_layer.weight.device
        base_dtype = base_layer.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, base_layer.in_features, device=base_device, dtype=base_dtype))
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, rank, device=base_device, dtype=base_dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        torch = _torch()
        base = self.base_layer(inputs)
        # CastedLinear (and some plain Linear layers under mixed precision) stores
        # its weight in one dtype (fp32) but receives activations in another
        # (bf16) -- cast the activation to lora_a's dtype before this matmul, not
        # base_layer.weight's dtype, since those two can legitimately differ.
        # Confirmed by a real run: "expected mat1 and mat2 to have the same
        # dtype, but got: c10::BFloat16 != float" once the device mismatch (the
        # earlier bug) was fixed.
        update = self.dropout(inputs).to(dtype=self.lora_a.dtype).matmul(self.lora_a.transpose(0, 1))
        update = update.matmul(self.lora_b.transpose(0, 1))
        return base + update.to(dtype=base.dtype) * torch.as_tensor(self.scaling, dtype=base.dtype, device=base.device)


class LoRACastedLinear(_BASE_MODULE):
    """Wrap HRM's custom CastedLinear layer with trainable LoRA weights."""

    def __init__(
        self,
        base_layer,
        *,
        rank: int = 16,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        nn = _nn()
        torch = _torch()
        super().__init__()
        if not _is_casted_linear_like(base_layer):
            raise TypeError("LoRACastedLinear can only wrap CastedLinear-like modules")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        out_features, in_features = base_layer.weight.shape
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # See LoRALinear's identical comment: must match the base layer's
        # device/dtype or torch.compile'd forward passes fail to trace.
        base_device = base_layer.weight.device
        base_dtype = base_layer.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=base_device, dtype=base_dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=base_device, dtype=base_dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        torch = _torch()
        base = self.base_layer(inputs)
        # CastedLinear (and some plain Linear layers under mixed precision) stores
        # its weight in one dtype (fp32) but receives activations in another
        # (bf16) -- cast the activation to lora_a's dtype before this matmul, not
        # base_layer.weight's dtype, since those two can legitimately differ.
        # Confirmed by a real run: "expected mat1 and mat2 to have the same
        # dtype, but got: c10::BFloat16 != float" once the device mismatch (the
        # earlier bug) was fixed.
        update = self.dropout(inputs).to(dtype=self.lora_a.dtype).matmul(self.lora_a.transpose(0, 1))
        update = update.matmul(self.lora_b.transpose(0, 1))
        return base + update.to(dtype=base.dtype) * torch.as_tensor(self.scaling, dtype=base.dtype, device=base.device)


def inject_lora_adapters(
    model,
    *,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_patterns: Sequence[str] = DEFAULT_LORA_TARGET_PATTERNS,
    fallback_to_all_linear: bool = False,
    freeze_backbone: bool = True,
) -> LoRAInjectionReport:
    """Replace matching Linear/CastedLinear modules with LoRA wrappers."""

    nn = _nn()
    lowered_patterns = tuple(pattern.lower() for pattern in target_patterns)
    injected: list[str] = []

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    def should_wrap(full_name: str, child) -> bool:  # type: ignore[no-untyped-def]
        if not _is_lora_wrappable(child, nn):
            return False
        if any(pattern in full_name.lower() for pattern in lowered_patterns):
            return True
        return fallback_to_all_linear

    def visit(module, prefix: str = "") -> None:  # type: ignore[no-untyped-def]
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if should_wrap(full_name, child):
                setattr(
                    module,
                    child_name,
                    _wrap_lora_layer(child, nn, rank=rank, alpha=alpha, dropout=dropout),
                )
                injected.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not injected:
        raise ValueError("no Linear or CastedLinear modules matched the LoRA target patterns")

    trainable = 0
    frozen = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return LoRAInjectionReport(
        injected_modules=tuple(injected),
        trainable_parameters=trainable,
        frozen_parameters=frozen,
    )


def lora_parameters(model) -> list:  # type: ignore[no-untyped-def]
    return [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]


def lora_state_dict(model) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def save_lora_checkpoint(model, path: str | Path, *, metadata: dict[str, object] | None = None) -> Path:  # type: ignore[no-untyped-def]
    torch = _torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "mythos_lora",
            "state_dict": lora_state_dict(model),
            "metadata": metadata or {},
        },
        output_path,
    )
    return output_path


def snapshot_frozen_parameters(model) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        name: parameter.detach().clone().cpu()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def changed_frozen_parameters(model, snapshot: dict[str, object], *, atol: float = 0.0) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    torch = _torch()
    changed: list[str] = []
    for name, before in snapshot.items():
        current = dict(model.named_parameters())[name].detach().cpu()
        if not torch.allclose(current, before, atol=atol, rtol=0.0):
            changed.append(name)
    return tuple(changed)


def count_parameters(parameters: Iterable) -> int:  # type: ignore[type-arg]
    return sum(parameter.numel() for parameter in parameters)


def _wrap_lora_layer(child, nn, *, rank: int, alpha: float | None, dropout: float):  # type: ignore[no-untyped-def]
    if isinstance(child, nn.Linear):
        return LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
    if _is_casted_linear_like(child):
        return LoRACastedLinear(child, rank=rank, alpha=alpha, dropout=dropout)
    raise TypeError(f"unsupported LoRA target module: {type(child).__name__}")


def _is_lora_wrappable(child, nn) -> bool:  # type: ignore[no-untyped-def]
    return isinstance(child, nn.Linear) or _is_casted_linear_like(child)


def _is_casted_linear_like(child) -> bool:  # type: ignore[no-untyped-def]
    casted_types = _casted_linear_types()
    if casted_types and isinstance(child, casted_types):
        return True
    if child.__class__.__name__ != "CastedLinear":
        return False
    weight = getattr(child, "weight", None)
    return weight is not None and getattr(weight, "ndim", None) == 2 and callable(getattr(child, "forward", None))


def _casted_linear_types() -> tuple[type, ...]:
    types: list[type] = []
    for module_name in ("models.layers", "layers"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        casted_linear = getattr(module, "CastedLinear", None)
        if isinstance(casted_linear, type):
            types.append(casted_linear)
    return tuple(types)
