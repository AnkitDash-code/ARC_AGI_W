from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from mythos.lora import LoRACastedLinear, inject_lora_adapters, lora_parameters


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        bias = self.bias.to(inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, self.weight.to(inputs.dtype), bias)


class FakeHRMAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = CastedLinear(8, 24)
        self.self_attn.o_proj = CastedLinear(8, 8)
        self.mlp = CastedLinear(8, 8)

    def forward(self, inputs):  # type: ignore[no-untyped-def]
        qkv = self.self_attn.qkv_proj(inputs)
        q = qkv[..., :8]
        return self.mlp(self.self_attn.o_proj(q))


def test_lora_injector_wraps_hrm_castedlinear_attention_targets() -> None:
    model = FakeHRMAttentionBlock()

    report = inject_lora_adapters(model, rank=2, target_patterns=("self_attn",))

    assert report.injected_modules == ("self_attn.qkv_proj", "self_attn.o_proj")
    assert isinstance(model.self_attn.qkv_proj, LoRACastedLinear)
    assert isinstance(model.self_attn.o_proj, LoRACastedLinear)
    assert isinstance(model.mlp, CastedLinear)
    assert lora_parameters(model)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "base_layer" in name)


def test_lora_castedlinear_forward_preserves_shape_and_dtype() -> None:
    layer = LoRACastedLinear(CastedLinear(8, 4), rank=2)
    inputs = torch.randn(3, 8, dtype=torch.float32)

    outputs = layer(inputs)

    assert outputs.shape == (3, 4)
    assert outputs.dtype == inputs.dtype
