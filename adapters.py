from __future__ import annotations
from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 32, nonlinearity: str = "relu"):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)
        self.act = nn.ReLU() if nonlinearity == "relu" else nn.GELU()
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)   # start near identity
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Linear(in_f, self.r, bias=False)
        self.B = nn.Linear(self.r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.B(self.A(self.dropout(x))) * self.scaling
        return base_out + lora_out

def _iter_transformer_layers(encoder: nn.TransformerEncoder) -> Iterable[nn.TransformerEncoderLayer]:
    for layer in encoder.layers:
        yield layer

def inject_adapters_into_encoder(encoder: nn.TransformerEncoder, hidden_dim: int, bottleneck: int, nonlinearity: str = "relu"):
    for layer in _iter_transformer_layers(encoder):
        if not hasattr(layer, "adapter"):
            layer.adapter = Adapter(hidden_dim, bottleneck, nonlinearity)

def wrap_lora_in_encoder_layers(
    encoder: nn.TransformerEncoder,
    targets: Tuple[str, ...] = ("attn_out", "ffn"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    for layer in _iter_transformer_layers(encoder):
        if "attn_out" in targets:
            proj = layer.self_attn.out_proj
            if not isinstance(proj, LoRALinear):
                layer.self_attn.out_proj = LoRALinear(proj, r=r, alpha=alpha, dropout=dropout)
        if "ffn" in targets:
            if not isinstance(layer.linear1, LoRALinear):
                layer.linear1 = LoRALinear(layer.linear1, r=r, alpha=alpha, dropout=dropout)
            if not isinstance(layer.linear2, LoRALinear):
                layer.linear2 = LoRALinear(layer.linear2, r=r, alpha=alpha, dropout=dropout)

def set_trainable_modules(model: nn.Module, trainable_names: Tuple[str, ...]):
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if any(t in n for t in trainable_names):
            p.requires_grad_(True)

def mark_layernorm_trainable(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
