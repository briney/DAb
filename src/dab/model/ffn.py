"""SwiGLU Feed-Forward Network implementation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(x) * gate."""

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        return F.silu(x) * gate


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    FFN(x) = W_down * SwiGLU(W_gate(x), W_up(x))

    Args:
        d_model: Model dimension
        d_ffn: FFN intermediate dimension (default: 8/3 * d_model)
        bias: Whether to include bias in linear layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_ffn is None:
            d_ffn = int(d_model * 8 / 3)
            d_ffn = ((d_ffn + 63) // 64) * 64

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_up = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)

        self.activation = SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        hidden = self.activation(gate, up)
        hidden = self.dropout(hidden)
        return self.w_down(hidden)


class FusedSwiGLUFFN(nn.Module):
    """Memory-efficient SwiGLU FFN with fused gate/up projection."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_ffn is None:
            d_ffn = int(d_model * 8 / 3)
            d_ffn = ((d_ffn + 63) // 64) * 64

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate_up = nn.Linear(d_model, d_ffn * 2, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)
