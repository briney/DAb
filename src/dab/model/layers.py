"""Pre-norm transformer block with chain-aware attention."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import EfficientChainAwareAttention
from .ffn import FusedSwiGLUFFN


class PreNormBlock(nn.Module):
    """
    Pre-norm transformer block with chain-aware attention and SwiGLU FFN.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention = EfficientChainAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
        )

        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FusedSwiGLUFFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        normed = self.attention_norm(x)
        attn_out = self.attention(normed, chain_ids, attention_mask)
        x = x + self.dropout(attn_out)

        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer blocks."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                PreNormBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_all_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        hidden_states = []

        for layer in self.layers:
            x = layer(x, chain_ids, attention_mask)
            if return_all_hidden_states:
                hidden_states.append(x)

        x = self.final_norm(x)

        if return_all_hidden_states:
            return x, hidden_states
        return x
