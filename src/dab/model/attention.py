"""
Hybrid Self-Attention and Cross-Attention with chain masking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .rope import RotaryPositionEmbedding


class EfficientChainAwareAttention(nn.Module):
    """
    Attention module supporting hybrid intra-chain (self) and inter-chain (cross) attention.

    For antibody sequences with multiple chains:
    1. Computes self-attention scores for all position pairs
    2. Computes cross-attention scores for all position pairs
    3. Creates composite: intra-chain pairs use self-attention, inter-chain use cross-attention

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        head_dim: Dimension per head (default: 64)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        inner_dim = n_heads * head_dim

        # Self-attention projections
        self.q_self = nn.Linear(d_model, inner_dim, bias=bias)
        self.k_self = nn.Linear(d_model, inner_dim, bias=bias)
        self.v_self = nn.Linear(d_model, inner_dim, bias=bias)

        # Cross-attention projections
        self.q_cross = nn.Linear(d_model, inner_dim, bias=bias)
        self.k_cross = nn.Linear(d_model, inner_dim, bias=bias)
        self.v_cross = nn.Linear(d_model, inner_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, d_model, bias=bias)

        # RoPE
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len=max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with efficient chain-aware attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V for both attention types
        q_self = rearrange(self.q_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_self = rearrange(self.k_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_self = rearrange(self.v_self(x), "b s (h d) -> b h s d", h=self.n_heads)

        q_cross = rearrange(self.q_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_cross = rearrange(self.k_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_cross = rearrange(self.v_cross(x), "b s (h d) -> b h s d", h=self.n_heads)

        # Apply RoPE
        q_self, k_self = self.rope(q_self, k_self)
        q_cross, k_cross = self.rope(q_cross, k_cross)

        # Compute raw attention scores
        scores_self = torch.matmul(q_self, k_self.transpose(-2, -1)) * self.scale
        scores_cross = torch.matmul(q_cross, k_cross.transpose(-2, -1)) * self.scale

        # Create chain masks
        chain_i = chain_ids.unsqueeze(-1)  # (batch, seq_len, 1)
        chain_j = chain_ids.unsqueeze(-2)  # (batch, 1, seq_len)
        intra_mask = (chain_i == chain_j).unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        inter_mask = ~intra_mask

        # Mask out irrelevant positions in each attention type
        scores_self = scores_self.masked_fill(inter_mask, float("-inf"))
        scores_cross = scores_cross.masked_fill(intra_mask, float("-inf"))

        # Apply padding mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool().unsqueeze(1).unsqueeze(2)
            scores_self = scores_self.masked_fill(padding_mask, float("-inf"))
            scores_cross = scores_cross.masked_fill(padding_mask, float("-inf"))

        # Compute attention weights
        attn_self = F.softmax(scores_self, dim=-1)
        attn_cross = F.softmax(scores_cross, dim=-1)

        # Handle NaN from all-masked rows
        attn_self = torch.nan_to_num(attn_self, nan=0.0)
        attn_cross = torch.nan_to_num(attn_cross, nan=0.0)

        attn_self = self.dropout(attn_self)
        attn_cross = self.dropout(attn_cross)

        # Compute outputs
        out_self = torch.matmul(attn_self, v_self)
        out_cross = torch.matmul(attn_cross, v_cross)

        # Combine outputs
        output = out_self + out_cross

        # Reshape and project
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)

        return output
