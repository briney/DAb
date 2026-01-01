"""
Hybrid Self-Attention and Cross-Attention with chain masking.

Implements the MINT-style attention mechanism where:
1. Self-attention scores (with RoPE) are computed for intra-chain pairs
2. Cross-attention scores (without RoPE) are computed for inter-chain pairs
3. Scores are merged before softmax based on chain membership
4. A single softmax normalizes across all positions
5. Weights are split after softmax for separate value multiplications
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .rope import RotaryPositionEmbedding


class ChainAwareAttention(nn.Module):
    """
    Attention module implementing MINT-style hybrid intra/inter-chain attention.

    For antibody sequences with multiple chains:
    1. Computes self-attention scores (with RoPE) for intra-chain pairs
    2. Computes cross-attention scores (without RoPE) for inter-chain pairs
    3. Merges scores before softmax: intra-chain uses self scores, inter-chain uses cross scores
    4. Applies single softmax to merged scores
    5. Splits attention weights after softmax for value multiplication

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
        self.dropout_p = dropout

        inner_dim = n_heads * head_dim

        # Self-attention projections (RoPE will be applied to Q and K)
        self.q_self = nn.Linear(d_model, inner_dim, bias=bias)
        self.k_self = nn.Linear(d_model, inner_dim, bias=bias)
        self.v_self = nn.Linear(d_model, inner_dim, bias=bias)

        # Cross-attention projections (no RoPE)
        self.q_cross = nn.Linear(d_model, inner_dim, bias=bias)
        self.k_cross = nn.Linear(d_model, inner_dim, bias=bias)
        self.v_cross = nn.Linear(d_model, inner_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, d_model, bias=bias)

        # RoPE (only applied to self-attention)
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len=max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def _create_chain_mask(
        self, chain_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        Create intra-chain mask and padding mask.

        Args:
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)

        Returns:
            intra_mask: Boolean mask where True = same chain (batch, 1, seq_len, seq_len)
            padding_mask: Additive mask with -inf for padding (batch, 1, 1, seq_len)
        """
        # Chain masks: (batch, seq_len, seq_len)
        chain_i = chain_ids.unsqueeze(-1)  # (batch, seq_len, 1)
        chain_j = chain_ids.unsqueeze(-2)  # (batch, 1, seq_len)
        intra_mask = (chain_i == chain_j).unsqueeze(1)  # (batch, 1, seq_len, seq_len)

        # Padding mask
        if attention_mask is not None:
            # Create additive mask: 0 where valid, -inf where padding
            padding_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            padding_mask = padding_mask.masked_fill(~attention_mask.bool(), float("-inf"))
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        else:
            padding_mask = None

        return intra_mask, padding_mask

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass with MINT-style chain-aware attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            need_weights: If True, return attention weights

        Returns:
            If need_weights is False:
                Output tensor of shape (batch, seq_len, d_model)
            If need_weights is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V for both attention types
        q_self = rearrange(self.q_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_self = rearrange(self.k_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_self = rearrange(self.v_self(x), "b s (h d) -> b h s d", h=self.n_heads)

        q_cross = rearrange(self.q_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_cross = rearrange(self.k_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_cross = rearrange(self.v_cross(x), "b s (h d) -> b h s d", h=self.n_heads)

        # Apply RoPE only to self-attention Q and K (not cross-attention)
        q_self, k_self = self.rope(q_self, k_self)

        # Compute raw attention scores
        scores_self = torch.matmul(q_self, k_self.transpose(-2, -1)) * self.scale
        scores_cross = torch.matmul(q_cross, k_cross.transpose(-2, -1)) * self.scale

        # Create chain masks
        intra_mask, padding_mask = self._create_chain_mask(chain_ids, attention_mask)

        # Convert intra_mask to float for torch.where and later multiplication
        intra_mask_float = intra_mask.float()

        # Merge attention scores before softmax:
        # Use self scores for intra-chain pairs, cross scores for inter-chain pairs
        merged_scores = torch.where(intra_mask, scores_self, scores_cross)

        # Apply padding mask
        if padding_mask is not None:
            merged_scores = merged_scores + padding_mask

        # Single softmax over all positions
        attn_weights = F.softmax(merged_scores, dim=-1)

        # Handle NaN from all-masked rows (e.g., all padding)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Compute weighted values using chain mask to route to appropriate values
        # Intra-chain positions use v_self, inter-chain positions use v_cross
        out_self = torch.matmul(attn_weights * intra_mask_float, v_self)
        out_cross = torch.matmul(attn_weights * (1.0 - intra_mask_float), v_cross)
        output = out_self + out_cross

        # Reshape and project
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output
