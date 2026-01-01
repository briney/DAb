"""Pre-norm transformer block with configurable attention."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .attention import ChainAwareAttention, MultiHeadAttention
from .ffn import FusedSwiGLUFFN


class PreNormBlock(nn.Module):
    """
    Pre-norm transformer block with configurable attention and SwiGLU FFN.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_norm_eps: float = 1e-6,
        use_chain_aware_attention: bool = True,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Select attention type based on config
        attention_cls = ChainAwareAttention if use_chain_aware_attention else MultiHeadAttention
        self.attention = attention_cls(
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
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_attentions: If True, return attention weights

        Returns:
            If output_attentions is False:
                Output tensor of shape (batch, seq_len, d_model)
            If output_attentions is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        normed = self.attention_norm(x)

        if output_attentions:
            attn_out, attn_weights = self.attention(
                normed, chain_ids, attention_mask, need_weights=True
            )
        else:
            attn_out = self.attention(normed, chain_ids, attention_mask, need_weights=False)

        x = x + self.dropout(attn_out)

        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        if output_attentions:
            return x, attn_weights
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer blocks."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        use_chain_aware_attention: bool = True,
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
                    use_chain_aware_attention=use_chain_aware_attention,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]] | tuple[
        Tensor, tuple[Tensor, ...], tuple[Tensor, ...]
    ]:
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_hidden_states: If True, return all hidden states (including input)
            output_attentions: If True, return attention weights from all layers

        Returns:
            If neither output_hidden_states nor output_attentions:
                Output tensor of shape (batch, seq_len, d_model)
            If output_hidden_states only:
                Tuple of (output, hidden_states) where hidden_states is a tuple of
                n_layers + 1 tensors (input embedding + each layer output before final norm)
            If output_attentions only:
                Tuple of (output, attentions) where attentions is a tuple of
                n_layers attention weight tensors
            If both:
                Tuple of (output, hidden_states, attentions)
        """
        all_hidden_states: tuple[Tensor, ...] = ()
        all_attentions: tuple[Tensor, ...] = ()

        # Include input embeddings in hidden states
        if output_hidden_states:
            all_hidden_states = (x,)

        for layer in self.layers:
            if output_attentions:
                x, attn_weights = layer(
                    x, chain_ids, attention_mask, output_attentions=True
                )
                all_attentions = all_attentions + (attn_weights,)
            else:
                x = layer(x, chain_ids, attention_mask, output_attentions=False)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        x = self.final_norm(x)

        # Build return value based on what was requested
        if output_hidden_states and output_attentions:
            return x, all_hidden_states, all_attentions
        elif output_hidden_states:
            return x, all_hidden_states
        elif output_attentions:
            return x, all_attentions
        return x
