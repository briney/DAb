"""Main DAb transformer model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..vocab import Vocab
from .embeddings import DAbEmbedding
from .layers import TransformerEncoder


@dataclass
class DAbConfig:
    """Configuration for DAb model."""

    vocab_size: int = 32
    padding_idx: int = Vocab.PAD_IDX

    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 4
    head_dim: int = 64
    d_ffn: Optional[int] = None

    max_seq_len: int = 320
    max_timesteps: int = 100
    use_timestep_embedding: bool = False

    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_ffn is None:
            self.d_ffn = int(self.d_model * 8 / 3)
            self.d_ffn = ((self.d_ffn + 63) // 64) * 64


class DAbModel(nn.Module):
    """
    Discrete Diffusion Antibody Language Model.

    Pre-norm transformer with RoPE, SwiGLU, and hybrid self/cross attention.
    """

    def __init__(self, config: DAbConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = DAbEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.padding_idx,
            max_timesteps=config.max_timesteps,
            use_timestep_embedding=config.use_timestep_embedding,
            dropout=config.embedding_dropout,
        )

        self.encoder = TransformerEncoder(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            d_ffn=config.d_ffn,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            max_seq_len=config.max_seq_len,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: Tensor,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            timesteps: Optional diffusion timesteps of shape (batch,)
            output_hidden_states: If True, return all hidden states (n_layers + 1 tensors)
            output_attentions: If True, return attention weights from all layers

        Returns:
            Dictionary with:
                - "logits": Output logits of shape (batch, seq_len, vocab_size)
                - "hidden_states": Final hidden states of shape (batch, seq_len, d_model)
                - "all_hidden_states": (optional) Tuple of n_layers + 1 hidden state tensors
                - "attentions": (optional) Tuple of n_layers attention weight tensors
        """
        embedded = self.embeddings(token_ids, timesteps)

        # Call encoder with appropriate flags
        encoder_outputs = self.encoder(
            embedded,
            chain_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Parse encoder outputs based on what was requested
        if output_hidden_states and output_attentions:
            hidden_states, all_hidden_states, all_attentions = encoder_outputs
        elif output_hidden_states:
            hidden_states, all_hidden_states = encoder_outputs
            all_attentions = None
        elif output_attentions:
            hidden_states, all_attentions = encoder_outputs
            all_hidden_states = None
        else:
            hidden_states = encoder_outputs
            all_hidden_states = None
            all_attentions = None

        logits = self.lm_head(hidden_states)

        output = {"logits": logits, "hidden_states": hidden_states}

        if all_hidden_states is not None:
            output["all_hidden_states"] = all_hidden_states

        if all_attentions is not None:
            output["attentions"] = all_attentions

        return output

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "DAbModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = DAbConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.state_dict(),
            },
            path,
        )
