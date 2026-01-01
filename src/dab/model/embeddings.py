"""Token and timestep embeddings for the diffusion model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional scaling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 1,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * self.scale


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, d_model: int, max_timesteps: int = 1000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps

        embeddings = self._build_embeddings(max_timesteps, d_model)
        self.register_buffer("embeddings", embeddings)

    def _build_embeddings(self, max_timesteps: int, d_model: int) -> Tensor:
        half_dim = d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        timesteps = torch.arange(max_timesteps, dtype=torch.float32)
        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if d_model % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(max_timesteps, 1)], dim=-1)

        return embeddings

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.embeddings[timesteps]


class LearnedTimestepEmbedding(nn.Module):
    """Learned timestep embedding with MLP projection."""

    def __init__(
        self,
        d_model: int,
        max_timesteps: int = 1000,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4

        self.sinusoidal = SinusoidalTimestepEmbedding(d_model, max_timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        emb = self.sinusoidal(timesteps)
        return self.mlp(emb)


class DAbEmbedding(nn.Module):
    """
    Combined embedding module for DAb model.

    Combines token embeddings with optional timestep embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 1,
        max_timesteps: int = 100,
        use_timestep_embedding: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)

        self.use_timestep_embedding = use_timestep_embedding
        if use_timestep_embedding:
            self.timestep_embedding = LearnedTimestepEmbedding(d_model, max_timesteps)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(
        self,
        token_ids: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        embeddings = self.token_embedding(token_ids)

        if self.use_timestep_embedding and timesteps is not None:
            timestep_emb = self.timestep_embedding(timesteps)
            embeddings = embeddings + timestep_emb.unsqueeze(1)

        return self.dropout(embeddings)
