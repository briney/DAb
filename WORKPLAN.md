# DAb: Discrete Diffusion Antibody Language Model

## Comprehensive Implementation Workplan

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Package Configuration](#3-package-configuration)
4. [Vocabulary and Tokenization](#4-vocabulary-and-tokenization)
5. [Model Architecture](#5-model-architecture)
6. [Discrete Diffusion](#6-discrete-diffusion)
7. [Data Loading](#7-data-loading)
8. [Training Infrastructure](#8-training-infrastructure)
9. [Encoding API](#9-encoding-api)
10. [Configuration System](#10-configuration-system)
11. [CLI Implementation](#11-cli-implementation)
12. [Logging and Checkpointing](#12-logging-and-checkpointing)
13. [Testing Suite](#13-testing-suite)
14. [Implementation Order](#14-implementation-order)

---

## 1. Project Overview

### 1.1 Goals

DAb is a discrete diffusion language model for antibody sequences. Key features:

- **Architecture**: Pre-norm transformer with RoPE, SwiGLU, and hybrid self/cross-attention
- **Training**: Discrete diffusion with information-weighted masking
- **Multi-chain**: Supports variable numbers of chains with intra-chain self-attention and inter-chain cross-attention
- **Flexibility**: Configurable via Hydra, multi-GPU via Accelerate

### 1.2 Design Principles

- **Step-driven training**: All intervals (logging, validation, checkpointing) specified in steps by default
- **Modularity**: Components designed for easy extension and modification
- **Modern Python**: Type hints, dataclasses, pyproject.toml packaging
- **Testability**: Comprehensive unit, integration, and end-to-end tests

---

## 2. Repository Structure

```
DAb/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   ├── config.yaml                 # Main config with defaults
│   ├── model/
│   │   ├── base.yaml               # Base model config
│   │   ├── small.yaml              # Small model variant
│   │   └── large.yaml              # Large model variant
│   ├── training/
│   │   ├── default.yaml            # Default training config
│   │   └── debug.yaml              # Fast debug config
│   ├── data/
│   │   └── default.yaml            # Data loading config
│   ├── diffusion/
│   │   ├── linear.yaml             # Linear noise schedule
│   │   ├── cosine.yaml             # Cosine noise schedule
│   │   └── sqrt.yaml               # Sqrt noise schedule
│   └── logging/
│       └── wandb.yaml              # WandB config
├── src/
│   └── dab/
│       ├── __init__.py
│       ├── __main__.py             # Entry point for `python -m dab`
│       ├── cli.py                  # Click CLI definitions
│       ├── version.py              # Version info
│       ├── vocab.py                # Vocabulary and tokenization
│       ├── model/
│       │   ├── __init__.py
│       │   ├── transformer.py      # Main transformer model
│       │   ├── attention.py        # Self/cross attention with chain masking
│       │   ├── embeddings.py       # Token + optional timestep embeddings
│       │   ├── rope.py             # Rotary position embeddings
│       │   ├── ffn.py              # SwiGLU feed-forward network
│       │   └── layers.py           # Pre-norm transformer block
│       ├── diffusion/
│       │   ├── __init__.py
│       │   ├── noise_schedule.py   # Linear, cosine, sqrt schedules
│       │   ├── masking.py          # Information-weighted masking
│       │   └── sampler.py          # Sampling/generation utilities
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py          # PyTorch Dataset implementation
│       │   ├── collator.py         # Batch collation with padding
│       │   ├── loader.py           # DataLoader factory with sampling weights
│       │   └── transforms.py       # Data augmentation/preprocessing
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # Main training loop
│       │   ├── optimizer.py        # Optimizer and scheduler factory
│       │   ├── metrics.py          # Training/eval metrics
│       │   └── checkpoint.py       # Checkpointing utilities
│       ├── encoding/
│       │   ├── __init__.py
│       │   ├── encoder.py          # Encoding API
│       │   └── pooling.py          # Pooling strategies
│       ├── logging/
│       │   ├── __init__.py
│       │   └── wandb_logger.py     # WandB integration
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── config.py           # Hydra config utilities
│       │   └── seed.py             # Reproducibility utilities
│       └── train.py                # Training entry point for accelerate
└── tests/
    ├── README.md                   # Test documentation
    ├── conftest.py                 # Pytest fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_vocab.py
    │   ├── test_rope.py
    │   ├── test_attention.py
    │   ├── test_ffn.py
    │   ├── test_embeddings.py
    │   ├── test_transformer.py
    │   ├── test_noise_schedule.py
    │   ├── test_masking.py
    │   ├── test_dataset.py
    │   ├── test_collator.py
    │   └── test_metrics.py
    ├── integration/
    │   ├── __init__.py
    │   ├── test_model_forward.py
    │   ├── test_diffusion_pipeline.py
    │   ├── test_data_pipeline.py
    │   └── test_encoding.py
    └── e2e/
        ├── __init__.py
        ├── test_training_run.py
        └── fixtures/
            └── toy_data.csv
```

---

## 3. Package Configuration

### 3.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dab"
version = "0.1.0"
description = "Discrete Diffusion Antibody Language Model"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
keywords = ["antibody", "protein", "language-model", "diffusion", "transformer"]

dependencies = [
    "torch>=2.0.0",
    "accelerate>=0.25.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "click>=8.0.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "numpy>=1.24.0",
    "wandb>=0.16.0",
    "tqdm>=4.65.0",
    "einops>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
dab = "dab.cli:main"

[project.urls]
Homepage = "https://github.com/your-org/DAb"
Documentation = "https://github.com/your-org/DAb#readme"
Repository = "https://github.com/your-org/DAb"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
dab = ["py.typed"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "e2e: end-to-end tests",
]
```

### 3.2 .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Hydra
outputs/
multirun/

# WandB
wandb/

# Checkpoints
checkpoints/
*.pt
*.ckpt

# Data
data/
*.csv
*.parquet
!tests/e2e/fixtures/*.csv

# OS
.DS_Store
Thumbs.db
```

---

## 4. Vocabulary and Tokenization

### 4.1 src/dab/vocab.py

```python
"""Vocabulary and tokenization for antibody sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor


@dataclass(frozen=True)
class Vocab:
    """
    Fixed 32-token vocabulary for antibody sequences.
    
    Special tokens:
        - <cls>: Classification/start token (index 0)
        - <pad>: Padding token (index 1)
        - <eos>: End of sequence token (index 2)
        - <unk>: Unknown token (index 3)
        - <mask>: Mask token for diffusion (index 31)
    
    Standard amino acids: L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C
    Non-standard: X (any), B (N/D), U (selenocysteine), O (pyrrolysine), Z (Q/E)
    Special characters: . (insertion), - (gap)
    """
    
    TOKENS: ClassVar[list[str]] = [
        "<cls>", "<pad>", "<eos>", "<unk>",
        "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y",
        "M", "H", "W", "C", "X", "B", "U", "O", "Z", ".", "-",
        "<mask>",
    ]
    
    # Special token indices
    CLS_IDX: ClassVar[int] = 0
    PAD_IDX: ClassVar[int] = 1
    EOS_IDX: ClassVar[int] = 2
    UNK_IDX: ClassVar[int] = 3
    MASK_IDX: ClassVar[int] = 31
    
    # Amino acid range (for sampling during generation)
    AA_START_IDX: ClassVar[int] = 4
    AA_END_IDX: ClassVar[int] = 30  # Exclusive
    
    def __post_init__(self) -> None:
        """Validate vocabulary consistency."""
        assert len(self.TOKENS) == 32
        assert self.TOKENS[self.CLS_IDX] == "<cls>"
        assert self.TOKENS[self.PAD_IDX] == "<pad>"
        assert self.TOKENS[self.EOS_IDX] == "<eos>"
        assert self.TOKENS[self.UNK_IDX] == "<unk>"
        assert self.TOKENS[self.MASK_IDX] == "<mask>"
    
    @classmethod
    def size(cls) -> int:
        """Return vocabulary size."""
        return len(cls.TOKENS)
    
    @classmethod
    def token_to_idx(cls, token: str) -> int:
        """Convert a single token to its index."""
        try:
            return cls.TOKENS.index(token)
        except ValueError:
            return cls.UNK_IDX
    
    @classmethod
    def idx_to_token(cls, idx: int) -> str:
        """Convert an index to its token."""
        if 0 <= idx < len(cls.TOKENS):
            return cls.TOKENS[idx]
        return cls.TOKENS[cls.UNK_IDX]
    
    @classmethod
    def encode(cls, sequence: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode a sequence string to token indices.
        
        Args:
            sequence: Amino acid sequence string
            add_special_tokens: Whether to add <cls> and <eos> tokens
            
        Returns:
            List of token indices
        """
        indices = [cls.token_to_idx(aa) for aa in sequence.upper()]
        if add_special_tokens:
            indices = [cls.CLS_IDX] + indices + [cls.EOS_IDX]
        return indices
    
    @classmethod
    def decode(cls, indices: list[int] | Tensor, remove_special_tokens: bool = True) -> str:
        """
        Decode token indices to a sequence string.
        
        Args:
            indices: Token indices (list or tensor)
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded sequence string
        """
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        
        special = {cls.CLS_IDX, cls.PAD_IDX, cls.EOS_IDX, cls.MASK_IDX}
        tokens = []
        for idx in indices:
            if remove_special_tokens and idx in special:
                continue
            tokens.append(cls.idx_to_token(idx))
        return "".join(tokens)
    
    @classmethod
    def get_padding_mask(cls, token_ids: Tensor) -> Tensor:
        """
        Create a boolean mask where True indicates non-padding positions.
        
        Args:
            token_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Boolean mask of shape (batch_size, seq_len)
        """
        return token_ids != cls.PAD_IDX
    
    @classmethod
    def get_special_tokens_mask(cls, token_ids: Tensor) -> Tensor:
        """
        Create a boolean mask where True indicates special token positions.
        
        Args:
            token_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Boolean mask of shape (batch_size, seq_len)
        """
        special_indices = torch.tensor(
            [cls.CLS_IDX, cls.PAD_IDX, cls.EOS_IDX, cls.UNK_IDX, cls.MASK_IDX],
            device=token_ids.device,
        )
        return torch.isin(token_ids, special_indices)


# Module-level instance for convenience
vocab = Vocab()
```

---

## 5. Model Architecture

### 5.1 src/dab/model/rope.py

```python
"""Rotary Position Embeddings (RoPE) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer attention.
    
    RoPE encodes position information by rotating query and key vectors
    in 2D subspaces, enabling relative position awareness without 
    explicit position embeddings in the input.
    
    Args:
        dim: Dimension of each attention head (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression of frequencies
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute sin/cos cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos cache for given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)
            position_ids: Optional position indices of shape (batch, seq_len)
        
        Returns:
            Tuple of rotated (query, key) tensors
        """
        seq_len = q.shape[2]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        if position_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            cos = self.cos_cached.squeeze(0).squeeze(0)
            sin = self.sin_cached.squeeze(0).squeeze(0)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
```

### 5.2 src/dab/model/ffn.py

```python
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
```

### 5.3 src/dab/model/attention.py

```python
"""
Hybrid Self-Attention and Cross-Attention with chain masking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

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
        self.scale = head_dim ** -0.5
        
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
```

### 5.4 src/dab/model/embeddings.py

```python
"""Token and timestep embeddings for the diffusion model."""

from __future__ import annotations

import math
from typing import Optional

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
        hidden_dim: Optional[int] = None,
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
        timesteps: Optional[Tensor] = None,
    ) -> Tensor:
        embeddings = self.token_embedding(token_ids)
        
        if self.use_timestep_embedding and timesteps is not None:
            timestep_emb = self.timestep_embedding(timesteps)
            embeddings = embeddings + timestep_emb.unsqueeze(1)
        
        return self.dropout(embeddings)
```

### 5.5 src/dab/model/layers.py

```python
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
        
        self.layers = nn.ModuleList([
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
        ])
        
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
```

### 5.6 src/dab/model/transformer.py

```python
"""Main DAb transformer model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
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
        return_hidden_states: bool = False,
    ) -> dict[str, Tensor]:
        hidden_states = self.embeddings(token_ids, timesteps)
        
        if return_hidden_states:
            hidden_states, all_hidden_states = self.encoder(
                hidden_states, chain_ids, attention_mask, return_all_hidden_states=True
            )
        else:
            hidden_states = self.encoder(hidden_states, chain_ids, attention_mask)
            all_hidden_states = None
        
        logits = self.lm_head(hidden_states)
        
        output = {"logits": logits, "hidden_states": hidden_states}
        if all_hidden_states is not None:
            output["all_hidden_states"] = all_hidden_states
        
        return output
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
        return n_params
    
    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "DAbModel":
        checkpoint = torch.load(path, map_location=map_location)
        config = DAbConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save_pretrained(self, path: str) -> None:
        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
        }, path)
```

### 5.7 src/dab/model/__init__.py

```python
"""DAb model components."""

from .attention import EfficientChainAwareAttention
from .embeddings import DAbEmbedding, LearnedTimestepEmbedding, SinusoidalTimestepEmbedding, TokenEmbedding
from .ffn import FusedSwiGLUFFN, SwiGLU, SwiGLUFFN
from .layers import PreNormBlock, TransformerEncoder
from .rope import RotaryPositionEmbedding
from .transformer import DAbConfig, DAbModel

__all__ = [
    "DAbModel", "DAbConfig", "PreNormBlock", "TransformerEncoder",
    "EfficientChainAwareAttention", "SwiGLU", "SwiGLUFFN", "FusedSwiGLUFFN",
    "TokenEmbedding", "SinusoidalTimestepEmbedding", "LearnedTimestepEmbedding",
    "DAbEmbedding", "RotaryPositionEmbedding",
]
```

---

## 6. Discrete Diffusion

### 6.1 src/dab/diffusion/noise_schedule.py

```python
"""Noise schedules for discrete diffusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import math

import torch
from torch import Tensor


class ScheduleType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""
    
    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps
    
    @abstractmethod
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        pass
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)


class LinearSchedule(NoiseSchedule):
    """mask_rate(t) = t / T"""
    
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        return timestep / self.num_timesteps


class CosineSchedule(NoiseSchedule):
    """mask_rate(t) = 1 - cos((t/T) * π/2)"""
    
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return 1 - torch.cos(t_normalized * math.pi / 2)
        return 1 - math.cos(t_normalized * math.pi / 2)


class SqrtSchedule(NoiseSchedule):
    """mask_rate(t) = sqrt(t / T)"""
    
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return torch.sqrt(t_normalized)
        return math.sqrt(t_normalized)


def create_schedule(schedule_type: str | ScheduleType, num_timesteps: int, **kwargs) -> NoiseSchedule:
    if isinstance(schedule_type, str):
        schedule_type = ScheduleType(schedule_type.lower())
    
    if schedule_type == ScheduleType.LINEAR:
        return LinearSchedule(num_timesteps)
    elif schedule_type == ScheduleType.COSINE:
        return CosineSchedule(num_timesteps)
    elif schedule_type == ScheduleType.SQRT:
        return SqrtSchedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
```

### 6.2 src/dab/diffusion/masking.py

```python
"""Information-weighted masking for discrete diffusion."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..vocab import Vocab
from .noise_schedule import NoiseSchedule


class InformationWeightedMasker:
    """
    Applies masking with preference for high-information positions.
    
    Weights: Non-templated CDR = 2, Templated CDR or Non-templated non-CDR = 1, Templated non-CDR = 0
    With multiplier=1.0: Non-templated CDR ~3x more likely than templated non-CDR.
    """
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        weight_multiplier: float = 1.0,
        mask_token_id: int = Vocab.MASK_IDX,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.weight_multiplier = weight_multiplier
        self.mask_token_id = mask_token_id
    
    def compute_weights(
        self,
        cdr_mask: Optional[Tensor],
        non_templated_mask: Optional[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        
        weights = torch.ones(batch_size, seq_len, device=device)
        
        if cdr_mask is not None:
            weights = weights + cdr_mask.float() * self.weight_multiplier
        
        if non_templated_mask is not None:
            weights = weights + non_templated_mask.float() * self.weight_multiplier
        
        weights = weights * attention_mask.float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return weights / weights_sum
    
    def apply_mask(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        cdr_mask: Optional[Tensor] = None,
        non_templated_mask: Optional[Tensor] = None,
        special_tokens_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        mask_rates = self.noise_schedule.get_mask_rate(timesteps)
        valid_counts = attention_mask.sum(dim=-1)
        
        if special_tokens_mask is not None:
            special_counts = (special_tokens_mask & attention_mask.bool()).sum(dim=-1)
            valid_counts = valid_counts - special_counts
        
        num_to_mask = (valid_counts.float() * mask_rates).round().long().clamp(min=0)
        
        maskable_positions = attention_mask.clone()
        if special_tokens_mask is not None:
            maskable_positions = maskable_positions & ~special_tokens_mask
        
        weights = self.compute_weights(cdr_mask, non_templated_mask, maskable_positions)
        
        noise = torch.rand_like(weights) * 1e-6
        scores = weights + noise
        scores = scores.masked_fill(~maskable_positions.bool(), float("-inf"))
        
        _, indices = scores.sort(dim=-1, descending=True)
        
        position_ranks = torch.zeros_like(indices)
        position_ranks.scatter_(
            dim=-1,
            index=indices,
            src=torch.arange(seq_len, device=device).expand(batch_size, -1)
        )
        
        mask_labels = position_ranks < num_to_mask.unsqueeze(-1)
        mask_labels = mask_labels & maskable_positions.bool()
        
        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id
        
        return masked_ids, mask_labels


class UniformMasker:
    """Simple uniform random masking without information weighting."""
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        mask_token_id: int = Vocab.MASK_IDX,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
    
    def apply_mask(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        special_tokens_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        mask_rates = self.noise_schedule.get_mask_rate(timesteps)
        rand = torch.rand(batch_size, seq_len, device=device)
        
        maskable = attention_mask.bool()
        if special_tokens_mask is not None:
            maskable = maskable & ~special_tokens_mask.bool()
        
        mask_labels = (rand < mask_rates.unsqueeze(-1)) & maskable
        
        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id
        
        return masked_ids, mask_labels
```

### 6.3 src/dab/diffusion/sampler.py

```python
"""Sampling utilities for generation/denoising."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..vocab import Vocab
from .noise_schedule import NoiseSchedule


class DiffusionSampler:
    """Sampler for generating sequences via iterative denoising."""
    
    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def _sample_from_logits(self, logits: Tensor) -> Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        if self.top_k is not None:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits[indices_to_remove] = float("-inf")
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[:-1])
    
    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        batch_size: int,
        seq_len: int,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
        num_steps: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tensor:
        num_steps = num_steps or self.noise_schedule.num_timesteps
        
        token_ids = torch.full((batch_size, seq_len), Vocab.MASK_IDX, dtype=torch.long, device=device)
        token_ids[:, 0] = Vocab.CLS_IDX
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        timesteps = torch.linspace(num_steps, 1, num_steps, device=device).long()
        
        iterator = tqdm(timesteps, desc="Sampling", disable=not show_progress)
        for t in iterator:
            current_rate = self.noise_schedule.get_mask_rate(t)
            next_rate = self.noise_schedule.get_mask_rate(t - 1) if t > 1 else 0.0
            
            is_masked = token_ids == Vocab.MASK_IDX
            outputs = model(token_ids, chain_ids, attention_mask)
            sampled = self._sample_from_logits(outputs["logits"])
            
            confidence = outputs["logits"].max(dim=-1).values
            confidence = confidence.masked_fill(~is_masked, float("-inf"))
            
            valid_mask = attention_mask.bool()
            num_valid = valid_mask.sum(dim=-1).float()
            num_to_unmask = ((current_rate - next_rate) * num_valid).round().long()
            
            for i in range(batch_size):
                if num_to_unmask[i] > 0:
                    masked_positions = is_masked[i].nonzero(as_tuple=True)[0]
                    if len(masked_positions) > 0:
                        k = min(num_to_unmask[i].item(), len(masked_positions))
                        top_confident = confidence[i, masked_positions].topk(k).indices
                        unmask_positions = masked_positions[top_confident]
                        token_ids[i, unmask_positions] = sampled[i, unmask_positions]
        
        return token_ids
```

### 6.4 src/dab/diffusion/__init__.py

```python
"""Discrete diffusion components."""

from .masking import InformationWeightedMasker, UniformMasker
from .noise_schedule import CosineSchedule, LinearSchedule, NoiseSchedule, ScheduleType, SqrtSchedule, create_schedule
from .sampler import DiffusionSampler

__all__ = [
    "NoiseSchedule", "LinearSchedule", "CosineSchedule", "SqrtSchedule",
    "ScheduleType", "create_schedule", "InformationWeightedMasker",
    "UniformMasker", "DiffusionSampler",
]
```

---

## 7. Data Loading

### 7.1 src/dab/data/dataset.py

```python
"""PyTorch Dataset for antibody sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class AntibodyDataset(Dataset):
    """
    Dataset for paired antibody heavy/light chain sequences.
    
    Reads data from CSV or Parquet files with columns:
    - heavy_chain, light_chain (required)
    - heavy_cdr_mask, light_cdr_mask (optional)
    - heavy_non_templated_mask, light_non_templated_mask (optional)
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 320,
        heavy_col: str = "heavy_chain",
        light_col: str = "light_chain",
        heavy_cdr_col: str = "heavy_cdr_mask",
        light_cdr_col: str = "light_cdr_mask",
        heavy_nt_col: str = "heavy_non_templated_mask",
        light_nt_col: str = "light_non_templated_mask",
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length
        
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy_cdr_col = heavy_cdr_col
        self.light_cdr_col = light_cdr_col
        self.heavy_nt_col = heavy_nt_col
        self.light_nt_col = light_nt_col
        
        self.df = self._load_data()
        
        if heavy_col not in self.df.columns or light_col not in self.df.columns:
            raise ValueError(f"Missing required columns: {heavy_col}, {light_col}")
        
        self.has_cdr_mask = heavy_cdr_col in self.df.columns and light_cdr_col in self.df.columns
        self.has_nt_mask = heavy_nt_col in self.df.columns and light_nt_col in self.df.columns
    
    def _load_data(self) -> pd.DataFrame:
        if self.data_path.suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        elif self.data_path.suffix in [".csv", ".tsv"]:
            sep = "\t" if self.data_path.suffix == ".tsv" else ","
            return pd.read_csv(self.data_path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _parse_mask(self, mask_str: str) -> Optional[list[int]]:
        if pd.isna(mask_str):
            return None
        if isinstance(mask_str, str):
            return [int(x) for x in mask_str.split(",")]
        return list(mask_str)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        
        result = {
            "heavy_chain": row[self.heavy_col],
            "light_chain": row[self.light_col],
        }
        
        if self.has_cdr_mask:
            result["heavy_cdr_mask"] = self._parse_mask(row[self.heavy_cdr_col])
            result["light_cdr_mask"] = self._parse_mask(row[self.light_cdr_col])
        else:
            result["heavy_cdr_mask"] = None
            result["light_cdr_mask"] = None
        
        if self.has_nt_mask:
            result["heavy_non_templated_mask"] = self._parse_mask(row[self.heavy_nt_col])
            result["light_non_templated_mask"] = self._parse_mask(row[self.light_nt_col])
        else:
            result["heavy_non_templated_mask"] = None
            result["light_non_templated_mask"] = None
        
        return result


class MultiDataset(Dataset):
    """Combines multiple datasets with weighted sampling."""
    
    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        self.lengths = {name: len(ds) for name, ds in datasets.items()}
        self.total_length = sum(self.lengths.values())
        
        self._build_index_map()
        
        if weights is None:
            weights = {name: 1.0 for name in self.dataset_names}
        
        total_weight = sum(weights.values())
        self.weights = {name: w / total_weight for name, w in weights.items()}
        self._build_sampling_probs()
    
    def _build_index_map(self) -> None:
        self.index_map = []
        for name in self.dataset_names:
            for local_idx in range(self.lengths[name]):
                self.index_map.append((name, local_idx))
    
    def _build_sampling_probs(self) -> None:
        probs = []
        for name, local_idx in self.index_map:
            prob = self.weights[name] / self.lengths[name]
            probs.append(prob)
        
        total = sum(probs)
        self.sampling_probs = [p / total for p in probs]
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_name, local_idx = self.index_map[idx]
        item = self.datasets[dataset_name][local_idx]
        item["_dataset"] = dataset_name
        return item
    
    def get_sampler_weights(self) -> torch.Tensor:
        return torch.tensor(self.sampling_probs)
```

### 7.2 src/dab/data/collator.py

```python
"""Batch collation for antibody sequences."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from ..vocab import Vocab


class AntibodyCollator:
    """
    Collates antibody sequences into padded batches.
    
    Format: [CLS] heavy light [EOS]
    Chain IDs: 0 for CLS/heavy, 1 for light/EOS
    """
    
    def __init__(self, max_length: int = 320, pad_to_max: bool = False) -> None:
        self.max_length = max_length
        self.pad_to_max = pad_to_max
    
    def _encode_pair(
        self,
        heavy: str,
        light: str,
        heavy_cdr: Optional[list[int]],
        light_cdr: Optional[list[int]],
        heavy_nt: Optional[list[int]],
        light_nt: Optional[list[int]],
    ) -> dict[str, list[int]]:
        heavy_ids = Vocab.encode(heavy, add_special_tokens=False)
        light_ids = Vocab.encode(light, add_special_tokens=False)
        
        token_ids = [Vocab.CLS_IDX] + heavy_ids + light_ids + [Vocab.EOS_IDX]
        chain_ids = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)
        special_mask = [1] + [0] * len(heavy_ids) + [0] * len(light_ids) + [1]
        
        if heavy_cdr is not None and light_cdr is not None:
            cdr_mask = [0] + heavy_cdr + light_cdr + [0]
        else:
            cdr_mask = None
        
        if heavy_nt is not None and light_nt is not None:
            nt_mask = [0] + heavy_nt + light_nt + [0]
        else:
            nt_mask = None
        
        return {
            "token_ids": token_ids,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
            "nt_mask": nt_mask,
            "special_mask": special_mask,
        }
    
    def _pad_sequence(self, seq: list[int], target_len: int, pad_value: int) -> list[int]:
        if len(seq) >= target_len:
            return seq[:target_len]
        return seq + [pad_value] * (target_len - len(seq))
    
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Tensor]:
        encoded = []
        for example in batch:
            enc = self._encode_pair(
                heavy=example["heavy_chain"],
                light=example["light_chain"],
                heavy_cdr=example.get("heavy_cdr_mask"),
                light_cdr=example.get("light_cdr_mask"),
                heavy_nt=example.get("heavy_non_templated_mask"),
                light_nt=example.get("light_non_templated_mask"),
            )
            encoded.append(enc)
        
        lengths = [len(e["token_ids"]) for e in encoded]
        pad_len = self.max_length if self.pad_to_max else min(max(lengths), self.max_length)
        
        token_ids, chain_ids, attention_mask, special_masks = [], [], [], []
        cdr_masks, nt_masks = [], []
        
        has_cdr = encoded[0]["cdr_mask"] is not None
        has_nt = encoded[0]["nt_mask"] is not None
        
        for enc in encoded:
            seq_len = min(len(enc["token_ids"]), pad_len)
            
            token_ids.append(self._pad_sequence(enc["token_ids"], pad_len, Vocab.PAD_IDX))
            chain_ids.append(self._pad_sequence(enc["chain_ids"], pad_len, 0))
            attention_mask.append([1] * seq_len + [0] * (pad_len - seq_len))
            special_masks.append(self._pad_sequence(enc["special_mask"], pad_len, 1))
            
            if has_cdr:
                cdr_masks.append(self._pad_sequence(enc["cdr_mask"], pad_len, 0))
            if has_nt:
                nt_masks.append(self._pad_sequence(enc["nt_mask"], pad_len, 0))
        
        result = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_masks, dtype=torch.bool),
            "cdr_mask": torch.tensor(cdr_masks, dtype=torch.long) if has_cdr else None,
            "non_templated_mask": torch.tensor(nt_masks, dtype=torch.long) if has_nt else None,
        }
        
        return result
```

### 7.3 src/dab/data/loader.py

```python
"""DataLoader factory with support for weighted sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader, WeightedRandomSampler

from .collator import AntibodyCollator
from .dataset import AntibodyDataset, MultiDataset


def create_dataloader(
    data_path: Union[str, Path],
    batch_size: int,
    max_length: int = 320,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    pad_to_max: bool = False,
) -> DataLoader:
    dataset = AntibodyDataset(data_path, max_length=max_length)
    collator = AntibodyCollator(max_length=max_length, pad_to_max=pad_to_max)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )


def create_multi_dataloader(
    data_paths: dict[str, Union[str, Path]],
    weights: Optional[dict[str, float]],
    batch_size: int,
    max_length: int = 320,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    pad_to_max: bool = False,
) -> DataLoader:
    datasets = {
        name: AntibodyDataset(path, max_length=max_length)
        for name, path in data_paths.items()
    }
    
    multi_dataset = MultiDataset(datasets, weights)
    
    sampler = WeightedRandomSampler(
        weights=multi_dataset.get_sampler_weights(),
        num_samples=len(multi_dataset),
        replacement=True,
    )
    
    collator = AntibodyCollator(max_length=max_length, pad_to_max=pad_to_max)
    
    return DataLoader(
        multi_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
```

### 7.4 src/dab/data/transforms.py

```python
"""Data augmentation and preprocessing transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any


class Transform(ABC):
    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        pass


class Compose(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            example = t(example)
        return example


class RandomChainSwap(Transform):
    """Randomly swap heavy and light chains with probability p."""
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.p:
            example = example.copy()
            example["heavy_chain"], example["light_chain"] = example["light_chain"], example["heavy_chain"]
            
            if example.get("heavy_cdr_mask") is not None:
                example["heavy_cdr_mask"], example["light_cdr_mask"] = example["light_cdr_mask"], example["heavy_cdr_mask"]
            
            if example.get("heavy_non_templated_mask") is not None:
                example["heavy_non_templated_mask"], example["light_non_templated_mask"] = \
                    example["light_non_templated_mask"], example["heavy_non_templated_mask"]
        
        return example
```

### 7.5 src/dab/data/__init__.py

```python
"""Data loading components."""

from .collator import AntibodyCollator
from .dataset import AntibodyDataset, MultiDataset
from .loader import create_dataloader, create_multi_dataloader
from .transforms import Compose, RandomChainSwap, Transform

__all__ = [
    "AntibodyDataset", "MultiDataset", "AntibodyCollator",
    "create_dataloader", "create_multi_dataloader",
    "Transform", "Compose", "RandomChainSwap",
]
```

---

## 8. Training Infrastructure

### 8.1 src/dab/training/metrics.py

```python
"""Training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class MetricAccumulator:
    """Accumulates metrics over steps for averaging."""
    
    _values: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    
    def update(self, name: str, value: float, count: int = 1) -> None:
        if name not in self._values:
            self._values[name] = 0.0
            self._counts[name] = 0
        
        self._values[name] += value * count
        self._counts[name] += count
    
    def compute(self, name: str) -> Optional[float]:
        if name not in self._values or self._counts[name] == 0:
            return None
        return self._values[name] / self._counts[name]
    
    def compute_all(self) -> dict[str, float]:
        return {name: self.compute(name) for name in self._values}
    
    def reset(self) -> None:
        self._values.clear()
        self._counts.clear()


def compute_masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask_labels: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute cross-entropy loss only on masked positions."""
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask_labels.view(-1)
    
    loss_per_token = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction="none"
    )
    
    masked_loss = loss_per_token * mask_flat.float()
    
    if reduction == "none":
        return masked_loss.view(batch_size, seq_len)
    elif reduction == "sum":
        return masked_loss.sum()
    else:
        num_masked = mask_flat.sum().clamp(min=1)
        return masked_loss.sum() / num_masked


def compute_accuracy(logits: Tensor, targets: Tensor, mask_labels: Tensor) -> tuple[float, int]:
    """Compute accuracy on masked positions."""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets) & mask_labels
    
    num_correct = correct.sum().item()
    num_total = mask_labels.sum().item()
    
    if num_total == 0:
        return 0.0, 0
    
    return num_correct / num_total, num_total


def compute_perplexity(loss: Tensor) -> Tensor:
    return torch.exp(loss)


@dataclass
class DiffusionMetrics:
    """Container for diffusion training metrics."""
    
    loss: float
    accuracy: float
    perplexity: float
    num_masked_tokens: int
    mask_rate: float
    
    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            "num_masked_tokens": self.num_masked_tokens,
            "mask_rate": self.mask_rate,
        }


def compute_diffusion_metrics(
    logits: Tensor,
    targets: Tensor,
    mask_labels: Tensor,
    attention_mask: Tensor,
) -> DiffusionMetrics:
    loss = compute_masked_cross_entropy(logits, targets, mask_labels)
    accuracy, num_masked = compute_accuracy(logits, targets, mask_labels)
    perplexity = compute_perplexity(loss).item()
    
    valid_tokens = attention_mask.sum().item()
    mask_rate = num_masked / valid_tokens if valid_tokens > 0 else 0.0
    
    return DiffusionMetrics(
        loss=loss.item(),
        accuracy=accuracy,
        perplexity=perplexity,
        num_masked_tokens=num_masked,
        mask_rate=mask_rate,
    )
```

### 8.2 src/dab/training/optimizer.py

```python
"""Optimizer and learning rate scheduler configuration."""

from __future__ import annotations

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR, _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Create AdamW optimizer with proper weight decay separation."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "bias" in name or "layer_norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 100000,
    num_warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """Create learning rate scheduler with warmup."""
    if scheduler_type == "constant":
        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == "linear":
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps)
        decay = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr_ratio, 
                        total_iters=num_training_steps - num_warmup_steps)
        return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[num_warmup_steps])
    
    elif scheduler_type == "cosine":
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps)
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr = base_lr * min_lr_ratio
        cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps])
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]
```

### 8.3 src/dab/training/checkpoint.py

```python
"""Checkpointing utilities for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n: int = 5
    save_best: bool = True
    best_metric: str = "val_loss"
    best_mode: str = "min"


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    def __init__(
        self,
        config: CheckpointConfig,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric_value: Optional[float] = None
        self.saved_checkpoints: list[Path] = []
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.config.best_mode == "min":
            return current < best
        return current > best
    
    def save(
        self,
        step: int,
        epoch: int,
        metrics: Optional[dict[str, float]] = None,
        extra_state: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        
        checkpoint_path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        
        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        if self.config.save_best and metrics is not None:
            metric_value = metrics.get(self.config.best_metric)
            if metric_value is not None:
                if self.best_metric_value is None or self._is_better(metric_value, self.best_metric_value):
                    self.best_metric_value = metric_value
                    best_path = self.save_dir / "best_checkpoint.pt"
                    torch.save(checkpoint, best_path)
        
        return checkpoint_path
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        if load_best:
            path = self.save_dir / "best_checkpoint.pt"
        elif checkpoint_path is not None:
            path = Path(checkpoint_path)
        else:
            checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            path = checkpoints[-1]
        
        checkpoint = torch.load(path, map_location=map_location)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return {
            "step": checkpoint["step"],
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }
    
    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.config.save_every_n_steps == 0
```

### 8.4 src/dab/training/trainer.py

```python
"""Main training loop with Accelerate integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..diffusion import InformationWeightedMasker, NoiseSchedule, UniformMasker
from ..logging import WandbLogger
from ..model import DAbModel
from .checkpoint import CheckpointConfig, CheckpointManager
from .metrics import MetricAccumulator, compute_diffusion_metrics, compute_masked_cross_entropy
from .optimizer import create_optimizer, create_scheduler, get_lr


@dataclass
class TrainingConfig:
    # Duration (step-driven by default)
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    
    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    
    # Diffusion
    noise_schedule: str = "cosine"
    num_timesteps: int = 100
    weight_multiplier: float = 1.0
    
    # Intervals (in steps)
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5
    save_best: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Mixed precision
    mixed_precision: str = "no"


class Trainer:
    """Main trainer class with Accelerate integration."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: DAbModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
    ) -> None:
        self.config = config
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )
        
        self.optimizer = create_optimizer(
            model, lr=config.learning_rate, weight_decay=config.weight_decay, betas=config.betas
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler_type,
            num_training_steps=config.max_steps,
            num_warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            model, self.optimizer, train_dataloader, self.scheduler
        )
        
        self.eval_dataloader = self.accelerator.prepare(eval_dataloader) if eval_dataloader else None
        
        if noise_schedule is None:
            from ..diffusion import create_schedule
            noise_schedule = create_schedule(config.noise_schedule, config.num_timesteps)
        
        self.masker = InformationWeightedMasker(noise_schedule, config.weight_multiplier)
        self.uniform_masker = UniformMasker(noise_schedule)
        
        checkpoint_config = CheckpointConfig(
            save_dir=config.checkpoint_dir,
            save_every_n_steps=config.save_every_n_steps,
            keep_last_n=config.keep_last_n_checkpoints,
            save_best=config.save_best,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config,
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
        )
        
        self.metrics = MetricAccumulator()
        self.global_step = 0
        self.epoch = 0
        self.logger: Optional[WandbLogger] = None
    
    def set_logger(self, logger: WandbLogger) -> None:
        self.logger = logger
    
    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["token_ids"].device
        batch_size = batch["token_ids"].shape[0]
        
        timesteps = self.masker.noise_schedule.sample_timesteps(batch_size, device)
        
        if batch.get("cdr_mask") is not None or batch.get("non_templated_mask") is not None:
            masked_ids, mask_labels = self.masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                cdr_mask=batch.get("cdr_mask"),
                non_templated_mask=batch.get("non_templated_mask"),
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
        else:
            masked_ids, mask_labels = self.uniform_masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
        
        return {"masked_ids": masked_ids, "mask_labels": mask_labels, "timesteps": timesteps}
    
    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask_output = self._apply_masking(batch)
        
        outputs = self.model(
            token_ids=mask_output["masked_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
        )
        
        return loss
    
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_metrics = MetricAccumulator()
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", 
                         disable=not self.accelerator.is_local_main_process):
            mask_output = self._apply_masking(batch)
            
            outputs = self.model(
                token_ids=mask_output["masked_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            metrics = compute_diffusion_metrics(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_output["mask_labels"],
                attention_mask=batch["attention_mask"],
            )
            
            eval_metrics.update("loss", metrics.loss)
            eval_metrics.update("accuracy", metrics.accuracy)
            eval_metrics.update("perplexity", metrics.perplexity)
        
        self.model.train()
        
        return {
            "val_loss": eval_metrics.compute("loss"),
            "val_accuracy": eval_metrics.compute("accuracy"),
            "val_perplexity": eval_metrics.compute("perplexity"),
        }
    
    def train(self) -> None:
        self.model.train()
        
        if self.config.max_epochs is not None:
            steps_per_epoch = len(self.train_dataloader)
            total_steps = self.config.max_epochs * steps_per_epoch
        else:
            total_steps = self.config.max_steps
        
        progress_bar = tqdm(
            total=total_steps, desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.update(self.global_step)
        
        while self.global_step < total_steps:
            self.epoch += 1
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    self.metrics.update("train_loss", loss.item())
                    
                    if self.global_step % self.config.log_every_n_steps == 0:
                        log_metrics = self.metrics.compute_all()
                        log_metrics["learning_rate"] = get_lr(self.optimizer)
                        log_metrics["epoch"] = self.epoch
                        log_metrics["step"] = self.global_step
                        
                        if self.logger is not None:
                            self.logger.log(log_metrics, step=self.global_step)
                        
                        self.metrics.reset()
                    
                    if self.config.eval_every_n_steps > 0 and self.global_step % self.config.eval_every_n_steps == 0:
                        eval_metrics = self.evaluate()
                        if self.logger is not None and eval_metrics:
                            self.logger.log(eval_metrics, step=self.global_step)
                    
                    if self.checkpoint_manager.should_save(self.global_step):
                        eval_metrics = self.evaluate() if self.eval_dataloader else {}
                        self.checkpoint_manager.save(step=self.global_step, epoch=self.epoch, metrics=eval_metrics)
                    
                    if self.global_step >= total_steps:
                        break
        
        progress_bar.close()
        
        if self.accelerator.is_main_process:
            final_metrics = self.evaluate() if self.eval_dataloader else {}
            self.checkpoint_manager.save(step=self.global_step, epoch=self.epoch, metrics=final_metrics)
            
            if self.logger is not None:
                self.logger.finish()
```

### 8.5 src/dab/training/__init__.py

```python
"""Training components."""

from .checkpoint import CheckpointConfig, CheckpointManager
from .metrics import DiffusionMetrics, MetricAccumulator, compute_accuracy, compute_diffusion_metrics, compute_masked_cross_entropy, compute_perplexity
from .optimizer import create_optimizer, create_scheduler, get_lr
from .trainer import Trainer, TrainingConfig

__all__ = [
    "Trainer", "TrainingConfig", "CheckpointManager", "CheckpointConfig",
    "create_optimizer", "create_scheduler", "get_lr",
    "MetricAccumulator", "DiffusionMetrics", "compute_masked_cross_entropy",
    "compute_accuracy", "compute_perplexity", "compute_diffusion_metrics",
]
```

---

## 9. Encoding API

### 9.1 src/dab/encoding/pooling.py

```python
"""Pooling strategies for sequence embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class PoolingType(str, Enum):
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"
    MEAN_MAX = "mean_max"


class PoolingStrategy(ABC):
    @abstractmethod
    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        pass


class MeanPooling(PoolingStrategy):
    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class CLSPooling(PoolingStrategy):
    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return hidden_states[:, 0, :]


class MaxPooling(PoolingStrategy):
    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if attention_mask is None:
            return hidden_states.max(dim=1).values
        
        mask = attention_mask.unsqueeze(-1).bool()
        masked_hidden = hidden_states.masked_fill(~mask, float("-inf"))
        return masked_hidden.max(dim=1).values


class MeanMaxPooling(PoolingStrategy):
    def __init__(self) -> None:
        self.mean_pool = MeanPooling()
        self.max_pool = MaxPooling()
    
    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        mean_out = self.mean_pool(hidden_states, attention_mask)
        max_out = self.max_pool(hidden_states, attention_mask)
        return torch.cat([mean_out, max_out], dim=-1)


def create_pooling(pooling_type: str | PoolingType) -> PoolingStrategy:
    if isinstance(pooling_type, str):
        pooling_type = PoolingType(pooling_type.lower())
    
    if pooling_type == PoolingType.MEAN:
        return MeanPooling()
    elif pooling_type == PoolingType.CLS:
        return CLSPooling()
    elif pooling_type == PoolingType.MAX:
        return MaxPooling()
    elif pooling_type == PoolingType.MEAN_MAX:
        return MeanMaxPooling()
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
```

### 9.2 src/dab/encoding/encoder.py

```python
"""Encoding API for extracting embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from ..data.collator import AntibodyCollator
from ..model import DAbModel
from .pooling import PoolingStrategy, create_pooling


class DAbEncoder:
    """High-level API for encoding antibody sequences."""
    
    def __init__(
        self,
        model: DAbModel,
        device: Union[str, torch.device] = "cpu",
        pooling: Optional[Union[str, PoolingStrategy]] = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        
        if pooling is None:
            self.pooling = None
        elif isinstance(pooling, str):
            self.pooling = create_pooling(pooling)
        else:
            self.pooling = pooling
        
        self.collator = AntibodyCollator(max_length=model.config.max_seq_len, pad_to_max=False)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "cpu",
        pooling: Optional[str] = None,
    ) -> "DAbEncoder":
        model = DAbModel.from_pretrained(model_path, map_location=device)
        return cls(model, device=device, pooling=pooling)
    
    def _prepare_input(self, heavy_chain: str, light_chain: str) -> dict[str, Tensor]:
        example = {
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "heavy_cdr_mask": None,
            "light_cdr_mask": None,
            "heavy_non_templated_mask": None,
            "light_non_templated_mask": None,
        }
        batch = self.collator([example])
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    
    def _prepare_batch(self, heavy_chains: list[str], light_chains: list[str]) -> dict[str, Tensor]:
        examples = [
            {
                "heavy_chain": h, "light_chain": l,
                "heavy_cdr_mask": None, "light_cdr_mask": None,
                "heavy_non_templated_mask": None, "light_non_templated_mask": None,
            }
            for h, l in zip(heavy_chains, light_chains)
        ]
        batch = self.collator(examples)
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    
    @torch.no_grad()
    def encode(
        self,
        heavy_chain: str,
        light_chain: str,
        return_numpy: bool = False,
    ) -> Union[Tensor, np.ndarray]:
        """Encode a single antibody sequence pair."""
        batch = self._prepare_input(heavy_chain, light_chain)
        
        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        hidden_states = outputs["hidden_states"]
        
        if self.pooling is not None:
            embeddings = self.pooling(hidden_states, batch["attention_mask"])
            embeddings = embeddings.squeeze(0)
        else:
            seq_len = batch["attention_mask"].sum().item()
            embeddings = hidden_states[0, :seq_len, :]
        
        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings
    
    @torch.no_grad()
    def encode_batch(
        self,
        heavy_chains: list[str],
        light_chains: list[str],
        return_numpy: bool = False,
        batch_size: int = 32,
    ) -> Union[Tensor, np.ndarray, list]:
        """Encode a batch of antibody sequence pairs."""
        assert len(heavy_chains) == len(light_chains)
        
        all_embeddings = []
        
        for i in range(0, len(heavy_chains), batch_size):
            batch_heavy = heavy_chains[i:i + batch_size]
            batch_light = light_chains[i:i + batch_size]
            
            batch = self._prepare_batch(batch_heavy, batch_light)
            
            outputs = self.model(
                token_ids=batch["token_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            hidden_states = outputs["hidden_states"]
            
            if self.pooling is not None:
                embeddings = self.pooling(hidden_states, batch["attention_mask"])
                all_embeddings.append(embeddings)
            else:
                for j in range(hidden_states.shape[0]):
                    seq_len = batch["attention_mask"][j].sum().item()
                    emb = hidden_states[j, :seq_len, :]
                    if return_numpy:
                        emb = emb.cpu().numpy()
                    all_embeddings.append(emb)
        
        if self.pooling is not None:
            result = torch.cat(all_embeddings, dim=0)
            if return_numpy:
                return result.cpu().numpy()
            return result
        else:
            return all_embeddings
    
    def get_embedding_dim(self) -> int:
        dim = self.model.config.d_model
        if isinstance(self.pooling, MeanMaxPooling):
            return dim * 2
        return dim
```

### 9.3 src/dab/encoding/__init__.py

```python
"""Encoding API for extracting embeddings."""

from .encoder import DAbEncoder
from .pooling import CLSPooling, MaxPooling, MeanMaxPooling, MeanPooling, PoolingStrategy, PoolingType, create_pooling

__all__ = [
    "DAbEncoder", "PoolingStrategy", "PoolingType",
    "MeanPooling", "CLSPooling", "MaxPooling", "MeanMaxPooling", "create_pooling",
]
```

---

## 10. Logging

### 10.1 src/dab/logging/wandb_logger.py

```python
"""Weights & Biases logging integration."""

from __future__ import annotations

from typing import Any, Optional

import wandb


class WandbLogger:
    """Wrapper for WandB logging."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        resume: bool = False,
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            tags=tags,
            notes=notes,
            resume="allow" if resume else None,
        )
    
    def log(self, metrics: dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        wandb.log(metrics, step=step, commit=commit)
    
    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)
    
    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self) -> None:
        wandb.finish()
    
    @property
    def run_id(self) -> str:
        return self.run.id
    
    @property
    def run_url(self) -> str:
        return self.run.url
```

### 10.2 src/dab/logging/__init__.py

```python
"""Logging utilities."""

from .wandb_logger import WandbLogger

__all__ = ["WandbLogger"]
```

---

## 11. Utilities

### 11.1 src/dab/utils/seed.py

```python
"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed: int, device: torch.device = torch.device("cpu")) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
```

### 11.2 src/dab/utils/config.py

```python
"""Configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, path: str | Path) -> None:
    OmegaConf.save(config, path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    return OmegaConf.merge(*configs)


def to_dict(config: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(config, resolve=True)
```

### 11.3 src/dab/utils/__init__.py

```python
"""Utility functions."""

from .config import load_config, merge_configs, save_config, to_dict
from .seed import get_generator, set_seed

__all__ = ["set_seed", "get_generator", "load_config", "save_config", "merge_configs", "to_dict"]
```

---

## 12. Main Package

### 12.1 src/dab/__init__.py

```python
"""DAb: Discrete Diffusion Antibody Language Model."""

from .model import DAbConfig, DAbModel
from .encoding import DAbEncoder
from .vocab import Vocab, vocab
from .version import __version__

__all__ = ["DAbModel", "DAbConfig", "DAbEncoder", "Vocab", "vocab", "__version__"]
```

### 12.2 src/dab/version.py

```python
"""Version information."""

__version__ = "0.1.0"
```

### 12.3 src/dab/__main__.py

```python
"""Entry point for python -m dab."""

from .cli import main

if __name__ == "__main__":
    main()
```

---

## 13. CLI Implementation

### 13.1 src/dab/cli.py

```python
"""Command-line interface for DAb."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def main() -> None:
    """DAb: Discrete Diffusion Antibody Language Model."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--config-dir", type=click.Path(exists=True), default="configs", help="Config directory")
@click.option("--train-data", "-t", type=click.Path(exists=True), required=True, help="Training data path")
@click.option("--eval-data", "-e", type=click.Path(exists=True), help="Evaluation data path")
@click.option("--output-dir", "-o", type=click.Path(), default="outputs", help="Output directory")
@click.option("--name", "-n", default="dab_experiment", help="Experiment name")
@click.option("--resume", type=click.Path(exists=True), help="Checkpoint to resume from")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--wandb/--no-wandb", default=True, help="Enable/disable WandB")
@click.argument("overrides", nargs=-1)
def train(
    config: Optional[str],
    config_dir: str,
    train_data: str,
    eval_data: Optional[str],
    output_dir: str,
    name: str,
    resume: Optional[str],
    seed: int,
    wandb: bool,
    overrides: tuple[str, ...],
) -> None:
    """
    Train a DAb model.
    
    Examples:
    
        dab train --train-data data/train.csv --eval-data data/val.csv
        
        dab train -t data/train.csv model=small training.batch_size=64
    """
    from .train import run_training
    
    run_training(
        config_path=config,
        config_dir=config_dir,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=output_dir,
        name=name,
        resume_from=resume,
        seed=seed,
        use_wandb=wandb,
        overrides=list(overrides),
    )


@main.command()
@click.option("--checkpoint", "-c", type=click.Path(exists=True), required=True, help="Model checkpoint")
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input file")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file (.pt or .npy)")
@click.option("--pooling", "-p", type=click.Choice(["mean", "cls", "max", "mean_max", "none"]), 
              default="none", help="Pooling strategy")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--device", "-d", default=None, help="Device to run on")
def encode(
    checkpoint: str,
    input: str,
    output: str,
    pooling: str,
    batch_size: int,
    device: Optional[str],
) -> None:
    """
    Encode antibody sequences using a trained model.
    
    Examples:
    
        dab encode -c checkpoints/best.pt -i data/seqs.csv -o embeddings.pt
        
        dab encode -c model.pt -i seqs.parquet -o emb.npy --pooling mean
    """
    import pandas as pd
    import torch
    import numpy as np
    
    from .encoding import DAbEncoder
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f"Loading model from {checkpoint}...")
    
    pooling_strategy = None if pooling == "none" else pooling
    encoder = DAbEncoder.from_pretrained(checkpoint, device=device, pooling=pooling_strategy)
    
    click.echo(f"Loading data from {input}...")
    
    if input.endswith(".parquet"):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)
    
    heavy_chains = df["heavy_chain"].tolist()
    light_chains = df["light_chain"].tolist()
    
    click.echo(f"Encoding {len(heavy_chains)} sequences...")
    
    return_numpy = output.endswith(".npy")
    embeddings = encoder.encode_batch(heavy_chains, light_chains, return_numpy=return_numpy, batch_size=batch_size)
    
    click.echo(f"Saving embeddings to {output}...")
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output.endswith(".npy"):
        if isinstance(embeddings, list):
            np.save(output, np.array(embeddings, dtype=object))
        else:
            np.save(output, embeddings)
    elif output.endswith(".pt"):
        torch.save(embeddings, output)
    else:
        raise ValueError(f"Unknown output format: {output}")
    
    click.echo("Done!")


if __name__ == "__main__":
    main()
```

### 13.2 src/dab/train.py

```python
"""Training entry point for use with accelerate launch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .data import create_dataloader
from .diffusion import create_schedule
from .logging import WandbLogger
from .model import DAbConfig, DAbModel
from .training import Trainer, TrainingConfig
from .utils import set_seed


def run_training(
    config_path: Optional[str] = None,
    config_dir: str = "configs",
    train_data: Optional[str] = None,
    eval_data: Optional[str] = None,
    output_dir: str = "outputs",
    name: str = "dab_experiment",
    resume_from: Optional[str] = None,
    seed: int = 42,
    use_wandb: bool = True,
    overrides: Optional[list[str]] = None,
) -> None:
    """Main training function."""
    config_dir_path = Path(config_dir).absolute()
    
    with initialize_config_dir(config_dir=str(config_dir_path), version_base=None):
        override_list = overrides or []
        override_list.extend([f"name={name}", f"seed={seed}", f"output_dir={output_dir}"])
        
        if train_data:
            override_list.append(f"data.train_data={train_data}")
        if eval_data:
            override_list.append(f"data.eval_data={eval_data}")
        
        cfg = compose(config_name="config", overrides=override_list)
    
    print(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.seed)
    
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path / "config.yaml")
    
    # Create model
    model_config = DAbConfig(
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        head_dim=cfg.model.head_dim,
        d_ffn=cfg.model.d_ffn,
        max_seq_len=cfg.model.max_seq_len,
        max_timesteps=cfg.model.max_timesteps,
        use_timestep_embedding=cfg.model.use_timestep_embedding,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        embedding_dropout=cfg.model.embedding_dropout,
    )
    model = DAbModel(model_config)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create data loaders
    train_loader = create_dataloader(
        data_path=cfg.data.train_data,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
        pad_to_max=cfg.data.pad_to_max,
    )
    
    eval_loader = None
    if cfg.data.eval_data:
        eval_loader = create_dataloader(
            data_path=cfg.data.eval_data,
            batch_size=cfg.training.batch_size,
            max_length=cfg.data.max_length,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
            pad_to_max=cfg.data.pad_to_max,
        )
    
    # Create noise schedule and training config
    noise_schedule = create_schedule(cfg.diffusion.schedule_type, cfg.diffusion.num_timesteps)
    
    training_config = TrainingConfig(
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        max_grad_norm=cfg.training.max_grad_norm,
        scheduler_type=cfg.training.scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        min_lr_ratio=cfg.training.min_lr_ratio,
        noise_schedule=cfg.diffusion.schedule_type,
        num_timesteps=cfg.diffusion.num_timesteps,
        weight_multiplier=cfg.diffusion.weight_multiplier,
        log_every_n_steps=cfg.training.log_every_n_steps,
        eval_every_n_steps=cfg.training.eval_every_n_steps,
        save_every_n_steps=cfg.training.save_every_n_steps,
        checkpoint_dir=cfg.training.checkpoint_dir,
        keep_last_n_checkpoints=cfg.training.keep_last_n_checkpoints,
        save_best=cfg.training.save_best,
        seed=cfg.seed,
        mixed_precision=cfg.training.mixed_precision,
    )
    
    # Create trainer
    trainer = Trainer(
        config=training_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        noise_schedule=noise_schedule,
    )
    
    # Set up logging
    if use_wandb and cfg.logging.enabled:
        logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.logging.entity,
            tags=list(cfg.logging.tags) if cfg.logging.tags else None,
            notes=cfg.logging.notes,
        )
        trainer.set_logger(logger)
    
    # Resume if specified
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.checkpoint_manager.load(resume_from)
    
    # Train
    trainer.train()


def main() -> None:
    """Entry point for `accelerate launch -m dab.train`."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", "-t", required=True)
    parser.add_argument("--eval-data", "-e", default=None)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", "-o", default="outputs")
    parser.add_argument("--name", "-n", default="dab_experiment")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    run_training(
        config_dir=args.config_dir,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        name=args.name,
        resume_from=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        overrides=unknown,
    )


if __name__ == "__main__":
    main()
```

---

## 14. Configuration System

### 14.1 configs/config.yaml

```yaml
defaults:
  - model: base
  - training: default
  - data: default
  - diffusion: cosine
  - logging: wandb
  - _self_

name: dab_experiment
seed: 42
output_dir: outputs/${name}
```

### 14.2 configs/model/base.yaml

```yaml
vocab_size: 32
padding_idx: 1

d_model: 256
n_layers: 16
n_heads: 4
head_dim: 64
d_ffn: null

max_seq_len: 320
max_timesteps: 100
use_timestep_embedding: false

dropout: 0.1
attention_dropout: 0.1
embedding_dropout: 0.1
```

### 14.3 configs/model/small.yaml

```yaml
vocab_size: 32
padding_idx: 1

d_model: 128
n_layers: 4
n_heads: 2
head_dim: 64
d_ffn: null

max_seq_len: 320
max_timesteps: 100
use_timestep_embedding: false

dropout: 0.1
attention_dropout: 0.1
embedding_dropout: 0.1
```

### 14.4 configs/model/large.yaml

```yaml
vocab_size: 32
padding_idx: 1

d_model: 512
n_layers: 24
n_heads: 8
head_dim: 64
d_ffn: null

max_seq_len: 320
max_timesteps: 100
use_timestep_embedding: false

dropout: 0.1
attention_dropout: 0.1
embedding_dropout: 0.1
```

### 14.5 configs/training/default.yaml

```yaml
max_steps: 100000
max_epochs: null

batch_size: 32
gradient_accumulation_steps: 1

learning_rate: 1e-4
weight_decay: 0.01
betas: [0.9, 0.999]
max_grad_norm: 1.0

scheduler_type: cosine
warmup_steps: 1000
min_lr_ratio: 0.1

log_every_n_steps: 10
eval_every_n_steps: 500
save_every_n_steps: 1000

checkpoint_dir: ${output_dir}/checkpoints
keep_last_n_checkpoints: 5
save_best: true

mixed_precision: "no"
```

### 14.6 configs/training/debug.yaml

```yaml
max_steps: 100
max_epochs: null

batch_size: 4
gradient_accumulation_steps: 1

learning_rate: 1e-4
weight_decay: 0.01
betas: [0.9, 0.999]
max_grad_norm: 1.0

scheduler_type: constant
warmup_steps: 10
min_lr_ratio: 0.1

log_every_n_steps: 1
eval_every_n_steps: 50
save_every_n_steps: 50

checkpoint_dir: ${output_dir}/checkpoints
keep_last_n_checkpoints: 2
save_best: false

mixed_precision: "no"
```

### 14.7 configs/data/default.yaml

```yaml
train_data: null
eval_data: null
dataset_weights: null

max_length: 320
num_workers: 4
pin_memory: true
drop_last: true
pad_to_max: false

heavy_col: heavy_chain
light_col: light_chain
heavy_cdr_col: heavy_cdr_mask
light_cdr_col: light_cdr_mask
heavy_nt_col: heavy_non_templated_mask
light_nt_col: light_non_templated_mask
```

### 14.8 configs/diffusion/linear.yaml

```yaml
schedule_type: linear
num_timesteps: 100
weight_multiplier: 1.0
```

### 14.9 configs/diffusion/cosine.yaml

```yaml
schedule_type: cosine
num_timesteps: 100
weight_multiplier: 1.0
```

### 14.10 configs/diffusion/sqrt.yaml

```yaml
schedule_type: sqrt
num_timesteps: 100
weight_multiplier: 1.0
```

### 14.11 configs/logging/wandb.yaml

```yaml
enabled: true
project: dab
entity: null
name: ${name}
tags: []
notes: null
```

---

## 15. Testing Suite

### 15.1 tests/README.md

```markdown
# DAb Test Suite

This directory contains the test suite for DAb, organized into three categories.

## Test Categories

### Unit Tests (`tests/unit/`)

Tests for individual components in isolation.

| Test File | Purpose |
|-----------|---------|
| `test_vocab.py` | Vocabulary encoding/decoding, special tokens, masks |
| `test_rope.py` | Rotary position embedding computation and application |
| `test_attention.py` | Self/cross attention, chain masking, attention scores |
| `test_ffn.py` | SwiGLU activation, FFN forward pass, dimensions |
| `test_embeddings.py` | Token embeddings, timestep embeddings, combined embeddings |
| `test_transformer.py` | Full model forward pass, config validation, weight init |
| `test_noise_schedule.py` | Linear/cosine/sqrt schedules, mask rate computation |
| `test_masking.py` | Information-weighted masking, uniform masking |
| `test_dataset.py` | Data loading, column parsing, mask extraction |
| `test_collator.py` | Batch collation, padding, chain ID generation |
| `test_metrics.py` | Loss computation, accuracy, perplexity |

### Integration Tests (`tests/integration/`)

Tests for component interactions and data flow.

| Test File | Purpose |
|-----------|---------|
| `test_model_forward.py` | Full model forward pass with realistic inputs |
| `test_diffusion_pipeline.py` | Noise → mask → forward → loss pipeline |
| `test_data_pipeline.py` | Load → collate → batch data flow |
| `test_encoding.py` | Model → encoder → embeddings pipeline |

### End-to-End Tests (`tests/e2e/`)

Tests for complete workflows.

| Test File | Purpose |
|-----------|---------|
| `test_training_run.py` | Full training loop on toy data (few steps) |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dab --cov-report=html

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run excluding slow tests
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_batch`: A sample collated batch
- `small_model`: A small DAb model
- `small_config`: Configuration for small model
- `noise_schedule`: A cosine noise schedule
- `toy_dataloader`: DataLoader with toy data
```

### 15.2 tests/conftest.py

```python
"""Pytest fixtures for DAb tests."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from dab.data import AntibodyCollator, create_dataloader
from dab.diffusion import create_schedule
from dab.model import DAbConfig, DAbModel


@pytest.fixture
def small_config() -> DAbConfig:
    return DAbConfig(
        vocab_size=32, d_model=64, n_layers=2, n_heads=2, head_dim=32,
        max_seq_len=64, max_timesteps=10, dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0,
    )


@pytest.fixture
def small_model(small_config: DAbConfig) -> DAbModel:
    return DAbModel(small_config)


@pytest.fixture
def sample_sequences() -> list[dict]:
    return [
        {"heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "light_chain": "DIQMTQSPSSLSASVGDRVTITC",
         "heavy_cdr_mask": None, "light_cdr_mask": None, "heavy_non_templated_mask": None, "light_non_templated_mask": None},
        {"heavy_chain": "QVQLQQSGAELARPGASVKMSCKAS", "light_chain": "DIVMTQSPDSLAVSLGERATINC",
         "heavy_cdr_mask": None, "light_cdr_mask": None, "heavy_non_templated_mask": None, "light_non_templated_mask": None},
    ]


@pytest.fixture
def sample_batch(sample_sequences: list[dict]) -> dict[str, torch.Tensor]:
    collator = AntibodyCollator(max_length=64)
    return collator(sample_sequences)


@pytest.fixture
def noise_schedule():
    return create_schedule("cosine", num_timesteps=10)


@pytest.fixture
def toy_data_path(tmp_path: Path) -> Path:
    data = {
        "heavy_chain": ["EVQLVESGGGLVQPGGSLRLSCAAS", "QVQLQQSGAELARPGASVKMSCKAS", "EVQLLESGGGLVQPGGSLRLSCAAS"],
        "light_chain": ["DIQMTQSPSSLSASVGDRVTITC", "DIVMTQSPDSLAVSLGERATINC", "EIVMTQSPATLSVSPGERATLSC"],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "toy_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def toy_dataloader(toy_data_path: Path):
    return create_dataloader(data_path=toy_data_path, batch_size=2, max_length=64, shuffle=False, num_workers=0)
```

### 15.3 tests/unit/test_vocab.py

```python
"""Tests for vocabulary and tokenization."""

import pytest
import torch

from dab.vocab import Vocab


class TestVocab:
    def test_vocab_size(self):
        assert Vocab.size() == 32
    
    def test_special_token_indices(self):
        assert Vocab.CLS_IDX == 0
        assert Vocab.PAD_IDX == 1
        assert Vocab.EOS_IDX == 2
        assert Vocab.UNK_IDX == 3
        assert Vocab.MASK_IDX == 31
    
    def test_token_to_idx(self):
        assert Vocab.token_to_idx("<cls>") == 0
        assert Vocab.token_to_idx("L") == 4
        assert Vocab.token_to_idx("<mask>") == 31
    
    def test_unknown_token(self):
        assert Vocab.token_to_idx("?") == Vocab.UNK_IDX
    
    def test_encode_simple(self):
        sequence = "LAG"
        encoded = Vocab.encode(sequence, add_special_tokens=False)
        assert encoded == [4, 5, 6]
    
    def test_encode_with_special_tokens(self):
        sequence = "LA"
        encoded = Vocab.encode(sequence, add_special_tokens=True)
        assert encoded[0] == Vocab.CLS_IDX
        assert encoded[-1] == Vocab.EOS_IDX
        assert len(encoded) == 4
    
    def test_roundtrip(self):
        sequence = "EVQLVESGGGLVQ"
        encoded = Vocab.encode(sequence, add_special_tokens=False)
        decoded = Vocab.decode(encoded, remove_special_tokens=True)
        assert decoded == sequence
    
    def test_padding_mask(self):
        token_ids = torch.tensor([[0, 4, 5, 1, 1], [0, 4, 5, 6, 2]])
        mask = Vocab.get_padding_mask(token_ids)
        expected = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        assert torch.equal(mask, expected)
```

### 15.4 tests/unit/test_noise_schedule.py

```python
"""Tests for noise schedules."""

import pytest
import torch

from dab.diffusion.noise_schedule import CosineSchedule, LinearSchedule, SqrtSchedule, create_schedule


class TestLinearSchedule:
    def test_endpoints(self):
        schedule = LinearSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(0) == 0.0
        assert schedule.get_mask_rate(100) == 1.0
    
    def test_midpoint(self):
        schedule = LinearSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(50) == 0.5


class TestCosineSchedule:
    def test_endpoints(self):
        schedule = CosineSchedule(num_timesteps=100)
        assert schedule.get_mask_rate(0) == pytest.approx(0.0, abs=1e-6)
        assert schedule.get_mask_rate(100) == pytest.approx(1.0, abs=1e-6)
    
    def test_monotonic(self):
        schedule = CosineSchedule(num_timesteps=100)
        prev_rate = 0.0
        for t in range(1, 101):
            rate = schedule.get_mask_rate(t)
            assert rate > prev_rate
            prev_rate = rate


class TestCreateSchedule:
    def test_create_linear(self):
        schedule = create_schedule("linear", 100)
        assert isinstance(schedule, LinearSchedule)
    
    def test_create_cosine(self):
        schedule = create_schedule("cosine", 100)
        assert isinstance(schedule, CosineSchedule)
    
    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_schedule("invalid", 100)
```

### 15.5 tests/integration/test_model_forward.py

```python
"""Integration tests for model forward pass."""

import pytest
import torch

from dab.model import DAbConfig, DAbModel


class TestModelForward:
    @pytest.fixture
    def model(self) -> DAbModel:
        config = DAbConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32, max_seq_len=64)
        return DAbModel(config)
    
    def test_basic_forward(self, model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.cat([torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)], dim=1).long()
        
        outputs = model(token_ids, chain_ids)
        
        assert "logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 32)
        assert outputs["hidden_states"].shape == (batch_size, seq_len, 64)
    
    def test_forward_with_attention_mask(self, model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -8:] = 0
        
        outputs = model(token_ids, chain_ids, attention_mask=attention_mask)
        assert outputs["logits"].shape == (batch_size, seq_len, 32)
    
    def test_gradient_flow(self, model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()
        
        outputs = model(token_ids, chain_ids)
        loss = outputs["logits"].sum()
        loss.backward()
        
        assert model.embeddings.token_embedding.embedding.weight.grad is not None
```

### 15.6 tests/integration/test_diffusion_pipeline.py

```python
"""Integration tests for diffusion pipeline."""

import pytest
import torch

from dab.diffusion import InformationWeightedMasker, create_schedule
from dab.model import DAbConfig, DAbModel
from dab.training.metrics import compute_masked_cross_entropy
from dab.vocab import Vocab


class TestDiffusionPipeline:
    @pytest.fixture
    def setup(self):
        config = DAbConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32, max_seq_len=64)
        model = DAbModel(config)
        schedule = create_schedule("cosine", num_timesteps=10)
        masker = InformationWeightedMasker(schedule)
        return model, masker, schedule
    
    def test_mask_forward_loss_pipeline(self, setup):
        model, masker, schedule = setup
        batch_size, seq_len = 2, 32
        
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = Vocab.CLS_IDX
        token_ids[:, -1] = Vocab.EOS_IDX
        
        chain_ids = torch.cat([torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)], dim=1).long()
        attention_mask = torch.ones(batch_size, seq_len)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, -1] = True
        
        timesteps = schedule.sample_timesteps(batch_size, token_ids.device)
        masked_ids, mask_labels = masker.apply_mask(
            token_ids=token_ids, timesteps=timesteps, attention_mask=attention_mask, special_tokens_mask=special_tokens_mask
        )
        
        outputs = model(masked_ids, chain_ids, attention_mask)
        loss = compute_masked_cross_entropy(outputs["logits"], token_ids, mask_labels)
        
        assert not torch.isnan(loss)
        assert loss > 0
        assert mask_labels.sum() > 0
```

### 15.7 tests/e2e/test_training_run.py

```python
"""End-to-end training test."""

import pytest
import torch

from dab.data import create_dataloader
from dab.diffusion import create_schedule
from dab.model import DAbConfig, DAbModel
from dab.training import Trainer, TrainingConfig


@pytest.mark.e2e
@pytest.mark.slow
class TestTrainingRun:
    @pytest.fixture
    def toy_data(self, tmp_path):
        import pandas as pd
        data = {"heavy_chain": ["EVQLVESGGGLVQPGG"] * 20, "light_chain": ["DIQMTQSPSSLSAS"] * 20}
        df = pd.DataFrame(data)
        path = tmp_path / "train.csv"
        df.to_csv(path, index=False)
        return path
    
    @pytest.fixture
    def tiny_config(self):
        return DAbConfig(d_model=32, n_layers=1, n_heads=1, head_dim=32, max_seq_len=64, dropout=0.0)
    
    def test_training_runs(self, toy_data, tiny_config, tmp_path):
        model = DAbModel(tiny_config)
        train_loader = create_dataloader(toy_data, batch_size=4, max_length=64, shuffle=True, num_workers=0)
        
        training_config = TrainingConfig(
            max_steps=5, batch_size=4, learning_rate=1e-4, warmup_steps=1,
            log_every_n_steps=1, eval_every_n_steps=0, save_every_n_steps=0,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        
        trainer = Trainer(config=training_config, model=model, train_dataloader=train_loader)
        trainer.train()
        
        assert trainer.global_step == 5
```

### 15.8 tests/e2e/fixtures/toy_data.csv

```csv
heavy_chain,light_chain
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,DIQMTQSPSSLSASVGDRVTITC
QVQLQQSGAELARPGASVKMSCKASGYTFT,DIVMTQSPDSLAVSLGERATINC
EVQLLESGGGLVQPGGSLRLSCAASGFTFS,EIVMTQSPATLSVSPGERATLSC
QVQLVQSGAEVKKPGASVKVSCKASGYTFT,DIQMTQSPSSLSASVGDRVTITC
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,DIVMTQSPDSLAVSLGERATINC
```

---

## 16. Implementation Order

### Phase 1: Core Infrastructure
1. Set up repository structure
2. Create `pyproject.toml` and package configuration
3. Implement `vocab.py`
4. Create Hydra configs

### Phase 2: Model Architecture
1. Implement RoPE (`rope.py`)
2. Implement SwiGLU FFN (`ffn.py`)
3. Implement chain-aware attention (`attention.py`)
4. Implement embeddings (`embeddings.py`)
5. Implement transformer layers (`layers.py`)
6. Implement main model (`transformer.py`)
7. Write unit tests for each component

### Phase 3: Diffusion Components
1. Implement noise schedules (`noise_schedule.py`)
2. Implement masking (`masking.py`)
3. Implement sampler (`sampler.py`)
4. Write unit tests

### Phase 4: Data Pipeline
1. Implement dataset (`dataset.py`)
2. Implement collator (`collator.py`)
3. Implement dataloader factory (`loader.py`)
4. Implement transforms (`transforms.py`)
5. Write integration tests

### Phase 5: Training Infrastructure
1. Implement metrics (`metrics.py`)
2. Implement optimizer utilities (`optimizer.py`)
3. Implement checkpointing (`checkpoint.py`)
4. Implement trainer (`trainer.py`)
5. Implement WandB logger
6. Write integration tests

### Phase 6: CLI and Encoding
1. Implement pooling strategies (`pooling.py`)
2. Implement encoder API (`encoder.py`)
3. Implement CLI (`cli.py`)
4. Implement training entry point (`train.py`)
5. Write integration tests

### Phase 7: Testing and Polish
1. Complete end-to-end tests
2. Run full test suite
3. Fix any bugs
4. Documentation review
5. Final cleanup

---

## Appendix: Quick Start

After implementation, users can:

```bash
# Install
pip install -e .

# Train
dab train --train-data data/train.csv --eval-data data/val.csv --name my_experiment

# With Accelerate for multi-GPU
accelerate launch -m dab.train --train-data data/train.csv

# Encode sequences
dab encode --checkpoint checkpoints/best.pt --input seqs.csv --output embeddings.pt

# Python API
from dab import DAbEncoder

encoder = DAbEncoder.from_pretrained("checkpoints/best.pt", pooling="mean")
embedding = encoder.encode("EVQLVES...", "DIQMTQ...")
```
