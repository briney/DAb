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
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "O",
        "Z",
        ".",
        "-",
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
