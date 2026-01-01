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
        expected = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        assert torch.equal(mask, expected)

    def test_special_tokens_mask(self):
        token_ids = torch.tensor([[0, 4, 5, 31, 2]])  # CLS, L, A, MASK, EOS
        mask = Vocab.get_special_tokens_mask(token_ids)
        expected = torch.tensor([[True, False, False, True, True]])
        assert torch.equal(mask, expected)
