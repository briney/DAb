"""Tests for chain-aware attention."""

import pytest
import torch

from dab.model.attention import EfficientChainAwareAttention


class TestEfficientChainAwareAttention:
    @pytest.fixture
    def attention(self):
        return EfficientChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, dropout=0.0, max_seq_len=128
        )

    def test_forward_shape(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)], dim=1
        ).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_with_attention_mask(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[:, -5:] = 0  # Mask last 5 positions

        out = attention(x, chain_ids, attention_mask=attention_mask)
        assert out.shape == x.shape

    def test_single_chain(self, attention):
        """Test with single chain (all same chain_id)."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_multiple_chains(self, attention):
        """Test with multiple chains."""
        batch, seq_len, d_model = 2, 30, 64
        x = torch.randn(batch, seq_len, d_model)
        # Three chains of 10 tokens each
        chain_ids = torch.cat(
            [
                torch.zeros(batch, 10),
                torch.ones(batch, 10),
                torch.full((batch, 10), 2),
            ],
            dim=1,
        ).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_deterministic_without_dropout(self, attention):
        """Test that output is deterministic without dropout."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attention.eval()
        out1 = attention(x, chain_ids)
        out2 = attention(x, chain_ids)

        assert torch.allclose(out1, out2)
