"""Tests for embedding layers."""

import math

import pytest
import torch

from dab.model.embeddings import (
    DAbEmbedding,
    LearnedTimestepEmbedding,
    SinusoidalTimestepEmbedding,
    TokenEmbedding,
)


class TestTokenEmbedding:
    @pytest.fixture
    def embedding(self):
        return TokenEmbedding(vocab_size=32, d_model=64, padding_idx=1)

    def test_forward_shape(self, embedding):
        token_ids = torch.randint(0, 32, (2, 10))
        out = embedding(token_ids)
        assert out.shape == (2, 10, 64)

    def test_scaling(self):
        emb_scaled = TokenEmbedding(vocab_size=32, d_model=64, scale=True)
        emb_unscaled = TokenEmbedding(vocab_size=32, d_model=64, scale=False)

        # Make them share the same weights
        emb_unscaled.embedding.weight.data = emb_scaled.embedding.weight.data.clone()

        # Use non-padding tokens to avoid zeros
        token_ids = torch.randint(2, 30, (2, 10))
        out_scaled = emb_scaled(token_ids)
        out_unscaled = emb_unscaled(token_ids)

        # Check that scaled output is sqrt(d_model) times the unscaled output
        expected_ratio = math.sqrt(64)
        assert torch.allclose(out_scaled, out_unscaled * expected_ratio)

    def test_padding_idx(self, embedding):
        token_ids = torch.tensor([[1, 1, 1]])  # All padding
        out = embedding(token_ids)
        assert torch.allclose(out, torch.zeros_like(out))


class TestSinusoidalTimestepEmbedding:
    @pytest.fixture
    def embedding(self):
        return SinusoidalTimestepEmbedding(d_model=64, max_timesteps=100)

    def test_forward_shape(self, embedding):
        timesteps = torch.tensor([0, 10, 50, 99])
        out = embedding(timesteps)
        assert out.shape == (4, 64)

    def test_different_timesteps_different_embeddings(self, embedding):
        t1 = torch.tensor([0])
        t2 = torch.tensor([50])
        out1 = embedding(t1)
        out2 = embedding(t2)
        assert not torch.allclose(out1, out2)


class TestLearnedTimestepEmbedding:
    @pytest.fixture
    def embedding(self):
        return LearnedTimestepEmbedding(d_model=64, max_timesteps=100)

    def test_forward_shape(self, embedding):
        timesteps = torch.tensor([0, 10, 50, 99])
        out = embedding(timesteps)
        assert out.shape == (4, 64)


class TestDAbEmbedding:
    def test_without_timestep(self):
        embedding = DAbEmbedding(
            vocab_size=32,
            d_model=64,
            use_timestep_embedding=False,
        )
        token_ids = torch.randint(0, 32, (2, 10))
        out = embedding(token_ids)
        assert out.shape == (2, 10, 64)

    def test_with_timestep(self):
        embedding = DAbEmbedding(
            vocab_size=32,
            d_model=64,
            use_timestep_embedding=True,
            max_timesteps=100,
        )
        token_ids = torch.randint(0, 32, (2, 10))
        timesteps = torch.tensor([10, 50])
        out = embedding(token_ids, timesteps=timesteps)
        assert out.shape == (2, 10, 64)

    def test_timestep_ignored_when_disabled(self):
        embedding = DAbEmbedding(
            vocab_size=32,
            d_model=64,
            use_timestep_embedding=False,
        )
        token_ids = torch.randint(0, 32, (2, 10))
        timesteps = torch.tensor([10, 50])

        out_with_t = embedding(token_ids, timesteps=timesteps)
        out_without_t = embedding(token_ids)

        assert torch.allclose(out_with_t, out_without_t)
