"""Tests for SwiGLU Feed-Forward Networks."""

import pytest
import torch

from dab.model.ffn import FusedSwiGLUFFN, SwiGLU, SwiGLUFFN


class TestSwiGLU:
    def test_forward(self):
        swiglu = SwiGLU()
        x = torch.randn(2, 10, 64)
        gate = torch.randn(2, 10, 64)
        out = swiglu(x, gate)
        assert out.shape == x.shape


class TestSwiGLUFFN:
    @pytest.fixture
    def ffn(self):
        return SwiGLUFFN(d_model=64, d_ffn=128, dropout=0.0)

    def test_forward_shape(self, ffn):
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_default_d_ffn(self):
        ffn = SwiGLUFFN(d_model=64)
        # d_ffn should be approximately 8/3 * d_model, rounded to nearest 64
        assert ffn.d_ffn % 64 == 0
        assert ffn.d_ffn > ffn.d_model

    def test_custom_d_ffn(self):
        ffn = SwiGLUFFN(d_model=64, d_ffn=256)
        assert ffn.d_ffn == 256


class TestFusedSwiGLUFFN:
    @pytest.fixture
    def ffn(self):
        return FusedSwiGLUFFN(d_model=64, d_ffn=128, dropout=0.0)

    def test_forward_shape(self, ffn):
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_matches_unfused(self):
        """Test that fused and unfused produce same output dimensions."""
        d_model, d_ffn = 64, 128
        fused = FusedSwiGLUFFN(d_model=d_model, d_ffn=d_ffn, dropout=0.0)
        unfused = SwiGLUFFN(d_model=d_model, d_ffn=d_ffn, dropout=0.0)

        x = torch.randn(2, 10, d_model)
        out_fused = fused(x)
        out_unfused = unfused(x)

        assert out_fused.shape == out_unfused.shape
