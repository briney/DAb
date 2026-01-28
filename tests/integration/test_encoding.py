"""Integration tests for the encoding API."""

import pytest
import torch

from dab import DAbEncoder
from dab.encoding import MeanMaxPooling, create_pooling
from dab.model import DAbConfig, DAbModel


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    config = DAbConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )
    return DAbModel(config)


@pytest.fixture
def encoder(small_model):
    """Create an encoder with no pooling."""
    return DAbEncoder(small_model, device="cpu", pooling=None)


@pytest.fixture
def mean_encoder(small_model):
    """Create an encoder with mean pooling."""
    return DAbEncoder(small_model, device="cpu", pooling="mean")


class TestDAbEncoder:
    def test_encode_single_no_pooling(self, encoder):
        heavy = "EVQLVESGGGLVQPGRSLRLSCAAS"
        light = "DIQMTQSPSSVSASVGDRVTITC"

        embedding = encoder.encode(heavy, light)

        # Should return full sequence embeddings
        assert embedding.ndim == 2
        assert embedding.shape[1] == 64  # d_model
        # Sequence length: CLS + heavy + light + EOS
        expected_len = 1 + len(heavy) + len(light) + 1
        assert embedding.shape[0] == expected_len

    def test_encode_single_with_pooling(self, mean_encoder):
        heavy = "EVQLVESGGGLVQPGRSLRLSCAAS"
        light = "DIQMTQSPSSVSASVGDRVTITC"

        embedding = mean_encoder.encode(heavy, light)

        # Should return pooled embedding
        assert embedding.ndim == 1
        assert embedding.shape[0] == 64  # d_model

    def test_encode_return_numpy(self, mean_encoder):
        heavy = "EVQLVESGGGLVQPGRSLRLSCAAS"
        light = "DIQMTQSPSSVSASVGDRVTITC"

        embedding = mean_encoder.encode(heavy, light, return_numpy=True)

        assert not isinstance(embedding, torch.Tensor)
        assert embedding.shape == (64,)

    def test_encode_batch_with_pooling(self, mean_encoder):
        heavy_chains = [
            "EVQLVESGGGLVQPGRSLRLSCAAS",
            "QVQLQQSGAELARPGAS",
        ]
        light_chains = [
            "DIQMTQSPSSVSASVGDRVTITC",
            "DIVMTQSPLSLPVTPGEPAS",
        ]

        embeddings = mean_encoder.encode_batch(heavy_chains, light_chains)

        assert embeddings.shape == (2, 64)

    def test_encode_batch_no_pooling(self, encoder):
        heavy_chains = [
            "EVQLVESGGGLVQPGRSLRLSCAAS",
            "QVQLQQSGAELARPGAS",
        ]
        light_chains = [
            "DIQMTQSPSSVSASVGDRVTITC",
            "DIVMTQSPLSLPVTPGEPAS",
        ]

        embeddings = encoder.encode_batch(heavy_chains, light_chains)

        # Should return list of variable-length embeddings
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

        # Check individual shapes
        expected_len_0 = 1 + len(heavy_chains[0]) + len(light_chains[0]) + 1
        expected_len_1 = 1 + len(heavy_chains[1]) + len(light_chains[1]) + 1
        assert embeddings[0].shape == (expected_len_0, 64)
        assert embeddings[1].shape == (expected_len_1, 64)

    def test_encode_batch_mismatched_lengths_raises(self, mean_encoder):
        heavy_chains = ["EVQLVES", "QVQLQQS"]
        light_chains = ["DIQMTQ"]  # Only one light chain

        with pytest.raises(ValueError, match="must match"):
            mean_encoder.encode_batch(heavy_chains, light_chains)

    def test_get_embedding_dim(self, small_model):
        encoder_no_pool = DAbEncoder(small_model, pooling=None)
        encoder_mean = DAbEncoder(small_model, pooling="mean")
        encoder_mean_max = DAbEncoder(small_model, pooling="mean_max")

        assert encoder_no_pool.get_embedding_dim() == 64
        assert encoder_mean.get_embedding_dim() == 64
        assert encoder_mean_max.get_embedding_dim() == 128


class TestEncoderWithPoolingStrategy:
    def test_encoder_with_pooling_instance(self, small_model):
        pooling = create_pooling("cls")
        encoder = DAbEncoder(small_model, device="cpu", pooling=pooling)

        heavy = "EVQLVESGGGLVQPGRSLRLSCAAS"
        light = "DIQMTQSPSSVSASVGDRVTITC"

        embedding = encoder.encode(heavy, light)

        assert embedding.shape == (64,)


class TestEncoderBatching:
    def test_batch_size_parameter(self, mean_encoder):
        # Create many sequences
        heavy_chains = ["EVQLVES"] * 10
        light_chains = ["DIQMTQ"] * 10

        # Process with different batch sizes
        emb_bs2 = mean_encoder.encode_batch(heavy_chains, light_chains, batch_size=2)
        emb_bs5 = mean_encoder.encode_batch(heavy_chains, light_chains, batch_size=5)

        # Results should be the same regardless of batch size
        assert torch.allclose(emb_bs2, emb_bs5)

    def test_large_batch(self, mean_encoder):
        # Test with more sequences than batch size
        heavy_chains = ["EVQLVESGGGLVQ"] * 100
        light_chains = ["DIQMTQSPSS"] * 100

        embeddings = mean_encoder.encode_batch(
            heavy_chains, light_chains, batch_size=32
        )

        assert embeddings.shape == (100, 64)


class TestGetAttentions:
    """Tests for the get_attentions method."""

    def test_get_attentions_all_layers(self, encoder):
        """Test getting attention from all layers (default)."""
        heavy = "EVQLVESGGGLVQPGRSLRLSCAAS"
        light = "DIQMTQSPSSVSASVGDRVTITC"

        result = encoder.get_attentions(heavy, light)

        assert "attentions" in result
        assert "n_layers" in result
        assert "seq_len" in result

        attentions = result["attentions"]
        n_layers = result["n_layers"]
        seq_len = result["seq_len"]

        assert isinstance(attentions, tuple)
        assert len(attentions) == n_layers
        assert n_layers == 2  # small_model has 2 layers

        expected_seq_len = 1 + len(heavy) + len(light) + 1
        assert seq_len == expected_seq_len
        for attn in attentions:
            assert attn.shape == (2, seq_len, seq_len)  # 2 heads

    def test_get_attentions_single_layer(self, encoder):
        """Test getting attention from a single layer."""
        result = encoder.get_attentions("EVQLVES", "DIQMTQ", layer=0)
        assert isinstance(result["attentions"], torch.Tensor)
        assert result["attentions"].ndim == 3

    def test_get_attentions_negative_index(self, encoder):
        """Test negative indexing for last layer."""
        result_last = encoder.get_attentions("EVQLVES", "DIQMTQ", layer=-1)
        result_explicit = encoder.get_attentions("EVQLVES", "DIQMTQ", layer=1)
        assert torch.allclose(result_last["attentions"], result_explicit["attentions"])

    def test_get_attentions_layer_list(self, encoder):
        """Test getting attention from multiple layers."""
        result = encoder.get_attentions("EVQLVES", "DIQMTQ", layer=[0, -1])
        assert isinstance(result["attentions"], tuple)
        assert len(result["attentions"]) == 2

    def test_get_attentions_invalid_layer_raises(self, encoder):
        """Test out-of-range layer index raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            encoder.get_attentions("EVQLVES", "DIQMTQ", layer=10)

    def test_get_attentions_sums_to_one(self, encoder):
        """Test attention weights sum to 1 along key dimension."""
        result = encoder.get_attentions("EVQLVES", "DIQMTQ", layer=0)
        attn_sums = result["attentions"].sum(dim=-1)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)
