"""Tests for the main DAb transformer model."""

import pytest
import torch

from dab.model import DAbConfig, DAbModel


class TestDAbConfig:
    def test_default_values(self):
        config = DAbConfig()
        assert config.vocab_size == 32
        assert config.d_model == 256
        assert config.n_layers == 16

    def test_d_ffn_auto_computed(self):
        config = DAbConfig(d_model=64)
        assert config.d_ffn is not None
        assert config.d_ffn > config.d_model
        assert config.d_ffn % 64 == 0

    def test_custom_d_ffn(self):
        config = DAbConfig(d_model=64, d_ffn=256)
        assert config.d_ffn == 256


class TestDAbModel:
    @pytest.fixture
    def config(self):
        return DAbConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
        )

    @pytest.fixture
    def model(self, config):
        return DAbModel(config)

    def test_forward_basic(self, model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

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

        outputs = model(token_ids, chain_ids, attention_mask=attention_mask)

        assert outputs["logits"].shape == (batch_size, seq_len, 32)

    def test_forward_output_hidden_states(self, model, config):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(token_ids, chain_ids, output_hidden_states=True)

        assert "all_hidden_states" in outputs
        # Should be n_layers + 1 (embedding + each layer output)
        assert len(outputs["all_hidden_states"]) == config.n_layers + 1

        # Check shapes
        for hidden_state in outputs["all_hidden_states"]:
            assert hidden_state.shape == (batch_size, seq_len, config.d_model)

    def test_forward_output_attentions(self, model, config):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(token_ids, chain_ids, output_attentions=True)

        assert "attentions" in outputs
        # Should be n_layers attention weight tensors
        assert len(outputs["attentions"]) == config.n_layers

        # Each attention output is a single merged attention weight tensor
        for attn_weights in outputs["attentions"]:
            assert attn_weights.shape == (batch_size, config.n_heads, seq_len, seq_len)

    def test_forward_output_both(self, model, config):
        """Test returning both hidden states and attentions."""
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(
            token_ids, chain_ids, output_hidden_states=True, output_attentions=True
        )

        assert "all_hidden_states" in outputs
        assert "attentions" in outputs
        assert len(outputs["all_hidden_states"]) == config.n_layers + 1
        assert len(outputs["attentions"]) == config.n_layers

    def test_forward_with_multiple_chains(self, model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.cat(
            [
                torch.zeros(batch_size, seq_len // 2),
                torch.ones(batch_size, seq_len // 2),
            ],
            dim=1,
        ).long()

        outputs = model(token_ids, chain_ids)

        assert outputs["logits"].shape == (batch_size, seq_len, 32)

    def test_get_num_params(self, model):
        n_params = model.get_num_params(non_embedding=True)
        assert n_params > 0

        n_params_with_emb = model.get_num_params(non_embedding=False)
        assert n_params_with_emb > n_params

    def test_save_and_load(self, model, tmp_path):
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        loaded_model = DAbModel.from_pretrained(str(save_path))

        # Check configs match
        assert loaded_model.config.d_model == model.config.d_model
        assert loaded_model.config.n_layers == model.config.n_layers

        # Check outputs match
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(token_ids, chain_ids)
            out2 = loaded_model(token_ids, chain_ids)

        assert torch.allclose(out1["logits"], out2["logits"])

    def test_weight_tying(self, model):
        """Test that embedding weights are tied to lm_head."""
        assert model.lm_head.weight is model.embeddings.token_embedding.embedding.weight

    def test_standard_attention_mode(self):
        """Test model with standard MultiHeadAttention instead of ChainAwareAttention."""
        config = DAbConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=False,  # Use standard attention
        )
        model = DAbModel(config)

        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(token_ids, chain_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 32)

    def test_attention_mode_comparison(self):
        """Test that both attention modes produce valid outputs with same config."""
        base_config = dict(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
        )

        # Create both models
        chain_aware_model = DAbModel(DAbConfig(**base_config, use_chain_aware_attention=True))
        standard_model = DAbModel(DAbConfig(**base_config, use_chain_aware_attention=False))

        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, 32, (batch_size, seq_len))
        chain_ids = torch.zeros(batch_size, seq_len).long()

        chain_aware_model.eval()
        standard_model.eval()

        with torch.no_grad():
            out1 = chain_aware_model(token_ids, chain_ids)
            out2 = standard_model(token_ids, chain_ids)

        # Both should produce valid outputs with same shape
        assert out1["logits"].shape == out2["logits"].shape
        assert not torch.isnan(out1["logits"]).any()
        assert not torch.isnan(out2["logits"]).any()
