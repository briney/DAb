"""Tests for diffusion sampler."""

import pytest
import torch

from dab.diffusion.noise_schedule import LinearSchedule
from dab.diffusion.sampler import DiffusionSampler
from dab.tokenizer import AA_END_IDX, AA_START_IDX, tokenizer


class MockModel:
    """Mock model for testing sampler."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def __call__(self, token_ids, chain_ids, attention_mask=None):
        batch_size, seq_len = token_ids.shape
        # Return random logits, but make amino acids more likely
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        # Boost amino acid logits
        logits[:, :, AA_START_IDX:AA_END_IDX] += 2.0
        return {"logits": logits}


class TestDiffusionSampler:
    @pytest.fixture
    def sampler(self):
        schedule = LinearSchedule(num_timesteps=10)
        return DiffusionSampler(noise_schedule=schedule, temperature=1.0)

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    def test_sample_shape(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()

        output = sampler.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )

        assert output.shape == (batch_size, seq_len)

    def test_sample_starts_with_cls(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()

        output = sampler.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )

        # First token should be CLS
        assert (output[:, 0] == tokenizer.cls_token_id).all()

    def test_sample_mostly_unmasked_at_end(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()

        # Use more steps to ensure better unmasking
        schedule = LinearSchedule(num_timesteps=50)
        sampler_more_steps = DiffusionSampler(noise_schedule=schedule, temperature=1.0)

        output = sampler_more_steps.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )

        # After sampling, most tokens should be unmasked (allow a few remaining due to rounding)
        mask_count = (output == tokenizer.mask_token_id).sum()
        assert mask_count < seq_len * batch_size * 0.1  # Less than 10% remaining masked

    def test_sample_with_attention_mask(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = sampler.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            attention_mask=attention_mask,
            show_progress=False,
        )

        assert output.shape == (batch_size, seq_len)

    def test_temperature_effect(self, mock_model):
        schedule = LinearSchedule(num_timesteps=5)
        sampler_low_temp = DiffusionSampler(noise_schedule=schedule, temperature=0.1)
        sampler_high_temp = DiffusionSampler(noise_schedule=schedule, temperature=2.0)

        batch_size, seq_len = 10, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()

        # With low temperature, outputs should be more deterministic
        # Just verify both work without error
        output_low = sampler_low_temp.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )
        output_high = sampler_high_temp.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )

        assert output_low.shape == output_high.shape

    def test_top_k_sampling(self, mock_model):
        schedule = LinearSchedule(num_timesteps=5)
        sampler = DiffusionSampler(noise_schedule=schedule, top_k=5)

        batch_size, seq_len = 2, 32
        chain_ids = torch.zeros(batch_size, seq_len).long()

        output = sampler.sample(
            model=mock_model,
            batch_size=batch_size,
            seq_len=seq_len,
            chain_ids=chain_ids,
            show_progress=False,
        )

        assert output.shape == (batch_size, seq_len)

    def test_sample_conditional_shape(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = tokenizer.cls_token_id
        chain_ids = torch.zeros(batch_size, seq_len).long()
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_positions[:, 10:20] = True  # Mask positions 10-19

        output = sampler.sample_conditional(
            model=mock_model,
            token_ids=token_ids,
            chain_ids=chain_ids,
            mask_positions=mask_positions,
            show_progress=False,
        )

        assert output.shape == token_ids.shape

    def test_sample_conditional_preserves_unmasked(self, sampler, mock_model):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = tokenizer.cls_token_id
        chain_ids = torch.zeros(batch_size, seq_len).long()
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_positions[:, 10:20] = True

        output = sampler.sample_conditional(
            model=mock_model,
            token_ids=token_ids,
            chain_ids=chain_ids,
            mask_positions=mask_positions,
            show_progress=False,
        )

        # Unmasked positions should be preserved
        unmasked = ~mask_positions
        assert (output[unmasked] == token_ids[unmasked]).all()
