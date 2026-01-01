"""Tests for masking utilities."""

import pytest
import torch

from dab.diffusion.masking import InformationWeightedMasker, UniformMasker
from dab.diffusion.noise_schedule import LinearSchedule
from dab.vocab import Vocab


class TestUniformMasker:
    @pytest.fixture
    def masker(self):
        schedule = LinearSchedule(num_timesteps=100)
        return UniformMasker(noise_schedule=schedule)

    def test_apply_mask_shape(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([50, 50])
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, timesteps, attention_mask)

        assert masked_ids.shape == token_ids.shape
        assert mask_labels.shape == token_ids.shape

    def test_masked_positions_have_mask_token(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([100, 100])  # High mask rate
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, timesteps, attention_mask)

        # Where mask_labels is True, masked_ids should be MASK_IDX
        assert (masked_ids[mask_labels] == Vocab.MASK_IDX).all()

    def test_respects_attention_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([100, 100])
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -10:] = 0  # Last 10 positions are padding

        _, mask_labels = masker.apply_mask(token_ids, timesteps, attention_mask)

        # Padding positions should not be masked
        assert not mask_labels[:, -10:].any()

    def test_respects_special_tokens_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = Vocab.CLS_IDX
        token_ids[:, -1] = Vocab.EOS_IDX
        timesteps = torch.tensor([100, 100])
        attention_mask = torch.ones(batch_size, seq_len)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, -1] = True

        _, mask_labels = masker.apply_mask(
            token_ids, timesteps, attention_mask, special_tokens_mask=special_tokens_mask
        )

        # Special token positions should not be masked
        assert not mask_labels[:, 0].any()
        assert not mask_labels[:, -1].any()


class TestInformationWeightedMasker:
    @pytest.fixture
    def masker(self):
        schedule = LinearSchedule(num_timesteps=100)
        return InformationWeightedMasker(noise_schedule=schedule, weight_multiplier=1.0)

    def test_apply_mask_shape(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([50, 50])
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, timesteps, attention_mask)

        assert masked_ids.shape == token_ids.shape
        assert mask_labels.shape == token_ids.shape

    def test_compute_weights_uniform(self, masker):
        batch_size, seq_len = 2, 10
        attention_mask = torch.ones(batch_size, seq_len)

        weights = masker.compute_weights(
            cdr_mask=None, non_templated_mask=None, attention_mask=attention_mask
        )

        # Without CDR/NT masks, weights should be uniform
        expected = torch.ones(batch_size, seq_len) / seq_len
        assert torch.allclose(weights, expected)

    def test_compute_weights_with_cdr(self, masker):
        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 5:8] = 1  # Positions 5-7 are CDR

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # CDR positions should have higher weights
        assert weights[0, 5] > weights[0, 0]
        assert weights[0, 6] > weights[0, 0]
        assert weights[0, 7] > weights[0, 0]

    def test_respects_special_tokens_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([100, 100])
        attention_mask = torch.ones(batch_size, seq_len)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True

        _, mask_labels = masker.apply_mask(
            token_ids, timesteps, attention_mask, special_tokens_mask=special_tokens_mask
        )

        assert not mask_labels[:, 0].any()

    def test_mask_count_matches_rate(self, masker):
        batch_size, seq_len = 4, 100
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        timesteps = torch.tensor([50, 50, 50, 50])  # 50% mask rate
        attention_mask = torch.ones(batch_size, seq_len)

        _, mask_labels = masker.apply_mask(token_ids, timesteps, attention_mask)

        # Should mask approximately 50 tokens per sequence
        mask_counts = mask_labels.sum(dim=-1)
        assert (mask_counts >= 40).all()  # Allow some tolerance
        assert (mask_counts <= 60).all()
