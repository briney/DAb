"""Tests for masking utilities."""

import pytest
import torch

from dab.diffusion.masking import InformationWeightedMasker, UniformMasker
from dab.diffusion.noise_schedule import LinearSchedule
from dab.tokenizer import tokenizer


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
        assert (masked_ids[mask_labels] == tokenizer.mask_token_id).all()

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
        token_ids[:, 0] = tokenizer.cls_token_id
        token_ids[:, -1] = tokenizer.eos_token_id
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
        return InformationWeightedMasker(
            noise_schedule=schedule,
            cdr_weight_multiplier=1.0,
            nongermline_weight_multiplier=1.0,
        )

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

    def test_compute_weights_with_detailed_cdr(self, masker):
        """Test weight computation with detailed CDR mask (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)."""
        batch_size, seq_len = 1, 12
        attention_mask = torch.ones(batch_size, seq_len)
        # Detailed CDR mask: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
        cdr_mask = torch.tensor([[0, 0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0]])

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # All CDR positions (values 1, 2, or 3) should have higher weights than FW (0)
        fw_weight = weights[0, 0].item()
        assert weights[0, 2].item() > fw_weight  # CDR1
        assert weights[0, 3].item() > fw_weight  # CDR1
        assert weights[0, 5].item() > fw_weight  # CDR2
        assert weights[0, 6].item() > fw_weight  # CDR2
        assert weights[0, 8].item() > fw_weight  # CDR3
        assert weights[0, 9].item() > fw_weight  # CDR3
        # All CDR types should have same weight boost
        assert torch.isclose(weights[0, 2], weights[0, 5])
        assert torch.isclose(weights[0, 5], weights[0, 8])

    def test_separate_cdr_and_nongermline_multipliers(self):
        """Test that CDR and nongermline multipliers are applied independently."""
        schedule = LinearSchedule(num_timesteps=100)

        # Create masker with higher CDR weight
        masker_high_cdr = InformationWeightedMasker(
            noise_schedule=schedule,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=0.5,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 2:4] = 1  # CDR at positions 2-3
        nt_mask = torch.zeros(batch_size, seq_len)
        nt_mask[:, 6:8] = 1  # Nongermline at positions 6-7

        weights = masker_high_cdr.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=nt_mask, attention_mask=attention_mask
        )

        # CDR should have weight = 1 + 2.0 = 3.0 (unnormalized)
        # NT should have weight = 1 + 0.5 = 1.5 (unnormalized)
        # FW should have weight = 1.0 (unnormalized)
        # CDR should have higher weight than NT
        assert weights[0, 2] > weights[0, 6]
        assert weights[0, 3] > weights[0, 7]
        # NT should have higher weight than FW
        assert weights[0, 6] > weights[0, 0]

    def test_zero_cdr_multiplier(self):
        """Test that zero CDR multiplier gives CDR same weight as framework."""
        schedule = LinearSchedule(num_timesteps=100)
        masker = InformationWeightedMasker(
            noise_schedule=schedule,
            cdr_weight_multiplier=0.0,
            nongermline_weight_multiplier=1.0,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 5:8] = 1

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # With zero multiplier, all weights should be equal (uniform)
        expected = torch.ones(batch_size, seq_len) / seq_len
        assert torch.allclose(weights, expected)

    def test_high_nongermline_multiplier(self):
        """Test that high nongermline multiplier increases nongermline weight."""
        schedule = LinearSchedule(num_timesteps=100)
        masker = InformationWeightedMasker(
            noise_schedule=schedule,
            cdr_weight_multiplier=1.0,
            nongermline_weight_multiplier=5.0,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 2:4] = 1
        nt_mask = torch.zeros(batch_size, seq_len)
        nt_mask[:, 6:8] = 1

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=nt_mask, attention_mask=attention_mask
        )

        # NT should now have higher weight than CDR (1 + 5.0 vs 1 + 1.0)
        assert weights[0, 6] > weights[0, 2]
        assert weights[0, 7] > weights[0, 3]

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
