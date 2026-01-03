"""Integration tests for the diffusion pipeline."""

import pytest
import torch

from dab.diffusion import (
    CosineSchedule,
    DiffusionSampler,
    InformationWeightedMasker,
    UniformMasker,
    create_schedule,
)
from dab.model import DAbConfig, DAbModel
from dab.tokenizer import tokenizer
from dab.training import compute_masked_cross_entropy


@pytest.fixture
def model():
    """Create a small model for testing."""
    config = DAbConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=128,
        max_timesteps=100,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )
    return DAbModel(config)


@pytest.fixture
def noise_schedule():
    """Create a cosine noise schedule."""
    return create_schedule("cosine", num_timesteps=100)


@pytest.fixture
def sample_batch():
    """Create a sample batch with CDR masks."""
    heavy = "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"
    light = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA"

    heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
    light_ids = tokenizer.encode(light, add_special_tokens=False)

    tokens = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
    chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

    # Create CDR mask (positions 10-15 and 25-30 are CDRs)
    cdr_mask = [0] * len(tokens)
    for i in range(10, 16):
        if i < len(tokens):
            cdr_mask[i] = 1
    for i in range(25, 31):
        if i < len(tokens):
            cdr_mask[i] = 1

    # Special tokens mask
    special_mask = [True] + [False] * (len(tokens) - 2) + [True]

    return {
        "token_ids": torch.tensor([tokens, tokens]),
        "chain_ids": torch.tensor([chains, chains]),
        "attention_mask": torch.ones(2, len(tokens)),
        "cdr_mask": torch.tensor([cdr_mask, cdr_mask]),
        "special_tokens_mask": torch.tensor([special_mask, special_mask]),
    }


class TestDiffusionTrainingStep:
    def test_uniform_masking_forward_loss(self, model, noise_schedule, sample_batch):
        """Test full training step with uniform masking."""
        masker = UniformMasker(noise_schedule)

        # Sample timesteps
        batch_size = sample_batch["token_ids"].shape[0]
        timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

        # Apply masking
        masked_ids, mask_labels = masker.apply_mask(
            token_ids=sample_batch["token_ids"],
            timesteps=timesteps,
            attention_mask=sample_batch["attention_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        # Forward pass
        outputs = model(
            token_ids=masked_ids,
            chain_ids=sample_batch["chain_ids"],
            attention_mask=sample_batch["attention_mask"],
        )

        # Compute loss
        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=sample_batch["token_ids"],
            mask_labels=mask_labels,
        )

        assert loss.ndim == 0
        assert loss > 0
        assert not torch.isnan(loss)

    def test_information_weighted_masking_forward_loss(
        self, model, noise_schedule, sample_batch
    ):
        """Test full training step with information-weighted masking."""
        # Set seed for reproducibility - ensures timesteps aren't too low
        # (very low timesteps → near-zero mask rate → no masked tokens → zero loss)
        torch.manual_seed(42)

        masker = InformationWeightedMasker(
            noise_schedule, cdr_weight_multiplier=2.0, nongermline_weight_multiplier=1.0
        )

        batch_size = sample_batch["token_ids"].shape[0]
        timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

        # Apply masking with CDR weighting
        masked_ids, mask_labels = masker.apply_mask(
            token_ids=sample_batch["token_ids"],
            timesteps=timesteps,
            attention_mask=sample_batch["attention_mask"],
            cdr_mask=sample_batch["cdr_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        # Forward pass
        outputs = model(
            token_ids=masked_ids,
            chain_ids=sample_batch["chain_ids"],
            attention_mask=sample_batch["attention_mask"],
        )

        # Compute loss
        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=sample_batch["token_ids"],
            mask_labels=mask_labels,
        )

        assert loss.ndim == 0
        assert loss > 0

    def test_training_step_gradient_update(self, model, noise_schedule, sample_batch):
        """Test that a training step updates parameters."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        masker = UniformMasker(noise_schedule)
        batch_size = sample_batch["token_ids"].shape[0]
        timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

        # Get initial parameters
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Training step
        masked_ids, mask_labels = masker.apply_mask(
            token_ids=sample_batch["token_ids"],
            timesteps=timesteps,
            attention_mask=sample_batch["attention_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            token_ids=masked_ids,
            chain_ids=sample_batch["chain_ids"],
            attention_mask=sample_batch["attention_mask"],
        )

        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=sample_batch["token_ids"],
            mask_labels=mask_labels,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed


class TestDiffusionSampling:
    def test_sampling_pipeline(self, model, noise_schedule, sample_batch):
        """Test the full sampling pipeline."""
        model.eval()
        sampler = DiffusionSampler(noise_schedule)

        # Get dimensions from sample batch
        batch_size = 1
        seq_len = sample_batch["token_ids"].shape[1]
        chain_ids = sample_batch["chain_ids"][0:1]

        # Sample from scratch
        with torch.no_grad():
            sampled = sampler.sample(
                model=model,
                batch_size=batch_size,
                seq_len=seq_len,
                chain_ids=chain_ids,
                num_steps=10,
                show_progress=False,
            )

        # Check output shape and that tokens are valid
        assert sampled.shape == (batch_size, seq_len)
        assert (sampled >= 0).all()
        assert (sampled < 32).all()

    def test_conditional_sampling_preserves_context(
        self, model, noise_schedule, sample_batch
    ):
        """Test that conditional sampling preserves unmasked tokens."""
        model.eval()
        sampler = DiffusionSampler(noise_schedule)

        token_ids = sample_batch["token_ids"][0:1].clone()
        chain_ids = sample_batch["chain_ids"][0:1]

        # Create mask for positions 5-15
        mask_positions = torch.zeros_like(token_ids, dtype=torch.bool)
        mask_positions[0, 5:15] = True
        original_tokens = token_ids.clone()

        with torch.no_grad():
            sampled = sampler.sample_conditional(
                model=model,
                token_ids=token_ids,
                chain_ids=chain_ids,
                mask_positions=mask_positions,
                num_steps=10,
                show_progress=False,
            )

        # Check that unmasked positions are preserved
        assert torch.equal(sampled[0, :5], original_tokens[0, :5])
        assert torch.equal(sampled[0, 15:], original_tokens[0, 15:])


class TestNoiseScheduleIntegration:
    def test_all_schedule_types_work(self, model, sample_batch):
        """Test that all noise schedule types work in the pipeline."""
        for schedule_type in ["linear", "cosine", "sqrt"]:
            schedule = create_schedule(schedule_type, num_timesteps=50)
            masker = UniformMasker(schedule)

            batch_size = sample_batch["token_ids"].shape[0]
            timesteps = schedule.sample_timesteps(batch_size, device="cpu")

            masked_ids, mask_labels = masker.apply_mask(
                token_ids=sample_batch["token_ids"],
                timesteps=timesteps,
                attention_mask=sample_batch["attention_mask"],
                special_tokens_mask=sample_batch["special_tokens_mask"],
            )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=sample_batch["chain_ids"],
                attention_mask=sample_batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=sample_batch["token_ids"],
                mask_labels=mask_labels,
            )

            assert not torch.isnan(loss), f"NaN loss for schedule type: {schedule_type}"
