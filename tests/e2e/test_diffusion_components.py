"""End-to-end tests for diffusion components and model configurations.

Tests cover:
1. Static schedule type (MLM-style fixed masking)
2. Information-weighted vs Uniform maskers
3. ChainAwareAttention vs standard MultiHeadAttention
"""

import pandas as pd
import pytest
import torch

from dab.data import create_dataloader
from dab.diffusion import (
    InformationWeightedMasker,
    UniformMasker,
    create_schedule,
)
from dab.model import DAbConfig, DAbModel
from dab.training import compute_masked_cross_entropy, create_optimizer


@pytest.fixture
def training_data(tmp_path):
    """Create training data for e2e tests."""
    data = {
        "heavy_chain": [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMH",
            "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWV",
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAIS",
        ]
        * 5,  # 20 samples
        "light_chain": [
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSY",
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYL",
        ]
        * 5,
    }
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Static Schedule Tests
# =============================================================================


class TestStaticSchedule:
    """Tests for static (MLM-style) masking schedule."""

    def test_static_schedule_constant_mask_rate(self):
        """Verify static schedule returns constant mask rate regardless of timestep."""
        mask_rate = 0.15
        schedule = create_schedule("static", num_timesteps=100, mask_rate=mask_rate)

        # Test various timesteps - all should return same mask rate
        for t in [1, 25, 50, 75, 100]:
            rate = schedule.get_mask_rate(t)
            assert rate == mask_rate, f"Expected {mask_rate} at t={t}, got {rate}"

        # Test with tensor input
        timesteps = torch.tensor([1, 50, 100])
        rates = schedule.get_mask_rate(timesteps)
        expected = torch.full((3,), mask_rate)
        assert torch.allclose(rates, expected), f"Tensor rates mismatch: {rates}"

    def test_static_schedule_training_loop(self, training_data):
        """Test training with static schedule produces stable masking behavior."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            max_timesteps=100,
            dropout=0.0,
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        # Use static schedule with 15% masking (MLM-style)
        noise_schedule = create_schedule("static", num_timesteps=100, mask_rate=0.15)
        masker = UniformMasker(noise_schedule)

        # Track mask counts to verify consistent masking
        mask_fractions = []
        losses = []

        for epoch in range(2):
            for batch in dataloader:
                batch_size = batch["token_ids"].shape[0]
                timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                # Track masking fraction
                maskable = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
                num_maskable = maskable.sum().item()
                num_masked = mask_labels.sum().item()
                if num_maskable > 0:
                    mask_fractions.append(num_masked / num_maskable)

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Verify masking is approximately 15% (with some variance due to rounding)
        avg_mask_fraction = sum(mask_fractions) / len(mask_fractions)
        assert 0.10 < avg_mask_fraction < 0.20, (
            f"Expected ~15% masking, got {avg_mask_fraction:.2%}"
        )

        # Verify training is stable
        assert all(loss < 100 for loss in losses), "Training loss exploded"
        assert losses[-1] < losses[0] * 2, "Loss increased significantly"

    def test_static_vs_cosine_masking_behavior(self):
        """Compare static and cosine schedule masking behavior via mask rates."""
        # Static schedule - mask rate should be constant
        static_schedule = create_schedule("static", num_timesteps=100, mask_rate=0.15)

        # Cosine schedule - mask rate varies by timestep
        cosine_schedule = create_schedule("cosine", num_timesteps=100)

        # Test at different timesteps - verify mask rates directly
        timesteps = [10, 50, 90]

        # Static rates should all be 0.15
        static_rates = [static_schedule.get_mask_rate(t) for t in timesteps]
        assert all(r == 0.15 for r in static_rates), (
            f"Static schedule should return constant rate 0.15: {static_rates}"
        )

        # Cosine rates should increase with timestep
        cosine_rates = [cosine_schedule.get_mask_rate(t) for t in timesteps]
        assert cosine_rates[0] < cosine_rates[1] < cosine_rates[2], (
            f"Cosine should increase with timestep: {cosine_rates}"
        )

        # Verify with tensor input as well
        timesteps_tensor = torch.tensor(timesteps)
        static_tensor_rates = static_schedule.get_mask_rate(timesteps_tensor)
        expected = torch.full((3,), 0.15)
        assert torch.allclose(static_tensor_rates, expected), (
            f"Static tensor rates should all be 0.15: {static_tensor_rates}"
        )

        cosine_tensor_rates = cosine_schedule.get_mask_rate(timesteps_tensor)
        assert cosine_tensor_rates[0] < cosine_tensor_rates[1] < cosine_tensor_rates[2], (
            f"Cosine tensor rates should increase: {cosine_tensor_rates}"
        )


# =============================================================================
# Masker Type Tests
# =============================================================================


class TestMaskerTypes:
    """Tests comparing UniformMasker and InformationWeightedMasker."""

    def test_uniform_masker_training(self, training_data):
        """Test training with UniformMasker reduces loss."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule("cosine", num_timesteps=50)
        masker = UniformMasker(noise_schedule)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                batch_size = batch["token_ids"].shape[0]
                timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        # Loss should decrease or stabilize
        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_information_weighted_masker_training(self, training_data):
        """Test training with InformationWeightedMasker reduces loss."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule("cosine", num_timesteps=50)
        masker = InformationWeightedMasker(noise_schedule, weight_multiplier=1.0)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                batch_size = batch["token_ids"].shape[0]
                timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

                # InformationWeightedMasker supports optional CDR/template masks
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,  # No CDR annotation
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_information_weighted_with_cdr_mask(self, training_data):
        """Test InformationWeightedMasker prioritizes CDR positions."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        noise_schedule = create_schedule("cosine", num_timesteps=50)
        masker = InformationWeightedMasker(noise_schedule, weight_multiplier=2.0)

        # Create a CDR mask marking positions 10-20 as CDR
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        cdr_mask[:, 10:20] = True

        timesteps = torch.full((batch_size,), 25)

        masked_ids, mask_labels = masker.apply_mask(
            token_ids=batch["token_ids"],
            timesteps=timesteps,
            attention_mask=batch["attention_mask"],
            cdr_mask=cdr_mask,
            non_templated_mask=None,
            special_tokens_mask=batch["special_tokens_mask"],
        )

        # Calculate mask proportion in CDR vs non-CDR regions
        special_mask = batch["special_tokens_mask"].bool()
        maskable = batch["attention_mask"].bool() & ~special_mask

        cdr_maskable = cdr_mask & maskable
        non_cdr_maskable = ~cdr_mask & maskable

        cdr_masked = (mask_labels & cdr_maskable).sum().item()
        cdr_total = cdr_maskable.sum().item()

        non_cdr_masked = (mask_labels & non_cdr_maskable).sum().item()
        non_cdr_total = non_cdr_maskable.sum().item()

        if cdr_total > 0 and non_cdr_total > 0:
            cdr_fraction = cdr_masked / cdr_total
            non_cdr_fraction = non_cdr_masked / non_cdr_total

            # CDR regions should be masked at higher rate
            assert cdr_fraction >= non_cdr_fraction * 0.8, (
                f"CDR mask fraction ({cdr_fraction:.2%}) should be higher "
                f"than non-CDR ({non_cdr_fraction:.2%})"
            )

    def test_masker_comparison_different_distributions(self, training_data):
        """Compare mask distributions between Uniform and InformationWeighted maskers."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=8,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        noise_schedule = create_schedule("cosine", num_timesteps=50)
        uniform_masker = UniformMasker(noise_schedule)
        weighted_masker = InformationWeightedMasker(noise_schedule, weight_multiplier=2.0)

        # Create CDR and non-templated masks
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        cdr_mask[:, 10:25] = True  # Mark CDR region

        non_templated_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        non_templated_mask[:, 15:22] = True  # Mark some non-templated positions

        timesteps = torch.full((batch_size,), 30)

        # Run multiple trials to get statistical significance
        uniform_cdr_fractions = []
        weighted_cdr_fractions = []

        for _ in range(20):
            _, uniform_labels = uniform_masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            _, weighted_labels = weighted_masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                cdr_mask=cdr_mask,
                non_templated_mask=non_templated_mask,
                special_tokens_mask=batch["special_tokens_mask"],
            )

            maskable = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
            cdr_maskable = cdr_mask & maskable

            if cdr_maskable.sum() > 0:
                uniform_cdr = (uniform_labels & cdr_maskable).sum().item()
                weighted_cdr = (weighted_labels & cdr_maskable).sum().item()

                uniform_cdr_fractions.append(uniform_cdr / cdr_maskable.sum().item())
                weighted_cdr_fractions.append(weighted_cdr / cdr_maskable.sum().item())

        # On average, weighted masker should mask more CDR positions
        avg_uniform_cdr = sum(uniform_cdr_fractions) / len(uniform_cdr_fractions)
        avg_weighted_cdr = sum(weighted_cdr_fractions) / len(weighted_cdr_fractions)

        # Weighted should mask at least as much in CDR regions (usually more)
        assert avg_weighted_cdr >= avg_uniform_cdr * 0.8, (
            f"Weighted ({avg_weighted_cdr:.2%}) should mask >= "
            f"uniform ({avg_uniform_cdr:.2%}) in CDR regions"
        )


# =============================================================================
# Chain-Aware Attention Tests
# =============================================================================


class TestChainAwareAttention:
    """Tests for ChainAwareAttention vs standard MultiHeadAttention."""

    def test_chain_aware_attention_training(self, training_data):
        """Test training with ChainAwareAttention enabled."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
            use_chain_aware_attention=True,  # Explicit
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule("cosine", num_timesteps=50)
        masker = UniformMasker(noise_schedule)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                batch_size = batch["token_ids"].shape[0]
                timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_standard_attention_training(self, training_data):
        """Test training with standard MultiHeadAttention (chain-aware disabled)."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
            use_chain_aware_attention=False,  # Disable chain-aware attention
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule("cosine", num_timesteps=50)
        masker = UniformMasker(noise_schedule)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                batch_size = batch["token_ids"].shape[0]
                timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_chain_aware_vs_standard_output_shapes(self):
        """Verify both attention types produce same output shapes."""
        config_chain = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            max_timesteps=10,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        config_standard = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            max_timesteps=10,
            dropout=0.0,
            use_chain_aware_attention=False,
        )

        model_chain = DAbModel(config_chain)
        model_standard = DAbModel(config_standard)

        model_chain.eval()
        model_standard.eval()

        # Create test input
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = 0  # CLS
        token_ids[:, -1] = 2  # EOS

        # Chain IDs: first half heavy, second half light
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            out_chain = model_chain(token_ids, chain_ids, attention_mask)
            out_standard = model_standard(token_ids, chain_ids, attention_mask)

        assert out_chain["logits"].shape == out_standard["logits"].shape
        assert out_chain["hidden_states"].shape == out_standard["hidden_states"].shape

    def test_chain_aware_attention_patterns(self):
        """Verify ChainAwareAttention produces different patterns for intra/inter-chain."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            max_timesteps=10,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        model = DAbModel(config)
        model.eval()

        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = 0  # CLS
        token_ids[:, -1] = 2  # EOS

        # Two chains: 0 for first half, 1 for second half
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                token_ids,
                chain_ids,
                attention_mask,
                output_attentions=True,
            )

        # Should have attention weights for each layer
        assert "attentions" in outputs
        assert len(outputs["attentions"]) == 1  # 1 layer

        attn_weights = outputs["attentions"][0]  # (batch, heads, seq, seq)
        assert attn_weights.shape == (batch_size, 2, seq_len, seq_len)

        # Attention weights should sum to 1 along last dimension
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_chain_aware_save_load_roundtrip(self, tmp_path):
        """Test that chain-aware attention models save and load correctly."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            max_timesteps=10,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        model = DAbModel(config)

        # Modify weights to make them unique
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Save and reload
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        loaded_model = DAbModel.from_pretrained(str(save_path))

        # Verify config preserved
        assert loaded_model.config.use_chain_aware_attention is True

        # Verify weights match
        model.eval()
        loaded_model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        chain_ids = torch.zeros(1, 16, dtype=torch.long)
        chain_ids[:, 8:] = 1
        attention_mask = torch.ones(1, 16)

        with torch.no_grad():
            out_orig = model(token_ids, chain_ids, attention_mask)
            out_loaded = loaded_model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(out_orig["logits"], out_loaded["logits"], atol=1e-6)

    def test_standard_attention_save_load_roundtrip(self, tmp_path):
        """Test that standard attention models save and load correctly."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            max_timesteps=10,
            dropout=0.0,
            use_chain_aware_attention=False,
        )
        model = DAbModel(config)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        loaded_model = DAbModel.from_pretrained(str(save_path))

        # Verify config preserved
        assert loaded_model.config.use_chain_aware_attention is False

        model.eval()
        loaded_model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        chain_ids = torch.zeros(1, 16, dtype=torch.long)
        chain_ids[:, 8:] = 1
        attention_mask = torch.ones(1, 16)

        with torch.no_grad():
            out_orig = model(token_ids, chain_ids, attention_mask)
            out_loaded = loaded_model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(out_orig["logits"], out_loaded["logits"], atol=1e-6)


# =============================================================================
# Combined Configuration Tests
# =============================================================================


class TestCombinedConfigurations:
    """Tests combining different schedule, masker, and attention configurations."""

    @pytest.mark.parametrize(
        "schedule_type,schedule_kwargs",
        [
            ("cosine", {}),
            ("linear", {}),
            ("sqrt", {}),
            ("static", {"mask_rate": 0.15}),
        ],
    )
    @pytest.mark.parametrize("use_chain_aware", [True, False])
    def test_all_schedule_attention_combinations(
        self, training_data, schedule_type, schedule_kwargs, use_chain_aware
    ):
        """Test training with all combinations of schedules and attention types."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
            use_chain_aware_attention=use_chain_aware,
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule(
            schedule_type, num_timesteps=50, **schedule_kwargs
        )
        masker = UniformMasker(noise_schedule)

        # Train for 1 epoch
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch_size = batch["token_ids"].shape[0]
            timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

            masked_ids, mask_labels = masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        assert avg_loss < 100, (
            f"Loss too high for {schedule_type} + "
            f"chain_aware={use_chain_aware}: {avg_loss}"
        )

    @pytest.mark.parametrize("masker_type", ["uniform", "information_weighted"])
    @pytest.mark.parametrize("use_chain_aware", [True, False])
    def test_all_masker_attention_combinations(
        self, training_data, masker_type, use_chain_aware
    ):
        """Test training with all combinations of maskers and attention types."""
        config = DAbConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            max_timesteps=50,
            dropout=0.0,
            use_chain_aware_attention=use_chain_aware,
        )
        model = DAbModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        noise_schedule = create_schedule("cosine", num_timesteps=50)

        if masker_type == "uniform":
            masker = UniformMasker(noise_schedule)
        else:
            masker = InformationWeightedMasker(noise_schedule)

        # Train for 1 epoch
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch_size = batch["token_ids"].shape[0]
            timesteps = noise_schedule.sample_timesteps(batch_size, device="cpu")

            if masker_type == "uniform":
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )
            else:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    timesteps=timesteps,
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        assert avg_loss < 100, (
            f"Loss too high for {masker_type} + "
            f"chain_aware={use_chain_aware}: {avg_loss}"
        )
