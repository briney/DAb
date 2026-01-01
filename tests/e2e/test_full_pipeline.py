"""End-to-end tests for the complete DAb pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dab import DAbConfig, DAbEncoder, DAbModel
from dab.data import create_dataloader
from dab.diffusion import DiffusionSampler, UniformMasker, create_schedule
from dab.training import compute_masked_cross_entropy, create_optimizer
from dab.vocab import Vocab


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data files."""
    train_data = {
        "heavy_chain": [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMH",
            "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWV",
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAIS",
        ],
        "light_chain": [
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSY",
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYL",
        ],
    }

    train_path = tmp_path / "train.csv"
    pd.DataFrame(train_data).to_csv(train_path, index=False)

    return {"train": train_path}


@pytest.fixture
def trained_model(sample_data, tmp_path):
    """Create and train a small model."""
    # Create model
    config = DAbConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=1,
        head_dim=32,
        max_seq_len=128,
        max_timesteps=50,
        dropout=0.0,
    )
    model = DAbModel(config)
    model.train()

    # Setup training
    dataloader = create_dataloader(
        data_path=sample_data["train"],
        batch_size=2,
        max_length=128,
        num_workers=0,
    )
    optimizer = create_optimizer(model, lr=1e-3)
    noise_schedule = create_schedule("cosine", num_timesteps=50)
    masker = UniformMasker(noise_schedule)

    # Train for a few steps
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

    # Save model
    model_path = tmp_path / "trained_model.pt"
    model.save_pretrained(str(model_path))

    return model_path


class TestTrainEncodeGenerate:
    def test_train_save_load_encode(self, trained_model):
        """Test the full train -> save -> load -> encode pipeline."""
        # Load the trained model
        encoder = DAbEncoder.from_pretrained(trained_model, pooling="mean")

        # Encode some sequences
        heavy = "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"
        light = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA"

        embedding = encoder.encode(heavy, light)

        assert embedding.shape == (32,)  # d_model
        assert not torch.isnan(embedding).any()

    def test_batch_encoding_after_training(self, trained_model):
        """Test batch encoding with a trained model."""
        encoder = DAbEncoder.from_pretrained(trained_model, pooling="mean")

        heavy_chains = [
            "EVQLVESGGGLVQ",
            "QVQLQQSGAELARP",
            "EVQLVQSGAEVKKP",
        ]
        light_chains = [
            "DIQMTQSPSS",
            "DIVMTQSPLS",
            "EIVLTQSPGT",
        ]

        embeddings = encoder.encode_batch(heavy_chains, light_chains, batch_size=2)

        assert embeddings.shape == (3, 32)
        assert not torch.isnan(embeddings).any()

    def test_sampling_with_trained_model(self, trained_model):
        """Test sequence sampling with a trained model."""
        model = DAbModel.from_pretrained(trained_model)
        model.eval()

        noise_schedule = create_schedule("cosine", num_timesteps=50)
        sampler = DiffusionSampler(noise_schedule)

        # Start with a partially masked sequence
        heavy = "EVQLVESGGGLVQ"
        light = "DIQMTQSPSS"

        heavy_ids = Vocab.encode(heavy, add_special_tokens=False)
        light_ids = Vocab.encode(light, add_special_tokens=False)

        tokens = [Vocab.CLS_IDX] + heavy_ids + light_ids + [Vocab.EOS_IDX]
        chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

        token_ids = torch.tensor([tokens])
        chain_ids = torch.tensor([chains])

        # Create mask for CDR-like region (positions 5-10)
        mask_positions = torch.zeros_like(token_ids, dtype=torch.bool)
        mask_positions[0, 5:10] = True

        # Sample conditionally
        with torch.no_grad():
            sampled = sampler.sample_conditional(
                model=model,
                token_ids=token_ids,
                chain_ids=chain_ids,
                mask_positions=mask_positions,
                num_steps=10,
                show_progress=False,
            )

        # Check that we got valid amino acids
        assert sampled.shape == token_ids.shape
        assert (sampled >= 0).all() and (sampled < 32).all()

        # Decode and verify it's valid
        decoded = Vocab.decode(sampled[0].tolist())
        assert len(decoded) > 0


class TestEncoderOutputFormats:
    def test_numpy_output(self, trained_model):
        """Test numpy output format."""
        encoder = DAbEncoder.from_pretrained(trained_model, pooling="mean")

        embedding = encoder.encode(
            "EVQLVES", "DIQMTQ", return_numpy=True
        )

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)

    def test_batch_numpy_output(self, trained_model):
        """Test batch numpy output format."""
        encoder = DAbEncoder.from_pretrained(trained_model, pooling="mean")

        embeddings = encoder.encode_batch(
            ["EVQLVES", "QVQLQQS"],
            ["DIQMTQ", "DIVMTQ"],
            return_numpy=True,
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 32)


class TestDifferentPoolingStrategies:
    def test_all_pooling_strategies(self, trained_model):
        """Test all pooling strategies produce valid outputs."""
        for pooling in ["mean", "cls", "max", "mean_max"]:
            encoder = DAbEncoder.from_pretrained(trained_model, pooling=pooling)

            embedding = encoder.encode("EVQLVESGGGLVQ", "DIQMTQSPSS")

            expected_dim = 64 if pooling == "mean_max" else 32
            assert embedding.shape == (expected_dim,), f"Failed for {pooling}"
            assert not torch.isnan(embedding).any(), f"NaN in {pooling}"

    def test_no_pooling_returns_sequence(self, trained_model):
        """Test that no pooling returns full sequence embeddings."""
        encoder = DAbEncoder.from_pretrained(trained_model, pooling=None)

        heavy = "EVQLVES"
        light = "DIQMTQ"

        embedding = encoder.encode(heavy, light)

        # Should be (seq_len, d_model)
        expected_len = 1 + len(heavy) + len(light) + 1  # CLS + heavy + light + EOS
        assert embedding.shape == (expected_len, 32)


class TestVocabRoundtrip:
    def test_encode_decode_roundtrip(self):
        """Test that sequences survive encode/decode roundtrip."""
        sequences = [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "ACDEFGHIKLMNPQRSTVWY",  # All standard amino acids
        ]

        for seq in sequences:
            encoded = Vocab.encode(seq, add_special_tokens=False)
            decoded = Vocab.decode(encoded)
            assert decoded == seq, f"Roundtrip failed for {seq}"

    def test_special_tokens_handling(self):
        """Test special token handling in encode/decode."""
        seq = "EVQLVES"

        # With special tokens
        with_special = Vocab.encode(seq, add_special_tokens=True)
        assert with_special[0] == Vocab.CLS_IDX
        assert with_special[-1] == Vocab.EOS_IDX

        # Decode should strip special tokens
        decoded = Vocab.decode(with_special)
        assert decoded == seq
