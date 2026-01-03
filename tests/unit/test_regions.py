"""Tests for antibody region extraction."""

import pytest
import torch

from dab.eval.regions import (
    AntibodyRegion,
    CDR_REGIONS,
    FW_REGIONS,
    HEAVY_REGIONS,
    LIGHT_REGIONS,
    aggregate_region_masks,
    extract_region_masks,
)


class TestAntibodyRegion:
    """Tests for AntibodyRegion enum."""

    def test_all_regions_exist(self):
        """Test that all expected regions are defined."""
        expected = [
            "cdr1_h", "cdr2_h", "cdr3_h",
            "cdr1_l", "cdr2_l", "cdr3_l",
            "fw1_h", "fw2_h", "fw3_h", "fw4_h",
            "fw1_l", "fw2_l", "fw3_l", "fw4_l",
        ]
        actual = [r.value for r in AntibodyRegion]
        assert set(expected) == set(actual)

    def test_cdr_regions_grouping(self):
        """Test CDR_REGIONS contains only CDR regions."""
        for region in CDR_REGIONS:
            assert "cdr" in region.value

    def test_fw_regions_grouping(self):
        """Test FW_REGIONS contains only framework regions."""
        for region in FW_REGIONS:
            assert "fw" in region.value

    def test_heavy_regions_grouping(self):
        """Test HEAVY_REGIONS contains only heavy chain regions."""
        for region in HEAVY_REGIONS:
            assert region.value.endswith("_h")

    def test_light_regions_grouping(self):
        """Test LIGHT_REGIONS contains only light chain regions."""
        for region in LIGHT_REGIONS:
            assert region.value.endswith("_l")


class TestExtractRegionMasks:
    """Tests for extract_region_masks function."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch with CDR and chain annotations.

        Sequence structure (length 40):
        - Position 0: CLS (special)
        - Positions 1-19: Heavy chain
            - 1-3: FW1_H
            - 4-6: CDR1_H
            - 7-9: FW2_H
            - 10-12: CDR2_H
            - 13-15: FW3_H
            - 16-18: CDR3_H
            - 19: FW4_H
        - Positions 20-38: Light chain
            - 20-22: FW1_L
            - 23-25: CDR1_L
            - 26-28: FW2_L
            - 29-31: CDR2_L
            - 32-34: FW3_L
            - 35-37: CDR3_L
            - 38: FW4_L
        - Position 39: EOS (special)
        """
        batch_size = 2
        seq_len = 40

        # Token IDs (just placeholders)
        token_ids = torch.randint(4, 24, (batch_size, seq_len))

        # Attention mask (all valid)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Special tokens mask (CLS at 0, EOS at 39)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, 39] = True

        # Chain IDs: 0 for CLS + heavy, 1 for light + EOS
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, 20:] = 1

        # CDR mask: mark CDR positions as 1
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Heavy chain CDRs
        cdr_mask[:, 4:7] = 1   # CDR1_H (positions 4-6)
        cdr_mask[:, 10:13] = 1  # CDR2_H (positions 10-12)
        cdr_mask[:, 16:19] = 1  # CDR3_H (positions 16-18)
        # Light chain CDRs
        cdr_mask[:, 23:26] = 1  # CDR1_L (positions 23-25)
        cdr_mask[:, 29:32] = 1  # CDR2_L (positions 29-31)
        cdr_mask[:, 35:38] = 1  # CDR3_L (positions 35-37)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
        }

    def test_extract_all_regions(self, sample_batch):
        """Test extracting all region masks."""
        region_masks = extract_region_masks(sample_batch)

        # Should have all 14 regions
        assert len(region_masks) == 14

        # Check CDR1_H positions
        cdr1_h_mask = region_masks[AntibodyRegion.CDR1_H]
        assert cdr1_h_mask.shape == (2, 40)
        assert cdr1_h_mask[0, 4:7].all()  # Positions 4-6 should be True
        assert not cdr1_h_mask[0, :4].any()  # Positions before should be False
        assert not cdr1_h_mask[0, 7:10].any()  # Positions after should be False

    def test_extract_specific_regions(self, sample_batch):
        """Test extracting only specific regions."""
        regions = {AntibodyRegion.CDR1_H, AntibodyRegion.CDR3_L}
        region_masks = extract_region_masks(sample_batch, regions)

        assert len(region_masks) == 2
        assert AntibodyRegion.CDR1_H in region_masks
        assert AntibodyRegion.CDR3_L in region_masks

    def test_cdr_positions_correct(self, sample_batch):
        """Test that CDR positions are correctly identified."""
        region_masks = extract_region_masks(sample_batch)

        # CDR1_H: positions 4-6
        cdr1_h = region_masks[AntibodyRegion.CDR1_H][0]
        assert cdr1_h[4:7].all()
        assert cdr1_h.sum() == 3

        # CDR3_L: positions 35-37
        cdr3_l = region_masks[AntibodyRegion.CDR3_L][0]
        assert cdr3_l[35:38].all()
        assert cdr3_l.sum() == 3

    def test_framework_positions_correct(self, sample_batch):
        """Test that framework positions are correctly inferred."""
        region_masks = extract_region_masks(sample_batch)

        # FW1_H: positions 1-3 (between CLS and CDR1_H)
        fw1_h = region_masks[AntibodyRegion.FW1_H][0]
        assert fw1_h[1:4].all()
        assert fw1_h.sum() == 3

        # FW2_H: positions 7-9 (between CDR1_H and CDR2_H)
        fw2_h = region_masks[AntibodyRegion.FW2_H][0]
        assert fw2_h[7:10].all()
        assert fw2_h.sum() == 3

    def test_no_cdr_mask_raises_error(self):
        """Test that missing cdr_mask raises ValueError."""
        batch = {
            "token_ids": torch.zeros(2, 10, dtype=torch.long),
            "chain_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="cdr_mask is required"):
            extract_region_masks(batch)

    def test_batch_processing(self, sample_batch):
        """Test that batch processing works correctly."""
        region_masks = extract_region_masks(sample_batch)

        # All sequences in batch should have same regions
        for region, mask in region_masks.items():
            assert mask.shape[0] == 2  # batch_size
            # Both sequences should have same pattern
            assert torch.equal(mask[0], mask[1])


class TestAggregateRegionMasks:
    """Tests for aggregate_region_masks function."""

    @pytest.fixture
    def sample_region_masks(self):
        """Create sample region masks for testing aggregation."""
        batch_size = 2
        seq_len = 20

        masks = {}
        for region in AntibodyRegion:
            masks[region] = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Set some positions
        masks[AntibodyRegion.CDR1_H][:, 2:4] = True
        masks[AntibodyRegion.CDR2_H][:, 5:7] = True
        masks[AntibodyRegion.CDR3_H][:, 8:10] = True
        masks[AntibodyRegion.FW1_H][:, 0:2] = True
        masks[AntibodyRegion.FW2_H][:, 4:5] = True

        return masks

    def test_aggregate_all(self, sample_region_masks):
        """Test 'all' aggregation returns individual regions."""
        result = aggregate_region_masks(sample_region_masks, "all")

        assert "cdr1_h" in result
        assert "cdr2_h" in result
        assert "fw1_h" in result
        assert len(result) == len(sample_region_masks)

    def test_aggregate_cdr(self, sample_region_masks):
        """Test 'cdr' aggregation groups CDRs and frameworks."""
        result = aggregate_region_masks(sample_region_masks, "cdr")

        assert "cdr" in result
        assert "fw" in result
        assert len(result) == 2

        # CDR mask should include all CDR positions
        cdr_mask = result["cdr"]
        assert cdr_mask[:, 2:4].all()  # CDR1_H
        assert cdr_mask[:, 5:7].all()  # CDR2_H
        assert cdr_mask[:, 8:10].all()  # CDR3_H

    def test_aggregate_chain(self, sample_region_masks):
        """Test 'chain' aggregation groups by chain."""
        result = aggregate_region_masks(sample_region_masks, "chain")

        assert "heavy" in result
        assert "light" in result
        assert len(result) == 2

    def test_aggregate_region_type(self, sample_region_masks):
        """Test 'region_type' aggregation groups by CDR/FW number."""
        result = aggregate_region_masks(sample_region_masks, "region_type")

        assert "cdr1" in result
        assert "cdr2" in result
        assert "cdr3" in result
        assert "fw1" in result
        assert "fw2" in result

    def test_invalid_aggregate_by(self, sample_region_masks):
        """Test that invalid aggregate_by raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregate_by"):
            aggregate_region_masks(sample_region_masks, "invalid")
