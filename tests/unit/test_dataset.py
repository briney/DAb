"""Tests for dataset classes."""

import pytest
import pandas as pd

from dab.data.dataset import AntibodyDataset, MultiDataset


class TestAntibodyDataset:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        data = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL", "QVQLQQSGAELARPGASVKM"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT", "DIVMTQSPDSLAVSLGERAT"],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_data.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def sample_csv_with_masks(self, tmp_path):
        data = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL", "QVQLQQSGAELARPGASVKM"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT", "DIVMTQSPDSLAVSLGERAT"],
            "heavy_cdr_mask": ["0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0", "0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0"],
            "light_cdr_mask": ["0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0"],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_data_masks.csv"
        df.to_csv(path, index=False)
        return path

    def test_load_csv(self, sample_csv):
        dataset = AntibodyDataset(sample_csv)
        assert len(dataset) == 2

    def test_getitem(self, sample_csv):
        dataset = AntibodyDataset(sample_csv)
        item = dataset[0]

        assert "heavy_chain" in item
        assert "light_chain" in item
        assert item["heavy_chain"] == "EVQLVESGGGLVQPGGSLRL"
        assert item["light_chain"] == "DIQMTQSPSSLSASVGDRVT"

    def test_getitem_with_masks(self, sample_csv_with_masks):
        dataset = AntibodyDataset(sample_csv_with_masks)
        item = dataset[0]

        assert "heavy_cdr_mask" in item
        assert "light_cdr_mask" in item
        assert item["heavy_cdr_mask"] is not None
        assert len(item["heavy_cdr_mask"]) == 20

    def test_missing_columns(self, tmp_path):
        data = {"wrong_col": ["EVQL"]}
        df = pd.DataFrame(data)
        path = tmp_path / "bad_data.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError):
            AntibodyDataset(path)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("dummy")

        with pytest.raises(ValueError):
            AntibodyDataset(path)


class TestMultiDataset:
    @pytest.fixture
    def two_datasets(self, tmp_path):
        # Dataset 1
        data1 = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT"],
        }
        df1 = pd.DataFrame(data1)
        path1 = tmp_path / "data1.csv"
        df1.to_csv(path1, index=False)

        # Dataset 2
        data2 = {
            "heavy_chain": ["QVQLQQSGAELARPGASVKM", "EVQLLESGGGLVQPGGSLRL"],
            "light_chain": ["DIVMTQSPDSLAVSLGERAT", "EIVMTQSPATLSVSPGERAT"],
        }
        df2 = pd.DataFrame(data2)
        path2 = tmp_path / "data2.csv"
        df2.to_csv(path2, index=False)

        ds1 = AntibodyDataset(path1)
        ds2 = AntibodyDataset(path2)

        return {"ds1": ds1, "ds2": ds2}

    def test_combined_length(self, two_datasets):
        multi = MultiDataset(two_datasets)
        assert len(multi) == 3  # 1 + 2

    def test_getitem(self, two_datasets):
        multi = MultiDataset(two_datasets)
        item = multi[0]

        assert "heavy_chain" in item
        assert "_dataset" in item

    def test_sampling_weights(self, two_datasets):
        multi = MultiDataset(two_datasets)
        weights = multi.get_sampler_weights()

        assert len(weights) == 3
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_custom_weights(self, two_datasets):
        multi = MultiDataset(two_datasets, weights={"ds1": 2.0, "ds2": 1.0})

        # ds1 has 1 sample with weight 2/3, ds2 has 2 samples with weight 1/3 total
        weights = multi.get_sampler_weights()
        assert len(weights) == 3
