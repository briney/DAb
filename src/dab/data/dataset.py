"""PyTorch Dataset for antibody sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AntibodyDataset(Dataset):
    """
    Dataset for paired antibody heavy/light chain sequences.

    Reads data from CSV or Parquet files with columns:
    - heavy_chain, light_chain (required)
    - heavy_cdr_mask, light_cdr_mask (optional)
    - heavy_non_templated_mask, light_non_templated_mask (optional)
    - heavy_coords, light_coords (optional) - CA atom coordinates
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 320,
        heavy_col: str = "heavy_chain",
        light_col: str = "light_chain",
        heavy_cdr_col: str = "heavy_cdr_mask",
        light_cdr_col: str = "light_cdr_mask",
        heavy_nt_col: str = "heavy_non_templated_mask",
        light_nt_col: str = "light_non_templated_mask",
        load_coords: bool = False,
        heavy_coords_col: str = "heavy_coords",
        light_coords_col: str = "light_coords",
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length

        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy_cdr_col = heavy_cdr_col
        self.light_cdr_col = light_cdr_col
        self.heavy_nt_col = heavy_nt_col
        self.light_nt_col = light_nt_col

        self.load_coords = load_coords
        self.heavy_coords_col = heavy_coords_col
        self.light_coords_col = light_coords_col

        self.df = self._load_data()

        if heavy_col not in self.df.columns or light_col not in self.df.columns:
            raise ValueError(f"Missing required columns: {heavy_col}, {light_col}")

        self.has_cdr_mask = (
            heavy_cdr_col in self.df.columns and light_cdr_col in self.df.columns
        )
        self.has_nt_mask = (
            heavy_nt_col in self.df.columns and light_nt_col in self.df.columns
        )
        self.has_coords = (
            load_coords
            and heavy_coords_col in self.df.columns
            and light_coords_col in self.df.columns
        )

    def _load_data(self) -> pd.DataFrame:
        if self.data_path.suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        elif self.data_path.suffix in [".csv", ".tsv"]:
            sep = "\t" if self.data_path.suffix == ".tsv" else ","
            return pd.read_csv(self.data_path, sep=sep)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _parse_mask(self, mask_str: str) -> list[int] | None:
        if pd.isna(mask_str):
            return None
        if isinstance(mask_str, str):
            return [int(x) for x in mask_str.split(",")]
        return list(mask_str)

    def _parse_coords(self, coords_data: Any) -> np.ndarray | None:
        """Parse coordinate data from various formats.

        Supports:
        - numpy array (N, 3)
        - JSON string of list of lists
        - Comma-separated string of flattened coordinates

        Args:
            coords_data: Raw coordinate data from dataframe.

        Returns:
            Numpy array of shape (N, 3) or None if invalid.
        """
        if coords_data is None or (isinstance(coords_data, float) and pd.isna(coords_data)):
            return None

        if isinstance(coords_data, np.ndarray):
            return coords_data.astype(np.float32)

        if isinstance(coords_data, str):
            # Try JSON format first
            try:
                parsed = json.loads(coords_data)
                return np.array(parsed, dtype=np.float32)
            except json.JSONDecodeError:
                pass

            # Try comma-separated format (flattened)
            try:
                values = [float(x) for x in coords_data.split(",")]
                n_coords = len(values) // 3
                return np.array(values, dtype=np.float32).reshape(n_coords, 3)
            except (ValueError, TypeError):
                return None

        if isinstance(coords_data, (list, tuple)):
            return np.array(coords_data, dtype=np.float32)

        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        result = {
            "heavy_chain": row[self.heavy_col],
            "light_chain": row[self.light_col],
        }

        if self.has_cdr_mask:
            result["heavy_cdr_mask"] = self._parse_mask(row[self.heavy_cdr_col])
            result["light_cdr_mask"] = self._parse_mask(row[self.light_cdr_col])
        else:
            result["heavy_cdr_mask"] = None
            result["light_cdr_mask"] = None

        if self.has_nt_mask:
            result["heavy_non_templated_mask"] = self._parse_mask(row[self.heavy_nt_col])
            result["light_non_templated_mask"] = self._parse_mask(row[self.light_nt_col])
        else:
            result["heavy_non_templated_mask"] = None
            result["light_non_templated_mask"] = None

        if self.has_coords:
            result["heavy_coords"] = self._parse_coords(row[self.heavy_coords_col])
            result["light_coords"] = self._parse_coords(row[self.light_coords_col])
        else:
            result["heavy_coords"] = None
            result["light_coords"] = None

        return result


class MultiDataset(Dataset):
    """Combines multiple datasets with weighted sampling."""

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())

        self.lengths = {name: len(ds) for name, ds in datasets.items()}
        self.total_length = sum(self.lengths.values())

        self._build_index_map()

        if weights is None:
            weights = {name: 1.0 for name in self.dataset_names}

        total_weight = sum(weights.values())
        self.weights = {name: w / total_weight for name, w in weights.items()}
        self._build_sampling_probs()

    def _build_index_map(self) -> None:
        self.index_map = []
        for name in self.dataset_names:
            for local_idx in range(self.lengths[name]):
                self.index_map.append((name, local_idx))

    def _build_sampling_probs(self) -> None:
        probs = []
        for name, local_idx in self.index_map:
            prob = self.weights[name] / self.lengths[name]
            probs.append(prob)

        total = sum(probs)
        self.sampling_probs = [p / total for p in probs]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_name, local_idx = self.index_map[idx]
        item = self.datasets[dataset_name][local_idx]
        item["_dataset"] = dataset_name
        return item

    def get_sampler_weights(self) -> torch.Tensor:
        return torch.tensor(self.sampling_probs)
