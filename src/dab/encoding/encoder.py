"""Encoding API for extracting embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..data.collator import AntibodyCollator
from ..model import DAbModel
from .pooling import MeanMaxPooling, PoolingStrategy, create_pooling


class DAbEncoder:
    """High-level API for encoding antibody sequences.

    Parameters
    ----------
    model
        Trained DAbModel instance.
    device
        Device to run inference on.
    pooling
        Pooling strategy. Can be a string ("mean", "cls", "max", "mean_max")
        or a PoolingStrategy instance. If None, returns full sequence embeddings.

    Examples
    --------
    >>> encoder = DAbEncoder.from_pretrained("model.pt", pooling="mean")
    >>> embedding = encoder.encode("EVQLV...", "DIQMT...")
    >>> embeddings = encoder.encode_batch(heavy_list, light_list)
    """

    def __init__(
        self,
        model: DAbModel,
        device: str | torch.device = "cpu",
        pooling: str | PoolingStrategy | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)

        if pooling is None:
            self.pooling = None
        elif isinstance(pooling, str):
            self.pooling = create_pooling(pooling)
        else:
            self.pooling = pooling

        self.collator = AntibodyCollator(
            max_length=model.config.max_seq_len, pad_to_max=False
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "cpu",
        pooling: str | None = None,
    ) -> "DAbEncoder":
        """Load an encoder from a pretrained checkpoint.

        Parameters
        ----------
        model_path
            Path to the model checkpoint.
        device
            Device to load the model on.
        pooling
            Pooling strategy to use.

        Returns
        -------
        DAbEncoder
            Encoder instance.
        """
        model = DAbModel.from_pretrained(str(model_path), map_location=device)
        return cls(model, device=device, pooling=pooling)

    def _prepare_input(self, heavy_chain: str, light_chain: str) -> dict[str, Tensor]:
        """Prepare a single sequence pair for encoding."""
        example = {
            "heavy_chain": heavy_chain,
            "light_chain": light_chain,
            "heavy_cdr_mask": None,
            "light_cdr_mask": None,
            "heavy_non_templated_mask": None,
            "light_non_templated_mask": None,
        }
        batch = self.collator([example])
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _prepare_batch(
        self, heavy_chains: list[str], light_chains: list[str]
    ) -> dict[str, Tensor]:
        """Prepare a batch of sequence pairs for encoding."""
        examples = [
            {
                "heavy_chain": h,
                "light_chain": l,
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            }
            for h, l in zip(heavy_chains, light_chains)
        ]
        batch = self.collator(examples)
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def encode(
        self,
        heavy_chain: str,
        light_chain: str,
        return_numpy: bool = False,
    ) -> Tensor | np.ndarray:
        """Encode a single antibody sequence pair.

        Parameters
        ----------
        heavy_chain
            Heavy chain amino acid sequence.
        light_chain
            Light chain amino acid sequence.
        return_numpy
            If True, return numpy array instead of tensor.

        Returns
        -------
        Tensor or np.ndarray
            If pooling is set, returns shape (hidden_dim,) or (hidden_dim*2,) for mean_max.
            If pooling is None, returns shape (seq_len, hidden_dim).
        """
        batch = self._prepare_input(heavy_chain, light_chain)

        outputs = self.model(
            token_ids=batch["token_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        hidden_states = outputs["hidden_states"]

        if self.pooling is not None:
            embeddings = self.pooling(hidden_states, batch["attention_mask"])
            embeddings = embeddings.squeeze(0)
        else:
            seq_len = int(batch["attention_mask"].sum().item())
            embeddings = hidden_states[0, :seq_len, :]

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        heavy_chains: list[str],
        light_chains: list[str],
        return_numpy: bool = False,
        batch_size: int = 32,
    ) -> Tensor | np.ndarray | list:
        """Encode a batch of antibody sequence pairs.

        Parameters
        ----------
        heavy_chains
            List of heavy chain sequences.
        light_chains
            List of light chain sequences.
        return_numpy
            If True, return numpy arrays instead of tensors.
        batch_size
            Batch size for processing.

        Returns
        -------
        Tensor, np.ndarray, or list
            If pooling is set, returns stacked embeddings of shape (n, hidden_dim).
            If pooling is None, returns a list of variable-length embeddings.
        """
        if len(heavy_chains) != len(light_chains):
            raise ValueError(
                f"Number of heavy chains ({len(heavy_chains)}) must match "
                f"number of light chains ({len(light_chains)})"
            )

        all_embeddings = []

        for i in range(0, len(heavy_chains), batch_size):
            batch_heavy = heavy_chains[i : i + batch_size]
            batch_light = light_chains[i : i + batch_size]

            batch = self._prepare_batch(batch_heavy, batch_light)

            outputs = self.model(
                token_ids=batch["token_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            hidden_states = outputs["hidden_states"]

            if self.pooling is not None:
                embeddings = self.pooling(hidden_states, batch["attention_mask"])
                all_embeddings.append(embeddings)
            else:
                for j in range(hidden_states.shape[0]):
                    seq_len = int(batch["attention_mask"][j].sum().item())
                    emb = hidden_states[j, :seq_len, :]
                    if return_numpy:
                        emb = emb.cpu().numpy()
                    all_embeddings.append(emb)

        if self.pooling is not None:
            result = torch.cat(all_embeddings, dim=0)
            if return_numpy:
                return result.cpu().numpy()
            return result
        else:
            return all_embeddings

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension.

        Returns
        -------
        int
            The dimension of the output embeddings.
        """
        dim = self.model.config.d_model
        if isinstance(self.pooling, MeanMaxPooling):
            return dim * 2
        return dim
