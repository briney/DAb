"""Controlled masking for evaluation with seeded reproducibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..diffusion.masking import InformationWeightedMasker, UniformMasker
from ..diffusion.noise_schedule import create_schedule
from ..tokenizer import tokenizer

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EvalMasker:
    """Controlled masking for evaluation with seeded reproducibility.

    Wraps UniformMasker or InformationWeightedMasker with:
    - Configurable masking strategy (uniform vs information-weighted)
    - Configurable schedule (static vs diffusion schedules)
    - Seeded random generation for reproducibility

    Parameters
    ----------
    masker_type
        Type of masking: "uniform" or "information_weighted".
    schedule_type
        Type of schedule: "static", "cosine", "linear", "sqrt".
    mask_rate
        Target mask rate for static schedule (0.0-1.0).
    num_timesteps
        Number of timesteps for diffusion schedules.
    cdr_weight_multiplier
        Weight multiplier for CDR positions in information-weighted masking.
    nongermline_weight_multiplier
        Weight multiplier for nongermline positions in information-weighted masking.
    seed
        Random seed for reproducibility.
    selection_method
        Selection method for information-weighted masking: "ranked" or "sampled".
    """

    def __init__(
        self,
        masker_type: str = "uniform",
        schedule_type: str = "static",
        mask_rate: float = 0.15,
        num_timesteps: int = 100,
        cdr_weight_multiplier: float = 1.0,
        nongermline_weight_multiplier: float = 1.0,
        seed: int = 42,
        selection_method: str = "sampled",
    ) -> None:
        self.masker_type = masker_type
        self.schedule_type = schedule_type
        self.mask_rate = mask_rate
        self.num_timesteps = num_timesteps
        self.cdr_weight_multiplier = cdr_weight_multiplier
        self.nongermline_weight_multiplier = nongermline_weight_multiplier
        self.seed = seed
        self.selection_method = selection_method

        # Create noise schedule
        self.noise_schedule = create_schedule(
            schedule_type=schedule_type,
            num_timesteps=num_timesteps,
            mask_rate=mask_rate,
        )

        # Create underlying masker
        if masker_type == "uniform":
            self._masker = UniformMasker(
                noise_schedule=self.noise_schedule,
                mask_token_id=tokenizer.mask_token_id,
            )
        elif masker_type == "information_weighted":
            self._masker = InformationWeightedMasker(
                noise_schedule=self.noise_schedule,
                cdr_weight_multiplier=cdr_weight_multiplier,
                nongermline_weight_multiplier=nongermline_weight_multiplier,
                mask_token_id=tokenizer.mask_token_id,
                selection_method=selection_method,
            )
        else:
            raise ValueError(
                f"Unknown masker_type: {masker_type}. "
                "Must be 'uniform' or 'information_weighted'."
            )

    def get_generator(self, device: torch.device) -> torch.Generator:
        """Create a fresh seeded generator for an evaluation run.

        Creates a new generator with the configured seed each time,
        ensuring reproducible masking when called at the start of each
        evaluation run.

        Parameters
        ----------
        device
            Device to create the generator on.

        Returns
        -------
        torch.Generator
            Seeded generator for this device.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample timesteps with optional seeded generator.

        Parameters
        ----------
        batch_size
            Number of timesteps to sample.
        device
            Device to create timesteps on.
        generator
            Optional seeded generator for reproducibility.

        Returns
        -------
        Tensor
            Sampled timesteps of shape (batch_size,).
        """
        return torch.randint(
            1,
            self.num_timesteps + 1,
            (batch_size,),
            device=device,
            generator=generator,
        )

    def apply_mask(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply masking to a batch with optional seeded generator.

        Parameters
        ----------
        batch
            Batch dictionary with at least:
            - token_ids: (batch, seq_len) token IDs
            - attention_mask: (batch, seq_len) attention mask
            - special_tokens_mask: (batch, seq_len) optional special tokens mask
            - cdr_mask: (batch, seq_len) optional CDR mask (for info-weighted)
            - non_templated_mask: (batch, seq_len) optional non-templated mask
        generator
            Optional seeded generator for reproducibility.

        Returns
        -------
        tuple[Tensor, Tensor]
            - masked_ids: Token IDs with masked positions replaced
            - mask_labels: Boolean mask indicating which positions were masked
        """
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        special_tokens_mask = batch.get("special_tokens_mask")
        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Sample timesteps (with generator for reproducibility)
        timesteps = self.sample_timesteps(batch_size, device, generator)

        # Apply masking based on masker type
        if self.masker_type == "uniform":
            # For uniform masking with generator, we need to handle it manually
            # since UniformMasker uses torch.rand internally
            if generator is not None:
                return self._apply_uniform_mask_with_generator(
                    token_ids=token_ids,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    generator=generator,
                )
            return self._masker.apply_mask(
                token_ids=token_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
        else:
            # Information-weighted masking
            cdr_mask = batch.get("cdr_mask")
            non_templated_mask = batch.get("non_templated_mask")

            if generator is not None:
                return self._apply_info_weighted_mask_with_generator(
                    token_ids=token_ids,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    cdr_mask=cdr_mask,
                    non_templated_mask=non_templated_mask,
                    special_tokens_mask=special_tokens_mask,
                    generator=generator,
                )
            return self._masker.apply_mask(
                token_ids=token_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                cdr_mask=cdr_mask,
                non_templated_mask=non_templated_mask,
                special_tokens_mask=special_tokens_mask,
            )

    def _apply_uniform_mask_with_generator(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        special_tokens_mask: Tensor | None,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Apply uniform masking with seeded generator."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        mask_rates = self.noise_schedule.get_mask_rate(timesteps)
        rand = torch.rand(batch_size, seq_len, device=device, generator=generator)

        maskable = attention_mask.bool()
        if special_tokens_mask is not None:
            maskable = maskable & ~special_tokens_mask.bool()

        mask_labels = (rand < mask_rates.unsqueeze(-1)) & maskable

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = tokenizer.mask_token_id

        return masked_ids, mask_labels

    def _apply_info_weighted_mask_with_generator(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        cdr_mask: Tensor | None,
        non_templated_mask: Tensor | None,
        special_tokens_mask: Tensor | None,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Apply information-weighted masking with seeded generator."""
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        mask_rates = self.noise_schedule.get_mask_rate(timesteps)
        valid_counts = attention_mask.sum(dim=-1)

        if special_tokens_mask is not None:
            special_counts = (special_tokens_mask & attention_mask.bool()).sum(dim=-1)
            valid_counts = valid_counts - special_counts

        num_to_mask = (valid_counts.float() * mask_rates).round().long().clamp(min=0)

        maskable_positions = attention_mask.bool().clone()
        if special_tokens_mask is not None:
            maskable_positions = maskable_positions & ~special_tokens_mask.bool()

        # Compute weights
        weights = self._masker.compute_weights(cdr_mask, non_templated_mask, maskable_positions)

        # Compute scores based on selection method
        if self.selection_method == "ranked":
            # Deterministic top-K selection (small noise only for tie-breaking)
            noise = torch.rand(weights.shape, device=device, generator=generator) * 1e-6
            scores = weights + noise
        else:
            # Gumbel-top-k: weighted probabilistic sampling without replacement
            eps = 1e-10
            uniform = torch.rand(weights.shape, device=device, generator=generator)
            uniform = uniform.clamp(min=eps, max=1 - eps)
            gumbel_noise = -torch.log(-torch.log(uniform))
            scores = torch.log(weights + eps) + gumbel_noise

        scores = scores.masked_fill(~maskable_positions.bool(), float("-inf"))

        _, indices = scores.sort(dim=-1, descending=True)

        position_ranks = torch.zeros_like(indices)
        position_ranks.scatter_(
            dim=-1,
            index=indices,
            src=torch.arange(seq_len, device=device).expand(batch_size, -1),
        )

        mask_labels = position_ranks < num_to_mask.unsqueeze(-1)
        mask_labels = mask_labels & maskable_positions.bool()

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = tokenizer.mask_token_id

        return masked_ids, mask_labels


def create_eval_masker(cfg: DictConfig) -> EvalMasker:
    """Create an EvalMasker from configuration.

    Parameters
    ----------
    cfg
        Configuration with keys:
        - type: "uniform" or "information_weighted"
        - schedule: "static", "cosine", "linear", "sqrt"
        - mask_rate: float (for static schedule)
        - num_timesteps: int
        - cdr_weight_multiplier: float (for information_weighted)
        - nongermline_weight_multiplier: float (for information_weighted)
        - seed: int
        - selection_method: "ranked" or "sampled" (for information_weighted)

    Returns
    -------
    EvalMasker
        Configured evaluation masker.
    """
    return EvalMasker(
        masker_type=cfg.get("type", "uniform"),
        schedule_type=cfg.get("schedule", "static"),
        mask_rate=cfg.get("mask_rate", 0.15),
        num_timesteps=cfg.get("num_timesteps", 100),
        cdr_weight_multiplier=cfg.get("cdr_weight_multiplier", 1.0),
        nongermline_weight_multiplier=cfg.get("nongermline_weight_multiplier", 1.0),
        seed=cfg.get("seed", 42),
        selection_method=cfg.get("selection_method", "sampled"),
    )
