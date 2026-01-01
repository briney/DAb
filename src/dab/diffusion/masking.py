"""Information-weighted masking for discrete diffusion."""

from __future__ import annotations

import torch
from torch import Tensor

from ..vocab import Vocab
from .noise_schedule import NoiseSchedule


class InformationWeightedMasker:
    """
    Applies masking with preference for high-information positions.

    Weights: Non-templated CDR = 2, Templated CDR or Non-templated non-CDR = 1, Templated non-CDR = 0
    With multiplier=1.0: Non-templated CDR ~3x more likely than templated non-CDR.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        weight_multiplier: float = 1.0,
        mask_token_id: int = Vocab.MASK_IDX,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.weight_multiplier = weight_multiplier
        self.mask_token_id = mask_token_id

    def compute_weights(
        self,
        cdr_mask: Tensor | None,
        non_templated_mask: Tensor | None,
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        weights = torch.ones(batch_size, seq_len, device=device)

        if cdr_mask is not None:
            weights = weights + cdr_mask.float() * self.weight_multiplier

        if non_templated_mask is not None:
            weights = weights + non_templated_mask.float() * self.weight_multiplier

        weights = weights * attention_mask.float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return weights / weights_sum

    def apply_mask(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        cdr_mask: Tensor | None = None,
        non_templated_mask: Tensor | None = None,
        special_tokens_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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

        weights = self.compute_weights(cdr_mask, non_templated_mask, maskable_positions)

        noise = torch.rand_like(weights) * 1e-6
        scores = weights + noise
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
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels


class UniformMasker:
    """Simple uniform random masking without information weighting."""

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        mask_token_id: int = Vocab.MASK_IDX,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id

    def apply_mask(
        self,
        token_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Tensor,
        special_tokens_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        mask_rates = self.noise_schedule.get_mask_rate(timesteps)
        rand = torch.rand(batch_size, seq_len, device=device)

        maskable = attention_mask.bool()
        if special_tokens_mask is not None:
            maskable = maskable & ~special_tokens_mask.bool()

        mask_labels = (rand < mask_rates.unsqueeze(-1)) & maskable

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels
