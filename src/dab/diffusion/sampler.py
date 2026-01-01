"""Sampling utilities for generation/denoising."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..vocab import Vocab
from .noise_schedule import NoiseSchedule


class DiffusionSampler:
    """Sampler for generating sequences via iterative denoising."""

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.noise_schedule = noise_schedule
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def _sample_from_logits(self, logits: Tensor) -> Tensor:
        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.top_k is not None:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(
            probs.view(-1, probs.size(-1)), num_samples=1
        ).view(probs.shape[:-1])

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        batch_size: int,
        seq_len: int,
        chain_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        device: torch.device = torch.device("cpu"),
        num_steps: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Generate sequences via iterative denoising.

        Args:
            model: The model to use for predictions. Should accept
                   (token_ids, chain_ids, attention_mask) and return dict with "logits".
            batch_size: Number of sequences to generate.
            seq_len: Length of sequences to generate.
            chain_ids: Chain identity tensor of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            device: Device to use for generation.
            num_steps: Number of denoising steps (default: num_timesteps from schedule).
            show_progress: Whether to show progress bar.

        Returns:
            Generated token IDs of shape (batch_size, seq_len).
        """
        num_steps = num_steps or self.noise_schedule.num_timesteps

        # Start with all masked tokens (except CLS at position 0)
        token_ids = torch.full(
            (batch_size, seq_len), Vocab.MASK_IDX, dtype=torch.long, device=device
        )
        token_ids[:, 0] = Vocab.CLS_IDX

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        timesteps = torch.linspace(num_steps, 1, num_steps, device=device).long()

        iterator = tqdm(timesteps, desc="Sampling", disable=not show_progress)
        for t in iterator:
            current_rate = self.noise_schedule.get_mask_rate(t)
            next_rate = self.noise_schedule.get_mask_rate(t - 1) if t > 1 else 0.0

            is_masked = token_ids == Vocab.MASK_IDX
            outputs = model(token_ids, chain_ids, attention_mask)
            sampled = self._sample_from_logits(outputs["logits"])

            # Get confidence scores for masked positions
            confidence = outputs["logits"].max(dim=-1).values
            confidence = confidence.masked_fill(~is_masked, float("-inf"))

            # Calculate how many tokens to unmask this step
            valid_mask = attention_mask.bool()
            num_valid = valid_mask.sum(dim=-1).float()
            num_to_unmask = ((current_rate - next_rate) * num_valid).round().long()

            # Unmask the most confident predictions
            for i in range(batch_size):
                if num_to_unmask[i] > 0:
                    masked_positions = is_masked[i].nonzero(as_tuple=True)[0]
                    if len(masked_positions) > 0:
                        k = min(num_to_unmask[i].item(), len(masked_positions))
                        top_confident = confidence[i, masked_positions].topk(k).indices
                        unmask_positions = masked_positions[top_confident]
                        token_ids[i, unmask_positions] = sampled[i, unmask_positions]

        return token_ids

    @torch.no_grad()
    def sample_conditional(
        self,
        model: Callable,
        token_ids: Tensor,
        chain_ids: Tensor,
        mask_positions: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Conditionally generate tokens at specified masked positions.

        Args:
            model: The model to use for predictions.
            token_ids: Initial token IDs with some positions to regenerate.
            chain_ids: Chain identity tensor.
            mask_positions: Boolean tensor indicating which positions to regenerate.
            attention_mask: Optional attention mask.
            num_steps: Number of denoising steps.
            show_progress: Whether to show progress bar.

        Returns:
            Token IDs with masked positions filled in.
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        num_steps = num_steps or self.noise_schedule.num_timesteps

        # Apply masks to positions we want to regenerate
        token_ids = token_ids.clone()
        token_ids[mask_positions] = Vocab.MASK_IDX

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        timesteps = torch.linspace(num_steps, 1, num_steps, device=device).long()

        iterator = tqdm(timesteps, desc="Conditional sampling", disable=not show_progress)
        for t in iterator:
            current_rate = self.noise_schedule.get_mask_rate(t)
            next_rate = self.noise_schedule.get_mask_rate(t - 1) if t > 1 else 0.0

            is_masked = token_ids == Vocab.MASK_IDX
            outputs = model(token_ids, chain_ids, attention_mask)
            sampled = self._sample_from_logits(outputs["logits"])

            confidence = outputs["logits"].max(dim=-1).values
            confidence = confidence.masked_fill(~is_masked, float("-inf"))

            num_masked = mask_positions.sum(dim=-1).float()
            num_to_unmask = ((current_rate - next_rate) * num_masked).round().long()

            for i in range(batch_size):
                if num_to_unmask[i] > 0:
                    masked_positions_i = is_masked[i].nonzero(as_tuple=True)[0]
                    if len(masked_positions_i) > 0:
                        k = min(num_to_unmask[i].item(), len(masked_positions_i))
                        top_confident = confidence[i, masked_positions_i].topk(k).indices
                        unmask_pos = masked_positions_i[top_confident]
                        token_ids[i, unmask_pos] = sampled[i, unmask_pos]

        return token_ids
