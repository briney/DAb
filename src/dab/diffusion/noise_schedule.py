"""Noise schedules for discrete diffusion."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor


class ScheduleType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"
    STATIC = "static"
    POWER = "power"


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        pass

    @abstractmethod
    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """Compute NELBO weight for the given timestep.

        The NELBO weight is α'_t / (1 - α_t), where α_t = 1 - mask_rate(t).
        This weights early timesteps (low masking) more heavily, which leads
        to faster convergence per the MD4 paper (arxiv 2406.04329).

        Parameters
        ----------
        timestep
            The diffusion timestep(s), in range [1, num_timesteps].

        Returns
        -------
        float | Tensor
            The NELBO weight(s) for the given timestep(s).
        """
        pass

    def get_normalized_nelbo_weight(
        self,
        timestep: int | Tensor,
        normalize: str | None = None,
        clip_max: float = 10.0,
    ) -> float | Tensor:
        """Compute NELBO weight with optional normalization.

        Parameters
        ----------
        timestep
            The diffusion timestep(s), in range [1, num_timesteps].
        normalize
            Normalization method: None (raw), "clip", or "minmax".
        clip_max
            Maximum weight when using clip normalization.

        Returns
        -------
        float | Tensor
            The (optionally normalized) NELBO weight(s).
        """
        weight = self.get_nelbo_weight(timestep)

        if normalize is None:
            return weight

        if normalize == "clip":
            if isinstance(weight, Tensor):
                return torch.clamp(weight, max=clip_max)
            return min(weight, clip_max)

        if normalize == "minmax":
            # Compute min/max weights across the full timestep range
            all_timesteps = torch.arange(1, self.num_timesteps + 1, dtype=torch.float32)
            all_weights = self.get_nelbo_weight(all_timesteps)
            w_min = all_weights.min().item()
            w_max = all_weights.max().item()

            if w_max - w_min < 1e-8:
                # Avoid division by zero for constant weights
                return weight if isinstance(weight, Tensor) else weight

            if isinstance(weight, Tensor):
                return (weight - w_min) / (w_max - w_min)
            return (weight - w_min) / (w_max - w_min)

        raise ValueError(f"Unknown normalization method: {normalize}")

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        training_progress: float | None = None,
        curriculum_start: float = 0.1,
    ) -> Tensor:
        """Sample timesteps for a batch.

        Parameters
        ----------
        batch_size
            Number of timesteps to sample.
        device
            Device for the output tensor.
        training_progress
            Optional training progress from 0.0 (start) to 1.0 (end).
            When provided, enables curriculum learning that starts with
            a limited timestep range and gradually expands to the full range.
        curriculum_start
            Starting fraction of the timestep range for curriculum learning
            (default: 0.1, meaning start with the first 10% of timesteps).

        Returns
        -------
        Tensor
            Sampled timesteps of shape (batch_size,).
        """
        if training_progress is not None and training_progress < 1.0:
            # Curriculum: expand timestep range as training progresses
            # At progress=0: use curriculum_start fraction of timesteps
            # At progress=1: use full range
            effective_progress = curriculum_start + (1.0 - curriculum_start) * training_progress
            max_timestep = max(1, int(self.num_timesteps * effective_progress))
        else:
            max_timestep = self.num_timesteps
        return torch.randint(1, max_timestep + 1, (batch_size,), device=device)


class LinearSchedule(NoiseSchedule):
    """mask_rate(t) = t / T

    NELBO weight derivation:
    - α_t = 1 - t/T (unmasked rate)
    - α'_t = -1/T
    - weight = |α'_t| / (1 - α_t) = (1/T) / (t/T) = T/t
    """

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        return timestep / self.num_timesteps

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """NELBO weight = T / t (higher weight at early timesteps)."""
        if isinstance(timestep, Tensor):
            return self.num_timesteps / timestep.float()
        return self.num_timesteps / timestep


class CosineSchedule(NoiseSchedule):
    """mask_rate(t) = 1 - cos((t/T) * π/2)

    NELBO weight derivation:
    - α_t = cos(πt/2T) (unmasked rate)
    - α'_t = -(π/2T) * sin(πt/2T)
    - weight = |α'_t| / (1 - α_t) = (π/2T) * sin(πt/2T) / (1 - cos(πt/2T))
    - Using identity: sin(x)/(1-cos(x)) = cot(x/2), so weight = (π/2T) * cot(πt/4T)
    - Or equivalently: weight = (π/2T) * tan(π/4 - πt/4T) for numerical stability
    """

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return 1 - torch.cos(t_normalized * math.pi / 2)
        return 1 - math.cos(t_normalized * math.pi / 2)

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """NELBO weight = (π/2T) * sin(πt/2T) / (1 - cos(πt/2T))."""
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            t_normalized = t_normalized.float()
            sin_term = torch.sin(t_normalized * math.pi / 2)
            cos_term = torch.cos(t_normalized * math.pi / 2)
            # Add small epsilon to avoid division by zero at t=0
            denominator = (1 - cos_term).clamp(min=1e-8)
            return (math.pi / (2 * self.num_timesteps)) * sin_term / denominator
        else:
            sin_term = math.sin(t_normalized * math.pi / 2)
            cos_term = math.cos(t_normalized * math.pi / 2)
            denominator = max(1 - cos_term, 1e-8)
            return (math.pi / (2 * self.num_timesteps)) * sin_term / denominator


class SqrtSchedule(NoiseSchedule):
    """mask_rate(t) = sqrt(t / T)

    NELBO weight derivation:
    - α_t = 1 - sqrt(t/T) (unmasked rate)
    - α'_t = -1 / (2T * sqrt(t/T)) = -1 / (2 * sqrt(t*T))
    - weight = |α'_t| / (1 - α_t) = 1 / (2 * sqrt(t*T) * sqrt(t/T))
             = 1 / (2 * t/T) = T / (2t)
    """

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return torch.sqrt(t_normalized)
        return math.sqrt(t_normalized)

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """NELBO weight = T / (2t) (higher weight at early timesteps)."""
        if isinstance(timestep, Tensor):
            return self.num_timesteps / (2 * timestep.float())
        return self.num_timesteps / (2 * timestep)


class PowerSchedule(NoiseSchedule):
    """mask_rate(t) = (t / T)^power

    A generalized polynomial schedule where higher power values result in
    lower average masking rates. This allows for MLM-like average masking
    while maintaining a valid diffusion schedule.

    Average mask rates by power (with uniform timestep sampling):
    - power=1.0: 0.50 (equivalent to linear)
    - power=2.0: 0.33
    - power=3.0: 0.25
    - power=4.0: 0.20
    - power=5.0: 0.17

    NELBO weight derivation:
    - α_t = 1 - (t/T)^p (unmasked rate)
    - α'_t = -(p/T) * (t/T)^(p-1)
    - weight = |α'_t| / (1 - α_t) = (p/T) * (t/T)^(p-1) / (t/T)^p
             = (p/T) / (t/T) = p * T / t

    Parameters
    ----------
    num_timesteps
        Number of diffusion timesteps.
    power
        Exponent for the schedule (default: 4.0 for ~20% avg masking).
    """

    def __init__(self, num_timesteps: int, power: float = 4.0) -> None:
        super().__init__(num_timesteps)
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
        self.power = power

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return torch.pow(t_normalized, self.power)
        return t_normalized**self.power

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """NELBO weight = p * T / t (higher weight at early timesteps)."""
        if isinstance(timestep, Tensor):
            return self.power * self.num_timesteps / timestep.float()
        return self.power * self.num_timesteps / timestep


class StaticSchedule(NoiseSchedule):
    """Static masking rate for MLM-style training.

    Returns constant mask rate regardless of timestep, enabling
    traditional masked language model training where mask_rate is fixed
    (typically 15% as in BERT).

    For static schedules, NELBO weight is constant (1.0) since there is
    no timestep-dependent masking.

    Parameters
    ----------
    num_timesteps
        Number of timesteps (kept for API compatibility).
    mask_rate
        The constant masking rate to return (default: 0.15 for 15%).
    """

    def __init__(self, num_timesteps: int, mask_rate: float = 0.15) -> None:
        super().__init__(num_timesteps)
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
        self.mask_rate = mask_rate

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        """Return constant mask rate, ignoring timestep."""
        if isinstance(timestep, Tensor):
            return torch.full_like(timestep, self.mask_rate, dtype=torch.float)
        return self.mask_rate

    def get_nelbo_weight(self, timestep: int | Tensor) -> float | Tensor:
        """Return constant weight of 1.0 (no timestep dependence)."""
        if isinstance(timestep, Tensor):
            return torch.ones_like(timestep, dtype=torch.float)
        return 1.0


def create_schedule(
    schedule_type: str | ScheduleType, num_timesteps: int, **kwargs
) -> NoiseSchedule:
    if isinstance(schedule_type, str):
        schedule_type = ScheduleType(schedule_type.lower())

    if schedule_type == ScheduleType.LINEAR:
        return LinearSchedule(num_timesteps)
    elif schedule_type == ScheduleType.COSINE:
        return CosineSchedule(num_timesteps)
    elif schedule_type == ScheduleType.SQRT:
        return SqrtSchedule(num_timesteps)
    elif schedule_type == ScheduleType.POWER:
        power = kwargs.get("power", 4.0)
        return PowerSchedule(num_timesteps, power=power)
    elif schedule_type == ScheduleType.STATIC:
        mask_rate = kwargs.get("mask_rate", 0.15)
        return StaticSchedule(num_timesteps, mask_rate=mask_rate)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
