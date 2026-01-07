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
    """mask_rate(t) = t / T"""

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        return timestep / self.num_timesteps


class CosineSchedule(NoiseSchedule):
    """mask_rate(t) = 1 - cos((t/T) * π/2)"""

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return 1 - torch.cos(t_normalized * math.pi / 2)
        return 1 - math.cos(t_normalized * math.pi / 2)


class SqrtSchedule(NoiseSchedule):
    """mask_rate(t) = sqrt(t / T)"""

    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        t_normalized = timestep / self.num_timesteps
        if isinstance(t_normalized, Tensor):
            return torch.sqrt(t_normalized)
        return math.sqrt(t_normalized)


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


class StaticSchedule(NoiseSchedule):
    """Static masking rate for MLM-style training.

    Returns constant mask rate regardless of timestep, enabling
    traditional masked language model training where mask_rate is fixed
    (typically 15% as in BERT).

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
