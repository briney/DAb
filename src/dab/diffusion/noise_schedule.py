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


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    def __init__(self, num_timesteps: int) -> None:
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_mask_rate(self, timestep: int | Tensor) -> float | Tensor:
        pass

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)


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
    elif schedule_type == ScheduleType.STATIC:
        mask_rate = kwargs.get("mask_rate", 0.15)
        return StaticSchedule(num_timesteps, mask_rate=mask_rate)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
