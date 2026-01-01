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
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
