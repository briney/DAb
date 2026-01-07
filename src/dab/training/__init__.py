"""Training infrastructure for DAb."""

from .checkpoint import CheckpointConfig, CheckpointManager
from .flops import FLOPsConfig, FLOPsTracker
from .masking_frequency import MaskingFrequencyConfig, MaskingFrequencyTracker
from .metrics import (
    DiffusionMetrics,
    MetricAccumulator,
    compute_accuracy,
    compute_diffusion_metrics,
    compute_masked_cross_entropy,
    compute_perplexity,
    compute_weighted_masked_cross_entropy,
)
from .optimizer import create_optimizer, create_scheduler, get_lr
from .trainer import Trainer, TrainingConfig

__all__ = [
    # Checkpoint
    "CheckpointConfig",
    "CheckpointManager",
    # FLOPs tracking
    "FLOPsConfig",
    "FLOPsTracker",
    # Masking frequency tracking
    "MaskingFrequencyConfig",
    "MaskingFrequencyTracker",
    # Metrics
    "MetricAccumulator",
    "DiffusionMetrics",
    "compute_masked_cross_entropy",
    "compute_weighted_masked_cross_entropy",
    "compute_accuracy",
    "compute_perplexity",
    "compute_diffusion_metrics",
    # Optimizer
    "create_optimizer",
    "create_scheduler",
    "get_lr",
    # Trainer
    "TrainingConfig",
    "Trainer",
]
