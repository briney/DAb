"""Optimizer and learning rate scheduler configuration."""

from __future__ import annotations

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Create AdamW optimizer with proper weight decay separation."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bias" in name or "layer_norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_decay: str = "cosine",
    num_training_steps: int = 100000,
    num_warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """Create learning rate scheduler with warmup."""
    if scheduler_decay == "constant":

        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_decay == "linear":
        warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps
        )
        # Ensure total_iters is at least 1 to avoid edge cases
        decay_iters = max(1, num_training_steps - num_warmup_steps)
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=decay_iters,
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[num_warmup_steps]
        )

    elif scheduler_decay == "cosine":
        warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=num_warmup_steps
        )
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr = base_lr * min_lr_ratio
        # Ensure T_max is at least 1 to avoid division by zero
        t_max = max(1, num_training_steps - num_warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[num_warmup_steps]
        )

    else:
        raise ValueError(f"Unknown scheduler decay type: {scheduler_decay}")


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]
