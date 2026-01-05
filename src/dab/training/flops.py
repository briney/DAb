"""FLOPs tracking and MFU calculation for training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..model import DAbConfig


@dataclass
class FLOPsConfig:
    """Configuration for FLOPs and MFU tracking.

    Parameters
    ----------
    enabled
        Master switch to enable/disable FLOPs tracking.
    """

    enabled: bool = True


# Peak TFLOPS for common GPUs (FP16/BF16 tensor core performance)
GPU_PEAK_TFLOPS: dict[str, float] = {
    # NVIDIA H100
    "H100-SXM5-80GB": 989.0,
    "H100-PCIE-80GB": 756.0,
    "H100-NVL": 835.0,
    # NVIDIA A100
    "A100-SXM4-40GB": 312.0,
    "A100-SXM4-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    # NVIDIA V100
    "V100-SXM2-16GB": 125.0,
    "V100-SXM2-32GB": 125.0,
    "V100-PCIE-16GB": 112.0,
    "V100-PCIE-32GB": 112.0,
    # NVIDIA L40/L40S
    "L40": 181.0,
    "L40S": 362.0,
    # NVIDIA A10/A40
    "A10": 62.5,
    "A40": 149.7,
    # NVIDIA RTX 4090/3090
    "NVIDIA GeForce RTX 4090": 82.6,
    "NVIDIA GeForce RTX 4080": 48.7,
    "NVIDIA GeForce RTX 3090": 35.6,
    "NVIDIA GeForce RTX 3090 Ti": 40.0,
    "NVIDIA GeForce RTX 3080": 29.8,
}


def get_gpu_peak_tflops() -> float | None:
    """Detect GPU type and return theoretical peak TFLOPS.

    Returns
    -------
    float | None
        Peak TFLOPS for detected GPU, or None if unknown/unavailable.
    """
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_name(0)

    # Try exact match first
    if gpu_name in GPU_PEAK_TFLOPS:
        return GPU_PEAK_TFLOPS[gpu_name]

    # Try partial matching for common GPU families
    for known_gpu, tflops in GPU_PEAK_TFLOPS.items():
        if known_gpu in gpu_name or gpu_name in known_gpu:
            return tflops

    # Fallback heuristics based on GPU name patterns
    if "H100" in gpu_name:
        return 989.0
    if "A100" in gpu_name:
        return 312.0
    if "V100" in gpu_name:
        return 125.0
    if "L40S" in gpu_name:
        return 362.0
    if "L40" in gpu_name:
        return 181.0
    if "4090" in gpu_name:
        return 82.6
    if "3090" in gpu_name:
        return 35.6

    return None


def compute_model_flops_per_token(config: DAbConfig) -> int:
    """Compute FLOPs per token for forward pass.

    Formula breakdown per transformer layer:

    Attention:
    - Standard MHA: Q, K, V projections = 3 * 2 * d_model^2 per token
    - Chain-aware attention: doubles QKV = 6 * 2 * d_model^2 per token
    - QK^T matmul = 2 * seq_len * d_model per token
    - Attention @ V = 2 * seq_len * d_model per token
    - Output projection = 2 * d_model^2 per token

    FFN (SwiGLU):
    - Fused gate+up = 2 * d_model * (2 * d_ffn) per token
    - Down projection = 2 * d_ffn * d_model per token
    - Total = 6 * d_model * d_ffn per token

    Parameters
    ----------
    config
        Model configuration with architecture parameters.

    Returns
    -------
    int
        Estimated FLOPs per token for a single forward pass.
    """
    d_model = config.d_model
    n_layers = config.n_layers
    d_ffn = config.d_ffn
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size

    # QKV projection factor: 1x for standard attention, 2x for chain-aware
    qkv_factor = 2 if config.use_chain_aware_attention else 1

    flops_per_layer = 0

    # Attention QKV projections: qkv_factor * 3 * 2 * d_model^2 per token
    flops_per_layer += qkv_factor * 3 * 2 * d_model * d_model

    # Attention QK^T and Attn@V: 2 * 2 * seq_len * d_model per token
    flops_per_layer += 4 * seq_len * d_model

    # Output projection: 2 * d_model^2 per token
    flops_per_layer += 2 * d_model * d_model

    # FFN (SwiGLU): 3 * 2 * d_model * d_ffn per token
    flops_per_layer += 6 * d_model * d_ffn

    # Total for all layers
    total_flops = n_layers * flops_per_layer

    # LM head: 2 * d_model * vocab_size per token
    total_flops += 2 * d_model * vocab_size

    return total_flops


def compute_training_flops_per_token(config: DAbConfig) -> int:
    """Compute FLOPs per token for forward + backward pass.

    Standard approximation: backward pass ~= 2x forward pass FLOPs,
    so total training step = 3x forward pass.

    Parameters
    ----------
    config
        Model configuration.

    Returns
    -------
    int
        FLOPs per token for training (forward + backward).
    """
    return 3 * compute_model_flops_per_token(config)


class FLOPsTracker:
    """Tracks cumulative FLOPs during training.

    Computes MFU (Model FLOPs Utilization) when GPU peak is known.

    Parameters
    ----------
    config
        FLOPs tracking configuration.
    model_config
        Model architecture configuration for FLOPs calculation.
    world_size
        Number of GPUs for distributed training.
    """

    def __init__(
        self,
        config: FLOPsConfig,
        model_config: DAbConfig,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.world_size = world_size

        # Pre-compute FLOPs per token
        self._flops_per_token = compute_training_flops_per_token(model_config)

        # Detect GPU peak for MFU calculation
        self._gpu_peak_tflops = get_gpu_peak_tflops()

        # Cumulative counters
        self._cumulative_flops: int = 0
        self._cumulative_tokens: int = 0

        # For MFU calculation (track wall time)
        self._start_time: float | None = None
        self._total_elapsed_seconds: float = 0.0

    def start_timing(self) -> None:
        """Start wall-clock timing for MFU calculation."""
        self._start_time = time.perf_counter()

    def update(self, batch_size: int, seq_len: int) -> None:
        """Update cumulative FLOPs after a training step.

        Parameters
        ----------
        batch_size
            Number of sequences in the batch.
        seq_len
            Sequence length (tokens per sequence).
        """
        if not self.config.enabled:
            return

        # Tokens processed this step (across all GPUs)
        tokens = batch_size * seq_len * self.world_size

        # FLOPs for this step
        step_flops = tokens * self._flops_per_token

        self._cumulative_flops += step_flops
        self._cumulative_tokens += tokens

    def mark_step_end(self) -> None:
        """Mark end of step for timing accumulation."""
        if self._start_time is not None:
            self._total_elapsed_seconds += time.perf_counter() - self._start_time
            self._start_time = None

    def compute(self) -> dict[str, float]:
        """Compute metrics for logging.

        Returns
        -------
        dict[str, float]
            Dictionary with cumulative_tflops and optionally mfu/achieved_tflops.
        """
        if not self.config.enabled:
            return {}

        results: dict[str, float] = {}

        # Cumulative FLOPs (in TFLOPs for readability)
        results["cumulative_tflops"] = self._cumulative_flops / 1e12

        # MFU calculation
        if self._gpu_peak_tflops is not None and self._total_elapsed_seconds > 0:
            # Achieved TFLOPS
            achieved_tflops = (self._cumulative_flops / 1e12) / self._total_elapsed_seconds

            # Total theoretical peak (all GPUs)
            total_peak_tflops = self._gpu_peak_tflops * self.world_size

            # MFU as percentage
            mfu = (achieved_tflops / total_peak_tflops) * 100
            results["mfu"] = mfu
            results["achieved_tflops"] = achieved_tflops

        return results

    @property
    def flops_per_token(self) -> int:
        """FLOPs per token for training."""
        return self._flops_per_token

    @property
    def gpu_peak_tflops(self) -> float | None:
        """Detected GPU peak TFLOPS, or None if unknown."""
        return self._gpu_peak_tflops
