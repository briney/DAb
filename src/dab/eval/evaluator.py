"""Main evaluator class for orchestrating metric computation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Metric
from .masking import EvalMasker, create_eval_masker
from .per_position import PerPositionEvaluator, RegionMaskingEvaluator
from .regions import AntibodyRegion
from .registry import build_metrics

if TYPE_CHECKING:
    from accelerate import Accelerator
    from omegaconf import DictConfig

    from ..model import DAbModel


def _get_model_device(model: "DAbModel", accelerator: "Accelerator | None") -> torch.device:
    """Get the device the model is on.

    Args:
        model: The model to check.
        accelerator: Optional Accelerator instance.

    Returns:
        The device the model parameters are on.
    """
    if accelerator is not None:
        return accelerator.device
    return next(model.parameters()).device


class Evaluator:
    """Orchestrates evaluation metric computation.

    The Evaluator manages metric instantiation, caching, and computation
    for multiple evaluation datasets with potentially different metrics.

    Attributes:
        cfg: Full configuration object.
        model: The model to evaluate.
        accelerator: Optional Accelerator for distributed training.
    """

    def __init__(
        self,
        cfg: "DictConfig",
        model: "DAbModel",
        accelerator: "Accelerator | None" = None,
        objective: str = "diffusion",
    ) -> None:
        """Initialize the evaluator.

        Args:
            cfg: Full configuration object.
            model: The model to evaluate.
            accelerator: Optional Accelerate accelerator instance.
            objective: Training objective (e.g., "diffusion", "mlm").
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator
        self.objective = objective

        # Determine global coordinate availability
        data_cfg = cfg.get("data", {})
        self.has_coords = bool(data_cfg.get("load_coords", False))

        # Cache for metrics per eval dataset
        self._metrics_cache: dict[str, list[Metric]] = {}

        # Cache for whether attention weights are needed per eval dataset
        self._needs_attentions_cache: dict[str, bool] = {}

        # Initialize evaluation masker from config (if configured)
        self.eval_masker = self._build_eval_masker()

    def _build_eval_masker(self) -> EvalMasker | None:
        """Build evaluation masker from config if configured.

        Returns
        -------
        EvalMasker or None
            Configured EvalMasker if eval.masking is present in config,
            None otherwise.
        """
        eval_cfg = self.cfg.get("eval", {})
        masking_cfg = eval_cfg.get("masking", {})

        if not masking_cfg:
            return None

        return create_eval_masker(masking_cfg)

    def _get_metrics(self, eval_name: str) -> list[Metric]:
        """Get or build metrics for an evaluation dataset.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            List of Metric instances for this dataset.
        """
        if eval_name not in self._metrics_cache:
            metrics = build_metrics(
                cfg=self.cfg,
                objective=self.objective,
                has_coords=self.has_coords,
                eval_name=eval_name,
            )
            # If per-position region eval is enabled, skip redundant region metrics
            region_cfg = self._get_region_config(eval_name)
            if region_cfg.get("enabled", False):
                region_metric_names = {"region_acc", "region_ppl", "region_loss"}
                metrics = [m for m in metrics if m.name not in region_metric_names]
            self._metrics_cache[eval_name] = metrics
        return self._metrics_cache[eval_name]

    def _needs_attentions(self, eval_name: str) -> bool:
        """Check if any metrics need attention weights.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            True if any metric requires attention weights.
        """
        if eval_name not in self._needs_attentions_cache:
            metrics = self._get_metrics(eval_name)
            self._needs_attentions_cache[eval_name] = any(
                getattr(m, "needs_attentions", False) for m in metrics
            )
        return self._needs_attentions_cache[eval_name]

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        """Aggregate metric states across distributed processes.

        Args:
            metrics: List of metrics to aggregate.
        """
        if self.accelerator is None or self.accelerator.num_processes <= 1:
            return

        for metric in metrics:
            # Check if this metric uses object-based gathering
            state_objects = metric.state_objects()

            if state_objects is not None:
                # Use gather_for_metrics with use_gather_object for variable-length state
                gathered = self.accelerator.gather_for_metrics(
                    state_objects, use_gather_object=True
                )
                metric.load_state_objects(gathered)
            else:
                # Use tensor-based gathering for fixed-size state
                state_tensors = metric.state_tensors()
                if state_tensors:
                    gathered_tensors = []
                    for tensor in state_tensors:
                        # Move to accelerator device and gather
                        tensor = tensor.to(self.accelerator.device)
                        gathered = self.accelerator.gather(tensor)
                        # Sum across processes
                        if gathered.dim() > tensor.dim():
                            gathered = gathered.sum(dim=0)
                        gathered_tensors.append(gathered)
                    metric.load_state_tensors(gathered_tensors)

    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_name: str,
        masker: Any = None,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Masking priority:
        1. Use self.eval_masker if configured (controlled, reproducible eval)
        2. Use passed masker parameter (legacy behavior)
        3. Fall back to _create_eval_mask (15% random masking)

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            masker: Optional masker for creating mask labels.
                Ignored if self.eval_masker is configured.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = self._get_metrics(eval_name)

        if not metrics:
            return {}

        # Reset all metrics
        for metric in metrics:
            metric.reset()

        # Check if any metrics need attention weights
        needs_attentions = self._needs_attentions(eval_name)

        self.model.eval()
        device = _get_model_device(self.model, self.accelerator)

        # Determine if we should show progress bar
        show_progress = self.accelerator is None or self.accelerator.is_local_main_process

        # Get seeded generator for reproducible masking (if using eval_masker)
        generator = None
        if self.eval_masker is not None:
            generator = self.eval_masker.get_generator(device)

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Eval ({eval_name})", disable=not show_progress):
                # Move batch to device if not using accelerator
                if self.accelerator is None:
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Create mask labels for evaluation
                # Priority: 1) eval_masker (controlled), 2) passed masker, 3) fallback
                if self.eval_masker is not None:
                    # Use configured eval masker with seeded generator
                    masked_ids, mask_labels = self.eval_masker.apply_mask(
                        batch=batch,
                        generator=generator,
                    )
                elif masker is not None:
                    # Legacy: use passed masker
                    batch_size = batch["token_ids"].shape[0]
                    timesteps = masker.noise_schedule.sample_timesteps(batch_size, device)
                    masked_ids, mask_labels = masker.apply_mask(
                        token_ids=batch["token_ids"],
                        timesteps=timesteps,
                        attention_mask=batch["attention_mask"],
                        special_tokens_mask=batch.get("special_tokens_mask"),
                    )
                else:
                    # Default: random 15% masking for eval
                    mask_labels = self._create_eval_mask(batch, device)
                    masked_ids = batch["token_ids"].clone()
                    from ..tokenizer import tokenizer
                    masked_ids[mask_labels.bool()] = tokenizer.mask_token_id

                # Forward pass
                outputs = self.model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                    output_attentions=needs_attentions,
                )

                # Update all metrics
                for metric in metrics:
                    try:
                        metric.update(outputs, batch, mask_labels)
                    except Exception as e:
                        warnings.warn(f"Metric '{metric.name}' update failed: {e}")

        # Aggregate across distributed processes
        self._gather_metric_states(metrics)

        # Compute final metric values
        results: dict[str, float] = {}
        for metric in metrics:
            try:
                computed = metric.compute()
                results.update(computed)
            except Exception as e:
                warnings.warn(f"Metric '{metric.name}' compute failed: {e}")

        # Region-based evaluation (if enabled for this dataset)
        region_cfg = self._get_region_config(eval_name)
        if region_cfg.get("enabled", False):
            try:
                region_results = self._evaluate_regions(eval_loader, eval_name, region_cfg)
                # Prefix with "region/" and merge into results
                for key, value in region_results.items():
                    results[f"region/{key}"] = value
            except Exception as e:
                warnings.warn(f"Region evaluation failed: {e}")

        self.model.train()

        # Clear CUDA cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _create_eval_mask(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        mask_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Create a random mask for evaluation.

        Args:
            batch: Input batch dictionary.
            device: Device to create the mask on.
            mask_ratio: Fraction of tokens to mask.

        Returns:
            Binary mask tensor (batch, seq_len).
        """
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        special_tokens_mask = batch.get("special_tokens_mask")

        # Random mask
        rand = torch.rand_like(token_ids.float())
        mask_labels = (rand < mask_ratio).long()

        # Don't mask padding
        mask_labels = mask_labels * attention_mask

        # Don't mask special tokens if mask is provided
        if special_tokens_mask is not None:
            mask_labels = mask_labels * (~special_tokens_mask).long()

        return mask_labels.to(device)

    def _get_region_config(self, eval_name: str) -> dict[str, Any]:
        """Get merged region evaluation config for a dataset.

        Merges global eval.regions config with per-dataset overrides.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            Merged region configuration dictionary.
        """
        # Start with global config
        eval_cfg = self.cfg.get("eval", {})
        global_regions = dict(eval_cfg.get("regions", {}))

        # Get per-dataset override if present
        data_cfg = self.cfg.get("data", {})
        eval_datasets = data_cfg.get("eval", {})

        if isinstance(eval_datasets, str):
            # Single eval dataset, no per-dataset config
            return global_regions

        dataset_cfg = eval_datasets.get(eval_name, {})
        if isinstance(dataset_cfg, str):
            # Shorthand path, no per-dataset config
            return global_regions

        dataset_regions = dict(dataset_cfg.get("regions", {}))

        # Merge: dataset overrides global
        result = global_regions.copy()
        result.update(dataset_regions)
        return result

    def _show_progress(self) -> bool:
        """Check if progress bars should be shown."""
        return self.accelerator is None or self.accelerator.is_local_main_process

    def _evaluate_regions(
        self,
        eval_loader: DataLoader,
        eval_name: str,
        region_cfg: dict[str, Any],
    ) -> dict[str, float]:
        """Run region-based evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            region_cfg: Region evaluation configuration.

        Returns:
            Dictionary mapping region metric names to values.
        """
        mode = region_cfg.get("mode", "per_position")
        include = region_cfg.get("include")  # None = all regions
        position_batch_size = region_cfg.get("position_batch_size", 32)
        aggregate_strategies = region_cfg.get("aggregate", ["all"])

        # Parse include list to AntibodyRegion set
        if include is not None:
            regions = {AntibodyRegion(r) for r in include}
        else:
            regions = None  # All regions

        device = _get_model_device(self.model, self.accelerator)

        if mode == "per_position":
            return self._run_per_position_eval(
                eval_loader, regions, position_batch_size, aggregate_strategies
            )
        else:  # region_level
            return self._run_region_level_eval(
                eval_loader, regions, aggregate_strategies
            )

    def _run_per_position_eval(
        self,
        eval_loader: DataLoader,
        regions: set[AntibodyRegion] | None,
        position_batch_size: int,
        aggregate_strategies: list[str],
    ) -> dict[str, float]:
        """Run per-position region evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            regions: Set of regions to evaluate, or None for all.
            position_batch_size: Batch size for per-position evaluation.
            aggregate_strategies: List of aggregation strategies.

        Returns:
            Dictionary mapping region metric names to values.
        """
        device = _get_model_device(self.model, self.accelerator)
        evaluator = PerPositionEvaluator(
            model=self.model,
            position_batch_size=position_batch_size,
            device=device,
            show_progress=False,  # Outer tqdm handles progress
        )

        # Accumulate results across all samples
        region_accumulators: dict[str, dict[str, float]] = {}

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc="Region eval",
                disable=not self._show_progress()
            ):
                # Process each sample in the batch individually
                batch_size = batch["token_ids"].shape[0]
                for i in range(batch_size):
                    # Extract single sample
                    sample = {
                        k: v[i] if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Evaluate by region
                    try:
                        sample_results = evaluator.evaluate_by_region(sample, regions)
                    except Exception as e:
                        warnings.warn(f"Region evaluation failed for sample: {e}")
                        continue

                    # Accumulate results
                    for region_name, metrics in sample_results.items():
                        if region_name not in region_accumulators:
                            region_accumulators[region_name] = {
                                "correct": 0,
                                "total_loss": 0.0,
                                "total_prob": 0.0,
                                "count": 0,
                            }
                        acc = region_accumulators[region_name]
                        count = metrics.get("count", 0)
                        if count > 0:
                            acc["correct"] += metrics["accuracy"] * count
                            acc["total_loss"] += metrics["avg_loss"] * count
                            acc["total_prob"] += metrics["avg_prob"] * count
                            acc["count"] += count

        # Compute final metrics
        results: dict[str, float] = {}
        for region_name, acc in region_accumulators.items():
            if acc["count"] > 0:
                results[f"{region_name}_accuracy"] = acc["correct"] / acc["count"]
                results[f"{region_name}_loss"] = acc["total_loss"] / acc["count"]
                results[f"{region_name}_prob"] = acc["total_prob"] / acc["count"]

        # Apply aggregation strategies
        results.update(self._aggregate_region_results(region_accumulators, aggregate_strategies))

        return results

    def _run_region_level_eval(
        self,
        eval_loader: DataLoader,
        regions: set[AntibodyRegion] | None,
        aggregate_strategies: list[str],
    ) -> dict[str, float]:
        """Run region-level (full region masking) evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            regions: Set of regions to evaluate, or None for all.
            aggregate_strategies: List of aggregation strategies.

        Returns:
            Dictionary mapping region metric names to values.
        """
        device = _get_model_device(self.model, self.accelerator)
        evaluator = RegionMaskingEvaluator(model=self.model, device=device)

        # Accumulate results across all samples
        region_accumulators: dict[str, dict[str, float]] = {}

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc="Region eval",
                disable=not self._show_progress()
            ):
                # Process each sample in the batch individually
                batch_size = batch["token_ids"].shape[0]
                for i in range(batch_size):
                    # Extract single sample
                    sample = {
                        k: v[i] if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    # Evaluate all regions
                    try:
                        sample_results = evaluator.evaluate_all_regions(sample, regions)
                    except Exception as e:
                        warnings.warn(f"Region evaluation failed for sample: {e}")
                        continue

                    # Accumulate results
                    for region_name, metrics in sample_results.items():
                        if region_name not in region_accumulators:
                            region_accumulators[region_name] = {
                                "correct": 0,
                                "total_loss": 0.0,
                                "total_prob": 0.0,
                                "count": 0,
                            }
                        acc = region_accumulators[region_name]
                        count = metrics.get("count", 0)
                        if count > 0:
                            acc["correct"] += metrics["accuracy"] * count
                            acc["total_loss"] += metrics["avg_loss"] * count
                            acc["total_prob"] += metrics["avg_prob"] * count
                            acc["count"] += count

        # Compute final metrics
        results: dict[str, float] = {}
        for region_name, acc in region_accumulators.items():
            if acc["count"] > 0:
                results[f"{region_name}_accuracy"] = acc["correct"] / acc["count"]
                results[f"{region_name}_loss"] = acc["total_loss"] / acc["count"]
                results[f"{region_name}_prob"] = acc["total_prob"] / acc["count"]

        # Apply aggregation strategies
        results.update(self._aggregate_region_results(region_accumulators, aggregate_strategies))

        return results

    def _aggregate_region_results(
        self,
        region_accumulators: dict[str, dict[str, float]],
        aggregate_strategies: list[str],
    ) -> dict[str, float]:
        """Aggregate region results by various strategies.

        Args:
            region_accumulators: Per-region accumulated metrics.
            aggregate_strategies: List of aggregation strategies to apply.

        Returns:
            Aggregated metrics.
        """
        results: dict[str, float] = {}

        for strategy in aggregate_strategies:
            if strategy == "all":
                # Individual regions already in results, skip
                continue

            elif strategy in ("cdr", "fwr"):
                # Aggregate CDRs vs frameworks
                cdr_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
                fwr_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}

                for region_name, acc in region_accumulators.items():
                    if "cdr" in region_name:
                        for k in cdr_acc:
                            cdr_acc[k] += acc[k]
                    elif "fwr" in region_name:
                        for k in fwr_acc:
                            fwr_acc[k] += acc[k]

                if cdr_acc["count"] > 0:
                    results["cdr_accuracy"] = cdr_acc["correct"] / cdr_acc["count"]
                    results["cdr_loss"] = cdr_acc["total_loss"] / cdr_acc["count"]
                if fwr_acc["count"] > 0:
                    results["fwr_accuracy"] = fwr_acc["correct"] / fwr_acc["count"]
                    results["fwr_loss"] = fwr_acc["total_loss"] / fwr_acc["count"]

            elif strategy == "chain":
                # Aggregate by heavy vs light chain
                heavy_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
                light_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}

                for region_name, acc in region_accumulators.items():
                    if region_name.startswith("h"):
                        for k in heavy_acc:
                            heavy_acc[k] += acc[k]
                    elif region_name.startswith("l"):
                        for k in light_acc:
                            light_acc[k] += acc[k]

                if heavy_acc["count"] > 0:
                    results["heavy_accuracy"] = heavy_acc["correct"] / heavy_acc["count"]
                    results["heavy_loss"] = heavy_acc["total_loss"] / heavy_acc["count"]
                if light_acc["count"] > 0:
                    results["light_accuracy"] = light_acc["correct"] / light_acc["count"]
                    results["light_loss"] = light_acc["total_loss"] / light_acc["count"]

            elif strategy == "region_type":
                # Aggregate by CDR/FWR number across chains
                type_accs: dict[str, dict[str, float]] = {}

                for region_name, acc in region_accumulators.items():
                    # Extract type (e.g., "cdr1" from "hcdr1" or "lcdr1")
                    if region_name.startswith("h") or region_name.startswith("l"):
                        region_type = region_name[1:]  # Remove chain prefix
                    else:
                        region_type = region_name

                    if region_type not in type_accs:
                        type_accs[region_type] = {
                            "correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0
                        }
                    for k in type_accs[region_type]:
                        type_accs[region_type][k] += acc[k]

                for region_type, acc in type_accs.items():
                    if acc["count"] > 0:
                        results[f"{region_type}_accuracy"] = acc["correct"] / acc["count"]
                        results[f"{region_type}_loss"] = acc["total_loss"] / acc["count"]

        return results

    def evaluate_all(
        self,
        eval_loaders: dict[str, DataLoader],
        masker: Any = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on all configured evaluation datasets.

        Args:
            eval_loaders: Dictionary mapping eval dataset names to DataLoaders.
            masker: Optional masker for creating mask labels.

        Returns:
            Dictionary mapping eval dataset names to their metric results.
        """
        all_results: dict[str, dict[str, float]] = {}

        for eval_name, eval_loader in eval_loaders.items():
            results = self.evaluate(eval_loader, eval_name, masker)
            all_results[eval_name] = results

        return all_results

    def clear_cache(self) -> None:
        """Clear cached metrics and attention flags.

        Call this if the configuration changes during training.
        """
        self._metrics_cache.clear()
        self._needs_attentions_cache.clear()
