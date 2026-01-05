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
from .region_config import RegionEvalConfig, build_region_eval_config
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
        mask_labels_cache: list[torch.Tensor] | None = None,
        batch_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        """Run region-based evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            region_cfg: Region evaluation configuration dictionary.
            mask_labels_cache: Cached mask labels from main evaluation (for standard mode).
            batch_cache: Cached batches from main evaluation (for standard mode).

        Returns:
            Dictionary mapping region metric names to values.
        """
        # Build config object from dictionary
        config = build_region_eval_config(region_cfg)

        mode = config.mode
        position_batch_size = config.position_batch_size

        # Get enabled regions as AntibodyRegion set
        enabled_region_names = config.get_enabled_regions()
        if enabled_region_names:
            regions = {AntibodyRegion(r) for r in enabled_region_names}
        else:
            regions = None  # No individual regions enabled

        if mode == "standard":
            return self._run_standard_eval(
                eval_loader, eval_name, regions, config,
                mask_labels_cache, batch_cache
            )
        elif mode in ("per_position", "per-position"):
            return self._run_per_position_eval(
                eval_loader, regions, position_batch_size, config
            )
        else:  # region_level or region-level
            return self._run_region_level_eval(
                eval_loader, regions, config
            )

    def _get_regions_for_aggregates(
        self,
        config: RegionEvalConfig,
    ) -> set[AntibodyRegion]:
        """Get all regions needed to compute enabled aggregates.

        Args:
            config: Region evaluation configuration.

        Returns:
            Set of AntibodyRegion values needed for aggregate computation.
        """
        regions_needed: set[AntibodyRegion] = set()
        enabled_aggs = config.get_enabled_aggregates()

        # CDR regions
        cdr_regions = {
            AntibodyRegion.HCDR1, AntibodyRegion.HCDR2, AntibodyRegion.HCDR3,
            AntibodyRegion.LCDR1, AntibodyRegion.LCDR2, AntibodyRegion.LCDR3,
        }
        # FWR regions
        fwr_regions = {
            AntibodyRegion.HFWR1, AntibodyRegion.HFWR2, AntibodyRegion.HFWR3, AntibodyRegion.HFWR4,
            AntibodyRegion.LFWR1, AntibodyRegion.LFWR2, AntibodyRegion.LFWR3, AntibodyRegion.LFWR4,
        }
        # Heavy chain regions
        heavy_regions = {
            AntibodyRegion.HCDR1, AntibodyRegion.HCDR2, AntibodyRegion.HCDR3,
            AntibodyRegion.HFWR1, AntibodyRegion.HFWR2, AntibodyRegion.HFWR3, AntibodyRegion.HFWR4,
        }
        # Light chain regions
        light_regions = {
            AntibodyRegion.LCDR1, AntibodyRegion.LCDR2, AntibodyRegion.LCDR3,
            AntibodyRegion.LFWR1, AntibodyRegion.LFWR2, AntibodyRegion.LFWR3, AntibodyRegion.LFWR4,
        }

        if "all_cdr" in enabled_aggs:
            regions_needed |= cdr_regions
        if "all_fwr" in enabled_aggs:
            regions_needed |= fwr_regions
        if "heavy" in enabled_aggs:
            regions_needed |= heavy_regions
        if "light" in enabled_aggs:
            regions_needed |= light_regions
        if "overall" in enabled_aggs:
            regions_needed |= cdr_regions | fwr_regions

        return regions_needed

    def _compute_aggregate_metrics(
        self,
        region_accumulators: dict[str, dict[str, float]],
        config: RegionEvalConfig,
    ) -> dict[str, float]:
        """Compute aggregate metrics based on enabled aggregates.

        Args:
            region_accumulators: Per-region accumulated metrics.
            config: Region evaluation configuration.

        Returns:
            Aggregated metrics with / separator.
        """
        results: dict[str, float] = {}
        enabled_aggs = config.get_enabled_aggregates()

        # all_cdr: aggregate all CDR regions
        if "all_cdr" in enabled_aggs:
            cdr_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
            for region_name in ["hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3"]:
                if region_name in region_accumulators:
                    for k in cdr_acc:
                        cdr_acc[k] += region_accumulators[region_name][k]
            if cdr_acc["count"] > 0:
                results["all_cdr/accuracy"] = cdr_acc["correct"] / cdr_acc["count"]
                results["all_cdr/loss"] = cdr_acc["total_loss"] / cdr_acc["count"]
                results["all_cdr/prob"] = cdr_acc["total_prob"] / cdr_acc["count"]
                avg_loss = cdr_acc["total_loss"] / cdr_acc["count"]
                results["all_cdr/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # all_fwr: aggregate all FWR regions
        if "all_fwr" in enabled_aggs:
            fwr_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
            for region_name in [
                "hfwr1", "hfwr2", "hfwr3", "hfwr4",
                "lfwr1", "lfwr2", "lfwr3", "lfwr4"
            ]:
                if region_name in region_accumulators:
                    for k in fwr_acc:
                        fwr_acc[k] += region_accumulators[region_name][k]
            if fwr_acc["count"] > 0:
                results["all_fwr/accuracy"] = fwr_acc["correct"] / fwr_acc["count"]
                results["all_fwr/loss"] = fwr_acc["total_loss"] / fwr_acc["count"]
                results["all_fwr/prob"] = fwr_acc["total_prob"] / fwr_acc["count"]
                avg_loss = fwr_acc["total_loss"] / fwr_acc["count"]
                results["all_fwr/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # heavy: aggregate all heavy chain regions
        if "heavy" in enabled_aggs:
            heavy_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
            for region_name in [
                "hcdr1", "hcdr2", "hcdr3",
                "hfwr1", "hfwr2", "hfwr3", "hfwr4"
            ]:
                if region_name in region_accumulators:
                    for k in heavy_acc:
                        heavy_acc[k] += region_accumulators[region_name][k]
            if heavy_acc["count"] > 0:
                results["heavy/accuracy"] = heavy_acc["correct"] / heavy_acc["count"]
                results["heavy/loss"] = heavy_acc["total_loss"] / heavy_acc["count"]
                results["heavy/prob"] = heavy_acc["total_prob"] / heavy_acc["count"]
                avg_loss = heavy_acc["total_loss"] / heavy_acc["count"]
                results["heavy/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # light: aggregate all light chain regions
        if "light" in enabled_aggs:
            light_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
            for region_name in [
                "lcdr1", "lcdr2", "lcdr3",
                "lfwr1", "lfwr2", "lfwr3", "lfwr4"
            ]:
                if region_name in region_accumulators:
                    for k in light_acc:
                        light_acc[k] += region_accumulators[region_name][k]
            if light_acc["count"] > 0:
                results["light/accuracy"] = light_acc["correct"] / light_acc["count"]
                results["light/loss"] = light_acc["total_loss"] / light_acc["count"]
                results["light/prob"] = light_acc["total_prob"] / light_acc["count"]
                avg_loss = light_acc["total_loss"] / light_acc["count"]
                results["light/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # overall: aggregate all regions (excluding germline/nongermline which are position-based)
        if "overall" in enabled_aggs:
            overall_acc = {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}
            for name, acc in region_accumulators.items():
                # Skip germline/nongermline - they're position-based, not region-based
                if name in ("germline", "nongermline"):
                    continue
                for k in overall_acc:
                    overall_acc[k] += acc[k]
            if overall_acc["count"] > 0:
                results["overall/accuracy"] = overall_acc["correct"] / overall_acc["count"]
                results["overall/loss"] = overall_acc["total_loss"] / overall_acc["count"]
                results["overall/prob"] = overall_acc["total_prob"] / overall_acc["count"]
                avg_loss = overall_acc["total_loss"] / overall_acc["count"]
                results["overall/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # germline: already accumulated directly in region_accumulators
        if "germline" in enabled_aggs and "germline" in region_accumulators:
            acc = region_accumulators["germline"]
            if acc["count"] > 0:
                results["germline/accuracy"] = acc["correct"] / acc["count"]
                results["germline/loss"] = acc["total_loss"] / acc["count"]
                results["germline/prob"] = acc["total_prob"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results["germline/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # nongermline: already accumulated directly in region_accumulators
        if "nongermline" in enabled_aggs and "nongermline" in region_accumulators:
            acc = region_accumulators["nongermline"]
            if acc["count"] > 0:
                results["nongermline/accuracy"] = acc["correct"] / acc["count"]
                results["nongermline/loss"] = acc["total_loss"] / acc["count"]
                results["nongermline/prob"] = acc["total_prob"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results["nongermline/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        return results

    def _run_standard_eval(
        self,
        eval_loader: DataLoader,
        eval_name: str,
        regions: set[AntibodyRegion] | None,
        config: RegionEvalConfig,
        mask_labels_cache: list[torch.Tensor] | None = None,
        batch_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        """Run standard region evaluation using existing masking.

        This mode computes region metrics on positions that are both:
        - Masked during evaluation
        - Within the specified regions

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            regions: Set of regions to evaluate, or None for all.
            config: Region evaluation configuration.
            mask_labels_cache: Optional cached mask labels from main evaluation.
            batch_cache: Optional cached batches from main evaluation.

        Returns:
            Dictionary mapping region metric names to values.
        """
        from .regions import extract_region_masks

        device = _get_model_device(self.model, self.accelerator)

        # Accumulators per region: {region_name: {correct, total_loss, total_tokens}}
        region_accumulators: dict[str, dict[str, float]] = {}

        # Determine which regions to extract for aggregates
        regions_for_aggregates = self._get_regions_for_aggregates(config)
        all_regions_needed = (regions or set()) | regions_for_aggregates

        # Get seeded generator for reproducible masking
        generator = None
        if self.eval_masker is not None:
            generator = self.eval_masker.get_generator(device)

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc="Region eval (standard)",
                disable=not self._show_progress()
            ):
                # Move batch to device if not using accelerator
                if self.accelerator is None:
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Skip if no CDR mask available
                if batch.get("cdr_mask") is None:
                    continue

                # Create mask labels (same logic as main evaluate())
                if self.eval_masker is not None:
                    masked_ids, mask_labels = self.eval_masker.apply_mask(
                        batch=batch,
                        generator=generator,
                    )
                else:
                    mask_labels = self._create_eval_mask(batch, device)
                    masked_ids = batch["token_ids"].clone()
                    from ..tokenizer import tokenizer
                    masked_ids[mask_labels.bool()] = tokenizer.mask_token_id

                # Forward pass
                outputs = self.model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                logits = outputs["logits"]
                targets = batch["token_ids"]
                predictions = logits.argmax(dim=-1)

                # Compute per-token loss
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits_flat, targets_flat, reduction="none"
                ).view(batch_size, seq_len)

                # Compute per-token probabilities for correct tokens
                probs = torch.softmax(logits, dim=-1)
                target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

                # Extract region masks
                try:
                    target_regions = all_regions_needed if all_regions_needed else set(AntibodyRegion)
                    region_masks = extract_region_masks(batch, target_regions)
                except ValueError:
                    continue

                mask = mask_labels.bool()
                correct_mask = (predictions == targets) & mask

                # Process individual regions
                for region_name, region_mask in region_masks.items():
                    combined_mask = mask & region_mask
                    region_correct = (correct_mask & region_mask).sum().item()
                    region_loss = (loss_per_token * combined_mask.float()).sum().item()
                    region_total = combined_mask.sum().item()

                    if region_name.value not in region_accumulators:
                        region_accumulators[region_name.value] = {
                            "correct": 0,
                            "total_loss": 0.0,
                            "total_prob": 0.0,
                            "count": 0,
                        }
                    acc = region_accumulators[region_name.value]
                    region_prob = (target_probs * combined_mask.float()).sum().item()
                    acc["correct"] += region_correct
                    acc["total_loss"] += region_loss
                    acc["total_prob"] += region_prob
                    acc["count"] += region_total

                # Handle germline/nongermline aggregates (position-based, not region-based)
                enabled_aggs = config.get_enabled_aggregates()
                non_templated_mask = batch.get("non_templated_mask")
                if non_templated_mask is None:
                    # Warn once if germline/nongermline tracking enabled but mask missing
                    if "germline" in enabled_aggs or "nongermline" in enabled_aggs:
                        warnings.warn(
                            "germline/nongermline region tracking enabled but non_templated_mask "
                            "not found in batch. Ensure data has columns matching "
                            "heavy_nongermline_col and light_nongermline_col config settings."
                        )
                if non_templated_mask is not None:
                    if "germline" in enabled_aggs:
                        if "germline" not in region_accumulators:
                            region_accumulators["germline"] = {
                                "correct": 0,
                                "total_loss": 0.0,
                                "total_prob": 0.0,
                                "count": 0,
                            }
                        germline_mask = (non_templated_mask == 0)
                        combined = mask & germline_mask
                        acc = region_accumulators["germline"]
                        acc["correct"] += ((predictions == targets) & combined).sum().item()
                        acc["total_loss"] += (loss_per_token * combined.float()).sum().item()
                        acc["total_prob"] += (target_probs * combined.float()).sum().item()
                        acc["count"] += combined.sum().item()

                    if "nongermline" in enabled_aggs:
                        if "nongermline" not in region_accumulators:
                            region_accumulators["nongermline"] = {
                                "correct": 0,
                                "total_loss": 0.0,
                                "total_prob": 0.0,
                                "count": 0,
                            }
                        nongermline_mask = (non_templated_mask == 1)
                        combined = mask & nongermline_mask
                        acc = region_accumulators["nongermline"]
                        acc["correct"] += ((predictions == targets) & combined).sum().item()
                        acc["total_loss"] += (loss_per_token * combined.float()).sum().item()
                        acc["total_prob"] += (target_probs * combined.float()).sum().item()
                        acc["count"] += combined.sum().item()

        # Compute final metrics for enabled individual regions
        results: dict[str, float] = {}
        enabled_regions = config.get_enabled_regions()
        for region_name, acc in region_accumulators.items():
            if region_name in enabled_regions and acc["count"] > 0:
                results[f"{region_name}/accuracy"] = acc["correct"] / acc["count"]
                results[f"{region_name}/loss"] = acc["total_loss"] / acc["count"]
                results[f"{region_name}/prob"] = acc["total_prob"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results[f"{region_name}/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # Add aggregate metrics
        results.update(self._compute_aggregate_metrics(region_accumulators, config))

        return results

    def _run_per_position_eval(
        self,
        eval_loader: DataLoader,
        regions: set[AntibodyRegion] | None,
        position_batch_size: int,
        config: RegionEvalConfig,
    ) -> dict[str, float]:
        """Run per-position region evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            regions: Set of regions to evaluate, or None for all.
            position_batch_size: Batch size for per-position evaluation.
            config: Region evaluation configuration.

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

        # Determine which regions to evaluate for aggregates
        regions_for_aggregates = self._get_regions_for_aggregates(config)
        all_regions_needed = (regions or set()) | regions_for_aggregates

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc="Region eval (per-position)",
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
                        target_regions = all_regions_needed if all_regions_needed else None
                        sample_results = evaluator.evaluate_by_region(sample, target_regions)
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

        # Compute final metrics for enabled individual regions
        results: dict[str, float] = {}
        enabled_regions = config.get_enabled_regions()
        for region_name, acc in region_accumulators.items():
            if region_name in enabled_regions and acc["count"] > 0:
                results[f"{region_name}/accuracy"] = acc["correct"] / acc["count"]
                results[f"{region_name}/loss"] = acc["total_loss"] / acc["count"]
                results[f"{region_name}/prob"] = acc["total_prob"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results[f"{region_name}/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # Add aggregate metrics
        results.update(self._compute_aggregate_metrics(region_accumulators, config))

        return results

    def _run_region_level_eval(
        self,
        eval_loader: DataLoader,
        regions: set[AntibodyRegion] | None,
        config: RegionEvalConfig,
    ) -> dict[str, float]:
        """Run region-level (full region masking) evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            regions: Set of regions to evaluate, or None for all.
            config: Region evaluation configuration.

        Returns:
            Dictionary mapping region metric names to values.
        """
        device = _get_model_device(self.model, self.accelerator)
        evaluator = RegionMaskingEvaluator(model=self.model, device=device)

        # Accumulate results across all samples
        region_accumulators: dict[str, dict[str, float]] = {}

        # Determine which regions to evaluate for aggregates
        regions_for_aggregates = self._get_regions_for_aggregates(config)
        all_regions_needed = (regions or set()) | regions_for_aggregates

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                eval_loader,
                desc="Region eval (region-level)",
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
                        target_regions = all_regions_needed if all_regions_needed else None
                        sample_results = evaluator.evaluate_all_regions(sample, target_regions)
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

        # Compute final metrics for enabled individual regions
        results: dict[str, float] = {}
        enabled_regions = config.get_enabled_regions()
        for region_name, acc in region_accumulators.items():
            if region_name in enabled_regions and acc["count"] > 0:
                results[f"{region_name}/accuracy"] = acc["correct"] / acc["count"]
                results[f"{region_name}/loss"] = acc["total_loss"] / acc["count"]
                results[f"{region_name}/prob"] = acc["total_prob"] / acc["count"]
                avg_loss = acc["total_loss"] / acc["count"]
                results[f"{region_name}/ppl"] = torch.exp(torch.tensor(avg_loss)).item()

        # Add aggregate metrics
        results.update(self._compute_aggregate_metrics(region_accumulators, config))

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
