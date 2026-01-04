"""Main training loop with Accelerate integration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..diffusion import InformationWeightedMasker, NoiseSchedule, UniformMasker
from ..model import DAbModel
from .checkpoint import CheckpointConfig, CheckpointManager
from .metrics import (
    DiffusionMetrics,
    MetricAccumulator,
    compute_diffusion_metrics,
    compute_masked_cross_entropy,
)
from .optimizer import create_optimizer, create_scheduler, get_lr

if TYPE_CHECKING:
    from ..eval import Evaluator


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Duration (step-driven by default)
    max_steps: int = 100000
    max_epochs: int | None = None

    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_decay: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Diffusion
    noise_schedule: str = "cosine"
    num_timesteps: int = 100
    cdr_weight_multiplier: float = 1.0
    nongermline_weight_multiplier: float = 1.0

    # Intervals (in steps)
    log_steps: int = 10
    eval_steps: int = 500
    checkpoint_steps: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5
    save_best: bool = True

    # Reproducibility
    seed: int = 42

    # Mixed precision
    mixed_precision: str = "no"


class Trainer:
    """Main trainer class with Accelerate integration."""

    def __init__(
        self,
        config: TrainingConfig,
        model: DAbModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        eval_dataloaders: dict[str, DataLoader] | None = None,
        noise_schedule: NoiseSchedule | None = None,
        evaluator: "Evaluator | None" = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        self.config = config

        # Use provided accelerator or create a new one
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # Let accelerate handle mixed_precision from its config (accelerate config)
            # rather than overriding with our config value
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )

        self.optimizer = create_optimizer(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_decay=config.scheduler_decay,
            num_training_steps=config.max_steps,
            num_warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(model, self.optimizer, train_dataloader, self.scheduler)

        # Support both single eval_dataloader (legacy) and multiple eval_dataloaders
        self.eval_dataloader = (
            self.accelerator.prepare(eval_dataloader) if eval_dataloader else None
        )

        # Prepare multiple eval dataloaders if provided
        self.eval_dataloaders: dict[str, DataLoader] = {}
        if eval_dataloaders:
            for name, loader in eval_dataloaders.items():
                self.eval_dataloaders[name] = self.accelerator.prepare(loader)
        elif self.eval_dataloader is not None:
            # Use single eval_dataloader as "validation" if no multi-loader dict provided
            self.eval_dataloaders["validation"] = self.eval_dataloader

        # Store evaluator for advanced metrics
        self.evaluator = evaluator

        if noise_schedule is None:
            from ..diffusion import create_schedule

            noise_schedule = create_schedule(config.noise_schedule, config.num_timesteps)

        self.masker = InformationWeightedMasker(
            noise_schedule,
            cdr_weight_multiplier=config.cdr_weight_multiplier,
            nongermline_weight_multiplier=config.nongermline_weight_multiplier,
        )
        self.uniform_masker = UniformMasker(noise_schedule)

        checkpoint_config = CheckpointConfig(
            save_dir=config.checkpoint_dir,
            checkpoint_steps=config.checkpoint_steps,
            keep_last_n=config.keep_last_n_checkpoints,
            save_best=config.save_best,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config,
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
        )

        self.metrics = MetricAccumulator()
        self.global_step = 0
        self.epoch = 0.0
        self.steps_per_epoch = len(self.train_dataloader)
        self.logger = None

    def set_logger(self, logger) -> None:
        """Set the logger for training metrics."""
        self.logger = logger

    def set_evaluator(self, evaluator: "Evaluator") -> None:
        """Set the evaluator for advanced metrics.

        Args:
            evaluator: Evaluator instance for computing metrics.
        """
        self.evaluator = evaluator

    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply masking to a batch."""
        device = batch["token_ids"].device
        batch_size = batch["token_ids"].shape[0]

        timesteps = self.masker.noise_schedule.sample_timesteps(batch_size, device)

        if batch.get("cdr_mask") is not None or batch.get("non_templated_mask") is not None:
            masked_ids, mask_labels = self.masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                cdr_mask=batch.get("cdr_mask"),
                non_templated_mask=batch.get("non_templated_mask"),
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
        else:
            masked_ids, mask_labels = self.uniform_masker.apply_mask(
                token_ids=batch["token_ids"],
                timesteps=timesteps,
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch.get("special_tokens_mask"),
            )

        return {"masked_ids": masked_ids, "mask_labels": mask_labels, "timesteps": timesteps}

    def training_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, DiffusionMetrics]:
        """Execute a single training step.

        Returns:
            Tuple of (loss tensor for backprop, DiffusionMetrics with all metrics).
        """
        mask_output = self._apply_masking(batch)

        outputs = self.model(
            token_ids=mask_output["masked_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        metrics = compute_diffusion_metrics(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
            attention_mask=batch["attention_mask"],
        )

        # Recompute loss tensor for backprop (metrics.loss is a float)
        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
        )

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset (legacy single-dataset method)."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        eval_metrics = MetricAccumulator()

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            mask_output = self._apply_masking(batch)

            outputs = self.model(
                token_ids=mask_output["masked_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            metrics = compute_diffusion_metrics(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_output["mask_labels"],
                attention_mask=batch["attention_mask"],
            )

            eval_metrics.update("loss", metrics.loss)
            eval_metrics.update("accuracy", metrics.accuracy)
            eval_metrics.update("perplexity", metrics.perplexity)

        self.model.train()

        return {
            "val_loss": eval_metrics.compute("loss"),
            "val_accuracy": eval_metrics.compute("accuracy"),
            "val_perplexity": eval_metrics.compute("perplexity"),
        }

    def evaluate_all(self) -> dict[str, dict[str, float]]:
        """Run evaluation on all configured eval datasets.

        Uses the Evaluator if available for advanced metrics, otherwise
        falls back to basic metrics.

        Returns:
            Dictionary mapping eval dataset names to their metric results.
        """
        if not self.eval_dataloaders:
            return {}

        if self.evaluator is not None:
            # Use advanced evaluator
            return self.evaluator.evaluate_all(
                self.eval_dataloaders,
                masker=self.uniform_masker,
            )

        # Fall back to simple evaluation for each dataset
        all_results: dict[str, dict[str, float]] = {}
        self.model.eval()

        for eval_name, eval_loader in self.eval_dataloaders.items():
            eval_metrics = MetricAccumulator()

            for batch in tqdm(
                eval_loader,
                desc=f"Eval ({eval_name})",
                disable=not self.accelerator.is_local_main_process,
            ):
                mask_output = self._apply_masking(batch)

                outputs = self.model(
                    token_ids=mask_output["masked_ids"],
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                metrics = compute_diffusion_metrics(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_output["mask_labels"],
                    attention_mask=batch["attention_mask"],
                )

                eval_metrics.update("loss", metrics.loss)
                eval_metrics.update("accuracy", metrics.accuracy)
                eval_metrics.update("perplexity", metrics.perplexity)

            all_results[eval_name] = {
                "loss": eval_metrics.compute("loss"),
                "accuracy": eval_metrics.compute("accuracy"),
                "perplexity": eval_metrics.compute("perplexity"),
            }

        self.model.train()
        return all_results

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()

        if self.config.max_epochs is not None:
            steps_per_epoch = len(self.train_dataloader)
            total_steps = self.config.max_epochs * steps_per_epoch
        else:
            total_steps = self.config.max_steps

        self.accelerator.print(f"Starting training for {total_steps} steps...")

        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
            file=sys.stdout,  # Explicit stdout for proper flushing with accelerate
        )
        progress_bar.update(self.global_step)

        while self.global_step < total_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, step_metrics = self.training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.epoch = self.global_step / self.steps_per_epoch
                    progress_bar.update(1)

                    self.metrics.update("train_loss", step_metrics.loss)
                    self.metrics.update("train_accuracy", step_metrics.accuracy)
                    self.metrics.update("train_perplexity", step_metrics.perplexity)

                    # Pre-compute conditions for this step
                    should_log = self.global_step % self.config.log_steps == 0
                    should_eval = (
                        self.config.eval_steps > 0
                        and self.global_step % self.config.eval_steps == 0
                    )
                    should_checkpoint = self.checkpoint_manager.should_save(self.global_step)

                    # Cache eval results to avoid running eval twice
                    all_eval_metrics: dict[str, dict[str, float]] | None = None

                    # Logging
                    if should_log:
                        log_metrics = self.metrics.compute_all()
                        log_metrics["learning_rate"] = get_lr(self.optimizer)
                        log_metrics["epoch"] = self.epoch
                        log_metrics["step"] = self.global_step

                        if self.logger is not None:
                            # Use commit=False if eval will also log at this step
                            # to avoid wandb non-monotonic step warnings
                            self.logger.log(
                                log_metrics,
                                step=self.global_step,
                                commit=not should_eval,
                            )

                        self.metrics.reset()

                    # Evaluation
                    if should_eval:
                        all_eval_metrics = self.evaluate_all()
                        if self.logger is not None and all_eval_metrics:
                            # Use log_eval_all if available, otherwise flatten and log
                            if hasattr(self.logger, "log_eval_all"):
                                self.logger.log_eval_all(all_eval_metrics, step=self.global_step)
                            else:
                                # Flatten metrics for basic logging
                                flat_metrics = {}
                                for eval_name, metrics in all_eval_metrics.items():
                                    for metric_name, value in metrics.items():
                                        flat_metrics[f"{eval_name}/{metric_name}"] = value
                                self.logger.log(flat_metrics, step=self.global_step)

                    # Checkpointing - reuse eval results if already computed
                    # Only save checkpoints on main process to avoid file conflicts
                    if should_checkpoint and self.accelerator.is_main_process:
                        # Only run eval if we haven't already at this step
                        if all_eval_metrics is None and self.eval_dataloaders:
                            all_eval_metrics = self.evaluate_all()

                        # Flatten for checkpoint manager
                        if all_eval_metrics:
                            eval_metrics = {}
                            for eval_name, metrics in all_eval_metrics.items():
                                for metric_name, value in metrics.items():
                                    eval_metrics[f"{eval_name}/{metric_name}"] = value
                        else:
                            eval_metrics = {}

                        self.checkpoint_manager.save(
                            step=self.global_step, epoch=self.epoch, metrics=eval_metrics
                        )

                    if self.global_step >= total_steps:
                        break

        progress_bar.close()

        # Final checkpoint
        if self.accelerator.is_main_process:
            if self.eval_dataloaders:
                all_eval_metrics = self.evaluate_all()
                # Flatten for checkpoint manager
                final_metrics = {}
                for eval_name, metrics in all_eval_metrics.items():
                    for metric_name, value in metrics.items():
                        final_metrics[f"{eval_name}/{metric_name}"] = value
            else:
                final_metrics = {}
            self.checkpoint_manager.save(
                step=self.global_step, epoch=self.epoch, metrics=final_metrics
            )

            if self.logger is not None:
                self.logger.finish()
