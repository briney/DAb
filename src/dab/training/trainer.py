"""Main training loop with Accelerate integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..diffusion import InformationWeightedMasker, NoiseSchedule, UniformMasker
from ..model import DAbModel
from .checkpoint import CheckpointConfig, CheckpointManager
from .metrics import (
    MetricAccumulator,
    compute_diffusion_metrics,
    compute_masked_cross_entropy,
)
from .optimizer import create_optimizer, create_scheduler, get_lr


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Duration (step-driven by default)
    max_steps: int = 100000
    max_epochs: Optional[int] = None

    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Diffusion
    noise_schedule: str = "cosine"
    num_timesteps: int = 100
    weight_multiplier: float = 1.0

    # Intervals (in steps)
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000

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
        eval_dataloader: Optional[DataLoader] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
    ) -> None:
        self.config = config

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

        self.optimizer = create_optimizer(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler_type,
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

        self.eval_dataloader = (
            self.accelerator.prepare(eval_dataloader) if eval_dataloader else None
        )

        if noise_schedule is None:
            from ..diffusion import create_schedule

            noise_schedule = create_schedule(config.noise_schedule, config.num_timesteps)

        self.masker = InformationWeightedMasker(noise_schedule, config.weight_multiplier)
        self.uniform_masker = UniformMasker(noise_schedule)

        checkpoint_config = CheckpointConfig(
            save_dir=config.checkpoint_dir,
            save_every_n_steps=config.save_every_n_steps,
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
        self.epoch = 0
        self.logger = None

    def set_logger(self, logger) -> None:
        """Set the logger for training metrics."""
        self.logger = logger

    def _apply_masking(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply masking to a batch."""
        device = batch["token_ids"].device
        batch_size = batch["token_ids"].shape[0]

        timesteps = self.masker.noise_schedule.sample_timesteps(batch_size, device)

        if (
            batch.get("cdr_mask") is not None
            or batch.get("non_templated_mask") is not None
        ):
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

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step."""
        mask_output = self._apply_masking(batch)

        outputs = self.model(
            token_ids=mask_output["masked_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
        )

        return loss

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset."""
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

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()

        if self.config.max_epochs is not None:
            steps_per_epoch = len(self.train_dataloader)
            total_steps = self.config.max_epochs * steps_per_epoch
        else:
            total_steps = self.config.max_steps

        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.update(self.global_step)

        while self.global_step < total_steps:
            self.epoch += 1

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss = self.training_step(batch)
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
                    progress_bar.update(1)

                    self.metrics.update("train_loss", loss.item())

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        log_metrics = self.metrics.compute_all()
                        log_metrics["learning_rate"] = get_lr(self.optimizer)
                        log_metrics["epoch"] = self.epoch
                        log_metrics["step"] = self.global_step

                        if self.logger is not None:
                            self.logger.log(log_metrics, step=self.global_step)

                        self.metrics.reset()

                    # Evaluation
                    if (
                        self.config.eval_every_n_steps > 0
                        and self.global_step % self.config.eval_every_n_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        if self.logger is not None and eval_metrics:
                            self.logger.log(eval_metrics, step=self.global_step)

                    # Checkpointing
                    if self.checkpoint_manager.should_save(self.global_step):
                        eval_metrics = self.evaluate() if self.eval_dataloader else {}
                        self.checkpoint_manager.save(
                            step=self.global_step, epoch=self.epoch, metrics=eval_metrics
                        )

                    if self.global_step >= total_steps:
                        break

        progress_bar.close()

        # Final checkpoint
        if self.accelerator.is_main_process:
            final_metrics = self.evaluate() if self.eval_dataloader else {}
            self.checkpoint_manager.save(
                step=self.global_step, epoch=self.epoch, metrics=final_metrics
            )

            if self.logger is not None:
                self.logger.finish()
