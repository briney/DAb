"""Training entry point for use with accelerate launch."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .data import create_eval_dataloaders, create_train_dataloader
from .diffusion import create_schedule
from .logging import WandbLogger
from .model import DAbConfig, DAbModel
from .training import Trainer, TrainingConfig
from .utils import set_seed


def run_training(
    config_path: str | None = None,
    config_dir: str = "configs",
    train: str | None = None,
    output_dir: str = "outputs",
    name: str = "dab_experiment",
    resume_from: str | None = None,
    seed: int = 42,
    use_wandb: bool = True,
    overrides: list[str] | None = None,
) -> None:
    """Main training function.

    Parameters
    ----------
    config_path
        Optional path to a specific config file.
    config_dir
        Directory containing Hydra configs.
    train
        Path to training data (single dataset). For multi-dataset,
        configure via data.train in config or overrides.
    output_dir
        Output directory for checkpoints and logs.
    name
        Experiment name.
    resume_from
        Optional checkpoint path to resume from.
    seed
        Random seed.
    use_wandb
        Whether to enable WandB logging.
    overrides
        List of Hydra config overrides.
    """
    config_dir_path = Path(config_dir).absolute()

    with initialize_config_dir(config_dir=str(config_dir_path), version_base=None):
        override_list = overrides or []
        override_list.extend(
            [f"name={name}", f"seed={seed}", f"output_dir={output_dir}"]
        )

        # Handle CLI override for train path
        if train:
            override_list.append(f"data.train={train}")

        cfg = compose(config_name="config", overrides=override_list)

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path / "config.yaml")

    # Create model
    model_config = DAbConfig(
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ffn=cfg.model.d_ffn,
        ffn_multiplier=cfg.model.ffn_multiplier,
        max_seq_len=cfg.model.max_seq_len,
        max_timesteps=cfg.model.max_timesteps,
        use_timestep_embedding=cfg.model.use_timestep_embedding,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        embedding_dropout=cfg.model.embedding_dropout,
    )
    model = DAbModel(model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    # Create train dataloader (handles single or multi-dataset automatically)
    train_loader = create_train_dataloader(
        cfg=cfg.data,
        batch_size=cfg.train.batch_size,
    )

    # Create eval dataloaders (handles multiple eval datasets)
    eval_dataloaders = create_eval_dataloaders(
        cfg=cfg.data,
        default_batch_size=cfg.train.batch_size,
    )

    # Create noise schedule and training config
    noise_schedule = create_schedule(
        cfg.diffusion.schedule_type, cfg.diffusion.num_timesteps
    )

    training_config = TrainingConfig(
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.optimizer.learning_rate,
        weight_decay=cfg.train.optimizer.weight_decay,
        betas=tuple(cfg.train.optimizer.betas),
        max_grad_norm=cfg.train.max_grad_norm,
        scheduler_decay=cfg.train.scheduler.decay,
        warmup_steps=cfg.train.scheduler.warmup_steps,
        min_lr_ratio=cfg.train.scheduler.min_lr_ratio,
        noise_schedule=cfg.diffusion.schedule_type,
        num_timesteps=cfg.diffusion.num_timesteps,
        weight_multiplier=cfg.diffusion.weight_multiplier,
        log_steps=cfg.train.log_steps,
        eval_steps=cfg.train.eval_steps,
        checkpoint_steps=cfg.train.checkpoint_steps,
        checkpoint_dir=cfg.train.checkpoint_dir,
        keep_last_n_checkpoints=cfg.train.keep_last_n_checkpoints,
        save_best=cfg.train.save_best,
        seed=cfg.seed,
        mixed_precision=cfg.train.mixed_precision,
    )

    # Create trainer with eval dataloaders
    trainer = Trainer(
        config=training_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloaders=eval_dataloaders if eval_dataloaders else None,
        noise_schedule=noise_schedule,
    )

    # Set up logging
    if use_wandb and cfg.log.wandb.enabled:
        logger = WandbLogger(
            project=cfg.log.wandb.project,
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.log.wandb.entity,
            tags=list(cfg.log.wandb.tags) if cfg.log.wandb.tags else None,
            notes=cfg.log.wandb.notes,
        )
        trainer.set_logger(logger)

    # Resume if specified
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.checkpoint_manager.load(resume_from)

    # Train
    trainer.train()


def main() -> None:
    """Entry point for `accelerate launch -m dab.train`."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a DAb model")
    parser.add_argument(
        "--train", "-t", required=True, help="Training data path (single dataset)"
    )
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--name", "-n", default="dab_experiment", help="Experiment name")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")

    args, unknown = parser.parse_known_args()

    run_training(
        config_dir=args.config_dir,
        train=args.train,
        output_dir=args.output_dir,
        name=args.name,
        resume_from=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        overrides=unknown,
    )


if __name__ == "__main__":
    main()
