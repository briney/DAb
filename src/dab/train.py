"""Training entry point for use with accelerate launch."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .data import create_dataloader
from .diffusion import create_schedule
from .logging import WandbLogger
from .model import DAbConfig, DAbModel
from .training import Trainer, TrainingConfig
from .utils import set_seed


def run_training(
    config_path: str | None = None,
    config_dir: str = "configs",
    train_data: str | None = None,
    eval_data: str | None = None,
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
    train_data
        Path to training data.
    eval_data
        Optional path to evaluation data.
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

        if train_data:
            override_list.append(f"data.train_data={train_data}")
        if eval_data:
            override_list.append(f"data.eval_data={eval_data}")

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
        head_dim=cfg.model.head_dim,
        d_ffn=cfg.model.d_ffn,
        max_seq_len=cfg.model.max_seq_len,
        max_timesteps=cfg.model.max_timesteps,
        use_timestep_embedding=cfg.model.use_timestep_embedding,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        embedding_dropout=cfg.model.embedding_dropout,
    )
    model = DAbModel(model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    # Create data loaders
    train_loader = create_dataloader(
        data_path=cfg.data.train_data,
        batch_size=cfg.training.batch_size,
        max_length=cfg.data.max_length,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
        pad_to_max=cfg.data.pad_to_max,
    )

    eval_loader = None
    if cfg.data.eval_data:
        eval_loader = create_dataloader(
            data_path=cfg.data.eval_data,
            batch_size=cfg.training.batch_size,
            max_length=cfg.data.max_length,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
            pad_to_max=cfg.data.pad_to_max,
        )

    # Create noise schedule and training config
    noise_schedule = create_schedule(
        cfg.diffusion.schedule_type, cfg.diffusion.num_timesteps
    )

    training_config = TrainingConfig(
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        max_grad_norm=cfg.training.max_grad_norm,
        scheduler_type=cfg.training.scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        min_lr_ratio=cfg.training.min_lr_ratio,
        noise_schedule=cfg.diffusion.schedule_type,
        num_timesteps=cfg.diffusion.num_timesteps,
        weight_multiplier=cfg.diffusion.weight_multiplier,
        log_every_n_steps=cfg.training.log_every_n_steps,
        eval_every_n_steps=cfg.training.eval_every_n_steps,
        save_every_n_steps=cfg.training.save_every_n_steps,
        checkpoint_dir=cfg.training.checkpoint_dir,
        keep_last_n_checkpoints=cfg.training.keep_last_n_checkpoints,
        save_best=cfg.training.save_best,
        seed=cfg.seed,
        mixed_precision=cfg.training.mixed_precision,
    )

    # Create trainer
    trainer = Trainer(
        config=training_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        noise_schedule=noise_schedule,
    )

    # Set up logging
    if use_wandb and cfg.logging.enabled:
        logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.logging.entity,
            tags=list(cfg.logging.tags) if cfg.logging.tags else None,
            notes=cfg.logging.notes,
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
    parser.add_argument("--train-data", "-t", required=True, help="Training data path")
    parser.add_argument("--eval-data", "-e", default=None, help="Evaluation data path")
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--name", "-n", default="dab_experiment", help="Experiment name")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")

    args, unknown = parser.parse_known_args()

    run_training(
        config_dir=args.config_dir,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        name=args.name,
        resume_from=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        overrides=unknown,
    )


if __name__ == "__main__":
    main()
