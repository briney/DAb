"""Command-line interface for DAb."""

from __future__ import annotations

from pathlib import Path

import click

from .version import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """DAb: Discrete Diffusion Antibody Language Model."""
    pass


@main.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True),
    default="configs",
    help="Config directory",
)
@click.option(
    "--train-data",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Training data path",
)
@click.option(
    "--eval-data", "-e", type=click.Path(exists=True), help="Evaluation data path"
)
@click.option(
    "--output-dir", "-o", type=click.Path(), default="outputs", help="Output directory"
)
@click.option("--name", "-n", default="dab_experiment", help="Experiment name")
@click.option(
    "--resume", type=click.Path(exists=True), help="Checkpoint to resume from"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--wandb/--no-wandb", default=True, help="Enable/disable WandB")
@click.argument("overrides", nargs=-1)
def train(
    config: str | None,
    config_dir: str,
    train_data: str,
    eval_data: str | None,
    output_dir: str,
    name: str,
    resume: str | None,
    seed: int,
    wandb: bool,
    overrides: tuple[str, ...],
) -> None:
    """Train a DAb model.

    Examples:

        dab train --train-data data/train.csv --eval-data data/val.csv

        dab train -t data/train.csv model=small training.batch_size=64
    """
    from .train import run_training

    run_training(
        config_path=config,
        config_dir=config_dir,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=output_dir,
        name=name,
        resume_from=resume,
        seed=seed,
        use_wandb=wandb,
        overrides=list(overrides),
    )


@main.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Model checkpoint",
)
@click.option(
    "--input", "-i", type=click.Path(exists=True), required=True, help="Input file"
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file (.pt or .npy)"
)
@click.option(
    "--pooling",
    "-p",
    type=click.Choice(["mean", "cls", "max", "mean_max", "none"]),
    default="none",
    help="Pooling strategy",
)
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--device", "-d", default=None, help="Device to run on")
def encode(
    checkpoint: str,
    input: str,
    output: str,
    pooling: str,
    batch_size: int,
    device: str | None,
) -> None:
    """Encode antibody sequences using a trained model.

    Examples:

        dab encode -c checkpoints/best.pt -i data/seqs.csv -o embeddings.pt

        dab encode -c model.pt -i seqs.parquet -o emb.npy --pooling mean
    """
    import numpy as np
    import pandas as pd
    import torch

    from .encoding import DAbEncoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"Loading model from {checkpoint}...")

    pooling_strategy = None if pooling == "none" else pooling
    encoder = DAbEncoder.from_pretrained(checkpoint, device=device, pooling=pooling_strategy)

    click.echo(f"Loading data from {input}...")

    if input.endswith(".parquet"):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)

    heavy_chains = df["heavy_chain"].tolist()
    light_chains = df["light_chain"].tolist()

    click.echo(f"Encoding {len(heavy_chains)} sequences...")

    return_numpy = output.endswith(".npy")
    embeddings = encoder.encode_batch(
        heavy_chains, light_chains, return_numpy=return_numpy, batch_size=batch_size
    )

    click.echo(f"Saving embeddings to {output}...")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".npy"):
        if isinstance(embeddings, list):
            np.save(output, np.array(embeddings, dtype=object))
        else:
            np.save(output, embeddings)
    elif output.endswith(".pt"):
        torch.save(embeddings, output)
    else:
        raise click.ClickException(f"Unknown output format: {output}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
