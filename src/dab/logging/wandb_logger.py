"""Weights & Biases logging integration."""

from __future__ import annotations

from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Wrapper for WandB logging."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        resume: bool = False,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and WANDB_AVAILABLE

        if not self.enabled:
            self.run = None
            return

        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            tags=tags,
            notes=notes,
            resume="allow" if resume else None,
        )

    def log(
        self, metrics: dict[str, Any], step: int | None = None, commit: bool = True
    ) -> None:
        """Log metrics to WandB."""
        if not self.enabled:
            return
        wandb.log(metrics, step=step, commit=commit)

    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact to WandB."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch model gradients and parameters."""
        if not self.enabled:
            return
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the WandB run."""
        if not self.enabled:
            return
        wandb.finish()

    @property
    def run_id(self) -> str | None:
        """Get the WandB run ID."""
        if not self.enabled or self.run is None:
            return None
        return self.run.id

    @property
    def run_url(self) -> str | None:
        """Get the WandB run URL."""
        if not self.enabled or self.run is None:
            return None
        return self.run.url
