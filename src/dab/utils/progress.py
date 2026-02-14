"""Progress bar utilities using Rich."""

from __future__ import annotations

from typing import IO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


def create_progress(*, disable: bool = False, file: IO | None = None) -> Progress:
    """Create a configured Rich Progress instance.

    Parameters
    ----------
    disable
        If True, suppress all output. Used to silence progress bars on
        non-main processes in distributed training.
    file
        File object for console output. If None, uses stderr (Rich default).
        Pass ``sys.stdout`` for compatibility with Accelerate's output handling.
    """
    console = Console(file=file) if file is not None else None

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=disable,
    )
