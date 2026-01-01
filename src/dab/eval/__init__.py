"""Evaluation harness for computing in-training validation metrics.

This module provides a flexible, extensible system for computing various
metrics during training. It supports:

- Multiple evaluation datasets with independent configurations
- Per-dataset metric selection and parameterization
- Distributed training aggregation
- Metrics computed from logits, embeddings, or attention weights
- Fitting small models (like logistic regression) on representations

Example usage:
    from dab.eval import Evaluator, build_metrics

    # Create evaluator
    evaluator = Evaluator(cfg, model, accelerator)

    # Evaluate on a dataset
    results = evaluator.evaluate(eval_loader, "validation")

    # Or evaluate on all configured datasets
    all_results = evaluator.evaluate_all(eval_loaders)
"""

from .base import Metric, MetricBase
from .evaluator import Evaluator
from .registry import build_metrics, get_metric_class, list_metrics, register_metric

__all__ = [
    # Core classes
    "Metric",
    "MetricBase",
    "Evaluator",
    # Registry functions
    "register_metric",
    "get_metric_class",
    "list_metrics",
    "build_metrics",
]
