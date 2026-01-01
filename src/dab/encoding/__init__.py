"""Encoding API for extracting embeddings."""

from .encoder import DAbEncoder
from .pooling import (
    CLSPooling,
    MaxPooling,
    MeanMaxPooling,
    MeanPooling,
    PoolingStrategy,
    PoolingType,
    create_pooling,
)

__all__ = [
    "DAbEncoder",
    "PoolingStrategy",
    "PoolingType",
    "MeanPooling",
    "CLSPooling",
    "MaxPooling",
    "MeanMaxPooling",
    "create_pooling",
]
