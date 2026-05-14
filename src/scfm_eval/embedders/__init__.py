"""Embedders for foundation models. Registry is lazy — see registry.py."""
from .base import BaseEmbedder
from .registry import (
    get_embedder,
    get_embedder_class,
    list_embedders,
    register_embedder,
)

__all__ = [
    'BaseEmbedder',
    'get_embedder',
    'get_embedder_class',
    'list_embedders',
    'register_embedder',
]
