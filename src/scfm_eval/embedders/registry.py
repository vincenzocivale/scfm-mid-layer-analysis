"""Lazy registry of foundation-model embedders.

Each embedder is registered by name with a (module_path, class_name) pair so
that heavy ML dependencies (torch, tahoe_x1, ...) are imported only when the
embedder is actually instantiated. This lets the package be imported in
analysis notebooks / CI without GPU-stack overhead.
"""
from importlib import import_module
from typing import Dict, Optional, Tuple, Type

from .base import BaseEmbedder

# name → (module dotted path, class name)
_REGISTRY: Dict[str, Tuple[str, str]] = {}
# cache of already-imported classes
_CACHE: Dict[str, Type[BaseEmbedder]] = {}


def register_embedder(name: str, module: str, class_name: str) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Embedder {name!r} already registered")
    _REGISTRY[name] = (module, class_name)


def get_embedder_class(name: str) -> Type[BaseEmbedder]:
    if name in _CACHE:
        return _CACHE[name]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown embedder {name!r}. Registered: {sorted(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name]
    cls = getattr(import_module(module_path), class_name)
    _CACHE[name] = cls
    return cls


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Instantiate an embedder by registered name. Extra kwargs go to the constructor."""
    return get_embedder_class(name)(**kwargs)


def list_embedders() -> list:
    return sorted(_REGISTRY)


# Built-in registrations (importable cheaply — no torch/tahoe_x1 yet).
register_embedder('scfoundation', 'scfm_eval.embedders.scfoundation', 'scFoundationEmbedder')
register_embedder('tahoe', 'scfm_eval.embedders.tahoe', 'TahoeEmbedder')
