"""Runtime export utilities for LRKV low-rank KV cache models (vLLM latent format)."""
from __future__ import annotations

__all__ = ["FORMAT", "build_latent_layer", "export", "k_columns", "k_segments", "rope_basis", "row_layout"]


def __getattr__(name: str):
    if name in __all__:
        from . import export_latent
        return getattr(export_latent, name)
    raise AttributeError(f"module 'lrkv.runtime' has no attribute {name!r}")

