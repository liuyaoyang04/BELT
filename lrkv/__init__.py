"""lrkv: behavior-aligned low-rank K/V projections for the KV cache -- bases, rank allocation, checkpoints.

    lrkv.modules         canonical low-rank projection modules (checkpoint layout)
    lrkv.checkpoint      checkpoint format and loading
    lrkv.model           high-level loading (low-rank modules inside the stock attention)
    lrkv.ours            behavior-aligned Kronecker-surrogate per-head K + joint gated Value SVD
    lrkv.layer_parallel  per-layer CUDA worker pool for full bases
    lrkv.allocate        frozen-basis output-KD rank allocation CLI
    lrkv.allocation      exact-budget prefix selector (nonquantized)
    lrkv.basis           full bases and selected-rank export
    lrkv.whitened        streamed RoPE key curvature surrogate and weighted closed-form solvers
    lrkv.experiment_data shared calibration and rank-training token preparation (docs.jsonl)
    lrkv.compression     KV-cache size accounting for a checkpoint
    lrkv.arch            hybrid-stack (Qwen3.8) architecture helpers
"""

__version__ = "0.1.0"

_LAZY = {
    "HeadwiseLowRankLinear": ("lrkv.modules", "HeadwiseLowRankLinear"),
    "LowRankLinear": ("lrkv.modules", "LowRankLinear"),
    "load_base_model": ("lrkv.model", "load_base_model"),
    "load_compressed_model": ("lrkv.model", "load_compressed_model"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'lrkv' has no attribute {name!r}")


__all__ = list(_LAZY) + ["__version__"]
