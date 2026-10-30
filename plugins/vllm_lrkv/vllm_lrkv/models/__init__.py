"""Model definitions for LRKV true-latent Qwen3.5 inference."""
from __future__ import annotations

from .qwen3_5_latent import (
    Qwen3_5LowRankLatentForCausalLM,
    Qwen3_5LowRankLatentForConditionalGeneration,
)

__all__ = [
    "Qwen3_5LowRankLatentForCausalLM",
    "Qwen3_5LowRankLatentForConditionalGeneration",
]

