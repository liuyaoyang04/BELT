"""vLLM plugin (``vllm.general_plugins`` entry point) serving Algorithm 1 low-rank K/V exports with a latent KV cache.

Pinned to vLLM 0.32.0. Serves ``latent`` exports (``lrkv.runtime.export_latent``: BF16 latent rows with in-kernel
rebuilt RoPE keys) using architecture ``Qwen3_5LowRankLatentForConditionalGeneration`` (or
``Qwen3_5LowRankLatentForCausalLM``) and attention backend ``LRKV_LATENT`` (``latent_backend``: per-layer latent rows,
CuTe Hopper decode kernel, decompress + FlashAttention prefill).

Includes KV cache sizing for the hybrid model (``kv_cache``): block size and GDN page padding from the compressed
layers' own widths, and one compressed group plus GDN groups of the same layer count in the block-outermost layout.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(f"vllm.{__name__}")     # under "vllm" so vLLM's handler prints it

LATENT_ARCH = "Qwen3_5LowRankLatentForConditionalGeneration"
LATENT_ARCH_TEXT = "Qwen3_5LowRankLatentForCausalLM"
LATENT_BACKEND_NAME = "LRKV_LATENT"
LATENT_BACKEND_PATH = "vllm_lrkv.latent_backend.LatentAttentionBackend"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    for arch, stock in ((LATENT_ARCH, "Qwen3_5ForConditionalGeneration"),
                        (LATENT_ARCH_TEXT, "Qwen3_5ForCausalLM")):
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"vllm_lrkv.models.qwen3_5_latent:{arch}")
        # vLLM's per-architecture config hooks are keyed by name; without the stock one the GDN SSM state cache
        # stays at the model dtype (BF16) instead of the config's mamba_ssm_dtype (FP32)
        MODELS_CONFIG_MAP.setdefault(arch, MODELS_CONFIG_MAP[stock])
    if LATENT_BACKEND_NAME not in AttentionBackendEnum._member_map_:
        AttentionBackendEnum.register(LATENT_BACKEND_NAME, LATENT_BACKEND_PATH)
    from .kv_cache import install as install_kv_cache
    install_kv_cache()
    logger.info("vllm_lrkv: registered %s / %s, attention backend %s",
                LATENT_ARCH, LATENT_ARCH_TEXT, LATENT_BACKEND_NAME)
