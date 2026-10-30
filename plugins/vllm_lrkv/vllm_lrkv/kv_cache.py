"""KV cache block size and grouping for the compressed latent exports (format ``latent``) of hybrid
(GDN + full attention) models.

vLLM 0.32 sizes a hybrid model's attention block from the model-level head size, and the packing heuristic
(``kv_cache_utils._get_packed_kv_cache_groups``) finds no repeating page pattern among layers whose latent widths
differ, putting every GDN layer in a group of its own.

For ``latent`` models this module replaces both decisions; every other model goes through vLLM's code unchanged:

* block size: the smallest backend-aligned B with B * sum_i(page_i per token) >= n_latent * GDN state, so one block
  of all compressed layers holds as many bytes as n_latent GDN states; the GDN page is padded to that block's
  bytes / n_latent;
* groups: all compressed layers in one group and the GDN layers in groups of n_latent (model order), so every group
  fills the same block.
"""

from __future__ import annotations

from functools import wraps
import logging

logger = logging.getLogger(f"vllm.{__name__}")     # under "vllm" so vLLM's handler prints it


def _form(vllm_config) -> str | None:
    model_config = vllm_config.model_config
    block = getattr(model_config.hf_text_config, "kv_lowrank", None) if model_config is not None else None
    return block.get("format") if isinstance(block, dict) else None


def _applies(vllm_config) -> bool:
    return _form(vllm_config) == "latent"


def _mamba_page_size(vllm_config) -> int:
    """The unpadded GDN state page, computed the way ``Platform._align_hybrid_block_size`` does."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.kv_cache_interface import MambaSpec

    model_config = vllm_config.model_config
    model_cls, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config=model_config)
    if hasattr(model_cls, "get_mamba_specs_from_config"):
        return max(spec.page_size_bytes for spec in model_cls.get_mamba_specs_from_config(vllm_config))
    return MambaSpec(shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
                     dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config), block_size=-1).page_size_bytes


def _latent_page_sizes_1_token(vllm_config) -> list[int]:
    """Per-token cache bytes of every compressed layer of the model."""
    from vllm.utils.torch_utils import get_dtype_size
    block = vllm_config.model_config.hf_text_config.kv_lowrank
    size = get_dtype_size(vllm_config.model_config.dtype)
    return [int(info["row_width"]) * size for info in block["layers"].values()]


def _resize_block(platform_cls, vllm_config, backend_cls, pre_block_size: int) -> None:
    from math import lcm

    from vllm.config.vllm import set_current_vllm_config
    from vllm.utils.math_utils import cdiv
    from vllm.v1.attention.backend import MultipleOf

    cache_config = vllm_config.cache_config
    pages = _latent_page_sizes_1_token(vllm_config)
    mamba_page = _mamba_page_size(vllm_config)
    if not pages or mamba_page == 0:
        return
    n_latent, latent_bytes_1_token = len(pages), sum(pages)

    def smallest(sizes):
        return min(s.base if isinstance(s, MultipleOf) else s for s in sizes)

    with set_current_vllm_config(vllm_config):
        unit = max(smallest(backend_cls.get_supported_kernel_block_sizes()), pre_block_size, 1)
        backend_alignment = lcm(*(smallest(sizes) for b in platform_cls._find_non_ssm_backends(vllm_config)
                                  if (sizes := b.get_supported_kernel_block_sizes())))
    block_size = unit * cdiv(n_latent * mamba_page, unit * latent_bytes_1_token)
    block_size = backend_alignment * cdiv(block_size, backend_alignment)

    cache_config.block_size = block_size
    if cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = block_size
    padded = cdiv(block_size * latent_bytes_1_token, n_latent)
    cache_config.mamba_page_size_padded = padded if padded != mamba_page else None
    logger.info("vllm_lrkv: %d compressed layers (%d cache bytes per token in all), block size %d tokens; GDN page "
                "padded by %.2f%% to %d bytes so %d GDN states fill one block", n_latent, latent_bytes_1_token,
                block_size, 100 * (padded - mamba_page) / mamba_page, padded, n_latent)


def _align_hybrid_block_size(original):
    @wraps(original)
    def align(cls, vllm_config, backend_cls):
        pre_block_size = vllm_config.cache_config.block_size
        original(cls, vllm_config, backend_cls)
        if _applies(vllm_config):
            _resize_block(cls, vllm_config, backend_cls, pre_block_size)

    align._lrkv = True
    return align


def _packed_kv_cache_groups(original):
    @wraps(original)
    def groups_for(vllm_config, kv_cache_spec):
        from vllm.v1.core import kv_cache_utils
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

        from .latent_backend import LatentKVSpec

        if not _applies(vllm_config):
            return original(vllm_config, kv_cache_spec)
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_outermost or len({s.page_size_bytes for s in kv_cache_spec.values()}) <= 1:
            return original(vllm_config, kv_cache_spec)
        latent = {n: s for n, s in kv_cache_spec.items() if isinstance(s, LatentKVSpec)}
        mamba = {n: s for n, s in kv_cache_spec.items() if isinstance(s, MambaSpec)}
        latent_spec = UniformTypeKVCacheSpecs.from_specs(latent) if latent else None
        if latent_spec is None or len(latent) + len(mamba) != len(kv_cache_spec):
            return original(vllm_config, kv_cache_spec)

        groups = [KVCacheGroupSpec(list(latent), latent_spec)]
        names = list(mamba)                     # spec order is the model's layer order
        for i in range(0, len(names), len(latent)):
            chunk = {n: mamba[n] for n in names[i:i + len(latent)]}
            spec = UniformTypeKVCacheSpecs.from_specs(chunk)
            assert spec is not None
            groups.append(KVCacheGroupSpec(list(chunk), spec))
        kv_cache_utils._annotate_eagle_groups(
            vllm_config, kv_cache_spec, groups,
            use_trailing_layer_fallback=kv_cache_utils._uses_trailing_mtp_layers(vllm_config))
        kv_cache_utils._warn_if_unannotated_eagle_mamba(vllm_config, groups)
        logger.info("vllm_lrkv: KV cache groups: %d compressed layers, %d GDN groups of up to %d layers", len(latent),
                    len(groups) - 1, len(latent))
        return groups

    groups_for._lrkv = True
    return groups_for


def install() -> None:
    from vllm.platforms import interface
    from vllm.v1.core import kv_cache_utils

    if not getattr(kv_cache_utils._get_packed_kv_cache_groups, "_lrkv", False):
        kv_cache_utils._get_packed_kv_cache_groups = _packed_kv_cache_groups(kv_cache_utils._get_packed_kv_cache_groups)
    current = interface.Platform.__dict__["_align_hybrid_block_size"].__func__
    if not getattr(current, "_lrkv", False):
        interface.Platform._align_hybrid_block_size = classmethod(_align_hybrid_block_size(current))
