"""Qwen3.5-family model whose compressed full-attention layers keep a latent KV cache (``latent`` exports of
``lrkv.runtime.export_latent``, backend ``vllm_lrkv.latent_backend``).

The export leaves every layer the stock checkpoint's own except the compressed attention layers, which carry
``q_gate_proj`` (the query + gate projection), ``kv_latent_proj`` (the cache row's K and V parts, alignment gaps
as zero rows), ``k_up`` (U_g per KV head), ``v_up`` (U_v) and the stock ``q_norm`` / ``k_norm`` / ``o_proj``.
``LatentAttentionLayer`` projects the query and gate (q_norm and the partial RoPE in vLLM's fused Qwen3.5 kernel)
and the latent c = kv_latent_proj(x), and hands the query, c and the positions to ``LatentAttention`` (vLLM's
``Attention`` with the layer's own cache spec and the unified attention op). The op first turns c into the row the
cache keeps, in one kernel (``latent_ops.latent_rows_to_cache``):

    rho_g = rms(U_g c_g);  row = [c_K / rho | 0 | c_v | 0]

The RoPE keys are rebuilt on the fly from c_K / rho and the text positions inside the decode kernel and prefill
decompression, and the impl returns the per-head attention output [T, H, head_dim], gated and projected as in
the stock layer. A group's latent coordinates sit at the columns the layout's ``k_segments`` lists (packed, or the
RoPE coordinates of every group chunk-aligned at the front). Single GPU (TP1, data parallel across GPUs).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import unified_attention_with_output
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models import qwen3_5 as _q35
from vllm.model_executor.models.qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5ForCausalLM,
                                                Qwen3_5ForConditionalGeneration, Qwen3_5RMSNorm,
                                                mrope_positions_factor)
from vllm.model_executor.models.qwen3_next import Qwen3NextMLP, Qwen3NextRMSNorm, Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import _encode_layer_name

from ..cute_decode import CHUNK, RopeRebuild, rope_blocks

FORMATS = ("latent",)


def latent_layers(config) -> dict:
    block = getattr(config, "kv_lowrank", None)
    if not isinstance(block, dict) or block.get("format") not in FORMATS:
        return {}
    return {int(i): info for i, info in block["layers"].items()}


class _Weight(nn.Module):
    """A bare tensor parameter loadable by name (``<prefix>.weight``)."""

    def __init__(self, shape, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class LatentAttention(Attention):
    """vLLM's attention layer (forward-context registration, KV cache binding, the unified attention op) over one
    latent row per token: the KV cache spec is the layer's own row width; the op receives the latent projection as
    its "key" and the positions as its "value", writes the rows and attends (no separate cache-update op: the impl
    writes the rows on a second stream while it maps the decode query). ``latent`` (a ``LatentLayer``) is attached
    by the owning module after weight loading."""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int, row_width: int,
                 cache_config, prefix: str):
        from vllm_lrkv.latent_backend import LatentAttentionBackend
        super().__init__(num_heads, head_dim, scale, num_kv_heads=num_kv_heads, cache_config=cache_config,
                         quant_config=None, prefix=prefix, attn_backend=LatentAttentionBackend)
        self.row_width = row_width
        self.latent = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from vllm_lrkv.latent_backend import LatentKVSpec
        return LatentKVSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=1, head_size=self.row_width,
                            dtype=self.kv_cache_torch_dtype)

    def forward(self, query: torch.Tensor, latent: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """query [T, H, head_dim] (normed, rotated), latent [T, pe_offset] (kv_latent_proj), positions [T] or
        [3, T] -> attention output [T, H * head_dim]."""
        T = query.shape[0]
        output = torch.empty(T, self.num_heads, self.head_size, dtype=query.dtype, device=query.device)
        if self.use_direct_call:
            unified_attention_with_output(query, latent, positions, output, self.layer_name)
        else:
            torch.ops.vllm.unified_attention_with_output(query, latent, positions, output,
                                                         _encode_layer_name(self.layer_name))
        return output.view(T, -1)


class LatentAttentionLayer(nn.Module):
    """One compressed Qwen3.5 attention layer over the latent cache (query + gate, latent row, gated output)."""

    def __init__(self, config, info: dict, vllm_config: VllmConfig, prefix: str, reduce_results: bool = True):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("vllm_lrkv latent backend runs one model copy per GPU (TP1 + DP)")
        cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.rotary_dim = int(config.kv_lowrank["rotary_dim"])
        self.k_head_ranks = [int(r) for r in info["k_head_ranks"]]
        # every group's latent columns as [start, length] runs in latent order (one run when packed; two when the
        # RoPE coordinates are chunk-aligned at the front of the K part)
        packed = [[[sum(self.k_head_ranks[:g]), r]] for g, r in enumerate(self.k_head_ranks)]
        self.k_segments = [[(int(a), int(n)) for a, n in runs] for runs in info.get("k_segments") or packed]
        assert all(len(runs) <= 2 and sum(n for _, n in runs) == r
                   for runs, r in zip(self.k_segments, self.k_head_ranks)), self.k_segments
        self.lk = sum(self.k_head_ranks)
        self.v_rank = int(info["v_rank"])
        self.v_offset = int(info["v_offset"])
        self.pe_offset = int(info["pe_offset"])
        self.row_width = int(info["row_width"])
        assert self.row_width == self.pe_offset
        self.rope_groups = tuple((runs[0][0], int(rr)) for runs, rr in zip(self.k_segments, info["rope_ranks"]))
        assert all(runs[0][1] >= rr for runs, (_, rr) in zip(self.k_segments, self.rope_groups))
        mm = vllm_config.model_config.multimodal_config
        if mm is not None and not mm.language_model_only:
            raise NotImplementedError("latent exports rebuild the RoPE keys from text positions (token t at "
                                      "position t): serve with --language-model-only")
        self.eps = float(config.rms_norm_eps)
        dtype = vllm_config.model_config.dtype

        self.q_gate_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * 2 * self.head_dim,
                                                bias=getattr(config, "qkv_bias", False), quant_config=quant_config,
                                                prefix=f"{prefix}.q_gate_proj")
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False,
                                        reduce_results=reduce_results, quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.kv_latent_proj = ReplicatedLinear(self.hidden_size, self.pe_offset, bias=False, quant_config=None,
                                               prefix=f"{prefix}.kv_latent_proj")
        self.k_up = _Weight((self.num_kv_heads, self.head_dim, max(self.k_head_ranks)), dtype)
        self.v_up = _Weight((self.num_kv_heads * self.head_dim, self.v_rank), dtype)
        self.rotary_emb = get_rope(head_size=self.head_dim, max_position=config.max_position_embeddings,
                                   rope_parameters=config.rope_parameters, dual_chunk_attention_config=None,
                                   mrope_positions_factor=mrope_positions_factor(vllm_config))
        self.mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        if not (getattr(self.rotary_emb, "is_neox_style", False) and self.rotary_emb.rotary_dim == self.rotary_dim
                and (self.mrope_section is None or (type(self.rotary_emb) is MRotaryEmbedding
                                                    and getattr(self.rotary_emb, "mrope_interleaved", False)))):
            raise NotImplementedError("vllm_lrkv latent layers: partial NeoX RoPE with plain or interleaved MRoPE only")
        self.attn = LatentAttention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads,
                                    self.row_width, cache_config, prefix=f"{prefix}.attn")
        # filled by derive_latent_tensors() once the weights are loaded
        self.register_buffer("u_cat", torch.zeros(self.lk, self.num_kv_heads * self.head_dim, dtype=torch.bfloat16),
                             persistent=False)
        self.register_buffer("k_segs", torch.tensor([[x for run in runs + [(0, 0)] * (2 - len(runs)) for x in run]
                                                     for runs in self.k_segments], dtype=torch.int32),
                             persistent=False)
        self.register_buffer("rope_gain", torch.zeros(self.rotary_dim, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def derive_latent_tensors(self) -> None:
        """Absorbed-query map, the block-diagonal U^T of the cache rows, and the RoPE-key rebuild blocks from U_g
        and k_norm."""
        from vllm_lrkv.latent_backend import LatentLayer
        dev, dtype = self.k_up.weight.device, self.k_up.weight.dtype
        gain = 1.0 + self.k_norm.weight.float()          # zero-centered RMSNorm of the Qwen3.5 family
        U = self.k_up.weight.float()
        rot, D, G = self.rotary_dim, self.head_dim, self.num_kv_heads
        f32 = torch.float32
        n_k = math.ceil(self.v_offset / CHUNK)
        qmap = torch.zeros(G, D, n_k * CHUNK + G * rot, device=dev, dtype=f32)
        u_cat = torch.zeros(self.lk, G * D, device=dev, dtype=f32)
        for g, (runs, r) in enumerate(zip(self.k_segments, self.k_head_ranks)):
            cols = torch.cat([torch.arange(a, a + n, device=dev) for a, n in runs])
            qmap[g, :rot, n_k * CHUNK + g * rot:n_k * CHUNK + (g + 1) * rot] = torch.eye(rot, device=dev)
            qmap[g, rot:, cols] = gain[rot:, None] * U[g, rot:, :r]
            u_cat[cols, g * D:(g + 1) * D] = U[g, :, :r].T
        self.u_cat.copy_(u_cat)                          # exact: U is stored in BF16
        self.rope_gain.copy_(gain[:rot])

        # The rotary rows of U_g read the group's first rr_g latent coordinates only (the export's change of
        # basis), at columns off .. off + rr_g; times the k_norm gain they are the decode kernel's resident blocks,
        # and by cache column the prefill decompression's rows (BF16).
        u_rot = [gain[:rot, None] * U[g, :rot, :rr] for g, (off, rr) in enumerate(self.rope_groups)]
        cos_sin = self.rotary_emb.cos_sin_cache.to(torch.bfloat16).contiguous()
        blocks = rope_blocks(u_rot, self.rope_groups)    # fp32
        rope = RopeRebuild(groups=self.rope_groups, blocks=blocks.to(dtype), cos_sin=cos_sin)
        rope_up = torch.zeros(G, rot, n_k * CHUNK, device=dev, dtype=dtype)
        for g, ((off, rr), u) in enumerate(zip(self.rope_groups, u_rot)):
            rope_up[g, :, off:off + rr] = u.to(dtype)

        # U_v transposed to [G, r_v, D] in place of the loaded [G * D, r_v] (rows of D: aligned vector loads)
        if self.v_up.weight.dim() == 2:
            w = self.v_up.weight.data.view(G, D, self.v_rank).transpose(1, 2)
            self.v_up.weight.data = w.contiguous()
        self.attn.latent = LatentLayer(num_kv_heads=G, head_dim=D, rotary_dim=rot, v_rank=self.v_rank,
                                       v_offset=self.v_offset, pe_offset=self.pe_offset, n_k=n_k,
                                       qmap=qmap.to(dtype), v_up=self.v_up.weight,
                                       u_cat=self.u_cat, k_segs=self.k_segs, rope_gain=self.rope_gain,
                                       eps=self.eps, rotary=self.rotary_emb, mrope_section=self.mrope_section,
                                       rope=rope, rope_up=rope_up)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        qg, _ = self.q_gate_proj(hidden_states)
        if positions.ndim == 2 and self.mrope_section is None:
            positions = positions[0]
        q, _, gate = fused_qk_rmsnorm_rope_gate(
            qg, qg[:, :0], self.q_norm.weight, self.q_norm.weight, self.rotary_emb.cos_sin_cache, positions,
            self.q_norm.variance_epsilon, self.num_heads, 0, self.head_dim, self.rotary_dim, norm_beta=1.0,
            mrope_section=self.mrope_section if positions.ndim == 2 else None)
        c, _ = self.kv_latent_proj(hidden_states)                                    # [T, pe_offset]
        attn_out = self.attn(q.view(T, self.num_heads, self.head_dim), c, positions)
        attn_out = attn_out * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_out)
        return output


class LatentDecoderLayer(Qwen3_5DecoderLayer):
    """Qwen3.5 decoder layer whose full attention is ``LatentAttentionLayer`` when the layer is compressed."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        layer_idx = extract_layer_index(prefix)
        info = latent_layers(config).get(layer_idx)
        if layer_type != "full_attention" or info is None:
            super().__init__(vllm_config, layer_type, prefix)
            return
        nn.Module.__init__(self)
        parallel_config, quant_config = vllm_config.parallel_config, vllm_config.quant_config
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
        self.use_attn_reduce_scatter_for_moe = (parallel_config.use_sequence_parallel_moe
                                                and parallel_config.pipeline_parallel_size == 1 and is_moe_layer)
        self.self_attn = LatentAttentionLayer(config, info, vllm_config, prefix=f"{prefix}.self_attn",
                                              reduce_results=not self.use_attn_reduce_scatter_for_moe)
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                                    hidden_act=config.hidden_act, quant_config=quant_config, prefix=f"{prefix}.mlp")
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))


@contextlib.contextmanager
def _patched_decoder_layer():
    original = _q35.Qwen3_5DecoderLayer
    _q35.Qwen3_5DecoderLayer = LatentDecoderLayer
    try:
        yield
    finally:
        _q35.Qwen3_5DecoderLayer = original


def _derive_all(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LatentAttentionLayer):
            module.derive_latent_tensors()


class Qwen3_5LowRankLatentForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The stock model built with ``LatentDecoderLayer`` (``Qwen3_5Model`` resolves the layer class by name)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        with _patched_decoder_layer():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _derive_all(self)
        return loaded


class Qwen3_5LowRankLatentForCausalLM(Qwen3_5ForCausalLM):
    """Text-only variant (plain ``model.layers.*`` layout), used by the tiny end-to-end tests."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        with _patched_decoder_layer():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        _derive_all(self)
        return loaded

