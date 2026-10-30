"""``LRKV_LATENT``: the plugin's attention backend over the latent KV cache of ``latent`` exports
(``lrkv.runtime.export_latent``).

Every compressed layer keeps one row per token in its own page (``LatentKVSpec``: one head slot of ``row_width``
elements, BF16, no separate V; the widths differ per layer, so the cache is block-outermost and
``vllm_lrkv.kv_cache`` packs the compressed layers into one group). The model hands vLLM's attention op the normed,
rotated query of every head ([T, H, head_dim]), the latent projection of every new token as the "key" and the
positions as the "value". The op first writes the tokens' cache rows (rms-folded K latents and V latent in the
export's aligned layout; the decode kernel and prefill decompression rebuild the RoPE keys on the fly) to their
slots in one kernel (``latent_ops.latent_rows_to_cache``); with ``SIDE_STREAM_MIN_DECODES`` decode tokens or more
it does so on a second stream while the decode query is mapped, the two being independent and both small (a CUDA
graph keeps the fork and join, all inside the op). Then the impl splits the reordered batch:

* decode (one query token per request, at the front): the block-diagonal query (``latent_ops``), the CuTe
  single-pass kernel over the paged rows on the step's device schedule (``cute_decode.schedule``, computed once per
  step by the builder and shared by all layers), the merge of each request's segments, then the per-head V
  up-projection written straight into the output;
* prefill (everything else, chunked prefill and prefix-cache hits included): the rows of the requests' tokens are
  gathered through the block table, the model's own keys and values are rebuilt per KV head ([RoPE key | gained
  position-free channels] and U_v c_v) and FlashAttention varlen runs over them; the new tokens are already in the
  cache when the op runs. While the step's sequences hold at most ``PREFILL_CHUNK`` tokens together, one causal pass
  covers them whole; beyond that the new tokens attend to themselves, then to the cached contexts in chunks of
  ``PREFILL_CHUNK`` tokens, and the partial results are merged by log-sum-exp, so the temporary memory of a prefill
  stays bounded whatever the context length.

Decode batches run in FULL CUDA graphs (UNIFORM_SINGLE_TOKEN_DECODE); prefill runs eagerly between graph pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import ClassVar

import torch

from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadataBuilder,
                                       AttentionType, CommonAttentionMetadata, MultipleOf)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import FullAttentionSpec

from . import latent_ops
from .cute_decode import CHUNK, DecodeSchedule, RopeRebuild, latent_decode_cute, make_schedule, merge_segments, schedule

BACKEND_NAME = "LRKV_LATENT"
# Tokens a prefill step decompresses at once (all its prefill requests together); the temporary memory is about
# 11 KB per token (gathered rows, K, V, RoPE temporaries). 0: one pass over the whole sequences at any length.
PREFILL_CHUNK = int(os.environ.get("LRKV_PREFILL_CHUNK_TOKENS", 32768))
# Decode tokens from which the cache rows are written on a second stream.
SIDE_STREAM_MIN_DECODES = int(os.environ.get("LRKV_SIDE_STREAM_MIN_DECODES", 32))


@dataclass(frozen=True, kw_only=True)
class LatentKVSpec(FullAttentionSpec):
    """One compressed layer's cache: a single head slot of ``head_size`` (the row width) elements per token and no
    separate V, so a page is block_size x row width x dtype bytes. The KV cache manager treats it as full attention
    (prefix caching, FullAttentionManager)."""

    head_size_v: int = 0


@dataclass
class LatentLayer:
    """Constants of one compressed layer, attached to its attention layer by the model after weight loading."""

    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    v_rank: int
    v_offset: int
    pe_offset: int
    n_k: int                    # 64-column chunks of the K part the decode kernel reads
    qmap: torch.Tensor          # [Hkv, head_dim, n_k * 64 + Hkv * rot]: query -> block-diagonal decode query
    v_up: torch.Tensor          # [Hkv, r_v, head_dim]: U_v transposed (rows of head_dim, latent_ops.up_project)
    u_cat: torch.Tensor         # [lk, Hkv * head_dim] BF16: U_g^T at the group's columns (rho, the RoPE key input)
    k_segs: torch.Tensor        # [Hkv, 4] int32: every group's latent columns as two runs (start, length) x 2
    rope_gain: torch.Tensor     # [rot] fp32: k_norm gain of the rotary channels
    eps: float
    rotary: torch.nn.Module     # the layer's rotary embedding (its cos/sin cache)
    mrope_section: tuple | None
    rope: RopeRebuild           # RoPE keys rebuilt in-kernel from the K latent
    rope_up: torch.Tensor       # [Hkv, rot, n_k * 64] gain * rotary rows of U_g by column

    def block_diag_query(self, q: torch.Tensor) -> torch.Tensor:
        """[T, H, head_dim] normed rotated query -> [T, H, n_k * 64 + Hkv * rot] (cute_decode.block_diag_query)."""
        return latent_ops.block_diag_query(q, self.qmap)

    def up_project(self, a: torch.Tensor, out: torch.Tensor) -> None:
        """[T, H, r_v] latent-space attention output -> out [T, H, head_dim] (U_v of the head's KV group)."""
        latent_ops.up_project(a, self.v_up, out)

    def decompress(self, rows: torch.Tensor, positions: torch.Tensor):
        """Cached rows [S, row_width] at text positions [S] -> the model's keys and values [S, Hkv, head_dim] (RoPE
        channels first)."""
        rot = self.rotary_dim
        rows = rows.to(self.qmap.dtype)
        kc = min(self.n_k * CHUNK, rows.shape[1])     # the K chunks' columns (zero weights past the K part)
        # Same math as the decode kernel: gain * B_g c~_g rounded to the cache dtype, NeoX rotation by position
        x = torch.einsum("sk,gdk->sgd", rows[:, :kc], self.rope_up[:, :, :kc]).float()
        cos, sin = self.rope.cos_sin[positions].float().unsqueeze(1).chunk(2, dim=-1)
        x1, x2 = x.chunk(2, dim=-1)
        k_pe = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(rows.dtype)
        k_nope = torch.einsum("sk,gdk->sgd", rows[:, :kc], self.qmap[:, rot:, :kc])
        v = torch.einsum("sr,grd->sgd", rows[:, self.v_offset:self.v_offset + self.v_rank], self.v_up)
        return torch.cat((k_pe, k_nope), dim=-1), v


@dataclass
class LatentAttentionMetadata:
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    block_table: torch.Tensor           # [num_reqs, max_blocks] int32
    seq_lens: torch.Tensor              # [num_reqs] int32, including this step's tokens
    sched: DecodeSchedule | None        # the decode requests' work split, filled for this step
    # prefill requests, relative to the first prefill token
    prefill_cu_q: torch.Tensor | None = None       # [num_prefills + 1] int32
    prefill_cu_k: torch.Tensor | None = None       # [num_prefills + 1] int32, one pass: the whole sequences
    prefill_max_q: int = 0
    prefill_max_k: int = 0
    # the rows to read, one pass: every token of every request; chunked: the new tokens, then the chunks
    prefill_kv_block: torch.Tensor | None = None   # cache block of every row
    prefill_kv_offset: torch.Tensor | None = None  # its slot inside the block
    prefill_kv_pos: torch.Tensor | None = None     # its position (index in the request)
    # chunked: (first row, end row, cu_k [num_prefills + 1] int32, max_k) of every chunk of the cached contexts
    prefill_chunks: list | None = None


_LAST_BUILD: dict = {}
_SCHEDULES: dict = {}
_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """The device's second stream, for the cache rows of a step with decode tokens (created at the first such step,
    a warm-up one, before any CUDA graph capture)."""
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device)
    return stream


class LatentAttentionMetadataBuilder(AttentionMetadataBuilder[LatentAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        # persistent: FULL CUDA graphs read it by address; a step has at most max_num_batched_tokens requests
        if device not in _SCHEDULES:
            _SCHEDULES[device] = make_schedule(vllm_config.scheduler_config.max_num_batched_tokens, device)
        self.sched = _SCHEDULES[device]

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> LatentAttentionMetadata:
        cm = common_attn_metadata
        last = _LAST_BUILD.get(self.device)
        if last is not None and last[0] is cm and last[1] == self.block_size:
            return last[2]
        m = self._build(cm)
        _LAST_BUILD[self.device] = (cm, self.block_size, m)
        return m

    def _build(self, cm: CommonAttentionMetadata) -> LatentAttentionMetadata:
        nd, npf, ndt, npt = split_decodes_and_prefills(cm, decode_threshold=1)
        m = LatentAttentionMetadata(num_actual_tokens=cm.num_actual_tokens, num_decodes=nd, num_decode_tokens=ndt,
                                    num_prefills=npf, num_prefill_tokens=npt, block_table=cm.block_table_tensor,
                                    seq_lens=cm.seq_lens, sched=self.sched if nd else None)
        if nd:
            schedule(self.sched, cm.seq_lens[:nd])
        if npf == 0:
            return m
        dev = self.device
        qsl = cm.query_start_loc_cpu[nd:nd + npf + 1].to(torch.int64)
        if cm.seq_lens_cpu_upper_bound is not None:           # exact on prefill rows
            seq_cpu = cm.seq_lens_cpu_upper_bound[nd:nd + npf].to(torch.int64)
        else:
            seq_cpu = cm.seq_lens[nd:nd + npf].to("cpu", torch.int64)
        q_len = qsl[1:] - qsl[:-1]
        m.prefill_cu_q = (qsl - qsl[0]).to(dev, torch.int32, non_blocking=True)
        m.prefill_max_q = int(q_len.max())
        req_of = torch.arange(npf)
        if PREFILL_CHUNK <= 0 or int(seq_cpu.sum()) <= PREFILL_CHUNK:
            # one causal pass over every token of every request
            starts, lens = torch.zeros(npf, dtype=torch.int64), seq_cpu
            cu_k = torch.zeros(npf + 1, dtype=torch.int64)
            torch.cumsum(seq_cpu, 0, out=cu_k[1:])
            m.prefill_cu_k = cu_k.to(dev, torch.int32, non_blocking=True)
            m.prefill_max_k = int(seq_cpu.max())
        else:
            # the new tokens of every request, then the cached contexts back to back cut every PREFILL_CHUNK tokens:
            # chunk j holds the part of each context that falls in [j, j + 1) * PREFILL_CHUNK of that sequence
            ctx = seq_cpu - q_len
            ctx_start = torch.cumsum(ctx, 0) - ctx
            n_chunks = -(-int(ctx.sum()) // PREFILL_CHUNK)
            lo = torch.arange(n_chunks, dtype=torch.int64)[:, None] * PREFILL_CHUNK
            a = (lo - ctx_start).clamp(min=0).minimum(ctx)
            clen = (lo + PREFILL_CHUNK - ctx_start).clamp(min=0).minimum(ctx) - a       # [n_chunks, npf]
            starts, lens = torch.cat([ctx, a.flatten()]), torch.cat([q_len, clen.flatten()])
            req_of = req_of.repeat(n_chunks + 1)
            cu = torch.zeros(n_chunks, npf + 1, dtype=torch.int64)
            torch.cumsum(clen, 1, out=cu[:, 1:])
            cu_dev = cu.to(dev, torch.int32, non_blocking=True)
            ends = int(qsl[-1] - qsl[0]) + torch.cumsum(cu[:, -1], 0)
            m.prefill_chunks = [(int(e - cu[j, -1]), int(e), cu_dev[j], int(clen[j].max()))
                                for j, e in enumerate(ends)]
        # (block, offset, position) of every row: lens[i] consecutive positions from starts[i] of request req_of[i]
        cu_rows = torch.zeros(len(lens) + 1, dtype=torch.int64)
        torch.cumsum(lens, 0, out=cu_rows[1:])
        total = int(cu_rows[-1])
        seg = torch.repeat_interleave(torch.arange(len(lens), device=dev), lens.to(dev, non_blocking=True),
                                      output_size=total)
        pos = (starts.to(dev, non_blocking=True)[seg] + torch.arange(total, device=dev)
               - cu_rows.to(dev, non_blocking=True)[seg])
        req = req_of.to(dev, non_blocking=True)[seg]
        m.prefill_kv_block = cm.block_table_tensor[nd:nd + npf][req, pos // self.block_size].to(torch.int64)
        m.prefill_kv_offset = pos % self.block_size
        m.prefill_kv_pos = pos
        return m


class LatentAttentionImpl(AttentionImpl[LatentAttentionMetadata]):
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int | None = None,
                 alibi_slopes: list[float] | None = None, sliding_window: int | None = None,
                 kv_cache_dtype: str = "auto", logits_soft_cap: float | None = None,
                 attn_type: str = AttentionType.DECODER, kv_sharing_target_layer_name: str | None = None,
                 **kwargs) -> None:
        if (attn_type != AttentionType.DECODER or alibi_slopes is not None or sliding_window is not None
                or logits_soft_cap or kv_sharing_target_layer_name is not None):
            raise NotImplementedError("LRKV_LATENT: causal decoder self-attention only (no alibi, sliding window, "
                                      "soft cap or KV sharing)")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(f"LRKV_LATENT keeps a BF16 latent cache, not {kv_cache_dtype}")
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.fa_version = get_flash_attn_version(head_size=head_size, head_size_v=head_size)

    def do_kv_cache_update(self, layer, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                           slot_mapping: torch.Tensor) -> None:
        """key: the latent projection [T, pe_offset] of the new tokens, value: their positions."""
        lat: LatentLayer = layer.latent
        latent_ops.latent_rows_to_cache(key, slot_mapping, kv_cache[:, 0], lat.u_cat, lat.k_segs, lat.eps)

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: LatentAttentionMetadata, output: torch.Tensor, output_scale=None,
                output_block_scale=None) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("LRKV_LATENT: fused output quantization")
        if attn_metadata is None:                 # profiling run
            return output.fill_(0)
        m = attn_metadata
        lat: LatentLayer = layer.latent
        cache = kv_cache[:, 0]                    # [blocks, block_size, row_width], strided per block
        nd = m.num_decode_tokens
        slots = get_attention_context(layer.layer_name)[3]
        side = None
        if slots is not None:                     # the new tokens' rows (a dummy run has no slots)
            if m.num_decodes >= SIDE_STREAM_MIN_DECODES:
                main, side = torch.cuda.current_stream(), _side_stream(query.device)
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    self.do_kv_cache_update(layer, key, value, kv_cache, slots)
            else:
                self.do_kv_cache_update(layer, key, value, kv_cache, slots)
        if m.num_decodes:
            dev = query.device
            q_bd = lat.block_diag_query(query[:nd])
            if side is not None:
                main.wait_stream(side)            # the step's own rows are part of the context
            part = torch.empty(nd + m.sched.num_ctas, self.num_heads, lat.v_rank + 1, dtype=torch.float32,
                               device=dev)                              # one partial per segment
            a = torch.empty(nd, self.num_heads, lat.v_rank, dtype=query.dtype, device=dev)
            lse = torch.empty(nd, self.num_heads, dtype=torch.float32, device=dev)
            latent_decode_cute(q_bd, cache, m.block_table[:nd], m.seq_lens[:nd], m.sched, part, self.scale,
                               lat.v_offset, lat.v_rank, rope=lat.rope)
            merge_segments(part, m.sched, a, lse)
            lat.up_project(a, output[:nd])
        if m.num_prefills:
            npt = m.num_prefill_tokens
            q, out = query[nd:nd + npt], output[nd:nd + npt]
            if m.prefill_chunks is None:
                k, v = lat.decompress(cache[m.prefill_kv_block, m.prefill_kv_offset], m.prefill_kv_pos)
                flash_attn_varlen_func(q=q, k=k, v=v, max_seqlen_q=m.prefill_max_q, cu_seqlens_q=m.prefill_cu_q,
                                       max_seqlen_k=m.prefill_max_k, cu_seqlens_k=m.prefill_cu_k,
                                       softmax_scale=self.scale, causal=True, out=out, fa_version=self.fa_version)
            else:
                self._chunked_prefill(lat, q, out, cache, m)
        return output

    def _chunked_prefill(self, lat: LatentLayer, q: torch.Tensor, out: torch.Tensor, cache: torch.Tensor,
                         m: LatentAttentionMetadata) -> None:
        """The new tokens over themselves (causal), then over each chunk of the cached contexts (all of it precedes
        them), folded into ``out`` by log-sum-exp."""
        blk, off, pos = m.prefill_kv_block, m.prefill_kv_offset, m.prefill_kv_pos
        npt = q.shape[0]
        k, v = lat.decompress(cache[blk[:npt], off[:npt]], pos[:npt])
        _, lse = flash_attn_varlen_func(q=q, k=k, v=v, max_seqlen_q=m.prefill_max_q, cu_seqlens_q=m.prefill_cu_q,
                                        max_seqlen_k=m.prefill_max_q, cu_seqlens_k=m.prefill_cu_q,
                                        softmax_scale=self.scale, causal=True, out=out, return_softmax_lse=True,
                                        fa_version=self.fa_version)
        del k, v
        part, lse_next = torch.empty_like(out), torch.empty_like(lse)
        for i, (r0, r1, cu_k, max_k) in enumerate(m.prefill_chunks):
            k, v = lat.decompress(cache[blk[r0:r1], off[r0:r1]], pos[r0:r1])
            _, part_lse = flash_attn_varlen_func(q=q, k=k, v=v, max_seqlen_q=m.prefill_max_q,
                                                 cu_seqlens_q=m.prefill_cu_q, max_seqlen_k=max_k, cu_seqlens_k=cu_k,
                                                 softmax_scale=self.scale, causal=False, out=part,
                                                 return_softmax_lse=True, fa_version=self.fa_version)
            del k, v
            last = i == len(m.prefill_chunks) - 1
            merge_attn_states(out, part, part_lse, out, lse, None if last else lse_next)
            lse, lse_next = lse_next, lse


class LatentAttentionBackend(AttentionBackend):
    # the op writes the rows itself (LatentAttention calls no separate update); False keeps vLLM padding the slot
    # mapping to the token count of the key, as the row-writing kernel expects
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[LatentAttentionImpl]:
        return LatentAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[LatentAttentionMetadataBuilder]:
        return LatentAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]                   # the decode kernel's 64-token TMA tiles never cross a page

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Block-outermost only: the layers' row widths differ, so their pages share a block side by side."""
        from vllm.v1.kv_cache_layout import KVCacheLayout
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

