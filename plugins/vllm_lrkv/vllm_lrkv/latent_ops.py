"""Triton kernels around the latent decode of ``latent_backend`` (one launch each, reading and writing the layouts the
decode path uses, so no permute copies):

* ``block_diag_query``: the normed, rotated query [T, H, D] times each KV group's map [D, n_k * 64 + Hkv * rot] ->
  the block-diagonal query of ``cute_decode`` [T, H, n_k * 64 + Hkv * rot];
* ``up_project``: the merged latent output [T, H, r_v] times U_v of each head's group, stored transposed as
  [r_v, D] -> the attention output [T, H, D], written in place (a strided view such as ``output[:num_decodes]``).
  The weight tile then loads in 16-byte vectors along D whatever r_v is;
* ``latent_rows_to_cache``: writes the per-token cache rows ``[c_K / rho | 0 | c_v | 0]`` into the paged KV cache.

Rows of a group g are (token, head g * HPG + j); every program computes one [rows x columns] tile of one group.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


# per-step sizes are not specialized on, so a batch size not seen in warm-up does not compile during serving
@triton.jit(do_not_specialize=["num_rows"])
def _group_gemm(X, W, Out, num_rows, K, N, stride_xt, stride_xh, stride_wg, stride_wk, stride_wn, stride_ot,
                stride_oh, HPG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Out[t, g * HPG + j, :] = X[t, g * HPG + j, :] @ W[g] for the rows r = t * HPG + j of group g = program_id(2)."""
    g = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    t = (rows // HPG).to(tl.int64)
    h = g * HPG + rows % HPG
    mask_m = rows < num_rows
    mask_n = cols < N
    x_rows = X + t * stride_xt + h * stride_xh
    w = W + g * stride_wg
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        mask_k = ks < K
        x = tl.load(x_rows[:, None] + ks[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wt = tl.load(w + ks[:, None] * stride_wk + cols[None, :] * stride_wn, mask=mask_k[:, None] & mask_n[None, :],
                     other=0.0)
        acc = tl.dot(x, wt, acc)
    out = Out + t[:, None] * stride_ot + h[:, None] * stride_oh + cols[None, :]
    tl.store(out, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# (BM, BN, BK, num_stages, num_warps) by rows per group (6 per token), from sweeps on H800 (24 heads in 4 groups)
_QUERY_TILES = ((16, (16, 64, 64, 3, 4)), (48, (64, 64, 64, 3, 4)), (384, (64, 128, 64, 3, 4)),
                (None, (64, 128, 32, 3, 4)))
_UP_TILES = ((96, (16, 64, 128, 2, 4)), (192, (32, 64, 128, 2, 4)), (480, (32, 128, 128, 2, 4)),
             (None, (64, 128, 128, 2, 4)))


def _launch(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, stride_wk: int, stride_wn: int, K: int, N: int,
            tiles):
    T, H, _ = x.shape
    groups = w.shape[0]
    hpg = H // groups
    rows = T * hpg
    assert x.stride(-1) == 1 and out.stride(-1) == 1 and out.shape[:2] == x.shape[:2]
    bm, bn, bk, stages, warps = next(cfg for limit, cfg in tiles if limit is None or rows <= limit)
    _group_gemm[(triton.cdiv(rows, bm), triton.cdiv(N, bn), groups)](
        x, w, out, rows, K, N, x.stride(0), x.stride(1), w.stride(0), stride_wk, stride_wn, out.stride(0),
        out.stride(1), HPG=hpg, BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)


def block_diag_query(q: torch.Tensor, qmap: torch.Tensor) -> torch.Tensor:
    """q [T, H, D] (normed, rotated) and qmap [G, D, KQ] -> the decode kernel's block-diagonal query [T, H, KQ]."""
    T, H, D = q.shape
    KQ = qmap.shape[-1]
    out = torch.empty(T, H, KQ, device=q.device, dtype=q.dtype)
    _launch(q, qmap, out, qmap.stride(1), qmap.stride(2), D, KQ, _QUERY_TILES)
    return out


def up_project(a: torch.Tensor, v_up_t: torch.Tensor, out: torch.Tensor) -> None:
    """a [T, H, r_v] (merged latent output) and v_up_t [G, r_v, D] (U_v transposed) -> out [T, H, D] =
    U_v,g(h) a(h), in place."""
    R, D = v_up_t.shape[1], v_up_t.shape[2]
    assert a.shape[-1] == R and v_up_t.stride(2) == 1
    _launch(a, v_up_t, out, v_up_t.stride(1), v_up_t.stride(2), R, D, _UP_TILES)


@triton.jit(do_not_specialize=["num_tokens"])
def _latent_rows(C, Ut, KSegs, Slots, Cache, num_tokens, eps, stride_ct, stride_cb, stride_cn,
                 LK: tl.constexpr, PE: tl.constexpr, NG: tl.constexpr, PAGE: tl.constexpr,
                 BT: tl.constexpr, BK: tl.constexpr, BW: tl.constexpr):
    """Cache rows of the tokens of block program_id(0): for KV group g = program_id(1) < NG, c~_g = c_g / rho_g at the
    group's latent columns (two runs, KSegs[g] = (start, length, start, length): packed, or the RoPE coordinates
    apart from the rest); program_id(1) == NG copies columns [LK, PE) (the V part and the alignment gaps).
    Head dim 256 with 64 rotary channels."""
    D: tl.constexpr = 256
    g = tl.program_id(1)
    tok = tl.program_id(0) * BT + tl.arange(0, BT)
    live = tok < num_tokens
    slot = tl.load(Slots + tok, mask=live, other=-1).to(tl.int64)
    ok = live & (slot >= 0)                                   # padded tokens carry slot -1
    dst = Cache + (slot // PAGE) * stride_cb + (slot % PAGE) * stride_cn
    src = C + tok.to(tl.int64) * stride_ct
    if g == NG:
        for w0 in range(LK, PE, BW):
            cols = w0 + tl.arange(0, BW)
            m = ok[:, None] & (cols < PE)[None, :]
            x = tl.load(src[:, None] + cols[None, :], mask=m, other=0.0)
            tl.store(dst[:, None] + cols[None, :], x.to(Cache.dtype.element_ty), mask=m)
        return
    a_lo = tl.load(KSegs + 4 * g)
    a_n = tl.load(KSegs + 4 * g + 1)
    b_lo = tl.load(KSegs + 4 * g + 2)
    rank = a_n + tl.load(KSegs + 4 * g + 3)
    u = Ut + g * D
    c64 = tl.arange(0, 64)
    c128 = tl.arange(0, 128)
    # k_pre = U_g c_g (BF16 operands, FP32 accumulation) in channel blocks [0, 64), [64, 128), [128, 256)
    acc_r = tl.zeros([BT, 64], dtype=tl.float32)
    acc_a = tl.zeros([BT, 64], dtype=tl.float32)
    acc_b = tl.zeros([BT, 128], dtype=tl.float32)
    for kk in tl.static_range(D // BK):
        i = kk * BK + tl.arange(0, BK)                      # the group's latent coordinates and their columns
        ks = tl.where(i < a_n, a_lo + i, b_lo + i - a_n)
        mk = i < rank
        a = tl.load(src[:, None] + ks[None, :], mask=live[:, None] & mk[None, :], other=0.0)
        rows = u + ks[:, None] * (NG * D)
        acc_r = tl.dot(a, tl.load(rows + c64[None, :], mask=mk[:, None], other=0.0), acc_r)
        acc_a = tl.dot(a, tl.load(rows + 64 + c64[None, :], mask=mk[:, None], other=0.0), acc_a)
        acc_b = tl.dot(a, tl.load(rows + 128 + c128[None, :], mask=mk[:, None], other=0.0), acc_b)
    sq = tl.sum(acc_r * acc_r, 1) + tl.sum(acc_a * acc_a, 1) + tl.sum(acc_b * acc_b, 1)
    inv = tl.rsqrt(sq / D + eps)                              # 1 / rho_g per token
    for kk in tl.static_range(D // BK):
        i = kk * BK + tl.arange(0, BK)
        ks = tl.where(i < a_n, a_lo + i, b_lo + i - a_n)
        mk = i < rank
        m = ok[:, None] & mk[None, :]
        a = tl.load(src[:, None] + ks[None, :], mask=m, other=0.0).to(tl.float32) * inv[:, None]
        tl.store(dst[:, None] + ks[None, :], a.to(Cache.dtype.element_ty), mask=m)


def latent_rows_to_cache(c: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor,
                         u_cat: torch.Tensor, k_segs: torch.Tensor, eps: float) -> None:
    """Write the cache rows of new tokens: c [T, >= pe_offset] (the latent projection; its first pe_offset columns
    are [c_K | 0 | c_v | 0]) -> cache[slot] = [c_K / rho | 0 | c_v | 0] per group (RoPE keys are rebuilt in-kernel).

    u_cat [lk, Hkv * D] (BF16): U_g^T at the rows of the group's columns; k_segs [Hkv, 4] int32: every group's columns
    as two runs (start, length, start, length); cache [blocks, page, W].
    """
    T = slot_mapping.shape[0]
    if T == 0:
        return
    lk, ngd = u_cat.shape
    ng = k_segs.shape[0]
    assert k_segs.shape[1] == 4 and k_segs.is_contiguous()
    assert ngd == ng * 256, "head dim 256 (Qwen3.5 family)"
    W = cache.shape[-1]
    assert c.stride(-1) == 1 and cache.stride(-1) == 1 and u_cat.is_contiguous()
    bt, bk, warps = _ROWS_CFG
    _latent_rows[(triton.cdiv(T, bt), ng + 1)](
        c, u_cat, k_segs, slot_mapping, cache, T, eps, c.stride(0), cache.stride(0),
        cache.stride(1), LK=lk, PE=W, NG=ng, PAGE=cache.shape[1], BT=bt, BK=bk, BW=128, num_warps=warps)


_ROWS_CFG = (16, 128, 8)     # tokens per program, K block, warps (H800 sweep)

