"""Single-pass latent decode on Hopper in CuTe DSL with in-kernel rebuilt RoPE keys.

Cache row of a compressed layer, two parts that each start 16-byte aligned (TMA cannot start a box at a feature
offset that is not a multiple of 8 BF16 elements; the hardware raises an illegal instruction):

    [c~_1 | ... | c~_G | pad]  [c_v | pad]
     0                lk_pad    lk_pad   pe_off

The K segments inside the first part stay packed (or RoPE-aligned): the score GEMM reads the whole part and
the query is block diagonal, i.e. head h of group g carries its absorbed query q_lat(h) at the columns of c~_g and
zeros elsewhere, and its rotated query at the group's RoPE chunk. The RoPE keys are rebuilt from the K part on the
fly: k_pe,g(t) = RoPE_t(gain * B_g c~_g(t)), where the export turned every group's latent basis so that B_g (the
rotary rows of U_g) reads only the first rr_g = min(rot, r_g) coordinates of c~_g. Those coordinates span one or two
64-column chunks, so per tile and group one more wgmma [64 tokens x 64] (the cache chunks already in shared memory
times the group's resident [64 x 64] blocks of gain * B_g) gives the unrotated key; it is rounded to BF16 (a stored
key's precision) and rotated in registers, and the result is the register A operand of the group's RoPE score wgmma
(the accumulator layout is the operand layout, as P in FlashAttention 3). The cos / sin rows of the tile's 64
positions (the model's cos_sin_cache; a request's token t sits at position t, text only) come through the ring as
one more [64 x 64] chunk ahead of the K part, read into registers once per tile.

Work is split by a schedule computed on the device once per step (``schedule``; all layers share it): every
request's context is cut into 64-token tiles and the tiles of the whole batch, in request order, are dealt out in
equal runs to a fixed number of CTAs (one per SM). A run that crosses request boundaries becomes several segments
(request, tile range), so no CTA is empty while there is work, short and long requests mix, and the grid does not
depend on the lengths (FULL CUDA graphs). Each CTA walks its segments; warps 0-3 form the MMA warpgroup, warp 4 issues
TMA. Per segment the block-diagonal query of the request comes in through a one-stage pipeline (released after the
segment's last score GEMM, so the next segment's query loads while the values finish); the cache is streamed in
[64 tokens x 64 features] chunks (128-byte swizzled rows) through a ring of shared-memory stages; per 64-token tile:

1. scores S[tokens, heads] += chunk @ q_bd_chunk^T over the K-part chunks and rebuilt RoPE keys (wgmma, M = 64 tokens,
   N = 24 heads, K-major A);
2. masked online softmax per head: the column max is reduced over lanes and the four warps, the running sum stays
   per thread until the end;
3. P^T goes to shared memory and O^T[features, heads] += V_chunk^T @ P over the V chunks: the same swizzled chunk
   read as an MN-major A operand (the K-major and MN-major SW128 layouts of a 64 x 64 BF16 tile are the same bytes).

Each segment writes its normalized partial output and log-sum-exp ([segments, H, lv + 1]); ``merge_segments`` (Triton)
combines the segments of every request. Constraints: page size a multiple of 64, 24 query heads in 4 KV groups.

Compiled kernels persist across processes. ``cute.compile`` never consults CuTe DSL's own file cache (compile-only
calls turn it off), so each layout's kernel is exported once as an object file (``export_to_c``) into
``$LRKV_CUTE_CACHE_DIR`` (default ``$VLLM_CACHE_ROOT/lrkv_cute``; empty disables the cache), named by a hash of this
file, the DSL and quack versions, the GPU capability and the compile key, and later processes load it
(``cute.runtime.load_module``: milliseconds instead of about 3 s per layout).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
from importlib.metadata import version
import logging
import math
import os
import shutil
import tempfile
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator
import cutlass.pipeline as pipeline
from cutlass.tensor_utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils
from quack.layout_utils import reshape_acc_to_frgA, reshape_acc_to_mn
import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(f"vllm.{__name__}")     # under "vllm" so vLLM's handler prints it

H, G, ROT = 24, 4, 64
TILE_T = 64              # tokens per tile (wgmma M of the score GEMM, K of the value GEMM)
CHUNK = 64               # features per TMA box: one 128-byte swizzled row
K_BLOCKS = CHUNK // 16   # wgmma K steps per chunk (BF16)
MMA_WARPS = 4
NUM_THREADS = 32 * (MMA_WARPS + 1)
STAGES = 12              # cache ring: 12 x 8 KB; with the query (up to 52 KB) one CTA per SM
SMEM_BYTES = 227 * 1024  # opt-in shared memory per block (H800 / H100)
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
EXPERIMENT: frozenset = frozenset()


class LatentDecodeSm90:
    def __init__(self, lk_pad: int, lv: int, page: int, rope_groups: tuple, stages: int = STAGES, inflight: int = 1):
        """BF16 rows with in-kernel rebuilt RoPE keys. ``rope_groups``: ((k_off_g, rr_g) per group) listing the
        first rr_g latent coordinates of every group from which its RoPE keys are rebuilt."""
        assert lk_pad % 8 == 0 and page % TILE_T == 0, (lk_pad, page)
        assert rope_groups is not None and len(rope_groups) == G
        self.mt = BFloat16
        self.num_threads = NUM_THREADS
        self.lk_pad, self.lv, self.page = lk_pad, lv, page
        self.n_k = math.ceil(lk_pad / CHUNK)
        self.n_v = math.ceil(lv / CHUNK)
        self.n_q = self.n_k + G                     # query chunks: the K part, then one RoPE chunk per group
        # resident [64 x 64] blocks of every B_g: one per cache chunk its rr_g coordinates touch (one or two);
        # rope_at[c] lists the groups whose coordinates end in chunk c: (group, first chunk, first block)
        self.rope_blocks, self.rope_at = [], [[] for _ in range(self.n_k)]
        for g, (off, rr) in enumerate(rope_groups):
            assert 0 < rr <= ROT and off + rr <= lk_pad, (off, rr)
            c0, c1 = off // CHUNK, (off + rr - 1) // CHUNK
            self.rope_at[c1].append((g, c0, len(self.rope_blocks)))
            self.rope_blocks += [(g, c) for c in range(c0, c1 + 1)]
        self.n_rb = len(self.rope_blocks)
        # feature offset of every cache chunk of a tile, in consumption order: scores (K part), values; the
        # tile's cos / sin chunk comes first
        self.chunk_feat = [CHUNK * c for c in range(self.n_k)] + [lk_pad + CHUNK * v for v in range(self.n_v)]
        fixed = 2 * (self.n_q * H * CHUNK + self.n_rb * ROT * CHUNK + H * CHUNK) + 4 * 1024
        stages = min(stages, (SMEM_BYTES - fixed) // (2 * TILE_T * CHUNK))
        assert 0 <= inflight < stages
        self.stages = stages
        self.inflight = inflight                    # wgmma chunk groups left in flight before a stage is released

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mC: cute.Tensor, mBT: cute.Tensor, mSeq: cute.Tensor, mWork: cute.Tensor,
                 mCtaSeg: cute.Tensor, mPart: cute.Tensor, mR: cute.Tensor, mCS: cute.Tensor, num_ctas: Int32,
                 scale_log2: Float32, stream: cuda.CUstream):
        # cache [blocks, page, W] -> (page, W, blocks): tokens are M, features K; query [B, H, KQ] -> (H, KQ, B)
        mC_mkl = cute.make_tensor(mC.iterator, cute.select(mC.layout, mode=[1, 2, 0]))
        mQ_nkl = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))
        mt = self.mt
        c_layout = sm90_utils.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (TILE_T, H, CHUNK), mt, self.stages)
        v_layout = sm90_utils.make_smem_layout_a(LayoutEnum.COL_MAJOR, (CHUNK, H, TILE_T), mt, self.stages)
        # K-major B operands with 24 rows, built without a stage mode: slicing an (8, 3) mode aborts in this DSL
        b_atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, mt)
        q_layout = cute.tile_to_shape(b_atom, (H, CHUNK * self.n_q), order=(0, 1))
        chunk_b_layout = cute.tile_to_shape(b_atom, (H, CHUNK), order=(0, 1))
        # rebuilt RoPE keys: the resident blocks of B_g [rot x 64] side by side
        r_layout = cute.tile_to_shape(b_atom, (ROT, CHUNK * max(self.n_rb, 1)), order=(0, 1))
        r_block_layout = cute.tile_to_shape(b_atom, (ROT, CHUNK), order=(0, 1))
        tma_c, mC_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mC_mkl, cute.slice_(c_layout, (None, None, 0)), (TILE_T, CHUNK))
        tma_q, mQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ_nkl, chunk_b_layout, (H, CHUNK))
        mR_nkl = cute.make_tensor(mR.iterator, cute.select(mR.layout, mode=[1, 2, 0]))    # [1, rot, KR] -> (rot, KR, 1)
        tma_r, mR_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mR_nkl, r_block_layout, (ROT, CHUNK))
        # cos / sin rows [max_pos, rot] -> (max_pos, rot, 1): a [64 positions x 64] box into a ring stage; positions
        # past the table read as zeros
        mCS_mkl = cute.make_tensor(mCS.iterator, cute.append(mCS.layout, cute.make_layout(1)))
        tma_cs, mCS_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mCS_mkl, cute.slice_(c_layout, (None, None, 0)), (TILE_T, CHUNK))
        mma_qk = sm90_utils.make_trivial_tiled_mma(mt, mt, OperandMajorMode.K, OperandMajorMode.K,
                                                   Float32, (1, 1, 1), (TILE_T, H))
        mma_pv = sm90_utils.make_trivial_tiled_mma(mt, mt, OperandMajorMode.MN, OperandMajorMode.K,
                                                   Float32, (1, 1, 1), (CHUNK, H))
        # rebuilt keys: B_g c~_g [tokens, rot] from the cache chunks, then the RoPE score with the keys in registers
        mma_kpe = sm90_utils.make_trivial_tiled_mma(mt, mt, OperandMajorMode.K, OperandMajorMode.K,
                                                    Float32, (1, 1, 1), (TILE_T, ROT))
        mma_qk_rs = sm90_utils.make_trivial_tiled_mma(mt, mt, OperandMajorMode.K, OperandMajorMode.K,
                                                      Float32, (1, 1, 1), (TILE_T, H),
                                                      a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Storage:
            pipe: cute.struct.MemRange[Int64, 2 * self.stages]
            qpipe: cute.struct.MemRange[Int64, 2]
            rpipe: cute.struct.MemRange[Int64, 2]
            red: cute.struct.MemRange[Float32, MMA_WARPS * H]
            sC: cute.struct.Align[cute.struct.MemRange[mt, cute.cosize(c_layout)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[mt, cute.cosize(q_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[mt, cute.cosize(chunk_b_layout)], 1024]
            sR: cute.struct.Align[cute.struct.MemRange[mt, cute.cosize(r_layout)], 1024]

        self.storage = Storage
        self.kernel(tma_c, mC_tma, tma_q, mQ_tma, tma_r, mR_tma, tma_cs, mCS_tma, mBT, mSeq, mWork, mCtaSeg, mPart,
                    scale_log2, c_layout, v_layout, q_layout, chunk_b_layout, r_layout, mma_qk, mma_pv, mma_kpe,
                    mma_qk_rs).launch(grid=[num_ctas, 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tma_c: cute.CopyAtom, mC_tma: cute.Tensor, tma_q: cute.CopyAtom, mQ_tma: cute.Tensor,
               tma_r: cute.CopyAtom, mR_tma: cute.Tensor, tma_cs: cute.CopyAtom, mCS_tma: cute.Tensor,
               mBT: cute.Tensor, mSeq: cute.Tensor, mWork: cute.Tensor, mCtaSeg: cute.Tensor, mPart: cute.Tensor,
               scale_log2: Float32, c_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout,
               q_layout: cute.ComposedLayout, chunk_b_layout: cute.ComposedLayout, r_layout: cute.ComposedLayout,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma, mma_kpe: cute.TiledMma, mma_qk_rs: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta, _, _ = cute.arch.block_idx()
        st = SmemAllocator().allocate(self.storage)
        sC = st.sC.get_tensor(c_layout.outer, swizzle=c_layout.inner)
        sV = st.sC.get_tensor(v_layout.outer, swizzle=v_layout.inner)      # the same chunks, read as V^T
        sQ = st.sQ.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        sP = st.sP.get_tensor(chunk_b_layout.outer, swizzle=chunk_b_layout.inner)
        sR = st.sR.get_tensor(r_layout.outer, swizzle=r_layout.inner)
        red = st.red.get_tensor(cute.make_layout((MMA_WARPS, H), stride=(H, 1)))
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.pipe.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, MMA_WARPS),
            tx_count=TILE_T * CHUNK * 2, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), tidx=tidx)
        qpipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.qpipe.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, MMA_WARPS),
            tx_count=H * CHUNK * 2 * self.n_q, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), tidx=tidx)
        rpipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.rpipe.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, MMA_WARPS),
            tx_count=ROT * CHUNK * 2 * max(self.n_rb, 1), cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), tidx=tidx)
        seg_begin = mCtaSeg[cta]
        seg_end = mCtaSeg[cta + 1]

        if warp_idx == MMA_WARPS:
            # producer: once the resident B_g blocks (only for a CTA with work: no copy may be in flight when
            # the CTA exits), per segment the query chunks, then every chunk of every tile in consumption order
            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            qstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if seg_begin < seg_end:
                rstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                rpipe.producer_acquire(rstate)
                gR = mR_tma[None, None, 0]
                for b in cutlass.range_constexpr(self.n_rb):
                    trs, trg = cpasync.tma_partition(
                        tma_r, 0, cute.make_layout(1),
                        cute.group_modes(cute.local_tile(sR, (ROT, CHUNK), (0, b)), 0, 2),
                        cute.group_modes(cute.local_tile(gR, (ROT, CHUNK), (0, b)), 0, 2))
                    cute.copy(tma_r, trg, trs, tma_bar_ptr=rpipe.producer_get_barrier(rstate))
                rpipe.producer_commit(rstate)
            for s in cutlass.range(seg_begin, seg_end, unroll=1):
                b = mWork[s, 0]
                qpipe.producer_acquire(qstate)
                gQ = mQ_tma[None, None, b]
                for c in cutlass.range_constexpr(self.n_q):
                    tqs, tqg = cpasync.tma_partition(
                        tma_q, 0, cute.make_layout(1),
                        cute.group_modes(cute.local_tile(sQ, (H, CHUNK), (0, c)), 0, 2),
                        cute.group_modes(cute.local_tile(gQ, (H, CHUNK), (0, c)), 0, 2))
                    cute.copy(tma_q, tqg, tqs, tma_bar_ptr=qpipe.producer_get_barrier(qstate))
                qpipe.producer_commit(qstate)
                qstate.advance()
                for tile in cutlass.range(mWork[s, 1], mWork[s, 2], unroll=1):
                    t0 = tile * TILE_T
                    page_idx = t0 // self.page
                    gC_page = mC_tma[None, None, mBT[b, page_idx]]
                    tok = t0 - page_idx * self.page
                    # the tile's cos / sin rows: one chunk ahead of the K part
                    gCS = cute.local_tile(cute.domain_offset((t0, 0), mCS_tma[None, None, 0]), (TILE_T, CHUNK),
                                          (0, 0))
                    tss, tsg = cpasync.tma_partition(tma_cs, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                                                     cute.group_modes(gCS, 0, 2))
                    pipe.producer_acquire(state)
                    cute.copy(tma_cs, tsg, tss[(None, state.index)], tma_bar_ptr=pipe.producer_get_barrier(state))
                    pipe.producer_commit(state)
                    state.advance()
                    for c in cutlass.range_constexpr(len(self.chunk_feat)):
                        gCc = cute.local_tile(cute.domain_offset((tok, self.chunk_feat[c]), gC_page),
                                              (TILE_T, CHUNK), (0, 0))
                        tcs, tcg = cpasync.tma_partition(tma_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                                                         cute.group_modes(gCc, 0, 2))
                        pipe.producer_acquire(state)
                        cute.copy(tma_c, tcg, tcs[(None, state.index)], tma_bar_ptr=pipe.producer_get_barrier(state))
                        pipe.producer_commit(state)
                        state.advance()

        if warp_idx < MMA_WARPS:
            if seg_begin < seg_end:
                rstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                rpipe.consumer_wait(rstate)                         # resident until the CTA exits
            self.consume(pipe, self.stages, qpipe, sC, sV, sQ, sP, sR, red, mma_qk, mma_pv, mma_kpe, mma_qk_rs, mSeq,
                         mWork, mPart, tidx, warp_idx, seg_begin, seg_end, scale_log2)

    @cute.jit
    def consume(self, pipe, n_stages: int, qpipe, sC: cute.Tensor, sV: cute.Tensor, sQ: cute.Tensor,
                sP: cute.Tensor, sR: cute.Tensor, red: cute.Tensor, mma_qk: cute.TiledMma,
                mma_pv: cute.TiledMma, mma_kpe: cute.TiledMma, mma_qk_rs: cute.TiledMma, mSeq: cute.Tensor,
                mWork: cute.Tensor, mPart: cute.Tensor, tidx: Int32, warp_idx: Int32, seg_begin: Int32,
                seg_end: Int32, scale_log2: Float32):
        """The MMA warps over the chunks of the cache ring ``pipe`` (``n_stages`` stages)."""
        lane = tidx % 32
        thr_qk = mma_qk.get_slice(tidx)
        thr_pv = mma_pv.get_slice(tidx)
        tSrC = thr_qk.make_fragment_A(thr_qk.partition_A(sC))       # (mma, M, K blocks, stage)
        tSrQ = thr_qk.make_fragment_B(thr_qk.partition_B(sQ))       # (mma, N, K blocks of all query chunks)
        tOrV = thr_pv.make_fragment_A(thr_pv.partition_A(sV))
        tOrP = thr_pv.make_fragment_B(thr_pv.partition_B(sP))
        acc_s = cute.make_rmem_tensor(thr_qk.partition_shape_C((TILE_T, H)), Float32)
        # B_g c~_g [tokens, rot] from the cache chunks and the resident blocks, rotated in registers, then the A
        # operand (registers) of the group's RoPE score against its query chunk
        thr_kpe = mma_kpe.get_slice(tidx)
        thr_rs = mma_qk_rs.get_slice(tidx)
        tKrC = thr_kpe.make_fragment_A(thr_kpe.partition_A(sC))
        tKrR = thr_kpe.make_fragment_B(thr_kpe.partition_B(sR))     # (mma, N, K blocks of all resident blocks)
        acc_k = cute.make_rmem_tensor(thr_kpe.partition_shape_C((TILE_T, ROT)), Float32)
        k_mn = reshape_acc_to_mn(acc_k)                                  # (2 token rows, 16 channel columns)
        ck_mn = reshape_acc_to_mn(thr_kpe.partition_C(cute.make_identity_tensor((TILE_T, ROT))))
        tSrK = thr_rs.make_fragment_A(thr_rs.partition_shape_A((TILE_T, ROT)))
        tSrQ_rs = thr_rs.make_fragment_B(thr_rs.partition_B(sQ))
        cs = cute.make_rmem_tensor((cute.size(k_mn, mode=[0]), cute.size(k_mn, mode=[1])), BFloat16)  # cos | sin
        acc_o = [cute.make_rmem_tensor(thr_pv.partition_shape_C((CHUNK, H)), Float32) for _ in range(self.n_v)]
        s_mn = reshape_acc_to_mn(acc_s)                                  # (2 token rows, 6 head columns)
        c_mn = reshape_acc_to_mn(thr_qk.partition_C(cute.make_identity_tensor((TILE_T, H))))
        o_mn = [reshape_acc_to_mn(acc_o[v]) for v in range(self.n_v)]   # (2 feature rows, the same 6 heads)
        co_mn = reshape_acc_to_mn(thr_pv.partition_C(cute.make_identity_tensor((CHUNK, H))))
        n_rows = cute.size(s_mn, mode=[0])
        n_cols = cute.size(s_mn, mode=[1])
        col_max = cute.make_rmem_tensor((n_cols,), Float32)
        col_sum = cute.make_rmem_tensor((n_cols,), Float32)
        tile_max = cute.make_rmem_tensor((n_cols,), Float32)

        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, n_stages)
        qstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        for s in cutlass.range(seg_begin, seg_end, unroll=1):
            tile0 = mWork[s, 1]
            tile1 = mWork[s, 2]
            end = cutlass.min(tile1 * TILE_T, mSeq[mWork[s, 0]])          # tokens past the context are masked
            for v in cutlass.range_constexpr(self.n_v):
                acc_o[v].fill(0.0)
            for j in cutlass.range_constexpr(n_cols):
                col_max[j] = -Float32.inf
                col_sum[j] = 0.0
            qpipe.consumer_wait(qstate)
            for tile in cutlass.range(tile0, tile1, unroll=1):
                t0 = tile * TILE_T
                # 0. the cos / sin of this thread's token rows and channel pairs, from the tile's chunk
                pipe.consumer_wait(state)
                half = cute.size(k_mn, mode=[1]) // 2
                for i in cutlass.range_constexpr(cute.size(k_mn, mode=[0])):
                    row = ck_mn[i, 0][0]
                    for j in cutlass.range_constexpr(half):
                        d = ck_mn[i, j][1]
                        cs[i, j] = sC[row, d, state.index]
                        cs[i, j + half] = sC[row, d + ROT // 2, state.index]
                pipe.consumer_release(state)
                state.advance()
                # 1. scores over the K-part chunks; after the chunk where a group's rr_g coordinates end, B_g c~
                #    of the group (the chunks they span times its resident blocks), rotated and scored one chunk
                #    later while that chunk's score is in flight (when several groups end in one chunk, all but
                #    the last at once); chunk c - 2 is released after chunk c
                done = state.clone()
                prev = state.clone()
                pending = None
                for c in cutlass.range_constexpr(self.n_k):
                    pipe.consumer_wait(state)
                    warpgroup.fence()
                    for k in cutlass.range_constexpr(K_BLOCKS):
                        mma_qk.set(warpgroup.Field.ACCUMULATE, c > 0 or k > 0)
                        cute.gemm(mma_qk, acc_s, tSrC[None, None, k, state.index],
                                  tSrQ[None, None, c * K_BLOCKS + k], acc_s)
                    warpgroup.commit_group()
                    if cutlass.const_expr(pending is not None):
                        warpgroup.wait_group(1)              # all but this chunk's score: the group's B_g c~
                        self.rope_score(pending, k_mn, cs, tSrK, acc_k, mma_qk_rs, acc_s, tSrQ_rs)
                    elif cutlass.const_expr(c >= 2):
                        warpgroup.wait_group(2)              # chunk c - 2 is read
                    pending = None
                    groups = self.rope_at[c] if "no_rope" not in EXPERIMENT else []
                    for i, (g, c0, b0) in enumerate(groups):
                        warpgroup.fence()
                        for cc in cutlass.range_constexpr(c0, c + 1):
                            stage = prev.index if cc < c else state.index
                            for k in cutlass.range_constexpr(K_BLOCKS):
                                mma_kpe.set(warpgroup.Field.ACCUMULATE, cc > c0 or k > 0)
                                cute.gemm(mma_kpe, acc_k, tKrC[None, None, k, stage],
                                          tKrR[None, None, (b0 + cc - c0) * K_BLOCKS + k], acc_k)
                        warpgroup.commit_group()
                        if cutlass.const_expr(i < len(groups) - 1):
                            warpgroup.wait_group(0)
                            self.rope_score(g, k_mn, cs, tSrK, acc_k, mma_qk_rs, acc_s, tSrQ_rs)
                    if cutlass.const_expr(len(groups) > 0):
                        pending = groups[-1][0]
                    if cutlass.const_expr(c >= 2):
                        pipe.consumer_release(done)
                        done.advance()
                    prev = state.clone()
                    state.advance()
                if cutlass.const_expr(pending is not None):
                    warpgroup.wait_group(0)
                    self.rope_score(pending, k_mn, cs, tSrK, acc_k, mma_qk_rs, acc_s, tSrQ_rs)
                warpgroup.wait_group(0)
                for _ in cutlass.range_constexpr(min(2, self.n_k)):
                    pipe.consumer_release(done)
                    done.advance()
                if tile == tile1 - 1:
                    qpipe.consumer_release(qstate)                       # the producer may load the next query
                # 2. mask tokens past the context, then the per-head max over the tile's 64 tokens
                for j in cutlass.range_constexpr(n_cols if "no_softmax" not in EXPERIMENT else 0):
                    m = -Float32.inf
                    for i in cutlass.range_constexpr(n_rows):
                        s_mn[i, j] = s_mn[i, j] if t0 + c_mn[i, j][0] < end else -Float32.inf
                        m = cute.arch.fmax(m, s_mn[i, j])
                    for off in cutlass.range_constexpr(3):                     # lanes with the same lane % 4
                        m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=4 << off))
                    tile_max[j] = m
                if lane < 4:
                    for j in cutlass.range_constexpr(n_cols if "no_softmax" not in EXPERIMENT else 0):
                        red[warp_idx, c_mn[0, j][1]] = tile_max[j]
                if cutlass.const_expr("no_softmax" not in EXPERIMENT):
                    cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
                for j in cutlass.range_constexpr(n_cols if "no_softmax" not in EXPERIMENT else 0):
                    h = c_mn[0, j][1]
                    m_new = cute.arch.fmax(cute.arch.fmax(red[0, h], red[1, h]), cute.arch.fmax(red[2, h], red[3, h]))
                    m_new = cute.arch.fmax(m_new, col_max[j])
                    alpha = cute.math.exp2((col_max[j] - m_new) * scale_log2, fastmath=True)
                    col_max[j] = m_new
                    m_scaled = m_new * scale_log2
                    psum = Float32(0.0)
                    for i in cutlass.range_constexpr(n_rows):
                        p = cute.math.exp2(s_mn[i, j] * scale_log2 - m_scaled, fastmath=True)
                        psum += p
                        sP[h, c_mn[i, j][0]] = p.to(self.mt)
                    col_sum[j] = col_sum[j] * alpha + psum
                    for v in cutlass.range_constexpr(self.n_v):
                        for i in cutlass.range_constexpr(cute.size(o_mn[v], mode=[0])):
                            o_mn[v][i, j] = o_mn[v][i, j] * alpha
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
                # 3. values: O^T[features, heads] += V_chunk^T @ P, one accumulator per chunk (independent groups)
                done = state.clone()
                for v in cutlass.range_constexpr(self.n_v):
                    pipe.consumer_wait(state)
                    warpgroup.fence()
                    mma_pv.set(warpgroup.Field.ACCUMULATE, True)
                    for k in cutlass.range_constexpr(K_BLOCKS):
                        cute.gemm(mma_pv, acc_o[v], tOrV[None, None, k, state.index], tOrP[None, None, k],
                                  acc_o[v])
                    warpgroup.commit_group()
                    state.advance()
                    if v >= self.inflight:
                        warpgroup.wait_group(self.inflight)
                        pipe.consumer_release(done)
                        done.advance()
                warpgroup.wait_group(0)
                for _ in cutlass.range_constexpr(min(self.inflight, self.n_v)):
                    pipe.consumer_release(done)
                    done.advance()

            # epilogue: the per-head sum over lanes and warps, then the normalized partial output and its LSE
            for j in cutlass.range_constexpr(n_cols):
                for off in cutlass.range_constexpr(3):
                    col_sum[j] = col_sum[j] + cute.arch.shuffle_sync_bfly(col_sum[j], offset=4 << off)
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
            if lane < 4:
                for j in cutlass.range_constexpr(n_cols):
                    red[warp_idx, c_mn[0, j][1]] = col_sum[j]
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
            for j in cutlass.range_constexpr(n_cols):
                h = c_mn[0, j][1]
                total = red[0, h] + red[1, h] + red[2, h] + red[3, h]
                inv = 1.0 / total
                for v in cutlass.range_constexpr(self.n_v):
                    for i in cutlass.range_constexpr(cute.size(o_mn[v], mode=[0])):
                        f = v * CHUNK + co_mn[i, j][0]
                        if f < self.lv:
                            mPart[s, h, f] = o_mn[v][i, j] * inv
                if warp_idx == 0 and lane < 4:
                    mPart[s, h, self.lv] = (col_max[j] * scale_log2 + cute.math.log2(total, fastmath=True)) * LN2
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)   # red is reused by the next segment
            qstate.advance()

    @cute.jit
    def rope_score(self, g: int, k_mn: cute.Tensor, cs: cute.Tensor, tSrK: cute.Tensor, acc_k: cute.Tensor,
                   mma_qk_rs: cute.TiledMma, acc_s: cute.Tensor, tSrQ_rs: cute.Tensor):
        """Group ``g``'s rebuilt RoPE keys (B_g c~ in ``acc_k``, complete): rotated, then the register A operand of
        the group's RoPE score against its query chunk (accumulated into ``acc_s``, one more wgmma group)."""
        self.rotate(k_mn, cs)
        tSrK.store(reshape_acc_to_frgA(acc_k).load().to(self.mt))
        warpgroup.fence()
        mma_qk_rs.set(warpgroup.Field.ACCUMULATE, True)
        for k in cutlass.range_constexpr(K_BLOCKS):
            cute.gemm(mma_qk_rs, acc_s, tSrK[None, None, k], tSrQ_rs[None, None, (self.n_k + g) * K_BLOCKS + k],
                      acc_s)
        warpgroup.commit_group()

    @cute.jit
    def rotate(self, k_mn: cute.Tensor, cs: cute.Tensor):
        """In place on the rebuilt keys' accumulator [token rows, channel columns]: rounding to BF16 (a stored key's
        precision, as before vLLM's rotary embedding), then the NeoX rotation with this thread's cos | sin."""
        half = cute.size(k_mn, mode=[1]) // 2
        for i in cutlass.range_constexpr(cute.size(k_mn, mode=[0])):
            for j in cutlass.range_constexpr(half):
                x1 = k_mn[i, j].to(BFloat16).to(Float32)
                x2 = k_mn[i, j + half].to(BFloat16).to(Float32)
                cos = cs[i, j].to(Float32)
                sin = cs[i, j + half].to(Float32)
                k_mn[i, j] = x1 * cos - x2 * sin
                k_mn[i, j + half] = x2 * cos + x1 * sin


_compiled: dict = {}


def _cache_dir() -> str:
    """Directory of the exported kernels ("" disables the on-disk cache)."""
    path = os.environ.get("LRKV_CUTE_CACHE_DIR")
    if path is None:
        from vllm import envs
        path = os.path.join(envs.VLLM_CACHE_ROOT, "lrkv_cute")
    return path


@functools.cache
def _build_id() -> bytes:
    """What the generated code depends on besides the compile key: this file, the DSL, quack and the GPU."""
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        h.update(f.read())
    h.update(f"{version('nvidia-cutlass-dsl')} {version('quack-kernels')} {torch.cuda.get_device_capability()}"
             .encode())
    return h.digest()


def _load_or_compile(key: tuple, make_kernel, args: tuple):
    """The compiled kernel of ``key``: loaded from the on-disk cache, or compiled and exported to it."""
    cache_dir = _cache_dir()
    if not cache_dir:
        return cute.compile(make_kernel(), *args)
    prefix = "lrkv_decode_" + hashlib.sha256(_build_id() + repr(key).encode()).hexdigest()[:32]
    path = os.path.join(cache_dir, prefix + ".o")
    if os.path.exists(path):
        try:
            compiled = getattr(cute.runtime.load_module(path), prefix)
            logger.debug("vllm_lrkv: decode kernel %s loaded from %s", key, path)
            return compiled
        except Exception as e:
            logger.warning("vllm_lrkv: cannot load %s (%s), recompiling", path, e)
    t = time.perf_counter()
    compiled = cute.compile(make_kernel(), *args)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=f".{prefix}.", dir=cache_dir)
        try:
            compiled.export_to_c(tmp, prefix, function_prefix=prefix)
            os.replace(os.path.join(tmp, prefix + ".o"), path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        logger.warning("vllm_lrkv: cannot cache the decode kernel in %s (%s)", cache_dir, e)
    logger.info("vllm_lrkv: compiled the decode kernel %s in %.1f s, cached as %s", key, time.perf_counter() - t,
                path)
    return compiled


def _cute(t: torch.Tensor, leading_dim: int):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


@dataclass
class DecodeSchedule:
    """Device buffers of one step's work split (``schedule``), shared by all compressed layers of the step.

    work [capacity, 4] int32: segment k = (request, first tile, end tile, 0), in request and tile order, so the
    segments of request b are seg_base[b] .. seg_base[b + 1] - 1 and CTA c owns cta_seg[c] .. cta_seg[c + 1] - 1."""

    num_ctas: int
    work: torch.Tensor
    cta_seg: torch.Tensor           # [num_ctas + 1] int32
    seg_base: torch.Tensor          # [max_reqs + 1] int32

    @property
    def capacity(self) -> int:
        return self.work.shape[0]


def make_schedule(max_reqs: int, device, num_ctas: int | None = None) -> DecodeSchedule:
    """Buffers for up to ``max_reqs`` decode requests; one CTA per SM by default (one fits, by shared memory)."""
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    i32 = dict(device=device, dtype=torch.int32)
    return DecodeSchedule(num_ctas, torch.zeros(max_reqs + num_ctas, 4, **i32), torch.zeros(num_ctas + 1, **i32),
                          torch.zeros(max_reqs + 1, **i32))


@triton.jit(do_not_specialize=["num_reqs"])
def _schedule(Seq, num_reqs, num_ctas, min_tiles, Work, CtaSeg, SegBase, BLOCK_B: tl.constexpr,
              BLOCK_C: tl.constexpr, TILE: tl.constexpr):
    offs = tl.arange(0, BLOCK_B)
    live = offs < num_reqs
    tiles = (tl.load(Seq + offs, mask=live, other=0) + TILE - 1) // TILE
    first = tl.cumsum(tiles, 0) - tiles                       # first global tile of every request
    total_tiles = tl.sum(tiles, 0)
    per = tl.maximum(tl.maximum((total_tiles + num_ctas - 1) // num_ctas, min_tiles), 1)
    cta0 = first // per
    nseg = tl.where(tiles > 0, (first + tiles - 1) // per - cta0 + 1, 0)
    base = tl.cumsum(nseg, 0) - nseg
    total = tl.sum(nseg, 0)
    tl.store(SegBase + offs, base, mask=live)
    tl.store(SegBase + num_reqs, total)
    for j in range(0, tl.max(nseg, 0)):
        ok = live & (j < nseg)
        cta = cta0 + j
        g0 = tl.maximum(first, cta * per)
        g1 = tl.minimum(first + tiles, (cta + 1) * per)
        k = base + j
        tl.store(Work + k * 4, offs, mask=ok)
        tl.store(Work + k * 4 + 1, g0 - first, mask=ok)
        tl.store(Work + k * 4 + 2, g1 - first, mask=ok)
        tl.store(CtaSeg + cta, k, mask=ok & (g0 == cta * per))
    for c0 in range(0, num_ctas + 1, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        tl.store(CtaSeg + c, tl.full([BLOCK_C], 0, tl.int32) + total, mask=(c <= num_ctas) & (c * per >= total_tiles))


def schedule(sched: DecodeSchedule, seq_lens: torch.Tensor, min_tiles: int = 1) -> None:
    """Fill ``sched`` for the decode requests ``seq_lens`` [B] (int32, on the device): runs of
    max(ceil(tiles / num_ctas), ``min_tiles``) tiles per CTA. A request of length 0 (padding) gets no segment."""
    B = seq_lens.shape[0]
    assert B + 1 <= sched.seg_base.shape[0]
    _schedule[(1,)](seq_lens, B, sched.num_ctas, min_tiles, sched.work, sched.cta_seg, sched.seg_base,
                    BLOCK_B=max(16, triton.next_power_of_2(B)), BLOCK_C=256, TILE=TILE_T, num_warps=4)


def block_diag_query(q_lat: torch.Tensor, q_rot: torch.Tensor, k_off: list[int], k_ranks: list[int],
                     lk_pad: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """[B, H, n_k * 64 + G * rot] query of the score GEMM: q_lat(h) [B, H, >= r_g] at its group's K columns, the
    rotated query q_rot(h) [B, H, rot] at its group's RoPE chunk, zeros elsewhere (reference construction; the
    backend computes it with ``latent_ops.block_diag_query``)."""
    B = q_lat.shape[0]
    n_k = math.ceil(lk_pad / CHUNK)
    q = torch.zeros(B, H, n_k * CHUNK + G * ROT, device=q_lat.device, dtype=dtype)
    hpg = H // G
    for g in range(G):
        hs = slice(g * hpg, (g + 1) * hpg)
        q[:, hs, k_off[g]:k_off[g] + k_ranks[g]] = q_lat[:, hs, :k_ranks[g]]
        q[:, hs, n_k * CHUNK + g * ROT:n_k * CHUNK + (g + 1) * ROT] = q_rot[:, hs]
    return q


@dataclass
class RopeRebuild:
    """RoPE-key inputs of the decode kernel: RoPE keys rebuilt in-kernel from the K latent."""

    groups: tuple                   # ((k_off_g, rr_g) per group): latent columns the group's keys depend on
    blocks: torch.Tensor            # [1, rot, n_blocks * 64] BF16: resident blocks of gain * B_g (``rope_blocks``)
    cos_sin: torch.Tensor           # [max_pos, rot] BF16: the model's cos_sin_cache (cos half | sin half)


def rope_blocks(u_rot: list[torch.Tensor], groups) -> torch.Tensor:
    """[1, rot, n_blocks * 64] blocks in the dtype of u_rot: u_rot[g] [rot, >= rr_g] are the rotary rows of U_g
    (after the export's change of basis) times the k_norm gain, reading latent columns k_off_g .. k_off_g + rr_g;
    block (g, c) holds them at the columns of the 64-column cache chunk c, in the kernel's order (groups, chunks)."""
    blocks = []
    for U, (off, rr) in zip(u_rot, groups):
        for c in range(off // CHUNK, (off + rr - 1) // CHUNK + 1):
            blk = torch.zeros(ROT, CHUNK, dtype=U.dtype, device=U.device)
            lo, hi = max(off, c * CHUNK), min(off + rr, (c + 1) * CHUNK)
            blk[:, lo - c * CHUNK:hi - c * CHUNK] = U[:, lo - off:hi - off]
            blocks.append(blk)
    return torch.cat(blocks, dim=1).unsqueeze(0).contiguous()


def latent_decode_cute(q_bd: torch.Tensor, cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor,
                       sched: DecodeSchedule, part: torch.Tensor, sm_scale: float, lk_pad: int, lv: int,
                       stages: int = STAGES, inflight: int = 1, *, rope: RopeRebuild) -> None:
    """Partial attention of one compressed layer over the segments of ``sched`` (``schedule`` of these seq_lens)
    into ``part`` [>= B + num_ctas - 1, H, lv + 1] fp32: per segment the normalized latent V-space output and LSE.

    q_bd [B, H, n_k * 64 + G * rot] (``block_diag_query``); cache [num_blocks, page, W] BF16: the layer's (possibly
    strided) view with 16-byte aligned rows and parts; block_table [B, max_pages] int32; ``rope``: ``RopeRebuild``
    parameters for rebuilding the RoPE keys in-kernel.
    """
    page = cache.shape[1]
    assert cache.dtype == torch.bfloat16 and cache.stride(-1) == 1 and q_bd.dtype == torch.bfloat16, q_bd.dtype
    assert cache.stride(1) % 8 == 0 and cache.stride(0) % 8 == 0 and cache.data_ptr() % 16 == 0
    assert part.shape[0] >= q_bd.shape[0] + sched.num_ctas - 1 and part.shape[-1] == lv + 1
    assert part.dtype == torch.float32
    blocks, cos_sin = rope.blocks, rope.cos_sin
    assert cache.shape[-1] >= lk_pad + lv and blocks.shape[1] == ROT and cos_sin.shape[1] == ROT
    assert blocks.dtype == q_bd.dtype and blocks.is_contiguous()
    assert cos_sin.dtype == torch.bfloat16 and cos_sin.is_contiguous() and cos_sin.data_ptr() % 16 == 0
    groups = tuple(rope.groups)
    key = (lk_pad, lv, page, stages, inflight, groups) + tuple(sorted(EXPERIMENT))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_cute(q_bd, 2), _cute(cache, 2), _cute(block_table, 1), _cute(seq_lens, 0), _cute(sched.work, 1),
            _cute(sched.cta_seg, 0), _cute(part, 2), _cute(blocks, 2), _cute(cos_sin, 1), Int32(sched.num_ctas),
            Float32(sm_scale * LOG2E), stream)
    compiled = _compiled.get(key)
    if compiled is None:
        compiled = _compiled[key] = _load_or_compile(
            key, lambda: LatentDecodeSm90(lk_pad, lv, page, groups, stages, inflight), args)
    compiled(*args)


@triton.jit
def _merge_segments(Part, Out, Lse, SegBase, stride_ps, stride_ph, stride_ob, stride_oh, stride_lb,
                    LV: tl.constexpr, NH: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
                    BLOCK_V: tl.constexpr):
    b = tl.program_id(0)
    s0 = tl.load(SegBase + b)
    n = tl.load(SegBase + b + 1) - s0
    hs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    vs = tl.program_id(2) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_h = hs < NH
    mask_hv = mask_h[:, None] & (vs < LV)[None, :]
    m = tl.full([BLOCK_H], -float("inf"), tl.float32)
    den = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_V], tl.float32)
    for k0 in range(0, n, BLOCK_S):                   # one pass over blocks of segments, rescaling as the max grows
        ks = k0 + tl.arange(0, BLOCK_S)
        seg = (s0 + ks).to(tl.int64) * stride_ps
        mask_sh = (ks < n)[:, None] & mask_h[None, :]
        lse = tl.load(Part + seg[:, None] + hs[None, :] * stride_ph + LV, mask=mask_sh, other=-float("inf"))
        p = tl.load(Part + seg[:, None, None] + hs[None, :, None] * stride_ph + vs[None, None, :],
                    mask=mask_sh[:, :, None] & mask_hv[None, :, :], other=0.0)
        m_new = tl.maximum(m, tl.max(lse, 0))
        old = tl.exp(m - m_new)
        w = tl.exp(lse - m_new[None, :])
        acc = acc * old[:, None] + tl.sum(w[:, :, None] * p, 0)
        den = den * old + tl.sum(w, 0)
        m = m_new
    has = den > 0.0                                   # a padded request (no segment) gets zeros, not NaN
    out = tl.where(has[:, None], acc / den[:, None], 0.0)
    tl.store(Out + b.to(tl.int64) * stride_ob + hs[:, None] * stride_oh + vs[None, :], out, mask=mask_hv)
    if tl.program_id(2) == 0:
        tl.store(Lse + b * stride_lb + hs, tl.where(has, m + tl.log(den), -float("inf")), mask=mask_h)


def merge_segments(part: torch.Tensor, sched: DecodeSchedule, out: torch.Tensor, lse: torch.Tensor) -> None:
    """Segment partials ``part`` [segments, H, lv + 1] -> out [B, H, lv] (its dtype), lse [B, H] (fp32)."""
    B, nh, lv = out.shape
    assert out.stride(-1) == 1 and part.shape[-1] == lv + 1
    block_s, block_h, block_v = (32, 1, 128) if B < 32 else (2, 8, 256)
    _merge_segments[(B, triton.cdiv(nh, block_h), triton.cdiv(lv, block_v))](
        part, out, lse, sched.seg_base, part.stride(0), part.stride(1), out.stride(0), out.stride(1), lse.stride(0),
        LV=lv, NH=nh, BLOCK_S=block_s, BLOCK_H=block_h, BLOCK_V=block_v, num_warps=4)

