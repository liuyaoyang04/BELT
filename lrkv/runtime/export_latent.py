"""Write a low-rank checkpoint + base model as a latent-KV HF directory for the ``vllm_lrkv`` plugin.

    python -m lrkv.runtime.export_latent --model models/Qwen3.8-27B --checkpoint ckpt.pt --output DIR [--rope-align]

For every compressed attention layer the base shard is rewritten: ``k_proj`` / ``v_proj``
are dropped, ``q_proj`` is renamed ``q_gate_proj`` (so the stock weight mapper does not fold it into a fused qkv
projection) and three tensors are added:

    self_attn.kv_latent_proj.weight  [pe_offset, hidden]   [VS_1; ...; VS_Hkv; 0; VS_v; 0]: the cache row's K part and
                                                            V part, each starting 16-byte aligned (zero rows pad)
    self_attn.k_up.weight            [Hkv, head_dim, r_max] U_g per KV head (zero padded): the per-token rms, the RoPE
                                                            key and the absorbed query are derived from it at load
    self_attn.v_up.weight            [Hkv * head_dim, r_v]  U_v of the joint V

The cache row of a compressed layer is ``[c~_1 | ... | c~_Hkv | 0 | c_v | 0]`` with
c~_g = c_g / rms(U_g c_g); the latent projection produces its ``pe_offset`` columns directly (the zero rows
give the alignment gaps), and the decode kernel rebuilds the RoPE keys from the K latent on the fly:
k_pe,g = RoPE(gain_r * U_g,r c~_g). Each group's latent basis is first turned by an orthogonal Q_g
(QR of U_g,r^T: VS_g -> Q_g^T VS_g, U_g -> U_g Q_g, the product and every score unchanged) so that the rotary rows
of U_g Q_g are zero past the first min(rot, r_g) latent coordinates: the rebuild reads only those, a [rot x rot]
block per group whatever the rank.

``--rope-align`` (64 rotary channels) also reorders the K part: the leading min(rot, r_g) coordinates
of every group (the only ones its RoPE keys read) come first, every group with a full 64 of them in a 64-column
chunk of its own and the smaller ones after them (largest first), then the rest of every group in group order. The
decode kernel reads the row in 64-column chunks and rebuilds a group's keys once the chunks holding those
coordinates are in. ``k_segments`` in the layout lists every group's columns as [start, length] runs in latent order,
``rope_offsets`` the column of every group's first coordinate.

Every other tensor (q/o projections, norms, GDN layers, vision tower, MTP) is the base checkpoint's own;
shards without a compressed layer are copied byte for byte. config.json gets a ``kv_lowrank`` block (format
``latent``, per-layer ranks and row layout) and the plugin's architecture name. ``latent_export.json`` records
provenance and the cache accounting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import shutil
import time

import torch

from ..checkpoint import compressed_layers, layer_prefix, load_state_dict_file

PLUGIN_ARCHITECTURE = "Qwen3_5LowRankLatentForConditionalGeneration"
PLUGIN_ARCHITECTURE_TEXT = "Qwen3_5LowRankLatentForCausalLM"
FORMAT = "latent"
ALIGN = 8                      # BF16 elements per 16 bytes: TMA box starts and row strides
WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".gguf")
Q_GATE_NAME = "q_gate_proj"


def sha256_of(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _text_config(config: dict) -> dict:
    return config["text_config"] if "text_config" in config else config


def _rotary_dim(tc: dict, head_dim: int) -> int:
    rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    factor = rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)) or 1.0
    return int(head_dim * float(factor))


def _layer_name_prefix(weight_map: dict, i: int) -> str:
    """The base checkpoint's name prefix of layer ``i``'s attention (VLM layout or plain)."""
    for pre in (f"model.language_model.layers.{i}.self_attn.", f"model.layers.{i}.self_attn."):
        if pre + "q_proj.weight" in weight_map:
            return pre
    raise SystemExit(f"layer {i}: no q_proj in the base index")


def _k_blocks(sd: dict, i: int, tensors: dict, pre: str, num_kv_heads: int, head_dim: int):
    """Per-head (VS_g, U_g) of layer ``i`` from the checkpoint, or identity factors of the base k_proj."""
    p = layer_prefix(i)
    if p + "k_proj.VS.weight" in sd:
        if p + "k_proj.group_ranks" in sd:
            raise SystemExit(f"layer {i}: grouped K is not supported by latent export")
        VS, U = sd[p + "k_proj.VS.weight"].float(), sd[p + "k_proj.U.weight"].float()
        ranks = [int(r) for r in sd[p + "k_proj.head_ranks"].tolist()]
        blocks, col = [], 0
        for g, r in enumerate(ranks):
            blocks.append((VS[col:col + r], U[g * head_dim:(g + 1) * head_dim, col:col + r]))
            col += r
        return blocks
    W = tensors[pre + "k_proj.weight"].float()
    eye = torch.eye(head_dim, dtype=torch.float32)
    return [(W[g * head_dim:(g + 1) * head_dim], eye) for g in range(num_kv_heads)]


def _v_factors(sd: dict, i: int, tensors: dict, pre: str):
    p = layer_prefix(i)
    if p + "v_proj.VS.weight" in sd:
        if p + "v_proj.head_ranks" in sd:
            raise SystemExit(f"layer {i}: per-head V is not supported by latent export")
        return sd[p + "v_proj.VS.weight"].float(), sd[p + "v_proj.U.weight"].float()
    W = tensors[pre + "v_proj.weight"].float()
    return W, torch.eye(W.shape[0], dtype=torch.float32)


def _align(x: int, align: int = ALIGN) -> int:
    return -(-x // align) * align


def k_segments(k_head_ranks: list[int], rotary_dim: int, rope_align: bool = False) -> list[list[list[int]]]:
    """Cache columns of every KV group's latent coordinates as [start, length] runs in latent order: group after
    group, or (``rope_align``) the leading min(rot, r_g) coordinates of every group first, groups with rot of them
    in group order (never across a 64-column chunk boundary when rot divides 64), then the others largest first,
    then the rest of every group in group order (module docstring)."""
    if not rope_align:
        offsets = [sum(k_head_ranks[:g]) for g in range(len(k_head_ranks))]
        return [[[o, int(r)]] for o, r in zip(offsets, k_head_ranks)]
    rr = [min(rotary_dim, int(r)) for r in k_head_ranks]
    segments, col = [[] for _ in rr], 0
    for g in sorted(range(len(rr)), key=lambda g: -rr[g]):                 # stable: full groups in group order
        segments[g].append([col, rr[g]])
        col += rr[g]
    for g, r in enumerate(k_head_ranks):
        if r > rr[g]:
            segments[g].append([col, int(r) - rr[g]])
            col += int(r) - rr[g]
    return segments


def k_columns(layout: dict) -> list[int]:
    """Cache column of every K latent coordinate, groups in order (the order of the checkpoint's factors and of a
    packed row)."""
    segments = layout.get("k_segments") or k_segments(layout["k_head_ranks"], 0)
    return [start + j for runs in segments for start, n in runs for j in range(n)]


def row_layout(k_head_ranks: list[int], v_rank: int, num_kv_heads: int, rotary_dim: int,
               align: int = ALIGN, rope_align: bool = False) -> dict:
    """Offsets of one compressed layer's cache row: K part at 0, V part, ending at ``pe_offset`` (RoPE keys are
    rebuilt on the fly by the decode kernel); parts start at multiples of ``align`` elements. Also lists how many
    leading latent coordinates of every group the RoPE keys depend on, where every group's coordinates sit
    (``k_segments``, packed or ``rope_align``) and where its RoPE ones start."""
    if rope_align and 64 % rotary_dim:
        raise ValueError(f"rope_align keeps every group's RoPE coordinates in one 64-column chunk: rot {rotary_dim} "
                         "must divide 64")
    lk = sum(k_head_ranks)
    v_offset = _align(lk, align)
    pe_offset = _align(v_offset + v_rank, align)
    segments = k_segments(k_head_ranks, rotary_dim, rope_align)
    return {
        "k_head_ranks": list(k_head_ranks),
        "v_rank": int(v_rank),
        "lk": lk,
        "v_offset": v_offset,
        "pe_offset": pe_offset,
        "row_width": pe_offset,
        "rope_ranks": [min(rotary_dim, int(r)) for r in k_head_ranks],
        "k_segments": segments,
        "rope_offsets": [runs[0][0] for runs in segments],
    }


def rope_basis(U: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Orthogonal Q [r, r] (float64) with (U Q)[:rotary_dim, min(rotary_dim, r):] = 0 for U [head_dim, r]: the
    rotary rows of U Q depend on the first min(rotary_dim, r) latent coordinates only (identity when r <= rot)."""
    r = U.shape[1]
    if r <= rotary_dim:
        return torch.eye(r, dtype=torch.float64)
    Q, _ = torch.linalg.qr(U[:rotary_dim].double().T, mode="complete")     # U_r^T = Q R, R [r, rot] upper triangular
    return Q


def build_latent_layer(k_blocks, v_factors, num_kv_heads: int, head_dim: int, rotary_dim: int,
                       dtype: torch.dtype = torch.bfloat16, align: int = ALIGN,
                       rope_align: bool = False):
    """(kv_latent_proj [pe_offset, hidden], k_up [Hkv, head_dim, r_max], v_up [Hkv * head_dim, r_v], layout) from
    per-head K factors (VS_g [r_g, hidden], U_g [head_dim, r_g]) and joint V factors (VS_v, U_v [Hkv * head_dim,
    r_v]). Products are not formed: the factors are stored as they are, in ``dtype``, after the orthogonal change
    of each group's latent basis (``rope_basis``) for rebuilt RoPE keys."""
    ranks = [int(VS.shape[0]) for VS, _ in k_blocks]
    VS_v, U_v = (t.float() for t in v_factors)
    if len(ranks) != num_kv_heads or U_v.shape[0] != num_kv_heads * head_dim:
        raise ValueError(f"{len(ranks)} K blocks / U_v {tuple(U_v.shape)} for {num_kv_heads} KV heads")
    layout = row_layout(ranks, int(VS_v.shape[0]), num_kv_heads, rotary_dim, align, rope_align)
    segments = layout["k_segments"]
    hidden = int(VS_v.shape[1])
    proj = torch.zeros(layout["pe_offset"], hidden, dtype=torch.float32)
    k_up = torch.zeros(num_kv_heads, head_dim, max(ranks), dtype=torch.float32)
    for g, ((VS, U), r) in enumerate(zip(k_blocks, ranks)):
        Q = rope_basis(U, rotary_dim)
        VS, U = (Q.T @ VS.double()).float(), (U.double() @ Q).float()
        U[:rotary_dim, min(rotary_dim, r):] = 0.0                         # exactly zero (rounding residue only)
        i = 0
        for start, n in segments[g]:
            proj[start:start + n] = VS[i:i + n].float()
            i += n
        k_up[g, :, :r] = U.float()
    proj[layout["v_offset"]:layout["v_offset"] + layout["v_rank"]] = VS_v
    return proj.to(dtype), k_up.to(dtype), U_v.to(dtype), layout


def export(model_path: str, checkpoint: str, output: str, dtype: torch.dtype = torch.bfloat16,
           rope_align: bool = False) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    tc = _text_config(config)
    if "quantization_config" in config or "quantization_config" in tc:
        raise SystemExit(f"{model_path}: quantized base models are not supported; pass an unquantized BF16/FP16 model")
    head_dim = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"])
    num_kv_heads = int(tc["num_key_value_heads"])
    rot = _rotary_dim(tc, head_dim)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = dict(index["weight_map"])

    sd = load_state_dict_file(checkpoint)
    layers = compressed_layers(sd)
    if not layers:
        raise SystemExit("checkpoint compresses no layer")
    by_shard = defaultdict(list)
    for i in layers:
        pre = _layer_name_prefix(weight_map, i)
        shards = {weight_map[n] for n in weight_map if n.startswith(pre)}
        if len(shards) != 1:
            raise SystemExit(f"layer {i}: attention tensors span {len(shards)} shards (unsupported)")
        by_shard[shards.pop()].append((i, pre))

    os.makedirs(output, exist_ok=True)
    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        if os.path.isfile(src) and not name.endswith(WEIGHT_FILES) and name != "config.json" \
                and not name.startswith("pytorch_model"):
            shutil.copy2(src, os.path.join(output, name))

    layer_info, new_map = {}, dict(weight_map)
    for shard in sorted(set(weight_map.values())):
        if shard not in by_shard:
            shutil.copy2(os.path.join(model_path, shard), os.path.join(output, shard))
            continue
        tensors = {}
        with safe_open(os.path.join(model_path, shard), framework="pt") as base:
            metadata = base.metadata()
            for n in base.keys():
                tensors[n] = base.get_tensor(n)
        for i, pre in by_shard[shard]:
            k_blocks = _k_blocks(sd, i, tensors, pre, num_kv_heads, head_dim)
            v_factors = _v_factors(sd, i, tensors, pre)
            proj, k_up, v_up, layout = build_latent_layer(k_blocks, v_factors, num_kv_heads, head_dim, rot, dtype,
                                                          ALIGN, rope_align)
            for n in [n for n in tensors if n.startswith((pre + "k_proj.", pre + "v_proj."))]:
                del tensors[n]
                del new_map[n]
            for n in [n for n in tensors if n.startswith(pre + "q_proj.")]:
                renamed = n.replace(pre + "q_proj.", pre + Q_GATE_NAME + ".")
                tensors[renamed] = tensors.pop(n)
                new_map[renamed] = new_map.pop(n)
            for suffix, t in [("kv_latent_proj.weight", proj), ("k_up.weight", k_up), ("v_up.weight", v_up)]:
                tensors[pre + suffix] = t.contiguous()
                new_map[pre + suffix] = shard
            layer_info[str(i)] = {**layout, "k_compressed": bool(layers[i]["k"]), "v_compressed": bool(layers[i]["v"])}
        save_file({n: t.contiguous() for n, t in tensors.items()}, os.path.join(output, shard), metadata=metadata)

    index["weight_map"] = new_map
    with open(os.path.join(output, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    tc["kv_lowrank"] = {
        "format": FORMAT, "rotary_dim": rot, "nope_dim": head_dim - rot, "rope_slice": "rebuilt",
        "norm_folded": True, "alignment_elements": ALIGN, "latent_dtype": str(dtype).replace("torch.", ""),
        "cache_dtype": str(dtype).replace("torch.", ""), "rope_align": rope_align,
        "q_gate_proj": Q_GATE_NAME, "layers": layer_info, "base_architectures": config.get("architectures"),
        "checkpoint": os.path.abspath(checkpoint),
    }
    config["architectures"] = [PLUGIN_ARCHITECTURE if "text_config" in config else PLUGIN_ARCHITECTURE_TEXT]
    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    full = 2 * num_kv_heads * head_dim
    per_token = {i: v["row_width"] for i, v in layer_info.items()}
    packed = {i: v["lk"] + v["v_rank"] for i, v in layer_info.items()}
    record = {
        "base_model": os.path.abspath(model_path), "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": sha256_of(checkpoint), "format": FORMAT, "dtype": str(dtype),
        "cache_dtype": str(dtype).replace("torch.", ""),
        "compressed_layers": sorted(int(i) for i in layer_info), "layers": layer_info,
        "cache_elements_per_token": {"compressed_layers": per_token, "total": sum(per_token.values()),
                                     "packed_total": sum(packed.values()), "full_kv_total": full * len(per_token),
                                     "ratio": sum(per_token.values()) / (full * len(per_token))},
        "cache_bytes_ratio": sum(per_token.values()) * dtype.itemsize / (2 * full * len(per_token)),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(output, "latent_export.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description="Write a low-rank checkpoint as a latent-KV HF model directory.")
    p.add_argument("--model", required=True, help="Base model directory (safetensors shards + index)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--rope-align", action="store_true",
                   help="Reorder K columns so every group's RoPE coordinates sit in one 64-column chunk.")
    args = p.parse_args(argv)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    record = export(args.model, args.checkpoint, args.output, dtype=dtype, rope_align=args.rope_align)
    print(json.dumps({k: v for k, v in record.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
