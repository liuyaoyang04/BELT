"""Low-rank KV checkpoint format and loading.

Format
------
A checkpoint is a plain ``state_dict`` (``torch.save``). For every compressed
attention layer ``i`` (prefix ``model.layers.{i}.self_attn.``) it holds:

    k_proj.VS.weight    [sum_h r_h, hidden]        per-KV-head latent projections, heads back to back
    k_proj.U.weight     [Hkv * head_dim, sum_h r_h] block-diagonal per head
    k_proj.head_ranks   int32 [Hkv]                r_h for every KV head
    v_proj.VS.weight    [r_v, hidden]              joint latent projection for the layer
    v_proj.U.weight     [Hkv * head_dim, r_v]

Everything else (q_proj, o_proj, MLP, norms, embeddings, lm_head) is optional:
a key that is present overrides the base model's tensor, a key that is absent
keeps it. A layer may compress only K or only V; the other projection stays a
dense ``nn.Linear`` (and is wrapped at full rank when the latent-cache
attention is installed). A layer with neither key stays uncompressed.

CLI
---
    python -m lrkv.checkpoint inspect  ckpt.pt [--model MODEL_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .modules import HeadwiseLowRankLinear, LowRankLinear

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)\.(.+)$")
K_FACTOR_KEYS = ("VS.weight", "U.weight", "head_ranks")
V_FACTOR_KEYS = ("VS.weight", "U.weight")


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}.self_attn."


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def is_lowrank_state_dict(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.endswith("k_proj.VS.weight") or k.endswith("v_proj.VS.weight") for k in sd)


def compressed_layers(sd: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, bool]]:
    """{layer_index: {"k": bool, "v": bool}} for every layer carrying low-rank factors."""
    out: Dict[int, Dict[str, bool]] = {}
    for k in sd:
        m = LAYER_RE.match(k)
        if m is None or m.group(3) != "VS.weight":
            continue
        i = int(m.group(1))
        out.setdefault(i, {"k": False, "v": False})[m.group(2)[0]] = True
    return dict(sorted(out.items()))


def infer_skip_layers(sd: Dict[str, torch.Tensor], num_layers: int) -> Tuple[int, ...]:
    """Layers that carry no low-rank factor at all. The checkpoint is the authority."""
    return tuple(i for i in range(num_layers) if i not in compressed_layers(sd))


def layer_ranks(sd: Dict[str, torch.Tensor], i: int) -> Tuple[Optional[list], Optional[int]]:
    """(per-head k ranks or None, joint v rank or None)."""
    pre = layer_prefix(i)
    if pre + "k_proj.group_ranks" in sd:
        raise ValueError(f"layer {i}: grouped K checkpoints are not supported (K must be per-head)")
    k = sd[pre + "k_proj.head_ranks"].tolist() if pre + "k_proj.head_ranks" in sd else None
    v = int(sd[pre + "v_proj.VS.weight"].shape[0]) if pre + "v_proj.VS.weight" in sd else None
    return k, v


def factor_keys(sd: Dict[str, torch.Tensor]) -> Iterable[str]:
    for k in sd:
        m = LAYER_RE.match(k)
        if m is not None and m.group(3) in (K_FACTOR_KEYS if m.group(2) == "k_proj" else V_FACTOR_KEYS):
            yield k


def factors_only(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the low-rank factor tensors (drops any fine-tuned dense weights)."""
    return {k: sd[k] for k in factor_keys(sd)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_state_dict_file(path: str, mmap: bool = False) -> Dict[str, torch.Tensor]:
    """Read a checkpoint's state dict.

    Prefer eager reads on the shared filesystem: demand-paging a large
    checkpoint into GPU copies causes slow small reads. Local-disk callers can opt
    into mmap explicitly when avoiding an eager CPU allocation matters.
    """
    sd = torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not is_lowrank_state_dict(sd):
        raise ValueError(f"{path} carries no low-rank factors (no *.k_proj.VS.weight / *.v_proj.VS.weight keys)")
    return sd


def head_dim_of(config) -> int:
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


@torch.no_grad()
def install_lowrank_modules(model: nn.Module, sd: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, bool]]:
    """Replace k_proj / v_proj with low-rank modules shaped from the checkpoint's own tensors.

    Pruned factors have per-layer (and per-head) widths, so the modules must
    be built from the checkpoint before ``load_state_dict`` runs. Returns the
    {layer: {"k","v"}} map of what was installed.
    """
    layers = compressed_layers(sd)
    for i, which in layers.items():
        attn = model.model.layers[i].self_attn
        pre = layer_prefix(i)
        if which["k"]:
            VS, U = sd[pre + "k_proj.VS.weight"], sd[pre + "k_proj.U.weight"]
            if pre + "k_proj.group_ranks" in sd:
                raise ValueError(f"layer {i}: grouped K checkpoints are not supported (K must be per-head)")
            ranks = sd[pre + "k_proj.head_ranks"].tolist()
            ref = attn.k_proj.weight
            assert U.shape[0] == ref.shape[0] and VS.shape[1] == ref.shape[1], (
                f"layer {i}: k_proj factors {tuple(U.shape)} x {tuple(VS.shape)} do not match the base "
                f"projection {tuple(ref.shape)}")
            attn.k_proj = HeadwiseLowRankLinear(VS.shape[1], U.shape[0], ranks, U.shape[0] // len(ranks),
                                                device=ref.device, dtype=ref.dtype)
        if which["v"]:
            VS, U = sd[pre + "v_proj.VS.weight"], sd[pre + "v_proj.U.weight"]
            ref = attn.v_proj.weight
            assert U.shape[0] == ref.shape[0] and VS.shape[1] == ref.shape[1], (
                f"layer {i}: v_proj factors {tuple(U.shape)} x {tuple(VS.shape)} do not match the base "
                f"projection {tuple(ref.shape)}")
            if pre + "v_proj.head_ranks" in sd:
                raise ValueError(f"layer {i}: per-head V checkpoints are not supported (V is one joint latent)")
            attn.v_proj = LowRankLinear(VS.shape[1], U.shape[0], VS.shape[0], device=ref.device, dtype=ref.dtype)
    return layers


@dataclass
class LoadInfo:
    path: Optional[str]
    compressed: Dict[int, Dict[str, bool]]
    skip_layers: Tuple[int, ...]
    overridden_dense_keys: int          # non-factor tensors the checkpoint overrode
    unexpected_keys: list


def load_lowrank_checkpoint(model: nn.Module, checkpoint) -> LoadInfo:
    """Install low-rank modules from ``checkpoint`` (path or state dict) and load every tensor it holds.

    Raises if any factor tensor fails to load; with ``strict=False`` an
    unloaded factor would silently stay at its random init.
    """
    path = None
    if isinstance(checkpoint, str):
        path = checkpoint
        sd = load_state_dict_file(checkpoint)
    else:
        sd = checkpoint
    compressed = install_lowrank_modules(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    factor_set = set(factor_keys(sd))
    critical = [k for k in missing if k in factor_set]
    if critical:
        raise RuntimeError(f"{len(critical)} factor tensors failed to load, e.g. {critical[:5]}")
    if unexpected:
        print(f"  warning: {len(unexpected)} checkpoint tensors have no target in the model, e.g. {unexpected[:5]}")
    from .arch import attention_layer_indices
    kv_layers = attention_layer_indices(model)     # layers with a KV cache (hybrid stacks: full attention only)
    return LoadInfo(
        path=path,
        compressed=compressed,
        skip_layers=tuple(i for i in kv_layers if i not in compressed),
        overridden_dense_keys=len(sd) - len(factor_set),
        unexpected_keys=list(unexpected),
    )


def save_state_dict(sd: Dict[str, torch.Tensor], path: str, only_factors: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = factors_only(sd) if only_factors else sd
    torch.save({k: v.contiguous() for k, v in sd.items()}, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _describe(sd, model_dir: Optional[str] = None) -> dict:
    layers = compressed_layers(sd)
    n_factor = len(list(factor_keys(sd)))
    info = {
        "num_tensors": len(sd),
        "num_factor_tensors": n_factor,
        "num_other_tensors": len(sd) - n_factor,
        "compressed_layers": {str(i): w for i, w in layers.items()},
        "ranks": {},
    }
    for i in layers:
        k, v = layer_ranks(sd, i)
        info["ranks"][str(i)] = {"k_head_ranks": k, "v_rank": v}
    if model_dir:
        from .arch import config_attention_layer_indices
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        info["skip_layers"] = [i for i in config_attention_layer_indices(cfg) if i not in layers]
    return info


def main():
    p = argparse.ArgumentParser(description="Inspect low-rank KV checkpoints.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="Print the compressed layers and ranks of a checkpoint")
    pi.add_argument("checkpoint")
    pi.add_argument("--model", default=None, help="Model directory (config.json) to list skipped layers")
    pi.add_argument("--json", default=None, help="Write the description to this JSON file")
    args = p.parse_args()

    sd = load_state_dict_file(args.checkpoint)
    info = _describe(sd, args.model)
    print(json.dumps(info, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
