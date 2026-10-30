"""Architecture adapters for hybrid attention stacks (the Qwen3.5 family, e.g. Qwen3.8-27B).

A Qwen3.5 text model interleaves Gated DeltaNet layers (``layer.linear_attn``, a
fixed-size recurrent state, no KV cache) with full-attention layers
(``layer.self_attn``). Only the latter carry a KV cache, so only they are
compressed; every other layer is still propagated by the dense teacher. Their
attention differs from Llama/Qwen3 in three ways handled here and in the callers:

* partial RoPE: only the first ``rotary_dim`` = ``partial_rotary_factor * head_dim``
  channels of every head rotate (``rotate_half`` inside that slice), the rest
  are position-free. The rotary tables have width ``rotary_dim``.
* zero-centered q_norm / k_norm: the effective gain of ``Qwen3_5RMSNorm`` is
  ``1 + weight`` (its epsilon attribute is ``eps``, not ``variance_epsilon``).
* an output gate: ``q_proj`` emits a query and a gate per head, and
  ``attn_output * sigmoid(gate)`` enters ``o_proj``.

Llama / Qwen3 models pass through every helper unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

ATTENTION = "attention"
LINEAR_ATTENTION = "linear_attention"


def layer_kind(layer: nn.Module) -> str:
    """``attention`` for a layer with ``self_attn`` (and a KV cache), ``linear_attention`` otherwise."""
    if getattr(layer, "self_attn", None) is not None:
        return ATTENTION
    if getattr(layer, "linear_attn", None) is not None:
        return LINEAR_ATTENTION
    raise TypeError(f"{type(layer).__name__} has neither self_attn nor linear_attn")


def attention_layer_indices(model: nn.Module) -> list[int]:
    """Indices of the decoder layers with a KV cache (all of them for Llama / Qwen3)."""
    return [i for i, layer in enumerate(model.model.layers) if layer_kind(layer) == ATTENTION]


def config_attention_layer_indices(config: dict) -> list[int]:
    """Same as :func:`attention_layer_indices` from a config.json dict (text config of a VLM)."""
    config = text_config_dict(config)
    count = int(config["num_hidden_layers"])
    types = config.get("layer_types")
    if types is None:
        return list(range(count))
    return [i for i, kind in enumerate(types[:count]) if kind == "full_attention"]


def text_config_dict(config: dict) -> dict:
    """The language-model part of a config.json dict (``text_config`` of a VLM wrapper)."""
    return config.get("text_config", config) if "vocab_size" not in config else config


def read_text_config(model_dir) -> dict:
    return text_config_dict(json.loads((Path(model_dir) / "config.json").read_text()))


def rotary_dim(config) -> int:
    """Rotating channels per head: ``head_dim`` unless the config sets a partial rotary factor."""
    from .checkpoint import head_dim_of
    head_dim = head_dim_of(config)
    rope = getattr(config, "rope_parameters", None) or {}
    factor = rope.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)) or 1.0
    dim = int(head_dim * float(factor))
    if not 0 < dim <= head_dim or dim % 2:
        raise ValueError(f"invalid rotary dimension {dim} for head_dim {head_dim}")
    return dim


def pad_rope_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """Extend rotary tables of width ``rotary_dim`` to ``head_dim``: cos 1, sin 0 on the position-free channels.

    Together with ``lrkv.whitened.rotate_half(..., rotary_dim)`` this is exactly the
    HF partial rotation (the rotated slice first, the pass-through channels after it).
    """
    width = cos.shape[-1]
    if sin.shape != cos.shape or width > head_dim:
        raise ValueError("rotary tables must share a shape no wider than head_dim")
    if width == head_dim:
        return cos, sin
    shape = (*cos.shape[:-1], head_dim - width)
    return (torch.cat((cos, cos.new_ones(shape)), dim=-1), torch.cat((sin, sin.new_zeros(shape)), dim=-1))


def rms_norm_eps(norm: nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        value = getattr(norm, name, None)
        if value is not None:
            return float(value)
    raise AttributeError(f"{type(norm).__name__} exposes no RMSNorm epsilon")


@torch.no_grad()
def rms_norm_gain(norm: nn.Module) -> torch.Tensor:
    """Effective FP32 per-channel gain g of an RMSNorm, ``norm(x) = g * x / rms(x)``.

    ``weight`` for Llama-style norms, ``1 + weight`` for the zero-centered Qwen3.5
    norm. The candidate is confirmed against the module itself on a unit-RMS
    input, so an unknown convention fails instead of silently using a wrong gain.
    """
    weight = norm.weight.detach()
    # All-ones input has unit RMS; the two conventions differ by exactly 1 per channel, far above the
    # rounding of a half-precision module output.
    probe = torch.ones(1, weight.numel(), dtype=weight.dtype, device=weight.device)
    output = norm(probe)[0].float() * (1.0 + rms_norm_eps(norm)) ** 0.5
    weight = weight.float()
    for gain in (weight, 1.0 + weight):
        if float((output - gain).abs().max()) <= 1e-2 * (1.0 + float(gain.abs().max())):
            return gain
    raise ValueError(f"{type(norm).__name__}: unknown RMSNorm gain convention")


def gate_sigmoid(q_proj_output: torch.Tensor, num_heads: int, head_dim: int):
    """sigmoid(gate) [..., Hq, h] of a gated q_proj output [..., Hq * 2h], or None for an ungated one."""
    width = q_proj_output.shape[-1]
    if width == num_heads * head_dim:
        return None
    if width != 2 * num_heads * head_dim:
        raise ValueError(f"q_proj width {width} is neither Hq*h nor 2*Hq*h")
    per_head = q_proj_output.reshape(*q_proj_output.shape[:-1], num_heads, 2 * head_dim)
    return torch.sigmoid(per_head[..., head_dim:].float())


def has_output_gate(attn: nn.Module) -> bool:
    from .checkpoint import head_dim_of
    config = attn.config
    return attn.q_proj.out_features == 2 * int(config.num_attention_heads) * head_dim_of(config)
