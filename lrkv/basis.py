"""Full frozen bases and selected-rank export for ours.

These artifacts contain every rank prefix and its spectrum. A truncated
deployment checkpoint is deliberately not accepted as an allocation basis.
"""
from __future__ import annotations

import torch
from torch import nn

from .checkpoint import head_dim_of, layer_prefix
from .compression import kv_cache_report


def validate_bases(model, bases):
    """Validate full per-KV-head K bases and joint whole-layer V bases."""
    from .arch import attention_layer_indices
    expected = {layer_prefix(i) + p for i in attention_layer_indices(model)
                for p in ("k_proj", "v_proj")}
    if set(bases) != expected:
        raise ValueError("allocation bases must cover exactly all K/V projections")
    for name, factors in bases.items():
        linear = model.get_submodule(name)
        if not isinstance(linear, nn.Linear) or linear.bias is not None:
            raise ValueError("allocation requires original bias-free nn.Linear K/V")
        kv_heads = int(model.config.num_key_value_heads)
        groups = kv_heads if name.endswith("k_proj") else 1
        if len(factors) != groups or linear.out_features % groups:
            raise ValueError("allocation requires per-head K and joint V bases")
        width = linear.out_features // groups
        if width > linear.in_features:
            raise ValueError("full output-width bases require output width <= input width")
        for factor in factors:
            spectrum = factor["alloc_spectrum"]
            if spectrum.shape != (width,):
                raise ValueError("allocation needs a full, untruncated spectrum")
            if not bool(torch.isfinite(spectrum).all()) or bool((spectrum < 0).any()):
                raise ValueError("invalid allocation spectrum")
            if bool((spectrum[1:] > spectrum[:-1]).any()):
                raise ValueError("spectrum must be in descending order")
            if factor.get("parameter_space") == "full_weight":
                shapes = {"down": (linear.in_features, width), "up": (width, width)}
            elif factor.get("parameter_space") == "key_space":
                shapes = {"enc": (width, width), "dec": (width, width)}
            else:
                raise ValueError("unknown allocation basis parameter space")
            for key, shape in shapes.items():
                value = factor[key]
                if value.shape != shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"invalid allocation basis {name}/{key}")


@torch.no_grad()
def projection_factors(linear, factors, ranks, *, dtype=torch.float16):
    """Rank prefixes of one projection's groups: (VS [sum r, in], block-diagonal U [out, sum r])."""
    if len(ranks) != len(factors):
        raise ValueError("selected rank groups do not match bases")
    downs, ups = [], []
    for weight, factor, rank in zip(linear.weight.detach().chunk(len(factors)), factors, ranks, strict=True):
        capacity = len(factor["alloc_spectrum"])
        if type(rank) is not int or not 1 <= rank <= capacity:
            raise ValueError("selected rank is outside basis capacity")
        if factor["parameter_space"] == "full_weight":
            down = factor["down"][:, :rank].T
            up = factor["up"][:rank].T
        else:
            down = factor["enc"][:, :rank].T.float() @ weight.cpu().float()
            up = factor["dec"][:rank].T
        downs.append(down.to(device="cpu", dtype=dtype))
        ups.append(up.to(device="cpu", dtype=dtype))
    return torch.cat(downs), torch.block_diag(*ups)


@torch.no_grad()
def export_factors(model, bases, rank_map, *, budget=None, dtype=torch.float16):
    """Slice original balanced factors; no basis or model weight is trained."""
    validate_bases(model, bases)
    if set(rank_map) != set(bases):
        raise ValueError("selected rank map does not match bases")
    sd, total = {}, 0
    for name, factors in bases.items():
        ranks = rank_map[name]
        sd[name + ".VS.weight"], sd[name + ".U.weight"] = projection_factors(
            model.get_submodule(name), factors, ranks, dtype=dtype)
        total += sum(ranks)
        if name.endswith("k_proj"):
            sd[name + ".head_ranks"] = torch.tensor(ranks, dtype=torch.int32)
    if budget is not None and total != budget:
        raise ValueError("selected ranks violate the exact global budget")
    if any(not bool(torch.isfinite(value).all()) for value in sd.values()):
        raise ArithmeticError("nonfinite exported allocation factors")
    from .arch import attention_layer_indices
    config = model.config
    report = kv_cache_report(sd, len(model.model.layers), config.num_key_value_heads, head_dim_of(config),
                             layers=attention_layer_indices(model))
    return sd, report
