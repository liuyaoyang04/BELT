"""Build fixed-rank or full-basis K/V factors from an uncompressed local Qwen3.8-27B model.

    CUDA_VISIBLE_DEVICES=1 python -m lrkv.ours --model models/Qwen3.8-27B \
        --experiment-data data/experiment/qwen3_8_27b_docs_c512x8192_t8192.pt \
        --output checkpoints/ours_qwen3_8_27b_uniform.pt

K is fitted per KV head in closed form using the behavior-aligned Kronecker
curvature surrogate (``kronecker_surrogate``, FP16-rounded prior probabilities,
``k_norm`` Jacobian metric, partial-RoPE padding); V is fitted jointly per
full-attention layer via gated weighted SVD (``behavior_svd``, one shared latent
per layer across all KV heads, weighted by the second moment of the attention
output gate). Linear-attention layers in hybrid stacks have no KV cache and are
propagated by the dense teacher without factorization.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch
from torch import nn

from .arch import (
    ATTENTION,
    attention_layer_indices,
    gate_sigmoid,
    has_output_gate,
    layer_kind,
    pad_rope_tables,
    rms_norm_eps,
    rms_norm_gain,
    rotary_dim,
)
from .checkpoint import head_dim_of, layer_prefix, save_state_dict
from .experiment_data import _sha256, configure_offline
from .layer_parallel import check_deadline, validate_layer_devices
from .model import load_base_model
from .whitened import behavior_svd_closed_form, fit_key_factors, fit_value_factors, value_metrics


def _validate_calibration(payload) -> list[dict[str, torch.Tensor]]:
    batches = payload.get("batches") if isinstance(payload, dict) else payload
    if not isinstance(batches, list) or not batches:
        raise ValueError("calibration artifact must contain a nonempty batch list")
    sequence_length = None
    cleaned = []
    for batch in batches:
        if not isinstance(batch, dict) or "input_ids" not in batch:
            raise ValueError("every calibration batch must contain input_ids")
        input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("calibration batches must have shape (1, sequence_length)")
        sequence_length = sequence_length or input_ids.shape[1]
        if input_ids.shape[1] != sequence_length:
            raise ValueError("calibration batches must share one sequence length")
        attention_mask = torch.as_tensor(
            batch.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long
        )
        if (
            input_ids.shape[1] < 2
            or attention_mask.shape != input_ids.shape
            or not bool((attention_mask == 1).all())
        ):
            raise ValueError("calibration requires unpadded all-one attention masks")
        cleaned.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return cleaned


def query_indices(length, count, device):
    if count < 1:
        raise ValueError("query count must be positive")
    if count >= length:
        return torch.arange(length, device=device)
    return torch.unique(torch.linspace(0, length - 1, count, device=device).round().long(), sorted=True)


def move_cache(cache, device):
    return {
        key: (
            move_cache(value, device)
            if isinstance(value, dict)
            else tuple(x.to(device) for x in value)
            if isinstance(value, tuple)
            else value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in cache.items()
    }


class _Captured(Exception):
    pass


def key_norm(attn):
    """The per-head RMSNorm between k_proj and RoPE (``k_norm``), or None."""
    norm = getattr(attn, "k_norm", None)
    if norm is None:
        if getattr(attn, "q_norm", None) is not None:
            raise ValueError("q_norm without k_norm is not supported")
        return None
    if getattr(attn, "q_norm", None) is None:
        raise ValueError("k_norm without q_norm is not supported")
    return norm


def unit_rms_keys(keys, norm):
    """k_proj outputs [..., h] scaled to unit RMS per head, as ``norm`` does before its weight (FP32)."""
    keys = keys.float()
    return keys * torch.rsqrt(keys.square().mean(-1, keepdim=True) + rms_norm_eps(norm))


@torch.inference_mode()
def capture_layer_inputs(model, batches, device):
    """Intercept layer zero so only embeddings/one layer need GPU residency.

    Returns the layer-0 inputs, an output buffer, and the keyword arguments of every
    layer kind, ``{kind: kwargs}`` (``lrkv.arch.layer_kind``). A hybrid stack passes
    its linear-attention layers a different attention mask than its full-attention
    layers, so each kind's arguments are captured at its first layer with preceding
    layers bypassed.
    """
    dtype = next(model.parameters()).dtype
    inputs = torch.empty(
        len(batches), batches[0]["input_ids"].shape[1], model.config.hidden_size, dtype=dtype
    )
    layers = model.model.layers
    kinds = [layer_kind(layer) for layer in layers]
    names = ("attention_mask", "position_ids", "cache_position", "position_embeddings")
    cache, index = {}, 0

    class Catcher(nn.Module):
        def __init__(self, kind, record):
            super().__init__()
            self.kind, self.record = kind, record

        def forward(self, hidden_states, **kwargs):
            nonlocal index
            if self.record:
                inputs[index].copy_(hidden_states[0].detach().cpu())
                index += 1
            cache.setdefault(self.kind, {k: kwargs.get(k) for k in names})
            raise _Captured

    class Bypass(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    def run(batch):
        try:
            model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
        except _Captured:
            pass

    originals = list(layers)
    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    try:
        layers[0] = Catcher(kinds[0], True)
        for batch in batches:
            run(batch)
        for first in sorted(kinds.index(kind) for kind in set(kinds) if kind != kinds[0]):
            for j in range(first):
                layers[j] = Bypass()
            layers[first] = Catcher(kinds[first], False)
            run(batches[0])
    finally:
        for j, layer in enumerate(originals):
            layers[j] = layer
        model.model.embed_tokens.cpu()
        model.model.rotary_emb.cpu()
    if (
        index != len(batches)
        or set(cache) != set(kinds)
        or any(c.get("position_embeddings") is None for c in cache.values())
    ):
        raise RuntimeError("failed to capture teacher inputs and rotary tables")
    return inputs, torch.empty_like(inputs), move_cache(cache, device)


def forward_layer(layer, hidden, cache):
    """One dense teacher layer on one row, with the keyword arguments of its kind."""
    kwargs = cache[layer_kind(layer)] if isinstance(cache.get(layer_kind(layer)), dict) else cache
    result = layer(
        hidden.unsqueeze(0),
        **{k: v for k, v in kwargs.items() if v is not None},
        use_cache=False,
        output_attentions=True,
    )
    return result if isinstance(result, torch.Tensor) else result[0]


@torch.inference_mode()
def capture_state(layer, inputs, outputs, cache, count, device, *, deadline=None):
    indices = query_indices(inputs.shape[1], count, device)
    state = {k: [] for k in ("queries", "keys", "values", "attention")}

    def query_hook(_module, _args, output):
        state["queries"].append(output[:, indices].detach().float().cpu())

    def key_hook(_module, _args, output):
        state["keys"].append(output.detach().cpu())

    def value_hook(_module, _args, output):
        state["values"].append(output.detach().float().cpu())

    def attention_hook(_module, _args, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("ours requires eager teacher attention weights")
        state["attention"].append(output[1][:, :, indices].detach().half().cpu())

    config = layer.self_attn.config
    gated = has_output_gate(layer.self_attn)
    if gated:
        state["gates"] = []

    def gate_hook(_module, _args, output):
        state["gates"].append(
            gate_sigmoid(
                output[:, indices].detach(),
                int(config.num_attention_heads),
                head_dim_of(config),
            ).cpu()
        )

    query_source = layer.self_attn.q_proj if key_norm(layer.self_attn) is None else layer.self_attn.q_norm
    handles = [
        query_source.register_forward_hook(query_hook),
        layer.self_attn.k_proj.register_forward_hook(key_hook),
        layer.self_attn.v_proj.register_forward_hook(value_hook),
        layer.self_attn.register_forward_hook(attention_hook),
    ]
    if gated:
        handles.append(layer.self_attn.q_proj.register_forward_hook(gate_hook))
    try:
        for sample in range(inputs.shape[0]):
            check_deadline(deadline)
            hidden = forward_layer(layer, inputs[sample].to(device), cache)
            if outputs is not None:
                outputs[sample].copy_(hidden[0].cpu())
    finally:
        for handle in handles:
            handle.remove()
    return {k: torch.cat(v) for k, v in state.items()}, indices


@torch.inference_mode()
def propagate_layer(layer, inputs, outputs, cache, device, *, deadline=None):
    """Dense teacher forward only; bitwise equal to the outputs capture_state writes."""
    for sample in range(inputs.shape[0]):
        check_deadline(deadline)
        outputs[sample].copy_(forward_layer(layer, inputs[sample].to(device), cache)[0].cpu())


def deploy_tables(model, device, deploy_length):
    """Deployment RoPE grid: 4096 positions spread over the deployment length."""
    positions = torch.linspace(0, deploy_length - 1, 4096, device=device).round().long()[None]
    model.model.rotary_emb.to(device)
    try:
        cos, sin = (
            x[0].float() for x in model.model.rotary_emb(torch.zeros(1, 1, device=device), positions)
        )
    finally:
        model.model.rotary_emb.cpu()
    return pad_rope_tables(cos, sin, head_dim_of(model.config))


def key_rope_tables(cache, config):
    """The captured attention-layer rotary tables, padded to head_dim for the key curvature."""
    kwargs = cache[ATTENTION] if isinstance(cache.get(ATTENTION), dict) else cache
    cos, sin = kwargs["position_embeddings"]
    return pad_rope_tables(cos, sin, head_dim_of(config))


def _key_views(layer, state):
    config = layer.self_attn.config
    h, kv = head_dim_of(config), int(config.num_key_value_heads)
    b, n = state["keys"].shape[:2]
    keys = state["keys"].reshape(b, n, kv, h)
    queries = state["queries"].reshape(b, -1, kv, config.num_attention_heads // kv, h)
    weight = layer.self_attn.k_proj.weight.detach().reshape(kv, h, -1)
    return keys, queries, weight


def _layer_factors(layer, state, *, index, device, key_fit, kr, vr, full_basis, deadline):
    """K heads from key_fit(head) and the joint gated V of one captured layer (on device)."""
    gates = state.get("gates")
    gates = None if gates is None else gates.to(device)
    config = layer.self_attn.config
    h, kv = head_dim_of(config), int(config.num_key_value_heads)
    sd, bases = {}, {}
    down, up, checks, key_bases = [], [], [], []
    for head in range(kv):
        check_deadline(deadline)
        factors = key_fit(head)
        down.append(factors.down[:, :kr].T.to(device).half().cpu())
        up.append(factors.up[:kr].T.to(device).half().cpu())
        checks.append(factors.audit)
        if full_basis:
            key_bases.append(
                {
                    "parameter_space": "full_weight",
                    "down": factors.down.cpu(),
                    "up": factors.up.cpu(),
                    "alloc_spectrum": factors.spectrum.cpu(),
                }
            )
        del factors
    prefix = layer_prefix(index)
    sd[prefix + "k_proj.VS.weight"] = torch.cat(down)
    sd[prefix + "k_proj.U.weight"] = torch.block_diag(*up)
    sd[prefix + "k_proj.head_ranks"] = torch.full((kv,), kr, dtype=torch.int32)
    if full_basis:
        row, col = value_metrics(
            state["attention"].to(device),
            state["values"].to(device),
            layer.self_attn.o_proj.weight,
            kv,
            h,
            gates,
        )
        encoder, decoder, spectrum = behavior_svd_closed_form(row, col)
        bases[prefix + "k_proj"] = key_bases
        bases[prefix + "v_proj"] = [
            {
                "parameter_space": "key_space",
                "enc": encoder.cpu(),
                "dec": decoder.cpu(),
                "alloc_spectrum": spectrum.cpu(),
            }
        ]
        vs = encoder[:, :vr].T @ layer.self_attn.v_proj.weight.float()
        u = decoder[:vr].T
        del row, col, encoder, decoder, spectrum
    else:
        vs, u = fit_value_factors(
            state["attention"].to(device),
            state["values"].to(device),
            layer.self_attn.v_proj.weight,
            layer.self_attn.o_proj.weight,
            kv,
            h,
            vr,
            gates,
        )
    sd[prefix + "v_proj.VS.weight"] = vs.half().cpu()
    sd[prefix + "v_proj.U.weight"] = u.half().cpu()
    if any(not bool(torch.isfinite(v).all()) for v in sd.values()):
        raise ArithmeticError(f"nonfinite exported factors in layer {index}")
    return sd, bases, checks


def _whitened_layer(
    layer,
    inputs,
    outputs,
    cache,
    deploy,
    *,
    index,
    device,
    kr,
    vr,
    full_basis,
    query_tokens,
    query_chunk,
    deadline=None,
):
    """Capture one dense teacher layer (already on device) and fit its Kronecker-surrogate K heads and joint V."""
    state, indices = capture_state(layer, inputs, outputs, cache, query_tokens, device, deadline=deadline)
    keys, queries, weight = _key_views(layer, state)
    norm = key_norm(layer.self_attn)
    norm_weight = None
    if norm is not None:
        keys = unit_rms_keys(keys, norm)
        norm_weight = rms_norm_gain(norm).to(device)
    config = layer.self_attn.config
    cos, sin = key_rope_tables(cache, config)

    def key_fit(head):
        return fit_key_factors(
            keys[:, :, head],
            queries[:, :, head],
            cos,
            sin,
            indices,
            deploy,
            weight[head].T,
            query_chunk=query_chunk,
            key_norm_weight=norm_weight,
            rotary_dim=rotary_dim(config),
        )

    return _layer_factors(
        layer,
        state,
        index=index,
        device=device,
        key_fit=key_fit,
        kr=kr,
        vr=vr,
        full_basis=full_basis,
        deadline=deadline,
    )


def default_deploy_length(_config=None):
    """Default RoPE prior grid extent (32k context horizon)."""
    return 32768


@torch.inference_mode()
def factorize_model(
    model,
    batches,
    *,
    device="cuda",
    k_rank=None,
    v_rank=None,
    query_tokens=64,
    query_chunk=64,
    deploy_length=None,
    progress=print,
    full_basis=False,
    deadline=None,
    layer_devices=None,
):
    """Return canonical factors (or full allocation bases) and construction metadata."""
    device = torch.device(device)
    layer_devices = validate_layer_devices(layer_devices, teacher_device=device)
    check_deadline(deadline)
    if full_basis and (k_rank is not None or v_rank is not None):
        raise ValueError("full allocation bases cannot specify truncated ranks")
    batches = _validate_calibration(batches)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("load the teacher on CPU; layers are staged individually")
    if next(model.parameters()).dtype != torch.float16:
        raise ValueError("ours pins the teacher to torch.float16")
    if model.config._attn_implementation != "eager":
        raise ValueError("ours requires eager teacher attention")
    if query_tokens < 1 or query_chunk < 1:
        raise ValueError("invalid query configuration")
    attention = attention_layer_indices(model)
    attention_modules = [model.model.layers[i].self_attn for i in attention]
    normed = any(key_norm(attn) is not None for attn in attention_modules)
    gated = any(has_output_gate(attn) for attn in attention_modules)
    config = model.config
    h, kv = head_dim_of(config), int(config.num_key_value_heads)
    if config.num_attention_heads % kv or h % 2:
        raise ValueError("invalid GQA / rotary layout")
    kr = h // 2 if k_rank is None else k_rank
    vr = kv * h // 2 if v_rank is None else v_rank
    if not 1 <= kr <= h or not 1 <= vr <= kv * h:
        raise ValueError("rank exceeds projection dimensions")
    deploy_length = default_deploy_length(config) if deploy_length is None else deploy_length
    if deploy_length < 1:
        raise ValueError("deployment length must be positive")
    for attn in attention_modules:
        if attn.k_proj.bias is not None or attn.v_proj.bias is not None:
            raise ValueError("ours canonical export requires bias-free K/V projections")
    was_training = model.training
    model.eval()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    sd, audits, bases, assignment = {}, [], {}, None
    try:
        if layer_devices:
            output, audits, assignment = _parallel_whitened_layers(
                model,
                batches,
                device=device,
                devices=layer_devices,
                deploy_length=deploy_length,
                deadline=deadline,
                progress=progress,
                kr=kr,
                vr=vr,
                full_basis=full_basis,
                query_tokens=query_tokens,
                query_chunk=query_chunk,
            )
            if full_basis:
                bases = output
            else:
                sd = output
        else:
            inputs, outputs, cache = capture_layer_inputs(model, batches, device)
            deploy = deploy_tables(model, device, deploy_length)
            for index, layer in enumerate(model.model.layers):
                check_deadline(deadline)
                started = time.monotonic()
                layer.to(device)
                try:
                    if layer_kind(layer) != ATTENTION:
                        propagate_layer(layer, inputs, outputs, cache, device, deadline=deadline)
                        inputs, outputs = outputs, inputs
                        continue
                    layer_sd, layer_bases, checks = _whitened_layer(
                        layer,
                        inputs,
                        outputs,
                        cache,
                        deploy,
                        index=index,
                        device=device,
                        kr=kr,
                        vr=vr,
                        full_basis=full_basis,
                        query_tokens=query_tokens,
                        query_chunk=query_chunk,
                        deadline=deadline,
                    )
                    inputs, outputs = outputs, inputs
                    sd.update(layer_sd)
                    bases.update(layer_bases)
                    elapsed = time.monotonic() - started
                    audits.append({"layer": index, "k_heads": checks, "seconds": elapsed})
                    layout = "full allocation bases" if full_basis else f"K={kr}/head, V={vr}"
                    progress(f"ours layer {index + 1}/{len(model.model.layers)}: {layout}, {elapsed:.1f}s")
                    del layer_sd, layer_bases, checks
                finally:
                    layer.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        model.model.rotary_emb.cpu()
        model.train(was_training)
    flat = [
        {"layer": a["layer"], "head": h, **c["flat_curvature_fallback"]}
        for a in audits
        for h, c in enumerate(a["k_heads"])
        if "flat_curvature_fallback" in c
    ]
    if flat:
        progress(
            f"kronecker_surrogate: {len(flat)} K head(s) with flat teacher curvature kept the attention prior: "
            + ", ".join(f"L{f['layer']}/H{f['head']}" for f in flat)
        )
    construction = {
        "method": "ours",
        "key_method": "kronecker_surrogate",
        "value_method": "behavior_svd",
        "k_rank_per_head": None if full_basis else kr,
        "v_rank": None if full_basis else vr,
        "full_basis": full_basis,
        "skip_layers": [],
        "key_norm_metric": "jacobian" if normed else None,
        "flat_curvature_prior_fallback": flat,
        "position_encoding": "rope",
        "query_tokens": query_tokens,
        "query_chunk": query_chunk,
        "deploy_length": deploy_length,
        "deploy_grid": 4096,
        "prior_attention": "fp16",
        "row_floor": 0.05,
        "ridge": 1e-4,
        "teacher_dtype": "float16",
        "teacher_attention": "eager",
        "curvature_dtype": "float32",
        "attention_layers": attention,
        "linear_attention_layers": len(model.model.layers) - len(attention),
        "rotary_dim": rotary_dim(config),
        "head_dim": h,
        "value_output_gate": ("second_moment" if gated else None),
        "decomposition_dtype": "float64",
        "factor_dtype": "K:float64,V:float32" if full_basis else "float16",
        "allow_tf32": False,
        "iterations": 0,
        "post_fit_loss_audit": False,
        "layers": audits,
        "layer_execution": {
            "mode": "spawn_per_device" if layer_devices else "serial",
            "teacher_device": str(device),
            "layer_devices": list(layer_devices),
            "scheduling": "next_idle_worker" if layer_devices else "layer_order",
            "layer_device_assignment": assignment,
            "position_mass_accumulation": "fixed_order",
        },
    }
    return (bases if full_basis else sd), construction


def write_checkpoint(sd, metadata, output):
    """Atomic factor export with a read-back check and a provenance sidecar."""
    output = Path(output)
    manifest = output.with_suffix(output.suffix + ".json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"output already exists: {output} or {manifest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    meta_tmp = manifest.with_name(manifest.name + f".{os.getpid()}.tmp")
    try:
        save_state_dict(sd, str(temporary))
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if loaded.keys() != sd.keys() or any(not torch.equal(v, loaded[k]) for k, v in sd.items()):
            raise IOError("checkpoint read-back differs from fitted factors")
        metadata = {**metadata, "checkpoint_sha256": _sha256(temporary), "format": "lrkv_canonical_factors"}
        meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, output)
        os.replace(meta_tmp, manifest)
    finally:
        temporary.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)


@torch.inference_mode()
def _parallel_whitened_layers(
    model, batches, *, device, devices, deploy_length, deadline, progress, **options
):
    """Dense teacher propagation on ``device``; each layer's capture and fit on the next idle layer worker."""
    from .layer_parallel import LayerPool, LayerTask

    count = len(model.model.layers)
    attention = attention_layer_indices(model)
    inputs, spare, cache = capture_layer_inputs(model, batches, device)
    del spare
    deploy = deploy_tables(model, device, deploy_length)
    cpu_cache, cpu_deploy = move_cache(cache, "cpu"), tuple(t.cpu() for t in deploy)
    del deploy
    layout = "full allocation bases" if options["full_basis"] else f"K={options['kr']}/head, V={options['vr']}"
    results = {}
    with LayerPool(devices, options=options, deadline=deadline) as pool:

        def finish():
            index, worker_device, layer_sd, layer_bases, checks, seconds = pool.wait()
            results[index] = (
                layer_sd,
                layer_bases,
                {"layer": index, "k_heads": checks, "seconds": seconds},
                worker_device,
            )
            progress(
                f"ours layer {index + 1}/{count} on {worker_device}: {layout}, {seconds:.1f}s "
                f"({len(results)}/{len(attention)} done)"
            )

        for index, layer in enumerate(model.model.layers):
            check_deadline(deadline)
            if layer_kind(layer) == ATTENTION:
                while not pool.idle():
                    finish()
                with torch.inference_mode(False):
                    snapshot = copy.deepcopy(layer).requires_grad_(False)
                pool.submit(LayerTask(index, snapshot, inputs, cpu_cache, cpu_deploy))
                del snapshot
            if index + 1 < count:
                layer.to(device)
                try:
                    outputs = torch.empty_like(inputs)
                    propagate_layer(layer, inputs, outputs, cache, device, deadline=deadline)
                finally:
                    layer.cpu()
                inputs = outputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del inputs
        while pool.busy():
            finish()
    check_deadline(deadline)
    artifacts, audits, assignment = {}, [], []
    for index in attention:
        layer_sd, layer_bases, audit, worker_device = results.pop(index)
        artifacts.update(layer_bases if options["full_basis"] else layer_sd)
        audits.append(audit)
        assignment.append(worker_device)
    return artifacts, audits, assignment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local uncompressed Qwen3.8-27B model directory")
    parser.add_argument("--output", required=True, help="canonical factor checkpoint (.pt)")
    from .experiment_data import add_argument

    add_argument(parser)
    parser.add_argument("--device", default="cuda", help="visible device; set CUDA_VISIBLE_DEVICES externally")
    parser.add_argument(
        "--layer-devices",
        nargs="+",
        metavar="cuda:N",
        help="fit whole layers on spawned workers across logical CUDA devices while --device propagates the teacher",
    )
    parser.add_argument("--k-rank", type=int, help="default: half of head_dim")
    parser.add_argument("--v-rank", type=int, help="default: half of joint V width")
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--deploy-length", type=int, help="prior RoPE grid extent (default: 32768)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_offline()
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {args.model}")
    if Path(args.output).exists() or Path(args.output + ".json").exists():
        raise FileExistsError(f"output already exists: {args.output}")
    torch.set_num_threads(4)
    torch.manual_seed(3)
    model, _ = load_base_model(args.model, dtype=torch.float16, device_map="cpu", attn_implementation="eager")
    from . import experiment_data

    batches, calibration = experiment_data.calibration(args.experiment_data, args.model)
    sd, metadata = factorize_model(
        model,
        batches,
        device=args.device,
        k_rank=args.k_rank,
        v_rank=args.v_rank,
        query_tokens=args.query_tokens,
        query_chunk=args.query_chunk,
        deploy_length=args.deploy_length,
        layer_devices=args.layer_devices,
    )
    import transformers

    metadata.update(
        model=str(Path(args.model).resolve()),
        model_config=model.config.to_dict(),
        calibration=calibration,
        base_config_sha256=_sha256(Path(args.model) / "config.json"),
        device=args.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        versions={"torch": str(torch.__version__), "transformers": transformers.__version__},
        source_sha256={
            name: _sha256(Path(__file__).with_name(name))
            for name in ("ours.py", "whitened.py", "layer_parallel.py", "experiment_data.py", "arch.py")
        },
    )
    write_checkpoint(sd, metadata, args.output)
    print(f"wrote {args.output} and {args.output}.json")


if __name__ == "__main__":
    main()
