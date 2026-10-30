"""High-level loading: base model and low-rank checkpoint.

    model, tok, info = load_compressed_model(base, ckpt)

The low-rank modules run inside the stock HF attention (full-size KV cache): the
accuracy reference that rank training, fine-tuning and the dense export use.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .checkpoint import LoadInfo, load_lowrank_checkpoint
from .modules import is_lowrank_module

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
           "fp32": torch.float32, "float32": torch.float32}


def parse_dtype(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES[str(name).lower()]


def _transformers_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def gpu_memory_map(devices) -> dict:
    """``max_memory`` for ``from_pretrained``/``dispatch`` that places a model on exactly these CUDA indices."""
    devices = [torch.device("cuda", d) if isinstance(d, int) else torch.device(d) for d in devices]
    if not devices or any(d.type != "cuda" or d.index is None for d in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"expected distinct explicit CUDA devices, got {devices}")
    return {d.index: torch.cuda.get_device_properties(d.index).total_memory for d in devices}


def dispatch_layers(model: nn.Module, devices) -> dict:
    """Spread an already built (CPU) model over ``devices`` by whole decoder layers (naive pipeline).

    Parameters are balanced across the devices, as ``from_pretrained(device_map="auto")``
    does; accelerate hooks move activations between them. Undo with :func:`undispatch`.
    Returns the device map.
    """
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    max_memory = gpu_memory_map(devices)
    no_split = list(getattr(model, "_no_split_modules", None) or [])
    dtype = next(model.parameters()).dtype
    balanced = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split, dtype=dtype)
    device_map = infer_auto_device_map(model, max_memory=balanced, no_split_module_classes=no_split, dtype=dtype)
    if any(not isinstance(v, int) for v in device_map.values()):
        raise RuntimeError(f"model does not fit on {list(max_memory)}: {device_map}")
    dispatch_model(model, device_map)
    return device_map


def undispatch(model: nn.Module) -> nn.Module:
    """Remove the hooks of :func:`dispatch_layers` and bring every tensor back to the CPU."""
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(model, recurse=True)
    if hasattr(model, "hf_device_map"):
        del model.hf_device_map
    return model.cpu()


def load_base_model(model_path: str, dtype=torch.bfloat16, device_map="auto",
                    attn_implementation: Optional[str] = "sdpa", trust_remote_code: bool = False,
                    local_files_only: bool = False, max_memory: Optional[dict] = None):
    """Load a Llama-family (or Qwen3) causal LM and its tokenizer.

    ``max_memory`` (e.g. :func:`gpu_memory_map`) restricts ``device_map="auto"`` to some GPUs.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = parse_dtype(dtype)
    kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code,
                  local_files_only=local_files_only)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if _transformers_major() >= 5:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    # Tokenizer first: it is cheap and it lists the directory, so an unreadable
    # or unlistable model path fails here instead of after minutes of weight loading.
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    # A VLM checkpoint loaded as its language model (Qwen3.8: text_config.dtype bfloat16) ignores the
    # requested dtype under transformers 5.9; cast so every caller gets the precision it asked for.
    if any(p.dtype != dtype for p in model.parameters() if p.is_floating_point()):
        model.to(dtype)
    _assert_llama_layout(model)
    return model, tokenizer


def _assert_llama_layout(model: nn.Module) -> None:
    """Llama layout; a hybrid stack (Qwen3.5) may interleave layers without self_attn (lrkv.arch)."""
    ok = hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "rotary_emb")
    if ok:
        from .arch import attention_layer_indices
        indices = attention_layer_indices(model)
        ok = bool(indices) and all(hasattr(model.model.layers[i].self_attn, n) for i in indices
                                   for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if not ok:
        raise TypeError(f"{type(model).__name__} does not have the Llama layout "
                        "(model.model.layers[i].self_attn.{q,k,v,o}_proj and model.model.rotary_emb)")


def enable_cache(model: nn.Module) -> None:
    """Turn the KV cache on for both forward and generate.

    Loading with ``use_cache=False`` and then flipping ``config.use_cache`` leaves
    ``generation_config.use_cache`` False under transformers 5.x, so a plain
    ``generate()`` silently recomputes the whole sequence every step.
    """
    model.config.use_cache = True
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.use_cache = True


def describe_ranks(model: nn.Module) -> dict:
    """{layer: {"k_head_ranks": [...], "v_rank": r}} for every low-rank layer."""
    out = {}
    for i, block in enumerate(model.model.layers):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        if is_lowrank_module(attn.k_proj) or is_lowrank_module(attn.v_proj):
            out[i] = {
                "k_head_ranks": attn.k_proj.head_ranks_list if is_lowrank_module(attn.k_proj) else None,
                "v_rank": attn.v_proj.rank if is_lowrank_module(attn.v_proj) else None,
            }
    return out


def load_compressed_model(
    model_path: str,
    checkpoint,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation: Optional[str] = "sdpa",
    verbose: bool = True,
    local_files_only: bool = False,
):
    """Base model + low-rank checkpoint, low-rank modules inside the stock attention.

    ``checkpoint`` may be a path or a state dict.
    Returns (model, tokenizer, LoadInfo).
    """
    dtype = parse_dtype(dtype)
    model, tokenizer = load_base_model(model_path, dtype=dtype, device_map=device_map,
                                       attn_implementation=attn_implementation,
                                       local_files_only=local_files_only)
    info: LoadInfo = load_lowrank_checkpoint(model, checkpoint)
    if verbose:
        n = len(info.compressed)
        print(f"  checkpoint: {n} compressed layers, uncompressed {list(info.skip_layers)}"
              f", {info.overridden_dense_keys} dense tensors overridden")
    model.eval()
    enable_cache(model)
    return model, tokenizer, info
