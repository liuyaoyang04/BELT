"""Offline nonquantized rank allocation: basis -> train -> export.

`basis` builds full frozen K/V bases (`kronecker_surrogate` per KV head for K and
joint gated `behavior_svd` per full-attention layer for V) from the shared
experiment token artifact (`lrkv.experiment_data`). `train` learns per-block
rank cutoffs under an exact KV-cache budget with output KD and a
warmup-stable-decay (WSD) schedule. `export` slices the selected ranks into a
canonical factor-only checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .allocation import build_block_specs, install_belt_modules, rank_summary, selected_rank_map
from .basis import export_factors, validate_bases
from .experiment_data import _sha256, _tokenizer_sha256, configure_offline
from .layer_parallel import check_deadline, validate_layer_devices
from .model import load_base_model
from .ours import factorize_model, write_checkpoint

SEED = 20260805
PROTOCOL_WSD = "output-kd-exact-budget-rank-unit-wsd-v1"
BASIS_FORMAT = "lrkv_full_allocation_basis_v1"
ALLOCATION_FORMAT = "lrkv_rank_allocation_v1"


def _last_hidden(model, input_ids: torch.Tensor, *, belt_session: bool):
    selector = getattr(model, "_belt_selector", None)
    if belt_session:
        selector.begin_forward()
    try:
        return model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    finally:
        if belt_session:
            selector.end_forward()


@torch.no_grad()
def _teacher_hidden(teacher, input_ids: torch.Tensor) -> torch.Tensor:
    """Frozen-teacher last hidden states at the positions that predict a next token."""
    ids = input_ids.to(teacher.get_input_embeddings().weight.device, dtype=torch.long).unsqueeze(0)
    return _last_hidden(teacher, ids, belt_session=False)[:, :-1]


def _chunked_kd(
    student,
    teacher,
    input_ids: torch.Tensor,
    *,
    backward: bool,
    loss_scale: float = 1.0,
    token_chunk: int = 2048,
    teacher_hidden: torch.Tensor | None = None,
) -> float:
    """Output KD of one sequence.

    ``teacher_hidden`` supplies precomputed teacher hidden states (a teacher
    split); the teacher logits then come from the student's ``lm_head``, which
    is the same frozen matrix, and ``teacher`` may be None.
    """
    if input_ids.ndim != 1 or input_ids.numel() < 2 or token_chunk < 1:
        raise ValueError("KD requires a token sequence and a positive chunk size")
    student_device = student.get_input_embeddings().weight.device
    student_ids = input_ids.to(student_device, dtype=torch.long).unsqueeze(0)
    belt_session = getattr(student, "_belt_selector", None) is not None
    if teacher_hidden is None:
        teacher_hidden = _teacher_hidden(teacher, input_ids)
        teacher_head, config = teacher.lm_head, teacher.config
    else:
        teacher_head, config = student.lm_head, student.config
    if backward:
        student_hidden = _last_hidden(
            student, student_ids, belt_session=belt_session
        )[:, :-1]
    else:
        with torch.no_grad():
            student_hidden = _last_hidden(
                student, student_ids, belt_session=belt_session
            )[:, :-1]

    eos_ids = _eos_ids(config, getattr(student, "generation_config", None))
    valid = ~torch.isin(
        input_ids[1:], torch.tensor(eos_ids, dtype=input_ids.dtype, device=input_ids.device)
    )
    token_count = int(valid.sum())
    if token_count == 0:
        raise ValueError("KD block has no supervised tokens")
    teacher_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])[
        valid.to(teacher_hidden.device)
    ]
    student_base = student_hidden.detach().reshape(
        -1, student_hidden.shape[-1]
    )
    student_flat = student_base[valid.to(student_hidden.device)]
    if backward:
        student_flat = student_flat.detach().requires_grad_(True)
    total = 0.0
    for start in range(0, token_count, token_chunk):
        end = min(start + token_chunk, token_count)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_flat[start:end])
        if backward:
            student_logits = student.lm_head(student_flat[start:end])
            aligned_teacher = teacher_logits.to(student_logits.device)
            chunk = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(aligned_teacher, dim=-1),
                reduction="sum",
            )
            (chunk * (loss_scale / token_count)).backward()
        else:
            with torch.no_grad():
                student_logits = student.lm_head(student_flat[start:end])
                aligned_teacher = teacher_logits.to(student_logits.device)
                chunk = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(aligned_teacher, dim=-1),
                    reduction="sum",
                )
        total += float(chunk.detach())
    if backward:
        if student_flat.grad is None:
            raise RuntimeError("output KD produced no hidden-state gradient")
        hidden_gradient = torch.zeros_like(student_base)
        hidden_gradient[valid.to(student_hidden.device)] = student_flat.grad
        student_hidden.backward(hidden_gradient.reshape_as(student_hidden))
    return total / token_count


def _eos_ids(config, generation_config=None):
    """EOS ids of the model config and its generation config."""
    ids = []
    for source in (config, generation_config):
        value = getattr(source, "eos_token_id", None) if source is not None else None
        ids += [value] if isinstance(value, int) else list(value or [])
    return sorted(set(ids))


@torch.no_grad()
def evaluate_kd(
    student,
    teacher,
    rows,
    *,
    process_rank=0,
    world_size=1,
    token_chunk=2048,
    teacher_hidden=None,
    group=None,
):
    """Mean validation KD; ``teacher_hidden`` maps this rank's row indices to cached teacher states."""
    student.eval()
    values = torch.zeros(len(rows), device=next(student.parameters()).device, dtype=torch.float64)
    for index in range(process_rank, len(rows), world_size):
        values[index] = _chunked_kd(
            student,
            teacher,
            rows[index],
            backward=False,
            token_chunk=token_chunk,
            teacher_hidden=None if teacher_hidden is None else teacher_hidden[index],
        )
    if world_size > 1:
        dist.all_reduce(values, group=group)
    if not bool(torch.isfinite(values).all()):
        raise ArithmeticError("nonfinite validation KD")
    return {"values": values.cpu().tolist(), "mean": float(values.mean())}


def wsd_learning_rate(update, peak, warmup_updates, decay_start, decay_updates):
    """Warmup-stable-decay rate for ``update`` (1-based)."""
    if warmup_updates and update <= warmup_updates:
        return peak * update / warmup_updates
    if decay_start is not None and update > decay_start:
        remaining = decay_updates - (update - decay_start) + 1
        return peak * max(remaining, 0) / decay_updates
    return peak


def split_student_group(students):
    """Collective over all ranks: the process group of the ``students`` training ranks of a teacher split."""
    if dist.get_world_size() != 2 * students or students < 1:
        raise ValueError("a teacher split needs 2 x students ranks, students first")
    return dist.new_group(list(range(students)))


class TeacherSplit:
    """Student side of a teacher split."""

    def __init__(self, students, group, *, hidden_size, dtype, device):
        self.size, self.group, self.rank = students, group, dist.get_rank()
        if not 0 <= self.rank < students or dist.get_world_size() != 2 * students:
            raise ValueError("student ranks come first in a teacher split")
        self.peer = self.rank + students
        self.hidden_size, self.dtype, self.device = hidden_size, dtype, device
        self.validation_hidden = None
        self._flag = torch.zeros((), dtype=torch.int32, device=device)

    def receive(self, length):
        hidden = torch.empty((1, length - 1, self.hidden_size), dtype=self.dtype, device=self.device)
        dist.recv(hidden, src=self.peer)
        return hidden

    def receive_validation(self, rows):
        self.validation_hidden = {
            index: self.receive(rows.shape[1]) for index in range(self.rank, len(rows), self.size)
        }

    def signal(self, stop):
        """Tell every teacher rank whether training ended at this update (one per update)."""
        self._flag.fill_(int(stop))
        dist.broadcast(self._flag, src=0)


@torch.no_grad()
def serve_teacher(teacher, rows, *, train_samples, global_batch, students):
    """Teacher rank of a split: stream hidden states to student ``rank - students`` until it stops."""
    student = dist.get_rank() - students
    if not 0 <= student < students or dist.get_world_size() != 2 * students:
        raise ValueError("teacher ranks follow the student ranks in a teacher split")
    if global_batch % students or train_samples % global_batch or not 0 < train_samples < len(rows):
        raise ValueError("invalid teacher split batch")
    teacher.eval().requires_grad_(False)
    validation = rows[train_samples:]
    for index in range(student, len(validation), students):
        dist.send(_teacher_hidden(teacher, validation[index]).contiguous(), dst=student)
    offsets = range(student, global_batch, students)
    ahead = lambda update: [
        _teacher_hidden(teacher, rows[(update - 1) * global_batch + offset]).contiguous()
        for offset in offsets
    ]
    flag = torch.zeros((), dtype=torch.int32, device=teacher.get_input_embeddings().weight.device)
    updates = train_samples // global_batch
    pending = ahead(1)
    for update in range(1, updates + 1):
        for hidden in pending:
            dist.send(hidden, dst=student)
        pending = ahead(update + 1) if update < updates else []
        dist.broadcast(flag, src=0)
        if flag.item():
            break


def train_selector(
    student,
    teacher,
    rows,
    *,
    train_samples=4000,
    global_batch=4,
    learning_rate=2.0,
    validation_interval=25,
    patience=3,
    min_delta=0.0,
    min_updates=50,
    token_chunk=2048,
    progress=print,
    min_delta_relative=0.02,
    warmup_updates=10,
    stable_updates=300,
    decay_updates=50,
    split=None,
):
    """Train cutoff shifts in rank units using output KD and warmup-stable-decay (WSD)."""
    if rows.ndim != 2 or not 0 < train_samples < len(rows) or rows.shape[1] < 2:
        raise ValueError("training data needs separate train and validation token rows")
    if (
        global_batch < 1
        or train_samples % global_batch
        or validation_interval < 1
        or patience < 1
        or min_updates < 0
        or token_chunk < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(min_delta)
        or min_delta < 0
    ):
        raise ValueError("invalid training/early stopping configuration")
    if not (math.isfinite(min_delta_relative) and 0 <= min_delta_relative < 1):
        raise ValueError("min_delta_relative must lie in [0, 1)")
    if warmup_updates < 0 or decay_updates < 1 or (stable_updates is not None and stable_updates < 1):
        raise ValueError("invalid warmup/stable/decay configuration")
    if split is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        process_rank = dist.get_rank() if dist.is_initialized() else 0
        reduce_group = cached = None
        teacher.eval().requires_grad_(False)
    else:
        if teacher is not None or split.validation_hidden is None:
            raise ValueError("a teacher split trains without a local teacher, after receive_validation")
        world_size, process_rank, reduce_group, cached = (
            split.size,
            split.rank,
            split.group,
            split.validation_hidden,
        )
    if global_batch % world_size:
        raise ValueError("global batch must be divisible by world size")
    selector = student._belt_selector
    if any(p.requires_grad for p in student.parameters() if p is not selector.block_shifts):
        raise ValueError("rank allocation may train only selector shifts")
    evaluate = lambda: evaluate_kd(
        student,
        teacher,
        rows[train_samples:],
        process_rank=process_rank,
        world_size=world_size,
        token_chunk=token_chunk,
        teacher_hidden=cached,
        group=reduce_group,
    )
    initial = evaluate()
    best_value = reference = initial["mean"]
    best_shifts = selector.block_shifts.detach().clone()
    best_update = stale = completed = 0
    history = [{"update": 0, **initial}]
    current_lr = learning_rate
    optimizer = torch.optim.AdamW([selector.block_shifts], lr=current_lr, weight_decay=0)
    lr_schedule = [{"update": 0, "learning_rate": current_lr}]
    decay_start = None
    stop_reason = "maximum_updates"
    started = time.monotonic()
    updates = train_samples // global_batch
    for update in range(1, updates + 1):
        current_lr = wsd_learning_rate(update, learning_rate, warmup_updates, decay_start, decay_updates)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        decay_end = decay_start is not None and update >= decay_start + decay_updates
        student.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = selector.block_shifts.new_zeros((), dtype=torch.float64)
        for offset in range(process_rank, global_batch, world_size):
            row = rows[(update - 1) * global_batch + offset]
            hidden = None if split is None else split.receive(row.numel())
            train_loss += (
                _chunked_kd(
                    student,
                    teacher,
                    row,
                    backward=True,
                    loss_scale=1 / global_batch,
                    token_chunk=token_chunk,
                    teacher_hidden=hidden,
                )
                / global_batch
            )
        gradient = selector.block_shifts.grad
        if gradient is None:
            raise RuntimeError("selector gradient is missing")
        if world_size > 1:
            dist.all_reduce(gradient, group=reduce_group)
            dist.all_reduce(train_loss, group=reduce_group)
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(train_loss)):
            raise ArithmeticError("nonfinite rank training loss or gradient")
        optimizer.step()
        completed = update
        stop = False
        if not (update % validation_interval and update != updates and not decay_end):
            validation = evaluate()
            value = validation["mean"]
            if value < best_value:
                best_value, best_update = value, update
                best_shifts = selector.block_shifts.detach().clone()
            if value < reference - min_delta and value < reference * (1 - min_delta_relative):
                reference, stale = value, 0
            else:
                stale += 1
            record = {
                "update": update,
                "train_kd": float(train_loss),
                **validation,
                "best_update": best_update,
                "stale_validations": stale,
                "learning_rate": current_lr,
            }
            history.append(record)
            if process_rank == 0:
                progress(json.dumps(record))
            if decay_end:
                stop_reason, stop = "wsd_decay_complete", True
            elif decay_start is None:
                plateau = update >= min_updates and stale >= patience
                if plateau or (stable_updates is not None and update >= stable_updates):
                    decay_start = update
                    with torch.no_grad():
                        selector.block_shifts.copy_(best_shifts)
                    optimizer = torch.optim.AdamW([selector.block_shifts], lr=current_lr, weight_decay=0)
                    lr_schedule.append(
                        {
                            "update": update,
                            "event": "decay_start",
                            "learning_rate": current_lr,
                            "decay_updates": decay_updates,
                            "resumed_from": best_update,
                        }
                    )
                    if process_rank == 0:
                        progress(
                            json.dumps(
                                {
                                    "update": update,
                                    "event": "decay_start",
                                    "trigger": "plateau" if plateau else "stable_updates",
                                    "resumed_from": best_update,
                                }
                            )
                        )
        if split is not None:
            split.signal(stop)
        if stop:
            break
    with torch.no_grad():
        selector.block_shifts.copy_(best_shifts)
        if world_size > 1:
            dist.broadcast(selector.block_shifts, src=0, group=reduce_group)
    placement = (
        {}
        if split is None
        else {"teacher_split": {"student_ranks": split.size, "teacher_ranks": split.size}}
    )
    return {
        "initial_validation": initial,
        "final_validation": evaluate(),
        "selection": {
            "best_update": best_update,
            "updates_completed": completed,
            "stop_reason": stop_reason,
            "validations": history,
            "wall_seconds": time.monotonic() - started,
        },
        "optimizer": {
            **placement,
            "learning_rate": learning_rate,
            "weight_decay": 0,
            "lr_schedule": lr_schedule,
            "final_learning_rate": current_lr,
            "min_delta_relative": min_delta_relative,
            "schedule": "warmup_stable_decay",
            "warmup_updates": warmup_updates,
            "stable_updates": stable_updates,
            "decay_updates": decay_updates,
            "decay_started_at": decay_start,
            "decay_from_best": True,
            "validation_rows": int(len(rows) - train_samples),
            "temperature": selector.temperature,
            "global_batch": global_batch,
            "world_size": world_size,
            "validation_interval": validation_interval,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "early_stop_min_updates": min_updates,
            "kd_token_chunk": token_chunk,
            "kd_temperature": 1.0,
            "coordinate": "rank_units",
        },
    }


def rank_sha256(rank_map):
    return hashlib.sha256(
        json.dumps(rank_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check_output(path):
    if Path(path).exists():
        raise FileExistsError(path)


def _save(payload, output, *, json_file=False):
    output = Path(output)
    _check_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        if json_file:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        else:
            torch.save(payload, temporary)
            torch.load(temporary, map_location="cpu", weights_only=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(model_path):
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {path}")
    return {
        "model": str(path.resolve()),
        "base_config_sha256": _sha256(path / "config.json"),
        "tokenizer_sha256": _tokenizer_sha256(path),
    }


def _check_identity(payload, identity):
    for key in ("model", "base_config_sha256", "tokenizer_sha256"):
        if payload.get(key) != identity[key]:
            raise ValueError(f"artifact belongs to a different base model/tokenizer: {key}")


def _sources():
    return {
        name: _sha256(Path(__file__).with_name(name))
        for name in (
            "allocate.py",
            "allocation.py",
            "basis.py",
            "ours.py",
            "whitened.py",
            "layer_parallel.py",
            "experiment_data.py",
            "arch.py",
        )
    }


def build_basis(args):
    limit = getattr(args, "max_seconds", None)
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    deadline = None if limit is None else time.monotonic() + limit
    layer_devices = getattr(args, "layer_devices", None)
    validate_layer_devices(layer_devices, teacher_device=args.device)
    _check_output(args.output)
    identity = _identity(args.model)
    model, _ = load_base_model(
        args.model, dtype=torch.float16, device_map="cpu", attn_implementation="eager"
    )
    from . import experiment_data

    batches, calibration = experiment_data.calibration(args.experiment_data, args.model)
    bases, construction = factorize_model(
        model,
        batches,
        device=args.device,
        full_basis=True,
        query_tokens=args.query_tokens,
        query_chunk=args.query_chunk,
        deploy_length=args.deploy_length,
        deadline=deadline,
        layer_devices=layer_devices,
    )
    check_deadline(deadline)
    validate_bases(model, bases)
    _save(
        {
            "format": BASIS_FORMAT,
            **identity,
            "method": construction["method"],
            "bases": bases,
            "construction": construction,
            "calibration": calibration,
            "source_sha256": _sources(),
        },
        args.output,
    )
    print(f"wrote full {construction['method']} allocation bases: {args.output}")


def _load_basis(path, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != BASIS_FORMAT or payload.get("method") != "ours":
        raise ValueError("expected a supported full allocation basis artifact, not a checkpoint")
    _check_identity(payload, identity)
    return payload


def train(args):
    _check_output(args.output)
    identity = _identity(args.model)
    payload = _load_basis(args.basis, identity)
    from .arch import read_text_config

    config = read_text_config(args.model)
    from . import experiment_data

    data = experiment_data.load(args.experiment_data, identity, config["vocab_size"])
    experiment_data.require_same_basis(payload["calibration"], data, args.experiment_data)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    process_rank = int(os.environ.get("RANK", "0"))
    device = torch.device(args.device)
    if world_size > 1:
        if device.type != "cuda":
            raise ValueError("CLI distributed allocation requires CUDA")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    placement = None
    if args.student_gpus is not None:
        if args.student_gpus < 1 or args.teacher_gpus < 0:
            raise ValueError("--student-gpus must be positive and --teacher-gpus nonnegative")
        if args.teacher_split == "on":
            raise ValueError("--teacher-split on and --student-gpus are exclusive")
        args.teacher_split = "off"
        block = args.student_gpus + args.teacher_gpus
        first = int(os.environ.get("LOCAL_RANK", "0")) * block
        if first + block > torch.cuda.device_count():
            raise ValueError(
                f"rank needs visible GPUs {first}..{first + block - 1}; "
                f"{torch.cuda.device_count()} are visible"
            )
        student_devices = list(range(first, first + args.student_gpus))
        teacher_devices = list(range(first + args.student_gpus, first + block)) or student_devices
        device = torch.device("cuda", first)
        torch.cuda.set_device(device)
        placement = {
            "student_gpus": args.student_gpus,
            "teacher_gpus": args.teacher_gpus,
            "layout": "per-rank naive pipeline, whole decoder layers balanced by parameters",
        }
    split = args.teacher_split == "on" or (
        args.teacher_split == "auto" and world_size > args.global_batch
    )
    if split and (world_size < 2 or world_size % 2):
        raise ValueError("a teacher split needs an even number of ranks: students first, then teachers")
    students = world_size // 2 if split else world_size
    if args.global_batch % students or data["train_samples"] % args.global_batch:
        raise ValueError("global batch must divide train samples and be divisible by the training ranks")

    def load(role="student"):
        if placement is None:
            return load_base_model(
                args.model, dtype=torch.bfloat16, device_map={"": str(device)}, attn_implementation="sdpa"
            )[0]
        from .model import gpu_memory_map

        return load_base_model(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            max_memory=gpu_memory_map(student_devices if role == "student" else teacher_devices),
        )[0]

    try:
        if world_size > 1:
            dist.init_process_group("nccl", device_id=device)
        link = None
        if split:
            group = split_student_group(students)
            if process_rank >= students:
                del payload["bases"]
                serve_teacher(
                    load(),
                    data["input_ids"],
                    train_samples=data["train_samples"],
                    global_batch=args.global_batch,
                    students=students,
                )
                return
        student = load()
        selector, blocks = install_belt_modules(
            student, payload["bases"], retain_ratio=args.retain_ratio, temperature=args.temperature
        )
        del payload["bases"]
        if split:
            teacher = None
            link = TeacherSplit(
                students, group, hidden_size=config["hidden_size"], dtype=torch.bfloat16, device=device
            )
            link.receive_validation(data["input_ids"][data["train_samples"] :])
        else:
            teacher = load("teacher")
        result = train_selector(
            student,
            teacher,
            data["input_ids"],
            train_samples=data["train_samples"],
            global_batch=args.global_batch,
            learning_rate=args.learning_rate,
            validation_interval=args.validation_interval,
            patience=args.patience,
            min_delta=args.min_delta,
            min_updates=args.min_updates,
            token_chunk=args.token_chunk,
            min_delta_relative=args.min_delta_relative,
            warmup_updates=args.warmup_updates,
            stable_updates=args.stable_updates,
            decay_updates=args.decay_updates,
            split=link,
        )
        if placement is not None:
            result["optimizer"]["model_parallel"] = placement
        if process_rank == 0:
            ranks = selector.hard_ranks()
            rank_map = selected_rank_map(ranks, blocks)
            result.update(
                format=ALLOCATION_FORMAT,
                protocol=PROTOCOL_WSD,
                **identity,
                method=payload["method"],
                basis={
                    "path": str(Path(args.basis).resolve()),
                    "sha256": _sha256(Path(args.basis)),
                    "trainable": False,
                    "calibration": payload["calibration"],
                },
                training_data={
                    "path": str(Path(args.experiment_data).resolve()),
                    "sha256": _sha256(Path(args.experiment_data)),
                    **{k: v for k, v in data.items() if k not in ("input_ids", "calibration_input_ids")},
                },
                rank_budget={
                    "target": selector.budget,
                    "selected": sum(ranks),
                    "capacity": sum(b.capacity for b in blocks),
                    "exact": sum(ranks) == selector.budget,
                    "retain_ratio": args.retain_ratio,
                    "cost": "compact_latent_elements",
                },
                final_ranks=rank_summary(ranks, blocks),
                selected_ranks=rank_map,
                selected_ranks_sha256=rank_sha256(rank_map),
                selector_parameters=selector.block_shifts.detach().cpu().tolist(),
                seed=SEED,
                dtype="bfloat16",
                quantization=False,
                source_sha256=_sources(),
            )
            _save(result, args.output, json_file=True)
            print(f"wrote selected ranks: {args.output}")
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def export(args):
    _check_output(args.output)
    _check_output(args.output + ".json")
    identity = _identity(args.model)
    payload = _load_basis(args.basis, identity)
    allocation = json.loads(Path(args.allocation).read_text())
    _check_identity(allocation, identity)
    if (
        allocation.get("format") != ALLOCATION_FORMAT
        or allocation.get("protocol") != PROTOCOL_WSD
        or allocation.get("method") != payload["method"]
        or allocation["basis"]["sha256"] != _sha256(Path(args.basis))
        or allocation["selected_ranks_sha256"] != rank_sha256(allocation["selected_ranks"])
    ):
        raise ValueError("allocation provenance or selected rank hash mismatch")
    blocks = build_block_specs(payload["bases"], retain_ratio=allocation["rank_budget"]["retain_ratio"])
    budget = sum(b.initial_rank for b in blocks)
    if allocation["rank_budget"]["target"] != budget:
        raise ValueError("allocation budget does not match full basis capacities")
    model, _ = load_base_model(args.model, dtype=torch.float16, device_map="cpu", attn_implementation="sdpa")
    sd, report = export_factors(model, payload["bases"], allocation["selected_ranks"], budget=budget)
    metadata = {
        **identity,
        "method": payload["method"],
        "rank_allocation": allocation["protocol"],
        "allocation": allocation,
        "allocation_sha256": _sha256(Path(args.allocation)),
        "construction": payload["construction"],
        "calibration": payload["calibration"],
        "factor_dtype": "float16",
        "quantization": False,
        "kv_cache": report,
        "source_sha256": _sources(),
    }
    write_checkpoint(sd, metadata, args.output)
    from .compression import format_report

    print(format_report(report, per_layer=True))
    print(f"wrote allocated checkpoint: {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    basis = sub.add_parser("basis", help="generate full frozen bases from original model")
    basis.add_argument("--query-tokens", type=int, default=64)
    basis.add_argument("--query-chunk", type=int, default=64)
    basis.add_argument("--deploy-length", type=int, help="prior RoPE grid extent (default: 32768)")
    basis.add_argument(
        "--layer-devices",
        nargs="+",
        metavar="cuda:N",
        help="fit whole layers on spawned workers across logical CUDA devices while --device propagates the teacher",
    )
    basis.add_argument("--max-seconds", type=float, help="construction deadline; no partial final artifact")

    training = sub.add_parser("train", help="learn rank cutoffs using output KD and WSD")
    training.add_argument("--basis", required=True)
    training.add_argument("--retain-ratio", type=float, default=0.5)
    training.add_argument("--temperature", type=float, default=4.0)
    training.add_argument("--learning-rate", type=float, default=2.0, help="WSD peak learning rate in rank units")
    training.add_argument("--global-batch", type=int, default=4)
    training.add_argument(
        "--teacher-split",
        choices=("auto", "on", "off"),
        default="auto",
        help="run the frozen teacher on dedicated ranks; auto splits when world_size > --global-batch",
    )
    training.add_argument(
        "--student-gpus",
        type=int,
        help="model parallel per rank: spread each rank's student over this many visible GPUs",
    )
    training.add_argument(
        "--teacher-gpus",
        type=int,
        default=1,
        help="with --student-gpus: GPUs per rank for the frozen teacher (0 shares the student's)",
    )
    training.add_argument("--validation-interval", type=int, default=25)
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--min-delta", type=float, default=0.0)
    training.add_argument("--min-updates", type=int, default=50)
    training.add_argument("--token-chunk", type=int, default=2048)
    training.add_argument("--min-delta-relative", type=float, default=0.02)
    training.add_argument("--warmup-updates", type=int, default=10, help="WSD linear warmup length")
    training.add_argument("--stable-updates", type=int, default=300, help="WSD maximum stable phase updates")
    training.add_argument("--decay-updates", type=int, default=50, help="WSD linear decay updates (>= 1)")

    exporting = sub.add_parser("export", help="slice selected ranks into a canonical checkpoint")
    exporting.add_argument("--basis", required=True)
    exporting.add_argument("--allocation", required=True)

    for command in (basis, training, exporting):
        command.add_argument("--model", required=True, help="local original model directory")
        command.add_argument("--output", required=True)
    from .experiment_data import add_argument

    for command in (basis, training):
        add_argument(command)
        command.add_argument("--device", default="cuda", help="visible device, CUDA_VISIBLE_DEVICES set externally")
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.global_batch < 1:
            parser.error("--global-batch must be positive")
        if args.decay_updates < 1:
            parser.error("--decay-updates must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_offline()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    {"basis": build_basis, "train": train, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
