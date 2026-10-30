"""Persistent, spawn-only CUDA workers that fit whole per-layer K/V bases.

The parent propagates the dense FP16 teacher layer by layer and hands each
layer (a CPU copy) and its input hidden states to an idle worker. A worker
captures that layer's teacher statistics and fits every K head and the joint V
with the serial per-layer code (``lrkv.ours._whitened_layer``) on its own
device. There is no cross-device reduction, so the bases are bitwise equal to
serial execution on the same GPU model and software stack. Only CPU tensors
cross processes.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import util as mp_util
from multiprocessing.connection import wait
import os
import tempfile
import time
import traceback

import torch
import torch.multiprocessing as mp
from torch import nn

# sun_path holds 108 bytes including the terminating NUL.
_SOCKET_PATH_MAX = 107
# multiprocessing names its fd-passing listener <tempdir>/pymp-XXXXXXXX/listener-XXXXXXXX.
_SOCKET_SUFFIX = len("/pymp-xxxxxxxx/listener-xxxxxxxx")
_LISTENER_SUFFIX = len("/listener-xxxxxxxx")
_SHORT_SOCKET_BASES = ("/dev/shm", "/tmp")


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("basis construction time limit reached; no final artifact was saved")


def _fits(directory, suffix):
    return len(os.fsencode(os.path.abspath(directory))) + suffix <= _SOCKET_PATH_MAX


def ensure_short_socket_dir():
    """Pin this process's multiprocessing temp directory to a path whose AF_UNIX sockets fit sun_path.

    CPU tensors cross processes by passing file descriptors (torch's default sharing strategy) through a
    UNIX-socket listener that multiprocessing creates under ``tempfile.gettempdir()``. A long TMPDIR, e.g. a
    result folder on a network file system, overflows the 108-byte socket path and the first tensor send
    fails with ``OSError: AF_UNIX path too long``. Only this empty, socket-only directory moves: large
    temporary files still follow TMPDIR. Order: ``LRKV_SOCKET_DIR`` (must fit), TMPDIR if it fits, then
    /dev/shm, /tmp. multiprocessing removes the directory at interpreter exit. Call it in every process
    before its first tensor send (spawned children do not inherit the choice).
    """
    override = os.environ.get("LRKV_SOCKET_DIR")
    if override and not (os.path.isdir(override) and _fits(override, _SOCKET_SUFFIX)):
        raise RuntimeError(f"LRKV_SOCKET_DIR={override!r} must be an existing directory shorter than "
                           f"{_SOCKET_PATH_MAX - _SOCKET_SUFFIX} bytes")
    candidates = [override] if override else [tempfile.gettempdir(), *_SHORT_SOCKET_BASES]
    for base in candidates:
        if base and os.path.isdir(base) and os.access(base, os.W_OK | os.X_OK) and _fits(base, _SOCKET_SUFFIX):
            break
    else:
        raise RuntimeError("no writable directory short enough for AF_UNIX sockets; set LRKV_SOCKET_DIR")
    previous = tempfile.tempdir
    tempfile.tempdir = base
    try:
        # Cached for the life of the process; a directory created earlier (before this call) is kept.
        chosen = mp_util.get_temp_dir()
    finally:
        tempfile.tempdir = previous
    if not _fits(chosen, _LISTENER_SUFFIX):
        raise RuntimeError(f"multiprocessing already uses {chosen!r}, too long for AF_UNIX sockets; call "
                           "ensure_short_socket_dir() before any tensor is sent between processes")
    return chosen


def _numeric_settings():
    return {
        "threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "linalg_library": torch.backends.cuda.preferred_linalg_library(),
    }


def _apply_numeric_settings(settings):
    """Reproduce the parent's _numeric_settings() in a spawned worker."""
    torch.set_num_threads(settings["threads"])
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings["fp16_reduction"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings["bf16_reduction"]
    torch.use_deterministic_algorithms(settings["deterministic"], warn_only=settings["warn_only"])
    torch.backends.cuda.preferred_linalg_library(settings["linalg_library"])


def validate_layer_devices(devices, *, teacher_device):
    """Resolve explicit logical CUDA indices, without changing GPU visibility."""
    if devices is None:
        return ()
    if torch.device(teacher_device).type != "cuda":
        raise ValueError("layer parallelism requires a CUDA teacher device")
    if isinstance(devices, (str, torch.device)) or not devices:
        raise ValueError("layer devices must be a nonempty sequence, e.g. ['cuda:0', 'cuda:1']")
    resolved = []
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError("layer devices require explicit logical CUDA indices, e.g. cuda:0")
        if device in resolved:
            raise ValueError(f"duplicate layer device: {device}")
        resolved.append(device)
    count = torch.cuda.device_count()
    if any(device.index >= count for device in resolved):
        raise ValueError(f"layer device is not visible: {resolved}; visible CUDA device count is {count}")
    teacher = torch.device(teacher_device)
    if teacher.index is not None and teacher.index >= count:
        raise ValueError(f"teacher device is not visible: {teacher}")
    return tuple(str(device) for device in resolved)


@dataclass(frozen=True)
class LayerTask:
    index: int
    layer: nn.Module            # CPU copy of the dense teacher decoder layer
    inputs: torch.Tensor        # [B, N, hidden] FP16 dense-teacher inputs of this layer
    cache: dict                 # CPU copy of the captured layer keyword arguments
    deploy: tuple               # CPU RoPE tables of the whitened prior's deployment grid


def _worker(connection, device, options, deadline, settings):
    index = None
    try:
        ensure_short_socket_dir()          # results are sent back as shared CPU tensors
        _apply_numeric_settings(settings)
        torch.cuda.set_device(device)
        from .ours import _whitened_layer, move_cache
        check_deadline(deadline)
        connection.send(("ready",))
        with torch.inference_mode():
            while True:
                task = connection.recv()
                if task is None:
                    break
                index = task.index
                check_deadline(deadline)
                started = time.monotonic()
                layer = task.layer.to(device)
                cache = move_cache(task.cache, device)
                deploy = tuple(t.to(device) for t in task.deploy)
                layer_sd, bases, checks = _whitened_layer(layer, task.inputs, None, cache, deploy,
                    index=index, device=torch.device(device), deadline=deadline, **options)
                del layer, cache, deploy, task
                torch.cuda.empty_cache()
                connection.send(("ok", index, layer_sd, bases, checks, time.monotonic() - started))
                del layer_sd, bases, checks
                index = None
    except BaseException as exc:
        try:
            connection.send(("error", index, type(exc).__name__, str(exc), traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class LayerPool:
    """One spawned worker per device, at most one layer in flight per worker.

    ``submit`` hands a layer to the first idle worker; ``wait`` blocks until any
    busy worker returns. A worker error, exit or the deadline tears down all
    workers when the pool is left through an exception.
    """
    def __init__(self, devices, *, options, deadline=None):
        self.devices = validate_layer_devices(devices, teacher_device="cuda")
        self.options, self.deadline = dict(options), deadline
        self.processes, self.connections, self._busy = [], [], {}
        self._started, self._closed = False, False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close(force=exc_type is not None)

    def start(self):
        if self._closed:
            raise RuntimeError("layer pool is closed")
        if self._started:
            return self
        ensure_short_socket_dir()          # before the first task tensor is sent
        context = mp.get_context("spawn")
        settings = _numeric_settings()
        try:
            for device in self.devices:
                check_deadline(self.deadline)
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, device, self.options, self.deadline, settings),
                    name=f"whitened-layer-{device}", daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.processes.append(process)
                self.connections.append(parent)
            pending = set(range(len(self.devices)))
            while pending:
                worker, message = self._receive(pending)
                if message != ("ready",):
                    raise RuntimeError("unexpected layer worker startup message")
                pending.remove(worker)
            self._started = True
            return self
        except BaseException:
            self.close(force=True)
            raise

    def idle(self):
        return [worker for worker in range(len(self.devices)) if worker not in self._busy]

    def busy(self):
        return bool(self._busy)

    def submit(self, task):
        """Send one layer to the first idle worker and return that worker's device."""
        if not self._started or self._closed:
            raise RuntimeError("layer pool must be started and not closed")
        idle = self.idle()
        if not idle:
            raise RuntimeError("no idle layer worker; wait for a result first")
        check_deadline(self.deadline)
        worker = idle[0]
        try:
            # CPU storages cross via shared memory; the parent never writes a sent buffer again.
            self.connections[worker].send(task)
        except (OSError, EOFError) as exc:
            raise RuntimeError(f"could not dispatch layer {task.index} to the worker on "
                               f"{self.devices[worker]}") from exc
        self._busy[worker] = task.index
        return self.devices[worker]

    def wait(self):
        """Block until a busy worker returns: (index, device, state dict, bases, K audits, seconds)."""
        if not self._busy:
            raise RuntimeError("no layer in flight")
        worker, message = self._receive(set(self._busy))
        index = self._busy.pop(worker)
        if message[0] != "ok" or message[1] != index:
            raise RuntimeError("layer worker returned an unexpected result")
        _, _, layer_sd, bases, checks, seconds = message
        return index, self.devices[worker], layer_sd, bases, checks, seconds

    def _receive(self, pending):
        while True:
            check_deadline(self.deadline)
            ready = wait([self.connections[worker] for worker in pending], timeout=0.05)
            if ready:
                worker = self.connections.index(ready[0])
                try:
                    message = ready[0].recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"layer worker on {self.devices[worker]} disconnected") from exc
                if message[0] == "error":
                    _, index, kind, detail, trace = message
                    error = TimeoutError if kind == "TimeoutError" else RuntimeError
                    raise error(f"layer worker on {self.devices[worker]}, layer {index}: {kind}: {detail}\n{trace}")
                return worker, message
            for worker in pending:
                code = self.processes[worker].exitcode
                if code is not None:
                    raise RuntimeError(f"layer worker on {self.devices[worker]} exited with code {code}")

    def close(self, *, force=False):
        if self._closed:
            return
        self._closed = True
        force = force or bool(self._busy)
        for process, connection in zip(self.processes, self.connections):
            if not process.is_alive():
                continue
            if force:
                process.terminate()
                continue
            try:
                connection.send(None)
            except (OSError, EOFError):
                process.terminate()
        stop = time.monotonic() + 3
        for process in self.processes:
            process.join(timeout=max(0, stop - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.kill()
                process.join()
        for connection in self.connections:
            connection.close()
        self._busy.clear()
