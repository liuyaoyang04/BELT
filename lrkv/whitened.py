"""Behavior-aligned Kronecker curvature surrogate for K and weighted-SVD factors for V.

Only the closed-form surrogate construction is exposed; no training or iterative fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
import math
import torch


@torch.no_grad()
def fit_key_factors(keys, queries, cos, sin, indices, deploy, weight, *,
                    query_chunk=64, dtype=torch.float32,
                    key_norm_weight=None, rotary_dim=None):
    """Fit one KV head; weight is [input_dim, head_dim], keys are FP16 teacher outputs.

    With ``key_norm_weight`` (Qwen3-style per-head RMSNorm on k_proj outputs), ``keys``
    are the unit-RMS k_proj outputs and ``queries`` the q_norm outputs; the curvature
    uses the RMSNorm Jacobian (see ``RoPEKeyCurvature``) and the factors still act on
    the k_proj output, so the deployed latent is ``x @ down``.

    ``rotary_dim`` < head_dim is partial RoPE (Qwen3.8): the tables, including the
    deployment grid, are then padded to head_dim (``lrkv.arch.pad_rope_tables``).
    """
    h = weight.shape[1]
    curvature = RoPEKeyCurvature(
        keys, keys, queries, cos, sin, indices, query_chunk=query_chunk,
        dtype=dtype, device=weight.device, teacher_weight=torch.eye(h, device=weight.device, dtype=dtype),
        key_norm_weight=key_norm_weight, rotary_dim=rotary_dim)
    row, col = attention_prior_metrics(curvature, *deploy, attention_dtype=torch.float16)
    fallback = None
    try:
        fit = fit_kronecker_whitened(curvature, row, col)
        row_metric, col_metric, checks = fit.row_metric, fit.col_metric, fit.history
    except FlatCurvatureError as exc:
        # The head's attention does not respond to its keys (e.g. saturated one-hot heads): the
        # one-step update has no data term, only rounding noise. Keep its starting point, the prior.
        row_metric, col_metric, checks = row, col, []
        fallback = {"metrics": "attention_prior_R0_S0", "reason": str(exc)}
    factors = weighted_key_svd(torch.eye(h, device=weight.device, dtype=torch.float64), row_metric, col_metric)
    factors.down = weight.double() @ factors.down
    factors.audit.update(method="kronecker_surrogate", prior_attention_dtype="torch.float16",
                         curvature_dtype=str(dtype), decomposition_dtype="torch.float64",
                         query_rows=curvature.row_count, checks=checks)
    if fallback is not None:
        factors.audit["flat_curvature_fallback"] = fallback
    if key_norm_weight is not None:
        factors.audit["key_norm"] = "k_proj_output_space_rmsnorm_jacobian"
    if rotary_dim is not None and rotary_dim != h:
        factors.audit["rotary_dim"] = int(rotary_dim)
    return factors


@torch.no_grad()
def value_metrics(attention, values, o_weight, num_kv_heads, head_dim, gates=None):
    """Joint-V weighted-SVD metrics, preserving FP32 operation order.

    attention [B,Hq,Q,N] is captured from the FP16 eager teacher; values
    [B,N,Hkv*h]. Every query head aggregates the entire joint V vector.

    ``gates`` [B,Q,Hq,h]: sigmoid of an attention output gate at the same query
    rows (Qwen3.5), which scales each head's attention output before ``o_proj``.
    Query t's output error through head q is then W_q diag(s_tq) E y_tq, and
    the column block of head q becomes (W_q^T W_q) * mean_t(s_tq s_tq^T), the
    exact expected metric when the gate is independent of y. None keeps the
    ungated metric (bitwise unchanged).
    """
    attention, values = attention.float(), values.float()
    width = num_kv_heads * head_dim
    heads = attention.shape[1]
    if heads % num_kv_heads or values.shape[-1] != width:
        raise ValueError("invalid joint V / GQA layout")
    row = torch.zeros(width, width, device=attention.device)
    for head in range(heads):
        weighted = (attention[:, head] @ values).reshape(-1, width)
        row += weighted.T @ weighted
    col = torch.zeros_like(row)
    o_weight = o_weight.to(attention.device).float()
    repeats = heads // num_kv_heads
    if gates is not None and (gates.ndim != 4 or gates.shape[0] != attention.shape[0]
                              or gates.shape[1:] != (attention.shape[2], heads, head_dim)):
        raise ValueError("gates must have shape [B, Q, Hq, head_dim] matching the attention rows")
    for kv in range(num_kv_heads):
        block = torch.zeros(head_dim, head_dim, device=attention.device)
        for repeat in range(repeats):
            head = kv * repeats + repeat
            w = o_weight[:, head * head_dim:(head + 1) * head_dim]
            if gates is None:
                block += w.T @ w
            else:
                s = gates[:, :, head].reshape(-1, head_dim).to(attention.device).float()
                block += (w.T @ w) * (s.T @ s / s.shape[0])
        col[kv * head_dim:(kv + 1) * head_dim, kv * head_dim:(kv + 1) * head_dim] = block
    return row, col


@torch.no_grad()
def fit_value_factors(attention, values, weight, o_weight, num_kv_heads, head_dim, rank, gates=None):
    row, col = value_metrics(attention, values, o_weight, num_kv_heads, head_dim, gates)
    encoder, decoder, _ = behavior_svd_closed_form(row, col, capacity=rank)
    # Multiply in FP32 before any rounding to FP16.
    return encoder.T @ weight.float(), decoder.T



def rotate_half(value: torch.Tensor, rotary_dim: int | None = None) -> torch.Tensor:
    """The signed pairing J of RoPE. With ``rotary_dim`` < width (partial RoPE, Qwen3.5) J acts on the
    first ``rotary_dim`` channels as HF does and is zero on the position-free rest, whose padded
    tables are cos 1 / sin 0 (``lrkv.arch.pad_rope_tables``)."""
    if rotary_dim is not None and rotary_dim != value.shape[-1]:
        rotated = rotate_half(value[..., :rotary_dim])
        return torch.cat((rotated, torch.zeros_like(value[..., rotary_dim:])), dim=-1)
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(value, cos, sin, rotary_dim=None):
    return value * cos + rotate_half(value, rotary_dim) * sin


def transpose_rope(value, cos, sin, rotary_dim=None):
    """Apply the TRANSPOSE, including any amplitude scaling in the tables.

    This need not be an inverse. Keeping sin inside rotate_half also handles
    unequal table entries in the two halves of the rotary map. J is
    antisymmetric for full and partial rotation alike.
    """
    return value * cos - rotate_half(value * sin, rotary_dim)


@dataclass
class CurvatureBlock:

    inputs: torch.Tensor                 # [N, d]
    effective_queries: torch.Tensor      # [T, N, h], includes score scaling
    probabilities: torch.Tensor          # [T, N]
    log_probabilities: torch.Tensor      # [T, N], masked entries are -inf

    @property
    def weights(self):
        return self.probabilities.square()

    def _moments(self):
        """Return M=sum p B and A=(sum w) M - 2 sum w B for the centered attention-distribution curvature."""
        x, g, p, w = self.inputs, self.effective_queries, self.probabilities, self.weights
        mean = x.T @ (p[..., None] * g)
        coefficients = w.sum(-1, keepdim=True) * p - 2 * w
        source = x.T @ (coefficients[..., None] * g)
        return mean, source

    def _row_terms(self, moments):
        """Return key weights and an UNSYMMETRIZED centering correction for the identity partial trace.

        Pool the query and head axes before multiplying, never materializing [query,d,d].
        """
        g = self.effective_queries
        mean, source = moments
        weights = (self.weights * (g * g).sum(-1)).sum(0)
        left = source.transpose(0, 1).flatten(1)
        right = mean.transpose(0, 1).flatten(1)
        return weights, left @ right.T

    def _col_terms(self, moments, input_quadratic):
        """Identity partial trace using sequence-level diag(X X^T)."""
        g = self.effective_queries
        mean, source = moments
        weights = self.weights * input_quadratic[None]
        leading = (g.transpose(-1, -2) @ (weights[..., None] * g)).sum(0)
        correction = source.flatten(0, 1).T @ mean.flatten(0, 1)
        return leading + correction


class RoPEKeyCurvature:

    """Replay frozen teacher rows to apply the exact empirical curvature.

    inputs: [B,N,d], before k_proj (after the layer's input normalization).
    keys: [B,N,h], before RoPE, including a frozen key bias if present.
    queries: [B,Q,G,h], before RoPE, all GQA queries sharing this KV head.
    cos/sin: [N,h] or [1 or B,N,h], actual model tables including scaling.
    query_indices: [Q], query offsets in these sequences, used for causality.
    visible_mask: optional Boolean [1 or B,Q,N], further restricts causality.
    teacher_weight/bias: optional original linear map. When supplied, keys
    are recomputed from X in curvature precision rather than treating cached
    projection-rounding residuals as fixed token-dependent biases.

    Tensors may be stored on CPU; only one sequence/query block is staged
    onto device. Ours uses this operator in head-dimensional key coordinates.

    key_norm_weight: optional [h] weight of a per-head RMSNorm between the key
    projection and RoPE (Qwen3 ``k_norm``). The (recomputed) keys ``k`` are then
    the unit-RMS projection outputs u, scores use the normalized keys u * w,
    and each pair's effective query is the RMSNorm Jacobian applied to it,
    P_u (w * g) with P_u = I - u u^T / h: an error along u only rescales the
    key and is removed by the normalization. The curvature is then that of a
    map acting on the projection output. queries are the q_norm outputs.
    """

    def __init__(
        self, inputs, keys, queries, cos, sin, query_indices, *,
        query_chunk=8, visible_mask=None,
        device=None, dtype=torch.float64,
        teacher_weight=None, teacher_bias=None, query_transform=None, key_norm_weight=None,
        rotary_dim=None,
    ):
        if inputs.ndim != 3 or keys.ndim != 3 or queries.ndim != 4:
            raise ValueError("expected inputs [B,N,d], keys [B,N,h], queries [B,Q,G,h]")
        batch, length, self.input_dim = inputs.shape
        self.head_dim = keys.shape[-1]
        if not batch or not length or not self.input_dim or self.head_dim % 2 or not self.head_dim:
            raise ValueError("nonempty inputs and a positive even rotary head dimension are required")
        if keys.shape[:2] != (batch, length) or queries.shape[0] != batch or queries.shape[-1] != self.head_dim:
            raise ValueError("incompatible inputs, keys and queries")
        indices = torch.as_tensor(query_indices, dtype=torch.long).detach().cpu()
        if indices.ndim != 1 or len(indices) != queries.shape[1] or not len(indices) or not queries.shape[2]:
            raise ValueError("query_indices must match nonempty query rows")
        if bool(((indices < 0) | (indices >= length)).any()):
            raise ValueError("query indices are outside the key sequence")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("curvature dtype must be float32 or float64")
        tables = []
        for table in (cos, sin):
            table = table.unsqueeze(0) if table.ndim == 2 else table
            if table.ndim != 3 or table.shape[0] not in (1, batch) or table.shape[1:] != (length, self.head_dim):
                raise ValueError("RoPE tables must have shape [1 or B,N,h]")
            tables.append(table.detach())
        if visible_mask is not None:
            if visible_mask.dtype != torch.bool or visible_mask.ndim != 3 or visible_mask.shape[0] not in (1, batch) or visible_mask.shape[1:] != (len(indices), length):
                raise ValueError("visible_mask must be Boolean [1 or B,Q,N]")
            visible_mask = visible_mask.detach()
        self.inputs, self.keys, self.queries = inputs.detach(), keys.detach(), queries.detach()
        self.cos, self.sin = tables
        self.query_indices, self.visible_mask = indices, visible_mask
        self.query_chunk = int(query_chunk)
        self.device, self.dtype = torch.device(device or keys.device), dtype
        self.row_count = batch * queries.shape[1] * queries.shape[2]
        if teacher_weight is not None and teacher_weight.shape != (self.input_dim, self.head_dim):
            raise ValueError("teacher_weight must have shape [input_dim, head_dim]")
        if teacher_bias is not None and (teacher_weight is None or teacher_bias.shape != (self.head_dim,)):
            raise ValueError("teacher_bias requires teacher_weight and shape [head_dim]")
        self.teacher_weight = None if teacher_weight is None else teacher_weight.detach().to(self.device, dtype)
        self.teacher_bias = None if teacher_bias is None else teacher_bias.detach().to(self.device, dtype)
        # Optional [h, h] map applied to the effective queries g (after RoPE and scaling);
        # scores and probabilities still use the raw teacher queries. This expresses
        # curvature statistics in query-side whitened coordinates without a metric product.
        if query_transform is not None and query_transform.shape != (self.head_dim, self.head_dim):
            raise ValueError("query_transform must have shape [head_dim, head_dim]")
        self.query_transform = None if query_transform is None else query_transform.detach().to(self.device, dtype)
        if key_norm_weight is not None and key_norm_weight.shape != (self.head_dim,):
            raise ValueError("key_norm_weight must have shape [head_dim]")
        self.key_norm_weight = (None if key_norm_weight is None
                                else key_norm_weight.detach().to(self.device, dtype))
        if rotary_dim is not None and (rotary_dim % 2 or not 0 < rotary_dim <= self.head_dim):
            raise ValueError("rotary_dim must be even and in (0, head_dim]")
        self.rotary_dim = None if rotary_dim in (None, self.head_dim) else int(rotary_dim)

    def score_keys(self, k):
        """Keys the teacher scores against: k itself, or the normalized keys u * w."""
        return k if self.key_norm_weight is None else k * self.key_norm_weight

    def normalized_queries(self, g, k):
        """RMSNorm Jacobian on the query side of each pair: P_u (w * g) for g [T, N, h], u = k [N, h]."""
        if self.key_norm_weight is None:
            return g
        weighted = g * self.key_norm_weight
        return weighted - k[None] * ((weighted * k[None]).sum(-1, keepdim=True) / self.head_dim)

    def sequences(self) -> Iterator[tuple[torch.Tensor, Iterator[CurvatureBlock]]]:
        """Stage each X once and keep its query blocks together for reuse."""
        for sample in range(self.inputs.shape[0]):
            x = self.inputs[sample].to(self.device, self.dtype)
            yield x, self._sequence_blocks(sample, x)

    def _sequence_blocks(self, sample, x) -> Iterator[CurvatureBlock]:
        indices = self.query_indices.to(self.device)
        repeats = self.queries.shape[2]
        positions = indices.repeat_interleave(repeats)
        key_positions = torch.arange(self.keys.shape[1], device=self.device)
        k = (self.keys[sample].to(self.device, self.dtype)
             if self.teacher_weight is None else x @ self.teacher_weight)
        if self.teacher_bias is not None:
            k = k + self.teacher_bias
        q = self.queries[sample].to(self.device, self.dtype)
        cos = self.cos[0 if self.cos.shape[0] == 1 else sample].to(self.device, self.dtype)
        sin = self.sin[0 if self.sin.shape[0] == 1 else sample].to(self.device, self.dtype)
        rotated = apply_rope(q, cos[indices, None], sin[indices, None], self.rotary_dim).flatten(0, 1)
        score_k = self.score_keys(k)
        supplied = None
        if self.visible_mask is not None:
            supplied = self.visible_mask[0 if self.visible_mask.shape[0] == 1 else sample]
            supplied = supplied.to(self.device).repeat_interleave(repeats, dim=0)
        for begin in range(0, len(positions), self.query_chunk):
            end = min(begin + self.query_chunk, len(positions))
            g = transpose_rope(rotated[begin:end, None], cos[None], sin[None], self.rotary_dim) / math.sqrt(self.head_dim)
            visible = key_positions[None] <= positions[begin:end, None]
            if supplied is not None:
                visible = visible & supplied[begin:end]
            if not bool(visible.any(-1).all()):
                raise ValueError("a query has no visible keys")
            scores = (g * score_k[None]).sum(-1)
            scores = scores.masked_fill(~visible, -torch.inf)
            log_p = scores.log_softmax(-1)
            if not bool(torch.isfinite(log_p[visible]).all()):
                raise ValueError("nonfinite teacher scores/probabilities")
            g = self.normalized_queries(g, k)
            if self.query_transform is not None:
                g = g @ self.query_transform
            yield CurvatureBlock(x, g, log_p.exp(), log_p)

    def marginals(self):
        """Compute T(I_h), T*(I_d) together in ONE teacher-data pass.

        Identity metrics need no dense multiplication. Both sides share the
        RoPE/softmax rows and centered means; no [query,d,d] tensor is built.
        """
        row = torch.zeros(self.input_dim, self.input_dim, device=self.device, dtype=self.dtype)
        col = torch.zeros(self.head_dim, self.head_dim, device=self.device, dtype=self.dtype)
        for x, sequence in self.sequences():
            weights = torch.zeros(x.shape[0], device=self.device, dtype=self.dtype)
            diagonal = x.square().sum(-1)
            for block in sequence:
                moments = block._moments()
                chunk_weights, correction = block._row_terms(moments)
                weights.add_(chunk_weights)
                row.add_(correction)
                col.add_(block._col_terms(moments, diagonal))
            row.addmm_(x.T, weights[:, None] * x)
        return ((row + row.T) / (2 * self.row_count),
                (col + col.T) / (2 * self.row_count))

    def attention_marginals(self, probability_dtype=None):
        """Teacher attention mass per (query head, key position) and per (sequence, key position).

        Rotary matrix products only; no [query, key, head_dim] tensor. These are
        the two attention statistics used by the attention-weighted prior metrics (R0, S0).
        ``probability_dtype`` rounds each probability (e.g. to float16) before
        accumulation to match FP16 teacher attention probabilities.
        """
        indices = self.query_indices.to(self.device)
        repeats = self.queries.shape[2]
        positions = indices.repeat_interleave(repeats)
        key_positions = torch.arange(self.keys.shape[1], device=self.device)
        position_mass = torch.zeros(repeats, self.keys.shape[1], device=self.device, dtype=self.dtype)
        row_weights = torch.zeros(self.inputs.shape[0], self.keys.shape[1], device=self.device, dtype=self.dtype)
        for sample in range(self.inputs.shape[0]):
            x = self.inputs[sample].to(self.device, self.dtype)
            k = (self.keys[sample].to(self.device, self.dtype)
                 if self.teacher_weight is None else x @ self.teacher_weight)
            if self.teacher_bias is not None:
                k = k + self.teacher_bias
            cos = self.cos[0 if self.cos.shape[0] == 1 else sample].to(self.device, self.dtype)
            sin = self.sin[0 if self.sin.shape[0] == 1 else sample].to(self.device, self.dtype)
            q = self.queries[sample].to(self.device, self.dtype)
            rotated_q = apply_rope(q, cos[indices, None], sin[indices, None], self.rotary_dim).flatten(0, 1)
            rotated_k = apply_rope(self.score_keys(k), cos, sin, self.rotary_dim)
            supplied = None
            if self.visible_mask is not None:
                supplied = self.visible_mask[0 if self.visible_mask.shape[0] == 1 else sample]
                supplied = supplied.to(self.device).repeat_interleave(repeats, dim=0)
            for begin in range(0, len(positions), self.query_chunk):
                end = min(begin + self.query_chunk, len(positions))
                scores = (rotated_q[begin:end] @ rotated_k.T) / math.sqrt(self.head_dim)
                visible = key_positions[None] <= positions[begin:end, None]
                if supplied is not None:
                    visible = visible & supplied[begin:end]
                if not bool(visible.any(-1).all()):
                    raise ValueError("a query has no visible keys")
                p = scores.masked_fill(~visible, -torch.inf).softmax(-1)
                if not bool(torch.isfinite(p).all()):
                    raise ValueError("nonfinite teacher probabilities")
                if probability_dtype is not None:
                    p = p.to(probability_dtype).to(self.dtype)
                # Rows are ordered query-major, GQA head minor. Sum each head's rows with a
                # fixed-order reduction: CUDA index_add_ uses atomics and is not reproducible.
                for head in range(min(repeats, end - begin)):
                    position_mass[(begin + head) % repeats].add_(p[head::repeats].sum(0))
                row_weights[sample].add_(p.sum(0))
        return position_mass, row_weights


class FlatCurvatureError(ValueError, ArithmeticError):
    """The whitened curvature marginals of a head are at rounding level (no usable teacher curvature)."""


@dataclass
class KroneckerFit:
    row_metric: torch.Tensor
    col_metric: torch.Tensor
    curvature_trace: float
    damping: float
    history: list[dict]
    converged: bool
    method: str
    data_passes: int


def _trace_normalized(metric, ridge):
    dim = metric.shape[0]
    symmetric = (metric.double() + metric.double().T) * 0.5
    trace = float(symmetric.trace())
    if not math.isfinite(trace) or trace <= 0:
        raise ValueError("metric trace must be finite and positive")
    return symmetric * (dim / trace) + ridge * torch.eye(dim, dtype=torch.float64, device=metric.device)


@torch.no_grad()
def attention_prior_metrics(curvature: RoPEKeyCurvature, deploy_cos, deploy_sin, *, row_floor=0.05, ridge=1e-4,
                            attention_dtype=None):
    """Attention-weighted key prior metrics (R0, S0) from streamed attention statistics.

    Attention-weighted key
    Gram with a row floor (R0) and query/RoPE statistics with position mass, a
    floor and the deployment position grid (S0); both trace-normalized with a
    ridge. The attention probabilities are recomputed here instead of captured.
    Requires a key-space curvature (inputs are the teacher keys).
    """
    h = curvature.head_dim
    if curvature.input_dim != h:
        raise ValueError("attention prior metrics are defined in key coordinates")
    position_mass, row_weights = curvature.attention_marginals(attention_dtype)   # [G, N], [B, N]
    position_mass = position_mass + row_floor * position_mass.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    indices = curvature.query_indices.to(curvature.device)
    cos = curvature.cos[0].to(curvature.device, curvature.dtype)          # [N, h]; single table
    sin = curvature.sin[0].to(curvature.device, curvature.dtype)
    if curvature.cos.shape[0] != 1:
        raise ValueError("attention prior metrics assume one shared RoPE table")
    deploy_cos = deploy_cos.to(curvature.device, curvature.dtype)
    deploy_sin = deploy_sin.to(curvature.device, curvature.dtype)
    deploy_count = max(deploy_cos.shape[0], 1)
    deploy_cc, deploy_ss, deploy_cs = deploy_cos.T @ deploy_cos, deploy_sin.T @ deploy_sin, deploy_cos.T @ deploy_sin
    queries = curvature.queries.to(curvature.device, curvature.dtype)     # [B, Q, G, h]
    rotated = apply_rope(queries, cos[indices][None, :, None], sin[indices][None, :, None], curvature.rotary_dim)
    col = torch.zeros(h, h, device=curvature.device, dtype=curvature.dtype)
    for g in range(queries.shape[2]):
        query = rotated[:, :, g].reshape(-1, h)
        half = rotate_half(query, curvature.rotary_dim)
        gram, half_gram, cross = query.T @ query, half.T @ half, query.T @ half
        mass = position_mass[g]
        cc, ss, cs = (cos * mass[:, None]).T @ cos, (sin * mass[:, None]).T @ sin, (cos * mass[:, None]).T @ sin
        col += gram * cc + half_gram * ss - cross * cs - cross.T * cs.T
        scale = float(mass.sum()) / deploy_count
        col += gram * (deploy_cc * scale) + half_gram * (deploy_ss * scale) - cross * (deploy_cs * scale) - cross.T * (deploy_cs.T * scale)
    if curvature.key_norm_weight is not None:
        # Query side of a normalized key: w * g. The rank-one P_u part of the RMSNorm
        # Jacobian is key dependent and is left to the exact whitened marginals.
        col = col * curvature.key_norm_weight[:, None] * curvature.key_norm_weight[None, :]
    row_weights = row_weights + row_floor * row_weights.mean().clamp_min(1e-12)
    row = torch.zeros(h, h, device=curvature.device, dtype=torch.float64)
    for sample in range(curvature.inputs.shape[0]):
        keys = curvature.inputs[sample].to(curvature.device, curvature.dtype)
        row.add_((keys.T @ (row_weights[sample][:, None] * keys)).double())
    return _trace_normalized((row + row.T) * 0.5, ridge), _trace_normalized(col.double(), ridge)


def _inverse_sqrt_and_sqrt(metric):
    values, vectors = torch.linalg.eigh((metric.double() + metric.double().T) * 0.5)
    if float(values.min()) <= 0:
        raise ValueError("prior metric must be positive definite")
    return (vectors * values.rsqrt()) @ vectors.T, (vectors * values.sqrt()) @ vectors.T


@torch.no_grad()
def fit_kronecker_whitened(curvature: RoPEKeyCurvature, prior_row, prior_col, *, ridge=1e-4) -> "KroneckerFit":
    """Closed-form Kronecker curvature surrogate update in prior-whitened coordinates, with ridge shrinkage.

    In the prior-whitened coordinates k -> R0^-1/2 k, g -> S0^-1/2 g the exact
    centered curvature's identity partial traces are A = R0^-1/2 T(S0^-1) R0^-1/2
    and B = S0^-1/2 T*(R0^-1) S0^-1/2. They are accumulated directly on the
    whitened data (no ill-conditioned metric products, so the passes can run in
    float32), trace-normalized and ridged, and mapped back:
    R = R0^1/2 (h A / tr A + ridge I) R0^1/2, S = S0^1/2 (h B / tr B + ridge I) S0^1/2.
    """
    h = curvature.head_dim
    if curvature.input_dim != h or curvature.teacher_weight is None:
        raise ValueError("the whitened construction needs a key-space curvature (inputs = teacher keys)")
    row_inv_sqrt, row_sqrt = _inverse_sqrt_and_sqrt(prior_row.to(curvature.device))
    col_inv_sqrt, col_sqrt = _inverse_sqrt_and_sqrt(prior_col.to(curvature.device))
    u = row_inv_sqrt.to(curvature.dtype)
    whitened_keys = torch.stack([curvature.inputs[b].to(curvature.device, curvature.dtype) @ u
                                 for b in range(curvature.inputs.shape[0])])
    whitened = RoPEKeyCurvature(
        whitened_keys, curvature.keys, curvature.queries, curvature.cos, curvature.sin, curvature.query_indices,
        query_chunk=curvature.query_chunk, visible_mask=curvature.visible_mask,
        device=curvature.device, dtype=curvature.dtype, teacher_weight=None, teacher_bias=None,
        query_transform=col_inv_sqrt.to(curvature.dtype), key_norm_weight=curvature.key_norm_weight,
        rotary_dim=curvature.rotary_dim,
    )
    # Scores must come from the raw teacher keys: keys were passed explicitly, inputs are whitened.
    a, b = (m.double() for m in whitened.marginals())
    del whitened, whitened_keys
    tau_a, tau_b = float(a.trace()), float(b.trace())
    if not math.isfinite(tau_a) or tau_a <= 0 or not math.isfinite(tau_b) or tau_b <= 0:
        raise FlatCurvatureError("flat or invalid whitened curvature")
    # Equal in exact arithmetic; the gap measures the data passes' accumulation error.
    discrepancy = abs(tau_a - tau_b) / tau_a
    if discrepancy > (1e-3 if curvature.dtype == torch.float32 else 1e-6):
        raise FlatCurvatureError(f"whitened marginal traces disagree: {discrepancy:.3e}")
    eye = torch.eye(h, device=curvature.device, dtype=torch.float64)
    conditioned, min_ratios = [], []
    tolerance = 1e-4 if curvature.dtype == torch.float32 else 1e-8
    for m in (a, b):
        values, vectors = torch.linalg.eigh(m)
        ratio = float(values.min()) / max(float(values.max()), 1e-300)
        min_ratios.append(ratio)
        if ratio < -tolerance:
            raise FlatCurvatureError(f"materially indefinite whitened marginal: min/max eigenvalue {ratio:.3e}")
        # As in the reference conditioning: clamp round-off negatives, keep the trace of the clamped matrix.
        conditioned.append((vectors * values.clamp_min(0)) @ vectors.T)
    a, b = conditioned
    tau_a, tau_b = float(a.trace()), float(b.trace())
    new_row = row_sqrt @ (a * (h / tau_a) + ridge * eye) @ row_sqrt
    new_col = col_sqrt @ (b * (h / tau_b) + ridge * eye) @ col_sqrt
    return KroneckerFit((new_row + new_row.T) / 2, (new_col + new_col.T) / 2, tau_a, ridge,
                        [{"whitened_trace_relative_discrepancy": discrepancy, "pass_dtype": str(curvature.dtype),
                          "min_over_max_eigenvalue": min_ratios}],
                        False, "kronecker_surrogate_whitened_marginals", 2)


@dataclass
class WeightSpaceFactors:
    """Direct factors W_hat=down @ up, using the paper's weight convention."""
    down: torch.Tensor
    up: torch.Tensor
    spectrum: torch.Tensor
    audit: dict = field(default_factory=dict)


def _metric_cholesky(metric):
    matrix = (metric.double() + metric.double().T) * 0.5
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("nonfinite metric")
    scale = float(matrix.diagonal().mean())
    if scale <= 0:
        raise ValueError("metric must have positive trace")
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    for relative in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        jitter = relative * scale
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * eye)
        if int(info.max()) == 0:
            return factor, jitter
    raise torch.linalg.LinAlgError("key metric is not positive definite after numerical jitter")


@torch.no_grad()
def weighted_key_svd(weight, row_metric, col_metric) -> WeightSpaceFactors:
    """Solve ||L(W-W_hat)C||_F for every rank prefix, preserving its scale.

    A numerical jitter, if needed, is recorded in the returned audit and
    changes the corresponding metric by jitter*I. There is no hidden ridge
    or per-factor trace normalization. Factors remain float64 until export.
    """
    if weight.ndim != 2 or row_metric.shape != (weight.shape[0], weight.shape[0]) or col_metric.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("expected W [d,h], row_metric [d,d], col_metric [h,h]")
    left, row_jitter = _metric_cholesky(row_metric.to(weight.device))
    right, col_jitter = _metric_cholesky(col_metric.to(weight.device))
    transformed = left.T @ weight.double() @ right
    u, spectrum, vh = torch.linalg.svd(transformed, full_matrices=False)
    sqrt_s = spectrum.clamp_min(0).sqrt()
    down = torch.linalg.solve_triangular(left.T, u * sqrt_s, upper=True)
    up = torch.linalg.solve_triangular(right.T, (sqrt_s[:, None] * vh).T, upper=True).T
    return WeightSpaceFactors(down, up, spectrum, {
        "parameter_space": "full_weight", "row_jitter": row_jitter,
        "col_jitter": col_jitter, "spectrum_scale": "empirical_curvature_no_trace_normalization",
    })


def _normalized_metric(metric: torch.Tensor, ridge: float) -> torch.Tensor:
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError(f"metric must be square, got {tuple(metric.shape)}")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    dim = metric.shape[0]
    symmetric = 0.5 * (metric.double() + metric.double().T)
    trace = symmetric.diagonal().sum()
    if not torch.isfinite(trace) or float(trace) <= 0:
        raise ValueError("metric trace must be finite and positive")
    return symmetric * (dim / trace) + ridge * torch.eye(
        dim, dtype=torch.float64, device=metric.device
    )


def _cholesky_with_jitter(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix.double() + matrix.double().T)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    scale = matrix.diagonal().abs().mean().clamp_min(1e-12)
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        factor, info = torch.linalg.cholesky_ex(matrix + scale * jitter * eye)
        if int(info.max()) == 0:
            return factor
    raise torch.linalg.LinAlgError("metric is not positive definite after jitter")


def behavior_svd_closed_form(
    row_metric: torch.Tensor,
    col_metric: torch.Tensor,
    capacity: int | None = None,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve ``min tr((M-I)^T R (M-I) S)`` for every rank prefix."""

    if row_metric.shape != col_metric.shape:
        raise ValueError("row and column metrics must have the same shape")
    dim = row_metric.shape[0]
    capacity = dim if capacity is None else int(capacity)
    if not 1 <= capacity <= dim:
        raise ValueError(f"capacity must be in [1, {dim}]")
    row = _normalized_metric(row_metric, ridge)
    col = _normalized_metric(col_metric, ridge)
    left = _cholesky_with_jitter(row)
    right = _cholesky_with_jitter(col)
    u, singular_values, vh = torch.linalg.svd(
        left.T @ right, full_matrices=False
    )
    u, singular_values, vh = (
        u[:, :capacity],
        singular_values[:capacity],
        vh[:capacity],
    )
    sqrt_s = singular_values.clamp_min(0).sqrt()
    encoder = torch.linalg.solve_triangular(
        left.T, u * sqrt_s, upper=True
    )
    decoder = torch.linalg.solve_triangular(
        right.T, vh.T * sqrt_s, upper=True
    ).T
    return (
        encoder.float().contiguous(),
        decoder.float().contiguous(),
        singular_values.float().contiguous(),
    )
