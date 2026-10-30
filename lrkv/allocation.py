"""Nonquantized exact-budget prefix allocation.

Only cutoff shifts are trained; full decomposition bases stay frozen.
The budget counts compact latent elements, before per-layer K padding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

def allocation_basis_components(weight: torch.Tensor, factor: Mapping):
    """Return PyTorch-layout down/up maps with the spectrum factored out.

    Legacy encoders act on projected keys; full_weight factors act directly
    on the projection's input. Both expose the same nested rank masks.
    """
    spectrum = factor["alloc_spectrum"].to(weight.device, torch.float32).clone()
    root = spectrum.clamp_min(torch.finfo(torch.float32).tiny).sqrt()
    space = factor.get("parameter_space", "key_space")
    if space == "full_weight":
        down_raw = factor["down"].to(weight.device, torch.float32)
        up_raw = factor["up"].to(weight.device, torch.float32)
        if down_raw.shape != (weight.shape[1], len(spectrum)) or up_raw.shape != (len(spectrum), weight.shape[0]):
            raise ValueError("full-weight allocation basis has incompatible dimensions")
        down = (down_raw / root[None]).T
        up = (up_raw / root[:, None]).T
    elif space == "key_space":
        encoder = factor["enc"].to(weight.device, torch.float32)
        decoder = factor["dec"].to(weight.device, torch.float32)
        down = (encoder / root[None]).T @ weight.detach().float()
        up = (decoder / root[:, None]).T
    else:
        raise ValueError(f"unknown basis parameter space {space!r}")
    return down, up, spectrum




@dataclass(frozen=True)
class BeltBlockSpec:
    module: str
    group: int
    projection: str
    capacity: int
    initial_rank: int
    min_rank: int = 1


class _ExactBudgetSigmoid(torch.autograd.Function):
    """Capped-simplex projection with an exact global mass constraint."""

    @staticmethod
    def forward(ctx, scores, budget, temperature):
        if scores.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        target = int(budget)
        count = int(scores.numel())
        if not 0 <= target <= count:
            raise ValueError(f"budget must be in [0, {count}], got {target}")
        tau = float(temperature)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be finite and positive")

        work = scores.detach().double()
        if target == 0:
            probabilities = torch.zeros_like(work)
        elif target == count:
            probabilities = torch.ones_like(work)
        else:
            tau_tensor = work.new_tensor(tau)
            low = work.min() - 64 * tau_tensor
            high = work.max() + 64 * tau_tensor
            for _ in range(48):
                midpoint = (low + high) * 0.5
                mass = torch.sigmoid((work - midpoint) / tau_tensor).sum()
                low = torch.where(mass > target, midpoint, low)
                high = torch.where(mass > target, high, midpoint)
            threshold = (low + high) * 0.5
            for _ in range(2):
                probabilities = torch.sigmoid((work - threshold) / tau_tensor)
                slope = (probabilities * (1 - probabilities)).sum()
                threshold += (
                    (probabilities.sum() - target)
                    * tau_tensor
                    / slope.clamp_min(torch.finfo(work.dtype).tiny)
                )
            probabilities = torch.sigmoid((work - threshold) / tau_tensor)

        derivative = probabilities * (1 - probabilities)
        ctx.save_for_backward(derivative)
        ctx.temperature = tau
        ctx.input_dtype = scores.dtype
        return probabilities

    @staticmethod
    def backward(ctx, grad_output):
        (derivative,) = ctx.saved_tensors
        denominator = derivative.sum().clamp_min(
            torch.finfo(derivative.dtype).tiny
        )
        mean = (grad_output.double() * derivative).sum() / denominator
        gradient = derivative * (grad_output.double() - mean) / ctx.temperature
        return gradient.to(ctx.input_dtype), None, None


def exact_budget_soft_mask(
    scores: torch.Tensor, budget: int, temperature: float
) -> torch.Tensor:
    return _ExactBudgetSigmoid.apply(scores, int(budget), float(temperature))


class ExactBudgetPrefixSelector(nn.Module):
    """One learned cutoff per block with exact-budget prefix selection."""

    def __init__(
        self,
        capacities: Sequence[int],
        initial_ranks: Sequence[int],
        *,
        budget: int,
        min_ranks: int | Sequence[int] = 1,
        temperature: float,
    ) -> None:
        super().__init__()
        capacities = tuple(map(int, capacities))
        initial_ranks = tuple(map(int, initial_ranks))
        minimums = (
            (int(min_ranks),) * len(capacities)
            if isinstance(min_ranks, int)
            else tuple(map(int, min_ranks))
        )
        if not capacities or not (
            len(capacities) == len(initial_ranks) == len(minimums)
        ):
            raise ValueError("capacities, ranks, and minimums must be aligned")
        for capacity, rank, minimum in zip(
            capacities, initial_ranks, minimums, strict=True
        ):
            if not 0 <= minimum <= rank <= capacity:
                raise ValueError("every block requires 0 <= min <= rank <= capacity")
        if sum(initial_ranks) != int(budget):
            raise ValueError("initial ranks must sum to the deployment budget")

        shifts = []
        block_indices = []
        positions = []
        offsets = [0]
        for block, (capacity, rank, minimum) in enumerate(
            zip(capacities, initial_ranks, minimums, strict=True)
        ):
            shifts.append(rank - 0.5)
            for position in range(minimum, capacity):
                block_indices.append(block)
                positions.append(position)
            offsets.append(len(positions))

        self.block_shifts = nn.Parameter(torch.tensor(shifts, dtype=torch.float32))
        self.register_buffer(
            "optional_block_indices",
            torch.tensor(block_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "optional_positions",
            torch.tensor(positions, dtype=torch.float32),
            persistent=False,
        )
        self.capacities = capacities
        self.min_ranks = minimums
        self.offsets = tuple(offsets)
        self.budget = int(budget)
        self.optional_budget = self.budget - sum(minimums)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self._active_masks: tuple[torch.Tensor, ...] | None = None

    def optional_scores(self) -> torch.Tensor:
        return self.block_shifts[self.optional_block_indices] - self.optional_positions

    def _hard_optional_mask(self, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(scores.numel(), device=scores.device, dtype=torch.float64)
        if self.optional_budget:
            mask[torch.topk(scores.detach(), self.optional_budget, sorted=False).indices] = 1
        return mask

    def _split(self, optional: torch.Tensor) -> tuple[torch.Tensor, ...]:
        masks = []
        for block, (capacity, minimum) in enumerate(
            zip(self.capacities, self.min_ranks, strict=True)
        ):
            required = torch.ones(minimum, device=optional.device, dtype=optional.dtype)
            mask = torch.cat((required, optional[self.offsets[block] : self.offsets[block + 1]]))
            if mask.numel() != capacity:
                raise RuntimeError("BELT block offsets are inconsistent")
            masks.append(mask)
        return tuple(masks)

    def block_masks(self, mode: str | None = None) -> tuple[torch.Tensor, ...]:
        mode = ("ste" if self.training else "hard") if mode is None else mode
        if mode not in {"hard", "soft", "ste"}:
            raise ValueError(f"unknown mask mode {mode!r}")
        scores = self.optional_scores()
        hard = self._hard_optional_mask(scores)
        if mode == "hard":
            optional = hard
        else:
            soft = exact_budget_soft_mask(
                scores, self.optional_budget, self.temperature
            )
            optional = soft if mode == "soft" else soft + (hard - soft).detach()
        return self._split(optional)

    def begin_forward(self) -> None:
        if self._active_masks is not None:
            raise RuntimeError("BELT forward session is already active")
        self._active_masks = self.block_masks()

    def end_forward(self) -> None:
        if self._active_masks is None:
            raise RuntimeError("BELT forward session is not active")
        self._active_masks = None

    def mask_for_block(self, index: int) -> torch.Tensor:
        if self._active_masks is not None:
            return self._active_masks[int(index)]
        return self.block_masks()[int(index)]

    @torch.no_grad()
    def hard_ranks(self) -> tuple[int, ...]:
        return tuple(int(mask.sum()) for mask in self.block_masks("hard"))


def build_block_specs(bases: Mapping[str, Sequence[Mapping]], *, retain_ratio=0.5) -> list[BeltBlockSpec]:
    if not math.isfinite(retain_ratio) or not 0 < retain_ratio <= 1:
        raise ValueError("retain_ratio must be in (0, 1]")
    blocks = []
    for module_name in sorted(bases):
        projection = module_name.rsplit(".", 1)[1]
        for group, factor in enumerate(bases[module_name]):
            capacity = int(factor["alloc_spectrum"].numel())
            blocks.append(
                BeltBlockSpec(
                    module_name, group, projection, capacity, max(1, int(capacity * retain_ratio))
                )
            )
    target = int(sum(b.capacity for b in blocks) * retain_ratio)
    if not blocks or target < len(blocks):
        raise ValueError("rank budget cannot give every block at least one rank")
    # Proportional integer initialization, deterministic across processes.
    ranks = [b.initial_rank for b in blocks]
    while sum(ranks) != target:
        add = sum(ranks) < target
        eligible = [i for i, b in enumerate(blocks)
                    if (ranks[i] < b.capacity if add else ranks[i] > b.min_rank)]
        index = max(eligible, key=lambda i: ((blocks[i].capacity * retain_ratio - ranks[i])
                    if add else (ranks[i] - blocks[i].capacity * retain_ratio), -i))
        ranks[index] += 1 if add else -1
    return [BeltBlockSpec(b.module, b.group, b.projection, b.capacity, rank, b.min_rank)
            for b, rank in zip(blocks, ranks, strict=True)]


class _BeltGroup(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        factor: Mapping[str, torch.Tensor],
        bias: torch.Tensor | None,
        selector: ExactBudgetPrefixSelector,
        block_index: int,
    ) -> None:
        super().__init__()
        down, up, spectrum = allocation_basis_components(weight, factor)
        self.register_buffer("down", down.to(weight.dtype).contiguous())
        self.register_buffer("up", up.to(weight.dtype).contiguous())
        self.register_buffer("spectrum", spectrum)
        self.register_buffer(
            "bias", None if bias is None else bias.detach().to(weight.dtype)
        )
        object.__setattr__(self, "_selector_ref", weakref.ref(selector))
        self.block_index = int(block_index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = F.linear(inputs, self.down)
        selector = self._selector_ref()
        if selector is None:
            raise RuntimeError("BELT selector was released")
        mask = selector.mask_for_block(self.block_index).to(
            latent.device, latent.dtype
        )
        return F.linear(
            latent * self.spectrum.to(latent.device, latent.dtype) * mask,
            self.up,
            self.bias,
        )


class BehaviorBeltLinear(nn.Module):
    def __init__(self, groups: Sequence[_BeltGroup]) -> None:
        super().__init__()
        self.groups = nn.ModuleList(groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([group(inputs) for group in self.groups], dim=-1)


def install_belt_modules(
    model,
    bases: Mapping[str, Sequence[Mapping]],
    *,
    temperature: float = 4.0,
    retain_ratio: float = 0.5,
) -> tuple[ExactBudgetPrefixSelector, list[BeltBlockSpec]]:
    if hasattr(model, "_belt_selector"):
        raise ValueError("rank allocation is already installed")
    from .basis import validate_bases
    validate_bases(model, bases)
    blocks = build_block_specs(bases, retain_ratio=retain_ratio)
    selector = ExactBudgetPrefixSelector(
        [block.capacity for block in blocks],
        [block.initial_rank for block in blocks],
        budget=sum(block.initial_rank for block in blocks),
        min_ranks=[block.min_rank for block in blocks],
        temperature=temperature,
    )
    indices: dict[str, list[int]] = {}
    for index, block in enumerate(blocks):
        indices.setdefault(block.module, []).append(index)

    for module_name, factors in bases.items():
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        original = getattr(parent, child_name)
        group_dim = original.out_features // len(factors)
        weights = original.weight.detach().reshape(
            len(factors), group_dim, original.in_features
        )
        biases = (
            None
            if original.bias is None
            else original.bias.detach().reshape(len(factors), group_dim)
        )
        groups = [
            _BeltGroup(
                weights[group],
                factor,
                None if biases is None else biases[group],
                selector,
                indices[module_name][group],
            )
            for group, factor in enumerate(factors)
        ]
        setattr(parent, child_name, BehaviorBeltLinear(groups).to(original.weight.device))

    selector = selector.to(next(model.parameters()).device)
    model.add_module("_belt_selector", selector)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selector.parameters():
        parameter.requires_grad_(True)

    model.register_forward_pre_hook(lambda _module, _args: selector.begin_forward())

    def end_forward(_module, _args, output):
        if selector._active_masks is not None:
            selector.end_forward()
        return output

    model.register_forward_hook(end_forward, always_call=True)
    return selector, blocks


def selected_rank_map(
    ranks: Sequence[int], blocks: Sequence[BeltBlockSpec]
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for rank, block in zip(ranks, blocks, strict=True):
        result.setdefault(block.module, []).append(int(rank))
    return result


def rank_summary(ranks: Sequence[int], blocks: Sequence[BeltBlockSpec]) -> dict:
    result = {}
    for projection in ("k_proj", "v_proj"):
        selected = sum(
            rank for rank, block in zip(ranks, blocks, strict=True)
            if block.projection == projection
        )
        capacity = sum(
            block.capacity for block in blocks if block.projection == projection
        )
        result[projection] = {
            "rank": selected,
            "capacity": capacity,
            "retain_ratio": selected / capacity,
        }
    result["combined"] = {
        "rank": sum(ranks),
        "capacity": sum(block.capacity for block in blocks),
        "retain_ratio": sum(ranks) / sum(block.capacity for block in blocks),
    }
    return result
