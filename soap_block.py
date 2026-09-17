"""Experimental SOAP with covariance tracked in a moving block-Jacobi basis.

The dense ``SOAP`` tracker forms ``Q.T @ C @ Q`` and retracts a dense tangent
step.  ``BlockSOAP`` instead stores that matrix directly as ``B`` and changes
coordinates with a small set of disjoint exact 2x2 rotations.  A block refresh
therefore preserves orthogonality without QR or Newton--Schulz.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from soap import SOAP, _offdiag, _sym


RotationBatch = tuple[Tensor, Tensor, Tensor, Tensor]
BlockBatch = tuple[Tensor, Tensor]


def _rotate_axis(
    tensor: Tensor,
    axis: int,
    rotations: Optional[RotationBatch],
    *,
    squared: bool = False,
) -> Tensor:
    """Apply T.T along one axis for a list of disjoint plane rotations."""
    if rotations is None or rotations[0].numel() == 0:
        return tensor
    first, second, cosine, sine = rotations
    result = tensor.clone()
    old_first = tensor.index_select(axis, first)
    old_second = tensor.index_select(axis, second)
    coefficient_shape = [1] * tensor.ndim
    coefficient_shape[axis] = first.numel()
    cosine = cosine.reshape(coefficient_shape)
    sine = sine.reshape(coefficient_shape)
    if squared:
        cosine_sq = cosine.square()
        sine_sq = sine.square()
        new_first = cosine_sq * old_first + sine_sq * old_second
        new_second = sine_sq * old_first + cosine_sq * old_second
    else:
        new_first = cosine * old_first + sine * old_second
        new_second = -sine * old_first + cosine * old_second
    result.index_copy_(axis, first, new_first)
    result.index_copy_(axis, second, new_second)
    return result


def _select_block_rotations(
    basis_covariance: Tensor,
    pairs_per_refresh: int,
    angle_cap: float,
    min_correlation: float,
    candidate_multiplier: int,
    selection: str = "top_correlation",
    round_index: int = 0,
) -> Optional[RotationBatch]:
    """Choose disjoint high-correlation pairs and return Jacobi rotations."""
    size = basis_covariance.shape[0]
    requested_pairs = int(pairs_per_refresh)
    max_pairs = size // 2 if requested_pairs == 0 else min(requested_pairs, size // 2)
    if max_pairs <= 0 or size < 2:
        return None

    b = _sym(basis_covariance.float())
    diagonal = b.diagonal().clamp_min(0.0)
    if selection == "round_robin":
        # Circle scheduling visits every pair in size-1 rounds (with one dummy
        # participant for odd sizes) and never reads scores back to the host.
        participant_count = size if size % 2 == 0 else size + 1
        dummy = size if participant_count != size else None
        ring = list(range(participant_count - 1))
        rounds = len(ring)
        shift = int(round_index) % rounds
        ring = ring[shift:] + ring[:shift]
        participants = [participant_count - 1] + ring
        scheduled = [
            (participants[index], participants[-1 - index])
            for index in range(participant_count // 2)
        ]
        scheduled = [
            (i, j) for i, j in scheduled if i != dummy and j != dummy
        ]
        windows = max(1, math.ceil(len(scheduled) / max_pairs))
        window = (int(round_index) // rounds) % windows
        scheduled = scheduled[window * max_pairs : (window + 1) * max_pairs]
        selected_first = [pair[0] for pair in scheduled]
        selected_second = [pair[1] for pair in scheduled]
    elif selection == "top_correlation":
        denominator = (
            diagonal.sqrt().unsqueeze(1) * diagonal.sqrt().unsqueeze(0)
        ).clamp_min(1e-12)
        correlation = (b / denominator).abs()
        scores = torch.triu(correlation, diagonal=1)
        candidate_count = min(
            size * (size - 1) // 2,
            max_pairs * max(1, int(candidate_multiplier)),
        )
        values, indices = scores.flatten().topk(candidate_count)
        candidate_values = values.tolist()
        candidate_indices = indices.tolist()
        used: set[int] = set()
        selected_first = []
        selected_second = []
        for score, flat_index in zip(candidate_values, candidate_indices):
            if score < min_correlation:
                break
            i, j = divmod(flat_index, size)
            if i == j or i in used or j in used:
                continue
            selected_first.append(i)
            selected_second.append(j)
            used.add(i)
            used.add(j)
            if len(selected_first) == max_pairs:
                break
    else:
        raise ValueError(f"Unknown pair selection: {selection!r}")
    if not selected_first:
        return None
    first = torch.tensor(selected_first, device=b.device, dtype=torch.long)
    second = torch.tensor(selected_second, device=b.device, dtype=torch.long)
    bij = b[first, second]
    theta = 0.5 * torch.atan2(
        2.0 * bij,
        b[first, first] - b[second, second],
    )
    # The diagonalizing angle is periodic by pi/2. Choose the nearest branch so
    # clipping moves in a descent direction rather than toward the far solution.
    theta = torch.where(theta > math.pi / 4, theta - math.pi / 2, theta)
    theta = torch.where(theta < -math.pi / 4, theta + math.pi / 2, theta)
    theta = theta.clamp(min=-float(angle_cap), max=float(angle_cap))
    if selection == "round_robin" and min_correlation > 0.0:
        selected_scale = (
            diagonal[first].sqrt() * diagonal[second].sqrt()
        ).clamp_min(1e-12)
        selected_correlation = b[first, second].abs() / selected_scale
        theta = torch.where(
            selected_correlation >= float(min_correlation),
            theta,
            torch.zeros_like(theta),
        )
    return first, second, theta.cos(), theta.sin()


def _rotate_basis_covariance(
    basis_covariance: Tensor,
    rotations: Optional[RotationBatch],
) -> Tensor:
    """Apply B <- T.T B T for disjoint rotations."""
    return _sym(_rotate_axis(_rotate_axis(basis_covariance, 0, rotations), 1, rotations))


def _apply_block_axis(
    tensor: Tensor,
    axis: int,
    blocks: Optional[BlockBatch],
    *,
    squared: bool = False,
) -> Tensor:
    """Apply block-diagonal T.T along an axis."""
    if blocks is None:
        return tensor
    indices, transitions = blocks
    if indices.numel() == 0:
        return tensor
    moved = tensor.movedim(axis, 0)
    selected = moved[indices]
    batch, width = indices.shape
    selected_flat = selected.reshape(batch, width, -1)
    operators = transitions.square() if squared else transitions
    transformed = torch.bmm(operators.transpose(1, 2), selected_flat)
    transformed = transformed.reshape_as(selected)
    result = moved.clone()
    result[indices] = transformed
    return result.movedim(0, axis)


def _select_cayley_blocks(
    basis_covariance: Tensor,
    block_size: int,
    step_size: float,
    damping: float,
    rotation_cap: float,
    min_correlation: float,
    unbiased_scale: bool = True,
    round_index: int = 0,
    seed: int = 0,
) -> Optional[BlockBatch]:
    """Build reproducibly shuffled blocks and near-identity Cayley transforms."""
    size = basis_covariance.shape[0]
    width = min(int(block_size), size)
    block_count = size // width
    if width < 2 or block_count == 0:
        return None
    generator = torch.Generator(device=basis_covariance.device)
    generator.manual_seed((int(seed) + int(round_index)) % (2**63 - 1))
    indices = torch.randperm(
        size,
        device=basis_covariance.device,
        generator=generator,
    )
    indices = indices[: block_count * width].reshape(block_count, width)
    b = _sym(basis_covariance.float())
    local = b[indices.unsqueeze(2), indices.unsqueeze(1)]
    diagonal = local.diagonal(dim1=-2, dim2=-1)
    gap = diagonal.unsqueeze(2) - diagonal.unsqueeze(1)
    tau = diagonal.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    tau = tau.clamp_min(1e-12) * float(damping)
    rotation = -local * gap / (gap.square() + tau.square()).clamp_min(1e-24)
    rotation = 0.5 * (rotation - rotation.transpose(1, 2))
    sampling_scale = (
        (size - 1) / (width - 1)
        if unbiased_scale and width < size
        else 1.0
    )
    rotation = rotation * (float(step_size) * sampling_scale)

    if min_correlation > 0.0:
        scale = (
            diagonal.clamp_min(0.0).sqrt().unsqueeze(2)
            * diagonal.clamp_min(0.0).sqrt().unsqueeze(1)
        ).clamp_min(1e-12)
        correlation = (local - torch.diag_embed(diagonal)) / scale
        active = correlation.abs().amax(dim=(-2, -1)) >= float(min_correlation)
        rotation = rotation * active.to(rotation.dtype).reshape(-1, 1, 1)

    if rotation_cap > 0.0:
        spectral_bound = rotation.abs().sum(dim=-1).amax(dim=-1)
        cap = rotation.new_tensor(float(rotation_cap))
        factor = (cap / spectral_bound.clamp_min(1e-30)).clamp(max=1.0)
        rotation = rotation * factor.reshape(-1, 1, 1)

    identity = torch.eye(width, device=rotation.device, dtype=rotation.dtype)
    identity = identity.expand(block_count, -1, -1)
    # Cayley(rotation) = (I - rotation/2)^-1 (I + rotation/2), which is
    # orthogonal for skew rotation and agrees with I + rotation to first order.
    transitions = torch.linalg.solve(
        identity - 0.5 * rotation,
        identity + 0.5 * rotation,
    )
    return indices, transitions


def _transform_basis_covariance_blocks(
    basis_covariance: Tensor,
    blocks: Optional[BlockBatch],
) -> Tensor:
    return _sym(
        _apply_block_axis(
            _apply_block_axis(basis_covariance, 0, blocks),
            1,
            blocks,
        )
    )


class BlockSOAP(SOAP):
    """SOAP with moving-basis covariance and exact block-Jacobi tracking.

    Extra arguments:
        basis_pairs_per_refresh: Maximum disjoint 2x2 rotations per factor;
            zero uses a complete parallel matching.
        basis_pair_angle_cap: Maximum absolute angle for any plane rotation.
        basis_pair_min_correlation: Ignore pairs below this normalized covariance.
        basis_pair_selection: ``"round_robin"`` avoids synchronization and
            eventually visits every pair; ``"top_correlation"`` greedily spends
            work on the largest current correlations.
        basis_pair_candidate_multiplier: Top candidates inspected per desired pair.
        basis_block_size: Tracker block width. Two uses exact Jacobi pairs;
            values above two use reproducibly shuffled blocks and damped Cayley
            updates.
        basis_block_unbiased_scale: Compensate for the probability that a pair
            appears in a random block; the block spectral cap still applies.
        basis_block_rotation_cap: Spectral bound for each sampled block's skew
            generator before applying its Cayley transform.
        basis_block_seed: Seed for the reproducible shuffled-block schedule.

    ``top_correlation`` selection transfers a short top-k candidate list to the
    host once per active factor refresh. This prototype favors clarity and
    correctness; a production kernel should keep selection and rotations fused.
    """

    def __init__(
        self,
        params,
        *args,
        basis_pairs_per_refresh: int = 0,
        basis_pair_angle_cap: float = math.pi / 8.0,
        basis_pair_min_correlation: float = 1e-3,
        basis_pair_selection: str = "round_robin",
        basis_pair_candidate_multiplier: int = 8,
        basis_block_size: int = 2,
        basis_block_unbiased_scale: bool = True,
        basis_block_rotation_cap: float = 0.25,
        basis_block_seed: int = 0,
        **kwargs,
    ):
        if int(basis_pairs_per_refresh) < 0:
            raise ValueError("basis_pairs_per_refresh must be >= 0")
        if not 0.0 < float(basis_pair_angle_cap) <= math.pi / 4.0:
            raise ValueError("basis_pair_angle_cap must be in (0, pi/4]")
        if float(basis_pair_min_correlation) < 0.0:
            raise ValueError("basis_pair_min_correlation must be non-negative")
        if basis_pair_selection not in ("round_robin", "top_correlation"):
            raise ValueError(
                "basis_pair_selection must be 'round_robin' or 'top_correlation'"
            )
        if int(basis_pair_candidate_multiplier) < 1:
            raise ValueError("basis_pair_candidate_multiplier must be >= 1")
        if int(basis_block_size) < 2:
            raise ValueError("basis_block_size must be >= 2")
        if float(basis_block_rotation_cap) < 0.0:
            raise ValueError("basis_block_rotation_cap must be non-negative")
        super().__init__(params, *args, **kwargs)
        for group in self.param_groups:
            group.setdefault("basis_pairs_per_refresh", int(basis_pairs_per_refresh))
            group.setdefault("basis_pair_angle_cap", float(basis_pair_angle_cap))
            group.setdefault(
                "basis_pair_min_correlation", float(basis_pair_min_correlation)
            )
            group.setdefault("basis_pair_selection", basis_pair_selection)
            group.setdefault(
                "basis_pair_candidate_multiplier",
                int(basis_pair_candidate_multiplier),
            )
            group.setdefault("basis_block_size", int(basis_block_size))
            group.setdefault(
                "basis_block_unbiased_scale", bool(basis_block_unbiased_scale)
            )
            group.setdefault(
                "basis_block_rotation_cap", float(basis_block_rotation_cap)
            )
            group.setdefault("basis_block_seed", int(basis_block_seed))

    def _cold_start_basis(self, state: dict) -> None:
        bases: list[Optional[Tensor]] = []
        moving_covariances: list[Optional[Tensor]] = []
        for covariance in state["GG"]:
            if covariance is None:
                bases.append(None)
                moving_covariances.append(None)
                continue
            q = self._eigh_basis(covariance)
            bases.append(q)
            moving_covariances.append(_sym(q.T @ covariance.float() @ q))
        state["Q"] = bases
        state["GG"] = moving_covariances

    def _accumulate_projected_covariance(
        self,
        projected_grad: Tensor,
        state: dict,
        group: dict,
    ) -> None:
        # Parent accumulation is coordinate-agnostic; here both its input and
        # stored factors are already expressed in the current basis.
        super()._accumulate_covariance(projected_grad, state, group)

    def _rotate_optimizer_tensor(
        self,
        tensor: Tensor,
        all_rotations: list[Optional[RotationBatch]],
        group: dict,
        *,
        squared: bool,
    ) -> Tensor:
        working, original_shape, permuted_shape = self._prepare_transform(tensor, group)
        for axis, rotations in enumerate(all_rotations):
            if rotations is not None:
                working = _rotate_axis(working, axis, rotations, squared=squared)
        return self._restore_transform(
            working,
            original_shape,
            permuted_shape,
            group["merge_dims"],
        )

    def _transform_optimizer_tensor_blocks(
        self,
        tensor: Tensor,
        all_blocks: list[Optional[BlockBatch]],
        group: dict,
        *,
        squared: bool,
    ) -> Tensor:
        working, original_shape, permuted_shape = self._prepare_transform(tensor, group)
        for axis, blocks in enumerate(all_blocks):
            if blocks is not None:
                working = _apply_block_axis(working, axis, blocks, squared=squared)
        return self._restore_transform(
            working,
            original_shape,
            permuted_shape,
            group["merge_dims"],
        )

    @staticmethod
    def _moving_basis_metrics(
        b: Tensor,
        q: Tensor,
        q_old: Tensor,
    ) -> dict[str, float]:
        offdiag = _offdiag(b.float())
        diagonal = b.diagonal().float().clamp_min(1e-12)
        correlation = offdiag / (
            diagonal.sqrt().unsqueeze(0) * diagonal.sqrt().unsqueeze(1)
        )
        count = max(1, b.numel() - b.shape[0])
        identity = torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
        return {
            "diag_err": float(
                offdiag.norm() / b.float().norm().clamp_min(1e-12)
            ),
            "corr_err": float((correlation.square().sum() / count).sqrt()),
            "orth_err": float(
                (q.T @ q - identity).norm() / math.sqrt(max(1, q.shape[1]))
            ),
            "basis_drift": float(
                (q - q_old).norm() / math.sqrt(max(1, q.shape[1]))
            ),
        }

    def _dense_reset_in_moving_basis(
        self,
        b: Tensor,
        q_old: Tensor,
        method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if method == "eigh":
            transition = self._eigh_basis(b)
        else:
            transition, r = torch.linalg.qr(b.float(), mode="reduced")
            signs = r.diagonal().sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            transition = transition * signs.unsqueeze(0)
        q_new = q_old.float() @ transition
        b_new = _sym(transition.T @ b.float() @ transition)
        return q_new, b_new, transition

    def _refresh_basis(self, state: dict, group: dict) -> None:
        old_basis = state["Q"]
        state["basis_refreshes"] += 1
        reset_frequency = group["basis_reset_frequency"]
        hard_reset = (
            reset_frequency > 0
            and state["basis_refreshes"] % reset_frequency == 0
        )
        collect_stats = (
            group["basis_track_stats"]
            and state["basis_refreshes"] % group["basis_track_stats_frequency"] == 0
        )

        new_basis: list[Optional[Tensor]] = []
        new_covariances: list[Optional[Tensor]] = []
        all_rotations: list[Optional[RotationBatch]] = []
        all_blocks: list[Optional[BlockBatch]] = []
        dense_transitions: list[Optional[Tensor]] = []

        for b, q_old in zip(state["GG"], old_basis):
            if b is None or q_old is None:
                new_basis.append(None)
                new_covariances.append(None)
                all_rotations.append(None)
                all_blocks.append(None)
                dense_transitions.append(None)
                continue

            if hard_reset:
                q_new, b_new, transition = self._dense_reset_in_moving_basis(
                    b, q_old, group["basis_reset_method"]
                )
                rotations = None
            else:
                if group["basis_block_size"] == 2:
                    rotations = _select_block_rotations(
                        b,
                        group["basis_pairs_per_refresh"],
                        group["basis_pair_angle_cap"],
                        group["basis_pair_min_correlation"],
                        group["basis_pair_candidate_multiplier"],
                        group["basis_pair_selection"],
                        state["basis_refreshes"] - 1,
                    )
                    blocks = None
                    q_new = _rotate_axis(q_old.float(), 1, rotations)
                    b_new = _rotate_basis_covariance(b, rotations)
                else:
                    rotations = None
                    blocks = _select_cayley_blocks(
                        b,
                        group["basis_block_size"],
                        group["basis_lr"],
                        group["basis_jacobi_damping"],
                        group["basis_block_rotation_cap"],
                        group["basis_pair_min_correlation"],
                        group["basis_block_unbiased_scale"],
                        state["basis_refreshes"] - 1,
                        group["basis_block_seed"],
                    )
                    q_new = _apply_block_axis(q_old.float(), 1, blocks)
                    b_new = _transform_basis_covariance_blocks(b, blocks)
                transition = None

            new_basis.append(q_new.float())
            new_covariances.append(b_new.float())
            all_rotations.append(rotations)
            all_blocks.append(None if hard_reset else blocks)
            dense_transitions.append(transition)
            if collect_stats:
                metrics = self._moving_basis_metrics(b_new, q_new, q_old.float())
                metrics["hard_reset"] = float(hard_reset)
                if self._basis_stats_accum is None:
                    self._basis_stats_accum = {"count": 0.0}
                self._basis_stats_accum["count"] += 1.0
                for key, value in metrics.items():
                    self._basis_stats_accum[key] = (
                        self._basis_stats_accum.get(key, 0.0) + value
                    )

        if hard_reset:
            state["exp_avg"] = self._project_with_basis(
                state["exp_avg"], dense_transitions, group
            ).float()
            state["exp_avg_sq"] = self._project_with_basis(
                state["exp_avg_sq"],
                [
                    None if transition is None else transition.square()
                    for transition in dense_transitions
                ],
                group,
            ).clamp_min_(0.0).float()
        else:
            if group["basis_block_size"] == 2:
                state["exp_avg"] = self._rotate_optimizer_tensor(
                    state["exp_avg"], all_rotations, group, squared=False
                ).float()
            else:
                state["exp_avg"] = self._transform_optimizer_tensor_blocks(
                    state["exp_avg"], all_blocks, group, squared=False
                ).float()
            if group["transport_second_moment_on_warm"]:
                if group["basis_block_size"] == 2:
                    state["exp_avg_sq"] = self._rotate_optimizer_tensor(
                        state["exp_avg_sq"], all_rotations, group, squared=True
                    ).clamp_min_(0.0).float()
                else:
                    state["exp_avg_sq"] = self._transform_optimizer_tensor_blocks(
                        state["exp_avg_sq"], all_blocks, group, squared=True
                    ).clamp_min_(0.0).float()

        state["Q"] = new_basis
        state["GG"] = new_covariances

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._basis_stats_accum = None
        self.latest_basis_stats = {}
        self.just_gathered_basis_stats = False

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("BlockSOAP does not support sparse gradients")

                grad = parameter.grad.detach()
                state = self.state[parameter]
                if "step" not in state:
                    self._init_state(grad, state, group)
                    # The initial observation is in world coordinates; cold
                    # start diagonalizes it and converts GG to moving coordinates.
                    super()._accumulate_covariance(grad, state, group)
                    self._cold_start_basis(state)
                    continue

                state["step"] += 1
                projected_grad = self._project_with_basis(
                    grad, state["Q"], group
                ).float()
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(projected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    projected_grad, projected_grad, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                update = exp_avg / bias_correction1
                update = update / (
                    exp_avg_sq / bias_correction2
                ).sqrt().add_(group["eps"])
                update = self._project_back_with_basis(update, state["Q"], group)
                if group["normalize_grads"]:
                    update = update / update.square().mean().sqrt().clamp_min(1e-30)
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

                self._accumulate_projected_covariance(projected_grad, state, group)
                if state["step"] % group["precondition_frequency"] == 0:
                    self._refresh_basis(state, group)

        self._finalize_stats()
        return loss


TurboBlockSOAP = BlockSOAP


__all__ = ["BlockSOAP", "TurboBlockSOAP"]
