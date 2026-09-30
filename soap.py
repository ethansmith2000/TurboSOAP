"""TurboSOAP: SOAP with a slim, warm-started eigenbasis tracker.

For each eligible tensor axis the optimizer maintains a covariance EMA ``C``
and a full orthogonal basis ``Q``. Gradients are projected into those bases,
Adam is applied elementwise, and the update is projected back.

Cold start uses ``eigh``. Between optional hard resets, each basis follows the
double-bracket (off-diagonal gauge) flow for

    0.5 * ||offdiag(Q.T @ C @ Q)||_F^2.

The pairwise flow is damped by the squared Rayleigh-quotient gap, producing a
regularized Jacobi step without the inverse-gap noise amplification of the old
tracker. Tracker steps are clipped but never normalized upward, so they vanish
as the covariance becomes diagonal in the tracked basis.
"""

from __future__ import annotations

import math
from itertools import chain
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


_NS_MAX_SINGULAR = 1.25


def _sym(x: Tensor) -> Tensor:
    return 0.5 * (x + x.T)


def _offdiag(x: Tensor) -> Tensor:
    return x - torch.diag_embed(x.diagonal())


def _orthogonalize_ns(x: Tensor, iterations: int = 2) -> Tensor:
    """Polar retraction using Higham's cubic Newton--Schulz iteration."""
    if x.numel() == 0:
        return x.float()

    z = x.float()
    gram = z.T @ z
    # ||gram||_inf bounds its largest eigenvalue for symmetric gram.
    upper_sq = gram.abs().sum(dim=-1).amax().clamp_min(1e-12)
    scale = (_NS_MAX_SINGULAR / upper_sq.sqrt()).clamp(max=1.0)
    z = z * scale
    gram = gram * scale.square()

    num_iterations = max(1, int(iterations))
    for iteration in range(num_iterations):
        gram_sq = gram @ gram
        correction = -10.0 * gram + 3.0 * gram_sq
        z = 0.125 * (15.0 * z + z @ correction)
        if iteration + 1 < num_iterations:
            gram = z.T @ z
    return z


def _normalized_covariance(covariance: Tensor) -> Tensor:
    scale = (covariance.trace().abs() / max(1, covariance.shape[0])).clamp_min(1e-12)
    return covariance.float() / scale


def _damped_jacobi_generator(
    basis_covariance: Tensor,
    damping: float,
) -> Tensor:
    """Return a skew generator that descends off-diagonal covariance energy.

    For B = Q.T C Q, the raw double-bracket generator is

        Omega_ij = B_ij * (B_jj - B_ii).

    Dividing by ``gap**2 + tau**2`` yields a damped first-order Jacobi angle.
    Unlike inverse-|gap| scaling, the numerator also vanishes with the gap, so
    near-degenerate directions are not amplified.
    """
    b = _sym(basis_covariance.float())
    diagonal = b.diagonal()
    gap = diagonal.unsqueeze(1) - diagonal.unsqueeze(0)
    tau = diagonal.abs().amax().clamp_min(1e-12) * float(damping)
    omega = -b * gap / (gap.square() + tau.square()).clamp_min(1e-24)
    return 0.5 * (omega - omega.T)


def _clip_rotation(
    rotation: Tensor,
    limit_value: float,
    mode: str = "average",
) -> Tensor:
    """Clip a rotation without scaling small steps up.

    ``average`` preserves the original average-column-norm policy.  The safer
    ``spectral_bound`` mode uses the maximum absolute row sum, which bounds the
    spectral norm of a skew-symmetric matrix but can be more conservative.
    """
    if rotation.numel() == 0 or limit_value <= 0.0:
        return rotation
    if mode == "average":
        magnitude = rotation.float().norm() / math.sqrt(max(1, rotation.shape[0]))
    elif mode == "spectral_bound":
        # For skew matrices, ||A||_1 == ||A||_inf and
        # ||A||_2 <= sqrt(||A||_1 ||A||_inf) == ||A||_inf.
        magnitude = rotation.float().abs().sum(dim=-1).amax()
    else:
        raise ValueError(f"Unknown rotation cap mode: {mode!r}")
    limit = rotation.new_tensor(float(limit_value))
    scale = (limit / magnitude.clamp_min(1e-30)).clamp(max=1.0)
    return rotation * scale.to(rotation.dtype)


def _largest_pair_jacobi_step(q: Tensor, b: Tensor, max_angle: float) -> Tensor:
    """Exactly rotate the most correlated 2x2 block of ``b``."""
    offdiag = _offdiag(b).abs()
    flat_index = offdiag.argmax()
    size = b.shape[0]
    i = flat_index // size
    j = flat_index % size
    bij = b[i, j]
    theta = 0.5 * torch.atan2(2.0 * bij, b[i, i] - b[j, j])
    theta = torch.where(theta > math.pi / 4, theta - math.pi / 2, theta)
    theta = torch.where(theta < -math.pi / 4, theta + math.pi / 2, theta)
    theta = theta.clamp(min=-float(max_angle), max=float(max_angle))
    cosine = theta.cos()
    sine = theta.sin()

    qi = q[:, i].clone()
    qj = q[:, j].clone()
    result = q.clone()
    result[:, i] = cosine * qi + sine * qj
    result[:, j] = -sine * qi + cosine * qj
    return result


def _basis_metrics(covariance: Tensor, q: Tensor, q_old: Optional[Tensor] = None) -> dict[str, float]:
    """Small diagnostic set for a full square covariance basis."""
    c = covariance.float()
    qf = q.float()
    b = _sym(qf.T @ c @ qf)
    off = _offdiag(b)
    c_norm_sq = c.square().sum().clamp_min(1e-24)
    diag_error = (off.square().sum() / c_norm_sq).sqrt()

    diagonal = b.diagonal().clamp_min(1e-12)
    corr = off / (diagonal.sqrt().unsqueeze(0) * diagonal.sqrt().unsqueeze(1))
    offdiag_count = max(1, b.numel() - b.shape[0])
    corr_error = (corr.square().sum() / offdiag_count).sqrt()

    identity = torch.eye(qf.shape[1], device=qf.device, dtype=qf.dtype)
    orth_error = (qf.T @ qf - identity).norm() / math.sqrt(max(1, qf.shape[1]))
    result = {
        "diag_err": float(diag_error),
        "corr_err": float(corr_error),
        "orth_err": float(orth_error),
    }
    if q_old is not None:
        result["basis_drift"] = float(
            (qf - q_old.float()).norm() / math.sqrt(max(1, qf.shape[1]))
        )
    return result


class SOAP(Optimizer):
    """SOAP with one warm basis tracker and periodic hard resets.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Outer Adam learning rate.
        betas: Adam first- and second-moment decay rates.
        shampoo_beta: Covariance EMA decay. ``-1`` uses ``betas[1]``.
        eps: Adam denominator epsilon.
        weight_decay: AdamW-style decoupled weight decay.
        precondition_frequency: Optimizer updates between basis refreshes.
        precondition_frequency_after_warmup: Optional fixed refresh frequency
            used after ``precondition_frequency_warmup_steps``. Zero keeps one
            fixed frequency throughout.
        precondition_frequency_warmup_steps: Optimizer updates to retain the
            base fixed frequency before switching to the optional later one.
        basis_residual_threshold: If positive, treat ``precondition_frequency``
            as the interval between residual checks and refresh a basis only
            when ``||offdiag(Q.T C Q)||_F / ||Q.T C Q||_F`` reaches this
            threshold. Disabled by default.
        basis_residual_max_age: Maximum optimizer updates between basis
            refreshes when residual gating is enabled. This bounds staleness
            after quiet checks. Must be at least ``precondition_frequency``.
        basis_residual_warmup_steps: Keep the fixed refresh cadence for this
            many optimizer updates before enabling residual gating.
        max_precond_dim: Axes larger than this are not preconditioned.
        merge_dims: Merge adjacent tensor dimensions before preconditioning.
        precondition_1d: Build a covariance basis for vector parameters.
        precondition_mode: ``"all"`` tracks every eligible factor;
            ``"smaller_side"`` tracks only the smaller axis of 2D tensors;
            ``"aspect_ratio"`` does so only when the larger axis is at least
            ``precondition_aspect_ratio`` times the smaller axis.
        precondition_aspect_ratio: Rectangularity threshold for
            ``precondition_mode="aspect_ratio"``.
        normalize_grads: RMS-normalize the final parameter update.
        data_format: ``"channels_first"`` or ``"channels_last"``.
        basis_lr: Step size for the damped Jacobi gauge flow.
        basis_lr_age_compensation: Increase ``basis_lr`` as the elapsed age
            between accepted refreshes grows. The compounded step is
            ``1 - (1 - basis_lr) ** (age / reference_age)`` and remains
            subject to ``basis_rotation_cap``.
        basis_lr_reference_age: Reference refresh age for compounded basis
            motion. Zero uses ``precondition_frequency``.
        basis_rotation_cap: Rotation limit interpreted according to
            ``basis_rotation_cap_mode``. Zero disables clipping.
        basis_rotation_cap_mode: ``"average"`` preserves the original cap;
            ``"spectral_bound"`` strictly controls a spectral-norm upper bound.
        basis_jacobi_damping: Relative Tikhonov damping for pairwise gaps.
        basis_stall_pair_threshold: If positive, use an exact capped 2x2 Jacobi
            step when the dense generator is this small relative to remaining
            off-diagonal covariance. This adds a device synchronization.
        basis_reset_frequency: Hard-reset interval measured in basis refreshes.
            ``0`` disables periodic resets. Cold start always uses ``eigh``.
        basis_reset_max_age: If positive, schedule subsequent hard resets by
            optimizer-step age instead of accepted-refresh count. The first
            reset retains the staggered window implied by
            ``basis_reset_frequency * precondition_frequency``.
        basis_reset_stagger: Spread each parameter's first reset across a
            centered one-period window, then retain the configured cadence.
            This smooths synchronized reset spikes without changing the
            long-run reset frequency.
        basis_reset_method: ``"qr"`` (one orthogonal iteration) or ``"eigh"``.
        basis_ns_iterations: Newton--Schulz iterations per gauge retraction.
        covariance_compute_dtype: Outer-product matmul dtype. On CPU, bf16
            requests use fp32. Covariance state itself always remains fp32.
        basis_track_stats: Populate ``latest_basis_stats`` on basis refreshes.
        basis_track_stats_frequency: Collect those statistics every N refreshes.
        transport_second_moment_on_warm: Apply squared-overlap diagonal variance
            transport after warm rotations as well as hard resets.

    Optimizer moments, covariance matrices, and master bases are kept in fp32.
    Momentum is transported only when Q actually changes. Elementwise second
    moments remain attached to continuously tracked columns; hard resets
    transport them through squared old/new basis overlaps.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.999),
        shampoo_beta: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        precondition_frequency_after_warmup: int = 0,
        precondition_frequency_warmup_steps: int = 0,
        basis_residual_threshold: float = 0.0,
        basis_residual_max_age: int = 0,
        basis_residual_warmup_steps: int = 0,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        precondition_mode: str = "all",
        precondition_aspect_ratio: float = 2.0,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        basis_lr: float = 0.5,
        basis_lr_age_compensation: bool = False,
        basis_lr_reference_age: int = 0,
        basis_rotation_cap: float = 0.1,
        basis_rotation_cap_mode: str = "average",
        basis_jacobi_damping: float = 1e-2,
        basis_stall_pair_threshold: float = 0.0,
        basis_reset_frequency: int = 50,
        basis_reset_max_age: int = 0,
        basis_reset_stagger: bool = False,
        basis_reset_method: str = "qr",
        basis_ns_iterations: int = 2,
        covariance_compute_dtype: str = "bfloat16",
        basis_track_stats: bool = True,
        basis_track_stats_frequency: int = 1,
        transport_second_moment_on_warm: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if len(betas) != 2 or not all(0.0 <= float(beta) < 1.0 for beta in betas):
            raise ValueError(f"betas must be two values in [0, 1), got {betas!r}")
        if shampoo_beta != -1.0 and not 0.0 <= float(shampoo_beta) < 1.0:
            raise ValueError("shampoo_beta must be -1 or in [0, 1)")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if int(precondition_frequency) < 1:
            raise ValueError("precondition_frequency must be >= 1")
        if int(precondition_frequency_after_warmup) < 0:
            raise ValueError(
                "precondition_frequency_after_warmup must be non-negative"
            )
        if int(precondition_frequency_warmup_steps) < 0:
            raise ValueError(
                "precondition_frequency_warmup_steps must be non-negative"
            )
        if (
            int(precondition_frequency_after_warmup) > 0
            and float(basis_residual_threshold) > 0.0
        ):
            raise ValueError(
                "fixed post-warmup frequency and residual gating are "
                "mutually exclusive"
            )
        if float(basis_residual_threshold) < 0.0:
            raise ValueError("basis_residual_threshold must be non-negative")
        if int(basis_residual_max_age) < 0:
            raise ValueError("basis_residual_max_age must be non-negative")
        if int(basis_residual_warmup_steps) < 0:
            raise ValueError("basis_residual_warmup_steps must be non-negative")
        if float(basis_residual_threshold) > 0.0 and int(
            basis_residual_max_age
        ) < int(precondition_frequency):
            raise ValueError(
                "basis_residual_max_age must be at least precondition_frequency "
                "when residual gating is enabled"
            )
        if int(max_precond_dim) < 1:
            raise ValueError("max_precond_dim must be >= 1")
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError("data_format must be 'channels_first' or 'channels_last'")
        if precondition_mode not in ("all", "smaller_side", "aspect_ratio"):
            raise ValueError(
                "precondition_mode must be 'all', 'smaller_side', or 'aspect_ratio'"
            )
        if float(precondition_aspect_ratio) < 1.0:
            raise ValueError("precondition_aspect_ratio must be >= 1")
        if basis_lr < 0.0:
            raise ValueError("basis_lr must be non-negative")
        if basis_lr_age_compensation and basis_lr >= 1.0:
            raise ValueError(
                "basis_lr must be below 1 when age compensation is enabled"
            )
        if int(basis_lr_reference_age) < 0:
            raise ValueError("basis_lr_reference_age must be non-negative")
        if basis_rotation_cap < 0.0:
            raise ValueError("basis_rotation_cap must be non-negative")
        if basis_rotation_cap_mode not in ("average", "spectral_bound"):
            raise ValueError(
                "basis_rotation_cap_mode must be 'average' or 'spectral_bound'"
            )
        if basis_jacobi_damping <= 0.0:
            raise ValueError("basis_jacobi_damping must be positive")
        if basis_stall_pair_threshold < 0.0:
            raise ValueError("basis_stall_pair_threshold must be non-negative")
        if int(basis_reset_frequency) < 0:
            raise ValueError("basis_reset_frequency must be >= 0")
        if int(basis_reset_max_age) < 0:
            raise ValueError("basis_reset_max_age must be non-negative")
        if int(basis_reset_max_age) > 0 and int(basis_reset_frequency) <= 0:
            raise ValueError(
                "basis_reset_frequency must be positive when "
                "basis_reset_max_age is enabled"
            )
        if basis_reset_method not in ("qr", "eigh"):
            raise ValueError("basis_reset_method must be 'qr' or 'eigh'")
        if int(basis_ns_iterations) < 1:
            raise ValueError("basis_ns_iterations must be >= 1")
        if covariance_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("covariance_compute_dtype must be 'float32' or 'bfloat16'")
        if int(basis_track_stats_frequency) < 1:
            raise ValueError("basis_track_stats_frequency must be >= 1")

        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            shampoo_beta=float(shampoo_beta),
            eps=float(eps),
            weight_decay=float(weight_decay),
            precondition_frequency=int(precondition_frequency),
            precondition_frequency_after_warmup=int(
                precondition_frequency_after_warmup
            ),
            precondition_frequency_warmup_steps=int(
                precondition_frequency_warmup_steps
            ),
            basis_residual_threshold=float(basis_residual_threshold),
            basis_residual_max_age=int(basis_residual_max_age),
            basis_residual_warmup_steps=int(basis_residual_warmup_steps),
            max_precond_dim=int(max_precond_dim),
            merge_dims=bool(merge_dims),
            precondition_1d=bool(precondition_1d),
            precondition_mode=precondition_mode,
            precondition_aspect_ratio=float(precondition_aspect_ratio),
            normalize_grads=bool(normalize_grads),
            basis_lr=float(basis_lr),
            basis_lr_age_compensation=bool(basis_lr_age_compensation),
            basis_lr_reference_age=int(basis_lr_reference_age),
            basis_rotation_cap=float(basis_rotation_cap),
            basis_rotation_cap_mode=basis_rotation_cap_mode,
            basis_jacobi_damping=float(basis_jacobi_damping),
            basis_stall_pair_threshold=float(basis_stall_pair_threshold),
            basis_reset_frequency=int(basis_reset_frequency),
            basis_reset_max_age=int(basis_reset_max_age),
            basis_reset_stagger=bool(basis_reset_stagger),
            basis_reset_method=basis_reset_method,
            basis_ns_iterations=int(basis_ns_iterations),
            covariance_compute_dtype=covariance_compute_dtype,
            basis_track_stats=bool(basis_track_stats),
            basis_track_stats_frequency=int(basis_track_stats_frequency),
            transport_second_moment_on_warm=bool(transport_second_moment_on_warm),
        )
        super().__init__(params, defaults)
        self._data_format = data_format
        self.latest_basis_stats: dict[str, float] = {}
        self.just_gathered_basis_stats = False
        self._basis_stats_accum: Optional[dict[str, float]] = None
        self.last_basis_refreshes = 0
        self.last_hard_reset_events = 0
        self._basis_reset_first_refresh: dict[int, int] = {}
        for group in self.param_groups:
            reset_frequency = int(group["basis_reset_frequency"])
            eligible = [
                parameter
                for parameter in group["params"]
                if self._parameter_has_preconditioner(parameter, group)
            ]
            for index, parameter in enumerate(eligible):
                first_refresh = reset_frequency
                if group["basis_reset_stagger"] and reset_frequency > 1:
                    first_refresh = (
                        reset_frequency // 2
                        + 1
                        + (index * reset_frequency) // len(eligible)
                    )
                self._basis_reset_first_refresh[id(parameter)] = first_refresh

    # ------------------------------------------------------------------
    # Tensor layout and basis transforms
    # ------------------------------------------------------------------

    def _merge_dims(self, tensor: Tensor, max_precond_dim: int) -> Tensor:
        if self._data_format == "channels_last" and tensor.dim() == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        new_shape: list[int] = []
        current = 1
        for size in tensor.shape:
            candidate = current * int(size)
            if candidate > max_precond_dim:
                if current > 1:
                    new_shape.append(current)
                    current = int(size)
                else:
                    new_shape.append(int(size))
                    current = 1
            else:
                current = candidate
        if current > 1 or not new_shape:
            new_shape.append(current)
        return tensor.reshape(new_shape)

    def _prepare_transform(self, tensor: Tensor, group: dict):
        original_shape = tensor.shape
        permuted_shape = None
        if group["merge_dims"]:
            if self._data_format == "channels_last" and tensor.dim() == 4:
                permuted_shape = tensor.permute(0, 3, 1, 2).shape
            tensor = self._merge_dims(tensor, group["max_precond_dim"])
        return tensor, original_shape, permuted_shape

    @staticmethod
    def _restore_transform(
        tensor: Tensor,
        original_shape: torch.Size,
        permuted_shape: Optional[torch.Size],
        merge_dims: bool,
    ) -> Tensor:
        if not merge_dims:
            return tensor
        if permuted_shape is not None:
            return tensor.reshape(permuted_shape).permute(0, 2, 3, 1)
        return tensor.reshape(original_shape)

    def _project_with_basis(self, tensor: Tensor, basis: list[Optional[Tensor]], group: dict) -> Tensor:
        """Apply Q.T along every active mode."""
        tensor, original_shape, permuted_shape = self._prepare_transform(tensor, group)
        if tensor.device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        compute_dtype = tensor.dtype

        if tensor.dim() == 2 and len(basis) == 2:
            left, right = basis
            if left is not None:
                tensor = left.to(compute_dtype).T @ tensor
            if right is not None:
                tensor = tensor @ right.to(compute_dtype)
        else:
            for matrix in basis:
                if matrix is None:
                    if tensor.dim() > 1:
                        tensor = tensor.permute(list(range(1, tensor.dim())) + [0])
                else:
                    tensor = torch.tensordot(
                        tensor, matrix.to(compute_dtype), dims=[[0], [0]]
                    )

        return self._restore_transform(
            tensor, original_shape, permuted_shape, group["merge_dims"]
        )

    def _project_back_with_basis(
        self,
        tensor: Tensor,
        basis: list[Optional[Tensor]],
        group: dict,
    ) -> Tensor:
        """Apply Q along every active mode."""
        tensor, original_shape, permuted_shape = self._prepare_transform(tensor, group)
        if tensor.device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        compute_dtype = tensor.dtype

        if tensor.dim() == 2 and len(basis) == 2:
            left, right = basis
            if left is not None:
                tensor = left.to(compute_dtype) @ tensor
            if right is not None:
                tensor = tensor @ right.to(compute_dtype).T
        else:
            for matrix in basis:
                if matrix is None:
                    if tensor.dim() > 1:
                        tensor = tensor.permute(list(range(1, tensor.dim())) + [0])
                else:
                    tensor = torch.tensordot(
                        tensor, matrix.to(compute_dtype), dims=[[0], [1]]
                    )

        return self._restore_transform(
            tensor, original_shape, permuted_shape, group["merge_dims"]
        )

    # ------------------------------------------------------------------
    # Covariance and basis lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _covariance_dtype(grad: Tensor, group: dict) -> torch.dtype:
        if group["covariance_compute_dtype"] == "bfloat16" and grad.is_cuda:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _parameter_has_preconditioner(parameter: Tensor, group: dict) -> bool:
        if parameter.ndim == 0:
            return False
        if parameter.ndim == 1:
            return (
                group["precondition_1d"]
                and int(parameter.shape[0]) <= group["max_precond_dim"]
            )
        dimensions = list(parameter.shape)
        eligible_axes = {
            axis
            for axis, size in enumerate(dimensions)
            if int(size) <= group["max_precond_dim"]
        }
        if not eligible_axes:
            return False
        mode = group.get("precondition_mode")
        if parameter.ndim == 2 and mode in ("smaller_side", "aspect_ratio"):
            smaller = min(dimensions)
            larger = max(dimensions)
            one_sided = mode == "smaller_side" or (
                larger / max(1, smaller)
                >= float(group["precondition_aspect_ratio"])
            )
            if one_sided:
                smaller_axis = min(
                    range(2), key=lambda axis: int(dimensions[axis])
                )
                return smaller_axis in eligible_axes
        return True

    def _init_state(
        self,
        grad: Tensor,
        state: dict,
        group: dict,
        parameter: Optional[Tensor] = None,
    ) -> None:
        state["step"] = 0
        state["basis_refreshes"] = 0
        state["basis_active_refreshes"] = 0
        state["basis_hard_reset_events"] = 0
        state["basis_hard_reset_factors"] = 0
        state["basis_last_hard_reset_step"] = 0
        state["basis_last_refresh_step"] = 0
        state["basis_last_residual_check_step"] = 0
        state["basis_residual_checks"] = 0
        state["basis_residual_skips"] = 0
        state["basis_residual_threshold_refreshes"] = 0
        state["basis_residual_max_age_refreshes"] = 0
        state["basis_residual_ratio_sum"] = 0.0
        state["basis_residual_ratio_max"] = 0.0
        state["basis_residual_ratio_last"] = 0.0
        state["basis_residual_ratio_samples"] = []
        state["basis_effective_lr_samples"] = 0
        state["basis_effective_lr_sum"] = 0.0
        state["basis_effective_lr_max"] = 0.0
        state["basis_reset_first_refresh"] = int(
            self._basis_reset_first_refresh.get(
                id(parameter), group["basis_reset_frequency"]
            )
        )
        state["basis_reset_first_step"] = (
            state["basis_reset_first_refresh"] * group["precondition_frequency"]
        )
        state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

        working_grad = grad
        if group["merge_dims"] and grad.dim() > 1:
            working_grad = self._merge_dims(grad, group["max_precond_dim"])
        dimensions = [working_grad.shape[0]] if working_grad.dim() == 1 else list(working_grad.shape)

        covariance: list[Optional[Tensor]] = []
        smaller_axis = None
        mode = group.get("precondition_mode")
        use_smaller_side = mode == "smaller_side"
        if working_grad.dim() == 2 and mode == "aspect_ratio":
            smaller = min(int(working_grad.shape[0]), int(working_grad.shape[1]))
            larger = max(int(working_grad.shape[0]), int(working_grad.shape[1]))
            use_smaller_side = (
                larger / max(1, smaller)
                >= float(group["precondition_aspect_ratio"])
            )
        if working_grad.dim() == 2 and use_smaller_side:
            smaller_axis = min(
                range(2), key=lambda axis: int(working_grad.shape[axis])
            )
        for axis, size in enumerate(dimensions):
            skip = (
                int(size) > group["max_precond_dim"]
                or (smaller_axis is not None and axis != smaller_axis)
                or (working_grad.dim() == 1 and not group["precondition_1d"])
            )
            covariance.append(
                None
                if skip
                else torch.zeros(
                    int(size), int(size), device=grad.device, dtype=torch.float32
                )
            )
        state["GG"] = covariance
        state["Q"] = [None for _ in covariance]

    def _accumulate_covariance(self, grad: Tensor, state: dict, group: dict) -> None:
        beta = group["shampoo_beta"]
        if beta < 0.0:
            beta = group["betas"][1]

        working_grad = grad
        if group["merge_dims"] and grad.dim() > 1:
            working_grad = self._merge_dims(grad, group["max_precond_dim"])
        g = working_grad.to(self._covariance_dtype(working_grad, group))

        if g.dim() == 1:
            covariance = state["GG"][0] if state["GG"] else None
            if covariance is not None:
                outer = g.unsqueeze(1) @ g.unsqueeze(0)
                covariance.lerp_(outer.float(), 1.0 - beta)
            return

        if g.dim() == 2:
            if state["GG"][0] is not None:
                state["GG"][0].lerp_((g @ g.T).float(), 1.0 - beta)
            if state["GG"][1] is not None:
                state["GG"][1].lerp_((g.T @ g).float(), 1.0 - beta)
            return

        for axis, covariance in enumerate(state["GG"]):
            if covariance is None:
                continue
            contract = list(chain(range(axis), range(axis + 1, g.dim())))
            outer = torch.tensordot(g, g, dims=[contract, contract])
            covariance.lerp_(outer.float(), 1.0 - beta)

    @staticmethod
    def _eigh_basis(covariance: Tensor) -> Tensor:
        c = _sym(covariance.float())
        try:
            _, q = torch.linalg.eigh(c)
        except RuntimeError:
            _, q = torch.linalg.eigh(c.double())
            q = q.float()
        return torch.flip(q, dims=[1]).contiguous()

    def _cold_start_basis(self, state: dict) -> None:
        state["Q"] = [
            None if covariance is None else self._eigh_basis(covariance)
            for covariance in state["GG"]
        ]

    @staticmethod
    def _align_reset_basis(q_old: Tensor, q_new: Tensor) -> Tensor:
        """Align a reset basis to old columns when the overlap gives a permutation."""
        overlap = q_old.float().T @ q_new.float()
        assignment = overlap.abs().argmax(dim=1)
        if int(torch.unique(assignment).numel()) == q_new.shape[1]:
            q_new = q_new[:, assignment]
            overlap = q_old.float().T @ q_new.float()
            signs = overlap.diagonal().sign()
        else:
            # Squared-overlap variance transport below remains correct even when
            # no clean permutation exists. Align only each new column's sign.
            old_index = overlap.abs().argmax(dim=0)
            new_index = torch.arange(q_new.shape[1], device=q_new.device)
            signs = overlap[old_index, new_index].sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return q_new * signs.unsqueeze(0)

    def _hard_reset_one(self, covariance: Tensor, q_old: Tensor, method: str) -> Tensor:
        if method == "eigh":
            q_new = self._eigh_basis(covariance)
        else:
            q_new, r = torch.linalg.qr(covariance.float() @ q_old.float(), mode="reduced")
            signs = r.diagonal().sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            q_new = q_new * signs.unsqueeze(0)
        return self._align_reset_basis(q_old, q_new).float()

    @staticmethod
    def _basis_covariance(covariance: Tensor, q: Tensor) -> Tensor:
        c = _normalized_covariance(covariance)
        return _sym(q.float().T @ c @ q.float())

    @staticmethod
    def _basis_residual_ratio(basis_covariance: Tensor) -> Tensor:
        b = basis_covariance.float()
        return _offdiag(b).norm() / b.norm().clamp_min(1e-12)

    @staticmethod
    def _gauge_step_one(
        covariance: Tensor,
        q_old: Tensor,
        group: dict,
        basis_covariance: Optional[Tensor] = None,
        effective_basis_lr: Optional[float] = None,
    ) -> Tensor:
        q = q_old.float()
        b = (
            SOAP._basis_covariance(covariance, q)
            if basis_covariance is None
            else _sym(basis_covariance.float())
        )
        omega = _damped_jacobi_generator(b, group["basis_jacobi_damping"])
        basis_lr = (
            group["basis_lr"]
            if effective_basis_lr is None
            else float(effective_basis_lr)
        )
        rotation = _clip_rotation(
            basis_lr * omega,
            group["basis_rotation_cap"],
            group.get("basis_rotation_cap_mode", "average"),
        )
        stall_threshold = group.get("basis_stall_pair_threshold", 0.0)
        if stall_threshold > 0.0:
            offdiag_norm = _offdiag(b).norm()
            if (
                float(offdiag_norm) > 0.0
                and float(rotation.norm()) <= float(stall_threshold * offdiag_norm)
            ):
                # Equal or poorly separated Rayleigh quotients can make the
                # double-bracket generator vanish away from a diagonal basis.
                return _largest_pair_jacobi_step(
                    q,
                    b,
                    max_angle=(
                        group["basis_rotation_cap"]
                        if group["basis_rotation_cap"] > 0.0
                        else math.pi / 4.0
                    ),
                )
        candidate = q + q @ rotation
        return _orthogonalize_ns(candidate, group["basis_ns_iterations"])

    def _transport_first_moment(
        self,
        state: dict,
        old_basis: list[Optional[Tensor]],
        new_basis: list[Optional[Tensor]],
        group: dict,
    ) -> None:
        world_momentum = self._project_back_with_basis(state["exp_avg"], old_basis, group)
        state["exp_avg"] = self._project_with_basis(world_momentum, new_basis, group).float()

    def _transport_second_moment_for_reset(
        self,
        state: dict,
        old_basis: list[Optional[Tensor]],
        new_basis: list[Optional[Tensor]],
        group: dict,
    ) -> None:
        transitions: list[Optional[Tensor]] = []
        for q_old, q_new in zip(old_basis, new_basis):
            if q_old is None or q_new is None:
                transitions.append(None)
            else:
                transitions.append((q_old.float().T @ q_new.float()).square())
        transported = self._project_with_basis(state["exp_avg_sq"], transitions, group)
        state["exp_avg_sq"] = transported.clamp_min_(0.0).float()

    def _record_metrics(
        self,
        covariance: Tensor,
        q_new: Tensor,
        q_old: Tensor,
        hard_reset: bool,
    ) -> None:
        metrics = _basis_metrics(covariance, q_new, q_old)
        metrics["hard_reset"] = float(hard_reset)
        if self._basis_stats_accum is None:
            self._basis_stats_accum = {"count": 0.0}
        self._basis_stats_accum["count"] += 1.0
        for key, value in metrics.items():
            self._basis_stats_accum[key] = self._basis_stats_accum.get(key, 0.0) + value

    @staticmethod
    def _should_hard_reset(state: dict, group: dict) -> bool:
        reset_frequency = int(group["basis_reset_frequency"])
        if reset_frequency <= 0:
            return False
        reset_max_age = int(group.get("basis_reset_max_age", 0))
        if reset_max_age > 0:
            if int(state.get("basis_hard_reset_events", 0)) == 0:
                return int(state["step"]) >= int(
                    state["basis_reset_first_step"]
                )
            return int(state["step"]) - int(
                state["basis_last_hard_reset_step"]
            ) >= reset_max_age
        first_refresh = int(
            state.get("basis_reset_first_refresh", reset_frequency)
        )
        refreshes = int(state["basis_refreshes"])
        return (
            refreshes >= first_refresh
            and (refreshes - first_refresh) % reset_frequency == 0
        )

    def _record_hard_reset(self, state: dict) -> None:
        active_factors = sum(factor is not None for factor in state["Q"])
        if active_factors == 0:
            return
        self.last_hard_reset_events += 1
        state["basis_hard_reset_events"] = int(
            state.get("basis_hard_reset_events", 0)
        ) + 1
        state["basis_hard_reset_factors"] = int(
            state.get("basis_hard_reset_factors", 0)
        ) + active_factors
        state["basis_last_hard_reset_step"] = int(state["step"])

    @staticmethod
    def _effective_basis_lr(state: dict, group: dict) -> float:
        basis_lr = float(group["basis_lr"])
        if not group.get("basis_lr_age_compensation", False):
            return basis_lr
        reference_age = int(group.get("basis_lr_reference_age", 0))
        if reference_age <= 0:
            reference_age = int(group["precondition_frequency"])
        age = max(
            0,
            int(state["step"]) - int(state.get("basis_last_refresh_step", 0)),
        )
        if basis_lr == 0.0 or age == 0:
            return 0.0
        # log1p/expm1 stays accurate for small tracker steps and fractional ages.
        return -math.expm1((age / reference_age) * math.log1p(-basis_lr))

    def _refresh_basis(
        self,
        state: dict,
        group: dict,
        basis_covariances: Optional[list[Optional[Tensor]]] = None,
    ) -> None:
        old_basis = state["Q"]
        effective_basis_lr = self._effective_basis_lr(state, group)
        state["basis_refreshes"] += 1
        state["basis_last_refresh_step"] = int(state["step"])
        state["basis_last_residual_check_step"] = int(state["step"])
        if any(factor is not None for factor in old_basis):
            state["basis_active_refreshes"] = int(
                state.get("basis_active_refreshes", 0)
            ) + 1
            self.last_basis_refreshes += 1
        hard_reset = self._should_hard_reset(state, group)
        if hard_reset:
            self._record_hard_reset(state)
        else:
            state["basis_effective_lr_samples"] = int(
                state.get("basis_effective_lr_samples", 0)
            ) + 1
            state["basis_effective_lr_sum"] = float(
                state.get("basis_effective_lr_sum", 0.0)
            ) + effective_basis_lr
            state["basis_effective_lr_max"] = max(
                float(state.get("basis_effective_lr_max", 0.0)),
                effective_basis_lr,
            )

        new_basis: list[Optional[Tensor]] = []
        if basis_covariances is None:
            basis_covariances = [None for _ in old_basis]
        for covariance, q_old, basis_covariance in zip(
            state["GG"], old_basis, basis_covariances
        ):
            if covariance is None or q_old is None:
                new_basis.append(None)
                continue
            if hard_reset:
                q_new = self._hard_reset_one(
                    covariance, q_old, group["basis_reset_method"]
                )
            else:
                q_new = self._gauge_step_one(
                    covariance,
                    q_old,
                    group,
                    basis_covariance=basis_covariance,
                    effective_basis_lr=effective_basis_lr,
                )
            new_basis.append(q_new.float())
            collect_stats = (
                group["basis_track_stats"]
                and state["basis_refreshes"] % group["basis_track_stats_frequency"] == 0
            )
            if collect_stats:
                self._record_metrics(covariance, q_new, q_old, hard_reset)

        self._transport_first_moment(state, old_basis, new_basis, group)
        if hard_reset or group["transport_second_moment_on_warm"]:
            self._transport_second_moment_for_reset(state, old_basis, new_basis, group)
        state["Q"] = new_basis

    def _residual_gated_refreshes(
        self,
        candidates: list[dict],
        group: dict,
    ) -> None:
        """Refresh candidates selected by a coarse covariance-residual check.

        Residuals remain device-side until one batched transfer per parameter
        group.  A successful check reuses its basis covariance in the gauge
        step, avoiding a second ``Q.T @ C @ Q`` calculation.
        """
        if not candidates:
            return

        threshold = float(group["basis_residual_threshold"])
        maximum_age = int(group["basis_residual_max_age"])
        checked: list[tuple[dict, list[Optional[Tensor]], Tensor]] = []
        for state in candidates:
            age = int(state["step"]) - int(state["basis_last_refresh_step"])
            if age >= maximum_age:
                state["basis_residual_max_age_refreshes"] += 1
                self._refresh_basis(state, group)
                continue

            state["basis_last_residual_check_step"] = int(state["step"])
            basis_covariances: list[Optional[Tensor]] = []
            factor_residuals: list[Tensor] = []
            for covariance, q in zip(state["GG"], state["Q"]):
                if covariance is None or q is None:
                    basis_covariances.append(None)
                    continue
                b = self._basis_covariance(covariance, q)
                basis_covariances.append(b)
                factor_residuals.append(self._basis_residual_ratio(b))
            if not factor_residuals:
                # Parameters without an active factor retain the fixed cadence.
                self._refresh_basis(state, group)
                continue
            checked.append(
                (state, basis_covariances, torch.stack(factor_residuals).amax())
            )

        if not checked:
            return
        residual_values = torch.stack([entry[2] for entry in checked]).tolist()
        for (state, basis_covariances, _), residual in zip(
            checked, residual_values
        ):
            residual = float(residual)
            state["basis_residual_checks"] += 1
            state["basis_residual_ratio_sum"] += residual
            state["basis_residual_ratio_max"] = max(
                float(state["basis_residual_ratio_max"]), residual
            )
            state["basis_residual_ratio_last"] = residual
            state["basis_residual_ratio_samples"].append(residual)
            if residual >= threshold:
                state["basis_residual_threshold_refreshes"] += 1
                self._refresh_basis(
                    state,
                    group,
                    basis_covariances=basis_covariances,
                )
            else:
                state["basis_residual_skips"] += 1

    @staticmethod
    def _fixed_refresh_frequency(state: dict, group: dict) -> int:
        later_frequency = int(group["precondition_frequency_after_warmup"])
        if (
            later_frequency > 0
            and int(state["step"])
            >= int(group["precondition_frequency_warmup_steps"])
        ):
            return later_frequency
        return int(group["precondition_frequency"])

    # ------------------------------------------------------------------
    # Optimizer update
    # ------------------------------------------------------------------

    def _finalize_stats(self) -> None:
        if self._basis_stats_accum is None:
            self.latest_basis_stats = {}
            self.just_gathered_basis_stats = False
            return
        count = self._basis_stats_accum.pop("count")
        self.latest_basis_stats = {
            key: value / count for key, value in self._basis_stats_accum.items()
        }
        self.just_gathered_basis_stats = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._basis_stats_accum = None
        self.latest_basis_stats = {}
        self.just_gathered_basis_stats = False
        self.last_basis_refreshes = 0
        self.last_hard_reset_events = 0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            residual_gating = group["basis_residual_threshold"] > 0.0
            residual_candidates: list[dict] = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                grad = parameter.grad.detach()
                state = self.state[parameter]

                if "step" not in state:
                    self._init_state(grad, state, group, parameter)
                    self._accumulate_covariance(grad, state, group)
                    self._cold_start_basis(state)
                    continue  # Initial covariance/basis observation.

                state["step"] += 1
                projected_grad = self._project_with_basis(grad, state["Q"], group).float()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(projected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    projected_grad, projected_grad, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                update = exp_avg / bias_correction1
                update = update / (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                update = self._project_back_with_basis(update, state["Q"], group)

                if group["normalize_grads"]:
                    update = update / update.square().mean().sqrt().clamp_min(1e-30)

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

                self._accumulate_covariance(grad, state, group)
                residual_gate_active = residual_gating and int(
                    state["step"]
                ) >= group["basis_residual_warmup_steps"]
                fixed_frequency = self._fixed_refresh_frequency(state, group)
                if not residual_gate_active and state["step"] % fixed_frequency == 0:
                    self._refresh_basis(state, group)
                elif residual_gate_active:
                    check_age = int(state["step"]) - int(
                        state["basis_last_residual_check_step"]
                    )
                    refresh_age = int(state["step"]) - int(
                        state["basis_last_refresh_step"]
                    )
                    if (
                        check_age >= group["precondition_frequency"]
                        or refresh_age >= group["basis_residual_max_age"]
                    ):
                        residual_candidates.append(state)

            if residual_gating:
                self._residual_gated_refreshes(residual_candidates, group)

        self._finalize_stats()
        return loss


TurboSlimSOAP = SOAP


__all__ = ["SOAP", "TurboSlimSOAP"]
