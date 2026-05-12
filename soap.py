"""
GradStepSOAP: SOAP with configurable eigenbasis refresh

Standard SOAP computes eigenbases Q_k of the Kronecker covariance factors
every K steps via eigendecomposition, an O(n^3) LAPACK call. This variant
also supports a warm-started Newton-Schulz basis refresh path built from
pure matmuls, so the refresh rule itself can be benchmarked cleanly.

The core SOAP mechanism is preserved exactly:
  Q^T @ grad @ Q          →  project to eigenbasis
  Adam(exp_avg, exp_avg_sq) →  per-element adaptation in decorrelated space
  Q @ update @ Q^T         →  project back

References:
  SOAP  — https://arxiv.org/abs/2409.11321
  Shampoo — https://arxiv.org/abs/1802.09568
"""

import torch
import torch.optim as optim
from itertools import chain


_SOAP_NS_MAX_SINGULAR = 1.25


def _soap_sym(A):
    return 0.5 * (A + A.T)


def _soap_ns_higham_cubic(X):
    Z = X
    gram = Z.T @ Z

    # Normalization based on top eigenvalue/singular value to prevent explosions
    upper_sq = gram.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(0)
    scale = (_SOAP_NS_MAX_SINGULAR / (upper_sq.sqrt() + 1e-7)).clamp(max=1.0)
    Z = Z * scale
    gram = gram * scale.square()

    gram_sq = gram @ gram
    correction = -10.0 * gram + 3.0 * gram_sq
    Z = 0.125 * (15.0 * Z + Z @ correction)
    return Z


def _soap_ns_retract(X):
    return _soap_ns_higham_cubic(X)


def _soap_normalize_direction(direction):
    rms = direction.norm() / max(1.0, float(direction.numel()) ** 0.5)
    return direction / rms.clamp(min=1e-12)


def _soap_update_direction_moment(moment, key: str, value, beta: float):
    current = moment.get(key)
    if current is None or current.shape != value.shape or current.device != value.device:
        current = torch.zeros_like(value)
        moment[key] = current
        moment[f"{key}_step"] = 0
    step_key = f"{key}_step"
    moment[step_key] = int(moment.get(step_key, 0)) + 1
    current.mul_(beta).add_(value, alpha=1.0 - beta)
    bias_correction = 1.0 - beta ** moment[step_key]
    return current / max(bias_correction, 1e-12)


def _soap_normalize_direction_by_mode(
    direction,
    normalizer: str,
    moment=None,
    beta: float = 0.0,
):
    if normalizer == "none":
        return direction
    if normalizer == "rms":
        if moment is not None and beta > 0.0:
            var = _soap_update_direction_moment(
                moment, "rms", direction.square().mean(), beta
            )
            return direction / var.sqrt().clamp(min=1e-12)
        return _soap_normalize_direction(direction)
    if normalizer == "row":
        row_var = direction.square().mean(dim=1, keepdim=True)
        if moment is not None and beta > 0.0:
            row_var = _soap_update_direction_moment(moment, "row", row_var, beta)
        denom = row_var.sqrt()
        return direction / denom.clamp(min=1e-12)
    if normalizer == "col":
        col_var = direction.square().mean(dim=0, keepdim=True)
        if moment is not None and beta > 0.0:
            col_var = _soap_update_direction_moment(moment, "col", col_var, beta)
        denom = col_var.sqrt()
        return direction / denom.clamp(min=1e-12)
    if normalizer == "row_col":
        row_var = direction.square().mean(dim=1, keepdim=True)
        col_var = direction.square().mean(dim=0, keepdim=True)
        if moment is not None and beta > 0.0:
            row_var = _soap_update_direction_moment(moment, "row", row_var, beta)
            col_var = _soap_update_direction_moment(moment, "col", col_var, beta)
        denom = (0.5 * (row_var + col_var)).sqrt()
        return direction / denom.clamp(min=1e-12)
    raise ValueError(f"Unknown basis_direction_normalizer: {normalizer!r}")


def _soap_brockett_projected_direction(Cq, q, weights):
    grad = Cq * weights.unsqueeze(0)
    return grad - q @ _soap_sym(q.T @ grad)


def _soap_tangent_direction(Cq, q):
    diag = (q * Cq).sum(dim=0)
    return Cq - q * diag.unsqueeze(0)


def _soap_eigengap_precondition_direction(Cq, q, direction, gap_floor: float):
    B = q.T @ Cq
    diag = torch.diagonal(B)
    gaps = (diag.unsqueeze(1) - diag.unsqueeze(0)).abs()
    floor = diag.abs().max().clamp(min=1e-12) * max(float(gap_floor), 1e-12)
    inv_gaps = gaps.clamp(min=floor).reciprocal()
    generator = q.T @ direction
    generator = 0.5 * (generator - generator.T)
    return q @ (generator * inv_gaps)


def _soap_maybe_precondition_direction(
    Cq, q, direction, preconditioner: str, gap_floor: float
):
    if preconditioner == "none":
        return direction
    if preconditioner == "eigengap":
        return _soap_eigengap_precondition_direction(Cq, q, direction, gap_floor)
    raise ValueError(f"Unknown basis_direction_preconditioner: {preconditioner!r}")


def _soap_strict_brockett_weights(q, weight_power: float):
    n = q.shape[1]
    base = torch.linspace(float(n), 1.0, n, device=q.device, dtype=q.dtype)
    weights = base.pow(weight_power)
    return weights / weights.mean().clamp(min=1e-8)


def _soap_basis_step_brockett_weighted_full(
    C,
    q,
    weights,
    direction_preconditioner: str,
    eigengap_floor: float,
    eta: float,
    direction_normalizer: str,
):
    Cq = C @ q
    direction = _soap_brockett_projected_direction(Cq, q, weights)
    direction = _soap_maybe_precondition_direction(
        Cq, q, direction, direction_preconditioner, eigengap_floor
    )
    direction = _soap_normalize_direction_by_mode(direction, direction_normalizer)
    return _soap_ns_retract(q + eta * direction)


def _soap_basis_step_tangent_full(
    C,
    q,
    direction_preconditioner: str,
    eigengap_floor: float,
    eta: float,
    direction_normalizer: str,
):
    Cq = C @ q
    direction = _soap_tangent_direction(Cq, q)
    direction = _soap_maybe_precondition_direction(
        Cq, q, direction, direction_preconditioner, eigengap_floor
    )
    direction = _soap_normalize_direction_by_mode(direction, direction_normalizer)
    return _soap_ns_retract(q + eta * direction)


def _make_soap_brockett_fixed_kernel(
    direction_method: str,
    direction_preconditioner: str,
    eigengap_floor: float,
    direction_normalizer: str,
    substeps: int,
):
    use_eigengap = direction_preconditioner == "eigengap"
    use_tangent = direction_method == "tangent"

    def kernel(C, q, weights, eta: float):
        current = q
        for _ in range(substeps):
            Cq = C @ current
            if use_tangent:
                direction = _soap_tangent_direction(Cq, current)
            else:
                direction = _soap_brockett_projected_direction(Cq, current, weights)

            if use_eigengap:
                direction = _soap_eigengap_precondition_direction(
                    Cq,
                    current,
                    direction,
                    eigengap_floor,
                )

            direction = _soap_normalize_direction_by_mode(
                direction,
                direction_normalizer,
            )
            current = _soap_ns_retract(current + eta * direction)
        return current

    return kernel


class SOAP(optim.Optimizer):
    r"""
    SOAP with configurable warm eigenbasis refresh.

    Maintains GG (covariance EMA) and Q (orthogonal eigenbasis) per
    Kronecker factor mode. Q is updated every ``precondition_frequency``
    steps (default 1) using a configurable warm refresh method from
    ``{"none", "tangent", "brockett", "eigh", "qr"}``. Cold start always uses eigh.

    exp_avg and exp_avg_sq live in the eigenbasis. When Q changes,
    exp_avg is un-projected from old Q and re-projected into new Q, while
    exp_avg_sq is re-sorted by estimated eigenvalue to match the new basis.

    Args:
        params: Parameters to optimise.
        lr: Learning rate (default: 3e-3).
        betas: Adam (β₁, β₂) (default: (0.95, 0.999)).
        shampoo_beta: EMA beta for Kronecker factor accumulation.
            -1 uses betas[1] (default: 0.999).
        eps: Adam epsilon (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 0.01).
        precondition_frequency: How often to update Q. Standard SOAP
            uses 10+ (default: 10).
        max_precond_dim: Skip preconditioning for dims above this
            (default: 10000).
        merge_dims: Merge small dims for >2D tensors (default: False).
        precondition_1d: Precondition 1D params (default: False).
        normalize_grads: RMS-normalise the final update (default: False).
        data_format: "channels_first" or "channels_last"
            (default: "channels_first").
        basis_stage1_method: "none", "tangent", "brockett", "eigh", or "qr"
            for the warm-refresh method. "tangent" uses the cheaper projected
            covariance direction; "brockett" adds ordered diagonalization
            pressure. "qr" matches the reference SOAP QR refresh: one power
            iteration from the current basis followed by QR (default: "tangent").
        Warm tracker retractions use one Higham cubic Newton-Schulz step.
        adam_second_moment: Shape of Adam's second-moment denominator in the
            SOAP basis. "elementwise" is standard SOAP; "axis_product" is an
            Adafactor-style product of one variance vector per tensor mode;
            "axis_geomean" uses a one-third-power product of axis variances;
            "axis_logblend" interpolates between axis_sum and axis_product
            denominators in log space; "axis_sum" and "axis_max" are softer
            factored variants; "scalar" uses one RMS variance per tensor
            (default: "elementwise").
        basis_step_size: Initial step size for the Stiefel-manifold trackers
            (``tangent``/``brockett``). Ignored by ``eigh``
            (default: 0.01).
        basis_substeps: Maximum tracker/retraction step budget per refresh for
            ``tangent``/``brockett`` (default: 1).
        basis_normalize_covariance: Normalize covariance by mean diagonal
            before tracker updates. This preserves eigenvectors while making
            step sizes more comparable across layers (default: True).
        basis_direction_normalizer: Normalize tracker directions before
            applying ``basis_step_size``. "rms" matches the previous scalar RMS
            behavior; "row", "col", and "row_col" apply factored matrix
            normalization to the tracker direction; "none" leaves direction
            magnitudes intact (default: "col").
        basis_direction_normalizer_beta: EMA beta for stateful direction
            normalizers. -1 uses Adam beta2. 0 preserves instantaneous
            normalization from prior versions (default: 0.99).
        basis_step_controller: Step-size controller for warm basis refreshes.
            ``"correction_ratio"`` adapts per-factor step scales to maximize
            diagonalization-error reduction within the substep budget.
            ``"fixed"`` uses ``basis_step_size`` directly (default:
            "correction_ratio").
        basis_target_correction_ratio: Stop a refresh once this fraction of the
            starting diagonalization error has been removed (default: 0.99).
        basis_min_marginal_correction_ratio: Stop after an accepted substep when
            its additional error reduction, normalized by starting error, falls
            below this threshold (default: 0.03).
        basis_backtrack_factor: Multiplicative shrink when a trial step fails
            to improve diagonalization error (default: 0.7).
        basis_growth_factor: Multiplicative growth for the next refresh when
            the current refresh appears step-size limited (default: 1.5).
        basis_step_size_min: Lower clamp for controller-adjusted step sizes
            (default: 0.00001).
        basis_step_size_max: Upper clamp for controller-adjusted step sizes
            (default: 1000.0).
        basis_compute_dtype: Internal dtype for warm basis-refresh matmuls.
            ``"float32"`` keeps the previous behavior; ``"bfloat16"`` runs
            Brockett directions and Newton-Schulz retractions in bf16
            (default: "bfloat16").
        basis_storage_dtype: Dtype used to store and apply Q between refreshes.
            ``"float32"`` is most conservative; ``"bfloat16"`` reduces Q memory
            traffic and projection cast overhead (default: "bfloat16").
        basis_compile: Compile selected warm-refresh kernels with
            ``torch.compile`` when available (default: False).
        basis_track_stats: Record aggregate basis diagnostics in
            ``latest_basis_stats``. Disabling this skips diagnostics that are
            not needed for controller/fallback decisions (default: True).
        basis_metric_frequency: Run tracker metric checks once every N tracker
            refreshes. Metric refreshes handle accept/reject, fallback, stats,
            and controller scale updates; intervening refreshes use the current
            fixed step without metrics (default: 5).
        covariance_compute_dtype: Dtype used for gradient outer products before
            fp32 covariance EMA accumulation. ``"bfloat16"`` can reduce matmul
            cost for approximate whitening statistics (default: "bfloat16").
        basis_direction_preconditioner: Optional preconditioner applied to the
            Brockett tracker direction before normalization. "eigengap" scales
            pairwise rotations by inverse estimated eigengap (default: "eigengap").
        basis_eigengap_floor: Relative floor for eigengap preconditioning
            (default: 1e-3).
        brockett_weight_power: Power applied to the Brockett ordering weights.
            0 removes ordering pressure; values > 1 bias the update more
            strongly toward the top spectrum (default: 1.0).
        basis_fallback_diag_err_threshold: After a warm refresh, fall back to
            ``basis_fallback_method`` if normalized diagonalization error
            remains above this threshold. Values <= 0 disable the fallback
            (default: 0.3).
        basis_fallback_method: "eigh" or "qr" fallback refresh used by the
            threshold fallback and optional warmup phase (default: "qr").
        basis_warmup_steps: Number of optimizer steps to refresh directly with
            ``basis_fallback_method`` before switching to ``basis_stage1_method``
            (default: 500).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.999),
        shampoo_beta: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        basis_stage1_method: str = "tangent",
        adam_second_moment: str = "elementwise",
        adam_second_moment_blend_alpha: float = 0.5,
        basis_step_size: float = 0.01,
        basis_substeps: int = 1,
        basis_normalize_covariance: bool = True,
        basis_direction_normalizer: str = "col",
        basis_direction_normalizer_beta: float = 0.99,
        basis_step_controller: str = "correction_ratio",
        basis_target_correction_ratio: float = 0.99,
        basis_min_marginal_correction_ratio: float = 0.03,
        basis_backtrack_factor: float = 0.7,
        basis_growth_factor: float = 1.5,
        basis_step_size_min: float = 0.00001,
        basis_step_size_max: float = 1000.0,
        basis_compute_dtype: str = "bfloat16",
        basis_storage_dtype: str = "bfloat16",
        basis_compile: bool = False,
        basis_track_stats: bool = True,
        basis_metric_frequency: int = 5,
        covariance_compute_dtype: str = "bfloat16",
        basis_direction_preconditioner: str = "eigengap",
        basis_eigengap_floor: float = 1e-3,
        brockett_weight_power: float = 1.0,
        basis_fallback_diag_err_threshold: float = 0.3,
        basis_fallback_method: str = "qr",
        basis_warmup_steps: int = 250,
    ):
        if basis_stage1_method not in ("none", "tangent", "brockett", "eigh", "qr"):
            raise ValueError(
                "basis_stage1_method must be 'none', 'tangent', 'brockett', "
                f"'eigh', or 'qr', got {basis_stage1_method!r}"
            )
        if adam_second_moment not in (
            "elementwise",
            "axis_product",
            "axis_geomean",
            "axis_logblend",
            "axis_sum",
            "axis_max",
            "scalar",
        ):
            raise ValueError(
                "adam_second_moment must be 'elementwise', 'axis_product', "
                "'axis_geomean', 'axis_logblend', 'axis_sum', 'axis_max', "
                f"or 'scalar', got {adam_second_moment!r}"
            )
        if not 0.0 <= float(adam_second_moment_blend_alpha) <= 1.0:
            raise ValueError(
                "adam_second_moment_blend_alpha must be in [0, 1], got "
                f"{adam_second_moment_blend_alpha!r}"
            )
        if adam_second_moment != "elementwise" and merge_dims:
            raise ValueError(
                "adam_second_moment modes other than 'elementwise' are not "
                "currently supported with merge_dims=True"
            )
        if basis_direction_normalizer not in ("none", "rms", "row", "col", "row_col"):
            raise ValueError(
                "basis_direction_normalizer must be 'none', 'rms', 'row', "
                f"'col', or 'row_col', got {basis_direction_normalizer!r}"
            )
        if float(basis_direction_normalizer_beta) != -1.0 and not (
            0.0 <= float(basis_direction_normalizer_beta) < 1.0
        ):
            raise ValueError(
                "basis_direction_normalizer_beta must be -1 or in [0, 1), got "
                f"{basis_direction_normalizer_beta!r}"
            )
        if int(basis_substeps) < 1:
            raise ValueError(f"basis_substeps must be >= 1, got {basis_substeps!r}")
        if basis_step_controller not in ("fixed", "correction_ratio"):
            raise ValueError(
                "basis_step_controller must be 'fixed' or 'correction_ratio', got "
                f"{basis_step_controller!r}"
            )
        if not 0.0 <= float(basis_target_correction_ratio) <= 1.0:
            raise ValueError(
                "basis_target_correction_ratio must be in [0, 1], got "
                f"{basis_target_correction_ratio!r}"
            )
        if float(basis_min_marginal_correction_ratio) < 0:
            raise ValueError(
                "basis_min_marginal_correction_ratio must be >= 0, got "
                f"{basis_min_marginal_correction_ratio!r}"
            )
        if not 0.0 < float(basis_backtrack_factor) < 1.0:
            raise ValueError(
                "basis_backtrack_factor must be in (0, 1), got "
                f"{basis_backtrack_factor!r}"
            )
        if float(basis_growth_factor) <= 1.0:
            raise ValueError(
                f"basis_growth_factor must be > 1, got {basis_growth_factor!r}"
            )
        if float(basis_step_size_min) <= 0:
            raise ValueError(
                f"basis_step_size_min must be > 0, got {basis_step_size_min!r}"
            )
        if float(basis_step_size_max) < float(basis_step_size_min):
            raise ValueError(
                "basis_step_size_max must be >= basis_step_size_min, got "
                f"{basis_step_size_max!r} < {basis_step_size_min!r}"
            )
        if basis_direction_preconditioner not in ("none", "eigengap"):
            raise ValueError(
                "basis_direction_preconditioner must be 'none' or 'eigengap', got "
                f"{basis_direction_preconditioner!r}"
            )
        if float(basis_eigengap_floor) < 0:
            raise ValueError(
                f"basis_eigengap_floor must be >= 0, got {basis_eigengap_floor!r}"
            )
        if float(brockett_weight_power) < 0:
            raise ValueError(
                f"brockett_weight_power must be >= 0, got {brockett_weight_power!r}"
            )
        if basis_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                "basis_compute_dtype must be 'float32' or 'bfloat16', got "
                f"{basis_compute_dtype!r}"
            )
        if basis_storage_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                "basis_storage_dtype must be 'float32' or 'bfloat16', got "
                f"{basis_storage_dtype!r}"
            )
        if covariance_compute_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                "covariance_compute_dtype must be 'float32' or 'bfloat16', got "
                f"{covariance_compute_dtype!r}"
            )
        if int(basis_metric_frequency) < 1:
            raise ValueError(
                f"basis_metric_frequency must be >= 1, got {basis_metric_frequency!r}"
            )
        if float(basis_fallback_diag_err_threshold) < 0:
            raise ValueError(
                "basis_fallback_diag_err_threshold must be >= 0, got "
                f"{basis_fallback_diag_err_threshold!r}"
            )
        if basis_fallback_method not in ("eigh", "qr"):
            raise ValueError(
                "basis_fallback_method must be 'eigh' or 'qr', got "
                f"{basis_fallback_method!r}"
            )
        if int(basis_warmup_steps) < 0:
            raise ValueError(
                f"basis_warmup_steps must be >= 0, got {basis_warmup_steps!r}"
            )
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, merge_dims=merge_dims,
            precondition_1d=precondition_1d, normalize_grads=normalize_grads,
            basis_stage1_method=basis_stage1_method,
            adam_second_moment=adam_second_moment,
            adam_second_moment_blend_alpha=adam_second_moment_blend_alpha,
            basis_step_size=basis_step_size,
            basis_substeps=basis_substeps,
            basis_normalize_covariance=basis_normalize_covariance,
            basis_direction_normalizer=basis_direction_normalizer,
            basis_direction_normalizer_beta=float(basis_direction_normalizer_beta),
            basis_step_controller=basis_step_controller,
            basis_target_correction_ratio=basis_target_correction_ratio,
            basis_min_marginal_correction_ratio=basis_min_marginal_correction_ratio,
            basis_backtrack_factor=basis_backtrack_factor,
            basis_growth_factor=basis_growth_factor,
            basis_step_size_min=basis_step_size_min,
            basis_step_size_max=basis_step_size_max,
            basis_compute_dtype=basis_compute_dtype,
            basis_storage_dtype=basis_storage_dtype,
            basis_compile=basis_compile,
            basis_track_stats=basis_track_stats,
            basis_metric_frequency=int(basis_metric_frequency),
            covariance_compute_dtype=covariance_compute_dtype,
            basis_direction_preconditioner=basis_direction_preconditioner,
            basis_eigengap_floor=basis_eigengap_floor,
            brockett_weight_power=brockett_weight_power,
            basis_fallback_diag_err_threshold=basis_fallback_diag_err_threshold,
            basis_fallback_method=basis_fallback_method,
            basis_warmup_steps=int(basis_warmup_steps),
        )
        super().__init__(params, defaults)
        self._data_format = data_format
        self.latest_basis_stats = {}
        self.just_gathered_basis_stats = False
        self._basis_compiled_fns = {}
        self._basis_weight_cache = {}

    # ==================================================================
    # Dimension merging (from reference SOAP)
    # ==================================================================

    def _merge_dims(self, grad, max_precond_dim):
        assert self._data_format in ("channels_first", "channels_last")
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        new_shape, curr = [], 1
        for sh in grad.shape:
            tmp = curr * sh
            if tmp > max_precond_dim:
                if curr > 1:
                    new_shape.append(curr)
                    curr = sh
                else:
                    new_shape.append(sh)
                    curr = 1
            else:
                curr = tmp
        if curr > 1 or len(new_shape) == 0:
            new_shape.append(curr)
        return grad.reshape(new_shape)

    # ==================================================================
    # Matrix primitives
    # ==================================================================

    @staticmethod
    def _dtype_from_name(dtype_name):
        if dtype_name == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @classmethod
    def _basis_dtype(cls, group):
        return cls._dtype_from_name(group["basis_compute_dtype"])

    @classmethod
    def _basis_storage_dtype(cls, group):
        return cls._dtype_from_name(group["basis_storage_dtype"])

    @classmethod
    def _covariance_compute_dtype(cls, group):
        return cls._dtype_from_name(group["covariance_compute_dtype"])

    @staticmethod
    def _matmul_dtype(lhs, rhs):
        if lhs.dtype == rhs.dtype:
            return lhs.dtype
        priority = {
            torch.float16: 0,
            torch.bfloat16: 1,
            torch.float32: 2,
            torch.float64: 3,
        }
        lhs_priority = priority.get(lhs.dtype)
        rhs_priority = priority.get(rhs.dtype)
        if lhs_priority is None or rhs_priority is None:
            return lhs.dtype
        return lhs.dtype if lhs_priority <= rhs_priority else rhs.dtype

    def _maybe_compile_basis_fn(self, key, fn, group):
        if not bool(group["basis_compile"]) or not hasattr(torch, "compile"):
            return fn
        if key not in self._basis_compiled_fns:
            # Compile one fixed basis-step kernel at a time. Keep metric checks,
            # early stopping, fallback, and logging in Python outside this boundary.
            self._basis_compiled_fns[key] = torch.compile(fn, fullgraph=True)
        return self._basis_compiled_fns[key]

    def _fixed_brockett_kernel(self, group, max_substeps: int):
        key = (
            "fixed_brockett",
            group["basis_stage1_method"],
            group["basis_direction_preconditioner"],
            float(group["basis_eigengap_floor"]),
            group["basis_direction_normalizer"],
            int(max_substeps),
            bool(group["basis_compile"]),
        )
        if key not in self._basis_compiled_fns:
            fn = _make_soap_brockett_fixed_kernel(
                group["basis_stage1_method"],
                group["basis_direction_preconditioner"],
                float(group["basis_eigengap_floor"]),
                group["basis_direction_normalizer"],
                int(max_substeps),
            )
            if bool(group["basis_compile"]) and hasattr(torch, "compile"):
                fn = torch.compile(fn, fullgraph=True)
            self._basis_compiled_fns[key] = fn
        return self._basis_compiled_fns[key]

    def _brockett_weights(self, q, group):
        weight_power = float(group["brockett_weight_power"])
        key = (
            q.shape[1],
            q.device,
            q.dtype,
            weight_power,
        )
        weights = self._basis_weight_cache.get(key)
        if weights is None:
            weights = _soap_strict_brockett_weights(q, weight_power)
            self._basis_weight_cache[key] = weights
        return weights

    def _ns_orthogonalize(self, X, group):
        """Newton-Schulz polar retraction."""
        orig_dtype = X.dtype
        work = X.to(self._basis_dtype(group))
        return _soap_ns_retract(work).to(dtype=orig_dtype)

    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.T)

    @staticmethod
    def _normalize_covariance(C):
        scale = torch.trace(C).abs() / max(1, C.shape[0])
        scale = scale.clamp(min=1e-12)
        return C / scale

    @classmethod
    def _cast_basis_list(cls, orth_list, group):
        storage_dtype = cls._basis_storage_dtype(group)
        return [
            q.to(storage_dtype) if not (isinstance(q, list) or len(q) == 0) else []
            for q in orth_list
        ]

    @staticmethod
    def _basis_metric_tensors_impl(C, Q_new, Q_old=None):
        B = Q_new.T @ C @ Q_new
        B_sq = B.square()
        B_norm_sq = B_sq.sum().clamp(min=1e-24)
        diag_sq = torch.diagonal(B).square().sum()
        diag_error = ((B_norm_sq - diag_sq).clamp(min=0.0) / B_norm_sq).sqrt()

        metrics = {
            "diag_err": diag_error,
        }
        if Q_old is not None:
            drift = (Q_new - Q_old).norm() / max(1.0, float(Q_old.numel()) ** 0.5)
            metrics["drift"] = drift
        return metrics

    def _basis_metric_tensors(self, C, Q_new, Q_old=None, group=None):
        if group is None or not bool(group["basis_compile"]) or not hasattr(torch, "compile"):
            return self._basis_metric_tensors_impl(C, Q_new, Q_old)
        key = "basis_metric_tensors"
        if key not in self._basis_compiled_fns:
            self._basis_compiled_fns[key] = torch.compile(self._basis_metric_tensors_impl)
        return self._basis_compiled_fns[key](C, Q_new, Q_old)

    def _basis_metrics(self, C, Q_new, Q_old=None, group=None):
        return {
            key: float(value.item())
            for key, value in self._basis_metric_tensors(C, Q_new, Q_old, group).items()
        }

    @staticmethod
    def _float_metrics(metrics):
        return {key: float(value.item()) for key, value in metrics.items()}

    @staticmethod
    def _metric_float(value):
        return float(value.item()) if torch.is_tensor(value) else float(value)

    @staticmethod
    def _metric_dict_value(metrics, key):
        return SOAP._metric_float(metrics[key])

    @staticmethod
    def _metric_improvement(before, after):
        return SOAP._metric_float(before) - SOAP._metric_float(after)

    @staticmethod
    def _metric_leq(lhs, rhs):
        lhs_value = lhs.item() if torch.is_tensor(lhs) else lhs
        rhs_value = rhs.item() if torch.is_tensor(rhs) else rhs
        return lhs_value <= rhs_value

    @staticmethod
    def _metrics_with_improvement(metrics, before):
        out = SOAP._float_metrics(metrics)
        out["diag_err_before"] = SOAP._metric_float(before)
        out["diag_err_improvement"] = SOAP._metric_improvement(before, metrics["diag_err"])
        return out

    def _record_basis_stats(self, stats):
        if not stats:
            return
        if not bool(stats.pop("_track_stats", True)):
            return
        if not hasattr(self, "_basis_stats_accum") or self._basis_stats_accum is None:
            self._basis_stats_accum = {"count": 0.0}
        self._basis_stats_accum["count"] += 1.0
        for key, value in stats.items():
            self._basis_stats_accum[key] = self._basis_stats_accum.get(key, 0.0) + float(value)

    def _finalize_basis_stats(self):
        accum = getattr(self, "_basis_stats_accum", None)
        if accum is None or accum.get("count", 0.0) == 0.0:
            self.latest_basis_stats = {}
            self.just_gathered_basis_stats = False
            return
        count = accum["count"]
        self.latest_basis_stats = {
            key: value / count for key, value in accum.items() if key != "count"
        }
        self.just_gathered_basis_stats = True

    @staticmethod
    def _basis_scale_list(precond_list):
        return [
            1.0 if not (isinstance(m, list) or len(m) == 0) else None
            for m in precond_list
        ]

    @staticmethod
    def _clamp_basis_eta(eta, group):
        return min(
            float(group["basis_step_size_max"]),
            max(float(group["basis_step_size_min"]), float(eta)),
        )

    @classmethod
    def _eta_to_scale(cls, eta, group):
        base_eta = max(float(group["basis_step_size"]), 1e-12)
        return cls._clamp_basis_eta(eta, group) / base_eta

    @staticmethod
    def _direction_normalizer_beta(group):
        beta = float(group["basis_direction_normalizer_beta"])
        if beta < 0.0:
            beta = float(group["betas"][1])
        if group["basis_direction_normalizer"] == "none":
            return 0.0
        return beta

    @staticmethod
    def _direction_moment_list(state, precond_list):
        moments = state.get("basis_direction_moments")
        if moments is None or len(moments) != len(precond_list):
            moments = [None if isinstance(m, list) or len(m) == 0 else {} for m in precond_list]
            state["basis_direction_moments"] = moments
        return moments

    def _fallback_basis_one(self, m, q0, group):
        method = group["basis_fallback_method"]
        if method == "eigh":
            return self._get_orthogonal_matrix_eigh([m], group)[0]
        if method == "qr":
            return torch.linalg.qr(m.data.float() @ q0.data.float(), mode="reduced")[0]
        raise ValueError(f"Unknown basis_fallback_method: {method!r}")

    def _fallback_basis_list(self, precond_list, orth_list, group):
        final_Q = []
        for m, q_old in zip(precond_list, orth_list):
            if isinstance(m, list) or len(m) == 0:
                final_Q.append([])
                continue
            final_Q.append(self._fallback_basis_one(m, q_old, group))
        return final_Q

    def _maybe_fallback_to_basis(self, m, C_metric, q_candidate, q0, metrics, group):
        threshold = float(group["basis_fallback_diag_err_threshold"])
        if threshold <= 0 or metrics["diag_err"] <= threshold:
            metrics["eigh_fallback"] = 0.0
            metrics["qr_fallback"] = 0.0
            metrics["diag_err_after_fallback"] = metrics["diag_err"]
            return q_candidate, metrics

        fallback_method = group["basis_fallback_method"]
        q_fallback = self._fallback_basis_one(m, q0, group)
        fallback_metrics = self._basis_metrics(C_metric, q_fallback, q0, group)
        for key, value in metrics.items():
            if key not in fallback_metrics:
                fallback_metrics[key] = value
        fallback_metrics["eigh_fallback"] = 1.0 if fallback_method == "eigh" else 0.0
        fallback_metrics["qr_fallback"] = 1.0 if fallback_method == "qr" else 0.0
        fallback_metrics["fallback_from_diag_err"] = metrics["diag_err"]
        fallback_metrics["diag_err_after_fallback"] = fallback_metrics["diag_err"]
        return q_fallback, fallback_metrics

    def _transport_axis_second_moments(self, state, old_orth_list, new_orth_list, group):
        if group["adam_second_moment"] not in (
            "axis_product",
            "axis_geomean",
            "axis_logblend",
            "axis_sum",
            "axis_max",
        ):
            return
        axis_second_moment = state.get("exp_avg_sq_axis")
        if axis_second_moment is None:
            return

        for ind, (q_old, q_new) in enumerate(zip(old_orth_list, new_orth_list)):
            if isinstance(q_old, list) or isinstance(q_new, list):
                continue
            if len(q_old) == 0 or len(q_new) == 0 or q_old.shape != q_new.shape:
                continue
            values = axis_second_moment[ind].reshape(-1)
            transition = (q_old.data.float().T @ q_new.data.float()).square()
            transported = values.float() @ transition.to(values.device)
            axis_second_moment[ind] = transported.to(axis_second_moment[ind].dtype).reshape_as(
                axis_second_moment[ind]
            )
        state["exp_avg_sq_axis"] = axis_second_moment

    def _finalize_basis_change(self, state, old_orth_list, new_orth_list, group):
        new_orth_list = self._cast_basis_list(new_orth_list, group)
        self._transport_axis_second_moments(state, old_orth_list, new_orth_list, group)
        state["Q"] = new_orth_list

    def _basis_step_brockett(
        self,
        C,
        q,
        eta,
        direction_normalizer,
        group,
        direction_moment=None,
    ):
        direction_preconditioner = group["basis_direction_preconditioner"]
        eigengap_floor = float(group["basis_eigengap_floor"])
        beta = self._direction_normalizer_beta(group)
        if direction_moment is not None and beta > 0.0:
            Cq = C @ q
            weights = self._brockett_weights(q, group)
            direction = _soap_brockett_projected_direction(Cq, q, weights)
            direction = _soap_maybe_precondition_direction(
                Cq, q, direction, direction_preconditioner, eigengap_floor
            )
            direction = _soap_normalize_direction_by_mode(
                direction,
                direction_normalizer,
                moment=direction_moment,
                beta=beta,
            )
            return _soap_ns_retract(q + float(eta) * direction)
        fn = self._maybe_compile_basis_fn(
            "basis_step_brockett_weighted_full",
            _soap_basis_step_brockett_weighted_full,
            group,
        )
        weights = self._brockett_weights(q, group)
        return fn(
            C,
            q,
            weights,
            direction_preconditioner,
            eigengap_floor,
            float(eta),
            direction_normalizer,
        )

    def _basis_step_tangent(
        self,
        C,
        q,
        eta,
        direction_normalizer,
        group,
        direction_moment=None,
    ):
        direction_preconditioner = group["basis_direction_preconditioner"]
        eigengap_floor = float(group["basis_eigengap_floor"])
        beta = self._direction_normalizer_beta(group)
        if direction_moment is not None and beta > 0.0:
            Cq = C @ q
            direction = _soap_tangent_direction(Cq, q)
            direction = _soap_maybe_precondition_direction(
                Cq, q, direction, direction_preconditioner, eigengap_floor
            )
            direction = _soap_normalize_direction_by_mode(
                direction,
                direction_normalizer,
                moment=direction_moment,
                beta=beta,
            )
            return _soap_ns_retract(q + float(eta) * direction)
        fn = self._maybe_compile_basis_fn(
            "basis_step_tangent_full",
            _soap_basis_step_tangent_full,
            group,
        )
        return fn(
            C,
            q,
            direction_preconditioner,
            eigengap_floor,
            float(eta),
            direction_normalizer,
        )

    def _run_stiefel_tracker(
        self,
        state,
        precond_list,
        orth_list,
        basis_scales,
        group,
        basis_step_fn,
        use_metrics: bool,
    ):
        """Run a cost-budgeted Stiefel tracker with NS retraction."""
        base_eta = float(group["basis_step_size"])
        max_substeps = max(1, int(group["basis_substeps"]))
        normalize_covariance = bool(group["basis_normalize_covariance"])
        direction_normalizer = group["basis_direction_normalizer"]
        use_controller = group["basis_step_controller"] == "correction_ratio"
        target_ratio = float(group["basis_target_correction_ratio"])
        min_marginal_ratio = float(group["basis_min_marginal_correction_ratio"])
        backtrack_factor = float(group["basis_backtrack_factor"])
        growth_factor = float(group["basis_growth_factor"])
        direction_beta = self._direction_normalizer_beta(group)
        direction_moments = self._direction_moment_list(state, precond_list)

        new_Q = []
        for factor_idx, (m, o) in enumerate(zip(precond_list, orth_list)):
            if isinstance(m, list) or len(m) == 0:
                new_Q.append([])
                continue

            scale = basis_scales[factor_idx]
            eta = self._clamp_basis_eta(base_eta * (1.0 if scale is None else float(scale)), group)
            work_dtype = self._basis_dtype(group)
            if not use_metrics:
                C = m.data.to(work_dtype)
                if normalize_covariance:
                    C = self._normalize_covariance(C)
                current_q = o.data.to(work_dtype)
                if direction_beta > 0.0:
                    direction_moment = direction_moments[factor_idx]
                    for _ in range(max_substeps):
                        current_q = basis_step_fn(
                            C,
                            current_q,
                            eta,
                            direction_normalizer,
                            group,
                            direction_moment,
                        )
                    new_Q.append(current_q.to(self._basis_storage_dtype(group)))
                    continue
                weights = (
                    None
                    if group["basis_stage1_method"] == "tangent"
                    else self._brockett_weights(current_q, group)
                )
                fixed_kernel = self._fixed_brockett_kernel(group, max_substeps)
                current_q = fixed_kernel(C, current_q, weights, float(eta))
                new_Q.append(current_q.to(self._basis_storage_dtype(group)))
                continue

            C_metric = m.data.float()
            if normalize_covariance:
                C_metric = self._normalize_covariance(C_metric)
            C = C_metric if work_dtype == C_metric.dtype else C_metric.to(work_dtype)
            q0 = o.data.float()
            q0_work = q0 if work_dtype == q0.dtype else q0.to(work_dtype)
            best_q = q0
            initial_metrics_t = self._basis_metric_tensors(C_metric, q0, q0, group)
            best_metrics_t = dict(initial_metrics_t)
            best_diag_error = best_metrics_t["diag_err"]
            steps_taken = 0
            accepted_steps = 0
            rejected_steps = 0
            current_q = q0_work
            initial_diag_error = self._metric_float(initial_metrics_t["diag_err"])
            previous_best_diag_error = initial_diag_error
            final_marginal_ratio = 0.0
            direction_moment = direction_moments[factor_idx]

            for substep_idx in range(max_substeps):
                candidate_q = basis_step_fn(
                    C,
                    current_q,
                    eta,
                    direction_normalizer,
                    group,
                    direction_moment,
                )
                candidate_q_metric = candidate_q.float()
                candidate_metrics_t = self._basis_metric_tensors(
                    C_metric, candidate_q_metric, q0, group
                )

                improves_diag = self._metric_leq(
                    candidate_metrics_t["diag_err"], best_diag_error
                )
                if improves_diag:
                    candidate_diag_error = self._metric_float(candidate_metrics_t["diag_err"])
                    marginal_ratio = (
                        (previous_best_diag_error - candidate_diag_error)
                        / max(initial_diag_error, 1e-12)
                    )
                    correction_ratio = (
                        (initial_diag_error - candidate_diag_error)
                        / max(initial_diag_error, 1e-12)
                    )
                    best_diag_error = candidate_metrics_t["diag_err"]
                    best_q = candidate_q_metric
                    best_metrics_t = candidate_metrics_t
                    current_q = candidate_q
                    steps_taken = substep_idx + 1
                    accepted_steps += 1
                    previous_best_diag_error = candidate_diag_error
                    final_marginal_ratio = marginal_ratio
                    if use_controller and (
                        correction_ratio >= target_ratio
                        or marginal_ratio < min_marginal_ratio
                    ):
                        break
                elif use_controller:
                    rejected_steps += 1
                    eta = self._clamp_basis_eta(eta * backtrack_factor, group)
                else:
                    break

            best_metrics = self._metrics_with_improvement(
                best_metrics_t, initial_metrics_t["diag_err"]
            )
            best_metrics["steps_taken"] = float(steps_taken)
            best_metrics["accepted_steps"] = float(accepted_steps)
            best_metrics["rejected_steps"] = float(rejected_steps)
            best_metrics["substeps"] = float(max_substeps)
            best_metrics["accepted_fraction"] = float(accepted_steps) / float(max_substeps)
            best_metrics["basis_step_size"] = float(eta)
            best_metrics["basis_step_size_initial"] = float(
                self._clamp_basis_eta(base_eta * (1.0 if scale is None else float(scale)), group)
            )
            best_metrics["basis_correction_ratio"] = (
                best_metrics["diag_err_improvement"] / max(initial_diag_error, 1e-12)
            )
            best_metrics["basis_marginal_correction_ratio"] = float(final_marginal_ratio)
            best_metrics["basis_scale"] = float(1.0 if scale is None else scale)

            if use_controller:
                new_eta = eta
                used_full_budget = steps_taken == max_substeps and rejected_steps == 0
                if accepted_steps == 0:
                    new_eta = eta
                elif (
                    used_full_budget
                    and best_metrics["basis_correction_ratio"] < target_ratio
                    and final_marginal_ratio >= min_marginal_ratio
                ):
                    new_eta = eta * growth_factor
                new_scale = self._eta_to_scale(new_eta, group)
                basis_scales[factor_idx] = new_scale
                best_metrics["basis_scale_after"] = float(
                    1.0 if new_scale is None else new_scale
                )
                best_metrics["basis_step_size_after"] = float(
                    self._clamp_basis_eta(base_eta * best_metrics["basis_scale_after"], group)
                )
            best_metrics["_track_stats"] = bool(group["basis_track_stats"])
            self._record_basis_stats(best_metrics)
            new_Q.append(best_q.to(self._basis_storage_dtype(group)))

        return new_Q

    # ==================================================================
    # Initialisation
    # ==================================================================

    def _init_preconditioner(self, grad, state, group):
        """Initialise GG (covariance) and Q (eigenbasis, initially None)."""
        max_precond_dim = group["max_precond_dim"]
        precondition_1d = group["precondition_1d"]
        merge_dims = group["merge_dims"]
        shampoo_beta = group["shampoo_beta"]
        if shampoo_beta < 0:
            shampoo_beta = group["betas"][1]

        state["shampoo_beta"] = shampoo_beta
        state["precondition_frequency"] = group["precondition_frequency"]

        g = grad
        if merge_dims and g.dim() > 1:
            g = self._merge_dims(g, max_precond_dim)

        state["GG"] = []
        dims = [g.shape[0]] if g.dim() == 1 else list(g.shape)
        for sh in dims:
            skip = sh > max_precond_dim or (g.dim() == 1 and not precondition_1d)
            if skip:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(sh, sh, device=grad.device, dtype=torch.float32)
                )

        state["Q"] = None
        state["basis_scales"] = self._basis_scale_list(state["GG"])

    # ==================================================================
    # Kronecker factor accumulation
    # ==================================================================

    def _accumulate_outer_products(self, grad, state, group):
        """EMA update of GG with per-mode outer products of grad."""
        beta = state["shampoo_beta"]
        max_precond_dim = group["max_precond_dim"]
        merge_dims = group["merge_dims"]
        precondition_1d = group["precondition_1d"]

        g = grad
        if merge_dims and g.dim() > 1:
            g = self._merge_dims(g, max_precond_dim)
        g_work = g.to(self._covariance_compute_dtype(group))

        if g.dim() == 1:
            if precondition_1d and g.shape[0] <= max_precond_dim:
                outer = g_work.unsqueeze(1) @ g_work.unsqueeze(0)
                state["GG"][0].lerp_(outer.float(), 1 - beta)
        else:
            if g_work.dim() == 2:
                if g_work.shape[0] <= max_precond_dim:
                    state["GG"][0].lerp_((g_work @ g_work.T).float(), 1 - beta)
                if g_work.shape[1] <= max_precond_dim:
                    state["GG"][1].lerp_((g_work.T @ g_work).float(), 1 - beta)
            else:
                for idx, sh in enumerate(g_work.shape):
                    if sh <= max_precond_dim:
                        contract = list(chain(range(idx), range(idx + 1, g_work.dim())))
                        outer = torch.tensordot(g_work, g_work, dims=[contract, contract])
                        state["GG"][idx].lerp_(outer.float(), 1 - beta)

    # ==================================================================
    # Eigenbasis: project / project_back / update
    # ==================================================================

    def _project(self, grad, state, group):
        """Q^T @ grad along each mode (rotate to eigenbasis)."""
        original_shape = grad.shape
        permuted_shape = None
        if group["merge_dims"]:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, group["max_precond_dim"])
        if grad.dim() == 2 and len(state["Q"]) == 2:
            left, right = state["Q"]
            if len(left) > 0:
                matmul_dtype = self._matmul_dtype(left, grad)
                grad = left.T.to(matmul_dtype) @ grad.to(matmul_dtype)
            if len(right) > 0:
                matmul_dtype = self._matmul_dtype(grad, right)
                grad = grad.to(matmul_dtype) @ right.to(matmul_dtype)
            if group["merge_dims"]:
                if permuted_shape is not None:
                    grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    grad = grad.reshape(original_shape)
            return grad
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                grad = grad.permute(list(range(1, grad.dim())) + [0])
        if group["merge_dims"]:
            if permuted_shape is not None:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(self, grad, state, group):
        """Q @ grad along each mode (rotate back from eigenbasis)."""
        original_shape = grad.shape
        permuted_shape = None
        if group["merge_dims"]:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, group["max_precond_dim"])
        if grad.dim() == 2 and len(state["Q"]) == 2:
            left, right = state["Q"]
            if len(left) > 0:
                matmul_dtype = self._matmul_dtype(left, grad)
                grad = left.to(matmul_dtype) @ grad.to(matmul_dtype)
            if len(right) > 0:
                matmul_dtype = self._matmul_dtype(grad, right)
                grad = grad.to(matmul_dtype) @ right.T.to(matmul_dtype)
            if group["merge_dims"]:
                if permuted_shape is not None:
                    grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    grad = grad.reshape(original_shape)
            return grad
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                grad = grad.permute(list(range(1, grad.dim())) + [0])
        if group["merge_dims"]:
            if permuted_shape is not None:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _get_orthogonal_matrix_eigh(self, mat, group):
        """Eigenbasis via eigh."""
        final = []
        for m in mat:
            if isinstance(m, list) or len(m) == 0:
                final.append([])
                continue
            m_f = m.data.float()
            d = m_f.shape[0]
            try:
                _, Q = torch.linalg.eigh(
                    m_f + 1e-30 * torch.eye(d, device=m_f.device)
                )
            except Exception:
                _, Q = torch.linalg.eigh(
                    m_f.to(torch.float64)
                    + 1e-30 * torch.eye(d, device=m_f.device)
                )
                Q = Q.to(m_f.dtype)
            Q = torch.flip(Q, [1])
            final.append(Q)
        return final

    def _reorder_exp_avg_sq(self, state, orth_list, precond_list, group):
        """Sort exp_avg_sq along each mode to follow current estimated eig ordering."""
        axis_second_moment = state.get("exp_avg_sq_axis")
        use_axis_second_moment = (
            group["adam_second_moment"]
            in ("axis_product", "axis_geomean", "axis_logblend", "axis_sum", "axis_max")
            and axis_second_moment is not None
        )
        max_precond_dim = group["max_precond_dim"]
        if group["merge_dims"] and not use_axis_second_moment:
            exp_avg_sq = self._merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        reordered_orth_list = []
        eig_work_dtype = self._basis_dtype(group)
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if isinstance(m, list) or len(m) == 0:
                reordered_orth_list.append([])
                continue
            m_work = m.data.to(eig_work_dtype)
            o_work = o.data.to(eig_work_dtype)
            est_eig = torch.diagonal(o_work.T @ m_work @ o_work)
            sort_idx = torch.argsort(est_eig, descending=True)
            if use_axis_second_moment:
                axis_second_moment[ind] = axis_second_moment[ind].index_select(ind, sort_idx)
            else:
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            reordered_orth_list.append(o[:, sort_idx].to(self._basis_storage_dtype(group)))

        if group["merge_dims"] and not use_axis_second_moment:
            exp_avg_sq = exp_avg_sq.reshape(state["exp_avg_sq"].shape)
        if use_axis_second_moment:
            state["exp_avg_sq_axis"] = axis_second_moment
        else:
            state["exp_avg_sq"] = exp_avg_sq
        return reordered_orth_list

    @staticmethod
    def _axis_second_moment_shapes(shape, device, dtype):
        moments = []
        for dim, size in enumerate(shape):
            moment_shape = [1] * len(shape)
            moment_shape[dim] = size
            moments.append(torch.zeros(moment_shape, device=device, dtype=dtype))
        return moments

    def _update_axis_second_moment(self, state, grad_projected, beta2, eps, mode, blend_alpha):
        if "exp_avg_sq_axis" not in state:
            state["exp_avg_sq_axis"] = self._axis_second_moment_shapes(
                grad_projected.shape,
                grad_projected.device,
                grad_projected.dtype,
            )

        grad_sq = grad_projected.square()
        axis_moments = state["exp_avg_sq_axis"]
        for dim, axis_moment in enumerate(axis_moments):
            reduce_dims = tuple(i for i in range(grad_sq.dim()) if i != dim)
            axis_mean = (
                grad_sq.mean(dim=reduce_dims, keepdim=True)
                if reduce_dims
                else grad_sq
            )
            axis_moment.mul_(beta2).add_(axis_mean, alpha=1.0 - beta2)

        if mode == "axis_logblend":
            var_sum = sum(axis_moments) / float(len(axis_moments))
            var_product = axis_moments[0]
            for axis_moment in axis_moments[1:]:
                var_product = var_product * axis_moment
            if len(axis_moments) > 1:
                global_var = axis_moments[0].mean().clamp(min=eps)
                var_product = var_product / global_var.pow(len(axis_moments) - 1)
            denom_sum = var_sum.clamp_min(eps * eps).sqrt()
            denom_product = var_product.clamp_min(eps * eps).sqrt()
            log_denom = denom_sum.log().lerp(denom_product.log(), float(blend_alpha))
            return log_denom.exp().add_(eps)
        else:
            var_estimate = axis_moments[0]
            for axis_moment in axis_moments[1:]:
                var_estimate = var_estimate * axis_moment
            if len(axis_moments) > 1:
                global_var = axis_moments[0].mean().clamp(min=eps)
                var_estimate = var_estimate / global_var.pow(len(axis_moments) - 1)

        return var_estimate.clamp_min(eps * eps).sqrt().add_(eps)

    def _update_second_moment(self, state, grad_projected, beta2, eps, group):
        mode = group["adam_second_moment"]
        if mode == "elementwise":
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)
            return exp_avg_sq.sqrt().add_(eps)
        if mode == "scalar":
            if "exp_avg_sq_scalar" not in state:
                state["exp_avg_sq_scalar"] = torch.zeros(
                    (), device=grad_projected.device, dtype=grad_projected.dtype
                )
            state["exp_avg_sq_scalar"].mul_(beta2).add_(
                grad_projected.square().mean(), alpha=1.0 - beta2
            )
            return state["exp_avg_sq_scalar"].clamp_min(eps * eps).sqrt().add_(eps)
        if grad_projected.dim() == 0:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)
            return exp_avg_sq.sqrt().add_(eps)
        return self._update_axis_second_moment(
            state,
            grad_projected,
            beta2,
            eps,
            mode,
            group["adam_second_moment_blend_alpha"],
        )

    def _apply_stage1_brockett(
        self, state, precond_list, orth_list, basis_scales, group, use_metrics: bool
    ):
        """Brockett-style ordered diagonalization step with NS retraction."""
        return self._run_stiefel_tracker(
            state,
            precond_list,
            orth_list,
            basis_scales,
            group,
            self._basis_step_brockett,
            use_metrics,
        )

    def _apply_stage1_tangent(
        self, state, precond_list, orth_list, basis_scales, group, use_metrics: bool
    ):
        """Cheaper tangent-space diagonalization step with NS retraction."""
        return self._run_stiefel_tracker(
            state,
            precond_list,
            orth_list,
            basis_scales,
            group,
            self._basis_step_tangent,
            use_metrics,
        )

    def _apply_stage1_eigh(self, precond_list, orth_list, group):
        """Warm eigendecomposition refresh from the current reordered basis."""
        _ = orth_list
        return self._get_orthogonal_matrix_eigh(precond_list, group)

    def _apply_stage1_qr(self, precond_list, orth_list, group):
        """Reference SOAP QR refresh: one power iteration, then QR."""
        final_Q = []
        for m, q_old in zip(precond_list, orth_list):
            if isinstance(m, list) or len(m) == 0:
                final_Q.append([])
                continue

            q0 = q_old.data.float()
            q_new = torch.linalg.qr(m.data.float() @ q0, mode="reduced")[0]
            C_metric = (
                self._normalize_covariance(m.data.float())
                if bool(group["basis_normalize_covariance"])
                else m.data.float()
            )
            if bool(group["basis_track_stats"]):
                metrics = self._basis_metrics(C_metric, q_new, q0, group)
                metrics["diag_err_before"] = self._basis_metrics(C_metric, q0, q0, group)[
                    "diag_err"
                ]
                metrics["diag_err_improvement"] = (
                    metrics["diag_err_before"] - metrics["diag_err"]
                )
                metrics["eigh_fallback"] = 0.0
                metrics["qr_fallback"] = 0.0
                metrics["diag_err_after_fallback"] = metrics["diag_err"]
                self._record_basis_stats(metrics)
            final_Q.append(q_new.to(self._basis_storage_dtype(group)))
        return final_Q

    def _apply_pipeline_fallback(self, precond_list, old_orth_list, candidate_orth_list, group):
        final_Q = []
        threshold = float(group["basis_fallback_diag_err_threshold"])
        track_stats = bool(group["basis_track_stats"])
        for m, q_old, q_candidate in zip(precond_list, old_orth_list, candidate_orth_list):
            if isinstance(m, list) or len(m) == 0:
                final_Q.append([])
                continue
            if threshold <= 0.0 and not track_stats:
                final_Q.append(q_candidate.to(self._basis_storage_dtype(group)))
                continue
            C_metric = (
                self._normalize_covariance(m.data.float())
                if bool(group["basis_normalize_covariance"])
                else m.data.float()
            )
            metrics = self._basis_metrics(C_metric, q_candidate.float(), q_old.float(), group)
            metrics["diag_err_before"] = self._basis_metrics(
                C_metric,
                q_old.float(),
                q_old.float(),
                group,
            )["diag_err"]
            metrics["diag_err_improvement"] = (
                metrics["diag_err_before"] - metrics["diag_err"]
            )
            q_final, metrics = self._maybe_fallback_to_basis(
                m, C_metric, q_candidate, q_old, metrics, group
            )
            metrics["diag_err_improvement"] = (
                metrics["diag_err_before"] - metrics["diag_err"]
            )
            metrics["_track_stats"] = track_stats
            self._record_basis_stats(metrics)
            final_Q.append(q_final.to(self._basis_storage_dtype(group)))
        return final_Q

    @staticmethod
    def _should_run_basis_metrics(state, group):
        frequency = int(group["basis_metric_frequency"])
        if frequency <= 1:
            return True
        refresh_period = max(1, int(state["precondition_frequency"]))
        steps_after_warmup = max(0, int(state["step"]) - int(group["basis_warmup_steps"]))
        refresh_index = max(1, (steps_after_warmup + refresh_period - 1) // refresh_period)
        return (refresh_index - 1) % frequency == 0

    def _update_preconditioner(self, grad, state, group):
        """Full eigenbasis preconditioner update cycle.

        1. Un-project exp_avg from old Q
        2. Accumulate outer products into GG
        3. Update Q (cold eigh or warm tracked/eigh)
        4. Re-project exp_avg into new Q
        """
        if state["Q"] is not None:
            state["exp_avg"] = self._project_back(state["exp_avg"], state, group)

        self._accumulate_outer_products(grad, state, group)

        first_q = state["Q"] is None
        if first_q:
            state["Q"] = self._cast_basis_list(
                self._get_orthogonal_matrix_eigh(state["GG"], group),
                group,
            )
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            precond_list = state["GG"]
            if "basis_scales" not in state or len(state["basis_scales"]) != len(precond_list):
                state["basis_scales"] = self._basis_scale_list(precond_list)
            old_orth_list = self._reorder_exp_avg_sq(state, state["Q"], precond_list, group)
            stage1_method = group["basis_stage1_method"]
            force_fallback = state["step"] <= int(group["basis_warmup_steps"])
            run_tracker_metrics = (
                stage1_method in ("tangent", "brockett")
                and not force_fallback
                and self._should_run_basis_metrics(state, group)
            )
            if force_fallback:
                stage1_Q = self._fallback_basis_list(precond_list, old_orth_list, group)
            elif stage1_method == "none":
                stage1_Q = old_orth_list
            elif stage1_method == "tangent":
                stage1_Q = self._apply_stage1_tangent(
                    state,
                    precond_list,
                    old_orth_list,
                    state["basis_scales"],
                    group,
                    run_tracker_metrics,
                )
            elif stage1_method == "brockett":
                stage1_Q = self._apply_stage1_brockett(
                    state,
                    precond_list,
                    old_orth_list,
                    state["basis_scales"],
                    group,
                    run_tracker_metrics,
                )
            elif stage1_method == "eigh":
                stage1_Q = self._apply_stage1_eigh(precond_list, old_orth_list, group)
            elif stage1_method == "qr":
                stage1_Q = self._apply_stage1_qr(precond_list, old_orth_list, group)
            else:
                raise ValueError(f"Unknown basis_stage1_method: {stage1_method!r}")

            if (
                force_fallback
                or stage1_method == "qr"
                or (stage1_method in ("tangent", "brockett") and not run_tracker_metrics)
                or (
                    stage1_method in ("tangent", "brockett")
                    and float(group["basis_fallback_diag_err_threshold"]) <= 0.0
                )
            ):
                final_Q = stage1_Q
            else:
                final_Q = self._apply_pipeline_fallback(precond_list, old_orth_list, stage1_Q, group)
            self._finalize_basis_change(state, old_orth_list, final_Q, group)

        # Project exp_avg (and exp_avg_sq on cold start) into eigenbasis.
        # On step 0 (cold start) we must project now so exp_avg/exp_avg_sq enter
        # the eigenbasis before the first actual parameter update.
        if first_q:
            state["exp_avg"] = self._project(state["exp_avg"], state, group)
            if group["adam_second_moment"] == "elementwise":
                state["exp_avg_sq"] = self._project(state["exp_avg_sq"], state, group).abs()
        elif state["step"] > 0:
            state["exp_avg"] = self._project(state["exp_avg"], state, group)

    # ==================================================================
    # Main step
    # ==================================================================

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            loss = closure()
        else:
            loss = None
        self._basis_stats_accum = None
        self.latest_basis_stats = {}
        self.just_gathered_basis_stats = False

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # ---- First-call init ----
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    self._init_preconditioner(grad, state, group)
                    self._update_preconditioner(grad, state, group)
                    continue  # skip first step (SOAP convention)

                state["step"] += 1

                # ---- Project → Adam in eigenbasis → Project back ----
                grad_projected = self._project(grad, state, group)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)

                denom = self._update_second_moment(
                    state, grad_projected, beta2, group["eps"], group
                )

                bc1 = 1.0 - beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                update_projected = exp_avg / bc1
                update_projected = update_projected / denom

                norm_grad = self._project_back(update_projected, state, group)

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

                p.add_(norm_grad, alpha=-group["lr"] * (bc2 ** 0.5))

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                self._update_preconditioner(grad, state, group)

        self._finalize_basis_stats()
        return loss

