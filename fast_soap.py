"""Opt-in fast J1 candidate; not the qualified research control.

Set float32 matmul precision to 'high' once before construction. Run optimizer
steps outside autocast. Persistent state is FP32; only the candidate/correction
products use BF16. GPU speed and learning quality require separate measurement.
"""
import math

import torch

from soap_reference import ResearchSOAP


def _bf16_mm_fp32(left, right):
    """BF16 inputs, FP32 output, without an intermediate BF16 result on CUDA."""
    left, right = left.bfloat16(), right.bfloat16()
    if left.is_cuda:
        return torch.mm(left, right, out_dtype=torch.float32)
    # CPU audit emulation: BF16 values are represented exactly in FP64. This
    # path checks the arithmetic contract, not the CUDA implementation or cost.
    # FP64 also avoids CPU codegen cancelling an inverse BF16->FP32 cast.
    return (left.double() @ right.double()).float()


def fast_warm_refresh(covariance, q):
    """Pure fullgraph kernel for matching square FP32 C/Q, with no mutation.

    Jacobi damping=.01, step=.5, skew-Frobenius cap=.5, two residual
    corrections. Relative B units remove trace(C) normalization. The 1e-30
    scale floor handles zero covariance; extreme scales can differ from J1.
    All FP32 products inherit the caller's precision setting.
    """
    b = q.T @ covariance @ q
    b = .5 * (b + b.T)
    diagonal = b.diagonal()
    scale = diagonal.abs().amax().clamp_min(1e-30)
    gap = (diagonal[:, None] - diagonal[None, :]) / scale
    # Symmetric B and antisymmetric gap already produce a skew generator.
    rotation = -.5 * (b / scale) * gap / (gap.square() + .01**2)
    bound = rotation.norm() / math.sqrt(2)
    rotation = rotation * (.5 / bound.clamp_min(1e-12)).clamp(max=1)
    y = q + _bf16_mm_fp32(q, rotation)
    for _ in range(2):
        residual = y.T @ y
        residual.diagonal().sub_(1.)
        y = y - .5 * _bf16_mm_fp32(y, residual)
    return y


class FastSOAP(ResearchSOAP):
    """Fixed-policy warm SOAP candidate sharing the reference update ordering.

    Two-sided dense FP32 parameters/state, permutation-only V on QR repair,
    unchanged V on warm refresh, caller-precision first-moment transport.
    compile_warm=False is an explicit eager audit path, not an error fallback.
    The complete optimizer step is still eager; the warm kernel is compiled.
    """

    def __init__(self, params, *, lr=.001, betas=(.9, .999), shampoo_beta=.999,
                 eps=1e-8, weight_decay=.05, precondition_frequency=20,
                 hard_reset_interval=200, max_precond_dim=10000,
                 compile_warm=True):
        if torch.get_float32_matmul_precision() != 'high':
            raise ValueError("FastSOAP requires caller-owned matmul precision 'high'")
        super().__init__(
            params, lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay, precondition_frequency=precondition_frequency,
            hard_reset_interval=hard_reset_interval, max_precond_dim=max_precond_dim,
            precondition_mode='all', covariance_compute_dtype='float32',
            basis_method='warm', variance_policy='permutation',
            transport_precision='inherit',
        )
        for group in self.param_groups:
            group.update(fast_soap_version=1, basis_lr=.5,
                         basis_jacobi_damping=.01, basis_rotation_cap=.5,
                         basis_rotation_cap_norm='skew_frobenius',
                         basis_ns_iterations=2)
        self._validate_fast_groups(self.param_groups)
        # CPU emulation converts through BF16. The installed CPU compiler's
        # 'pointless_convert' pattern removes intermediate rounding even with
        # emulate_precision_casts=True. CUDA uses real BF16 mm operands
        # and keeps its normal optimization passes. No global flags change.
        compile_options = {'emulate_precision_casts': True}
        if any(p.device.type == 'cpu' for group in self.param_groups for p in group['params']):
            compile_options['pattern_matcher'] = False
        self._warm_kernel = (
            torch.compile(fast_warm_refresh, fullgraph=True, dynamic=False,
                          options=compile_options)
            if compile_warm else fast_warm_refresh
        )

    @staticmethod
    def _validate_fast_groups(groups):
        expected = dict(fast_soap_version=1, basis_method='warm',
                        variance_policy='permutation', transport_precision='inherit',
                        covariance_compute_dtype='float32', precondition_mode='all',
                        basis_lr=.5, basis_jacobi_damping=.01,
                        basis_rotation_cap=.5, basis_rotation_cap_norm='skew_frobenius',
                        basis_ns_iterations=2)
        for group in groups:
            if any(group.get(key) != value for key, value in expected.items()):
                raise ValueError('Parameter groups must match the FastSOAP v1 policy')
            frequency = group['precondition_frequency']
            repair = group['hard_reset_interval']
            if (type(frequency) is not int or frequency < 1 or
                    type(repair) is not int or repair < 0 or
                    (repair and repair % frequency)):
                raise ValueError('Invalid FastSOAP refresh/repair schedule')
            if any(p.ndim != 2 or p.dtype != torch.float32 or p.layout != torch.strided
                   for p in group['params']):
                raise ValueError('FastSOAP requires dense 2D FP32 parameters')

    def load_state_dict(self, state_dict):
        # Validate policy before allowing the parent to replace parameter groups.
        groups = [dict(group, params=current['params']) for group, current in
                  zip(state_dict['param_groups'], self.param_groups)]
        self._validate_fast_groups(groups)
        return super().load_state_dict(state_dict)

    def _gauge_step_one(self, covariance, q, group):
        return self._warm_kernel(covariance, q)
