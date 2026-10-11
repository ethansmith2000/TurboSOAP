"""Per-instance warm hook; unchanged ResearchSOAP step/QR/M/V paths."""
from collections import Counter
from contextlib import contextmanager
from functools import partial
import math
import torch
from common_y_kernel import warm_prefix, retract_two


@contextmanager
def precision(mode):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(mode)
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def eager_hybrid(cov, q, mixed=False):
    with precision('high'):
        y = warm_prefix(cov, q, cov_mixed=False, candidate_mixed=mixed)
    with precision('highest'):
        return retract_two(y)


class HybridKernel:
    """Compiled callables hold no optimizer or state tensors; counters are scalar."""
    def __init__(self, compiler=torch.compile):
        self.prefix = compiler(partial(warm_prefix, cov_mixed=False, candidate_mixed=True),
                               mode='default', fullgraph=True, dynamic=False)
        self.retraction = compiler(retract_two, mode='default', fullgraph=True, dynamic=False)
        self.calls = Counter()
        self.probe = False
        self.probes = []

    def __call__(self, cov, q, group):
        validate_group(group)
        if cov.dtype != torch.float32 or q.dtype != torch.float32 or cov.shape != q.shape or q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError('Hybrid needs matching square FP32 covariance/basis')
        with precision('high'):
            y = self.prefix(cov, q)
        with precision('highest'):
            new = self.retraction(y)
        self.calls[str(len(q))] += 1
        if self.probe:
            # Additional reference/metrics are charged to this diagnostic call;
            # the complete optimizer timing sample is excluded from steady means.
            reference = eager_hybrid(cov, q, mixed=False)
            with precision('highest'):
                delta = new - q
                target = reference - q
                movement = (delta-target).norm()/target.norm().clamp_min(1e-12)
                cosine = (delta*target).sum()/(delta.norm()*target.norm()).clamp_min(1e-24)
                basis = (new-reference).norm()/reference.norm().clamp_min(1e-12)
                gram = (new.T@new-torch.eye(len(q), device=q.device)).norm()/math.sqrt(len(q))
                row = dict(dimension=len(q), basis_relative=float(basis), movement_relative=float(movement),
                           movement_cosine=float(cosine), gram=float(gram))
            row['passed'] = (all(math.isfinite(v) for v in row.values()) and
                row['basis_relative'] <= 1e-3 and row['movement_relative'] <= .1 and
                row['movement_cosine'] >= .995 and row['gram'] < .005)
            self.probes.append(row)
            if not row['passed']:
                raise FloatingPointError('Actual-state hybrid mixed screen failed: '+str(row))
        return new


def validate_group(g):
    required = dict(basis_method='warm', precondition_frequency=20, hard_reset_interval=200,
        variance_policy='permutation', transport_precision='highest', basis_lr=.5,
        basis_jacobi_damping=.01, basis_ns_iterations=2, basis_rotation_cap=.5,
        basis_rotation_cap_mode='skew_frobenius', research_movement_policy='fro_cubic2_cap05')
    if any(g.get(k) != v for k,v in required.items()):
        raise ValueError('Unsupported hybrid numerical policy')


def configure_hybrid(matrix, kernel):
    for group in matrix.param_groups:
        validate_group(group)
    matrix._gauge_step_one = kernel


def build_arm(model, config, arm, kernel):
    from cadence_training_contract import build_arm as baseline_arm
    from hybrid_training_contract import ARMS
    if arm not in ARMS:
        raise ValueError('Unknown hybrid pilot arm')
    opt=baseline_arm(model,config,'warm20' if arm=='hybrid20' else arm)
    if arm=='hybrid20':configure_hybrid(opt.matrix,kernel)
    return opt
