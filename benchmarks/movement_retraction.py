"""Experimental skew-Frobenius bounds; no production optimizer changes."""
import math
import torch
from soap import SOAP, _damped_jacobi_generator
try:
    from .warm_retraction import Retraction, retract
except ImportError:
    from warm_retraction import Retraction, retract


POLICIES = {
    'fro_cubic2_cap05': Retraction('fro_cubic2_cap05', order=3, iterations=2, cap_mode='skew_frobenius', cap=.5),
    'fro_quintic1_cap04': Retraction('fro_quintic1_cap04', cap_mode='skew_frobenius', cap=.4),
}


def clip_skew_frobenius(a, cap):
    """For real skew A, singular values occur in pairs: ||A||2 <= ||A||F/sqrt(2).

    Caller supplies the antisymmetric Jacobi generator. No spectral solve or
    tensor-dependent host branch is used. Small natural steps stay unchanged.
    """
    if cap <= 0:
        raise ValueError('cap must be positive')
    upper = a.norm() / math.sqrt(2)
    return a * (cap / upper.clamp_min(1e-12)).clamp(max=1)


def warm(cov, q, group, policy):
    b = SOAP._basis_covariance(cov, q)
    raw = group['basis_lr'] * _damped_jacobi_generator(b, group['basis_jacobi_damping'])
    a = clip_skew_frobenius(raw, policy.cap)
    return retract(q.float() + q.float() @ a, policy)


def configure_matrix(matrix, name):
    policy = POLICIES[name]
    for g in matrix.param_groups:
        g.update(research_movement_policy=name, basis_ns_iterations=policy.iterations,
                 basis_rotation_cap=policy.cap, basis_rotation_cap_mode='skew_frobenius')
    matrix._gauge_step_one = lambda cov, q, g: warm(cov, q, g, POLICIES[g['research_movement_policy']])
