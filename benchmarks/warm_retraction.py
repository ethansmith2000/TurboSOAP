"""Experimental fixed retractions; production SOAP remains unchanged."""
from dataclasses import dataclass

import torch

from soap import SOAP, _clip_rotation, _damped_jacobi_generator, _sym


@dataclass(frozen=True)
class Retraction:
    name: str
    order: int = 5
    iterations: int = 1
    scaled: bool = False
    normalize_covariance: bool = True
    cap_mode: str = 'average'
    cap: float = .1


VARIANTS = (
    Retraction('scaled_ns2', iterations=2, scaled=True),
    Retraction('scaled_ns6', iterations=6, scaled=True),
    Retraction('local_cubic1', order=3),
    Retraction('local_cubic2', order=3, iterations=2),
    Retraction('local_quintic1'),
    Retraction('local_quintic2', iterations=2),
    Retraction('rawcov_quintic1', normalize_covariance=False),
    Retraction('spectral_cubic1_cap05', order=3, cap_mode='spectral_bound', cap=.5),
    Retraction('spectral_cubic2_cap05', order=3, iterations=2, cap_mode='spectral_bound', cap=.5),
    Retraction('spectral_quintic1_cap05', cap_mode='spectral_bound', cap=.5),
    Retraction('spectral_quintic1_cap02', cap_mode='spectral_bound', cap=.2),
    Retraction('rawcov_spectral_quintic1_cap05', normalize_covariance=False, cap_mode='spectral_bound', cap=.5),
)


def candidate(cov, q, group, normalize_covariance=True):
    b = (SOAP._basis_covariance(cov, q) if normalize_covariance
         else _sym(q.float().T @ cov.float() @ q.float()))
    raw = group['basis_lr'] * _damped_jacobi_generator(b, group['basis_jacobi_damping'])
    rotation = _clip_rotation(raw, group['basis_rotation_cap'], group['basis_rotation_cap_mode'])
    return q.float() + q.float() @ rotation, rotation, raw


def retract(x, variant):
    """Fixed work only: no online measurement or Python tensor-dependent branch."""
    if variant.order not in (3, 5) or variant.iterations < 1:
        raise ValueError('Expected cubic/quintic and a positive iteration count')
    z = x.float()
    gram = z.T @ z
    if variant.scaled:
        upper = gram.abs().sum(dim=-1).amax().clamp_min(1e-12)
        scale = (1.25 / upper.sqrt()).clamp(max=1.0)
        z = z * scale
        gram = gram * scale.square()
    for index in range(variant.iterations):
        if variant.order == 3:
            z = 1.5 * z - .5 * (z @ gram)
        else:
            correction = -10.0 * gram + 3.0 * (gram @ gram)
            z = .125 * (15.0 * z + z @ correction)
        if index + 1 < variant.iterations:
            gram = z.T @ z
    return z


def warm(cov, q, group, variant):
    local_group = {**group, 'basis_rotation_cap_mode':variant.cap_mode,
                   'basis_rotation_cap':variant.cap}
    x, _, _ = candidate(cov, q, local_group, variant.normalize_covariance)
    return retract(x, variant)
