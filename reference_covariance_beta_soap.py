"""Covariance-memory experiment; adaptive beta2 and the qualified update stay fixed."""
from functools import partial
import math
from numbers import Real

import torch

from reference_blocked_soap import kind, tiles
from reference_sparse_soap import SparseReferenceBlockedSOAP


def covariance_with_beta(g, momentum, left_cov, right_cov, meta, shampoo_beta):
    moment = .9 * momentum + .1 * g
    t = tiles(g, meta)
    if meta[-2]:
        left_cov = shampoo_beta * left_cov + (1 - shampoo_beta) * (t @ t.transpose(-1, -2))
    if meta[-1]:
        right_cov = shampoo_beta * right_cov + (1 - shampoo_beta) * (t.transpose(-1, -2) @ t)
    return moment, left_cov, right_cov


class CovarianceBetaSOAP(SparseReferenceBlockedSOAP):
    """Change only covariance EMA decay; inherit the complete qualified step body."""

    def __init__(self, named_parameters, arm='qr', shampoo_beta=.999, refresh_interval=50, **kwargs):
        if arm not in ('qr', 'warm', 'sparse'):
            raise ValueError('Require qr, warm or sparse arm')
        if isinstance(shampoo_beta, bool) or not isinstance(shampoo_beta, Real) or not math.isfinite(shampoo_beta) or not 0 <= shampoo_beta < 1:
            raise ValueError('shampoo_beta must be a finite real in [0, 1)')
        super().__init__(named_parameters, arm='sparse', refresh_interval=refresh_interval, **kwargs)
        self.arm = arm
        self.shampoo_beta = float(shampoo_beta)
        # Preserve the original callable and literal arithmetic at the baseline.
        if self.shampoo_beta != .999:
            fn = partial(covariance_with_beta, shampoo_beta=self.shampoo_beta)
            self.covariance = torch.compile(fn, fullgraph=True, dynamic=False,
                options={'emulate_precision_casts': True}) if kwargs.get('compile_kernels', True) else fn

    def refresh_kind(self, call):
        return super().refresh_kind(call) if self.arm == 'sparse' else kind(call, self.arm)
