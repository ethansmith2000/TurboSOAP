"""Moderator check M1b (CPU, float64): how hard does the production Frobenius cap bind, vs n?

Production: a = min(1, 0.5 / (||A||_F/sqrt2)).  Hypothesis before running: ||A||_F of a skew correction
spread over many pairs grows with n while the cap is n-independent.  Result: on these inputs the pre-cap
||A||_F/sqrt2 is 0.63-1.10 with no clear trend in n (128-1024), so the cap binds (scale 0.46-0.79) at every n,
while the spectral norm stays 0.13-0.30.  We report the pre-cap ||A||_F/sqrt2, the scale a, the spectral norm
||A||_2 (what governs Newton-Schulz convergence), and r_C after one refresh with (i) production cap,
(ii) a spectral cap ||aA||_2 <= 0.5, (iii) no cap; step factor 0.5 (production) in all three.
"""
import math
import sys

import torch

sys.path.insert(0, '/workspace/TurboSOAP/research/reviews/panel_20261120/checks')
from m1_half_step import make_inputs, rc, polar  # noqa: E402

torch.manual_seed(1)


def skew(c, q, factor=.5, damping=.01):
    b = q.T @ c @ q
    b = .5 * (b + b.T)
    d = torch.diagonal(b)
    s = d.abs().max()
    h = (d[:, None] - d[None, :]) / s
    return -factor * (b / s) * h / (h.square() + damping ** 2)


print('--- cap binding ---')
for n in (128, 256, 512, 1024):
    for rot in (.05, .2):
        c, q = make_inputs(n, rot)
        a = skew(c, q)
        fro = (a.norm() / math.sqrt(2)).item()
        spec = torch.linalg.matrix_norm(a, ord=2).item()
        scale = min(1., .5 / fro)
        r_prod = rc(c, polar(q + q @ (scale * a)))
        r_spec = rc(c, polar(q + q @ (min(1., .5 / spec) * a)))
        r_none = rc(c, polar(q + q @ a))
        print(f'n={n} rot={rot} rC_in={rc(c, q):.4f} fro/sqrt2={fro:.3f} cap_scale={scale:.3f} '
              f'spec={spec:.3f} rC[prod cap]={r_prod:.4f} rC[spectral cap .5]={r_spec:.4f} rC[no cap]={r_none:.4f}',
              flush=True)
