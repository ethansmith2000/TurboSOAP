"""Moderator check M1 (CPU, float64 reference + production BF16 path).

Claim: for |h_ij| >> damping, the production skew A_ij -> -1/2 * B_ij/(d_i - d_j), so to first order
B+_ij ~= B_ij/2 (a half Jacobi step).  We measure the off-diagonal Frobenius reduction of one warm refresh
with step factor f in {0.5 (production), 1.0}, without the cap and with exact polar orthogonalization,
on synthetic inputs like the microbenchmark's (power-law spectrum, Q = qr(U(I+A)) with A ~ rot/sqrt(n)).
Also runs the production warm_basis (BF16, CPU emulation) for reference.
"""
import math
import sys

import torch

sys.path.insert(0, '/workspace/TurboSOAP')
from bf16_soap import warm_basis  # noqa: E402  (import only; nothing is modified)

torch.set_num_threads(4)
torch.manual_seed(0)


def make_inputs(n, rot, steps=40, beta=.95):
    lam = torch.arange(1, n + 1, dtype=torch.float64).pow(-1.0)
    s = (lam / lam.sum()).sqrt()
    u, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    c = torch.zeros(n, n, dtype=torch.float64)
    k = max(n // 4, 64)
    for _ in range(steps):
        g = u @ (s[:, None] * torch.randn(n, k, dtype=torch.float64))
        c = beta * c + (1 - beta) * g @ g.T
    a = torch.randn(n, n, dtype=torch.float64) * rot / math.sqrt(n)
    a = (a - a.T) / 2
    q, r = torch.linalg.qr(u @ (torch.eye(n, dtype=torch.float64) + a))
    q = q * torch.sign(torch.diagonal(r))
    return c, q


def rc(c, q):
    b = q.T @ c @ q
    off = b - torch.diag(torch.diagonal(b))
    return (off.norm() / b.norm()).item()


def polar(y):
    uu, _, vh = torch.linalg.svd(y)
    return uu @ vh


def warm_f64(c, q, factor, damping=.01, cap=None, exact_jacobi=False):
    b = q.T @ c @ q
    b = .5 * (b + b.T)
    d = torch.diagonal(b)
    s = d.abs().max()
    h = (d[:, None] - d[None, :]) / s
    if exact_jacobi:
        # simultaneous exact 2x2 Jacobi angles: tan(2 th) = 2 B_ij / (d_j - d_i), skew generator th_ij
        # principal branch |th| <= pi/4: th_ij = 1/2 atan(2 B_ij / (d_j - d_i)), antisymmetric by construction
        dd = d[None, :] - d[:, None]
        th = .5 * torch.atan2(2 * b * torch.sign(dd), dd.abs())
        th = th - torch.diag(torch.diagonal(th))
        a = .5 * (th - th.T)
    else:
        a = -factor * (b / s) * h / (h.square() + damping ** 2)
    if cap is not None:
        bound = a.norm() / math.sqrt(2)
        a = a * min(1., cap / max(bound.item(), 1e-12))
    return polar(q + q @ a)


if __name__ == '__main__':
    for n in (128, 256, 512):
        for rot in (.05, .2, .5):
            c, q = make_inputs(n, rot)
            r0 = rc(c, q)
            row = dict(n=n, rot=rot, rC_in=r0)
            row['f=0.5'] = rc(c, warm_f64(c, q, .5))
            row['f=0.5,cap.5'] = rc(c, warm_f64(c, q, .5, cap=.5))
            row['f=1.0'] = rc(c, warm_f64(c, q, 1.))
            row['f=0.75'] = rc(c, warm_f64(c, q, .75))
            row['jacobi_exact_sim'] = rc(c, warm_f64(c, q, None, exact_jacobi=True))
            qb = warm_basis(c.float().to(torch.bfloat16)[None], q.float().to(torch.bfloat16)[None],
                            torch.bfloat16, torch.bfloat16)[0].double()
            row['production_bf16'] = rc(c, qb)
            print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
