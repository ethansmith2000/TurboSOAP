"""Moderator check: Cholesky-QR variants vs Householder QR (CPU float32; TF32 emulated by rounding the Gram
operands to a 10-bit mantissa).  Inputs mimic the microbenchmark: power-law C (lambda_i ~ 1/i, EMA of 40 G G^T
with G = U diag(lambda^1/2) X, X n x n/4), Q = qr(U (I + A)), A skew ~ 0.2/sqrt(n), both rounded to BF16.

Variants (all use Z = C Q in FP32 and the Rayleigh sort):
  chol_fp32      unshifted CholQR, true FP32 Gram
  chol_tf32      unshifted CholQR, TF32-rounded Gram operands (what 'high' matmul precision gives on CUDA)
  chol2_fp32     CholQR2 (two passes), true FP32
  ridge_chol2    Z = (C + tau*dmax*I) Q, column-equilibrated, CholQR2 (Fable A1 / Neon A5), tau in {1e-4, 1e-3}
Reported: Cholesky failure (info != 0), normalized Gram error after BF16 storage, and distance to Householder QR's
basis = max over columns of (1 - |<q_i, q_i^HH>|) and mean of the same, after sign alignment (same column order).
"""
import math
import sys

import torch

torch.set_num_threads(4)
f32 = torch.float32


def tf32(x):
    # round-to-nearest to 10 explicit mantissa bits (TF32), keep FP32 exponent
    xi = x.contiguous().view(torch.int32)
    rnd = ((xi >> 13) & 1) + 0xFFF
    xi = (xi + rnd) & ~0x1FFF
    return xi.view(f32)


def bf16(x):
    return x.to(torch.bfloat16).to(f32)


def inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    lam = torch.arange(1, n + 1, dtype=torch.float64).pow(-1.0)
    s = (lam / lam.sum()).sqrt().to(f32)
    u, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    c = torch.zeros(n, n)
    for _ in range(40):
        gg = u @ (s[:, None] * torch.randn(n, n // 4, generator=g))
        c = .95 * c + .05 * gg @ gg.T
    a = torch.randn(n, n, generator=g) * .2 / math.sqrt(n)
    a = (a - a.T) / 2
    q, r = torch.linalg.qr(u @ (torch.eye(n) + a))
    q = q * torch.sign(torch.diagonal(r))
    return bf16(c), bf16(q)


def chol_pass(z, gram_round=None, shift=0.):
    zz = gram_round(z) if gram_round else z
    gram = zz.T @ zz
    if shift:
        gram = gram + shift * torch.eye(len(gram))
    L, info = torch.linalg.cholesky_ex(gram)
    if info != 0:
        return None
    return torch.linalg.solve_triangular(L.T, z, upper=True, left=False)


def run(n, seed=0):
    c, q = inputs(n, seed)
    z0 = c @ q
    order = (q * z0).sum(0).argsort(descending=True)
    dmax = (q * z0).sum(0).max()
    ref, r = torch.linalg.qr(z0[:, order])
    ref = ref * torch.sign(torch.diagonal(r))
    out = {}

    def score(qn):
        if qn is None or not torch.isfinite(qn).all():
            return 'FAIL'
        qb = bf16(qn)
        gram = ((qb.double().T @ qb.double() - torch.eye(n, dtype=torch.float64)).norm() / math.sqrt(n)).item()
        dots = (qn * ref).sum(0).abs()
        return f'gram={gram:.4f} col_dist_mean={(1 - dots).mean().item():.2e} max={(1 - dots).max().item():.2e}'

    z = z0[:, order]
    out['chol_fp32'] = score(chol_pass(z))
    out['chol_tf32'] = score(chol_pass(z, tf32))
    p1 = chol_pass(z)
    out['chol2_fp32'] = score(chol_pass(p1) if p1 is not None else None)
    for tau in (1e-4, 1e-3):
        zr = (c + tau * dmax * torch.eye(n)) @ q
        zr = zr[:, order]
        zr = zr / zr.norm(dim=0, keepdim=True)
        p1 = chol_pass(zr, shift=0.)
        if p1 is None:
            p1 = chol_pass(zr, shift=1e-6 * n)
        out[f'ridge{tau:g}_chol2_fp32'] = score(chol_pass(p1) if p1 is not None else None)
        p1t = chol_pass(zr, tf32)
        if p1t is None:
            p1t = chol_pass(zr, tf32, shift=1e-6 * n)
        out[f'ridge{tau:g}_chol2_tf32'] = score(chol_pass(p1t, tf32) if p1t is not None else None)
    kappa = torch.linalg.eigvalsh(c.double())
    print(f'n={n}: kappa(C)~{(kappa[-1] / kappa.clamp_min(1e-30)[0]).item():.2e}', flush=True)
    for k, v in out.items():
        print(f'   {k:22s} {v}', flush=True)


if __name__ == '__main__':
    for n in [int(x) for x in (sys.argv[1:] or ['512', '1024', '2048'])]:
        run(n)
