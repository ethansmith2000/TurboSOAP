"""One-sided Newton-Schulz coefficients for the warm retraction Y = Q(I+A), A skew.
sigma(Y) = sqrt(1+theta^2) >= 1 for exact Q; BF16 adds a little noise on both sides."""
import numpy as np, torch, json, sys

def minimax(lo, hi, degree=3, grid=4000, iters=3000):
    """Chebyshev fit of 1 by an odd polynomial on [lo,hi] (Lawson's algorithm)."""
    s = np.linspace(lo, hi, grid)
    powers = [1, 3, 5][: (degree + 1) // 2]
    V = np.stack([s ** p for p in powers], 1)
    w = np.full(grid, 1.0 / grid)
    for _ in range(iters):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(V * sw[:, None], sw, rcond=None)[0]
        err = np.abs(V @ c - 1)
        w = w * err; w = w / w.sum()
    return c, float(np.abs(V @ c - 1).max())


def apply_poly(y, coeffs, bf16):
    """y <- sum c_k y (y^T y)^k, written in residual form like production."""
    n = y.shape[-1]; eye = torch.eye(n, dtype=torch.float64)
    yd = y.double()
    gram = yd.mT @ yd                       # FP32-accum of BF16 operands ~ exact here
    e = gram - eye
    if bf16: e = e.to(torch.bfloat16).double()
    # rewrite p in powers of E: G = I+E
    c = list(coeffs) + [0, 0]
    c1, c3, c5 = c[0], c[1], c[2]
    a0 = c1 + c3 + c5; a1 = c3 + 2 * c5; a2 = c5
    corr = a1 * (yd @ e)
    if a2:
        e2 = e @ e
        if bf16: e2 = e2.to(torch.bfloat16).double()
        corr = corr + a2 * (yd @ e2)
    out = a0 * yd + corr
    return out.to(torch.bfloat16) if bf16 else out

def errs(y):
    yd = y.double(); n = yd.shape[-1]
    r = yd.mT @ yd - torch.eye(n, dtype=torch.float64)
    return (r.norm(dim=(-2, -1)) / n ** .5).max().item(), torch.linalg.matrix_norm(r, ord=2).max().item()

def trial(n, batch, cap, shape, schemes, bf16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64)).Q
    if bf16: q = q.to(torch.bfloat16)
    if shape == 'spread':
        a = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    else:  # rank-2k skew: spectral norm close to Frobenius
        u = torch.randn(batch, n, 2, generator=g, dtype=torch.float64)
        a = u[..., :1] @ u[..., 1:].mT
    a = a - a.mT
    a = a * (cap / (a.norm(dim=(-2, -1), keepdim=True) / 2 ** .5))   # production bound: ||A||_F/sqrt2 = cap
    qd = q.double()
    y = qd + qd @ (a.to(torch.bfloat16).double() if bf16 else a)
    if bf16: y = y.to(torch.bfloat16)
    out = {'input': errs(y)}
    for name, steps in schemes.items():
        z = y
        for cf in steps: z = apply_poly(z, cf, bf16)
        out[name] = errs(z)
    out['theta_max'] = torch.linalg.matrix_norm(a, ord=2).max().item()
    return out

if __name__ == '__main__':
    std = (1.5, -0.5)
    rows = []
    for cap in (0.5, 1.0):
        # production bound is ||A||_F/sqrt2 <= cap, so ||A||_2 <= cap
        hi = (1 + cap ** 2) ** .5 + 0.01
        lo = 0.99
        c3, t3 = minimax(lo, hi, 3)
        c5, t5 = minimax(lo, hi, 5)
        # second step after a tuned first step: interval shrinks to [1-t3, 1+t3]
        c3b, t3b = minimax(1 - t3 - .01, 1 + t3 + .01, 3)
        rows.append(dict(cap=cap, interval=[lo, hi], cubic=list(c3), cubic_err=t3, quintic=list(c5), quintic_err=t5,
                         std_err=max(abs(1.5 * s - .5 * s ** 3 - 1) for s in np.linspace(lo, hi, 4000))))
        schemes = {'std x1': [std], 'std x2 (production)': [std, std], 'tuned cubic x1': [tuple(c3)],
                   'tuned cubic x2': [tuple(c3), tuple(c3b)], 'tuned quintic x1': [tuple(c5)]}
        for shape in ('spread', 'rank2'):
            for bf16 in (False, True):
                r = trial(512, 4, cap, shape, schemes, bf16)
                rows.append(dict(cap=cap, shape=shape, bf16=bf16, **{k: v for k, v in r.items()}))
    for r in rows: print(json.dumps(r))
