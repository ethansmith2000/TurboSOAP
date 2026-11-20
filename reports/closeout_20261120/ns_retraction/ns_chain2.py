"""Four consecutive warm retractions from a BF16 basis, production-style rounding.
Pinned cubic: y <- y + c3 * y E with c3 = -1/(s(s+1)), s = sqrt(1 + theta_bound^2),
theta_bound = ||A||_F / sqrt(2) (the production cap quantity; >= ||A||_2 for skew A).
p(1) = 1 and p(s) = 1 exactly; c3 -> -1/2 (standard NS) as theta_bound -> 0."""
import json, torch
from ns_coeffs import errs
bf = lambda x: x.to(torch.bfloat16).double()

def skew(batch, n, cap, shape, g):
    if shape == 'spread':
        a = torch.randn(batch, n, n, generator=g, dtype=torch.float64); a = a - a.mT
    elif shape.startswith('rank'):
        k = int(shape[4:]); u = torch.randn(batch, n, 2 * k, generator=g, dtype=torch.float64)
        a = u[..., :k] @ u[..., k:].mT; a = a - a.mT
    else:
        u = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64)).Q
        th = 0.8 ** torch.arange(n // 2, dtype=torch.float64)
        j = torch.zeros(n, n, dtype=torch.float64)
        for i, t in enumerate(th): j[2*i, 2*i+1], j[2*i+1, 2*i] = t, -t
        a = u @ j @ u.mT
    return a * (cap / (a.norm(dim=(-2, -1), keepdim=True) / 2 ** .5))

def ns(y, c3):
    """Production residual form: y <- y + c3 * y (Y^T Y - I); c3 = -1/2 is standard NS."""
    yd = y.double(); e = yd.mT @ yd - torch.eye(y.shape[-1], dtype=torch.float64)
    return (yd + c3 * (yd @ bf(e))).to(torch.bfloat16)

def ns3(y):
    """Third-order NS: y (I - E/2 + 3E^2/8)."""
    yd = y.double(); e = yd.mT @ yd - torch.eye(y.shape[-1], dtype=torch.float64)
    e2 = bf(bf(e) @ bf(e))
    return (yd - .5 * (yd @ bf(e)) + .375 * (yd @ e2)).to(torch.bfloat16)

def retract(q, a, scheme):
    qd = q.double()
    tb = a.norm(dim=(-2, -1), keepdim=True) / 2 ** .5
    c3_pin = -1 / ((1 + tb ** 2).sqrt() * ((1 + tb ** 2).sqrt() + 1))
    if scheme.startswith('exp2'):          # identity kept in FP32 like production: q + Q (A + A^2/2)
        a2 = bf(a) @ bf(a)
        y = (qd + qd @ bf(a + .5 * a2)).to(torch.bfloat16)
    else:
        y = (qd + qd @ bf(a)).to(torch.bfloat16)
    if scheme == 'production: 2 std NS (5 mm)': y = ns(ns(y, -.5), -.5)
    elif scheme == 'C2: 1 std NS (3 mm)': y = ns(y, -.5)
    elif scheme == 'pinned cubic, 1 step (3 mm)': y = ns(y, c3_pin)
    elif scheme == 'third-order NS, 1 step (4 mm)': y = ns3(y)
    elif scheme == 'exp2 seed, no NS (2 mm)': pass
    elif scheme == 'exp2 seed + 1 std NS (4 mm)': y = ns(y, -.5)
    return y

schemes = ['production: 2 std NS (5 mm)', 'C2: 1 std NS (3 mm)', 'pinned cubic, 1 step (3 mm)',
           'third-order NS, 1 step (4 mm)', 'exp2 seed, no NS (2 mm)', 'exp2 seed + 1 std NS (4 mm)']
g0 = torch.Generator().manual_seed(7)
q_exact = torch.linalg.qr(torch.randn(4, 512, 512, generator=g0, dtype=torch.float64)).Q
print('BF16 floor (rounded exact Q):', [round(v, 5) for v in errs(q_exact.to(torch.bfloat16))])
for cap in (0.25, 0.5):
    for shape in ('spread', 'decay', 'rank8', 'rank1'):
        g = torch.Generator().manual_seed(2)
        q0 = q_exact.to(torch.bfloat16)
        As = [skew(4, 512, cap, shape, g) for _ in range(4)]
        row = dict(cap=cap, shape=shape, theta_max=round(max(torch.linalg.matrix_norm(a, ord=2).max().item() for a in As), 3))
        for s in schemes:
            q = q0; tr = []
            for a in As:
                q = retract(q, a, s); tr.append(errs(q))
            row[s] = [round(tr[-1][0], 5), round(max(t[1] for t in tr), 5)]
        print(json.dumps(row))
