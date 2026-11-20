"""Moderator checks of GPT-6.2-astra's P1 identities (CPU, float64 unless stated).

(1) B1 'ordered QR tangent': for B = D + E (D sorted descending, E small off-diagonal),
    qr(B).Q ~= I + K with K_ij = E_ij / d_j for i > j, K skew (positive-diagonal-R sign convention).
    Compare with the inverse-gap (Jacobi) generator E_ij/(d_j - d_i) and the production half step.
(2) A4: one NS correction maps E = Y^T Y - I to E' = -3/4 E^2 + 1/4 E^3; worst case at E = 0.25 -> -0.04297.
(3) Real skew K: ||K||_2 <= ||K||_F / sqrt(2).
(4) A5 transport reassociation FLOP counts (pure arithmetic, printed).
(5) Pitfall: BF16 covariance EMA can stick: an entry stops moving when 0.05 |P - C| < ~half an ulp.
    Simulate a slowly drifting target with small noise; compare BF16 round-to-nearest EMA with FP32 EMA.
"""
import math

import torch

torch.manual_seed(0)
dt = torch.float64

print('(1) ordered QR tangent')
for n, eps in [(8, 1e-3), (64, 1e-3), (64, 1e-2)]:
    d = torch.sort(torch.rand(n, dtype=dt) + .05, descending=True).values
    e = torch.randn(n, n, dtype=dt) * eps
    e = (e + e.T) / 2
    e.fill_diagonal_(0)
    b = torch.diag(d) + e
    q, r = torch.linalg.qr(b)
    q = q * torch.sign(torch.diagonal(r))          # positive-diagonal R
    k_qr = q - torch.eye(n, dtype=dt)
    lower = torch.tril(torch.ones(n, n, dtype=dt), -1).bool()
    k_pred = torch.zeros(n, n, dtype=dt)
    k_pred[lower] = (e / d[None, :])[lower]
    k_pred = k_pred - k_pred.T
    gap = d[None, :] - d[:, None]                   # d_j - d_i
    k_jac = torch.where(gap != 0, e / torch.where(gap == 0, torch.ones_like(gap), gap), torch.zeros_like(gap))
    k_jac = (k_jac - k_jac.T) / 2
    rel = lambda a: ((a - k_qr).norm() / k_qr.norm()).item()
    print(f'  n={n} eps={eps}: rel err of E_ij/d_j formula={rel(k_pred):.3e}; '
          f'of full Jacobi E/(d_j-d_i)={rel(k_jac):.3e}; of half Jacobi={rel(.5 * k_jac):.3e}')

print('(2) one Newton-Schulz correction')
for e0 in (.25, .09, .01):
    e1 = -.75 * e0 ** 2 + .25 * e0 ** 3
    e2 = -.75 * e1 ** 2 + .25 * e1 ** 3
    # direct check with a matrix
    k = torch.zeros(4, 4, dtype=dt); k[0, 1], k[1, 0] = math.sqrt(e0), -math.sqrt(e0)
    y = torch.eye(4, dtype=dt) + k
    y1 = y - .5 * y @ (y.T @ y - torch.eye(4, dtype=dt))
    y2 = y1 - .5 * y1 @ (y1.T @ y1 - torch.eye(4, dtype=dt))
    print(f'  e0={e0}: formula e1={e1:.5f} e2={e2:.2e}; matrix eig extremes after 1/2 corrections: '
          f'{torch.linalg.eigvalsh(y1.T @ y1 - torch.eye(4, dtype=dt)).min().item():.5f} / '
          f'{torch.linalg.eigvalsh(y2.T @ y2 - torch.eye(4, dtype=dt)).min().item():.2e}')

print('(3) skew spectral vs Frobenius/sqrt2')
worst = 0.
for _ in range(200):
    a = torch.randn(32, 32, dtype=dt); a = a - a.T
    worst = max(worst, (torch.linalg.matrix_norm(a, 2) / (a.norm() / math.sqrt(2))).item())
print(f'  max ||K||_2 / (||K||_F/sqrt2) over 200 random skews = {worst:.4f} (must be <= 1)')

print('(4) transport FLOPs (MACs) for the 109M shapes')
for m, n in [(2304, 768), (768, 768), (3072, 768), (768, 1536)]:
    orig = 2 * m * m * n + 2 * m * n * n
    if m > n:
        new = 2 * m * m * n + m * n * n + n ** 3
    elif n > m:
        new = m ** 3 + m * m * n + 2 * m * n * n
    else:
        new = orig
    print(f'  {m}x{n}: original {orig/1e9:.3f} GMAC, short-side reassociation {new/1e9:.3f} GMAC '
          f'({100*(orig-new)/orig:.1f}% fewer)')

print('(5) BF16 covariance EMA sticking')
torch.manual_seed(1)
n, steps, beta = 64, 400, .95
for noise in (0.0, .02, .2):
    u, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt))
    lam = torch.linspace(1, .01, n, dtype=dt)
    target0 = u @ torch.diag(lam) @ u.T
    drift = torch.randn(n, n, dtype=dt); drift = (drift - drift.T) * (.002 / math.sqrt(n))
    c32 = target0.clone()
    c16 = target0.float().to(torch.bfloat16)
    rot = torch.eye(n, dtype=dt)
    for t in range(steps):
        rot = rot @ torch.linalg.matrix_exp(drift)
        target = rot @ target0 @ rot.T
        sample = target + noise * (lambda z: (z + z.T) / 2)(torch.randn(n, n, dtype=dt))
        c32 = beta * c32 + (1 - beta) * sample
        c16 = (beta * c16.float() + (1 - beta) * sample.float()).to(torch.bfloat16)   # FP32 math, BF16 store
    err16 = ((c16.double() - c32).norm() / c32.norm()).item()
    stuck = ((c16.double() - target0.float().to(torch.bfloat16).double()).abs() == 0).double().mean().item()
    lag32 = ((c32 - target).norm() / target.norm()).item()
    print(f'  noise={noise}: ||C_bf16 - C_fp32||/||C_fp32|| = {err16:.4f}; FP32 EMA lag to current target = {lag32:.4f}; '
          f'fraction of BF16 entries never moved from their initial value = {stuck:.3f}')
