"""Moderator check M2 (CPU): basis-frame covariance EMA (prior 1).

Identity: with Gbar = Q_L^T G Q_R and orthogonal Q_R, Gbar Gbar^T = Q_L^T (G G^T) Q_L.  So B_L = Q_L^T C_L Q_L can be
kept as an EMA of Gbar Gbar^T between basis changes.  Check (a) the identity error when Q_R is a BF16-rounded basis
(normalized Gram error ~ 2e-3), (b) drift over 10 EMA steps vs forming Q^T C Q directly, (c) the cost after a basis
change: B+ = R^T B R with R = Q^T Q+ (2 products) versus Q+^T C Q+ (2 products) - i.e. no saving at the warm step
unless the rotation of B is cheaper than 2 products.
"""
import torch

torch.manual_seed(0)
torch.set_num_threads(4)
dt = torch.float64


def bf16(x):
    return x.float().to(torch.bfloat16).to(dt)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


for (m, n) in [(256, 256), (512, 256), (256, 768)]:
    ql, _ = torch.linalg.qr(torch.randn(m, m, dtype=dt))
    qr_, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt))
    qlb, qrb = bf16(ql), bf16(qr_)
    gram_err = ((qrb.T @ qrb - torch.eye(n, dtype=dt)).norm() / n ** .5).item()
    c = torch.zeros(m, m, dtype=dt)
    b = torch.zeros(m, m, dtype=dt)
    beta = .95
    for t in range(10):
        g = torch.randn(m, n, dtype=dt) * torch.linspace(1, .05, n, dtype=dt)
        gbar = qlb.T @ g @ qrb
        c = beta * c + (1 - beta) * g @ g.T
        b = beta * b + (1 - beta) * gbar @ gbar.T
    direct = qlb.T @ c @ qlb
    print(f'm={m} n={n} Q_R normalized Gram err={gram_err:.4f} '
          f'rel err(B_ema vs Q^T C Q) after 10 steps={rel(b, direct):.2e} '
          f'offdiag-only rel err={rel(b - torch.diag(b.diagonal()), direct - torch.diag(direct.diagonal())):.2e}')
print('Product count per warm refresh (per factor, n x n):')
print('  current: B = Q^T C Q (2) + Q A (1) + 2 x NS (2 each = 4)  = 7')
print('  B kept in frame: Q A (1) + NS (4) + rotate B: R^T B R (2) = 7, or first-order B + BA - AB (2) = 7')
print('  => saving needs a cheaper B rotation (e.g. only diag + first-order offdiag update) or NS done on R in the')
print('     small/structured frame; per ordinary step the EMA of Gbar Gbar^T costs the same as G G^T (m x n x m).')
