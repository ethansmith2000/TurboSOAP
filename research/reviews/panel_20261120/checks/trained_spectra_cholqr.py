"""Moderator check (CPU, read-only on the refresh-study snapshots): how near-singular / indefinite are the trained
BF16 covariance factors, and which Cholesky-QR variant reproduces Householder QR on them?

For widths d in {256, 512} and ages {500, 1000}, every factor:
  spectrum of sym(float(C_bf16)): lambda_min/lambda_max, fraction of eigenvalues < 0, < 1e-6, < 1e-4 of lambda_max.
Then one QR-type refresh from Q = eigenbasis of the previous snapshot (age - 10), Z = C Q sorted by Rayleigh quotient:
  hh       Householder QR (production qr_basis arithmetic, FP32)
  chol2    unshifted CholeskyQR2 (FP32)
  sgram    Gram-shifted CholeskyQR3 with a SMALL shift s = 1e-6 * ||Z||_2^2 (does not change the target Q)
  fukaya   Gram-shifted CholeskyQR3 with Fukaya's s = 11 (n^2 + n(n+1)) u ||Z||_2^2 (as in the Part 1b script)
  ridge(t) Z = (C + t*dmax*I) Q, then sgram  (changes the target), t in {1e-6, 1e-5, 1e-4}
Reported per variant (compact: topk_dist/col_dist/gram, or FAIL): failure (non-finite / cholesky info), BF16 Gram error, and agreement with hh on the top
k = n/8 columns (subspace distance 1 - ||U_k^T V_k||_F^2/k) and on all columns (mean 1 - |<q_i, hh_i>|).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path('/workspace/data/turbosoap_refresh_study_20261120')
U = 2.0 ** -24
f32 = torch.float32


def load(width, idx, slot):
    a = np.load(ROOT / f'd{width}' / f'f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(torch.bfloat16).float()


def gram_err(q):
    qb = q.to(torch.bfloat16).double()
    return ((qb.T @ qb - torch.eye(len(qb), dtype=torch.float64)).norm() / math.sqrt(len(qb))).item()


def tf32(x):
    xi = x.contiguous().view(torch.int32)
    xi = (xi + (((xi >> 13) & 1) + 0xFFF)) & ~0x1FFF
    return xi.view(f32)


def chol_pass(x, shift=0.):
    xx = tf32(x) if TF32 else x
    g = xx.T @ xx
    if shift:
        g = g + shift * torch.eye(len(g))
    L, info = torch.linalg.cholesky_ex(g)
    if info != 0:
        return None
    return torch.linalg.solve_triangular(L.T, x, upper=True, left=False)


def norm2sq(x):
    v = torch.ones(x.shape[1], 1) / math.sqrt(x.shape[1])
    g = x.T @ x
    for _ in range(20):
        v = g @ v
        v = v / v.norm()
    return (v.T @ g @ v).item()


def cholqr(x, shift_rel=None, fukaya=False, passes=2):
    n = x.shape[1]
    if fukaya:
        s = 11 * (n * n + n * (n + 1)) * U * norm2sq(x)
    elif shift_rel:
        s = shift_rel * norm2sq(x)
    else:
        s = 0.
    y = chol_pass(x, s)
    for _ in range(passes - 1 + (1 if s else 0)):
        if y is None or not torch.isfinite(y).all():
            return None
        y = chol_pass(y)
    if y is None or not torch.isfinite(y).all():
        return None
    return y


def agree(qn, hh):
    if qn is None:
        return 'FAIL'
    n = hh.shape[1]
    k = max(8, n // 8)
    sub = 1 - ((hh[:, :k].T @ qn[:, :k]).norm() ** 2 / k).item()
    col = (1 - (qn * hh).sum(0).abs()).mean().item()
    return f'{max(sub,0):.0e}/{max(col,0):.0e}/{gram_err(qn):.4f}'


widths = [int(w) for w in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['256', '512'])]
AGES = [int(a) for a in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['500', '1000'])]
TF32 = len(sys.argv) > 3 and sys.argv[3] == 'tf32'
for width in widths:
    meta = json.loads((ROOT / f'd{width}' / 'meta.json').read_text())
    for age in AGES:
        slot = meta['snapshot_ages'].index(age)
        for f in meta['factors']:
            c = load(width, f['index'], slot)
            c = .5 * (c + c.T)
            ev = torch.linalg.eigvalsh(c.double())
            lmax = ev[-1].item()
            stats = (f"lmin/lmax={ev[0].item()/lmax:+.1e} neg={(ev < 0).double().mean().item():.3f} "
                     f"<1e-6={(ev < 1e-6*lmax).double().mean().item():.3f} <1e-4={(ev < 1e-4*lmax).double().mean().item():.3f}")
            cp = load(width, f['index'], slot - 1)
            q = torch.linalg.eigh(.5 * (cp + cp.T)).eigenvectors.flip(-1)
            q = q.to(torch.bfloat16).float()
            z = c @ q
            order = (q * z).sum(0).argsort(descending=True)
            x = z[:, order]
            hh, r = torch.linalg.qr(x)
            hh = hh * torch.sign(torch.diagonal(r))
            res = dict(chol2=agree(cholqr(x), hh), sgram=agree(cholqr(x, 1e-6), hh),
                       fukaya=agree(cholqr(x, fukaya=True), hh))
            for t in (1e-6, 1e-5, 1e-4):
                dmax = (q * z).sum(0).max().item()
                xr = ((c + t * dmax * torch.eye(len(c))) @ q)[:, order]
                res[f'ridge{t:g}'] = agree(cholqr(xr, 1e-6), hh)
            print(f"d={width} age={age} f{f['index']} {f['param'].split('.')[-2]}.{f['side']} n={f['dim']} | {stats} | "
                  + ' '.join(f'{k}={v}' for k, v in res.items()), flush=True)
