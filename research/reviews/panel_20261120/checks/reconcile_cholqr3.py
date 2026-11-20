"""Reconcile the panel's C1 claim (small Gram shift: 1/48 failures) with refresh-study's replay_pb_checks.json
(Fukaya-shifted CholQR3: passes 2-3 fail on 11/16 age-10 trained factors).  CPU, IEEE FP32 (CPU matmul has no TF32),
read-only on the saved snapshots.

Matches refresh-study's setup: C = BF16 snapshot at the given age (NOT symmetrized, as production qr_basis),
Q = production eigh initialization at the previous snapshot (for age 10: eigh(sym(C_0)) -> BF16, exactly the
init path), Z = float(C) float(Q), Rayleigh sort, X = Z[:, pi].  ||X||_2^2 by 20 power iterations on X^T X.
Failure = any Cholesky pass with info != 0 or a non-finite result (same criterion as theirs: info by pass).
Accuracy vs a float64 Householder QR of the same X: max column angle after sign fixing (their metric) and the
top-n/8 subspace distance.

Variants: shift s = c * ||X||_2^2 on pass 1 with c in {Fukaya 11(n^2+n(n+1))u, 1e-4, 1e-5, 1e-6, 1e-7}, then two
unshifted passes ('3p'); plus 'c=1e-6, 2p+NS' = shifted pass + one unshifted pass + one FP32 Newton-Schulz step.
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


def load(width, idx, slot):
    a = np.load(ROOT / f'd{width}' / f'f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(torch.bfloat16)


def run(x, shift, unshifted, ns=0):
    n = x.shape[1]
    infos = []
    g = x.T @ x
    y = x
    for k in range(1 + unshifted):
        if k:
            g = y.T @ y
        L, info = torch.linalg.cholesky_ex(g + (shift * torch.eye(n) if k == 0 else 0))
        infos.append(int(info))
        if info != 0:
            return None, infos
        y = torch.linalg.solve_triangular(L.T, y, upper=True, left=False)
    for _ in range(ns):
        y = y - .5 * y @ (y.T @ y - torch.eye(n))
    if not torch.isfinite(y).all():
        return None, infos
    return y, infos


def accuracy(y, ref):
    n = ref.shape[1]
    cos = (y.double() * ref).sum(0).abs().clamp(max=1)
    ang = torch.acos(cos).max().item()
    k = max(8, n // 8)
    sub = 1 - ((ref[:, :k].T @ y[:, :k].double()).norm() ** 2 / k).item()
    yb = y.to(torch.bfloat16).double()
    gram = ((yb.T @ yb - torch.eye(n, dtype=torch.float64)).norm() / math.sqrt(n)).item()
    return ang, max(sub, 0.), gram


widths = [int(w) for w in sys.argv[1].split(',')] if len(sys.argv) > 1 else [256, 512]
ages = [int(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [10]
variants = [('fukaya', None, 2, 0), ('1e-4', 1e-4, 2, 0), ('1e-5', 1e-5, 2, 0), ('1e-6', 1e-6, 2, 0),
            ('1e-7', 1e-7, 2, 0), ('1e-6,2p+NS', 1e-6, 1, 1)]
tally = {v[0]: [0, 0] for v in variants}
worst = {v[0]: [0., 0.] for v in variants}
for width in widths:
    meta = json.loads((ROOT / f'd{width}' / 'meta.json').read_text())
    for age in ages:
        slot = meta['snapshot_ages'].index(age)
        for f in meta['factors']:
            c = load(width, f['index'], slot).float()
            cp = load(width, f['index'], slot - 1).float()
            q = torch.linalg.eigh(.5 * (cp + cp.T)).eigenvectors.flip(-1).to(torch.bfloat16).float()
            z = c @ q
            order = (q * z).sum(0).argsort(descending=True)
            x = z[:, order]
            n = x.shape[1]
            hh, r = torch.linalg.qr(x.double())
            hh = hh * torch.sign(torch.diagonal(r))
            g = x.T @ x
            v = torch.ones(n, 1) / math.sqrt(n)
            for _ in range(20):
                v = g @ v; v = v / v.norm()
            lam = (v.T @ g @ v).item()
            ev = torch.linalg.eigvalsh(.5 * (c + c.T).double())
            row = []
            for name, cr, unsh, ns in variants:
                s = (11 * (n * n + n * (n + 1)) * U if cr is None else cr) * lam
                y, infos = run(x, s, unsh, ns)
                tally[name][1] += 1
                if y is None:
                    tally[name][0] += 1
                    row.append(f'{name}=FAIL{infos}')
                else:
                    ang, sub, gram = accuracy(y, hh)
                    worst[name][0] = max(worst[name][0], ang); worst[name][1] = max(worst[name][1], sub)
                    row.append(f'{name}={ang:.1e}/{sub:.0e}/{gram:.4f}')
            print(f"d={width} age={age} f{f['index']} {f['param'].split('.')[-2]}.{f['side']} n={n} "
                  f"lmin/lmax={ev[0].item() / ev[-1].item():+.1e} neg={(ev < 0).double().mean().item():.2f} | "
                  + ' '.join(row), flush=True)
print('SUMMARY (failures / factors; worst max-column-angle; worst top-n/8 subspace distance), BF16 floor ~1.8e-3 rad:')
for name, (fl, tot) in tally.items():
    print(f'  {name:12s} {fl}/{tot}  angle<={worst[name][0]:.1e}  sub<={worst[name][1]:.0e}')
