"""Moderator check requested in P2 (Fable section 6; Neon section 6.2): on TRAINED covariances, how do the
Frobenius, Schatten-4 (GPT-6.2-astra's P2 bound) and spectral norms of the warm generator compare, for the
production generator and for the QR tangent (PB1, rho = 1e-4 and 1e-3, entries clamped to +-1)?

Input per factor: C_t = saved BF16 snapshot at age t, Q = eigenbasis of the snapshot at age t - 10 (an exact
refresh 10 ages earlier, i.e. the eigh10 chain's pre-refresh state), stored in BF16.  B = Q^T C_t Q in float64.
Reported: ||A||_F/sqrt2, (sum theta^4)^(1/4) = (0.5 ||A^T A||_F^2)^(1/4), ||A||_2, and their ratios, plus the
fraction of PB1 entries hitting the +-1 clamp.  CPU, read-only on the refresh-study snapshots.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path('/workspace/data/turbosoap_refresh_study_20261120')
dt = torch.float64


def load(width, idx, slot):
    a = np.load(ROOT / f'd{width}' / f'f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(torch.bfloat16).double()


def gen_prod(b):
    d = torch.diagonal(b); s = d.abs().max()
    h = (d[:, None] - d[None, :]) / s
    return -.5 * (b / s) * h / (h ** 2 + 1e-4)


def gen_pb1(b, rho):
    d = torch.diagonal(b); s = d.abs().max(); n = len(d)
    sg = torch.sign(d[None, :] - d[:, None])
    idx = torch.arange(n)
    sg = torch.where(sg == 0, torch.sign(idx[:, None] - idx[None, :]).to(dt), sg)
    den = torch.maximum(torch.maximum(d[:, None], d[None, :]), rho * s * torch.ones_like(b))
    a = b * sg / den
    a.fill_diagonal_(0)
    clamped = (a.abs() > 1).double().mean().item()
    return a.clamp(-1, 1), clamped


def norms(a):
    fro = (a.norm() / math.sqrt(2)).item()
    s4 = (0.5 * ((a.T @ a).norm() ** 2)).item() ** .25
    sp = torch.linalg.matrix_norm(a, 2).item()
    return fro, s4, sp


widths = [int(w) for w in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['256', '512'])]
ages = [int(a) for a in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['20', '500', '1000'])]
for width in widths:
    meta = json.loads((ROOT / f'd{width}' / 'meta.json').read_text())
    for age in ages:
        slot = meta['snapshot_ages'].index(age)
        for f in meta['factors']:
            c = load(width, f['index'], slot); c = .5 * (c + c.T)
            cp = load(width, f['index'], slot - 1); cp = .5 * (cp + cp.T)
            q = torch.linalg.eigh(cp).eigenvectors.flip(-1).float().to(torch.bfloat16).double()
            b = q.T @ c @ q; b = .5 * (b + b.T)
            off = b - torch.diag(torch.diagonal(b))
            rc = (off.norm() / b.norm()).item()
            out = []
            fro, s4, sp = norms(gen_prod(b))
            out.append(f'prod F={fro:.2f} S4={s4:.2f} 2={sp:.2f}')
            for rho in (1e-4, 1e-3):
                a, cl = gen_pb1(b, rho)
                fro, s4, sp = norms(a)
                out.append(f'pb1(rho={rho:g}) F={fro:.1f} S4={s4:.2f} 2={sp:.2f} F/2={fro/sp:.1f} clamp={cl:.3f}')
            print(f"d={width} age={age} f{f['index']} {f['param'].split('.')[-2]}.{f['side']} n={f['dim']} rC={rc:.2f} | "
                  + ' | '.join(out), flush=True)
