"""Close-out (2026-11-20): how much gradient energy a low-rank (top-r) eigenbasis keeps, and how fast it goes stale.

Uses the refresh study's saved covariance snapshots (BF16, every 10 ages) of 1-layer models. For each factor C at
a target age it reports tr(top-r eigenspace of C) / tr(C) for r = n/16, n/8, n/4, the same share captured by the
top-r basis computed 10 and 50 ages earlier, and, for each weight matrix, the product of its left and right shares
(the head-head block kept by two-sided M x r / r x N preconditioners, under a Kronecker approximation).

C is the beta = 0.95 EMA of G G^T (or G^T G), so the tail includes gradient noise; energy is not importance.

Input : /workspace/data/turbosoap_refresh_study_20261120/d{width}/f*_bf16.npy (local only, not in git)
Output: reports/closeout_20261120/lowrank_energy_d{width}_age{age}.json
Usage : python benchmarks/closeout_lowrank_energy.py [width] [age]
"""
import json
import pathlib
import sys

import numpy as np
import torch

DATA = pathlib.Path('/workspace/data/turbosoap_refresh_study_20261120')
OUT = pathlib.Path(__file__).resolve().parents[1] / 'reports' / 'closeout_20261120'
FRACTIONS = (1 / 16, 1 / 8, 1 / 4)
LAGS = (10, 50)


def load(width, factor, index):
    raw = np.load(DATA / f'd{width}' / f'f{factor}_bf16.npy', mmap_mode='r')[index]
    c = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).copy()).double()
    return (c + c.T) / 2


def top_basis(c):
    w, u = torch.linalg.eigh(c)
    return w.flip(0).clamp_min(0), u.flip(1)


def main(width=1024, age=1000):
    torch.set_num_threads(8)
    meta = json.loads((DATA / f'd{width}' / 'meta.json').read_text())
    ages = meta['snapshot_ages']
    factors = []
    for fi, fm in enumerate(meta['factors']):
        n = fm['dim']
        c = load(width, fi, ages.index(age))
        w, _ = top_basis(c)
        trace = w.sum()
        old = {lag: top_basis(load(width, fi, ages.index(age - lag)))[1] for lag in LAGS}
        rows = []
        for f in FRACTIONS:
            r = int(n * f)
            row = dict(r=r, fresh=(w[:r].sum() / trace).item())
            for lag, u in old.items():
                ur = u[:, :r]
                row[f'stale{lag}'] = (torch.trace(ur.T @ c @ ur) / trace).item()
            rows.append(row)
        factors.append(dict(param=fm['param'], side=fm['side'], n=n, shares=rows))
        print(f"{fm['param']:32s} {fm['side']} n={n:5d} " + '  '.join(
            f"r={x['r']}: {x['fresh']:.2f} (stale10 {x['stale10']:.2f}, stale50 {x['stale50']:.2f})" for x in rows))
    two_sided = {}
    for i in range(0, len(factors), 2):
        left, right = factors[i], factors[i + 1]
        assert left['param'] == right['param'] and (left['side'], right['side']) == ('L', 'R')
        two_sided[left['param']] = {f'r/n={f:g}': left['shares'][k]['fresh'] * right['shares'][k]['fresh']
                                    for k, f in enumerate(FRACTIONS)}
        print(left['param'], two_sided[left['param']])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'lowrank_energy_d{width}_age{age}.json').write_text(json.dumps(
        dict(width=width, age=age, seed=meta.get('seed'), factors=factors,
             two_sided_head_head_kronecker_approx=two_sided), indent=1))


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:3]))
