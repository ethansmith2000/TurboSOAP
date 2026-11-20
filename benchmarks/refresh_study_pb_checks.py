"""Follow-up checks for Part 2b (CPU, IEEE FP32/FP64; no GPU).

(a) Positive control: the panel's chained open-loop toy (checks/generator_compare.py
    ``chained``: n=128, lambda~1/i, skew drift 0.01 sqrt2/step, 32 samples/step,
    beta_C=.95, refresh every 10, QR at 50s; mean r_C and rho_rms over steps 100-1000)
    with this study's ``warm_generic`` (BF16 storage/products, 2 NS) for production
    and PB1 under no cap, Frobenius caps 0.5/1.0 and spectral cap 0.5.
(b) Where PB1's ||A||_F comes from on trained snapshots: basis = BF16 eigenbasis of
    C at age 740, applied to C at age 750; split ||A||_F^2 into head-head / head-tail /
    tail-tail pairs (head: B_jj >= 1e-2 max B_jj); compare with rho_rms sqrt(n_eff).
(c) Unshifted Cholesky-QR and shifted CholQR3 at IEEE FP32 (CPU, no TF32) on the
    trained age-10 snapshots with the production eigh initialization at age 0.
Writes reports/refresh_study_20261120/replay_pb_checks.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'research/reviews/panel_20261120/checks')]
from bf16_soap import warm_basis  # noqa: E402
from refresh_study_common import eigh_basis, cholqr_basis  # noqa: E402
from refresh_study_replay_pb import warm_generic  # noqa: E402
import refresh_study_microbench_ext as ext  # noqa: E402
import generator_compare as gc  # noqa: E402  (panel reference; read-only import)

BF16 = torch.bfloat16
DATA = Path('/workspace/data/turbosoap_refresh_study_20261120')
torch.set_num_threads(16)


def toy_control(n=128, steps=1000, seed=0):
    dt = torch.float64
    g = torch.Generator().manual_seed(seed)
    lam = torch.arange(1, n + 1, dtype=dt).pow(-1.0)
    s = (lam / lam.sum()).sqrt()
    u0, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt, generator=g))
    w = torch.randn(n, n, dtype=dt, generator=g)
    w = (w - w.T) / 2
    w = w * (.01 * math.sqrt(2) / w.norm())
    step_rot = torch.linalg.matrix_exp(w)
    u = u0.clone()
    c = torch.zeros(n, n, dtype=dt)
    cs = []
    for _ in range(steps + 1):
        gg = u @ (s[:, None] * torch.randn(n, 32, dtype=dt, generator=g))
        c = .95 * c + .05 * gg @ gg.T
        cs.append(c.clone())
        u = step_rot @ u
    variants = {'prod_warm_basis': None, 'pb1_nocap': dict(gen='pb1', rho=1e-4, cap=None),
                'pb1_frob0.5': dict(gen='pb1', rho=1e-4, cap=.5), 'pb1_frob1.0': dict(gen='pb1', rho=1e-4, cap=1.),
                'pb1_spec0.5': dict(gen='pb1', rho=1e-4, cap=.5, cap_kind='spectral')}
    res = {}
    for name in ['qr10', 'qr50'] + list(variants):
        q = torch.linalg.eigh(cs[0]).eigenvectors.flip(-1)
        rcs, rhos, mults = [], [], []
        for t in range(1, steps + 1):
            ct = cs[t]
            if t % 10 == 0:
                if name == 'qr10' or t % 50 == 0:
                    q = gc.qr_ref(ct, q)
                elif name != 'qr50':
                    cb, qb = ct.float().to(BF16)[None], q.float().to(BF16)[None]
                    if name == 'prod_warm_basis':
                        q = warm_basis(cb, qb, BF16, BF16)[0].double()
                    else:
                        st = {}
                        q = warm_generic(cb, qb, stats=st, **variants[name])[0].double()
                        if 'cap_mult' in st:
                            mults.append(st['cap_mult'][0])
            if t >= 100:
                rc, rr = gc.metrics(ct, q)
                rcs.append(rc)
                rhos.append(rr)
        res[name] = dict(mean_rC=sum(rcs) / len(rcs), mean_rho_rms=sum(rhos) / len(rhos),
                         mean_cap_mult=(sum(mults) / len(mults) if mults else None))
        print('toy', name, res[name], flush=True)
    res['panel_reference'] = dict(qr10=(0.0759, 0.2404), qr50=(0.1343, 0.3528), prod_bf16=(0.0977, 0.2938),
                                  qrtan_exact_polar_nocap=(0.0767, 0.2427))
    return res


def load(d, idx, slot):
    a = np.load(DATA / f'd{d}' / f'f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(BF16)


def a_energy(widths=(256, 1024)):
    out = []
    for d in widths:
        meta = json.loads((DATA / f'd{d}' / 'meta.json').read_text())
        for f in meta['factors']:
            if f['dim'] > 4096:
                continue
            c_old, c = load(d, f['index'], 74), load(d, f['index'], 75)
            q = eigh_basis(c_old[None])[0].double()
            b = q.T @ c.double() @ q
            b = .5 * (b + b.T)
            dg = torch.diagonal(b)
            s = dg.abs().max()
            sg = torch.sign(dg[None, :] - dg[:, None])
            idx = torch.arange(len(dg))
            sg = torch.where(sg == 0, torch.sign(idx[:, None] - idx[None, :]).double(), sg)
            den = torch.maximum(torch.maximum(dg[:, None], dg[None, :]), 1e-4 * s * torch.ones_like(b))
            a = b * sg / den
            a.fill_diagonal_(0)
            head = dg >= 1e-2 * s
            hh = head[:, None] & head[None, :]
            tt = (~head)[:, None] & (~head)[None, :]
            tot = float((a ** 2).sum())
            keep = dg >= 1e-4 * s
            sc = torch.where(keep, dg.clamp_min(1e-300).rsqrt(), torch.zeros_like(dg))
            rho = (b - torch.diag(dg)) * sc[:, None] * sc[None, :]
            neff = int(keep.sum())
            rho_rms = math.sqrt(float((rho ** 2).sum()) / max(neff, 1))
            fro = math.sqrt(tot)
            row = dict(width=d, param=f['param'], side=f['side'], dim=f['dim'], head_fraction=float(head.float().mean()),
                       a_frob=fro, frac_head_head=float((a ** 2)[hh].sum()) / tot,
                       frac_head_tail=float((a ** 2)[~hh & ~tt].sum()) / tot, frac_tail_tail=float((a ** 2)[tt].sum()) / tot,
                       rho_rms=rho_rms, n_eff=neff, bound_rho_rms_sqrt_neff=rho_rms * math.sqrt(neff),
                       implied_frob0p5_multiplier=min(1., .5 * math.sqrt(2) / fro))
            out.append(row)
            print('A', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out


def chol_fp32(widths=(256, 512)):
    out = []
    for d in widths:
        meta = json.loads((DATA / f'd{d}' / 'meta.json').read_text())
        for f in meta['factors']:
            c0, c10 = load(d, f['index'], 0)[None], load(d, f['index'], 1)[None]
            q = eigh_basis(c0)
            new, _, info = cholqr_basis(c10, q)
            new3, _, info3, shift = ext.cholqr3_basis(c10, q)
            ev = torch.linalg.eigvalsh(c10[0].double())
            row = dict(width=d, param=f['param'], side=f['side'], dim=f['dim'],
                       cholqr_info=int(info.flatten()[0]), cholqr_finite=bool(torch.isfinite(new.float()).all()),
                       cholqr3_info_by_pass=info3.flatten().tolist(), cholqr3_finite=bool(torch.isfinite(new3.float()).all()),
                       c_min_eig_over_max=float(ev[0] / ev[-1]), c_negative_eig_fraction=float((ev < 0).double().mean()))
            out.append(row)
            print('chol', row, flush=True)
    return out


def main():
    result = dict(description=__doc__, toy_control=toy_control(), a_energy=a_energy(), cholesky_ieee_fp32=chol_fp32())
    p = ROOT / 'reports/refresh_study_20261120/replay_pb_checks.json'
    p.write_text(json.dumps(result, indent=1, allow_nan=True))
    print('wrote', p)


if __name__ == '__main__':
    main()
