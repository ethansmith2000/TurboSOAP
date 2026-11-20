"""Part 2b of the refresh study: alternative warm-step generators on the saved
trained covariance trajectories (open-loop replay, quality), plus a timing mode.

Generators (all followed by Q+ = Q + Q A in BF16 and Newton--Schulz corrections
exactly as production ``warm_basis``):
  prod      production: A = -1/2 (B/s) h/(h^2+delta^2), h_ij = (d_i-d_j)/s, s = max|d|
  pb1       QR-tangent (panel PB1, checks/generator_compare.py ``gen_qrtan``):
            A_ij = B_ij sgn(d_j - d_i) / max(d_i, d_j, rho s), ties by column index
            (sign(i - j)), entries clamped to +-1
  relgap    production formula with the pair's own scale: m_ij = max(|d_i|,|d_j|,rho s),
            h_ij = (d_i-d_j)/m_ij, A = -1/2 (B/m) h/(h^2+delta^2)
  jac1      first-order Jacobi at factor 1 (production formula with -1 instead of -1/2)
Caps: 'frob' (production: ||A||_F/sqrt2 <= cap) or 'spectral' (||A||_2 <= cap,
estimated by 8 power iterations on A^T A from a fixed start vector).

With gen='prod', cap_kind='frob', cap=.5, damping=.01, ns=2 the function executes
production's operation sequence; a replay chain checks bit equality against
``bf16_soap.warm_basis`` at every age.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from bf16_soap import warm_basis, qr_basis, mm32  # noqa: E402
from refresh_study_common import eigh_basis, highest  # noqa: E402
import refresh_study_replay as rp  # noqa: E402

BF16, FP32 = torch.bfloat16, torch.float32

VARIANTS = {
    'prod': dict(gen='prod'),
    'pb1_r1e-4_c0.5': dict(gen='pb1', rho=1e-4, cap=.5),
    'pb1_r1e-4_c1.0': dict(gen='pb1', rho=1e-4, cap=1.0),
    'pb1_r1e-3_c0.5': dict(gen='pb1', rho=1e-3, cap=.5),
    'pb1_r1e-3_c1.0': dict(gen='pb1', rho=1e-3, cap=1.0),
    'pb1_r1e-4_spec0.5': dict(gen='pb1', rho=1e-4, cap=.5, cap_kind='spectral'),
    'relgap_r1e-4_c0.5': dict(gen='relgap', rho=1e-4, cap=.5),
    'jac1_c0.5': dict(gen='jac1', cap=.5),
    'pb1_r1e-4_c0.5_ns1': dict(gen='pb1', rho=1e-4, cap=.5, ns_steps=1),
}
SCHEDULES = ['warm10_qr50', 'warm10']


def warm_generic(cov, q, *, gen='prod', damping=.01, cap=.5, cap_kind='frob', rho=1e-4, ns_steps=2, clamp=1.,
                 stats=None):
    compute_dtype, basis_dtype = BF16, BF16
    qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
    b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
    b = .5 * (b + b.transpose(-1, -2))
    diagonal = b.diagonal(dim1=-2, dim2=-1)
    scale = diagonal.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if gen in ('prod', 'jac1'):
        factor = .5 if gen == 'prod' else 1.
        gap = (diagonal.unsqueeze(-1) - diagonal.unsqueeze(-2)) / scale.unsqueeze(-1)
        a = -factor * (b / scale.unsqueeze(-1)) * gap / (gap.square() + damping ** 2)
    elif gen == 'relgap':
        m = torch.maximum(torch.maximum(diagonal.abs().unsqueeze(-1), diagonal.abs().unsqueeze(-2)),
                          rho * scale.unsqueeze(-1))
        gap = (diagonal.unsqueeze(-1) - diagonal.unsqueeze(-2)) / m
        a = -.5 * (b / m) * gap / (gap.square() + damping ** 2)
    elif gen == 'pb1':
        n = b.shape[-1]
        di, dj = diagonal.unsqueeze(-1), diagonal.unsqueeze(-2)
        sg = torch.sign(dj - di)
        idx = torch.arange(n, device=b.device)
        tiebreak = torch.sign(idx.unsqueeze(-1) - idx.unsqueeze(-2)).to(b.dtype)
        sg = torch.where(sg == 0, tiebreak, sg)
        den = torch.maximum(torch.maximum(di, dj), rho * scale.unsqueeze(-1))
        a = b * sg / den
        a = a * (1 - torch.eye(n, device=b.device, dtype=b.dtype))
        if stats is not None:
            stats['clamped_fraction'] = float((a.abs() > clamp).float().mean())
        a = a.clamp(-clamp, clamp)
    else:
        raise ValueError(gen)
    if cap is not None:
        if cap_kind == 'frob':
            bound = torch.linalg.vector_norm(a, dim=(-2, -1), keepdim=True) / math.sqrt(2)
            mult = (cap / bound.clamp_min(1e-12)).clamp(max=1)
        elif cap_kind == 'spectral':
            v = torch.ones(a.shape[0], a.shape[-1], 1, device=a.device, dtype=a.dtype)
            for _ in range(8):
                v = torch.bmm(a.transpose(-1, -2), torch.bmm(a, v))
                v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
            bound = torch.bmm(a, v).norm(dim=-2, keepdim=True)
            mult = (cap / bound.clamp_min(1e-12)).clamp(max=1)
        else:
            raise ValueError(cap_kind)
        if stats is not None:
            stats['cap_binding'] = (mult.flatten() < 1).cpu().numpy().tolist()
            stats['cap_mult'] = mult.flatten().cpu().numpy().tolist()
        a = a * mult
    y = (q.float() + mm32(qb, a.to(compute_dtype))).to(basis_dtype)
    for _ in range(ns_steps):
        gram = mm32(y.transpose(-1, -2), y)
        eye = torch.eye(y.shape[-1], device=y.device, dtype=gram.dtype)
        residual = gram - eye
        y = (y.float() - .5 * mm32(y.to(compute_dtype), residual.to(compute_dtype))).to(basis_dtype)
    return y.contiguous()


def replay(args):
    rp.backend()
    torch.set_grad_enabled(False)
    width_dir = rp.DATA_ROOT / f'd{args.width}'
    meta = json.loads((width_dir / 'meta.json').read_text())
    factors = meta['factors']
    ages = meta['snapshot_ages']
    groups = {}
    for f in factors:
        groups.setdefault(f['dim'], []).append(f['index'])
    chains = [f'{v}|{s}' for v in VARIANTS for s in SCHEDULES] + ['prodcheck|warm10_qr50']
    result = dict(width=args.width, ages=ages, chains=chains, variants=VARIANTS, factors=factors, groups={},
                  open_loop=True, script_sha256=rp.sha(__file__))
    t_start = time.time()
    for dim, indices in sorted(groups.items()):
        ks = sorted({1, 8, max(8, dim // 8)})
        series = {ch: {ph: {} for ph in ('before', 'after')} for ch in chains}
        capstats = {ch: [] for ch in chains}
        exact = []
        qs = {}
        for slot, age in enumerate(ages):
            c = rp.load_group(width_dir, indices, slot, device='cpu' if args.cpu else 'cuda')
            with highest():
                evals, evecs = torch.linalg.eigh(.5 * (c.float() + c.float().transpose(-1, -2)))
            evecs = evecs.flip(-1)
            for ch in chains:
                var, sched = ch.split('|')
                if age == 0:
                    kind = 'init'
                elif sched == 'warm10_qr50':
                    kind = 'qr' if age % 50 == 0 else 'warm'
                else:
                    kind = 'warm'
                before = None
                if kind != 'init':
                    before = rp.metric_block(c, qs[ch], evecs, ks)
                    for key, v in before.items():
                        series[ch]['before'].setdefault(key, []).append(v)
                if kind == 'init':
                    qs[ch] = eigh_basis(c)
                elif kind == 'qr':
                    qs[ch] = qr_basis(c, qs[ch], BF16)[0]
                elif var == 'prodcheck':
                    qs[ch] = warm_basis(c, qs[ch], BF16, BF16)
                else:
                    st = {}
                    qs[ch] = warm_generic(c, qs[ch], stats=st, **VARIANTS[var])
                    capstats[ch].append(dict(age=age, **st))
                m = rp.metric_block(c, qs[ch], evecs, ks)
                for key, v in m.items():
                    series[ch]['after'].setdefault(key, []).append(v)
                    if age == 0:
                        series[ch]['before'].setdefault(key, []).append(np.full(len(indices), np.nan))
            exact.append(bool(torch.equal(qs['prod|warm10_qr50'], qs['prodcheck|warm10_qr50'])))
            del c, evals, evecs
            if slot % 20 == 0:
                msg = ' '.join(f"{v}={np.nanmean(series[v + '|warm10_qr50']['after']['rC'][-1]):.3f}"
                               for v in ('prod', 'pb1_r1e-4_c0.5', 'relgap_r1e-4_c0.5', 'jac1_c0.5'))
                print(f'd{args.width} dim{dim} age {age} t={time.time()-t_start:.0f}s rC {msg}', flush=True)
        result['groups'][str(dim)] = dict(
            factor_indices=indices, ks=ks,
            series={ch: {ph: {k: np.stack(v).tolist() for k, v in series[ch][ph].items()}
                         for ph in ('before', 'after')} for ch in chains},
            capstats={ch: v for ch, v in capstats.items() if v},
            prod_bit_equal_every_age=all(exact))
        del qs
        torch.cuda.empty_cache()
    result['elapsed_s'] = time.time() - t_start
    out = rp.REPORT / 'replay_pb_parts'
    out.mkdir(parents=True, exist_ok=True)
    rp.write_json(out / f'replay_pb_d{args.width}.json', result)
    print('wrote', out / f'replay_pb_d{args.width}.json', flush=True)


def timing(args):
    """Warm-variant cost on the 109M factor dims (batch 12) and n=8192 (batch 1),
    eager and compiled, with the Part 1 harness; exclusive GPU."""
    import refresh_study_microbench as mb
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.recompile_limit = 512
    torch._dynamo.config.accumulated_recompile_limit = 8192
    torch.set_grad_enabled(False)
    bench = mb.Bench(4000., 8, 40, 2)
    rows, exact = [], []
    env = mb.env_record()
    t0 = time.time()
    for n, batch in [(768, 12), (1536, 12), (2304, 12), (3072, 12), (8192, 1)]:
        c, q = mb.make_factor_inputs(n, batch, seed=5000 + n)
        prod = warm_basis(c, q, BF16, BF16)
        exact.append(dict(n=n, batch=batch, eager_bit_equal=bool(torch.equal(prod, warm_generic(c, q)))))
        for name, kw in VARIANTS.items():
            fn = (lambda kw: (lambda cc, qq: warm_generic(cc, qq, **kw)))(kw)
            for compiled in (False, True):
                st = bench.run_block([(name, fn, lambda: (c, q))], compiled)
                r = mb.row('warm_variant', n, batch, 'bf16', compiled, st[name], extra=dict(variant=name))
                rows.append(r)
                print(name, n, batch, 'C' if compiled else 'E', r['median_ms'], r.get('error'), flush=True)
        del c, q, prod
        torch.cuda.empty_cache()
    out = dict(env=env, env_end=mb.env_record(), rows=rows, bit_exact=exact, elapsed_s=time.time() - t0,
               variants=VARIANTS)
    rp.write_json(rp.REPORT / 'replay_pb_timing.json', out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replay')
    r.add_argument('--width', type=int, required=True)
    r.add_argument('--data-root', default=None)
    r.add_argument('--report-dir', default=None)
    r.add_argument('--cpu', action='store_true')
    t = sub.add_parser('time')
    t.add_argument('--report-dir', default=None)
    args = ap.parse_args()
    if getattr(args, 'data_root', None):
        rp.DATA_ROOT = Path(args.data_root)
    if getattr(args, 'report_dir', None):
        rp.REPORT = Path(args.report_dir)
    dict(replay=replay, time=timing)[args.cmd](args)


if __name__ == '__main__':
    main()
