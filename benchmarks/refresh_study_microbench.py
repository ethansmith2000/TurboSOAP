"""Refresh-cost microbenchmark for TurboSOAP (refresh study, 2026-11-20).

Times one basis refresh per factor, batched, for:
  warm_bf16      production ``warm_basis`` (BF16 C/Q storage, BF16 compute, BF16 basis)
  warm_fp32      ``warm_basis`` with FP32 C/Q storage and FP32 basis, BF16 compute
                 (the paper's ``compiled_fp32`` storage configuration)
  qr             production ``qr_basis`` (FP32 island incl. CQ, sort, QR, BF16 out)
  eigh           FP32 ``torch.linalg.eigh`` of sym(float(C)), descending, BF16 out
                 (exactly the initialization path of ``CompiledSOAP``)
  cholqr         Cholesky-QR comparator: Z=CQ (FP32), same sort as ``qr_basis``,
                 R=chol(Z^T Z)^T, Q+=Z R^{-1} via triangular solve, BF16 out.
                 Failures (info>0) and non-finite outputs are recorded, not tuned.
plus ``transport_moment`` (BF16) and the QR-only V permutation per matrix shape,
the compiled production event regions (``CompiledSOAP.step`` on the 109M
shapes, ordinary / warm / QR), and the warm-step ablation variants.

Backend settings match the 109M runs (benchmarks/bf16_storage_screen.py and
paper/REPRODUCIBILITY.md): deterministic algorithms with
CUBLAS_WORKSPACE_CONFIG=:4096:8, FP32 matmul precision 'high', BF16
reduced-precision reductions disabled, 4 CPU threads, compile options
fullgraph/dynamic=False/emulate_precision_casts=True.

Only production functions are imported from ``bf16_soap.py``; nothing there is
modified.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmarks')]
from bf16_soap import warm_basis, qr_basis, transport_moment, CompiledSOAP  # noqa: E402
from refresh_study_common import (warm_basis_param, eigh_basis, cholqr_basis,  # noqa: E402
                                  coordinate_matrix, residuals, gram_error, WARM_DEFAULTS)

BF16, FP32 = torch.bfloat16, torch.float32

# 109M TurboSOAP factor groups (batch 12 each): C_L/C_R dimensions per matrix shape.
PROD_SHAPES = [(2304, 768), (768, 768), (3072, 768), (768, 1536)]
PROD_FACTORS = [768] * 5 + [1536, 2304, 3072]
TARGETS = dict(ordinary_ms=10.14, warm_event_ms=61.61, qr_event_ms=601.18,
               qr_primitive_sec54_b12={768: 30.914, 1536: 80.535, 2304: 141.566, 3072: 215.218})

# One-at-a-time ablation around the production default (0.01, 0.5, 2 corrections, 1 step).
VARIANTS = [('default', {})]
VARIANTS += [(f'damping={d}', dict(damping=d)) for d in (.001, .003, .03, .1)]
VARIANTS += [(f'cap={c}', dict(cap=c)) for c in (.25, 1.0, None)]
VARIANTS += [(f'ns={k}', dict(ns_steps=k)) for k in (0, 1, 3)]
VARIANTS += [('warm_steps=2', dict(warm_steps=2))]


# ----------------------------------------------------------------------------- methods
def m_warm_bf16(c, q):
    return warm_basis(c, q, BF16, BF16)


def m_warm_fp32(c, q):
    return warm_basis(c, q, BF16, FP32)


def m_qr(c, q):
    return qr_basis(c, q, BF16)


def m_eigh(c, q):
    return eigh_basis(c)


def m_cholqr(c, q):
    return cholqr_basis(c, q)


METHODS = dict(warm_bf16=(m_warm_bf16, 'bf16'), warm_fp32=(m_warm_fp32, 'fp32'),
               qr=(m_qr, 'bf16'), eigh=(m_eigh, 'bf16'), cholqr=(m_cholqr, 'bf16'))


def m_transport(m, ol, orr, nl, nr):
    return transport_moment(m, ol, orr, nl, nr, BF16)


def m_vperm(v, lo, ro):
    v = v.gather(-2, lo.unsqueeze(-1).expand_as(v))
    return v.gather(-1, ro.unsqueeze(-2).expand_as(v))


def variant_fn(kw):
    cfg = {**WARM_DEFAULTS, **kw}

    def fn(c, q):
        return warm_basis_param(c, q, BF16, BF16, **cfg)
    return fn


# ----------------------------------------------------------------------------- inputs
@torch.no_grad()
def make_factor_inputs(n, batch, seed, alpha=1.0, steps=40, beta=.95, rot=.2):
    """Per element: SPD C = EMA_t of G_t G_t^T with G_t = U diag(lambda^1/2) X_t
    (power-law lambda_i ~ i^-alpha, X_t Gaussian n x n/4), stored in BF16; Q = QR
    (sign-fixed) of U(I+A), A skew with entries ~ rot/sqrt(n), stored in BF16."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    c_out = torch.empty(batch, n, n, device='cuda', dtype=BF16)
    q_out = torch.empty(batch, n, n, device='cuda', dtype=BF16)
    lam = torch.arange(1, n + 1, device='cuda', dtype=FP32).pow(-alpha)
    s = (lam / lam.sum()).sqrt()
    k = max(n // 4, 64)
    for i in range(batch):
        u = torch.linalg.qr(torch.randn(n, n, device='cuda', generator=g)).Q
        w = torch.zeros(n, n, device='cuda')
        for _ in range(steps):
            x = torch.randn(n, k, device='cuda', generator=g) / math.sqrt(k)
            w.mul_(beta).addmm_(x, x.T, alpha=1 - beta)
            del x
        w /= (1 - beta ** steps)
        us = u * s
        c = us @ w @ us.T
        del w, us
        c_out[i] = .5 * (c + c.T)
        del c
        a = torch.randn(n, n, device='cuda', generator=g) * (rot / math.sqrt(n))
        a = .5 * (a - a.T)
        qq, rr = torch.linalg.qr(u + u @ a)
        q_out[i] = qq * torch.sign(rr.diagonal())
        del a, qq, rr, u
    return c_out, q_out


@torch.no_grad()
def quality(c, q):
    """Per-element r_C / Gram error at FP32 'highest' (chunked to bound memory)."""
    rcs, eqs, fin = [], [], True
    for i in range(c.shape[0]):
        b = coordinate_matrix(c[i:i + 1], q[i:i + 1])
        rcs.append(float(residuals(b)[0]))
        eqs.append(float(gram_error(q[i:i + 1])))
        fin &= bool(torch.isfinite(q[i]).all())
        del b
    t = torch.tensor(rcs)
    return dict(rC_mean=float(t.nanmean()), gram_err_mean=sum(eqs) / len(eqs), gram_err_max=max(eqs),
                output_finite=fin, nonfinite_factors=sum(1 for i in range(q.shape[0]) if not torch.isfinite(q[i]).all()))


# ----------------------------------------------------------------------------- timing
def compile_fn(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})


def cuda_time(fn, args):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def quantiles(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return float(t.median()), float(t.quantile(.25)), float(t.quantile(.75))


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower()


def err_text(exc):
    return ('OOM: ' if is_oom(exc) else '') + type(exc).__name__ + ': ' + str(exc)[:400]


class Bench:
    def __init__(self, budget_ms, min_reps, max_reps, warmups):
        self.budget_ms, self.min_reps, self.max_reps, self.warmups = budget_ms, min_reps, max_reps, warmups

    def run_block(self, entries, compiled):
        """entries: list of (key, fn, args_fn). Compile/first call, warmups, then
        two interleaved passes (forward and reverse order), each preceded by one
        warmup call per entry.  Returns key -> dict(samples, info, error)."""
        state = {}
        for key, fn, args in entries:
            info = dict(compiled=compiled)
            try:
                f = compile_fn(fn) if compiled else fn
                t0 = time.perf_counter()
                out, _ = cuda_time(f, args())
                info['first_call_s'] = time.perf_counter() - t0
                del out
                est = []
                for _ in range(self.warmups):
                    out, ms = cuda_time(f, args())
                    del out
                    est.append(ms)
                reps = int(min(self.max_reps, max(self.min_reps, math.ceil(self.budget_ms / max(min(est), 1e-3)))))
                reps += reps % 2
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                out, _ = cuda_time(f, args())
                del out
                info['peak_extra_gib'] = (torch.cuda.max_memory_allocated() - base) / 2**30
                state[key] = dict(f=f, args=args, reps=reps, samples=[], info=info, error=None)
            except Exception as exc:  # compile failure / OOM / solver error
                state[key] = dict(error=err_text(exc), info=info, samples=None, trace=traceback.format_exc()[-1500:])
                torch.cuda.empty_cache()
        live = [k for k, _, _ in entries if state[k]['error'] is None]
        for order in (live, live[::-1]):
            for key in order:
                s = state[key]
                if s['error'] is not None:
                    continue
                try:
                    out, _ = cuda_time(s['f'], s['args']())
                    del out
                    for _ in range(s['reps'] // 2):
                        out, ms = cuda_time(s['f'], s['args']())
                        del out
                        s['samples'].append(ms)
                except Exception as exc:
                    s['error'] = err_text(exc)
                    torch.cuda.empty_cache()
        return state


def row(method, n, batch, storage, compiled, st, m=None, extra=None):
    r = dict(method=method, n=n, m=m if m is not None else n, batch=batch, storage_dtype=storage,
             compiled=compiled, median_ms=None, p25_ms=None, p75_ms=None, ok=False, error=st.get('error'))
    if st.get('error') is None and st.get('samples'):
        med, p25, p75 = quantiles(st['samples'])
        r.update(median_ms=med, p25_ms=p25, p75_ms=p75, ok=True, reps=len(st['samples']),
                 per_factor_median_ms=med / batch)
    r.update({k: v for k, v in st.get('info', {}).items() if k != 'compiled'})
    if extra:
        r.update(extra)
    return r


# ----------------------------------------------------------------------------- event regions
def event_regions(reps=20, warmups=3):
    """Production CompiledSOAP.step on 4 groups x 12 matrices of the 109M shapes,
    plus a fused AdamW over a 50257x768 tied-embedding stand-in (the 109M
    optimizer_ms also covered auxiliary AdamW).  Events forced via ``opt.call``.
    Ordinary, warm, QR interleaved per repetition.  Reports CUDA-event and
    synchronized host-wall medians."""
    torch.manual_seed(0)
    params = []
    for (m, n) in PROD_SHAPES:
        for _ in range(12):
            p = (torch.randn(m, n, device='cuda') * .02).contiguous()
            p.grad = torch.randn_like(p) * 1e-3
            params.append(p)
    emb = torch.randn(50257, 768, device='cuda') * .02
    emb.grad = torch.randn_like(emb) * 1e-3
    opt = CompiledSOAP(params, lr=.0015)
    aux = torch.optim.AdamW([emb], lr=6e-4, fused=True)
    opt.step()  # cold: eager eigh initialization
    ages = dict(ordinary=11, warm=10, qr=50)

    def run(event, with_aux=True):
        opt.call = ages[event]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        opt.step()
        if with_aux:
            aux.step()
        end.record()
        torch.cuda.synchronize()
        assert opt.last_event == event, (opt.last_event, event)
        return float(start.elapsed_time(end)), 1000 * (time.perf_counter() - t0)

    out = {}
    t0 = time.perf_counter()
    for event in ages:
        run(event)  # compile
    compile_s = time.perf_counter() - t0
    for _ in range(warmups):
        for event in ages:
            run(event)
    samples = {e: dict(cuda=[], wall=[], matrix_only_cuda=[]) for e in ages}
    for r in range(reps):
        order = list(ages) if r % 2 == 0 else list(ages)[::-1]
        for event in order:
            c, w = run(event)
            samples[event]['cuda'].append(c)
            samples[event]['wall'].append(w)
            c2, _ = run(event, with_aux=False)
            samples[event]['matrix_only_cuda'].append(c2)
    for event, s in samples.items():
        med, p25, p75 = quantiles(s['cuda'])
        wm, w25, w75 = quantiles(s['wall'])
        mo, _, _ = quantiles(s['matrix_only_cuda'])
        out[event] = dict(median_ms=med, p25_ms=p25, p75_ms=p75, wall_median_ms=wm, wall_p25_ms=w25,
                          wall_p75_ms=w75, matrix_only_median_ms=mo, reps=reps)
    out['compile_s_first_three_events'] = compile_s
    out['warm_minus_ordinary_ms'] = out['warm']['median_ms'] - out['ordinary']['median_ms']
    out['qr_minus_ordinary_ms'] = out['qr']['median_ms'] - out['ordinary']['median_ms']
    out['note'] = ('Synthetic random gradients; values do not affect op counts. Events forced by setting '
                   'opt.call; state evolves across repetitions. 109M reference: ordinary 10.14, warm 61.61, '
                   'QR 601.18 ms (synchronized host wall, training loop).')
    del opt, aux, params, emb
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- variants
def variant_rows(bench, modes, dims=(768, 1536, 2304, 3072), batch=12):
    rows, exact = [], []
    for n in dims:
        c, q = make_factor_inputs(n, batch, seed=5000 + n)
        prod = warm_basis(c, q, BF16, BF16)
        for name, kw in VARIANTS:
            fn = variant_fn(kw)
            out = fn(c, q)
            if name == 'default':
                exact.append(dict(n=n, batch=batch, compiled=False, bit_equal=bool(torch.equal(out, prod))))
            e = quality(c, out)
            extra = dict(variant=name, **{'out_' + k: v for k, v in e.items()})
            del out
            for compiled in modes:
                st = bench.run_block([(name, fn, lambda: (c, q))], compiled)
                rows.append(row('warm_variant', n, batch, 'bf16', compiled, st[name], extra=extra))
                print('   variant', name, n, 'C' if compiled else 'E', rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
        # compiled default vs compiled production
        try:
            cp = compile_fn(m_warm_bf16)(c, q)
            cv = compile_fn(variant_fn({}))(c, q)
            exact.append(dict(n=n, batch=batch, compiled=True, bit_equal=bool(torch.equal(cp, cv)),
                              compiled_vs_eager_production_equal=bool(torch.equal(cp, prod))))
        except Exception as exc:
            exact.append(dict(n=n, batch=batch, compiled=True, error=err_text(exc)))
        del c, q, prod
        torch.cuda.empty_cache()
    return rows, exact


# ----------------------------------------------------------------------------- driver
def loadavg():
    try:
        return Path('/proc/loadavg').read_text().split()[:3]
    except Exception:
        return None


def env_record():
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = None
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw',
                                       '--format=csv,noheader'], text=True).strip().splitlines()
    except Exception:
        smi = None
    return dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                num_threads=torch.get_num_threads(), repo_head=commit,
                bf16_soap_sha256=sha(ROOT / 'bf16_soap.py'), script_sha256=sha(__file__),
                common_sha256=sha(ROOT / 'benchmarks/refresh_study_common.py'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_claim_indices=os.environ.get('GPU_CLAIM_INDICES'), nvidia_smi=smi, loadavg=loadavg())


def reconcile(rows, events=None):
    out = {}
    for compiled in (False, True):
        get = {(r['method'], r['m'], r['n'], r['batch']): r for r in rows if r['compiled'] == compiled}

        def med(method, m, n, b=12):
            r = get.get((method, m, n, b))
            return r['median_ms'] if r and r['ok'] else None

        def total(method):
            vals = [med(method, d, d) for d in PROD_FACTORS]
            return None if any(v is None for v in vals) else sum(vals)

        transport = [med('transport_moment', m, n) for m, n in PROD_SHAPES]
        vperm = [med('v_permute', m, n) for m, n in PROD_SHAPES]
        tr = None if None in transport else sum(transport)
        vp = None if None in vperm else sum(vperm)
        block = dict(transport_sum_ms=tr, v_permute_sum_ms=vp)
        for method in METHODS:
            refresh = total(method)
            ev = None if refresh is None or tr is None else refresh + tr + (vp if method in ('qr', 'cholqr', 'eigh') else 0.)
            block[method] = dict(factor_refresh_sum_ms=refresh, event_extra_ms=ev,
                                 per_dim_ms={str(d): med(method, d, d) for d in sorted(set(PROD_FACTORS))})
        block['warm_target_ms'] = TARGETS['warm_event_ms'] - TARGETS['ordinary_ms']
        block['qr_target_ms'] = TARGETS['qr_event_ms'] - TARGETS['ordinary_ms']
        if block['warm_bf16']['event_extra_ms'] is not None:
            block['warm_ratio_to_target'] = block['warm_bf16']['event_extra_ms'] / block['warm_target_ms']
        if block['qr']['event_extra_ms'] is not None:
            block['qr_ratio_to_target'] = block['qr']['event_extra_ms'] / block['qr_target_ms']
        block['qr_primitive_vs_sec54'] = {str(d): dict(this=med('qr', d, d), sec54=v,
                                          ratio=(med('qr', d, d) / v) if med('qr', d, d) else None)
                                          for d, v in TARGETS['qr_primitive_sec54_b12'].items()}
        out['compiled' if compiled else 'eager'] = block
    if events:
        out['event_regions'] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
                                for k, v in events.items()}
        out['event_regions_vs_109M'] = dict(
            ordinary=dict(this=events['ordinary']['wall_median_ms'], target=TARGETS['ordinary_ms']),
            warm=dict(this=events['warm']['wall_median_ms'], target=TARGETS['warm_event_ms']),
            qr=dict(this=events['qr']['wall_median_ms'], target=TARGETS['qr_event_ms']))
    out['factor_groups'] = dict(factors_batch12=PROD_FACTORS, matrix_shapes_batch12=PROD_SHAPES)
    out['targets'] = {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in TARGETS.items()}
    out['note'] = ('Event extras = sum over the 8 batch-12 factor refreshes + 4 transports (+4 V permutations for '
                   'QR-type refreshes), each primitive timed in isolation; targets are 109M event means minus the '
                   'ordinary-call mean, i.e. they also absorb compiled-region fusion differences.')
    return out


def make_figure(json_path, pdf_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = json.loads(Path(json_path).read_text())
    rows = [r for r in d['rows'] if r['method'] in METHODS]
    colors = dict(warm_bf16='#2a78d6', qr='#eb6834', eigh='#1baf7a', cholqr='#eda100', warm_fp32='#e87ba4')
    labels = dict(warm_bf16='warm (BF16, production)', warm_fp32='warm (FP32 storage)', qr='QR (production)',
                  eigh='FP32 eigh', cholqr='Cholesky-QR (FP32)')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for method in METHODS:
        for compiled, ls in ((True, '-'), (False, ':')):
            pts = sorted((r['n'], r['median_ms'] / r['batch']) for r in rows
                         if r['method'] == method and r['batch'] == 12 and r['compiled'] == compiled and r['ok'])
            if pts:
                ax.plot(*zip(*pts), ls, color=colors[method], marker='o' if compiled else None, ms=3,
                        label=labels[method] if compiled else None, lw=1.6 if compiled else 1)
            bad = [r['n'] for r in rows if r['method'] == method and r['batch'] == 12 and r['compiled'] == compiled and not r['ok']]
            for n in bad:
                ax.axvline(n, color=colors[method], alpha=.15, lw=6)
                if not compiled and method == 'warm_bf16':
                    ax.annotate('eager warm OOM (batch 12)', (n, .12), fontsize=7, ha='right', color='0.3',
                                xytext=(-6, 0), textcoords='offset points')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('factor dimension n')
    ax.set_ylabel('ms per factor (batch 12 / 12)')
    ax.set_title('(a) refresh cost per factor; solid compiled, dotted eager', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.3, which='both', lw=.5)
    ax = axes[1]
    for method in ('warm_bf16', 'warm_fp32', 'eigh', 'cholqr'):
        for batch, ls in ((12, '-'), (1, '--')):
            get = {(r['n']): r['median_ms'] for r in rows if r['method'] == method and r['batch'] == batch and r['compiled'] and r['ok']}
            ref = {(r['n']): r['median_ms'] for r in rows if r['method'] == 'qr' and r['batch'] == batch and r['compiled'] and r['ok']}
            pts = sorted((n, get[n] / ref[n]) for n in get if n in ref)
            if pts:
                ax.plot(*zip(*pts), ls, color=colors[method], marker='o' if batch == 12 else None, ms=3,
                        label=f'{labels[method]} / QR' if batch == 12 else None)
    ax.axhline(1, color='k', lw=.6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('factor dimension n')
    ax.set_ylabel('time ratio to production QR')
    ax.set_title('(b) ratio to QR, compiled; solid batch 12, dashed batch 1', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.3, which='both', lw=.5)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(str(pdf_path).replace('.pdf', '.png'), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', default='256,512,768,1024,1536,2048,2304,3072,4096,6144,8192')
    ap.add_argument('--batches', default='1,12')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--modes', default='eager,compiled')
    ap.add_argument('--budget-ms', type=float, default=4000.)
    ap.add_argument('--min-reps', type=int, default=8)
    ap.add_argument('--max-reps', type=int, default=40)
    ap.add_argument('--warmups', type=int, default=2)
    ap.add_argument('--event-reps', type=int, default=20)
    ap.add_argument('--skip-events', action='store_true')
    ap.add_argument('--skip-variants', action='store_true')
    ap.add_argument('--variant-dims', default='768,1536,2304,3072')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'reports/refresh_study_20261120/microbench.json'))
    args = ap.parse_args()
    out_path = Path(args.out)
    fig_path = out_path.parent / 'refresh_cost_vs_dim.pdf'
    if args.plot_only:
        make_figure(out_path, fig_path)
        return

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.recompile_limit = 512
    torch._dynamo.config.accumulated_recompile_limit = 8192
    torch.set_grad_enabled(False)

    ns = [int(x) for x in args.ns.split(',')]
    batches = [int(x) for x in args.batches.split(',')]
    methods = args.methods.split(',')
    modes = [m == 'compiled' for m in args.modes.split(',')]
    bench = Bench(args.budget_ms, args.min_reps, args.max_reps, args.warmups)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, inputs_meta, events, variants, exact = [], [], None, [], []
    env_start = env_record()
    t_start = time.time()

    def dump(final=False):
        payload = dict(env=env_start, protocol=vars(args), inputs=inputs_meta, rows=rows, events=events,
                       variant_rows=variants, variant_bit_exact=exact, elapsed_s=time.time() - t_start,
                       complete=final)
        if final:
            payload['env_end'] = env_record()
            payload['reconciliation'] = reconcile(rows, events)
        tmp = out_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=True))
        tmp.replace(out_path)

    if not args.skip_events:
        try:
            events = event_regions(reps=args.event_reps)
        except Exception as exc:
            events = dict(error=err_text(exc), trace=traceback.format_exc()[-2000:])
        print('event regions', json.dumps(events), flush=True)
        dump()

    shapes_done = set()
    for batch in batches:
        for n in ns:
            print(f'== n={n} batch={batch} t={time.time()-t_start:.0f}s load={loadavg()}', flush=True)
            c = q = None
            try:
                c, q = make_factor_inputs(n, batch, seed=1000 + n + 7 * batch)
            except Exception as exc:
                c = q = None
                torch.cuda.empty_cache()
                inputs_meta.append(dict(n=n, batch=batch, error=err_text(exc)))
            if c is not None:
                try:
                    inputs_meta.append(dict(n=n, batch=batch, **{'input_' + k: v for k, v in quality(c, q).items()}))
                except Exception as exc:
                    inputs_meta.append(dict(n=n, batch=batch, quality_error=err_text(exc)))
                    torch.cuda.empty_cache()
            print('  input', inputs_meta[-1], flush=True)
            bf16_methods = [mm for mm in methods if METHODS[mm][1] == 'bf16']
            fp32_methods = [mm for mm in methods if METHODS[mm][1] == 'fp32']
            # output quality (eager, once), outside timing
            extras = {}
            for name in methods:
                fn, storage = METHODS[name]
                if c is None:
                    extras[name] = dict(output_check_error='input generation failed')
                    continue
                try:
                    cc, qq = (c.float(), q.float()) if storage == 'fp32' else (c, q)
                    res = fn(cc, qq)
                    del cc, qq
                    info = None
                    if isinstance(res, tuple):
                        if name == 'cholqr':
                            info = res[2]
                        res = res[0]
                    e = quality(c, res)
                    if info is not None:
                        e['chol_failures'] = int((info != 0).sum())
                    extras[name] = {'out_' + k: v for k, v in e.items()}
                    del res, info
                except Exception as exc:
                    extras[name] = dict(output_check_error=err_text(exc))
                torch.cuda.empty_cache()
            for compiled in modes:
                if c is None:
                    for name in methods:
                        rows.append(row(name, n, batch, METHODS[name][1], compiled,
                                        dict(error='input generation failed: ' + str(inputs_meta[-1].get('error'))),
                                        extra=extras.get(name)))
                    continue
                entries = [(name, METHODS[name][0], (lambda: (c, q))) for name in bf16_methods]
                st = bench.run_block(entries, compiled)
                for name in bf16_methods:
                    rows.append(row(name, n, batch, 'bf16', compiled, st[name], extra=extras.get(name)))
                    print('  ', ('C' if compiled else 'E'), name, rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
                del st
                torch.cuda.empty_cache()
            if fp32_methods and c is not None:
                cf, qf = c.float(), q.float()
                del c, q
                c = q = None
                torch.cuda.empty_cache()
                for compiled in modes:
                    entries = [(name, METHODS[name][0], (lambda: (cf, qf))) for name in fp32_methods]
                    st = bench.run_block(entries, compiled)
                    for name in fp32_methods:
                        rows.append(row(name, n, batch, 'fp32', compiled, st[name], extra=extras.get(name)))
                        print('  ', ('C' if compiled else 'E'), name, rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
                    del st
                    torch.cuda.empty_cache()
                del cf, qf
            c = q = None
            torch.cuda.empty_cache()
            # transport / V permutation: square n x n plus production shapes
            shape_list = [(n, n)] + [s for s in PROD_SHAPES if s[0] == n and s != (n, n)]
            for (mm, nn) in shape_list:
                if (mm, nn, batch) in shapes_done:
                    continue
                shapes_done.add((mm, nn, batch))
                try:
                    g = torch.Generator(device='cuda').manual_seed(7 + mm + nn + batch)
                    rnd = lambda *s: (torch.randn(*s, device='cuda', generator=g) / math.sqrt(s[-1])).to(BF16)
                    t = dict(m=rnd(batch, mm, nn), ol=rnd(batch, mm, mm), orr=rnd(batch, nn, nn),
                             nl=rnd(batch, mm, mm), nr=rnd(batch, nn, nn), v=rnd(batch, mm, nn).abs(),
                             lo=torch.stack([torch.randperm(mm, device='cuda', generator=g) for _ in range(batch)]),
                             ro=torch.stack([torch.randperm(nn, device='cuda', generator=g) for _ in range(batch)]))
                except Exception as exc:
                    for compiled in modes:
                        rows.append(row('transport_moment', nn, batch, 'bf16', compiled, dict(error=err_text(exc)), m=mm))
                        rows.append(row('v_permute', nn, batch, 'bf16', compiled, dict(error=err_text(exc)), m=mm))
                    torch.cuda.empty_cache()
                    continue
                for compiled in modes:
                    entries = [('transport_moment', m_transport, lambda: (t['m'], t['ol'], t['orr'], t['nl'], t['nr'])),
                               ('v_permute', m_vperm, lambda: (t['v'], t['lo'], t['ro']))]
                    st = bench.run_block(entries, compiled)
                    for name in ('transport_moment', 'v_permute'):
                        rows.append(row(name, nn, batch, 'bf16', compiled, st[name], m=mm))
                        print('  ', ('C' if compiled else 'E'), name, (mm, nn), rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
                del t
                torch.cuda.empty_cache()
            dump()

    if not args.skip_variants:
        try:
            variants, exact = variant_rows(bench, modes, dims=[int(x) for x in args.variant_dims.split(',')])
        except Exception as exc:
            exact = [dict(error=err_text(exc), trace=traceback.format_exc()[-2000:])]
        print('variant bit-exact', exact, flush=True)
    dump(final=True)
    try:
        make_figure(out_path, fig_path)
    except Exception as exc:
        print('figure failed', err_text(exc), flush=True)
    print(json.dumps(reconcile(rows, events), indent=1))


if __name__ == '__main__':
    main()
