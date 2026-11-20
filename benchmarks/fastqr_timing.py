"""Fast-QR study timing (2026-11-20): 109M optimizer event regions per refresh kind, and per-factor primitives.

Event regions follow refresh_study_microbench.event_regions: ``opt.step()`` over 4 groups x 12 matrices of the
109M shapes plus fused AdamW on a 50257x768 stand-in, events forced by setting ``opt.call``, CUDA-event and
synchronized host-wall medians, interleaved repetitions. Differences from that harness: fresh random gradients
before every step (outside the timed region) after a 40-step warm-in, so the covariances are full rank and the
CholQR Gram is not trivially singular; and every timed CholQR event's fallback count is recorded.

Optimizers: production ``CompiledSOAP`` (ordinary / warm / Householder QR) and ``FastRefreshSOAP(qr_kind='cholqr',
warm_ns_steps=1)`` (its QR event is C1, its warm event C2; its ordinary event is the production function).

Backend settings as the 109M runs: deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, FP32 matmul
'high' (the CholQR region switches to IEEE itself), no BF16 reduced-precision reductions, 4 CPU threads.
Requires an exclusive GPU (gpu-claim).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
from bf16_soap import CompiledSOAP, qr_basis, warm_basis, transport_moment  # noqa: E402
import bf16_soap_fastqr as fq  # noqa: E402

BF16, FP32 = torch.bfloat16, torch.float32
PROD_SHAPES = [(2304, 768), (768, 768), (3072, 768), (768, 1536)]
PROD_FACTORS = [768] * 5 + [1536, 2304, 3072]
TARGETS = dict(ordinary=10.14, warm=61.61, qr=601.18)
OUT = ROOT / 'reports/fastqr_20261120'


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def quantiles(xs):
    t = torch.tensor(xs, dtype=torch.float64)
    return dict(median_ms=float(t.median()), p25_ms=float(t.quantile(.25)), p75_ms=float(t.quantile(.75)), reps=len(xs))


class ieee:
    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def setup():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


# ----------------------------------------------------------------------------- event regions
def make_params(seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    params = [(torch.randn(m, n, device='cuda', generator=g) * .02).contiguous()
              for (m, n) in PROD_SHAPES for _ in range(12)]
    emb = torch.randn(50257, 768, device='cuda', generator=g) * .02
    return params, emb


def fresh_grads(params, emb, gen):
    for p in params:
        p.grad = torch.randn(p.shape, device='cuda', generator=gen) * 1e-3
    emb.grad = torch.randn(emb.shape, device='cuda', generator=gen) * 1e-3


def event_regions(reps, warm_in=40):
    gen = torch.Generator(device='cuda').manual_seed(7)
    arms = {}
    for name, factory in [('production', lambda ps: CompiledSOAP(ps, lr=.0015)),
                          ('fast', lambda ps: fq.FastRefreshSOAP(ps, lr=.0015, qr_kind='cholqr', warm_ns_steps=1))]:
        params, emb = make_params(0)
        opt = factory(params)
        aux = torch.optim.AdamW([emb], lr=6e-4, fused=True)
        fresh_grads(params, emb, gen)
        opt.step()  # cold: eigh initialization
        for i in range(warm_in):  # ordinary calls build full-rank covariances
            fresh_grads(params, emb, gen)
            opt.call = 10 * (i // 9) + 1 + i % 9
            opt.step()
            assert opt.last_event == 'ordinary'
        arms[name] = dict(opt=opt, aux=aux, params=params, emb=emb)
    ages = dict(ordinary=11, warm=10, qr=50)
    cells = [(arm, ev) for arm in arms for ev in ages]

    def run(arm, event, with_aux=True):
        a = arms[arm]
        fresh_grads(a['params'], a['emb'], gen)
        a['opt'].call = ages[event]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        a['opt'].step()
        if with_aux:
            a['aux'].step()
        end.record()
        torch.cuda.synchronize()
        assert a['opt'].last_event == event
        return float(start.elapsed_time(end)), 1000 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for arm, ev in cells:
        run(arm, ev)
    compile_s = time.perf_counter() - t0
    for _ in range(3):
        for arm, ev in cells:
            run(arm, ev)
    fb0 = arms['fast']['opt'].refresh_counts()
    samples = {c: dict(cuda=[], wall=[], matrix_only=[]) for c in cells}
    for r in range(reps):
        for arm, ev in (cells if r % 2 == 0 else cells[::-1]):
            c, w = run(arm, ev)
            samples[(arm, ev)]['cuda'].append(c)
            samples[(arm, ev)]['wall'].append(w)
            samples[(arm, ev)]['matrix_only'].append(run(arm, ev, with_aux=False)[0])
    fb1 = arms['fast']['opt'].refresh_counts()
    out = dict(cells={}, compile_s=compile_s, targets_109M=TARGETS)
    for (arm, ev), s in samples.items():
        out['cells'][f'{arm}/{ev}'] = dict(cuda=quantiles(s['cuda']), wall=quantiles(s['wall']),
                                           matrix_only_cuda=quantiles(s['matrix_only']))
    out['cholqr_timed_events'] = fb1['cholqr_events'] - fb0['cholqr_events']
    out['cholqr_timed_factor_refreshes'] = fb1['cholqr_factor_refreshes'] - fb0['cholqr_factor_refreshes']
    out['cholqr_timed_fallbacks'] = fb1['fallback_factors'] - fb0['fallback_factors']
    out['cholqr_all_counts'] = fb1
    out['fallback_log'] = arms['fast']['opt'].fallback_log[:50]
    out['note'] = ('Synthetic Gaussian gradients redrawn every call; events forced via opt.call. 109M reference '
                   '(training-loop host wall, production): ordinary 10.14, warm 61.61, QR 601.18 ms.')
    del arms
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- per-factor primitives
@torch.no_grad()
def factor_inputs(n, batch, seed, k_ratio=4, steps=20):
    """C: EMA of G G^T with G = U diag(i^-1/2) X, X Gaussian n x n/k_ratio (power-law spectrum), BF16.
    Q: the eigenbasis of C perturbed by a small rotation (r_C about 0.1-0.2), BF16."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    cs, qs = [], []
    lam = torch.arange(1, n + 1, device='cuda', dtype=FP32).pow(-1.)
    for _ in range(batch):
        u = torch.linalg.qr(torch.randn(n, n, device='cuda', generator=g)).Q
        w = torch.zeros(n, n, device='cuda')
        for _ in range(steps):
            x = torch.randn(n, n // k_ratio, device='cuda', generator=g) / math.sqrt(n // k_ratio)
            w.mul_(.95).addmm_(x, x.T, alpha=.05)
        us = u * lam.sqrt()
        c = us @ w @ us.T
        cs.append((.5 * (c + c.T)).to(BF16))
        a = torch.randn(n, n, device='cuda', generator=g) * (.2 / math.sqrt(n))
        qq = torch.linalg.qr(u + u @ (a - a.T) / 2).Q
        qs.append(qq.to(BF16))
    return torch.stack(cs).contiguous(), torch.stack(qs).contiguous()


def cuda_ms(fn, args, reps):
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        out.append(float(s.elapsed_time(e)))
    return out


def compile_fn(fn):
    return torch.compile(fn, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})


def primitives(reps, dims=(768, 1536, 2304, 3072), batch=12):
    rows = []
    cfg = fq.CHOLQR_DEFAULTS
    for n in dims:
        c, q = factor_inputs(n, batch, 100 + n)
        methods = dict(
            householder_qr_basis=(compile_fn(lambda c, q: qr_basis(c, q, BF16)), False),
            cholqr_basis_ieee=(compile_fn(lambda c, q: fq.cholqr_basis(c, q, BF16, **cfg)), True),
            cholqr_basis_tf32=(compile_fn(lambda c, q: fq.cholqr_basis(c, q, BF16, **cfg)), False),
            warm_ns2=(compile_fn(lambda c, q: warm_basis(c, q, BF16, BF16)), False),
            warm_ns1=(compile_fn(lambda c, q: fq.warm_basis_ns(c, q, BF16, BF16, 1)), False),
        )
        x32 = torch.bmm(c.float(), q.float())
        g32 = torch.bmm(x32.mT, x32)
        u32 = torch.linalg.cholesky(g32 + 1e-6 * g32.diagonal(dim1=-2, dim2=-1).amax(-1)[:, None, None]
                                    * torch.eye(n, device='cuda'), upper=True)
        parts = dict(
            gram_ieee=(lambda x: torch.bmm(x.mT, x), (x32,), True),
            gram_tf32=(lambda x: torch.bmm(x.mT, x), (x32,), False),
            cholesky_ex=(lambda g: torch.linalg.cholesky_ex(g, upper=True), (g32,), True),
            solve_triangular_ieee=(lambda u, x: torch.linalg.solve_triangular(u, x, upper=True, left=False), (u32, x32), True),
            householder_qr_only=(lambda x: torch.linalg.qr(x).Q, (x32,), True),
        )
        samples = {k: [] for k in (*methods, *parts)}
        for k, (f, use_ieee) in methods.items():  # compile + warmup
            for _ in range(3):
                if use_ieee:
                    with ieee():
                        f(c, q)
                else:
                    f(c, q)
        for r in range(2):
            order = list(methods) if r == 0 else list(methods)[::-1]
            for k in order:
                f, use_ieee = methods[k]
                if use_ieee:
                    with ieee():
                        samples[k] += cuda_ms(f, (c, q), reps)
                else:
                    samples[k] += cuda_ms(f, (c, q), reps)
            for k, (f, args, use_ieee) in parts.items():
                cuda_ms(f, args, 1)
                if use_ieee:
                    with ieee():
                        samples[k] += cuda_ms(f, args, reps)
                else:
                    samples[k] += cuda_ms(f, args, reps)
        with ieee():
            _, _, ok = methods['cholqr_basis_ieee'][0](c, q)
        _, _, ok_tf32 = methods['cholqr_basis_tf32'][0](c, q)
        # one fallback: Householder of one element + its moment transport (eager)
        m_shape = next((s for s in PROD_SHAPES if n in s), (n, 768))
        mm = torch.randn(1, *m_shape, device='cuda').to(BF16)
        ql = torch.linalg.qr(torch.randn(1, m_shape[0], m_shape[0], device='cuda')).Q.to(BF16)
        qr_ = torch.linalg.qr(torch.randn(1, m_shape[1], m_shape[1], device='cuda')).Q.to(BF16)
        order = torch.arange(n, device='cuda').unsqueeze(0)

        def fallback():
            with ieee():
                qn = fq.householder_from_order(c[:1], q[:1], order, BF16)
                if m_shape[0] == n:
                    transport_moment(mm, ql, qr_, qn, qr_, BF16)
                else:
                    transport_moment(mm, ql, qr_, ql, qn, BF16)
        cuda_ms(fallback, (), 2)
        samples['one_fallback_eager'] = cuda_ms(fallback, (), reps)
        row = dict(n=n, batch=batch, cholqr_ok_ieee=int(ok.sum()), cholqr_ok_tf32=int(ok_tf32.sum()))
        for k, s in samples.items():
            row[k] = quantiles(s)
        rows.append(row)
        say(f'n={n}: ' + ', '.join(f'{k} {v["median_ms"]:.2f}' for k, v in row.items() if isinstance(v, dict)))
        del c, q, x32, g32, u32, methods
        torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=30)
    p.add_argument('--prim-reps', type=int, default=10)
    p.add_argument('--out', default=str(OUT / 'timing.json'))
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('GPU_CLAIM_INDICES'), 'run under gpu-claim'
    setup()
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               claim=os.environ.get('GPU_CLAIM_INDICES') or os.environ.get('CUDA_VISIBLE_DEVICES'), loadavg_start=os.getloadavg(),
               started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    say('env', env)
    result = dict(env=env)
    result['event_regions'] = event_regions(a.reps)
    for k, v in result['event_regions']['cells'].items():
        say(f'{k}: cuda {v["cuda"]["median_ms"]:.2f} ms, wall {v["wall"]["median_ms"]:.2f} ms, matrix-only {v["matrix_only_cuda"]["median_ms"]:.2f}')
    say('timed CholQR fallbacks:', result['event_regions']['cholqr_timed_fallbacks'], 'of',
        result['event_regions']['cholqr_timed_factor_refreshes'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    result['primitives'] = primitives(a.prim_reps)
    env['loadavg_end'] = os.getloadavg()
    Path(a.out).write_text(json.dumps(result, indent=1))
    say('done')


if __name__ == '__main__':
    main()
