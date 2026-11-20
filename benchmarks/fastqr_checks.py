"""Checks for bf16_soap_fastqr (fast-QR study, 2026-11-20).

  unit     CPU (or --device cuda, compiled): default FastRefreshSOAP == production CompiledSOAP bit for bit
           (warm10/QR50 and QR10), warm_basis_ns(.., 2) == warm_basis, the 'cholqr' event == production 'qr'
           event apart from the basis, and the Householder fallback on a factor that really fails.
  trained  Saved trained factors (/workspace/data/turbosoap_refresh_study_20261120/d<width>): along a
           production Householder QR10 chain (BF16, from production's eigh initialization), branch C1 CholQR3
           and Householder QR from the same input at every refresh age, and also chain CholQR10 (with
           fallback) to compare chained r_C. Writes JSON.
  compiled GPU: compiled IEEE CholQR3 vs eager IEEE vs eager TF32 on trained factors.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import bf16_soap  # noqa: E402
from bf16_soap import CompiledSOAP, STATE_KEYS, qr_basis, warm_basis, transport_moment, matrix_step  # noqa: E402
import bf16_soap_fastqr as fq  # noqa: E402

BF16, FP32, FP64 = torch.bfloat16, torch.float32, torch.float64
DATA = Path('/workspace/data/turbosoap_refresh_study_20261120')
OUT = ROOT / 'reports/fastqr_20261120'


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ============================================================================ unit checks
def make_params(shapes, per_shape, device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = []
    for (m, n) in shapes:
        for _ in range(per_shape):
            out.append((torch.randn(m, n, generator=g) * .02).to(device).contiguous())
    return out


def grads_for(params, call, device, rank_one=()):
    g = torch.Generator(device='cpu').manual_seed(10_000 + call)
    out = []
    for i, p in enumerate(params):
        if i in rank_one:  # only the first row nonzero: the left covariance is exactly rank 1, even in BF16
            a = torch.zeros(p.shape)
            a[0] = torch.randn(p.shape[1], generator=g) * 1e-3
            out.append(a.to(device))
        else:
            out.append((torch.randn(p.shape, generator=g) * 1e-3).to(device))
    return out


def run_optimizer(cls, shapes, calls, device, compile_kernels, rank_one=(), **kw):
    params = make_params(shapes, 3, device)
    opt = cls(params, lr=.0015, compile_kernels=compile_kernels, **kw)
    for call in range(1, calls + 1):
        for p, gr in zip(params, grads_for(params, call, device, rank_one)):
            p.grad = gr
        opt.step()
    return opt, params


def collect(opt, params):
    tensors = [p.detach().clone() for p in params]
    for group in opt.param_groups:
        s = opt.state[group['params'][0]]
        tensors += [s[k].clone() for k in STATE_KEYS]
    return tensors


def unit(device, compiled):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = {}
    shapes = [(48, 32), (32, 32), (64, 32), (32, 64)] if device == 'cpu' else \
        [(2304, 768), (768, 768), (3072, 768), (768, 1536)]
    calls = 121

    # 1. default path == production, two schedules
    for name, kw in [('warm10_qr50', dict(precondition_frequency=10, hard_reset_interval=50)),
                     ('qr10', dict(precondition_frequency=10, hard_reset_interval=10))]:
        a = collect(*run_optimizer(CompiledSOAP, shapes, calls, device, compiled, **kw))
        if compiled:
            torch._dynamo.reset()
        b = collect(*run_optimizer(fq.FastRefreshSOAP, shapes, calls, device, compiled, **kw))
        if compiled:
            torch._dynamo.reset()
        equal = all(torch.equal(x, y) for x, y in zip(a, b))
        results[f'default_equals_production_{name}'] = equal
        say(f'default == production ({name}, {calls} calls, compiled={compiled}):', equal)
        assert equal

    # 2. warm_basis_ns(2) == warm_basis; ns=1 differs; both eager
    g = torch.Generator(device='cpu').manual_seed(1)
    n = 96
    x = torch.randn(4, n, 3 * n, generator=g)
    c = (x @ x.transpose(-1, -2) / (3 * n)).to(BF16).to(device)
    q = torch.linalg.qr(torch.randn(4, n, n, generator=g)).Q.to(BF16).to(device)
    w2 = fq.warm_basis_ns(c, q, BF16, BF16, 2)
    results['warm_ns2_equals_production'] = bool(torch.equal(w2, warm_basis(c, q, BF16, BF16)))
    results['warm_ns1_differs'] = not torch.equal(fq.warm_basis_ns(c, q, BF16, BF16, 1), w2)
    say('warm_basis_ns(2) == warm_basis:', results['warm_ns2_equals_production'],
        '| ns=1 differs:', results['warm_ns1_differs'])
    assert results['warm_ns2_equals_production'] and results['warm_ns1_differs']

    # 3. householder_from_order(order of sorted_projection) == qr_basis (eager)
    ordered, order = fq.sorted_projection(c, q)
    hq, ho = qr_basis(c, q, BF16)
    results['householder_from_order_equals_qr_basis'] = bool(
        torch.equal(order, ho) and torch.equal(fq.householder_from_order(c, q, order, BF16), hq))
    say('householder_from_order == qr_basis:', results['householder_from_order_equals_qr_basis'])
    assert results['householder_from_order_equals_qr_basis']

    # 4. fast event functions == production event functions on identical inputs (eager)
    p0 = torch.randn(3, 48, 32, generator=g).to(device)
    gr = torch.randn(3, 48, 32, generator=g).to(device) * 1e-3
    xs = torch.randn(3, 48, 200, generator=g)
    ys = torch.randn(3, 32, 200, generator=g)
    state = (torch.randn(3, 48, 32, generator=g).to(BF16).to(device) * 1e-3,
             torch.rand(3, 48, 32, generator=g).to(BF16).to(device) * 1e-6,
             (xs @ xs.mT / 200).to(BF16).to(device), (ys @ ys.mT / 200).to(BF16).to(device),
             torch.linalg.qr(torch.randn(3, 48, 48, generator=g)).Q.to(BF16).to(device),
             torch.linalg.qr(torch.randn(3, 32, 32, generator=g)).Q.to(BF16).to(device))
    age, lr = torch.tensor(20., device=device), torch.tensor(.001, device=device)
    cfg = dict(beta1=.95, beta2=.95, cov_beta=.95, eps=1e-8, weight_decay=.05, compute_dtype=BF16, basis_dtype=BF16)
    fcfg = dict(cfg, warm_ns_steps=2, **fq.CHOLQR_DEFAULTS)
    pp, ps = matrix_step(p0, gr, state, age, lr, event='warm', **cfg)
    fp, fs, _ = fq.fast_matrix_step(p0, gr, state, age, lr, event='warm', **fcfg)
    results['fast_warm_ns2_event_equals_production'] = bool(torch.equal(pp, fp) and all(torch.equal(a, b) for a, b in zip(ps, fs)))
    pp, ps = matrix_step(p0, gr, state, age, lr, event='qr', **cfg)
    fp, fs, ex = fq.fast_matrix_step(p0, gr, state, age, lr, event='cholqr', **fcfg)
    same = torch.equal(pp, fp) and all(torch.equal(ps[i], fs[i]) for i in (2, 3))  # p, C_L, C_R
    same_v = torch.equal(ps[1], fs[1])  # V permutation (orders equal on this input)
    angle = max(column_angles(fs[i].double(), ps[i].double()).max().item() for i in (4, 5))
    results['cholqr_event_matches_qr_event'] = dict(params_and_covariances_bit_equal=bool(same),
                                                    v_permutation_bit_equal=bool(same_v),
                                                    max_basis_column_angle=angle, ok=bool(ex[3].all() and ex[4].all()))
    say('cholqr event vs production qr event:', results['cholqr_event_matches_qr_event'])
    assert same and same_v and angle < 1e-2

    # 5. fallback on a factor that really fails (first-row-only gradient -> exactly rank-1 left covariance)
    opt, params = run_optimizer(fq.FastRefreshSOAP, [(64, 48)], 10, device, compiled, rank_one=(1,),
                                qr_kind='cholqr', precondition_frequency=10, hard_reset_interval=10)
    s = opt.state[params[0]]
    before = {k: s[k].clone() for k in STATE_KEYS}
    for p, gg in zip(params, grads_for(params, 11, device, rank_one=(1,))):
        p.grad = gg
    pending_copy = {}  # call 11 = age 10 = the first QR event
    orig_resolve = opt._resolve_cholqr

    def spy():
        e = opt._pending[0]
        pending_copy.update(left_ok=e['extras'][3].clone(), right_ok=e['extras'][4].clone(),
                            left_order=e['extras'][1].clone(), pre_m=e['extras'][0].clone(),
                            ql_before_fallback=opt.state[params[0]]['ql'].clone())
        orig_resolve()
    opt._resolve_cholqr = spy
    opt.step()
    assert opt.last_event == 'qr'
    s = opt.state[params[0]]
    bad = (~pending_copy['left_ok']).nonzero().flatten().tolist()
    good = pending_copy['left_ok'].nonzero().flatten().tolist()
    expect_q = fq.householder_from_order(s['cl'], before['ql'], pending_copy['left_order'], BF16)
    production_q, _ = qr_basis(s['cl'], before['ql'], BF16)
    expect_m = transport_moment(pending_copy['pre_m'], before['ql'], before['qr'], s['ql'], s['qr'], BF16)
    fb = dict(failed_left_indices=bad, refresh_log=opt.refresh_log, fallback_log=opt.fallback_log,
              fallback_q_equals_householder=all(torch.equal(s['ql'][i], expect_q[i]) for i in bad),
              fallback_q_equals_production_qr_basis=all(torch.equal(s['ql'][i], production_q[i]) for i in bad),
              fallback_m_retransported=all(torch.equal(s['m'][i], expect_m[i]) for i in bad),
              untouched_elements_kept=all(torch.equal(s['ql'][i], pending_copy['ql_before_fallback'][i]) for i in good),
              states_finite=all(bool(torch.isfinite(s[k].float()).all()) for k in STATE_KEYS),
              fallback_gram_error=float(gram_error(s['ql'][bad].double()).max()) if bad else None)
    results['fallback'] = fb
    say('fallback:', fb)
    assert bad == [1] and fb['fallback_q_equals_householder'] and fb['fallback_m_retransported']
    assert fb['untouched_elements_kept'] and fb['states_finite'] and opt.refresh_counts()['fallback_factors'] >= 1
    if device == 'cpu' and not compiled:
        assert fb['fallback_q_equals_production_qr_basis']
    return results


# ============================================================================ metrics
def column_angles(a, b):
    """Per-column angle between a and b (b, n, k) after sign fixing (sign-invariant)."""
    num = (a * b).sum(-2).abs()
    den = a.norm(dim=-2) * b.norm(dim=-2)
    return torch.arccos((num / den.clamp_min(1e-300)).clamp(max=1.))


def gram_error(q):
    n = q.shape[-1]
    eye = torch.eye(n, dtype=q.dtype, device=q.device)
    return (q.transpose(-1, -2) @ q - eye).norm(dim=(-2, -1)) / math.sqrt(n)


def prefix_distance(a, b, k):
    """1 - ||A_k^T B_k||_F^2 / k for the first k columns (QR of sorted Z preserves nested column spans)."""
    return (1 - (a[..., :k].transpose(-1, -2) @ b[..., :k]).norm(dim=(-2, -1)) ** 2 / k).clamp_min(0)


def r_c(c, q):
    b = q.transpose(-1, -2) @ c @ q
    off = b - torch.diag_embed(torch.diagonal(b, dim1=-2, dim2=-1))
    return off.norm(dim=(-2, -1)) / b.norm(dim=(-2, -1)).clamp_min(1e-300)


# ============================================================================ trained factors
def load_snapshot(width, idx, slot, device):
    a = np.load(DATA / f'd{width}' / f'f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(BF16).unsqueeze(0).to(device)


def eigh_init(c):
    m = c.float()
    m = .5 * (m + m.transpose(-1, -2))
    return torch.linalg.eigh(m).eigenvectors.flip(-1).to(BF16).contiguous()


class precision:
    def __init__(self, ieee):
        self.ieee = ieee

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = not self.ieee if torch.cuda.is_available() else False

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev


def first_failing_pass(x, cfg):
    """Diagnose which CholQR3 pass fails (0 = shifted), eager, IEEE."""
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=FP32)
    g = x.transpose(-1, -2) @ x
    v = torch.full((1, n, 1), 1 / math.sqrt(n), device=x.device)
    for _ in range(cfg['power_iters']):
        v = g @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    lam = v.transpose(-1, -2) @ g @ v
    for k in range(1 + cfg['unshifted_passes']):
        if k:
            g = x.transpose(-1, -2) @ x
        u, info = torch.linalg.cholesky_ex(g + cfg['shift_rel'] * lam * eye if k == 0 else g, upper=True)
        if int(info.max()):
            return dict(pass_index=k, pivot=int(info.max()))
        x = torch.linalg.solve_triangular(u, x, upper=True, left=False)
    return dict(pass_index=None, pivot=None, nonfinite=not bool(torch.isfinite(x).all()))


@torch.no_grad()
def trained(width, device, factors, age_stride, cfg, out_path):
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    meta = json.loads((DATA / f'd{width}' / 'meta.json').read_text())
    slots = list(range(0, 101))
    rows, chains = [], []
    for f in (factors if factors is not None else range(len(meta['factors']))):
        info = meta['factors'][f]
        n = info['dim']
        t0 = time.time()
        c0 = load_snapshot(width, f, 0, device)
        q_hh = eigh_init(c0)          # production Householder QR10 chain
        q_cq = q_hh.clone()           # C1 CholQR10 chain (with Householder fallback)
        chain = dict(factor=f, n=n, param=info['param'], side=info['side'], ages=[], hh_after=[], hh_before=[],
                     cq_after=[], cq_before=[], cq_fallbacks=0, cq_fallback_ages=[])
        for slot in slots[1:]:
            age = slot * 10
            c = load_snapshot(width, f, slot, device)
            cd = c.double() if device == 'cpu' else c.float()
            # chain metrics before refresh (previous basis on the new covariance)
            with precision(ieee=True):
                chain['hh_before'].append(float(r_c(cd, q_hh.to(cd.dtype))))
                chain['cq_before'].append(float(r_c(cd, q_cq.to(cd.dtype))))
            branch = slot % age_stride == 0 or slot <= 5
            if branch:
                # same-input branch: production qr_basis (training precision) vs C1 (IEEE), from q_hh
                with precision(ieee=False):
                    q_prod, order_prod = qr_basis(c, q_hh, BF16)
                with precision(ieee=True):
                    x, order = fq.sorted_projection(c, q_hh)
                    qc32, ok = fq.cholqr3(x, **cfg)
                    qh32 = torch.linalg.qr(x).Q
                    qc7, ok7 = fq.cholqr3(x, **dict(cfg, shift_rel=1e-7))  # the panel's alternative shift
                ok = bool(ok.all())
                row = dict(factor=f, n=n, side=info['side'], param=info['param'], age=age, ok=ok,
                           order_equal_to_qr_basis=float((order == order_prod).float().mean()),
                           ok_shift_1e7=bool(ok7.all()))
                if bool(ok7.all()):
                    row['angle_max_first_half_shift_1e7'] = float(column_angles(qc7.double(), qh32.double())[0][:max(1, n // 2)].max())
                del qc7
                if not ok:
                    with precision(ieee=True):
                        row['failure'] = first_failing_pass(x, cfg)
                else:
                    hd, cdd = qh32.double(), qc32.double()
                    ang = column_angles(cdd, hd)[0]
                    floor = column_angles(qh32.to(BF16).double(), hd)[0]
                    cb, hb = qc32.to(BF16).double(), qh32.to(BF16).double()
                    k8, k2 = max(1, n // 8), n // 2
                    with precision(ieee=True):
                        rc_c = float(r_c(cd, cb.to(cd.dtype)))
                        rc_h = float(r_c(cd, hb.to(cd.dtype)))
                        rc_p = float(r_c(cd, q_prod.to(cd.dtype)))
                    row.update(
                        angle_max=float(ang.max()), angle_median=float(ang.median()),
                        angle_p99=float(ang.quantile(.99)) if n > 1 else float(ang.max()),
                        angle_max_first_half=float(ang[:k2].max()),
                        bf16_floor_angle_max=float(floor.max()),
                        bf16_angle_max=float(column_angles(cb, hb)[0].max()),
                        prefix_dist_n8=float(prefix_distance(cdd, hd, k8)),
                        prefix_dist_n2=float(prefix_distance(cdd, hd, k2)),
                        bf16_prefix_dist_n8=float(prefix_distance(cb, hb, k8)),
                        e_q_cholqr_bf16=float(gram_error(cb)), e_q_householder_bf16=float(gram_error(hb)),
                        e_q_production_bf16=float(gram_error(q_prod.double())),
                        r_c_cholqr=rc_c, r_c_householder_same_order=rc_h, r_c_production=rc_p)
                rows.append(row)
                q_hh_new = q_prod
            else:
                with precision(ieee=False):
                    q_hh_new, _ = qr_basis(c, q_hh, BF16)
            # CholQR chain step (C1 with fallback)
            with precision(ieee=True):
                qn, order, okc = fq.cholqr_basis(c, q_cq, BF16, **cfg)
                if not bool(okc.all()):
                    qn = fq.householder_from_order(c, q_cq, order, BF16)
                    chain['cq_fallbacks'] += 1
                    chain['cq_fallback_ages'].append(age)
            q_hh, q_cq = q_hh_new, qn
            with precision(ieee=True):
                chain['hh_after'].append(float(r_c(cd, q_hh.to(cd.dtype))))
                chain['cq_after'].append(float(r_c(cd, q_cq.to(cd.dtype))))
            chain['ages'].append(age)
        # late-window interval r_C: mean(after_t, before_{t+10}) over ages 500..1000
        for name in ('hh', 'cq'):
            after, before, ages = chain[f'{name}_after'], chain[f'{name}_before'], chain['ages']
            vals = [.5 * (after[i] + before[i + 1]) for i in range(len(ages) - 1) if 500 <= ages[i] < 1000]
            chain[f'{name}_late_interval'] = sum(vals) / len(vals)
        chains.append(chain)
        fr = [r for r in rows if r['factor'] == f]
        say(f'd={width} f{f} n={n} {info["side"]}: {time.time() - t0:.0f}s, branch fails '
            f'{sum(not r["ok"] for r in fr)}/{len(fr)}, chain fallbacks {chain["cq_fallbacks"]}, '
            f'late r_C interval hh {chain["hh_late_interval"]:.4f} cq {chain["cq_late_interval"]:.4f}, '
            f'max angle {max((r.get("angle_max_first_half", 0) for r in fr), default=0):.2e}, '
            f'max prefix n/8 {max((r.get("prefix_dist_n8", 0) for r in fr), default=0):.1e}')
        json_dump(out_path, dict(width=width, device=device, config=cfg, age_stride=age_stride,
                                 torch=torch.__version__, rows=rows, chains=chains,
                                 gpu=torch.cuda.get_device_name() if device == 'cuda' else None))
    return rows, chains


@torch.no_grad()
def compiled_check(out_path, cfg):
    """GPU: compiled cholqr_basis under IEEE vs eager IEEE vs eager TF32, on trained factors."""
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    fn = torch.compile(lambda c, q: fq.cholqr_basis(c, q, BF16, **cfg), fullgraph=True, dynamic=False,
                       options={'emulate_precision_casts': True})
    rows = []
    for width, f, slot in [(1024, 0, 50), (1024, 4, 50), (1024, 2, 100), (512, 4, 20), (512, 0, 3)]:
        c = load_snapshot(width, f, slot, 'cuda')
        q = eigh_init(load_snapshot(width, f, slot - 1, 'cuda'))
        out = {}
        for name, ieee, comp in [('compiled_ieee', True, True), ('eager_ieee', True, False), ('eager_tf32', False, False)]:
            with precision(ieee=ieee):
                qq, order, ok = (fn if comp else (lambda c, q: fq.cholqr_basis(c, q, BF16, **cfg)))(c, q)
                x, _ = fq.sorted_projection(c, q)
                if comp or not ieee:
                    gram = x.transpose(-1, -2) @ x  # the Gram under this precision
                else:
                    gram = x.transpose(-1, -2) @ x
            out[name] = (qq, order, ok, gram)
        with precision(ieee=True):
            x, _ = fq.sorted_projection(c, q)
            g64 = (x.double().transpose(-1, -2) @ x.double())
            hh = torch.linalg.qr(x).Q.double()
        row = dict(width=width, factor=f, age=slot * 10, n=c.shape[-1])
        for name, (qq, order, ok, gram) in out.items():
            row[name] = dict(ok=bool(ok.all()),
                             gram_rel_err_vs_fp64=float((gram.double() - g64).norm() / g64.norm()),
                             angle_max_vs_householder=float(column_angles(qq.double(), hh)[0].max()) if bool(ok.all()) else None,
                             e_q=float(gram_error(qq.double())))
        a, b = out['compiled_ieee'][0].double(), out['eager_ieee'][0].double()
        row['compiled_vs_eager_ieee_angle_max'] = float(column_angles(a, b)[0].max())
        row['compiled_vs_eager_ieee_bit_equal'] = bool(torch.equal(out['compiled_ieee'][0], out['eager_ieee'][0]))
        rows.append(row)
        say(json.dumps(row))
    json_dump(out_path, dict(rows=rows, config=cfg, torch=torch.__version__, gpu=torch.cuda.get_device_name()))


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('unit', 'trained', 'compiled'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--compiled', action='store_true')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--factors', type=int, nargs='*')
    p.add_argument('--age-stride', type=int, default=1, help='branch-compare every k-th refresh (plus ages<=50)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--out')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg = dict(fq.CHOLQR_DEFAULTS)
    if a.mode == 'unit':
        r = unit(a.device, a.compiled)
        json_dump(a.out or OUT / f'unit_{a.device}{"_compiled" if a.compiled else ""}.json',
                  dict(results=r, torch=torch.__version__, device=a.device, compiled=a.compiled,
                       gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None))
        say('ALL UNIT CHECKS PASSED')
    elif a.mode == 'trained':
        trained(a.width, a.device, a.factors, a.age_stride, cfg,
                a.out or OUT / f'trained_d{a.width}_{a.device}.json')
    else:
        compiled_check(a.out or OUT / 'compiled_ieee_check.json', cfg)


if __name__ == '__main__':
    main()
