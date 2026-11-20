"""Refresh-cost microbenchmark, extension (Part 1b of the 2026-11-20 refresh study).

Adds (i) larger factor dimensions n in {8192 (continuity rerun), 12288, 16384}
at batch 1/4/12, (ii) a shifted CholeskyQR3 baseline in FP32 (Fukaya et al.
style), run on the earlier grid as well, (iii) achieved TFLOP/s for every
method under an explicitly stated flop model, (iv) accuracy of the Cholesky-QR
variants against Householder QR on the same sorted Z = CQ.

Same backend settings, inputs, timing protocol and production functions as
``refresh_study_microbench.py`` (imported from there).  Writes new files only:
reports/refresh_study_20261120/microbench_ext.json and refresh_cost_vs_dim_ext.pdf.

Shifted CholeskyQR3 (FP32), for X = Z[:, pi] (n x n, m = n):
  G = X^T X;  s = 11 (m n + n (n + 1)) u ||X||_2^2  with u = 2^-24 and ||X||_2^2
  estimated as the Rayleigh quotient of 20 power iterations on G;
  R1 = chol(G + s I);  X1 = X R1^-1;  then two unshifted CholeskyQR passes.
No other safeguard; failures (info > 0) and non-finite outputs are recorded.
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import refresh_study_microbench as mb  # noqa: E402  (sets backend env defaults)
import torch  # noqa: E402
from bf16_soap import qr_basis  # noqa: E402
from refresh_study_common import highest  # noqa: E402

BF16, FP32 = torch.bfloat16, torch.float32
U32 = 2.0 ** -24

# Matmul/factorization flops per factor (n x n), multiply-add = 2 flops.
FLOP_MODEL = {
    'warm_bf16': (14., '4n^3 (Q^T C Q) + 2n^3 (Q A) + 2 x [2n^3 (Y^T Y) + 2n^3 (Y E)]; elementwise ignored'),
    'warm_fp32': (14., 'same as warm_bf16'),
    'qr': (14. / 3, '2n^3 (C Q) + 4/3 n^3 (geqrf) + 4/3 n^3 (orgqr)'),
    'eigh': (9., 'conventional 9n^3 for symmetric eigensolver with vectors (Golub & Van Loan); actual syevd count varies'),
    'cholqr': (16. / 3, '2n^3 (C Q) + 2n^3 (Z^T Z, full bmm) + n^3/3 (chol) + n^3 (trsm)'),
    'cholqr3': (12., '2n^3 (C Q) + 3 x [2n^3 (X^T X) + n^3/3 (chol) + n^3 (trsm)]; power iteration ignored'),
}


def cholqr3_basis(cov, q):
    cf, old = cov.float(), q.float()
    z = torch.bmm(cf, old)
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    x = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    m, n = x.shape[-2], x.shape[-1]
    gram = torch.bmm(x.transpose(-1, -2), x)
    v = torch.ones(x.shape[0], n, 1, device=x.device, dtype=x.dtype) / math.sqrt(n)
    for _ in range(20):
        v = torch.bmm(gram, v)
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    lam = torch.bmm(v.transpose(-1, -2), torch.bmm(gram, v))  # (b,1,1) ~ ||X||_2^2
    shift = 11. * (m * n + n * (n + 1)) * U32 * lam
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    lower, info1 = torch.linalg.cholesky_ex(gram + shift * eye)
    x = torch.linalg.solve_triangular(lower.transpose(-1, -2), x, upper=True, left=False)
    infos = [info1]
    for _ in range(2):
        gram = torch.bmm(x.transpose(-1, -2), x)
        lower, info = torch.linalg.cholesky_ex(gram)
        x = torch.linalg.solve_triangular(lower.transpose(-1, -2), x, upper=True, left=False)
        infos.append(info)
    return x.to(BF16).contiguous(), order, torch.stack(infos, -1), shift.flatten()


def m_cholqr3(c, q):
    return cholqr3_basis(c, q)


METHODS = dict(mb.METHODS)
METHODS['cholqr3'] = (m_cholqr3, 'bf16')


@torch.no_grad()
def agreement(c, q, q_new, order=None):
    """Per element: Householder QR (FP32) of the same sorted Z; column-wise angle
    between q_new and it after sign fixing, plus the BF16-rounding floor (angle
    between BF16(Q_qr) and Q_qr).  Returns lists over batch elements."""
    out = dict(max_angle=[], median_angle=[], rel_frob=[], floor_max_angle=[])
    for i in range(c.shape[0]):
        with highest():
            cf, old = c[i:i + 1].float(), q[i:i + 1].float()
            z = torch.bmm(cf, old)
            # Use the method's own column order when given (a separately computed sort
            # can differ at near-ties and show up as spurious pi/2 column angles).
            o = order[i:i + 1] if order is not None else (old * z).sum(-2).argsort(dim=-1, descending=True)
            ordered = z.gather(-1, o.unsqueeze(-2).expand_as(z))
            qq = torch.linalg.qr(ordered).Q[0]
            del cf, old, z, ordered

            def angles(a, b):
                dots = (a * b).sum(0)
                norms = a.norm(dim=0) * b.norm(dim=0)
                cos = (dots.abs() / norms.clamp_min(1e-30)).clamp(max=1)
                ang = torch.arccos(cos)
                ang = torch.where(torch.isfinite(ang), ang, torch.full_like(ang, float('nan')))
                return ang, torch.sign(dots)
            qn = q_new[i].float()
            ang, sgn = angles(qn, qq)
            out['max_angle'].append(float(ang.max()) if torch.isfinite(ang).all() else float('nan'))
            out['median_angle'].append(float(ang.median()) if torch.isfinite(ang).all() else float('nan'))
            out['rel_frob'].append(float((qn * sgn - qq).norm() / math.sqrt(qq.shape[0])))
            fl, _ = angles(qq.to(BF16).float(), qq)
            out['floor_max_angle'].append(float(fl.max()))
            del qq, qn
    return out


def tflops(method, n, ms):
    if ms is None or ms <= 0:
        return None
    return FLOP_MODEL[method][0] * n ** 3 / (ms * 1e-3) / 1e12


def run_cells(bench, ns, batches, methods, modes, rows, inputs_meta, dump, t_start):
    for batch in batches:
        for n in ns:
            print(f'== n={n} batch={batch} t={time.time()-t_start:.0f}s load={mb.loadavg()}', flush=True)
            c = q = None
            try:
                c, q = mb.make_factor_inputs(n, batch, seed=1000 + n + 7 * batch)
                inputs_meta.append(dict(n=n, batch=batch, **{'input_' + k: v for k, v in mb.quality(c, q).items()}))
            except Exception as exc:
                c = q = None
                torch.cuda.empty_cache()
                inputs_meta.append(dict(n=n, batch=batch, error=mb.err_text(exc)))
            print('  input', inputs_meta[-1], flush=True)
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
                    info = shift = order = None
                    if isinstance(res, tuple):
                        order = res[1]
                        if name == 'cholqr':
                            info = res[2]
                        if name == 'cholqr3':
                            info, shift = res[2], res[3]
                        res = res[0]
                    e = mb.quality(c, res)
                    if info is not None:
                        e['chol_failures'] = int((info != 0).flatten(1).any(1).sum()) if info.dim() > 1 else int((info != 0).sum())
                        if info.dim() > 1:
                            e['chol_failures_by_pass'] = (info != 0).sum(0).tolist()
                    if shift is not None:
                        e['shift_over_norm2sq'] = 11. * (2 * n * n + n) * U32
                    e['gram_err_frob_raw_mean'] = e['gram_err_mean'] * math.sqrt(n)
                    if name in ('cholqr', 'cholqr3', 'qr'):
                        e.update({'agree_' + k: v for k, v in agreement(c, q, res, order).items()})
                    extras[name] = {'out_' + k: v for k, v in e.items()}
                    del res, info
                except Exception as exc:
                    extras[name] = dict(output_check_error=mb.err_text(exc))
                torch.cuda.empty_cache()
            bf = [m for m in methods if METHODS[m][1] == 'bf16']
            fp = [m for m in methods if METHODS[m][1] == 'fp32']
            for compiled in modes:
                if c is None:
                    for name in methods:
                        rows.append(mb.row(name, n, batch, METHODS[name][1], compiled,
                                           dict(error='input generation failed: ' + str(inputs_meta[-1].get('error'))),
                                           extra=extras.get(name)))
                    continue
                st = bench.run_block([(m, METHODS[m][0], (lambda: (c, q))) for m in bf], compiled)
                for name in bf:
                    rows.append(mb.row(name, n, batch, 'bf16', compiled, st[name], extra=extras.get(name)))
                    print('  ', 'C' if compiled else 'E', name, rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
                del st
                torch.cuda.empty_cache()
            if fp and c is not None:
                try:
                    q_cpu = q.cpu()
                    cf = c.float()
                    del c
                    c = None
                    del q
                    q = None
                    torch.cuda.empty_cache()
                    qf = q_cpu.to('cuda').float()
                    del q_cpu
                except Exception as exc:
                    cf = qf = None
                    torch.cuda.empty_cache()
                    for compiled in modes:
                        for name in fp:
                            rows.append(mb.row(name, n, batch, 'fp32', compiled,
                                               dict(error='FP32 input copy failed: ' + mb.err_text(exc)), extra=extras.get(name)))
                    fp = []
                for compiled in (modes if fp else []):
                    st = bench.run_block([(m, METHODS[m][0], (lambda: (cf, qf))) for m in fp], compiled)
                    for name in fp:
                        rows.append(mb.row(name, n, batch, 'fp32', compiled, st[name], extra=extras.get(name)))
                        print('  ', 'C' if compiled else 'E', name, rows[-1]['median_ms'], rows[-1].get('error'), flush=True)
                    del st
                    torch.cuda.empty_cache()
                cf = qf = None
            c = q = None
            torch.cuda.empty_cache()
            for r in rows:
                if r['method'] in FLOP_MODEL and 'tflops' not in r:
                    r['flops_per_factor'] = FLOP_MODEL[r['method']][0] * r['n'] ** 3
                    r['tflops'] = tflops(r['method'], r['n'], r['median_ms'] / r['batch'] if r['ok'] else None)
            dump()


def crossovers(rows):
    out = {}
    for compiled in (True, False):
        for batch in sorted({r['batch'] for r in rows}):
            get = lambda m: {r['n']: r['median_ms'] for r in rows if r['method'] == m and r['batch'] == batch
                             and r['compiled'] == compiled and r['ok']}
            w, qd, c3 = get('warm_bf16'), get('qr'), get('cholqr3')
            ratios = {n: w[n] / qd[n] for n in sorted(w) if n in qd}
            ratios3 = {n: w[n] / c3[n] for n in sorted(w) if n in c3}
            cross = [n for n in sorted(ratios) if ratios[n] >= 1]
            out[f"{'compiled' if compiled else 'eager'}_b{batch}"] = dict(
                warm_over_qr={str(k): v for k, v in ratios.items()},
                warm_over_cholqr3={str(k): v for k, v in ratios3.items()},
                first_n_warm_ge_qr=cross[0] if cross else None,
                max_n_measured=max(ratios) if ratios else None)
    return out


def make_figure(path_json, path_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = json.loads(Path(path_json).read_text())
    rows = [r for r in d['rows'] if r['compiled']]
    colors = dict(warm_bf16='#2a78d6', qr='#eb6834', eigh='#1baf7a', cholqr='#eda100', warm_fp32='#e87ba4',
                  cholqr3='#008300')
    labels = dict(warm_bf16='warm (BF16, production)', warm_fp32='warm (FP32 storage)', qr='QR (production)',
                  eigh='FP32 eigh', cholqr='Cholesky-QR (FP32)', cholqr3='shifted CholeskyQR3 (FP32)')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {1: ('-', 'o'), 4: (':', 's'), 12: ('--', '^')}
    for method in colors:
        for batch, (ls, mk) in styles.items():
            pts = sorted((r['n'], r['median_ms'] / r['batch']) for r in rows
                         if r['method'] == method and r['batch'] == batch and r['ok'])
            fl = sorted((r['n'], r['tflops']) for r in rows
                        if r['method'] == method and r['batch'] == batch and r['ok'] and r.get('tflops'))
            # mark outputs that are non-finite (failed factorization) with an x
            bad = sorted((r['n'], r['median_ms'] / r['batch']) for r in rows if r['method'] == method and r['batch'] == batch
                         and r['ok'] and r.get('out_chol_failures'))
            if pts:
                axes[0].plot(*zip(*pts), ls, color=colors[method], marker=mk, ms=3.5, lw=1.5 if batch == 1 else 1,
                             label=labels[method] if batch == 1 else None)
            if bad:
                axes[0].plot(*zip(*bad), 'x', color='k', ms=7, mew=1.2)
            if fl:
                axes[1].plot(*zip(*fl), ls, color=colors[method], marker=mk, ms=3.5, lw=1.5 if batch == 1 else 1)
    for batch, (ls, mk) in styles.items():
        for den, col, lab in (('qr', '#eb6834', 'warm / QR'), ('cholqr3', '#008300', 'warm / shifted CholQR3')):
            w = {r['n']: r['median_ms'] for r in rows if r['method'] == 'warm_bf16' and r['batch'] == batch and r['ok']}
            q = {r['n']: r['median_ms'] for r in rows if r['method'] == den and r['batch'] == batch and r['ok']}
            pts = sorted((n, w[n] / q[n]) for n in w if n in q)
            if pts:
                axes[2].plot(*zip(*pts), ls, color=col, marker=mk, ms=3.5, label=f'{lab}, batch {batch}')
    axes[2].axhline(1, color='k', lw=.7)
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(alpha=.3, which='both', lw=.5)
        ax.set_xlabel('factor dimension n')
    axes[0].set_ylabel('ms per factor (compiled)')
    axes[0].set_title('(a) cost per factor; solid b1, dotted b4, dashed b12;\n× = output non-finite / Cholesky failed', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_ylabel('achieved TFLOP/s (flop model in JSON)')
    axes[1].set_title('(b) achieved rate, compiled', fontsize=9)
    axes[2].set_ylabel('time ratio')
    axes[2].set_title('(c) warm relative to QR and to shifted CholQR3', fontsize=9)
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path_pdf)
    fig.savefig(str(path_pdf).replace('.pdf', '.png'), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid-ns', default='256,512,768,1024,1536,2048,2304,3072,4096,6144')
    ap.add_argument('--grid-batches', default='1,12')
    ap.add_argument('--large-ns', default='8192,12288,16384')
    ap.add_argument('--large-batches', default='1,4,12')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--modes', default='eager,compiled')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--precision', default='high', choices=['high', 'highest'],
                    help="global FP32 matmul precision; 'highest' = IEEE FP32 Grams (no TF32) for the Cholesky variants")
    ap.add_argument('--out', default=str(HERE.parent / 'reports/refresh_study_20261120/microbench_ext.json'))
    args = ap.parse_args()
    out_path = Path(args.out)
    fig_path = out_path.parent / ('refresh_cost_vs_dim_ext.pdf' if out_path.name == 'microbench_ext.json'
                                  else out_path.stem + '.pdf')
    if args.plot_only:
        make_figure(out_path, fig_path)
        return
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision(args.precision)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.recompile_limit = 512
    torch._dynamo.config.accumulated_recompile_limit = 8192
    torch.set_grad_enabled(False)
    methods = args.methods.split(',')
    modes = [m == 'compiled' for m in args.modes.split(',')]
    rows, inputs_meta = [], []
    env_start = mb.env_record()
    t_start = time.time()

    def dump(final=False):
        payload = dict(env=env_start, protocol=vars(args), flop_model={k: dict(coefficient_of_n3=v[0], description=v[1])
                                                                       for k, v in FLOP_MODEL.items()},
                       tflops_definition='flops_per_factor / (median_ms / batch); compiled and eager rows alike',
                       cholqr3_definition=__doc__.split('Shifted CholeskyQR3 (FP32)')[1].strip(),
                       inputs=inputs_meta, rows=rows, elapsed_s=time.time() - t_start, complete=final)
        if final:
            payload['env_end'] = mb.env_record()
            payload['crossover'] = crossovers(rows)
        tmp = out_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=True))
        tmp.replace(out_path)

    grid = mb.Bench(4000., 8, 40, 2)
    large = mb.Bench(3000., 4, 20, 1)
    run_cells(grid, [int(x) for x in args.grid_ns.split(',') if x], [int(x) for x in args.grid_batches.split(',')],
              methods, modes, rows, inputs_meta, dump, t_start)
    run_cells(large, [int(x) for x in args.large_ns.split(',') if x], [int(x) for x in args.large_batches.split(',')],
              methods, modes, rows, inputs_meta, dump, t_start)
    dump(final=True)
    try:
        make_figure(out_path, fig_path)
    except Exception as exc:
        print('figure failed', mb.err_text(exc), traceback.format_exc(), flush=True)
    print(json.dumps(crossovers(rows), indent=1))


if __name__ == '__main__':
    main()
