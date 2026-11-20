"""Accuracy of the Cholesky-QR variants against Householder QR (Part 1b follow-up).

Same synthetic inputs as refresh_study_microbench(_ext) (seed 1000 + n + 7*batch),
batch 1.  For each method the Householder reference is torch.linalg.qr of the
SAME sorted Z = C Q, using the method's own returned column order (production
backend settings, FP32 matmul 'high'), so near-tie sort differences cannot
masquerade as accuracy loss.  Column-wise angle after sign fixing, computed at
'highest'.  The BF16 floor is the angle between BF16(Q_householder) and
Q_householder.  Writes reports/refresh_study_20261120/microbench_ext_accuracy.json.
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import refresh_study_microbench as mb  # noqa: E402
import refresh_study_microbench_ext as ext  # noqa: E402
import torch  # noqa: E402
from bf16_soap import qr_basis  # noqa: E402
from refresh_study_common import highest, cholqr_basis  # noqa: E402

BF16 = torch.bfloat16


@torch.no_grad()
def angles(a, b):
    with highest():
        a, b = a.float(), b.float()
        dots = (a * b).sum(0)
        cos = (dots.abs() / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-30)).clamp(max=1)
        ang = torch.arccos(cos)
        rel = (a * torch.sign(dots) - b).norm() / math.sqrt(b.shape[0])
    finite = bool(torch.isfinite(ang).all())
    return dict(max_angle=float(ang.max()) if finite else float('nan'),
                median_angle=float(ang.median()) if finite else float('nan'),
                p99_angle=float(ang.quantile(.99)) if finite and ang.numel() <= 16_000_000 else float('nan'),
                rel_frob=float(rel) if finite else float('nan'))


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--method-precision', default='high', choices=['high', 'highest'],
                    help="FP32 matmul precision while running the methods ('highest' = IEEE FP32 Grams, no TF32)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')  # inputs and Householder reference as in the 'high' run
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_grad_enabled(False)
    rows = []
    t0 = time.time()
    for n in (256, 512, 768, 1024, 1536, 2048, 2304, 3072, 4096, 6144, 8192, 12288, 16384):
        c, q = mb.make_factor_inputs(n, 1, seed=1000 + n + 7)
        row = dict(n=n, batch=1)
        for name, fn in (('qr', lambda: qr_basis(c, q, BF16)), ('cholqr', lambda: cholqr_basis(c, q)),
                         ('cholqr3', lambda: ext.cholqr3_basis(c, q))):
            try:
                torch.set_float32_matmul_precision(args.method_precision)
                try:
                    out = fn()
                finally:
                    torch.set_float32_matmul_precision('high')
                new, order = out[0], out[1]
                z = torch.bmm(c.float(), q.float())  # production precision ('high')
                ordered = z.gather(-1, order.unsqueeze(-2).expand_as(z))
                ref = torch.linalg.qr(ordered).Q[0]
                del z, ordered
                e = angles(new[0], ref)
                e['gram_err'] = float(mb.gram_error(new)[0]) if torch.isfinite(new).all() else float('nan')
                e['nonfinite'] = not bool(torch.isfinite(new).all())
                if name == 'cholqr3':
                    e['chol_info_by_pass'] = out[2].flatten().tolist()
                    e['shift_over_norm2sq'] = 11. * (2 * n * n + n) * ext.U32
                if name == 'cholqr':
                    e['chol_info'] = int(out[2].flatten()[0])
                if name == 'qr':
                    e['bf16_floor'] = angles(ref.to(BF16), ref)
                row[name] = e
                del out, new, ref
            except Exception as exc:
                row[name] = dict(error=mb.err_text(exc))
            torch.cuda.empty_cache()
        rows.append(row)
        print(n, {k: (v.get('max_angle'), v.get('gram_err'), v.get('error')) for k, v in row.items() if isinstance(v, dict)},
              flush=True)
        del c, q
        torch.cuda.empty_cache()
    out = dict(env=mb.env_record(), rows=rows, elapsed_s=time.time() - t0, description=__doc__,
               method_precision=args.method_precision)
    suffix = '' if args.method_precision == 'high' else '_highest'
    p = HERE.parent / f'reports/refresh_study_20261120/microbench_ext_accuracy{suffix}.json'
    p.write_text(json.dumps(out, indent=1, allow_nan=True))
    print('wrote', p)


if __name__ == '__main__':
    main()
