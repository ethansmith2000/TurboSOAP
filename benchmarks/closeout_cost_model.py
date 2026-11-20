"""Close-out (2026-11-20): a calibrated per-layer cost model for SOAP refresh schedules.

Answers: for a Transformer of width d trained with N tokens per optimizer step, how much time goes to
forward+backward versus SOAP's ordinary step and its basis refreshes, and when an approximate (warm)
refresh can be cheaper than an exact QR refresh.

Architecture (the paper's 109M pattern): per layer, fused QKV (3d x d), attention out (d x d), fused SwiGLU in
(4d x d, hidden 2d) and SwiGLU out (d x 2d): 10 d^2 parameters and Kronecker factors of size d (x5), 2d, 3d, 4d,
so sum n^3 = 104 d^3.

Flop model (per layer, per optimizer step):
  forward+backward      60 d^2 N  +  6 T d N   (causal attention, sequence length T)
  ordinary SOAP step    240 d^3   (rotate gradient in/out of both bases, update both covariances)
  one QR refresh        104 d^3 * 14/3   (Z = CQ, geqrf, orgqr; the refresh study's convention)
  one warm refresh      104 d^3 * 14     (7 full n x n products per factor)

Rates are calibrated on the measured 109M runs (d = 768, N = 32,768, 12 layers + an untied-size LM head).
Refresh times use the refresh study's measured per-factor costs (compiled, batch 12; batch 4 where 12 was not
run), so they carry the real cuSOLVER-QR and BF16-matmul throughput at each factor size.

Inputs : reports/refresh_study_20261120/microbench.json, microbench_ext.json
Outputs: reports/closeout_20261120/cost_model.json and cost_model.md
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RS = ROOT / 'reports' / 'refresh_study_20261120'
OUT = ROOT / 'reports' / 'closeout_20261120'

T, VOCAB = 1024, 50257
WIDTHS = [256, 512, 768, 1024, 2048, 4096]
TOKENS = [32768, 1 << 20, 4 << 20]
SUM_N3 = 104            # sum of n^3 over a layer's factors, in units of d^3
QR_FLOPS = SUM_N3 * 14 / 3
WARM_FLOPS = SUM_N3 * 14
ORD_FLOPS = 240
FB_FLOPS_PER_TOKEN = 60  # times d^2


def factor_sizes(d):
    return [d] * 5 + [2 * d, 3 * d, 4 * d]


def per_factor_costs():
    cost = {}
    for row in json.loads((RS / 'microbench.json').read_text())['rows']:
        if row['ok'] and row['compiled'] and row['batch'] == 12:
            cost[(row['method'], row['n'])] = row['per_factor_median_ms']
    for row in json.loads((RS / 'microbench_ext.json').read_text())['rows']:
        key = (row['method'], row['n'])
        if (row.get('ok') and row.get('compiled') and row['method'] in ('qr', 'warm_bf16')
                and row['batch'] in (4, 12) and key not in cost):
            cost[key] = row['per_factor_median_ms']
    return cost


def calibrate():
    d, n_tok, layers = 768, 32768, 12
    measured = dict(fb_ms=291.0, ordinary_ms=10.0, warm_extra_ms=52.0, qr_extra_ms=606.0)
    fb = layers * (FB_FLOPS_PER_TOKEN * d * d + 6 * T * d) * n_tok + 6 * VOCAB * d * n_tok
    rates = dict(
        fb_tflops=fb / (measured['fb_ms'] * 1e-3) / 1e12,
        ordinary_tflops=layers * ORD_FLOPS * d ** 3 / (measured['ordinary_ms'] * 1e-3) / 1e12,
        warm_tflops=layers * WARM_FLOPS * d ** 3 / (measured['warm_extra_ms'] * 1e-3) / 1e12,
        qr_tflops=layers * QR_FLOPS * d ** 3 / (measured['qr_extra_ms'] * 1e-3) / 1e12,
    )
    return measured, rates


def main():
    cost = per_factor_costs()
    measured, rates = calibrate()
    r_fb, r_ord = rates['fb_tflops'] * 1e12, rates['ordinary_tflops'] * 1e12
    rows = []
    for n_tok in TOKENS:
        for d in WIDTHS:
            sizes = factor_sizes(d)
            if not all(('qr', n) in cost and ('warm_bf16', n) in cost for n in sizes):
                continue
            qr = sum(cost[('qr', n)] for n in sizes)
            warm = sum(cost[('warm_bf16', n)] for n in sizes)
            fb = (FB_FLOPS_PER_TOKEN * d * d + 6 * T * d) * n_tok / r_fb * 1e3
            ordinary = ORD_FLOPS * d ** 3 / r_ord * 1e3
            sched = dict(QR10=qr / 10, QR20=qr / 20, warm10_QR50=(4 * warm + qr) / 50, QR50=qr / 50)
            rows.append(dict(
                d=d, tokens=n_tok, fb_ms=fb, ordinary_ms=ordinary, refresh_ms=sched,
                warm_over_qr_time=warm / qr,
                qr_tflops=QR_FLOPS * d ** 3 / (qr * 1e-3) / 1e12,
                warm_tflops=WARM_FLOPS * d ** 3 / (warm * 1e-3) / 1e12,
                pct_of_fb={k: 100 * v / fb for k, v in dict(ordinary=ordinary, **sched).items()},
            ))
    # warm10/QR50 is cheaper than QRk iff (matmul rate / QR rate) > 12 / (50/k - 1)
    breakeven = {f'QR{k}': 12 / (50 / k - 1) for k in (10, 20, 25)}
    amortized = {k: v for k, v in dict(QR10=QR_FLOPS / 10, QR20=QR_FLOPS / 20, QR50=QR_FLOPS / 50,
                                       warm10_QR50=(4 * WARM_FLOPS + QR_FLOPS) / 50).items()}
    result = dict(calibration_measured_109M=measured, calibrated_rates=rates,
                  flops_per_layer_d3=dict(ordinary=ORD_FLOPS, qr_event=QR_FLOPS, warm_event=WARM_FLOPS,
                                          amortized_per_step=amortized),
                  breakeven_matmul_over_qr_rate=breakeven, rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cost_model.json').write_text(json.dumps(result, indent=1))

    lines = ['# Calibrated per-layer cost model (generated by benchmarks/closeout_cost_model.py)', '',
             'Calibrated rates (TFLOP/s) at 109M: ' + ', '.join(f'{k} {v:.1f}' for k, v in rates.items()), '',
             'Amortized refresh flops per step (units of d^3 per layer): '
             + ', '.join(f'{k} {v:.0f}' for k, v in amortized.items()), '',
             'warm10/QR50 is cheaper than QRk only if QR runs below 1/x of the warm step\'s matmul rate: '
             + ', '.join(f'{k}: x = {v:.1f}' for k, v in breakeven.items()) + '; never against QR50.', '',
             '| d | tokens/step | fwd+bwd ms/layer | ordinary | QR10 | QR20 | warm10/QR50 | QR50 '
             '| QR TFLOP/s | warm TFLOP/s | warm/QR time |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        p = r['pct_of_fb']
        lines.append(f"| {r['d']} | {r['tokens']} | {r['fb_ms']:.1f} | {p['ordinary']:.1f}% | {p['QR10']:.1f}% "
                     f"| {p['QR20']:.1f}% | {p['warm10_QR50']:.1f}% | {p['QR50']:.1f}% | {r['qr_tflops']:.1f} "
                     f"| {r['warm_tflops']:.0f} | {r['warm_over_qr_time']:.3f} |")
    lines += ['', 'Percentages are of forward+backward time per layer (LM head excluded). If the optimizer work is '
              'split across data-parallel ranks, use the global batch for tokens/step; if every rank repeats it, '
              'use the per-rank batch.']
    (OUT / 'cost_model.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
