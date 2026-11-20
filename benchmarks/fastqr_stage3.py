"""Stage 3 of the fast-QR study: BF16 QR50 to 10k (seeds 204/205) and the FP32-storage QR10 screen (seed 204, 2k).
Gates, the two decision rules fixed in REPORT.md section 8 before launch, and a figure.

Writes reports/fastqr_20261120/stage3.json, fastqr_stage3.{pdf,png} and copies run evidence. Reads only.

  python benchmarks/fastqr_stage3.py           # the three stage-3 bundles
  python benchmarks/fastqr_stage3.py --smoke   # code-path check with stage-2/stage-1 runs standing in
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmarks')]
from fastqr_stage2 import CACHED_PROD, DATA, OUT, ces, fp32_reference, jl, run_record  # noqa: E402

CONTROL_QR50 = ROOT / 'reports/bf16_refresh_controls_20261027/runs/bf16_qr50'
LEARN = ROOT / 'reports/bf16_learning_20261026/runs'
# Rule A (REPORT.md section 8): Delta = CE(BF16 QR50) - CE(cached BF16 warm10/QR50) at call 10000, per seed
A_NULL, A_EFFECT = .003, .006
# Rule B: recovery fractions at call 2000 (secondary 1500); both must agree
B_REF = {2000: dict(bf16_qr10=3.99809, eager_qr10=3.98591, eager_warm=3.99575, fp32s_warm=3.99156, bf16_warm=3.99394),
         1500: dict(bf16_qr10=4.12314, eager_qr10=4.10837, eager_warm=4.12819, fp32s_warm=4.12808, bf16_warm=4.12237)}
B_HIGH, B_LOW = 2 / 3, 1 / 3


def losses(events_path):
    return {e['call']: e['loss'] for e in map(json.loads, Path(events_path).read_text().splitlines())
            if e['event'] == 'update'}


def first_difference(a, b):
    return next((c for c in sorted(set(a) & set(b)) if a[c] != b[c]), None)


def first_refresh_gate(rec, cached_events):
    d = first_difference(rec['losses'], losses(cached_events))
    return dict(reference=str(Path(cached_events).relative_to(ROOT)), first_differing_call=d,
                passed=d is None or d > 11)


def qr50_prefix_gate(rec):
    ref_loss = losses(CONTROL_QR50 / 'events.jsonl')
    ref_ce = ces(jl(CONTROL_QR50 / 'result.json')['validation'])
    return dict(reference=str(CONTROL_QR50.relative_to(ROOT)),
                losses_equal_1_2000=all(rec['losses'].get(c) == v for c, v in ref_loss.items()),
                validation_equal_through_2000=all(rec['validation'].get(c) == v for c, v in ref_ce.items()),
                first_differing_call=first_difference(rec['losses'], ref_loss),
                passed=all(rec['losses'].get(c) == v for c, v in ref_loss.items())
                and all(rec['validation'].get(c) == v for c, v in ref_ce.items()))


def classify_a(d):
    return 'no_resolvable_value' if d <= A_NULL else 'warm_improves' if d >= A_EFFECT else 'between'


def rule_a(seed, rec, qr10_rec):
    prod = ces(jl(CACHED_PROD[seed])['validation'])
    calls = sorted(c for c in prod if c > 0 and c in rec['validation'])
    delta = {c: rec['validation'][c] - prod[c] for c in calls}
    late = [c for c in calls if c >= 5000]
    out = dict(seed=seed, calls=calls, ce={c: rec['validation'][c] for c in calls},
               cached_production={c: prod[c] for c in calls}, delta_vs_cached_production=delta,
               late_mean_delta=st.mean(delta[c] for c in late) if late else None)
    if qr10_rec:
        out['context_qr50_minus_bf16_qr10'] = {c: rec['validation'][c] - qr10_rec['validation'][c] for c in calls
                                               if c in qr10_rec['validation']}
        out['bf16_qr10_delta_vs_cached_production'] = {c: qr10_rec['validation'][c] - prod[c] for c in calls
                                                       if c in qr10_rec['validation']}
    out['fp32_qr10_delta_vs_cached_production'] = {c: v - prod[c] for c, v in
                                                   fp32_reference(seed).get('fp32_qr10', {}).items() if c in prod and c > 0}
    last = max(calls)
    if last >= 10000:
        out['rule_a'] = dict(delta_10k=delta[last], category=classify_a(delta[last]),
                             late_mean=out['late_mean_delta'], late_category=classify_a(out['late_mean_delta']))
    return out


def rule_b(rec):
    out = {}
    for call, r in B_REF.items():
        x = rec['validation'].get(call)
        if x is None:
            continue
        f = (r['bf16_qr10'] - x) / (r['bf16_qr10'] - r['eager_qr10'])
        d_bf16, d_eager = r['bf16_qr10'] - r['bf16_warm'], r['eager_qr10'] - r['eager_warm']
        g = (d_bf16 - (x - r['fp32s_warm'])) / (d_bf16 - d_eager)
        verdict = ('storage' if f >= B_HIGH and g >= B_HIGH else
                   'compiled_path_or_bf16_compute' if f <= B_LOW and g <= B_LOW else 'partial_or_unresolved')
        out[call] = dict(x=x, f=f, g=g, refresh_effect_in_fp32_storage=x - r['fp32s_warm'],
                         refresh_effect_bf16=d_bf16, refresh_effect_eager=d_eager, verdict=verdict)
    return out


def figure(a_rows, b, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                         'axes.spines.right': False})
    blue, orange = '#2a78d6', '#eb6834'
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    axes[1].sharey(axes[0])
    series = [('fp32_qr10_delta_vs_cached_production', 'FP32 QR10', blue, (0, (4, 2)), False),
              ('bf16_qr10_delta_vs_cached_production', 'BF16 QR10', blue, '-', True),
              ('delta_vs_cached_production', 'BF16 QR50', orange, '-', True)]  # new run drawn last, on top
    for ax, a in zip(axes[:2], a_rows):
        ax.axhspan(-A_NULL, A_NULL, color='#e8e7e3', lw=0, zorder=0)
        ax.axhline(0, color='#8a8984', lw=.9, zorder=1)
        ax.axhline(A_EFFECT, color='#c3c2b7', lw=.8, ls=(0, (2, 2)), zorder=1)
        ends = []
        for key, name, color, ls, fill in series:
            d = a.get(key) or {}
            xs = [c for c in sorted(d) if c >= 1500]
            if not xs:
                continue
            ys = [d[c] for c in xs]
            ax.plot(xs, ys, color=color, ls=ls, lw=2, zorder=3)
            ax.scatter(xs, ys, s=30, zorder=4, linewidths=1.4, facecolors=color if fill else '#fcfcfb',
                       edgecolors='#fcfcfb' if fill else color)
            ends.append((ys[-1], xs[-1], name))
        last = -1e9
        for y, x, name in sorted(ends):
            yl = max(y, last + .0011)
            last = yl
            ax.annotate(f'{name}  {y:+.4f}', (x, y), xytext=(x + 250, yl), textcoords='data', fontsize=7,
                        color='#52514e', va='center',
                        arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=.6, shrinkA=0, shrinkB=3))
        ax.set_title(f'seed {a["seed"]}: CE − cached BF16 warm10/QR50', fontsize=9, loc='left', color='#0b0b0b')
        ax.set_xlabel('call')
        ax.grid(True, color='#e8e7e3', lw=.6, zorder=0)
        xmax = max(a['calls'])
        ax.set_xlim(1300, xmax + 2300)
        ax.set_xticks([t for t in (2000, 4000, 6000, 8000, 10000) if t <= xmax])
    axes[0].set_ylabel('validation CE difference')
    ax = axes[2]
    r = B_REF[2000]
    impl = ['FP32 eager', 'compiled\nFP32 storage', 'compiled\nBF16 storage']
    x = [0, 1, 2]
    qr10 = [r['eager_qr10'], b.get(2000, {}).get('x'), r['bf16_qr10']]
    warm = [r['eager_warm'], r['fp32s_warm'], r['bf16_warm']]
    ax.plot(x, warm, color='#1baf7a', lw=2, marker='o', markersize=6, markeredgecolor='#fcfcfb', label='warm10/QR50')
    xs = [i for i, v in zip(x, qr10) if v is not None]
    ax.plot(xs, [qr10[i] for i in xs], color=blue, lw=2, marker='o', markersize=6, markeredgecolor='#fcfcfb',
            label='QR10')
    if b.get(2000):
        ax.scatter([1], [b[2000]['x']], s=90, facecolors='none', edgecolors=blue, linewidths=1.4, zorder=5)
        ax.annotate(f'new: {b[2000]["x"]:.4f}\nindicative only:\nfirst-refresh gate failed\n(run-to-run nondeterministic)',
                    (1, b[2000]['x']), xytext=(10, 4), textcoords='offset points', fontsize=6.6, color='#52514e',
                    va='bottom')
    ax.set_xticks(x, impl)
    ax.set_xlim(-.4, 2.4)
    ax.set_title('seed 204, CE at call 2000 by implementation', fontsize=9, loc='left', color='#0b0b0b')
    ax.grid(True, axis='y', color='#e8e7e3', lw=.6, zorder=0)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2, marker='o', markersize=5,
                          markerfacecolor=c if f else '#fcfcfb', label=n) for _, n, c, ls, f in series]
    handles.append(plt.matplotlib.patches.Patch(color='#e8e7e3', label=f'±{A_NULL}: null band (rule A)'))
    handles.append(plt.Line2D([], [], color='#c3c2b7', ls=(0, (2, 2)), lw=.8, label=f'+{A_EFFECT}: effect line (rule A)'))
    fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False, fontsize=7.5)
    fig.suptitle('109M OWT: BF16 QR50 to 10,000 calls (rule A) and the FP32-storage QR10 screen (rule B)',
                 fontsize=9.5, x=.01, ha='left', color='#0b0b0b')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ('losses', 'refresh_log', 'fallback_log')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    s2 = {seed: DATA / f'stage2_seed{seed}_bf16_hh_qr10_c10000' for seed in (204, 205)}
    if a.smoke:  # stand-ins: stage-2 QR10 for QR50, stage-1 QR10 for the FP32-storage arm
        qr50 = {seed: (s2[seed], 'bf16_hh_qr10') for seed in (204, 205)}
        fp32 = (DATA / 'stage1_seed204_c2000', 'bf16_hh_qr10')
    else:
        qr50 = {seed: (DATA / f'stage3_seed{seed}_bf16_hh_qr50_c10000', 'bf16_hh_qr50') for seed in (204, 205)}
        fp32 = (DATA / 'stage3_seed204_fp32s_hh_qr10_c2000', 'fp32s_hh_qr10')
    out = dict(protocol='REPORT.md section 8 (fixed before launch)',
               thresholds=dict(rule_a=dict(null_le=A_NULL, effect_ge=A_EFFECT),
                               rule_b=dict(storage_if_f_and_g_ge=B_HIGH, path_if_f_and_g_le=B_LOW, references=B_REF)),
               qr50={}, fp32_storage_qr10=None)
    a_rows = []
    for seed, (bundle, arm) in qr50.items():
        if not (bundle / 'runs' / arm / 'result.json').exists():
            continue
        rec = run_record(bundle, arm)
        qr10 = run_record(s2[seed], 'bf16_hh_qr10') if (s2[seed] / 'runs/bf16_hh_qr10/result.json').exists() else None
        gates = dict(in_run=rec['gates'], completion_audit=rec['completion_audit'],
                     graph_breaks=rec['dynamo_graph_breaks'],
                     first_refresh=first_refresh_gate(rec, CACHED_PROD[seed].parent / 'events.jsonl'))
        if seed == 204:
            gates['qr50_control_prefix'] = qr50_prefix_gate(rec)
        row = rule_a(seed, rec, qr10)
        a_rows.append(row)
        out['qr50'][seed] = dict(run=strip(rec), gates=gates, analysis=row,
                                 max_spectral_gram=max(v for v in rec['spectral_gram_by_call'].values() if v is not None))
    cats = [out['qr50'][s]['analysis'].get('rule_a', {}).get('category') for s in out['qr50']]
    if len(cats) == 2 and all(cats):
        out['rule_a_verdict'] = cats[0] if cats[0] == cats[1] and cats[0] != 'between' else 'unresolved'
    bundle, arm = fp32
    b = {}
    if (bundle / 'runs' / arm / 'result.json').exists():
        rec = run_record(bundle, arm)
        b = rule_b(rec)
        out['fp32_storage_qr10'] = dict(
            run=strip(rec), rule_b=b,
            gates=dict(in_run=rec['gates'], completion_audit=rec['completion_audit'],
                       graph_breaks=rec['dynamo_graph_breaks'],
                       first_refresh=first_refresh_gate(rec, LEARN / 'compiled_fp32/events.jsonl')))
    if not a.smoke:
        from fastqr_report import copy_evidence
        for bundle in [v[0] for v in qr50.values()] + [fp32[0]]:
            if (bundle / 'completion_audit.json').exists():
                copy_evidence(bundle)
    suffix = '_smoke' if a.smoke else ''
    (OUT / f'stage3{suffix}.json').write_text(json.dumps(out, indent=1, default=str))
    if a_rows or b:
        figure(a_rows, b, OUT / f'fastqr_stage3{suffix}')
    summary = dict(rule_a_verdict=out.get('rule_a_verdict'),
                   rule_a={s: v['analysis'].get('rule_a') for s, v in out['qr50'].items()},
                   gates_qr50={s: {k: (g.get('passed') if isinstance(g, dict) and 'passed' in g else g)
                                   for k, g in v['gates'].items()} for s, v in out['qr50'].items()},
                   rule_b=b, gates_fp32=out['fp32_storage_qr10'] and out['fp32_storage_qr10']['gates'])
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
