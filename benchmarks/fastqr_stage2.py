"""Stage 2 of the fast-QR study (10k calls, seeds 204/205): gates, deltas against the cached production runs, the
pre-registered decision rule (REPORT.md section 6), C2+C1 fallbacks/orthogonality, and a figure.

Writes reports/fastqr_20261120/stage2.json, fastqr_stage2_delta.{pdf,png} and copies run evidence. Reads only.

  python benchmarks/fastqr_stage2.py            # the four stage-2 bundles
  python benchmarks/fastqr_stage2.py --smoke    # code-path check on the stage-1 bundle (2k calls, seed 204)
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmarks')]
OUT = ROOT / 'reports/fastqr_20261120'
DATA = Path('/workspace/data/fastqr_20261120')
STAGE1 = DATA / 'stage1_seed204_c2000'
ARMS = ('bf16_hh_qr10', 'bf16_w10_cq50_ns1')
LABEL = {'bf16_hh_qr10': 'BF16 Householder QR10', 'bf16_w10_cq50_ns1': 'C2+C1 (warm10 ns=1 / CholQR50)',
         'fp32_qr10': 'FP32-state stock QR10 (cached)', 'fp32_warm': 'FP32-state stock warm (cached)'}
CACHED_PROD = {204: ROOT / 'reports/bf16_replicate_20261027_seed204/runs/bf16/result.json',
               205: ROOT / 'reports/bf16_full_horizon_20261026_seed205/runs/bf16/result.json'}
FACTOR_NAME = {((2304, 768), 'L'): 'QKV-left (2304)', ((2304, 768), 'R'): 'QKV-right (768)',
               ((768, 768), 'L'): 'attn-out-left (768)', ((768, 768), 'R'): 'attn-out-right (768)',
               ((3072, 768), 'L'): 'SwiGLU-in-left (3072)', ((3072, 768), 'R'): 'SwiGLU-in-right (768)',
               ((768, 1536), 'L'): 'SwiGLU-out-left (768)', ((768, 1536), 'R'): 'SwiGLU-out-right (1536)'}
PER_EVENT = 12  # factors of each (shape, side) per refresh event: 12 layers
# Pre-registered thresholds (REPORT.md section 6, written before results)
SURVIVES, REMOVED, C2C1_TOL, GATE_SPECTRAL = -0.007, -0.003, 0.003, 0.07


def jl(path):
    return json.loads(Path(path).read_text())


def ces(validation):
    return {v['call']: v['ce'] for v in validation}


def fp32_reference(seed):
    if seed == 204:
        runs = {r['name']: r for r in jl(ROOT / 'reports/owt_restart_20261026/comparison.json')['runs']}
        return dict(fp32_qr10=ces(runs['primary_stock_qr']['validation']),
                    fp32_warm=ces(runs['primary_stock_warm']['validation']))
    if seed == 205:
        return dict(fp32_qr10=ces(jl(ROOT / 'reports/bf16_full_horizon_20261026_seed205/runs/qr10/result.json')
                                  ['validation']))
    return {}


def run_record(bundle, arm):
    folder = bundle / 'runs' / arm
    r = jl(folder / 'result.json')
    events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
    updates = {e['call']: e for e in events if e['event'] == 'update'}
    calls = max(updates)
    clean = [updates[c] for c in range(101, calls + 1)]
    windows = {}
    for lo in range(0, calls, 1000):
        xs = [updates[c] for c in range(max(lo + 1, 101), lo + 1001) if c in updates]
        windows[f'{lo + 1}-{lo + 1000}'] = dict(step_ms=st.mean(e['step_ms'] for e in xs),
                                                 fb_ms=st.mean(e['fb_ms'] for e in xs),
                                                 optimizer_ms=st.mean(e['optimizer_ms'] for e in xs))
    by_kind = {}
    for k in ('ordinary', 'warm', 'qr'):
        xs = sorted(e['optimizer_ms'] for e in clean if e['kind'] == k)
        if xs:
            by_kind[k] = dict(n=len(xs), mean_ms=st.mean(xs), median_ms=xs[len(xs) // 2])
    geometry = jl(folder / 'geometry.json')
    stats = jl(folder / 'fastqr_stats.json')
    gates = {name: any(e['event'] == name and e.get('passed') for e in events)
             for name in ('startup_match', 'initial_validation_match', 'initial_backward_repeat', 'final_state_finite')}
    audit = jl(bundle / 'completion_audit.json') if (bundle / 'completion_audit.json').exists() else None
    return dict(arm=arm, calls=calls, validation=ces(r['validation']), losses={c: updates[c]['loss'] for c in updates},
                mean_step_ms=st.mean(e['step_ms'] for e in clean), mean_fb_ms=st.mean(e['fb_ms'] for e in clean),
                mean_optimizer_ms=st.mean(e['optimizer_ms'] for e in clean), optimizer_by_kind=by_kind,
                windows_1k=windows, elapsed_seconds=r['elapsed_seconds'], peak_gpu_gib=r['peak_gpu_bytes'] / 2**30,
                max_normalized_gram=r['max_normalized_gram'],
                spectral_gram_by_call={g['call']: g.get('max_spectral_gram') for g in geometry},
                normalized_gram_by_call={g['call']: g.get('max_normalized_gram') for g in geometry},
                refresh_counts=stats.get('refresh_counts'), refresh_log=stats.get('refresh_log', []),
                fallback_log=stats.get('fallback_log', []), gates=gates,
                completion_audit=audit and audit['status'], dynamo_graph_breaks=r['dynamo_counters'].get('graph_break', {}))


def fallback_tables(rec):
    log, refresh = rec['fallback_log'], rec['refresh_log']
    if not refresh:
        return None
    win = defaultdict(lambda: [0, 0])
    for row in refresh:
        w = (row['call'] - 1) // 1000
        win[w][0] += row['fallbacks']
        win[w][1] += row['factors']
    events = len(refresh)
    per_factor = defaultdict(int)
    for f in log:
        per_factor[(tuple(f['shape']), f['side'])] += 1
    return dict(total=dict(fallbacks=sum(r['fallbacks'] for r in refresh), refreshes=sum(r['factors'] for r in refresh),
                           events=events, events_with_fallback=sum(r['fallbacks'] > 0 for r in refresh),
                           max_per_event=max(r['fallbacks'] for r in refresh)),
                by_1k_window={f'{w * 1000 + 1}-{w * 1000 + 1000}': dict(fallbacks=a, refreshes=b, rate=a / b)
                              for w, (a, b) in sorted(win.items())},
                by_factor={FACTOR_NAME[k]: dict(fallbacks=n, rate=n / (events * PER_EVENT))
                           for k, n in sorted(per_factor.items(), key=lambda t: -t[1])})


def prefix_gate(rec):
    """Seed 204: the stage-2 run must equal its stage-1 run bit for bit through call 2000."""
    s1 = run_record(STAGE1, rec['arm'])
    upto = s1['calls']
    out = dict(validation_equal=all(rec['validation'].get(c) == v for c, v in s1['validation'].items()),
               losses_equal=all(rec['losses'].get(c) == s1['losses'][c] for c in range(1, upto + 1)))
    if s1['refresh_log']:
        a = [r for r in rec['refresh_log'] if r['call'] <= upto]
        b = s1['refresh_log']
        out.update(refresh_log_equal=a == b,
                   fallback_log_equal=[f for f in rec['fallback_log'] if f['call'] <= upto] == s1['fallback_log'],
                   fallbacks_through_2000=sum(r['fallbacks'] for r in a), refreshes_through_2000=sum(r['factors'] for r in a))
    out['passed'] = all(v for k, v in out.items() if k.endswith('_equal'))
    return out


def classify(d):
    if d is None:
        return None
    return 'survives' if d <= SURVIVES else 'removed' if d >= REMOVED else 'between'


def seed_analysis(seed, records):
    prod = ces(jl(CACHED_PROD[seed])['validation'])
    ref = fp32_reference(seed)
    calls = sorted(c for c in prod if c > 0 and all(c in r['validation'] for r in records.values()))
    deltas = {arm: {c: rec['validation'][c] - prod[c] for c in calls} for arm, rec in records.items()}
    for name, v in ref.items():
        deltas[name] = {c: v[c] - prod[c] for c in calls if c in v}
    last = max(calls)
    late = [c for c in calls if c >= 5000]
    out = dict(seed=seed, calls=calls, cached_production=dict(path=str(CACHED_PROD[seed].relative_to(ROOT)),
                                                               ce={c: prod[c] for c in calls}),
               ce={arm: {c: rec['validation'][c] for c in calls} for arm, rec in records.items()},
               fp32_reference={k: {c: v[c] for c in calls if c in v} for k, v in ref.items()},
               delta_vs_cached_production=deltas,
               late_window_mean_delta={k: (st.mean(v[c] for c in late) if late and all(c in v for c in late) else None)
                                       for k, v in deltas.items()})
    if 'bf16_hh_qr10' in deltas and last >= 10000:
        d = deltas['bf16_hh_qr10'][last]
        out['qr10_decision'] = dict(delta_10k=d, category=classify(d),
                                    late_mean=out['late_window_mean_delta']['bf16_hh_qr10'],
                                    late_category=classify(out['late_window_mean_delta']['bf16_hh_qr10']),
                                    fp32_qr10_delta_10k=deltas.get('fp32_qr10', {}).get(last))
    if 'bf16_w10_cq50_ns1' in deltas and last >= 10000:
        d = deltas['bf16_w10_cq50_ns1'][last]
        out['c2c1_endpoint'] = dict(delta_10k=d, keeps_production=abs(d) <= C2C1_TOL)
    return out


def figure(analyses, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                         'axes.spines.right': False})
    style = {'bf16_hh_qr10': dict(color='#2a78d6', ls='-', fill=True),
             'fp32_qr10': dict(color='#2a78d6', ls=(0, (4, 2)), fill=False),
             'bf16_w10_cq50_ns1': dict(color='#1baf7a', ls='-', fill=True)}
    fig, axes = plt.subplots(1, len(analyses), figsize=(5.4 * len(analyses), 3.8), constrained_layout=True,
                             sharey=True, squeeze=False)
    short = {'bf16_hh_qr10': 'BF16 QR10', 'fp32_qr10': 'FP32 QR10', 'bf16_w10_cq50_ns1': 'C2+C1'}
    for ax, a in zip(axes[0], analyses):
        xs_all = [c for c in a['calls'] if c >= 1500]
        ax.axhspan(-.003, .003, color='#e8e7e3', lw=0, zorder=0)
        ax.axhline(0, color='#8a8984', lw=.9, zorder=1)
        ends = []
        for key, s in style.items():
            d = a['delta_vs_cached_production'].get(key)
            if not d:
                continue
            xs = [c for c in xs_all if c in d]
            ys = [d[c] for c in xs]
            ax.plot(xs, ys, color=s['color'], ls=s['ls'], lw=2, zorder=3)
            ax.scatter(xs, ys, s=30, zorder=4, linewidths=1.4, facecolors=s['color'] if s['fill'] else '#fcfcfb',
                       edgecolors='#fcfcfb' if s['fill'] else s['color'])
            ends.append((ys[-1], xs[-1], key))
        last = -1e9
        for y, x, key in sorted(ends):  # direct end labels, stacked at least 0.0011 CE apart
            yl = max(y, last + .0011)
            last = yl
            ax.annotate(f'{short[key]}  {y:+.4f}', (x, y), xytext=(x + 250, yl), textcoords='data', fontsize=7,
                        color='#52514e', va='center',
                        arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=.6, shrinkA=0, shrinkB=3))
        ax.set_title(f'seed {a["seed"]}', fontsize=9, loc='left', color='#0b0b0b')
        ax.set_xlabel('call')
        ax.grid(True, color='#e8e7e3', lw=.6, zorder=0)
        ax.set_xlim(xs_all[0] - 200, xs_all[-1] + 2300)
        ax.set_xticks([t for t in (2000, 4000, 6000, 8000, 10000) if t <= xs_all[-1]])
    axes[0][0].set_ylabel('CE − cached BF16 warm10/QR50 CE (0 line)')
    handles = [plt.Line2D([], [], color=s['color'], ls=s['ls'], lw=2, marker='o', markersize=5,
                          markerfacecolor=s['color'] if s['fill'] else '#fcfcfb', label=LABEL[k])
               for k, s in style.items()]
    handles.append(plt.matplotlib.patches.Patch(color='#e8e7e3', label='±0.003: production seed spread at 10k'))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle('109M OWT, 10,000 calls: CE relative to the cached production run of the same seed '
                 '(below 0 = better)', fontsize=9.5, x=.01, ha='left', color='#0b0b0b')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ('losses', 'refresh_log', 'fallback_log')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        plan = {204: {arm: STAGE1 for arm in ARMS}}
    else:
        plan = {seed: {arm: DATA / f'stage2_seed{seed}_{arm}_c10000' for arm in ARMS} for seed in (204, 205)}
    out = dict(protocol='REPORT.md section 6 (fixed before results)',
               thresholds=dict(survives_if_delta_le=SURVIVES, removed_if_delta_ge=REMOVED, c2c1_tolerance=C2C1_TOL,
                               spectral_gate=GATE_SPECTRAL), seeds={})
    analyses = []
    for seed, bundles in plan.items():
        records = {arm: run_record(b, arm) for arm, b in bundles.items() if (b / 'runs' / arm / 'result.json').exists()}
        if not records:
            continue
        seed_out = dict(runs={arm: strip(r) for arm, r in records.items()},
                        fallbacks={arm: fallback_tables(r) for arm, r in records.items()},
                        spectral_gate={arm: max(v for v in r['spectral_gram_by_call'].values() if v is not None)
                                       < GATE_SPECTRAL for arm, r in records.items()})
        if seed == 204:
            seed_out['prefix_gate'] = {arm: prefix_gate(r) for arm, r in records.items()}
        an = seed_analysis(seed, records)
        seed_out['analysis'] = an
        analyses.append(an)
        out['seeds'][seed] = seed_out
    if not a.smoke:
        from fastqr_report import copy_evidence
        for bundles in plan.values():
            for b in bundles.values():
                if (b / 'completion_audit.json').exists():
                    copy_evidence(b)
    dest = OUT / ('stage2_smoke.json' if a.smoke else 'stage2.json')
    dest.write_text(json.dumps(out, indent=1, default=str))
    figure(analyses, OUT / ('fastqr_stage2_delta_smoke' if a.smoke else 'fastqr_stage2_delta'))
    for seed, s in out['seeds'].items():
        an = s['analysis']
        print(f'seed {seed}:', json.dumps(dict(prefix=s.get('prefix_gate'), spectral=s['spectral_gate'],
                                               qr10=an.get('qr10_decision'), c2c1=an.get('c2c1_endpoint'),
                                               delta_last={k: v[max(an['calls'])] for k, v in
                                                           an['delta_vs_cached_production'].items()
                                                           if max(an['calls']) in v}), default=str))


if __name__ == '__main__':
    main()
