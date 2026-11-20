"""Collect the fast-QR study outputs into reports/fastqr_20261120 (summary.json, tables.md, figures, run evidence).

Inputs (any that exist): timing.json, unit_*.json, compiled_ieee_check.json, trained_d*_*.json in the report
directory, and a training bundle's comparison.json (``--bundle``). Writes nothing outside the report directory.
"""
import argparse
import json
import math
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fastqr_20261120'
REF = dict(ordinary=10.14, warm=61.61, qr=601.18)
LABEL = {
    'bf16_w10_qr50': 'warm10/QR50 (production)',
    'bf16_hh_qr10': 'Householder QR10',
    'bf16_cq_qr10': 'CholQR10 (C3)',
    'bf16_cq_qr20': 'CholQR20 (C1-20)',
    'bf16_w10_cq50_ns1': 'warm10 ns=1 / CholQR50 (C2+C1)',
    'bf16_w10_cq50': 'warm10 / CholQR50 (C1)',
    'bf16_w10_qr50_ns1': 'warm10 ns=1 / QR50 (C2)',
    'cached_bf16_warm10_qr50': 'paper warm10/QR50 (cached)',
    'bf16_qr20': 'paper Householder QR20 (cached)',
    'bf16_qr50': 'paper Householder QR50 (cached)',
}
FAMILY = {  # 1 Householder-every-k, 2 CholQR-every-k, 3 warm-based
    'bf16_hh_qr10': 0, 'bf16_qr20': 0, 'bf16_qr50': 0, 'bf16_cq_qr10': 1, 'bf16_cq_qr20': 1,
    'bf16_w10_qr50': 2, 'cached_bf16_warm10_qr50': 2, 'bf16_w10_cq50_ns1': 2, 'bf16_w10_cq50': 2,
    'bf16_w10_qr50_ns1': 2,
}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']  # reference categorical slots 1-3 (validated all-pairs)
FAMILY_NAMES = ['Householder QR every k ages', 'CholQR3 (C1) every k ages', 'warm-step schedules']


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------- timing
def timing_summary(t):
    cells = t['event_regions']['cells']

    def get(arm, ev, key='wall'):
        return cells[f'{arm}/{ev}'][key]['median_ms']
    rows = []
    for arm, ev, label in [('production', 'ordinary', 'ordinary (production)'),
                           ('production', 'warm', 'warm, 2 NS (production)'),
                           ('production', 'qr', 'Householder QR (production)'),
                           ('fast', 'ordinary', 'ordinary (FastRefreshSOAP)'),
                           ('fast', 'warm', 'warm, 1 NS (C2)'),
                           ('fast', 'qr', 'CholQR3 QR event (C1)')]:
        ref = REF.get(ev) if arm == 'production' else None
        rows.append(dict(arm=arm, event=ev, label=label, wall_ms=get(arm, ev), cuda_ms=get(arm, ev, 'cuda'),
                         matrix_only_cuda_ms=cells[f'{arm}/{ev}']['matrix_only_cuda']['median_ms'],
                         p25_wall=cells[f'{arm}/{ev}']['wall']['p25_ms'], p75_wall=cells[f'{arm}/{ev}']['wall']['p75_ms'],
                         ref_109M_ms=ref, ratio_to_ref=get(arm, ev) / ref if ref else None))
    o, w, q = get('production', 'ordinary'), get('production', 'warm'), get('production', 'qr')
    w1, cq = get('fast', 'warm'), get('fast', 'qr')
    sched = {
        'warm10/QR50 (production)': (45 * o + 4 * w + q) / 50,
        'Householder QR10': (9 * o + q) / 10,
        'Householder QR20': (19 * o + q) / 20,
        'CholQR10 (C3)': (9 * o + cq) / 10,
        'CholQR20 (C1-20)': (19 * o + cq) / 20,
        'warm10/CholQR50 (C1)': (45 * o + 4 * w + cq) / 50,
        'warm10 ns=1/QR50 (C2)': (45 * o + 4 * w1 + q) / 50,
        'warm10 ns=1/CholQR50 (C2+C1)': (45 * o + 4 * w1 + cq) / 50,
    }
    return dict(events=rows, amortized_optimizer_ms_per_call=sched,
                refresh_ms_per_call={k: v - o for k, v in sched.items()},
                cholqr_timed_fallbacks=t['event_regions']['cholqr_timed_fallbacks'],
                cholqr_timed_factor_refreshes=t['event_regions']['cholqr_timed_factor_refreshes'],
                primitives=t.get('primitives'), env=t['env'])


# ----------------------------------------------------------------------------- trained-factor checks
def trained_summary(paths):
    out = {}
    for p in paths:
        d = load(p)
        if not d:
            continue
        rows, chains = d['rows'], d['chains']
        ok = [r for r in rows if r['ok']]
        by_n = {}
        for r in rows:
            by_n.setdefault(r['n'], []).append(r)
        fails = [dict(factor=r['factor'], n=r['n'], side=r['side'], age=r['age'], failure=r.get('failure')) for r in rows if not r['ok']]
        w = d['width']
        hh = st.mean(c['hh_late_interval'] for c in chains)
        cq = st.mean(c['cq_late_interval'] for c in chains)
        out[f"d{w}_{d['device']}"] = dict(
            width=w, device=d['device'], factor_ages=len(rows), factors=len(chains),
            fail_shift_1e6=len(fails), fail_shift_1e7=sum(not r.get('ok_shift_1e7', True) for r in rows),
            failures_1e6=fails, failures_1e7=[dict(n=r['n'], age=r['age'], factor=r['factor']) for r in rows if not r.get('ok_shift_1e7', True)],
            failures_by_n={n: f'{sum(not r["ok"] for r in rs)}/{len(rs)}' for n, rs in sorted(by_n.items())},
            angle_max_all_columns=max(r['angle_max'] for r in ok),
            angle_max_first_half=max(r['angle_max_first_half'] for r in ok),
            angle_median=st.median(r['angle_median'] for r in ok),
            bf16_floor_angle_min=min(r['bf16_floor_angle_max'] for r in ok),
            bf16_floor_angle_max=max(r['bf16_floor_angle_max'] for r in ok),
            prefix_dist_n8_max=max(r['prefix_dist_n8'] for r in ok),
            bf16_prefix_dist_n8_median=st.median(r['bf16_prefix_dist_n8'] for r in ok),
            bf16_prefix_dist_n8_max=max(r['bf16_prefix_dist_n8'] for r in ok),
            e_q_cholqr_max=max(r['e_q_cholqr_bf16'] for r in ok), e_q_householder_max=max(r['e_q_householder_bf16'] for r in ok),
            r_c_abs_diff_max=max(abs(r['r_c_cholqr'] - r['r_c_householder_same_order']) for r in ok),
            order_equal_min=min(r['order_equal_to_qr_basis'] for r in rows),
            order_equal_mean=st.mean(r['order_equal_to_qr_basis'] for r in rows),
            chain_late_interval_householder=hh, chain_late_interval_cholqr=cq,
            chain_late_interval_abs_diff_max_factor=max(abs(c['hh_late_interval'] - c['cq_late_interval']) for c in chains),
            chain_fallbacks=sum(c['cq_fallbacks'] for c in chains),
            chain_fallback_ages={c['factor']: c['cq_fallback_ages'] for c in chains if c['cq_fallbacks']},
            angle_max_first_half_shift_1e7=max((r.get('angle_max_first_half_shift_1e7', 0) for r in rows), default=None))
    return out


# ----------------------------------------------------------------------------- training
def training_summary(comparison):
    def ce(v, call):
        return next((x['ce'] for x in v if x['call'] == call), None)
    rows = []
    for r in comparison['results']:
        rows.append(dict(arm=r['arm'], label=LABEL.get(r['arm'], r['arm']), launch='this',
                         ce={c: ce(r['validation'], c) for c in (50, 100, 200, 500, 1000, 1500, 2000)},
                         mean_step_ms=r['mean_step_ms'], mean_optimizer_ms=r['mean_optimizer_ms'], mean_fb_ms=r['mean_fb_ms'],
                         optimizer_by_kind=r['optimizer_by_kind'], refresh_counts=r['refresh_counts'],
                         fallback_log=r['fallback_log'], max_spectral_gram=r['max_spectral_gram'],
                         spectral_gram_by_call=r['spectral_gram_by_call'], max_normalized_gram=r['max_normalized_gram'],
                         peak_gpu_gib=r['peak_gpu_bytes'] / 2**30, elapsed_seconds=r['elapsed_seconds']))
    cached = [comparison['paper_cached_warm_prefix']] + comparison['paper_qr20_qr50']
    for r in cached:
        rows.append(dict(arm=r['case'], label=LABEL.get(r['case'], r['case']), launch='paper (cached)',
                         ce={c: ce(r['validation'], c) for c in (50, 100, 200, 500, 1000, 1500, 2000)},
                         mean_step_ms=r['mean_step_ms'], mean_optimizer_ms=r['mean_optimizer_ms']))
    prod = next((r for r in rows if r['arm'] == 'bf16_w10_qr50'), None)
    paper = next((r for r in rows if r['arm'] == 'cached_bf16_warm10_qr50'), None)
    repro = None
    if prod and paper:
        repro = dict(exact=all(prod['ce'][c] == paper['ce'][c] for c in prod['ce'] if prod['ce'][c] is not None),
                     max_abs_ce_diff=max(abs(prod['ce'][c] - paper['ce'][c]) for c in prod['ce'] if prod['ce'][c] is not None))
    return rows, repro


def figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                         'axes.spines.right': False})
    calls = (500, 1000, 2000)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), constrained_layout=True)
    for ax, call in zip(axes, calls):
        pts = [r for r in rows if r['ce'].get(call) is not None and r['arm'] in FAMILY]
        for r in pts:
            color = COLORS[FAMILY[r['arm']]]
            filled = r['launch'] == 'this'
            ax.scatter(r['mean_step_ms'], r['ce'][call], s=46, zorder=3, linewidths=1.6,
                       facecolors=color if filled else '#fcfcfb', edgecolors=color if not filled else '#fcfcfb')
        ys = sorted(((r['ce'][call], r) for r in pts), key=lambda t: t[0])
        span = (max(y for y, _ in ys) - min(y for y, _ in ys)) or 1
        xs = [r['mean_step_ms'] for r in pts]
        dx = .04 * ((max(xs) - min(xs)) or 10)
        last = -1e9
        for y, r in ys:  # stacked direct labels in data coordinates, at least 7.5% of the span apart
            yl = max(y, last + .075 * span)
            last = yl
            ax.annotate(LABEL[r['arm']].replace(' (cached)', '*'), (r['mean_step_ms'], y),
                        xytext=(r['mean_step_ms'] + dx, yl), textcoords='data', fontsize=6.6, color='#52514e',
                        va='center', arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=.6, shrinkA=0, shrinkB=3))
        ax.set_title(f'validation CE at call {call}', fontsize=9, color='#0b0b0b', loc='left')
        noise = {500: .0117, 1000: .0095, 2000: .0039}[call]  # production seed 205 - seed 204 at this call
        ax.text(.98, .02, f'seed 204 vs 205 spread (production): {noise:.4f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.6, color='#52514e')
        ax.set_xlabel('mean step time, calls 101-2000 (ms)')
        ax.grid(True, color='#e8e7e3', lw=.6, zorder=0)
        ax.margins(x=.35, y=.12)
    axes[0].set_ylabel('validation CE (1,024 rows)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=n, markersize=6) for c, n in zip(COLORS, FAMILY_NAMES)]
    handles.append(plt.Line2D([], [], marker='o', ls='', markerfacecolor='#fcfcfb', markeredgecolor='#52514e',
                              label='hollow, *: paper runs (cached, other launch)', markersize=6))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle('109M OWT, seed 204, 2,000 calls: validation CE against step time per refresh schedule',
                 fontsize=9.5, x=.01, ha='left', color='#0b0b0b')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


def copy_evidence(bundle):
    dest = OUT / 'runs'
    for name in ('manifest.json', 'comparison.json', 'completion_audit.json', 'progress.json'):
        if (bundle / name).exists():
            (OUT / f'stage_{bundle.name}').mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / name, OUT / f'stage_{bundle.name}' / name)
    for run in sorted((bundle / 'runs').glob('*')):
        d = dest / bundle.name / run.name
        d.mkdir(parents=True, exist_ok=True)
        for f in ('result.json', 'events.jsonl', 'geometry.json', 'fastqr_stats.json', 'cuda_parity.json'):
            if (run / f).exists():
                shutil.copy2(run / f, d / f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, action='append', default=[])
    a = p.parse_args()
    summary = {}
    t = load(OUT / 'timing.json')
    if t:
        summary['timing'] = timing_summary(t)
    summary['trained'] = trained_summary(sorted(OUT.glob('trained_d*_*.json')))
    summary['unit'] = {f.name: load(f)['results'] for f in sorted(OUT.glob('unit_*.json'))}
    c = load(OUT / 'compiled_ieee_check.json')
    if c:
        summary['compiled_ieee_check'] = c['rows']
    for bundle in a.bundle:
        comp = load(bundle / 'comparison.json')
        if not comp:
            continue
        rows, repro = training_summary(comp)
        summary.setdefault('training', {})[bundle.name] = dict(rows=rows, reproduces_paper_warm_prefix=repro)
        figure(rows, OUT / 'fastqr_ce_vs_step')
        copy_evidence(bundle)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items()}, default=str)[:3000])


if __name__ == '__main__':
    main()
