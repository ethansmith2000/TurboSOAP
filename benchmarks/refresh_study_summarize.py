"""Summarize the refresh-study replay/ablation (2026-11-20).

Reads reports/refresh_study_20261120/replay_parts/{replay,align}_d*.json, the
trajectory meta.json files and microbench.json (variant timings), and writes
replay.json, ablation.json, replay_quality.pdf and ablation.pdf, plus markdown
tables (tables.md) used by REPORT.md.  All replay results are open-loop.

Averaging: unweighted mean over a width's 8 factors (as paper Sec. 5.5), then
mean over the refresh ages in the window.  'after' = immediately after the
refresh at that age (or the held basis when the schedule does not refresh);
'before' = the previous basis evaluated on the covariance at that age, i.e.
just before that age's refresh.  Phase = age mod 50.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/refresh_study_20261120'
DATA_ROOT = Path('/workspace/data/turbosoap_refresh_study_20261120')
SCHEDULES = ['qr10', 'warm10_qr50', 'qr20', 'qr50', 'eigh10', 'cholqr10', 'none']
COLORS = {'warm10_qr50': '#2a78d6', 'qr10': '#eb6834', 'eigh10': '#1baf7a', 'cholqr10': '#eda100',
          'qr20': '#e87ba4', 'qr50': '#008300', 'none': '#6250d6'}
LABELS = {'warm10_qr50': 'warm10/QR50', 'qr10': 'QR10', 'qr20': 'QR20', 'qr50': 'QR50', 'eigh10': 'eigh10',
          'cholqr10': 'Cholesky-QR10', 'none': 'init only'}
WIDTH_RAMP = ['#86b6ef', '#5598e7', '#256abf', '#0d366b']
METRICS = ['rC', 'rCw', 'corr_rms', 'gram', 'top1', 'top8', 'topK']
WINDOWS = {'all': (10, 10 ** 9), 'late': (500, 10 ** 9)}


def nanmean(x, axis=None):
    x = np.asarray(x, dtype=float)
    with np.errstate(all='ignore'):
        if np.all(np.isnan(x)):
            return float('nan') if axis is None else np.nanmean(x, axis=axis)
        return np.nanmean(x, axis=axis)


def fnum(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def load_width(path):
    """-> dict(ages, chains, factors, data[chain][phase][metric] = (S, F) array, extras)."""
    r = json.loads(Path(path).read_text())
    ages = np.array(r['ages'])
    nf = len(r['factors'])
    data = {}
    for dim, g in r['groups'].items():
        kmax = max(g['ks'])
        for ch, ph_map in g['series'].items():
            for ph, met in ph_map.items():
                for key, arr in met.items():
                    name = key
                    if key.startswith('top') and key not in ('top1', 'top8'):
                        name = 'topK'
                    a = np.array(arr, dtype=float)
                    slot = data.setdefault(ch, {}).setdefault(ph, {}).setdefault(name, np.full((len(ages), nf), np.nan))
                    slot[:, g['factor_indices']] = a
                if 'top8' in met and kmax <= 8:  # no larger k for tiny dims
                    data[ch][ph].setdefault('topK', np.full((len(ages), nf), np.nan))[:, g['factor_indices']] = np.array(met['top8'])
    return dict(ages=ages, chains=r['chains'], factors=r['factors'], data=data, raw=r)


def summarize_chain(w, ch):
    ages = w['ages']
    phase = ages % 50
    out = {}
    for wname, (lo, hi) in WINDOWS.items():
        sel = (ages >= lo) & (ages <= hi)
        block = {}
        for ph in ('after', 'before'):
            for m in METRICS:
                arr = w['data'][ch][ph].get(m)
                if arr is None:
                    continue
                fm = nanmean(arr, axis=1)  # factor mean per age
                block[f'{m}_{ph}_mean'] = fnum(nanmean(fm[sel]))
                block[f'{m}_{ph}_by_phase'] = {str(p): fnum(nanmean(fm[sel & (phase == p)])) for p in (0, 10, 20, 30, 40)}
                if m == 'rC':
                    block[f'rC_{ph}_max_factor_age'] = fnum(np.nanmax(arr[sel]) if np.any(np.isfinite(arr[sel])) else float('nan'))
        # interval mean: average of (after at t, before at t+10) over the window
        a = nanmean(w['data'][ch]['after']['rC'], axis=1)
        b = nanmean(w['data'][ch]['before']['rC'], axis=1)
        idx = np.where(sel[:-1])[0]
        block['rC_interval_mean'] = fnum(nanmean(.5 * (a[idx] + b[idx + 1]))) if len(idx) else None
        aw = nanmean(w['data'][ch]['after']['rCw'], axis=1)
        bw = nanmean(w['data'][ch]['before']['rCw'], axis=1)
        block['rCw_interval_mean'] = fnum(nanmean(.5 * (aw[idx] + bw[idx + 1]))) if len(idx) else None
        out[wname] = block
    # per factor-dim breakdown (late, after and before)
    sel = ages >= WINDOWS['late'][0]
    dims = {}  # noqa
    for f in w['factors']:
        dims.setdefault(f['dim'], []).append(f['index'])
    out['late_rC_by_dim'] = {str(d): dict(
        after=fnum(nanmean(w['data'][ch]['after']['rC'][sel][:, ix])),
        before=fnum(nanmean(w['data'][ch]['before']['rC'][sel][:, ix])),
        n_factors=len(ix)) for d, ix in sorted(dims.items())}
    return out


def factor_mean_series(w, ch, ph, m='rC'):
    return nanmean(w['data'][ch][ph][m], axis=1)


def chol_summary(w):
    out = dict(refresh_events=0, factor_events=0, info_nonzero=0, nonfinite=0, first_failure_age={})
    for dim, g in w['raw']['groups'].items():
        for ev in g['cholqr_failures']:
            out['refresh_events'] += 1
            out['factor_events'] += len(ev['info_nonzero'])
            for j, (bad, nf) in enumerate(zip(ev['info_nonzero'], ev['nonfinite'])):
                out['info_nonzero'] += int(bad)
                out['nonfinite'] += int(nf)
                key = f"{dim}:{g['factor_indices'][j]}"
                if (bad or nf) and key not in out['first_failure_age']:
                    out['first_failure_age'][key] = ev['age']
    return out


def plot_replay(widths, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    ws = sorted(widths)
    # (a) sawtooth for the largest width over ages 500..700
    d = ws[-1] if len(ws) else None
    ax = axes[0, 0]
    if d is not None:
        w = widths[d]
        ages = w['ages']
        for ch in ('qr50', 'warm10_qr50', 'qr10', 'eigh10'):
            a, b = factor_mean_series(w, ch, 'after'), factor_mean_series(w, ch, 'before')
            xs, ys = [], []
            for i, age in enumerate(ages):
                if age < 500 or age > 700:
                    continue
                if np.isfinite(b[i]):
                    xs.append(age); ys.append(b[i])
                xs.append(age); ys.append(a[i])
            ax.plot(xs, ys, '-', color=COLORS[ch], lw=1.6, label=LABELS[ch])
        for x in range(500, 701, 50):
            ax.axvline(x, color='0.6', lw=.6, ls=':')
        ax.set_title(f'(a) d={d}: factor-mean $r_C$, before→after each refresh', fontsize=9)
        ax.set_xlabel('age')
        ax.set_ylabel('$r_C$')
        ax.legend(fontsize=7, frameon=False)
    # (b) warm10/QR50 build-up by phase, per width
    ax = axes[0, 1]
    for k, dd in enumerate(ws):
        w = widths[dd]
        s = summarize_chain(w, 'warm10_qr50')['late']
        q = summarize_chain(w, 'qr10')['late']
        xs, ys, yq = [], [], []
        for p in (10, 20, 30, 40):
            xs += [p - 10 + 9.5, p]
            ys += [s['rC_before_by_phase'][str(p)], s['rC_after_by_phase'][str(p)]]
            yq += [q['rC_before_by_phase'][str(p)], q['rC_after_by_phase'][str(p)]]
        xs += [49.5, 50]
        ys += [s['rC_before_by_phase']['0'], s['rC_after_by_phase']['0']]
        yq += [q['rC_before_by_phase']['0'], q['rC_after_by_phase']['0']]
        xs = [0] + xs
        ys = [s['rC_after_by_phase']['0']] + ys
        yq = [q['rC_after_by_phase']['0']] + yq
        ax.plot(xs, ys, '-', color=WIDTH_RAMP[k % 4], lw=1.6, marker='o', ms=3, label=f'warm10/QR50 d={dd}')
        ax.plot(xs, yq, '--', color=WIDTH_RAMP[k % 4], lw=1)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel('age mod 50 (QR at 0/50; dashed: QR10, same width)')
    ax.set_ylabel('$r_C$ (late window, factor mean)')
    ax.set_title('(b) build-up across the four warm refreshes', fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    def by_width(ax, key_after, key_before, title, ylabel, chains=SCHEDULES, log=False):
        for ch in chains:
            ya = [summarize_chain(widths[dd], ch)['late'].get(key_after) for dd in ws]
            yb = [summarize_chain(widths[dd], ch)['late'].get(key_before) for dd in ws] if key_before else None
            ax.plot(ws, ya, '-', color=COLORS[ch], marker='o', ms=4, lw=1.6, label=LABELS[ch])
            if yb is not None:
                ax.plot(ws, yb, ':', color=COLORS[ch], marker='o', ms=4, mfc='none', lw=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ws)
        ax.set_xticklabels([str(x) for x in ws])
        if log:
            ax.set_yscale('log')
        ax.set_xlabel('model width d')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
    shown = [c for c in SCHEDULES if c not in ('none', 'cholqr10')]
    by_width(axes[0, 2], 'rC_after_mean', 'rC_before_mean', '(c) $r_C$ late mean: after (solid) / before (dotted)', '$r_C$', chains=shown)
    axes[0, 2].legend(fontsize=7, frameon=False)
    by_width(axes[1, 0], 'rCw_after_mean', 'rCw_before_mean', '(d) diag-normalized $r_C^w$ (dirs ≥1e-4·max)', '$r_C^w$', chains=shown)
    by_width(axes[1, 1], 'topK_after_mean', 'topK_before_mean', '(e) top-k subspace overlap, k=max(8,n/8)', 'overlap', chains=shown)
    by_width(axes[1, 2], 'gram_after_mean', None, '(f) normalized Gram error $e_Q$ after refresh', '$e_Q$', log=True, chains=shown)
    axes[0, 2].text(.02, .97, 'not shown: init-only ($r_C$≈0.92);\nCholesky-QR10 non-finite from age 10', transform=axes[0, 2].transAxes,
                    fontsize=7, va='top', color='0.3')
    for ax in axes.flat:
        ax.grid(alpha=.3, lw=.5)
    fig.suptitle('Open-loop replay on QR10-driven covariance trajectories (1-layer models, OWT); late window = ages 500–1000',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=120)


def plot_ablation(abl, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ws = sorted(int(k) for k in abl['widths'])
    knobs = [('damping', [.001, .003, .01, .03, .1], lambda v: f'damping={v}' if v != .01 else 'default', 'damping δ', True),
             ('cap', [.25, .5, 1.0, None], lambda v: f'cap={v}' if v != .5 else 'default', 'Frobenius cap', False),
             ('ns', [0, 1, 2, 3], lambda v: f'ns={v}' if v != 2 else 'default', 'Newton–Schulz corrections', False),
             ('steps', [1, 2], lambda v: 'warm_steps=2' if v == 2 else 'default', 'warm steps per refresh', False)]
    timing = abl.get('timing', {})
    rows = [('rC_warm_after_late', '$r_C$ after warm refresh (late)'), ('rC_before_qr_late', '$r_C$ just before QR (late)'),
            ('gram_after_late', '$e_Q$ after refresh (late)'), ('time', 'ms / 12 factors (n=768, compiled)')]
    fig, axes = plt.subplots(len(rows), len(knobs), figsize=(13, 10))
    for j, (knob, values, name_of, xlabel, logx) in enumerate(knobs):
        xs = list(range(len(values))) if not logx else values
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            if key == 'time':
                t = [timing.get(name_of(v), {}).get('768', {}).get('compiled') for v in values]
                te = [timing.get(name_of(v), {}).get('768', {}).get('eager') for v in values]
                if any(x is not None for x in t):
                    ax.plot(xs, [np.nan if x is None else x for x in t], '-o', color='#2a78d6', ms=4, label='compiled')
                    ax.plot(xs, [np.nan if x is None else x for x in te], ':o', color='#2a78d6', mfc='none', ms=4, label='eager')
                    if j == 0:
                        ax.legend(fontsize=7, frameon=False)
                else:
                    ax.text(.5, .5, 'timing pending', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            else:
                diverged = []
                for k, d in enumerate(ws):
                    vals = abl['widths'][str(d)]
                    y = [None if (vals.get(name_of(v), {}).get('nonfinite_fraction') or 0) > 0 else vals.get(name_of(v), {}).get(key)
                         for v in values]
                    diverged += [xs[ii] for ii, v in enumerate(values) if (vals.get(name_of(v), {}).get('nonfinite_fraction') or 0) > 0]
                    ax.plot(xs, [np.nan if v is None else v for v in y], '-o', color=WIDTH_RAMP[k % 4], ms=4, lw=1.5,
                            label=f'd={d}')
                    ref = vals.get('_qr10', {}).get(key)
                    if ref is not None and key != 'gram_after_late':
                        ax.axhline(ref, color=WIDTH_RAMP[k % 4], lw=.8, ls='--')
                if key == 'gram_after_late':
                    ax.set_yscale('log')
                for x in sorted(set(diverged)):
                    ax.annotate('diverged\n(non-finite)', (x, .5), xycoords=('data', 'axes fraction'), ha='center',
                                fontsize=7, color='#e34948')
            if logx:
                ax.set_xscale('log')
            else:
                ax.set_xticks(xs)
                ax.set_xticklabels(['none' if v is None else str(v) for v in values])
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=.3, lw=.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, title='dashed: QR10', title_fontsize=7)
    fig.suptitle('Warm-step ablation inside warm10/QR50 (open-loop replay; one knob varied at a time around 0.01 / 0.5 / 2 / 1)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', default=str(REPORT))
    ap.add_argument('--data-root', default=str(DATA_ROOT))
    ap.add_argument('--late-start', type=int, default=500)
    args = ap.parse_args()
    WINDOWS['late'] = (args.late_start, 10 ** 9)
    rep = Path(args.report_dir)
    parts = sorted((rep / 'replay_parts').glob('replay_d*.json'), key=lambda p: int(p.stem.split('_d')[1]))
    widths = {int(p.stem.split('_d')[1]): load_width(p) for p in parts}
    replay = dict(open_loop=True, description=__doc__, widths={})
    ablation = dict(open_loop=True, widths={}, variants=[])
    for d, w in sorted(widths.items()):
        meta_path = Path(args.data_root) / f'd{d}' / 'meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        align_path = rep / 'replay_parts' / f'align_d{d}.json'
        align = json.loads(align_path.read_text()) if align_path.exists() else None
        entry = dict(
            training=dict(validation=meta.get('validation'), final_train_loss=(meta.get('losses') or [None])[-1],
                          elapsed_s=meta.get('elapsed_s'), median_step_s=(float(np.median(meta['step_seconds']))
                                                                          if meta.get('step_seconds') else None),
                          model_params=meta.get('model_params'), device=meta.get('device'),
                          gpu_claim=meta.get('gpu_claim')),
            factors=w['factors'],
            schedules={ch: summarize_chain(w, ch) for ch in SCHEDULES if ch in w['data']},
            cholqr=chol_summary(w),
            default_variant_bit_equal_every_age=all(g['default_variant_bit_equal_every_age']
                                                    for g in w['raw']['groups'].values()),
            alignment=None if align is None else dict(
                bit_equal_all_factors=all(r['bit_equal_all'] for r in align['rows']),
                max_abs_diff=max((r['max_abs_diff'] or 0) for r in align['rows'])),
            series_factor_mean_rC={ch: dict(after=[fnum(x) for x in factor_mean_series(w, ch, 'after')],
                                            before=[fnum(x) for x in factor_mean_series(w, ch, 'before')])
                                   for ch in SCHEDULES if ch in w['data']},
            ages=w['ages'].tolist())
        lw = entry['schedules']
        entry['paper_cell_phase40_after'] = {wn: dict(warm10_qr50=lw['warm10_qr50'][wn]['rC_after_by_phase']['40'],
                                                      qr10=lw['qr10'][wn]['rC_after_by_phase']['40'])
                                             for wn in WINDOWS}
        replay['widths'][str(d)] = entry
        # ablation
        av = {}
        for ch in w['chains']:
            if not ch.startswith('warm10_qr50|') and ch != 'qr10' and ch != 'qr50':
                continue
            s = summarize_chain(w, ch)
            late = s['late']
            row = dict(
                rC_warm_after_late=fnum(nanmean([late['rC_after_by_phase'][str(p)] for p in (10, 20, 30, 40)])),
                rC_before_qr_late=late['rC_before_by_phase']['0'],
                rC_after_phase40_late=late['rC_after_by_phase']['40'],
                rC_interval_mean_late=late['rC_interval_mean'],
                rCw_warm_after_late=fnum(nanmean([late['rCw_after_by_phase'][str(p)] for p in (10, 20, 30, 40)])),
                rCw_interval_mean_late=late['rCw_interval_mean'],
                topK_warm_after_late=fnum(nanmean([late['topK_after_by_phase'][str(p)] for p in (10, 20, 30, 40)])),
                top1_warm_after_late=fnum(nanmean([late['top1_after_by_phase'][str(p)] for p in (10, 20, 30, 40)])),
                gram_after_late=late['gram_after_mean'],
                gram_after_max_late=fnum(np.nanmax(w['data'][ch]['after']['gram'][w['ages'] >= WINDOWS['late'][0]], initial=-np.inf)),
                rC_after_all=s['all']['rC_after_mean'], rC_before_all=s['all']['rC_before_mean'])
            after = w['data'][ch]['after']['rC'][1:]
            bad = ~np.isfinite(after)
            row['nonfinite_fraction'] = float(bad.mean())
            row['first_nonfinite_age'] = (int(w['ages'][1:][bad.any(1)][0]) if bad.any() else None)
            key = ch.split('|')[1] if '|' in ch else '_' + ch
            av[key] = row
        base = av.get('default', {})
        for k, row in av.items():
            row['delta_vs_default'] = {m: (None if row.get(m) is None or base.get(m) is None else row[m] - base[m])
                                       for m in ('rC_warm_after_late', 'rC_before_qr_late', 'rC_interval_mean_late',
                                                 'rCw_interval_mean_late', 'gram_after_late')}
        ablation['widths'][str(d)] = av
        ablation['variants'] = [c.split('|')[1] for c in w['chains'] if '|' in c]
    # timing from microbench
    mb = rep / 'microbench.json'
    timing = {}
    if mb.exists():
        m = json.loads(mb.read_text())
        for r in m.get('variant_rows', []):
            if r.get('ok'):
                timing.setdefault(r['variant'], {}).setdefault(str(r['n']), {})['compiled' if r['compiled'] else 'eager'] = r['median_ms']
        ablation['timing_bit_exact_checks'] = m.get('variant_bit_exact')
    ablation['timing'] = timing
    ablation['timing_note'] = 'ms per batch-12 call of the warm step alone, exclusive GPU (microbench.json variant_rows)'
    (rep).mkdir(parents=True, exist_ok=True)
    (rep / 'replay.json').write_text(json.dumps(replay, indent=1, allow_nan=True))
    (rep / 'ablation.json').write_text(json.dumps(ablation, indent=1, allow_nan=True))
    if widths:
        plot_replay(widths, rep / 'replay_quality.pdf')
        plot_ablation(ablation, rep / 'ablation.pdf')
    # markdown tables
    lines = ['## Replay (late window, ages 500–1000; factor means)', '',
             '| d | schedule | r_C after | r_C before | r_C interval | r_C^w interval | top-1 after | top-K after | e_Q after |',
             '|---|---|---|---|---|---|---|---|---|']
    f = lambda x: '—' if x is None else f'{x:.3f}'
    g = lambda x: '—' if x is None else f'{x:.2e}'
    for d, e in replay['widths'].items():
        for ch, s in e['schedules'].items():
            L = s['late']
            lines.append(f"| {d} | {LABELS[ch]} | {f(L['rC_after_mean'])} | {f(L['rC_before_mean'])} | {f(L['rC_interval_mean'])} | "
                         f"{f(L['rCw_interval_mean'])} | {f(L['top1_after_mean'])} | {f(L['topK_after_mean'])} | {g(L['gram_after_mean'])} |")
    lines += ['', '## warm10/QR50 build-up by phase (late window; after/before refresh)', '',
              '| d | 0 after (QR) | 10 before/after | 20 before/after | 30 before/after | 40 before/after | 50 before (pre-QR) | QR10 phase-40 after |',
              '|---|---|---|---|---|---|---|---|']
    for d, e in replay['widths'].items():
        L = e['schedules']['warm10_qr50']['late']
        Q = e['schedules']['qr10']['late']
        cells = [f"{f(L['rC_before_by_phase'][str(p)])} / {f(L['rC_after_by_phase'][str(p)])}" for p in (10, 20, 30, 40)]
        lines.append(f"| {d} | {f(L['rC_after_by_phase']['0'])} | " + ' | '.join(cells) +
                     f" | {f(L['rC_before_by_phase']['0'])} | {f(Q['rC_after_by_phase']['40'])} |")
    lines += ['', '## Ablation (late window; warm10/QR50 chains)', '',
              '| d | variant | r_C after warm | r_C pre-QR | r_C interval | r_C^w interval | e_Q after | ms (n=768 b12, compiled) |',
              '|---|---|---|---|---|---|---|---|']
    for d, av in ablation['widths'].items():
        for k, row in av.items():
            t = timing.get(k, {}).get('768', {}).get('compiled') if not k.startswith('_') else None
            if row.get('nonfinite_fraction'):
                k = f"{k} (non-finite {row['nonfinite_fraction']:.0%}, first at age {row['first_nonfinite_age']})"
            lines.append(f"| {d} | {k} | {f(row['rC_warm_after_late'])} | {f(row['rC_before_qr_late'])} | "
                         f"{f(row['rC_interval_mean_late'])} | {f(row['rCw_interval_mean_late'])} | {g(row['gram_after_late'])} | "
                         f"{'—' if t is None else f'{t:.2f}'} |")
    (rep / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
