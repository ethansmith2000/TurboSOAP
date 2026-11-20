"""Summarize Part 2b (alternative warm-step generators; open-loop replay).

Inputs: replay_pb_parts/replay_pb_d*.json (variants x {warm10_qr50, warm10}),
replay_parts/replay_d*.json (Part 2 references QR10/QR20/QR50/eigh10 and the
production warm10/QR50 chain, used for a consistency check), optional
replay_pb_timing.json.  Outputs replay_pb.json, replay_pb.pdf, tables_pb.md.

Window 'late' = ages 500-1000.  Interval mean of a metric = mean over refresh
ages t of (after_t + before_{t+10}) / 2, after an unweighted mean over the
width's 8 factors.  rho_rms = RMS off-diagonal correlation of Q^T C Q over
coordinates with diagonal >= 1e-4 max (the panel's whitened metric; recorded
as ``corr_rms``).  Gap closed = (QR50 - X) / (QR50 - QR10) on interval means.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path[:0] = [str(HERE)]
import refresh_study_summarize as S  # noqa: E402

REPORT = HERE.parent / 'reports/refresh_study_20261120'
REFS = ['qr10', 'qr20', 'qr50', 'eigh10']
SHOWN = ['prod', 'pb1_r1e-4_c0.5', 'pb1_r1e-4_c1.0', 'pb1_r1e-4_spec0.5', 'relgap_r1e-4_c0.5', 'jac1_c0.5']
COLORS = dict(zip(SHOWN, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
LABEL = {'prod': 'production', 'pb1_r1e-4_c0.5': 'PB1 ρ=1e-4, F-cap 0.5', 'pb1_r1e-4_c1.0': 'PB1 ρ=1e-4, F-cap 1.0',
         'pb1_r1e-3_c0.5': 'PB1 ρ=1e-3, F-cap 0.5', 'pb1_r1e-3_c1.0': 'PB1 ρ=1e-3, F-cap 1.0',
         'pb1_r1e-4_spec0.5': 'PB1 ρ=1e-4, spectral cap 0.5', 'relgap_r1e-4_c0.5': 'relative-gap damping',
         'jac1_c0.5': 'Jacobi factor 1, F-cap 0.5', 'pb1_r1e-4_c0.5_ns1': 'PB1 ρ=1e-4, F-cap 0.5, 1 NS'}


def interval(w, ch, metric, lo):
    ages = w['ages']
    a = S.nanmean(w['data'][ch]['after'][metric], axis=1)
    b = S.nanmean(w['data'][ch]['before'][metric], axis=1)
    idx = np.where(ages[:-1] >= lo)[0]
    return S.fnum(S.nanmean(.5 * (a[idx] + b[idx + 1]))) if len(idx) else None


def chain_summary(w, ch, lo, dims_left):
    ages = w['ages']
    sel = ages >= lo
    out = {m: interval(w, ch, m, lo) for m in ('rC', 'rCw', 'corr_rms')}
    for m in ('rC', 'rCw', 'corr_rms', 'gram', 'top1', 'topK'):
        for ph in ('after', 'before'):
            arr = w['data'][ch][ph].get(m)
            if arr is not None:
                out[f'{m}_{ph}'] = S.fnum(S.nanmean(S.nanmean(arr, axis=1)[sel]))
    after = w['data'][ch]['after']['rC'][1:]
    bad = ~np.isfinite(after)
    out['nonfinite_fraction'] = float(bad.mean())
    out['first_nonfinite_age'] = int(ages[1:][bad.any(1)][0]) if bad.any() else None
    for m in ('rC', 'corr_rms'):
        arr = w['data'][ch]['after'][m][sel]
        out[f'{m}_after_p90'] = S.fnum(np.nanpercentile(arr, 90)) if np.isfinite(arr).any() else None
        arr_l = arr[:, dims_left]
        out[f'{m}_after_left3d4d'] = S.fnum(S.nanmean(arr_l))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', default=str(REPORT))
    ap.add_argument('--late-start', type=int, default=500)
    args = ap.parse_args()
    rep = Path(args.report_dir)
    lo = args.late_start
    parts = sorted((rep / 'replay_pb_parts').glob('replay_pb_d*.json'), key=lambda p: int(p.stem.split('_d')[1]))
    result = dict(open_loop=True, description=__doc__, late_start=lo, widths={})
    timing = {}
    tpath = rep / 'replay_pb_timing.json'
    if tpath.exists():
        t = json.loads(tpath.read_text())
        for r in t['rows']:
            if r['ok']:
                timing.setdefault(r['variant'], {})[f"{r['n']}x{r['batch']}_{'compiled' if r['compiled'] else 'eager'}"] = r['median_ms']
        result['timing_bit_exact'] = t.get('bit_exact')
    result['timing_ms'] = timing
    for p in parts:
        d = int(p.stem.split('_d')[1])
        w = S.load_width(p)
        ref = S.load_width(rep / 'replay_parts' / f'replay_d{d}.json')
        dmin = min(f['dim'] for f in w['factors'])
        dims_left = [f['index'] for f in w['factors'] if f['dim'] >= 3 * dmin]
        refs = {ch: chain_summary(ref, ch, lo, dims_left) for ch in REFS}
        entry = dict(references=refs, chains={}, gap_closed={}, capstats={})
        # consistency with Part 2 production chain
        a = np.array(ref['data']['warm10_qr50']['after']['rC'])
        b = np.array(w['data']['prod|warm10_qr50']['after']['rC'])
        entry['prod_matches_part2'] = bool(np.allclose(a, b, equal_nan=True, rtol=0, atol=0))
        entry['prod_bit_equal_to_production_every_age'] = all(g['prod_bit_equal_every_age'] for g in w['raw']['groups'].values())
        for ch in w['chains']:
            if ch.startswith('prodcheck'):
                continue
            s = chain_summary(w, ch, lo, dims_left)
            entry['chains'][ch] = s
            gc = {}
            for m in ('rC', 'rCw', 'corr_rms'):
                q50, q10, x = refs['qr50'][m], refs['qr10'][m], s[m]
                gc[m] = None if None in (q50, q10, x) or q50 == q10 else (q50 - x) / (q50 - q10)
            entry['gap_closed'][ch] = gc
            caps = []
            clamps = []
            for g in w['raw']['groups'].values():
                for ev in g.get('capstats', {}).get(ch, []):
                    if ev['age'] >= lo and 'cap_binding' in ev:
                        caps += [(bool(x), m) for x, m in zip(ev['cap_binding'], ev['cap_mult'])]
                    if ev['age'] >= lo and 'clamped_fraction' in ev:
                        clamps.append(ev['clamped_fraction'])
            entry['capstats'][ch] = dict(
                binding_rate=(float(np.mean([c for c, _ in caps])) if caps else None),
                mean_multiplier=(float(np.mean([m for _, m in caps])) if caps else None),
                clamped_entry_fraction=(float(np.mean(clamps)) if clamps else None))
        result['widths'][str(d)] = entry
    (rep / 'replay_pb.json').write_text(json.dumps(result, indent=1, allow_nan=True))
    # tables
    f = lambda x: '—' if x is None else f'{x:.3f}'
    pc = lambda x: '—' if x is None else f'{100 * x:.0f}%'
    lines = [f'## Part 2b headline (late window, ages {lo}–1000; interval means)', '',
             '| d | variant | schedule | r_C | r_C^w | ρ_rms | gap closed r_C / r_C^w / ρ_rms | e_Q after | cap binds (mean mult) | non-finite | r_C p90 | r_C left 3d/4d |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for d, e in result['widths'].items():
        for ref in REFS:
            s = e['references'][ref]
            ge = '—' if s['gram_after'] is None else format(s['gram_after'], '.2e')
            lines.append(f"| {d} | **{ref.upper()}** | ref | {f(s['rC'])} | {f(s['rCw'])} | {f(s['corr_rms'])} | | "
                         f"{ge} | | | {f(s['rC_after_p90'])} | {f(s['rC_after_left3d4d'])} |")
        for ch, s in e['chains'].items():
            v, sched = ch.split('|')
            g = e['gap_closed'][ch]
            cs = e['capstats'][ch]
            nf = '' if not s['nonfinite_fraction'] else f"{100 * s['nonfinite_fraction']:.0f}% (from {s['first_nonfinite_age']})"
            cap = '—' if cs['binding_rate'] is None else f"{100 * cs['binding_rate']:.0f}% ({cs['mean_multiplier']:.2f})"
            lines.append(f"| {d} | {LABEL.get(v, v)} | {sched} | {f(s['rC'])} | {f(s['rCw'])} | {f(s['corr_rms'])} | "
                         f"{pc(g['rC'])} / {pc(g['rCw'])} / {pc(g['corr_rms'])} | "
                         f"{'—' if s['gram_after'] is None else format(s['gram_after'], '.2e')} | {cap} | {nf} | "
                         f"{f(s['rC_after_p90'])} | {f(s['rC_after_left3d4d'])} |")
    if timing:
        lines += ['', '## Warm-variant cost (ms per call; compiled / eager)', '',
                  '| variant | 768×12 | 1536×12 | 2304×12 | 3072×12 | 8192×1 |', '|---|---|---|---|---|---|']
        for v, tv in timing.items():
            cells = [f"{tv.get(k + '_compiled', float('nan')):.2f} / {tv.get(k + '_eager', float('nan')):.2f}"
                     for k in ('768x12', '1536x12', '2304x12', '3072x12', '8192x1')]
            lines.append(f'| {LABEL.get(v, v)} | ' + ' | '.join(cells) + ' |')
    (rep / 'tables_pb.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    plot(result, rep / 'replay_pb.pdf')


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ws = sorted(int(k) for k in result['widths'])
    if not ws:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [('rC', '$r_C$ (interval mean)'), ('rCw', '$r_C^w$ (interval mean)'), ('corr_rms', 'ρ_rms (interval mean)')]
    for i, sched in enumerate(('warm10_qr50', 'warm10')):
        for j, (m, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for ref, ls in (('qr10', '--'), ('qr50', ':'), ('eigh10', '-.')):
                y = [result['widths'][str(d)]['references'][ref][m] for d in ws]
                ax.plot(ws, y, ls, color='0.35', lw=1.2, label=ref.upper() if ref != 'eigh10' else 'eigh10')
            for v in SHOWN[1:] + SHOWN[:1]:  # production drawn last (Jacobi-1 coincides when the cap binds)
                ch = f'{v}|{sched}'
                y, bad = [], []
                for d in ws:
                    s = result['widths'][str(d)]['chains'].get(ch, {})
                    y.append(np.nan if s.get('nonfinite_fraction') else s.get(m))
                    if s.get('nonfinite_fraction'):
                        bad.append(d)
                ax.plot(ws, [np.nan if x is None else x for x in y], '--o' if v == 'prod' else '-o', color=COLORS[v],
                        ms=4, lw=1.6, label=LABEL[v])
                for d in bad:
                    ax.annotate('×', (d, .5), xycoords=('data', 'axes fraction'), color=COLORS[v], ha='center')
            ax.set_xscale('log', base=2)
            ax.set_xticks(ws)
            ax.set_xticklabels([str(x) for x in ws])
            ax.grid(alpha=.3, lw=.5)
            ax.set_xlabel('model width d')
            ax.set_ylabel(ylabel)
            title = 'warm10/QR50' if sched == 'warm10_qr50' else 'warm10, no QR repair'
            ax.set_title(f'({"abcdef"[3 * i + j]}) {title}', fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle('Part 2b: alternative warm-step generators, open-loop replay on trained factors (ages 500–1000)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=120)


if __name__ == '__main__':
    main()
