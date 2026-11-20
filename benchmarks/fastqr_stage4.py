"""Stage 4 of the fast-QR study: the C6 front-loaded schedule (S = 300, 1000; seed 204; 10k calls).
Gates, the decision rule fixed in REPORT.md section 9 before approval, cost, the PB8 settling log and a figure.

Writes reports/fastqr_20261120/stage4.json, fastqr_stage4.{pdf,png} and copies run evidence. Reads only.

  python benchmarks/fastqr_stage4.py           # the two stage-4 bundles
  python benchmarks/fastqr_stage4.py --smoke   # code-path check, the stage-2 QR10 run standing in
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmarks')]
from fastqr_stage2 import CACHED_PROD, DATA, OUT, ces, jl  # noqa: E402

QR10 = DATA / 'stage2_seed204_bf16_hh_qr10_c10000' / 'runs/bf16_hh_qr10'
SWITCHES = (300, 1000)
G_MIN, ENDPOINT_TOL, PB8_LIMIT = .5, .003, 2.  # REPORT.md section 9


def load(folder):
    events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
    updates = {e['call']: e for e in events if e['event'] == 'update'}
    stats = jl(folder / 'fastqr_stats.json') if (folder / 'fastqr_stats.json').exists() else {}
    return dict(result=jl(folder / 'result.json'), events=events, updates=updates, stats=stats,
                validation=ces(jl(folder / 'result.json')['validation']))


def gates(run, ref, S, bundle):
    a, b = run['updates'], ref['updates']
    first = next((c for c in sorted(a) if c in b and a[c]['loss'] != b[c]['loss']), None)
    early_val = {c: v for c, v in ref['validation'].items() if c <= S + 11}
    audit = jl(bundle / 'completion_audit.json') if (bundle / 'completion_audit.json').exists() else None
    in_run = {n: any(e['event'] == n and e.get('passed') for e in run['events'])
              for n in ('startup_match', 'initial_validation_match', 'initial_backward_repeat', 'final_state_finite')}
    return dict(in_run=in_run, completion_audit=audit and audit['status'],
                event_counts=audit and audit['arms'][0]['event_counts'],
                unique_graphs_at_captures=audit and audit['arms'][0]['unique_graphs_at_captures'],
                graph_breaks=run['result']['dynamo_counters'].get('graph_break', {}),
                qr10_prefix=dict(first_differing_loss_call=first, losses_equal_through=S + 11,
                                 passed=first is None or first > S + 11,
                                 validation_equal=all(run['validation'].get(c) == v for c, v in early_val.items())),
                pb8_error=run['stats'].get('pb8_error'))


def cost(run, S):
    excluded = S + 11
    ups = [u for c, u in run['updates'].items() if 101 <= c and c != excluded]
    by_kind = {k: st.mean(u['optimizer_ms'] for u in ups if u['kind'] == k)
               for k in ('ordinary', 'warm', 'qr') if any(u['kind'] == k for u in ups)}
    extra_qr = Counter(u['kind'] for u in run['updates'].values())['qr'] - 199
    within = extra_qr * (by_kind['qr'] - by_kind['warm']) / 10000 if 'warm' in by_kind else None
    return dict(excluded_call=excluded, excluded_call_optimizer_ms=run['updates'].get(excluded, {}).get('optimizer_ms'),
                mean_optimizer_ms=st.mean(u['optimizer_ms'] for u in ups),
                mean_step_ms=st.mean(u['step_ms'] for u in ups), mean_fb_ms=st.mean(u['fb_ms'] for u in ups),
                optimizer_ms_by_kind=by_kind, extra_qr_events=extra_qr, within_run_extra_ms_per_call=within,
                vs_production_cached=st.mean(u['optimizer_ms'] for u in ups) - 26.41,
                vs_production_stage1_fresh=st.mean(u['optimizer_ms'] for u in ups) - 26.6)


def rule(run, prod, qr10):
    calls = sorted(c for c in prod if c > 0 and c in run['validation'])
    delta = {c: run['validation'][c] - prod[c] for c in calls}
    g = {c: (prod[c] - run['validation'][c]) / (prod[c] - qr10[c]) for c in (200, 500, 1000) if c in calls}
    d10 = delta.get(10000)
    passed = g.get(500, 0) >= G_MIN and d10 is not None and d10 <= ENDPOINT_TOL
    outcome = ('pass' if passed else 'fail_no_early_gain' if g.get(500, 0) < G_MIN else
               'fail_endpoint_harm' if d10 is not None else 'incomplete')
    return dict(calls=calls, ce={c: run['validation'][c] for c in calls}, delta_vs_production=delta, gap_closure=g,
                delta_10k=d10, endpoint_gain=d10 is not None and d10 <= -ENDPOINT_TOL, outcome=outcome,
                call2000_context=dict(delta=delta.get(2000), qr10_delta=qr10[2000] - prod[2000],
                                      difference=(delta.get(2000) or 0) - (qr10[2000] - prod[2000])),
                mean_delta_500_1500=st.mean(delta[c] for c in calls if 500 <= c <= 1500))


def pb8(run, S):
    rows = run['stats'].get('pb8') or []
    pre = {r['refresh_age']: r for r in rows if r['kind'] == 'pre'}
    post = {r['refresh_age']: r for r in rows if r['kind'] == 'post'}
    mid = {r['refresh_age']: r for r in rows if r['kind'] == 'mid'}
    cells = defaultdict(list)
    pairs = []
    for age, p in sorted(post.items()):
        if age in pre and pre[age]['p999'] > 0:
            ratio = p['p999'] / pre[age]['p999']
            phase = 'qr_phase' if age <= S else 'late'
            cells[(p['refresh_event'], phase)].append(ratio)
            pairs.append(dict(age=age, event=p['refresh_event'], phase=phase, ratio=ratio,
                              pre=pre[age]['p999'], post=p['p999'], mid=mid.get(age, {}).get('p999')))
    summary = {f'{e}/{ph}': dict(n=len(v), median=st.median(v), p95=sorted(v)[int(.95 * (len(v) - 1))], max=max(v))
               for (e, ph), v in sorted(cells.items())}
    verdict = None
    if summary:
        verdict = 'pb8_dropped' if all(c['median'] <= PB8_LIMIT for c in summary.values()) else 'pb8_kept'
    return dict(samples=len(rows), cells=summary, verdict=verdict, pairs=pairs)


def figure(rows, prod, qr10, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = {300: '#eb6834', 1000: '#1baf7a'}
    blue = '#2a78d6'
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    ax.axhspan(-ENDPOINT_TOL, ENDPOINT_TOL, color='#e8e7e3', lw=0, zorder=0)
    ax.axhline(0, color='#8a8984', lw=.9, zorder=1)
    calls = sorted(c for c in prod if c >= 200 and c in qr10)
    series = [(f'C6, S = {S}', colors[S], r['rule']['delta_vs_production']) for S, r in rows.items()]
    series.append(('BF16 QR10', blue, {c: qr10[c] - prod[c] for c in calls}))  # last: S = 1000 equals it to call 1011
    ends = []
    for name, color, d in series:
        xs = [c for c in calls if c in d]
        ys = [d[c] for c in xs]
        thin = name == 'BF16 QR10'
        ax.plot(xs, ys, color=color, lw=1.4 if thin else 2, ls=(0, (3, 1.5)) if thin else '-', zorder=5 if thin else 3)
        ax.scatter(xs, ys, s=18 if thin else 26, color=color, edgecolors='#fcfcfb', linewidths=1, zorder=6 if thin else 4)
        ends.append((ys[-1], xs[-1], name))
    last = -1e9
    for y, x, name in sorted(ends):
        yl = max(y, last + .006)
        last = yl
        ax.annotate(f'{name}  {y:+.4f}', (x, y), xytext=(x * 1.25, yl), textcoords='data', fontsize=7,
                    color='#52514e', va='center', arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=.6, shrinkB=3))
    for S in rows:
        ax.axvline(S + 11, color=colors[S], lw=.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(170, 10000 * 3.2)
    ax.set_xlabel('call (log scale); dotted: switch at call S + 11')
    ax.set_ylabel('CE − cached BF16 warm10/QR50 CE')
    ax.set_title('seed 204: validation CE relative to production (below 0 = better)', fontsize=9, loc='left')
    ax.grid(True, color='#e8e7e3', lw=.6, zorder=0)
    ax = axes[1]
    for S, r in rows.items():
        pairs = r['pb8']['pairs']
        for event, marker in (('qr', 'o'), ('warm', 's')):
            xs = [p['age'] for p in pairs if p['event'] == event]
            ys = [p['ratio'] for p in pairs if p['event'] == event]
            if xs:
                ax.scatter(xs, ys, s=10, marker=marker, color=colors[S], alpha=.55, linewidths=0,
                           label=f'S = {S}, after {event.upper() if event == "qr" else "warm"}')
    ax.axhline(PB8_LIMIT, color='#8a8984', lw=.9, ls=(0, (3, 2)))
    ax.text(12, PB8_LIMIT * .985, 'panel limit: 2×', fontsize=6.8, color='#52514e', va='top')
    ax.set_xscale('log')
    ax.set_xlabel('refresh age (log scale)')
    ax.set_ylabel('p99.9 |M/(√V+ε)|: after ÷ before refresh')
    ax.set_title('PB8 settling ratio per refresh', fontsize=9, loc='left')
    ax.grid(True, color='#e8e7e3', lw=.6, zorder=0)
    if any(r['pb8']['pairs'] for r in rows.values()):
        ax.legend(frameon=False, fontsize=7, markerscale=1.8)
    fig.suptitle('109M OWT, C6 front-loaded schedule (QR every 10 ages up to S, then warm10/QR50), 10,000 calls',
                 fontsize=9.5, x=.01, ha='left', color='#0b0b0b')
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    prod = ces(jl(CACHED_PROD[204])['validation'])
    ref = load(QR10)
    rows = {}
    for S in SWITCHES:
        if a.smoke:
            bundle, folder = QR10.parents[1], QR10
        else:
            bundle = DATA / f'stage4_seed204_bf16_fl{S}_w10_qr50_c10000'
            folder = bundle / 'runs' / f'bf16_fl{S}_w10_qr50'
        if not (folder / 'result.json').exists():
            continue
        run = load(folder)
        rows[S] = dict(gates=gates(run, ref, S, bundle), rule=rule(run, prod, ref['validation']), cost=cost(run, S),
                       pb8=pb8(run, S))
    passing = [S for S in SWITCHES if S in rows and rows[S]['rule']['outcome'] == 'pass']
    proposal = (None if not passing else min(passing))
    out = dict(protocol='REPORT.md section 9 (fixed before approval)',
               thresholds=dict(gap_closure_min=G_MIN, endpoint_tol=ENDPOINT_TOL, pb8_limit=PB8_LIMIT),
               runs={S: {k: v for k, v in r.items() if k != 'pb8'} | dict(pb8={k: v for k, v in r['pb8'].items()
                                                                              if k != 'pairs'})
                     for S, r in rows.items()},
               seed205_proposal=proposal,
               extra_early_gain_s1000_vs_s300=(rows[1000]['rule']['mean_delta_500_1500'] -
                                               rows[300]['rule']['mean_delta_500_1500'])
               if 300 in rows and 1000 in rows else None)
    if not a.smoke:
        from fastqr_report import copy_evidence
        for S in rows:
            copy_evidence(DATA / f'stage4_seed204_bf16_fl{S}_w10_qr50_c10000')
    suffix = '_smoke' if a.smoke else ''
    (OUT / f'stage4{suffix}.json').write_text(json.dumps(out, indent=1, default=str))
    (OUT / f'stage4_pb8_pairs{suffix}.json').write_text(json.dumps({S: r['pb8']['pairs'] for S, r in rows.items()}))
    if rows:
        figure(rows, prod, ref['validation'], OUT / f'fastqr_stage4{suffix}')
    print(json.dumps(dict(proposal=proposal, **{str(S): dict(outcome=r['rule']['outcome'], g=r['rule']['gap_closure'],
                                                              d10k=r['rule']['delta_10k'],
                                                              prefix=r['gates']['qr10_prefix'],
                                                              cost=r['cost']['mean_optimizer_ms'],
                                                              within=r['cost']['within_run_extra_ms_per_call'],
                                                              pb8=r['pb8']['verdict'])
                                                 for S, r in rows.items()}), indent=1, default=str))


if __name__ == '__main__':
    main()
