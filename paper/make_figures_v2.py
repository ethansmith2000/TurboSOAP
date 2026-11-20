"""Figures for the 20 Nov 2026 reframing (warm = approximate refresh).

CPU plotting from saved JSON, plus one small synthetic GPU/CPU illustration.
Writes new PDFs into overleaf/figures/ and leaves the hash-pinned figures and
scripts (plot_review_context.py, summarize_bf16_completion.py) untouched.

Matched BF16 QR10 evidence is read in place from reports/fastqr_20261120/
(summary.json: stage-1 same-launch arms; stage2.json: 10k runs, seeds 204/205).

Usage: python make_figures_v2.py [--skip-toy]
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'overleaf' / 'figures'
FASTQR = 'reports/fastqr_20261120'

C_ORD, C_WARM, C_QR = '#9aa9b8', '#e8a33d', '#1f63d9'
C_TURBO, C_QR20, C_QR50, C_STOCK = '#1f63d9', '#16845b', '#c44b22', '#555555'

plt.rcParams.update({
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'legend.fontsize': 7.5,
})


def load(rel):
    return json.loads((ROOT / rel).read_text())


def ce_at(validation, call):
    return {v['call']: v['ce'] for v in validation}[call]


# ---------------------------------------------------------------- hero ----
def matched_runs():
    """Matched BF16 QR10 evidence from the fast-QR study (read in place).

    Stage 1: seed 204, 2,000 calls, fresh warm10/QR50 and BF16 Householder QR10
    in one launch (same-launch timing). Stage 2: BF16 QR10 to 10,000 calls on
    seeds 204/205 against the cached main TurboSOAP runs and FP32 QR10 refs.
    """
    rows = {r['arm']: r for r in
            load(f'{FASTQR}/summary.json')['training']['stage1_seed204_c2000']['rows']}
    stage2 = load(f'{FASTQR}/stage2.json')['seeds']
    return rows['bf16_w10_qr50'], rows['bf16_hh_qr10'], stage2


def hero():
    warm, qr10, stage2 = matched_runs()
    assert warm['launch'] == qr10['launch'] == 'this'
    wk, qk = warm['optimizer_by_kind'], qr10['optimizer_by_kind']
    turbo_cycle = 45 * wk['ordinary']['mean_ms'] + 4 * wk['warm']['mean_ms'] + wk['qr']['mean_ms']
    qr10_cycle = 45 * qk['ordinary']['mean_ms'] + 5 * qk['qr']['mean_ms']
    assert abs(turbo_cycle / 50 - warm['mean_optimizer_ms']) < 0.1
    assert abs(qr10_cycle / 50 - qr10['mean_optimizer_ms']) < 0.1

    fig = plt.figure(figsize=(10.2, 3.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.25, 1],
                          hspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    bx_top = fig.add_subplot(gs[0, 1])
    bx_bot = fig.add_subplot(gs[1, 1], sharex=bx_top)

    # (a) one 50-age cycle, cumulative optimizer time, measured in one launch
    rows = [('TurboSOAP\nwarm10/QR50', 1.0, wk, False),
            ('QR10, same\nimplementation', 0.0, qk, True)]
    for label, y, kinds, all_qr in rows:
        left = 0.0
        for age in range(1, 51):
            if age % 10:
                kind = 'o'
            elif all_qr or age == 50:
                kind = 'q'
            else:
                kind = 'w'
            cost = kinds[{'o': 'ordinary', 'w': 'warm', 'q': 'qr'}[kind]]['mean_ms']
            color = {'o': C_ORD, 'w': C_WARM, 'q': C_QR}[kind]
            ax.barh(y, cost, left=left, height=0.55, color=color,
                    edgecolor='white' if kind != 'o' else color,
                    linewidth=0.4 if kind != 'o' else 0)
            left += cost
        ax.text(left + 60, y, f'{left/1000:.2f} s', va='center', fontsize=8.5,
                fontweight='bold')
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xlim(0, qr10_cycle * 1.16)
    ax.set_ylim(-0.45, 1.75)
    ax.set_xlabel('Cumulative optimizer time over one 50-call cycle (ms)')
    ax.set_title('(a) Same refresh cadence; four of five refreshes approximate',
                 loc='left')
    o, w, q = (wk[k]['mean_ms'] for k in ('ordinary', 'warm', 'qr'))
    ax.legend(handles=[Patch(color=C_ORD, label=f'ordinary call ({o:.0f} ms)'),
                       Patch(color=C_WARM, label=f'call with warm refresh ({w:.0f} ms)'),
                       Patch(color=C_QR, label=f'call with QR refresh ({q:.0f} ms)')],
              loc='upper right', frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.02))
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)

    # (b) CE minus TurboSOAP's (same seed) over the 10k horizon; broken y-axis:
    # top = late-training scale, bottom = the early gap.
    seed_color = {'204': '#2563eb', '205': '#c2410c'}
    ends = {}
    for seed, v in stage2.items():
        a = v['analysis']
        turbo = a['cached_production']['ce']
        calls = [c for c in a['calls'] if c >= 50]
        for key, src, ls, lab in [
                ('bf16_qr10', a['ce']['bf16_hh_qr10'], '-', 'BF16 QR10 (matched)'),
                ('fp32_qr10', a['fp32_reference']['fp32_qr10'], '--', 'FP32 eager QR10')]:
            d = [src[str(c)] - turbo[str(c)] for c in calls]
            ends[(seed, key)] = d[-1]
            for axis in (bx_top, bx_bot):
                axis.plot(calls, d, ls=ls, color=seed_color[seed], lw=1.3,
                          marker='.' if ls == '-' else None, markersize=3,
                          label=f'{lab}, seed {seed}')
    spread = abs(stage2['204']['analysis']['cached_production']['ce']['10000']
                 - stage2['205']['analysis']['cached_production']['ce']['10000'])
    for axis in (bx_top, bx_bot):
        axis.axhspan(-spread, spread, color='#999999', alpha=0.2, lw=0)
        axis.axhline(0, color='#555555', lw=0.7)
        axis.grid(alpha=.18)
        axis.set_xscale('log')
        axis.set_xlim(45, 11000)
    bx_top.set_ylim(-0.016, 0.008)
    bx_bot.set_ylim(-0.135, -0.016)
    bx_top.spines['bottom'].set_visible(False)
    bx_bot.spines['top'].set_visible(False)
    bx_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle='none',
              color='k', mec='k', mew=0.8, clip_on=False)
    bx_top.plot([0], [0], transform=bx_top.transAxes, **kw)
    bx_bot.plot([0], [1], transform=bx_bot.transAxes, **kw)
    bx_bot.set_xlabel('Training calls (log scale)')
    bx_top.set_title('(b) CE relative to TurboSOAP, 10k calls', loc='left')
    bx_bot.legend(frameon=False, loc='lower right', fontsize=6.2, handlelength=1.6,
                  borderaxespad=0.2)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.14, top=0.91, wspace=0.34)
    x0 = bx_top.get_position().x0
    fig.text(x0 - 0.072, 0.525, 'CE minus TurboSOAP (same seed)', rotation=90,
             va='center', ha='center', fontsize=9)
    fig.savefig(OUT / 'hero.pdf')
    plt.close(fig)
    return dict(turbo_cycle_ms=turbo_cycle, qr10_cycle_ms=qr10_cycle,
                same_launch_step_ms=dict(warm=warm['mean_step_ms'], qr10=qr10['mean_step_ms']),
                same_launch_optimizer_ms=dict(warm=warm['mean_optimizer_ms'],
                                              qr10=qr10['mean_optimizer_ms']),
                same_launch_step_saving_pct=100 * (1 - warm['mean_step_ms'] / qr10['mean_step_ms']),
                ce_minus_turbo_at_10k={f'{s}/{k}': d for (s, k), d in ends.items()},
                turbo_seed_spread_10k=spread)


# ------------------------------------------------- relabelled figures ----
def full_horizon():
    s = load('reports/bf16_completion_20261027/summary.json')
    colors = {204: '#2563eb', 205: '#c2410c'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout='constrained')
    for r in s['per_seed']:
        c = r['validation_curve']
        x = [v['call'] for v in c if v['call'] > 0]
        axes[0].plot(x, [v['stock_ce'] for v in c if v['call'] > 0], '-',
                     color=colors[r['seed']], lw=1.5, label=f"FP32 QR10, seed {r['seed']}")
        axes[0].plot(x, [v['bf16_ce'] for v in c if v['call'] > 0], '--',
                     color=colors[r['seed']], lw=1.5, label=f"TurboSOAP, seed {r['seed']}")
        axes[1].plot([v['call'] for v in c], [v['delta'] for v in c], marker='.',
                     color=colors[r['seed']], label=str(r['seed']))
    axes[0].set(title='Full-horizon validation CE', xlabel='Training calls',
                ylabel='Cross-entropy')
    axes[1].set(title='TurboSOAP minus FP32 QR10 CE', xlabel='Training calls',
                ylabel='CE difference')
    axes[1].axhline(0, color='#555', lw=.7)
    axes[0].legend(frameon=False)
    axes[1].legend(title='Model seed', frameon=False)
    for a in axes:
        a.grid(alpha=.18)
    fig.savefig(OUT / 'full_horizon_v2.pdf')
    plt.close(fig)


def review_tradeoff():
    ctx = json.loads((HERE / 'review_context.json').read_text())
    controls = load('reports/bf16_refresh_controls_20261027/comparison.json')
    pilot = load('reports/bf16_learning_20261026/comparison.json')
    reference = {r['call']: r['ce'] for r in pilot['results'][0]['validation']}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 3.5),
                                      gridspec_kw={'width_ratios': [1, 1.15]})
    for row, label, color, marker in [
        (controls['cached_warm_prefix'], 'TurboSOAP warm10/QR50', C_TURBO, 'o'),
        (controls['results'][0], 'BF16 QR20', C_QR20, 's'),
        (controls['results'][1], 'BF16 QR50', C_QR50, '^'),
    ]:
        v = row['validation']
        left.plot([x['call'] for x in v], [x['ce'] - reference[x['call']] for x in v],
                  label=label, color=color, marker=marker, markersize=3)
    _, qr10, _ = matched_runs()
    qc = sorted((int(c), ce) for c, ce in qr10['ce'].items() if int(c) in reference)
    left.plot([c for c, _ in qc], [ce - reference[c] for c, ce in qc],
              label='BF16 QR10 (matched)', color='#6b21a8', marker='D', markersize=3)
    left.axhline(0, color='#777777', lw=1, label='FP32 QR10 (original)')
    left.set(xlabel='Training calls', ylabel='CE minus FP32 QR10',
             title='Seed-204 prefix learning', xlim=(0, 2000), ylim=(-.015, .15))
    left.grid(alpha=.18)
    left.legend(frameon=False, loc='upper right')
    rows = ctx['prefix_amortized_optimizer_events']
    labels = ['FP32 QR10', 'Original warm\n(FP32 state)', 'Compiled FP32 state\n(BF16 arithmetic)',
              'TurboSOAP', 'BF16 QR20', 'BF16 QR50', 'BF16 QR10\n(matched launch)']
    assert [r['case'] for r in rows] == ['qr10', 'original', 'compiled_fp32', 'bf16',
                                         'bf16_qr20', 'bf16_qr50']
    qk = qr10['optimizer_by_kind']
    rows = rows + [dict(case='bf16_qr10', amortized_ms=dict(
        ordinary=0.9 * qk['ordinary']['mean_ms'], warm=0.0, qr=0.1 * qk['qr']['mean_ms']))]
    pos = list(range(len(rows)))
    base = [0.] * len(rows)
    for key, label, color in [('ordinary', 'Ordinary', C_ORD), ('warm', 'Warm', C_WARM),
                              ('qr', 'QR', C_QR)]:
        vals = [r['amortized_ms'][key] for r in rows]
        right.barh(pos, vals, left=base, label=label, color=color, height=.65)
        base = [a + b for a, b in zip(base, vals)]
    for i, t in enumerate(base):
        right.text(t + 1, i, f'{t:.1f}', va='center', fontsize=7.5)
    right.set_yticks(pos, labels)
    right.invert_yaxis()
    right.set(xlabel='Amortized optimizer ms/call', xlim=(0, 110),
              title='Recorded prefix event accounting')
    right.legend(ncol=1, frameon=False, loc='center right')
    right.grid(axis='x', alpha=.18)
    right.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / 'review_tradeoff_v2.pdf')
    plt.close(fig)


# ------------------------------------------------ synthetic tracking ----
# Settings fixed before any result was inspected (20 Nov 2026):
TOY = dict(n=128, k=32, steps=1000, drift=0.01, spectrum_power=1.0,
           cov_beta=0.95, seed=0, window=(500, 650))


def tracking_toy():
    import torch
    spec = importlib.util.spec_from_file_location('bf16_soap', ROOT / 'bf16_soap.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, k, T = TOY['n'], TOY['k'], TOY['steps']
    gen = torch.Generator(device='cpu').manual_seed(TOY['seed'])
    lam = torch.arange(1, n + 1, dtype=torch.float64) ** -TOY['spectrum_power']
    U0, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    skews, samples = [], []
    for _ in range(T):
        s = torch.randn(n, n, generator=gen, dtype=torch.float64)
        s = s - s.T
        skews.append(torch.matrix_exp(TOY['drift'] * s / (s.norm() / 2 ** .5)))
        samples.append(torch.randn(n, k, generator=gen, dtype=torch.float64))

    def resid(c, qb):
        b = qb.T @ c @ qb
        off = b - torch.diag(torch.diag(b))
        return (off.norm() / b.norm().clamp_min(1e-30)).item()

    schedules = {'QR10': (10, 10), 'warm10/QR50': (10, 50), 'QR50': (50, 50)}
    out = {}
    for name, (refresh, repair) in schedules.items():
        U = U0.clone()
        c = None
        q = None
        trace = []
        for t in range(T):
            U = U @ skews[t]
            g = (U * lam.sqrt()) @ samples[t]
            gg = (g @ g.T).float()
            c = gg if c is None else TOY['cov_beta'] * c + (1 - TOY['cov_beta']) * gg
            if q is None:
                ev, vec = torch.linalg.eigh(c)
                q = vec.flip(-1).to(torch.bfloat16)
            age = t  # first call initializes; refreshes at positive ages
            if age > 0 and age % repair == 0:
                qn, _ = mod.qr_basis(c[None].to(dev), q[None].to(dev), torch.bfloat16)
                q = qn[0].cpu()
            elif age > 0 and age % refresh == 0:
                q = mod.warm_basis(c[None].to(dev).to(torch.bfloat16), q[None].to(dev),
                                   torch.bfloat16, torch.bfloat16)[0].cpu()
            trace.append(resid(c.double(), q.double()))
        out[name] = trace

    colors = {'QR10': '#6b21a8', 'warm10/QR50': C_TURBO, 'QR50': C_QR50}
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    lo, hi = TOY['window']
    for name, tr in out.items():
        mean = sum(tr[100:]) / len(tr[100:])
        ax.plot(range(lo, hi), tr[lo:hi], color=colors[name], lw=1.2,
                label=f'{name}\n(mean {mean:.3f}, steps 100–{T})')
    for x in range(lo, hi):
        if x % 50 == 0:
            ax.axvline(x, color='#bbbbbb', lw=0.6, ls=':')
    ax.set(xlabel='Step', ylabel=r'$r_C$ (off-diagonal fraction)',
           title='Synthetic drifting covariance (illustrative)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(OUT / 'tracking_toy.pdf')
    plt.close(fig)
    return {name: dict(mean=sum(tr[100:]) / len(tr[100:]),
                       at_age_x90=[tr[a] for a in range(190, T, 50)][:3])
            for name, tr in out.items()}


if __name__ == '__main__':
    info = dict(hero=hero())
    full_horizon()
    review_tradeoff()
    if '--skip-toy' not in sys.argv:
        info['toy'] = tracking_toy()
        info['toy_settings'] = TOY
    elif (HERE / 'figures_v2.json').exists():  # keep the last toy results
        prev = json.loads((HERE / 'figures_v2.json').read_text())
        info.update({k: prev[k] for k in ('toy', 'toy_settings') if k in prev})
    (HERE / 'figures_v2.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
