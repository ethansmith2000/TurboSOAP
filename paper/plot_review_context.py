"""Render a manuscript review figure from existing scalar evidence, CPU only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    context = json.loads((HERE / 'review_context.json').read_text())
    controls_path = ROOT / 'reports/bf16_refresh_controls_20261027/comparison.json'
    pilot_path = ROOT / 'reports/bf16_learning_20261026/comparison.json'
    controls = json.loads(controls_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    reference = {r['call']: r['ce'] for r in pilot['results'][0]['validation']}
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'soap-review-20261029',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.3),
                                      gridspec_kw={'width_ratios': [1, 1.15]})
    for row, label, color, marker in [
        (controls['cached_warm_prefix'], 'Full BF16 warm10/QR50', '#1f63d9', 'o'),
        (controls['results'][0], 'BF16 QR20', '#16845b', 's'),
        (controls['results'][1], 'BF16 QR50', '#c44b22', '^'),
    ]:
        values = row['validation']
        left.plot([v['call'] for v in values],
                  [v['ce'] - reference[v['call']] for v in values],
                  label=label, color=color, marker=marker, markersize=3)
    left.axhline(0, color='#777777', linewidth=1, label='Stock QR10 reference')
    left.set(xlabel='Training calls', ylabel='CE minus original stock QR10',
             title='Seed-204 prefix learning', xlim=(0, 2000), ylim=(-.005, .15))
    left.grid(alpha=.18)
    left.legend(fontsize=8, frameon=False, loc='upper right')
    rows = context['prefix_amortized_optimizer_events']
    labels = ['Stock QR10', 'Original warm', 'Compiled FP32 state\n(BF16 arithmetic)',
              'Full BF16 warm', 'BF16 QR20', 'BF16 QR50']
    position = list(range(len(rows)))
    base = [0.] * len(rows)
    for key, label, color in [
        ('ordinary', 'Ordinary', '#66829c'),
        ('warm', 'Warm', '#edb54a'),
        ('qr', 'QR', '#1f63d9'),
    ]:
        values = [r['amortized_ms'][key] for r in rows]
        right.barh(position, values, left=base, label=label, color=color, height=.65)
        base = [a + b for a, b in zip(base, values)]
    for i, total in enumerate(base):
        right.text(total + 1, i, f'{total:.1f}', va='center', fontsize=8)
    right.set_yticks(position, labels)
    right.invert_yaxis()
    right.set(xlabel='Amortized optimizer ms/call', xlim=(0, 110),
              title='Recorded prefix event accounting')
    right.legend(ncol=3, fontsize=8, frameon=False, loc='lower right')
    right.grid(axis='x', alpha=.18)
    right.set_axisbelow(True)
    fig.subplots_adjust(left=.065, right=.98, top=.88, bottom=.19, wspace=.65)
    fig.text(.5, .035,
             'Timing: calls 101–2000. First four bars: development pilot; last two: new controls. '
             'Cross-launch context, not causal attribution.',
             ha='center', fontsize=8, color='#444444')
    paths = [HERE / 'figures/review_tradeoff.svg', HERE / 'figures/review_tradeoff.png']
    fig.savefig(paths[0], metadata={'Date': None})
    fig.savefig(paths[1], dpi=180)
    plt.close(fig)
    manifest = dict(
        generator='plot_review_context.py', generator_sha256=sha(Path(__file__)),
        matplotlib=matplotlib.__version__,
        inputs={str(p.relative_to(ROOT)): sha(p)
                for p in [HERE / 'review_context.json', controls_path, pilot_path]},
        outputs={str(p.relative_to(HERE)): sha(p) for p in paths},
        execution='CPU plotting from saved scalars; no GPU or new training')
    (HERE / 'review_figure.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Rendered review_tradeoff.svg/png with source hashes.')


if __name__ == '__main__':
    main()
