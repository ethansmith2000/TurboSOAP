"""Retained figure and decision evidence for the warmed budget campaign."""
import json
from pathlib import Path
import sys


def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text())
    lines=['# New 90-second comparison after disposable warmup','',
        f"Qualification: **{s['qualified']}**. {s['primary_updates']:,} measured updates plus "
        f"{s['warmup_updates']:,} disposable policy warmup updates in {s['all_in_seconds']:.1f}s all-in.",'',
        '| Method | Fixed3500 mean wall s | Wall spread | Optimizer spread | 90-second endpoint steps | Endpoint CE | Mean endpoint CE |',
        '|---|---:|---:|---:|---|---|---:|']
    for x in s['conditions']:
        mean=f"{x['mean_wall_ce']:.6f}" if x['mean_wall_ce'] is not None else 'missing'
        lines.append(f"| {x['kind']} | {x['mean_fixed_training_seconds']:.4f} | {100*x['wall_range_over_mean']:.3f}% | "
            f"{100*x['optimizer_range_over_mean']:.3f}% | {x['endpoint_steps']} | {x['endpoint_ces']} | {mean} |")
    lines+=['',
        'Use all three repeats per method and unchanged 3% fixed3500 wall / 2% steady optimizer limits. '
        'The repeat gate uses identical3500 work, not varying final endpoint lengths. Every endpoint '
        'must be the first25-update crossing of90 seconds with <=1s overshoot; no interpolation.',
        '', '![New measured endpoints and discarded warmup](warmed_budget.png)', '',
        '| Arm | Measured updates | Endpoint seconds | Overshoot s | CE | Discarded warmup s | Arm all-in s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in s['rows']:
        lines.append(f"| {x['label']} | {x['steps']} | {x['endpoint_seconds']} | {x['overshoot_seconds']} | "
            f"{x['wall_ce']} | {x['warmup_seconds']:.3f} | {x['all_in_seconds']:.3f} |")
    lines+=['',
        f"Exact historical/repeated fixed observations and probes: {s['numerical_exact']}; "
        f"400-update warmups exact and restored: {s['warmup_exact_and_restored']}; "
        f"observer complete: {s['observer_complete']}; measured compiler stable: {s['compiler_stable']}. "
        f"{s['refresh_guards']} guards and {s['matrix_probes']} targeted probes. "
        f"{s['full_strict_reference_failures']} full strict QR/reset reference disagreements retained separately.",'',
        'Global smoke compiles the model and checks training/evaluation parity. Each arm then warms '
        'its own optimizer policy for a fixed400 updates on disposable state, covering two QR200 cycles. '
        'Weights and complete RNG are restored, the warm optimizer discarded, and the measured optimizer '
        'starts empty at update1. No warmed training state survives into measurement. Initialization '
        'and all first50 measured updates count toward the budget; first50 are omitted only from event means.', '',
        'The90-second budget is measured training time: augmentation/model/optimizer/host dispatch and '
        'synchronization, excluding setup, disposable warmup, evaluation, offline guards/probes/hashes, '
        'observer flushes and logging. All-in runtime and warmup costs are retained separately. '
        'This is instrumented local scope, not total job latency or deployment throughput.', '',
        'Seed139, covariance .99, Adam(.9,.999), both factors, strict highest M/high basis, V carry/reindex. '
        'QR20 versus Fro .5/cubic2 warm20/QR200; original recipe/data/LR .0005/200-epoch schedule. '
        'Main/autograd both CPU116, original masks restored. No clocks, GC, other processes or defaults changed.', '',
        'Repeat endpoints share one seed and differ by stop step. Report means/ranges descriptively; '
        'they are not independent-seed confidence intervals or tuned-method comparisons. Prior failed '
        'endpoints retain their original scope. A failed campaign gate withholds wall-quality interpretation.', '',
        'Small-factor ViT provides learning-fidelity evidence. Next use a shallow larger-dimension '
        'trainer with actual1536+ factors to assess matched learning and full optimizer cost. Dimension-dependent '
        'step scaling and cheap scalar/row/column EMA normalization remain hypotheses for separate bounded '
        'ablations with reduction/state-update costs included.', '',
        'No saved weights/checkpoints/tensor snapshots or new profiler traces. Frozen source/config/protocol, '
        'warmup/reset hashes, metrics/logs/observer/mask records and inventories retained. '
        'See [decision](DECISION.md), [protocol](PROTOCOL.md) and [provenance](PROVENANCE.md).']
    (b/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for kind,color in [('qr','#586673'),('fro','#2f8b9c')]:
        rows=[r for r in s['rows'] if r['kind']==kind]
        axes[0].scatter([r['endpoint_seconds'] for r in rows],[r['wall_ce'] for r in rows],
                        c=color,label=kind,s=55)
    axes[0].axvline(90,color='#999999',linestyle='--',linewidth=1)
    axes[0].set(xlabel='Measured training seconds',ylabel='Validation CE',title='New wall endpoints (one seed)')
    axes[0].legend();axes[0].grid(alpha=.2)
    for i,r in enumerate(s['rows']):
        axes[1].bar(i,r['warmup_seconds'],color='#586673' if r['kind']=='qr' else '#2f8b9c')
    axes[1].set_xticks(range(6),[r['label'] for r in s['rows']])
    axes[1].set(ylabel='Seconds excluded from training budget',title='Fixed400-update disposable warmup')
    fig.savefig(b/'warmed_budget.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
