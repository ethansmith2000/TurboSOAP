"""Standalone report and scientific figure from actual paired CIFAR evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

NAMES=dict(qr='QR20',fro05='Radius .5 / cubic2',fro06='Radius .6 / cubic2',fro06strong='Radius .6 / cubic3')

def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text());r=json.loads((b/'screen.json').read_text())
    unique=[x for x in s['rows'] if not x['repeat']]
    primary=[x for x in unique if x['beta']==.99]
    base=next(x for x in primary if x['kind']=='fro05');qr=next(x for x in primary if x['kind']=='qr')
    def mean_cost(kind):
        values=[x['optimizer_ms'] for x in s['rows'] if x['kind']==kind and x['beta']==.99]
        return sum(values)/len(values)
    lines=['# Radius .6: paired actual CIFAR learning','',
        f"Completed 2026-10-06. {s['trainer_steps']:,} actual updates across 16 arms, "
        f"{s['wall_seconds']:.1f}s managed execution. Fixed-update qualification: "
        f"**{s['fixed_update_qualified']}**. Covariance .99 wall qualification: "
        f"**{s['beta099_wall_qualified']}**. Covariance .999 wall endpoints remain exploratory.",'',
        'Increasing radius alone gives essentially unchanged CE: .99 delta '
        '-.000095 and .999 delta +.000338 versus radius .5/cubic2. Cubic3 gives '
        'a small .99 gain (-.003876) and a .999 regression (+.005930), while '
        'adding about 1.86% mean optimizer cost at .99. All warm variants still '
        'trail QR at both fixed-update endpoints. No radius .6 policy is promoted.','',
        'The new wall gate fails: QR range/mean 3.383% exceeds the declared 3% limit; '
        'its optimizer variation 1.217% passes 2%. The three warm policy wall gates '
        'pass. All budget endpoints remain exploratory for the primary matched '
        'comparison. The previous fixed-placement qualification retains its own '
        'scope; it does not guarantee this later experiment.','',
        'Seed 139, both factors, strict M, high basis arithmetic, warm20/QR200, '
        'fixed V carry/reindex, original 200-epoch schedule and LR .0005. '
        'All arms start from identical model/RNG/input streams with fresh optimizer. '
        'Primary quality endpoint is 3500 updates; secondary endpoint is the first '
        '25-update boundary reaching 90 newly measured training seconds.','',
        '## Fixed-update learning','',
        '| Cov beta | Policy | CE at 3500 | Accuracy (%) | Mean optimizer ms | Mean training wall at 3500 (s) |',
        '|---:|---|---:|---:|---:|---:|']
    for x in unique:
        group=[z for z in s['rows'] if (z['kind'],z['beta'])==(x['kind'],x['beta'])]
        cost=sum(z['optimizer_ms'] for z in group)/len(group)
        wall=sum(z['training_seconds'] for z in group)/len(group)
        lines.append(f"| {x['beta']} | {NAMES[x['kind']]} | {x['ce']:.6f} | {100*x['accuracy']:.3f} | {cost:.4f} | {wall:.3f} |")
    lines+=['','At covariance .99, paired fixed 3500 changes relative to radius .5/cubic2:','']
    for x in primary:
        if x['kind'] in ('fro06','fro06strong'):
            lines.append(f"- {NAMES[x['kind']]}: CE delta {x['ce']-base['ce']:+.6f}, "
                         f"mean optimizer cost change {100*(mean_cost(x['kind'])/mean_cost('fro05')-1):+.2f}%. "
                         f"CE delta versus QR: {x['ce']-qr['ce']:+.6f}.")
    lines+=['','## New measured-wall endpoints','',
        '| Cov beta | Arm | Updates at 90s | CE | Actual seconds | Overshoot (s) |',
        '|---:|---|---:|---:|---:|---:|']
    for x in s['rows']:
        w=x['budget']
        values=f"{w['step']} | {w['ce']:.6f} | {w['seconds']:.3f} | {w['overshoot_seconds']:.3f}" if w else 'capped | unavailable | unavailable | unavailable'
        lines.append(f"| {x['beta']} | {x['label']} | {values} |")
    lines+=['',
        'All 16 wall endpoints are shown, including every repeat. Fixed-update cost means '
        'include all three .99 runs; .999 has one. A 25-update crossing boundary can '
        'change quality between repeats despite exact fixed-update training. '
        'Training wall includes model, optimizer and host dispatch; excludes initialization, '
        'evaluation, offline probes, hashes and logging. This is an operational budget, '
        'not equal cumulative optimizer time. LR is indexed by updates without horizon '
        'retuning. Historical failed 90-second comparisons remain exploratory and unchanged. '
        'No synthetic timing enters these learning endpoints.','',
        '## Three-repeat qualification at covariance .99','',
        '| Policy | Exact model/optimizer/metrics/probes | Wall range/mean | Optimizer range/mean | Gate |',
        '|---|---|---:|---:|---|']
    for c in s['controls']:
        lines.append(f"| {NAMES[c['kind']]} | {c['exact']} | {100*c['wall_range_over_mean']:.3f}% | {100*c['optimizer_range_over_mean']:.3f}% | {c['repeat_qualified']} |")
    lines+=['',
        'All three repeats enter each range/mean; limits remain 3% wall/2% optimizer. '
        'No trimming. Hashes and timing are compared at 3500 because final wall crossing '
        'may differ in update count. Repeats measure execution stability, not independent '
        'learning replication. .999 has one trajectory per policy and lacks its own '
        'wall-repeat qualification. Main/autograd CPUs 116/117 are recorded and restored.','',
        f"Placement records complete: {s['placement_recorded']}; in-arm compiler stable: "
        f"{s['compiler_stable']}. All 24 historical matched QR/.5 observations exact: "
        f"{all(x['exact'] for x in s['historical_anchors'])}. "
        f"{s['refresh_guards']:,} refresh guards; {s['matrix_probes']:,} targeted matrix probes.",'',
        'Per-refresh targeted criteria stay selected-basis update<1e-6, physical M<1% '
        'and normalized Gram<.05. Full strict-refresh disagreements, including QR/reset '
        'events, remain separate in summary.json. No old gate was relaxed.','',
        '## Retained QR timing diagnostics','',
        '| QR arm | Model/backward host (s) | Model/backward thread CPU (s) | Optimizer host (s) |',
        '|---|---:|---:|---:|']
    for run in r['runs']:
        if run['kind']=='qr' and run['covariance_beta']==.99:
            hh=[h for h in run['history'] if h['step']<=3500]
            host=sum(h['host']['phases']['model_backward']['host_wall_ms'] for h in hh)/1000
            cpu=sum(h['host']['phases']['model_backward']['thread_cpu_ms'] for h in hh)/1000
            opt=sum(h['host']['phases']['optimizer']['host_wall_ms'] for h in hh)/1000
            lines.append(f"| {run['label']} | {host:.3f} | {cpu:.3f} | {opt:.3f} |")
    lines+=['',
        'These recorded intervals track the QR wall difference despite stable compiler '
        'counters and recorded 116/117 placement. They do not establish a cause or '
        'measure isolated GPU kernels. Host/device intervals overlap; do not add '
        'their totals. No completed experiment was rerun to diagnose this failure.','',
        '![Actual learning and repeat qualification](radius_learning.png)','',
        '## Interpretation and follow-up','',
        'This is one seed at an early point on the original 200-epoch schedule. '
        'Fixed-update results can identify a candidate for seed271/811 replication; '
        'they cannot establish convergence, a tuned-method ranking or a default. '
        'Wall-quality interpretation is limited to .99 if its declared gate passes; '
        'otherwise all budget endpoints remain exploratory. Numerical correction '
        'and learning differences alone do not prove a mechanism. Larger-model, '
        'BF16 optimizer, natural compiled state ownership and exact H200 reproduction '
        'remain open. See DECISION.md for the result-specific next task.','',
        '## Evidence and retention','',
        '202 CPU tests passed before launch; nine focused policy checks passed. '
        'Frozen execution sources/config/protocol and SHA256 are in manifest.json; '
        'separate reporting source in analysis_source. No model weights, checkpoints '
        'or tensor snapshots saved. One GPU3 claim via private manifold supervisor; '
        'all historical evidence and existing dirty Python files preserved. '
        'See PROVENANCE.md and artifact_inventory.json for final integrity/resources.']
    (b/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors=dict(qr='#606975',fro05='#3474b0',fro06='#ce7529',fro06strong='#359067')
    for ax,beta in zip(axes[0],(.99,.999)):
        for run in r['runs']:
            if run['covariance_beta']!=beta or run.get('repeat_of'):continue
            oo=[o for o in run['observations'] if 'fixed_updates' in o['reasons']]
            ax.plot([o['step'] for o in oo],[o['validation']['ce'] for o in oo],marker='o',label=NAMES[run['kind']],color=colors[run['kind']])
        ax.set_title(f'Paired fixed-update learning: covariance {beta}');ax.set_xlabel('Training updates');ax.set_ylabel('Validation CE');ax.grid(alpha=.2)
    kinds=list(NAMES);xx=np.arange(4)
    for i,beta in enumerate((.99,.999)):
        values=[[x['budget']['ce'] for x in s['rows'] if x['kind']==k and x['beta']==beta and x['budget']] for k in kinds]
        means=[sum(v)/len(v) if v else float('nan') for v in values]
        errors=[[mean-min(v) if v else 0 for mean,v in zip(means,values)],
                [max(v)-mean if v else 0 for mean,v in zip(means,values)]]
        axes[1,0].bar(xx+(i-.5)*.36,means,yerr=errors,capsize=3,width=.34,label=f'Covariance {beta}'+(' (exploratory)' if beta==.999 or not s['beta099_wall_qualified'] else ''),color=['#417ba8','#c18b4a'][i])
    axes[1,0].set_title('New 90-second CE: mean and repeat range');axes[1,0].set_ylabel('CE');axes[1,0].set_xticks(xx,['QR','.5 / 2','.6 / 2','.6 / 3']);axes[1,0].legend(fontsize=8);axes[1,0].grid(axis='y',alpha=.2);axes[1,0].set_axisbelow(True)
    cc=s['controls']
    axes[1,1].bar(xx-.17,[100*c['wall_range_over_mean'] for c in cc],width=.32,label='Training wall',color='#667fa3')
    axes[1,1].bar(xx+.17,[100*c['optimizer_range_over_mean'] for c in cc],width=.32,label='Optimizer',color='#3d916c')
    axes[1,1].axhline(3,color='#667fa3',ls='--',lw=1,label='Wall 3% limit');axes[1,1].axhline(2,color='#3d916c',ls=':',lw=1,label='Optimizer 2% limit')
    axes[1,1].set_title('Three-repeat variation at covariance .99');axes[1,1].set_ylabel('Range / mean (%)');axes[1,1].set_xticks(xx,['QR','.5 / 2','.6 / 2','.6 / 3']);axes[1,1].legend(fontsize=8);axes[1,1].grid(axis='y',alpha=.2);axes[1,1].set_axisbelow(True)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=4,fontsize=9)
    fig.savefig(b/'radius_learning.png',dpi=160);plt.close(fig)
    a=b/'analysis_source';a.mkdir(exist_ok=True);shutil.copy2(Path(__file__),a/Path(__file__).name)
    (b/'analysis_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()},indent=2)+'\n')

if __name__=='__main__':main()
