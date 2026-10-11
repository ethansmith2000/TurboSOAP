"""Source-backed fixed-update V-ablation report and standalone figure."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import sys

NAMES=dict(qr_carry='QR / reindex',qr_overlap='QR / overlap',fro_carry='Fro .5 / carry',fro_overlap='Fro .5 / overlap')
def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text());r=json.loads((b/'screen.json').read_text())
    assert s['fixed_update_qualified']
    unique=[x for x in s['rows'] if not x['repeat']]
    effects=[]
    for beta in [.99,.999]:
        for basis in ['qr','fro']:
            carry=next(x for x in unique if x['kind']==basis+'_carry' and x['beta']==beta)
            overlap=next(x for x in unique if x['kind']==basis+'_overlap' and x['beta']==beta)
            costs={kind:st.mean(x['optimizer_ms'] for x in s['rows'] if x['kind']==basis+'_'+kind and x['beta']==beta) for kind in ['carry','overlap']}
            samples={kind:[x['optimizer_ms'] for x in s['rows'] if x['kind']==basis+'_'+kind and x['beta']==beta] for kind in ['carry','overlap']}
            effects.append(dict(beta=beta,basis=basis,ce_delta=overlap['ce']-carry['ce'],
                accuracy_delta=overlap['accuracy']-carry['accuracy'],optimizer_cost_change=costs['overlap']/costs['carry']-1,
                carry_optimizer_ms=costs['carry'],overlap_optimizer_ms=costs['overlap'],
                observed_cost_change_range=[min(samples['overlap'])/max(samples['carry'])-1,max(samples['overlap'])/min(samples['carry'])-1]))
    assert all(x['ce_delta']>0 for x in effects), 'Review the study-specific outcome text'
    maximum_cost_repeat_spread=max(x['optimizer_range_over_mean'] for x in s['controls'])
    lines=['# Diagonal variance transport: fixed-update CIFAR ablation','',
        f"Completed 2026-10-06. Fixed-update qualification: **{s['fixed_update_qualified']}**. "
        f"{s['trainer_steps']:,} updates across twelve arms in {s['wall_seconds']:.1f}s.",'',
        'Only V handling changes within each QR/Fro pair. Seed 139, covariance .99/.999, '
        'Adam(.9,.999), both factors, strict highest M, high basis/V arithmetic, LR .0005 '
        'and original 200-epoch schedule. Fro uses fixed radius .5/cubic2, warm20/QR200. '
        'QR refreshes every 20. Each arm gets identical initial model/RNG/input and a fresh optimizer. '
        'All arms stop at 3500; no wall endpoint or wall-quality gate is collected.','',
        '## Paired effects at 3500','',
        '| Cov beta | Basis | Overlap minus carry CE | Accuracy change (pp) | Mean carry ms | Mean overlap ms | Optimizer cost change |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for x in effects:
        lines.append(f"| {x['beta']} | {x['basis']} | {x['ce_delta']:+.6f} | {100*x['accuracy_delta']:+.3f} | {x['carry_optimizer_ms']:.4f} | {x['overlap_optimizer_ms']:.4f} | {100*x['optimizer_cost_change']:+.2f}% |")
    lines+=['',
        'Overlap worsens CE in every tested pair; it does not close the warm/QR gap. '
        'Retain carry/reindex; no overlap promotion or seed expansion. Negative CE delta '
        'would indicate improvement within that basis/beta. Cost means include '
        'both .99 execution repeats; .999 has one arm per method. Identical-seed repeats '
        'are execution checks, not independent learning replication or confidence intervals.','',
        '| Cov beta | Method | CE at 3500 | Accuracy (%) |',
        '|---:|---|---:|---:|']
    for x in unique:lines.append(f"| {x['beta']} | {NAMES[x['kind']]} | {x['ce']:.6f} | {100*x['accuracy']:.3f} |")
    lines+=['','![Learning and complete optimizer effects](variance_screen.png)','',
        '## What the variance operation means','',
        'Carry/reindex keeps V at warm events and permutes it with QR column ordering at '
        'QR events. Overlap computes Vnew=(U_L squared)^T Vold (U_R squared), where '
        'U=Qold^T Qnew, at every refresh, clamped nonnegative. It uses the existing '
        'ResearchSOAP path after strict M restores high matmul precision. This is not '
        'a strict-precision V comparison. The same basis policy is held fixed, while '
        'subsequent gradients/covariances evolve along each actual trajectory.','',
        'The completed float64 moment oracle establishes diagonal transport orientation '
        'and its path dependence: discarded cross moments prevent exact history recovery. '
        'Many small transports need not recover the diagonal from one large net rotation. '
        'That prior evidence was read and hashed, without rerunning it. This study measures '
        'learning consequences of the existing approximation; it does not establish that '
        'transport reconstructs the full historical second moment.','',
        'Offline same-basis shadows compare both V policies on identical incoming state '
        'and actual new Q/M. Update differences, cosine and V mass ratio are recorded in '
        'summary.json. They quantify local policy effects, not full-history error or a '
        'causal explanation of the full learning trajectory. No shadow feeds back.','',
        '## Qualification and cost scope','',
        f"All 18 historical fixed observations and their optimizer hashes exact: "
        f"{all(x['exact'] for x in s['anchors'])}. Four repeated methods exact: "
        f"{all(x['exact'] for x in s['controls'])}. Placement complete: {s['placement_recorded']}; "
        f"in-arm compiler stable: {s['compiler_stable']}.",'',
        f"{s['refresh_guards']:,} finite/Gram/nonnegative-V guards, {s['matrix_probes']:,} targeted "
        f"M probes and {s['variance_formula_checks']:,} actual-V formula checks. Explicit "
        'two-sided matched-high formula must equal executed V bitwise. CPU orientation '
        'tests compare one transformation against a full diagonal Kronecker oracle. '
        'Selected-basis M uses actual V in both updates, so it cannot substitute for '
        'the independent V formula check. Full strict-refresh differences remain '
        'separately recorded; prior QR/reset failures are preserved.','',
        '| Method | Exact repeat | Wall range/mean | Optimizer range/mean |',
        '|---|---|---:|---:|']
    for x in s['controls']:lines.append(f"| {NAMES[x['kind']]} | {x['exact']} | {100*x['wall_range_over_mean']:.3f}% | {100*x['optimizer_range_over_mean']:.3f}% |")
    lines+=['',
        f'Optimizer repeat spreads reach {100*maximum_cost_repeat_spread:.2f}%, larger than the observed mean cost changes. '
        'Those changes are descriptive and do not establish a stable marginal cost. '
        'Cost-plot whiskers at .99 show extrema of all ratios between the two execution '
        'repeats, not confidence intervals; .999 has a single pair and no repeat range. '
        'No new timing threshold is inferred from this table. Prior wall failures remain '
        'unchanged; no time-to-quality winner is selected. Complete optimizer CUDA-event '
        'times include covariance accumulation, Adam updates, both bases, M/V transport, '
        'casts and host dispatch. First 50 excluded from steady event means only. '
        'Init, evaluation, offline guards/probes/hashes/logging are excluded. Offline '
        'checks may perturb subsequent execution; these are instrumented actual costs, '
        'not uninstrumented throughput. No synthetic timing is attached to losses.','',
        '## Sampled state diagnostics','',
        '| Cov beta | Method | Event | Max physical M error (%) | Max same-basis V difference (%) | Max same-basis update difference (%) | Full strict-reference failures / samples |',
        '|---:|---|---|---:|---:|---:|---:|']
    for x in s['diagnostics']:
        row=next(z for z in s['rows'] if z['label']==x['label'])
        if row['repeat']:continue
        lines.append(f"| {row['beta']} | {NAMES[row['kind']]} | {x['event']} | {100*x['max_world_m']:.5f} | {100*x['max_same_basis_v_difference']:.3f} | {100*x['max_same_basis_update_difference']:.3f} | {x['full_strict_failures']} / {x['count']} |")
    lines+=['',
        'Each V/update difference compares the two policies on the same incoming state '
        'and actual basis. Rows from distinct trajectories need not have identical '
        'diagnostics. Full strict-reference failures compare a separately recomputed '
        'refresh; they are retained separately from the selected-basis M and matched-high '
        'V implementation gates. No exact-history V error is available for these states.','',
        '## Decision and evidence','',
        'See [DECISION.md](DECISION.md) for the outcome-specific next bounded task. This is one seed '
        'at an early point of the original schedule, not convergence or a tuned-method '
        'ranking. Useful effects require paired seeds 271/811 and longer horizons. '
        'Larger-factor overlap natural cycles/precision/compilation and current wall '
        'stability remain unqualified. No production/default changes or broad sweep.','',
        '206 CPU tests and 21 focused tests passed; five additional focused checks, including observer integration, passed before launch. Frozen execution sources, '
        'config/protocol, metrics/logs, hashes and masks are retained. Reporting source '
        'is separately frozen in analysis_source. No weights/checkpoints/tensor snapshots '
        'saved. One claimed GPU, private manifold supervisor. Existing source and '
        'historical evidence preserved. See [PROVENANCE.md](PROVENANCE.md) and artifact_inventory.json.']
    (b/'README.md').write_text('\n'.join(lines)+'\n');(b/'effects.json').write_text(json.dumps(effects,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors=dict(qr_carry='#586673',qr_overlap='#2f8b9c',fro_carry='#ca732b',fro_overlap='#639445')
    for ax,beta in zip(axes[0],[.99,.999]):
        for run in r['runs']:
            if run['covariance_beta']!=beta or run.get('repeat_of'):continue
            ax.plot([o['step'] for o in run['observations']],[o['validation']['ce'] for o in run['observations']],marker='o',label=NAMES[run['kind']],color=colors[run['kind']])
        ax.set_title(f'Paired actual learning: covariance {beta}');ax.set_ylabel('Validation CE');ax.set_xlabel('Training updates');ax.grid(alpha=.2)
    xx=np.arange(4);labels=[f"{x['basis']} / {x['beta']}" for x in effects]
    for ax,key,scale,title,ylabel in [(axes[1,0],'ce_delta',1,'V overlap effect at 3500','Overlap minus carry CE'),(axes[1,1],'optimizer_cost_change',100,'Observed mean optimizer cost difference','Overlap minus carry (%)')]:
        values=[scale*x[key] for x in effects];ax.bar(xx,values,color=['#3b9678' if v<0 else '#b5674b' for v in values]);ax.axhline(0,color='#555',lw=1);ax.set_xticks(xx,labels);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if key=='optimizer_cost_change':
            indices=[i for i,x in enumerate(effects) if x['beta']==.99]
            errors=[[values[i]-100*effects[i]['observed_cost_change_range'][0] for i in indices],
                    [100*effects[i]['observed_cost_change_range'][1]-values[i] for i in indices]]
            ax.errorbar(indices,[values[i] for i in indices],yerr=errors,fmt='none',ecolor='#333',capsize=5)
            ax.set_xlabel('.99 whiskers: execution ranges, not CI; .999: one pair',fontsize=9)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=4,fontsize=9)
    fig.savefig(b/'variance_screen.png',dpi=160);plt.close(fig)
    a=b/'analysis_source';a.mkdir(exist_ok=True);shutil.copy2(Path(__file__),a/Path(__file__).name)
    (b/'analysis_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()},indent=2)+'\n')

if __name__=='__main__':main()
