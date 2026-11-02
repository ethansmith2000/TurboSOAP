"""CPU-only, source-backed final bounded-study report and exportable figures."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bf16_completion_20261027'
INPUTS={
    'seed204':'reports/bf16_replicate_20261027_seed204/comparison.json',
    'seed205':'reports/bf16_full_horizon_20261026_seed205/comparison.json',
    'controls':'reports/bf16_refresh_controls_20261027/comparison.json',
    'qr_profile':'reports/bf16_qr_attribution_20261027/result.json',
    'review204':'reports/bf16_replicate_20261027_seed204/monitor_review.json',
    'profile_audit':'automation/qr_profile_review.json',
    'control_audit':'automation/refresh_controls_review.json',
}


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={k:json.loads((ROOT/v).read_text()) for k,v in INPUTS.items()}
    pairs=[(204,data['seed204']['cached_quality_reference'],data['seed204']['candidate']),
           (205,*data['seed205']['results'])]
    rows=[]
    for seed,stock,warm in pairs:
        sv={v['call']:v['ce'] for v in stock['validation']};wv={v['call']:v['ce'] for v in warm['validation']}
        assert all(c in sv for c in wv) and max(wv)==10000
        curve=[dict(call=c,stock_ce=sv[c],bf16_ce=wv[c],delta=wv[c]-sv[c]) for c in sorted(wv)]
        delta=curve[-1]['delta'];peak=max(curve,key=lambda r:r['delta'])
        rows.append(dict(seed=seed,stock_final_ce=sv[10000],bf16_final_ce=wv[10000],ce_delta=delta,
            perplexity_increase_percent=100*math.expm1(delta),peak_observed_ce_gap=peak,
            matrix_state_bytes=warm['matrix_state_bytes'],peak_gpu_bytes=warm['peak_gpu_bytes'],
            max_physical_m_error=warm['max_physical_m_error'],max_normalized_gram=warm['max_normalized_gram'],
            validation_curve=curve))
    controls=data['controls'];short=[];cached=controls['cached_warm_prefix']
    for r in [cached,*controls['results']]:
        assert r['clean_cycle_calls']==1900
        short.append(dict(case=r['case'],final_ce=r['validation'][-1]['ce'],mean_step_ms=r['mean_step_ms'],
            mean_optimizer_ms=r['mean_optimizer_ms'],validation=r['validation']))
    profile=data['profile_audit']['rows']
    assert len(profile)==8 and all(r['numerical']['finite'] for r in profile)
    summary=dict(status='completed',per_seed=rows,mean_paired_ce_delta=sum(r['ce_delta'] for r in rows)/2,
        seed204_matched_prefix_timing=data['seed204']['paired_prefix_timing'],
        seed205_matched_full_timing=dict(calls=[101,10000],stock_step_ms=data['seed205']['results'][0]['mean_step_ms'],
            bf16_step_ms=data['seed205']['results'][1]['mean_step_ms'],saving_percent=data['seed205']['measured_step_time_saving_percent']),
        stock_matrix_state_bytes=data['seed205']['results'][0]['matrix_state_bytes'],
        controls=short,qr_profile=profile,
        sources={k:dict(path=str(ROOT/v),sha256=hashlib.sha256((ROOT/v).read_bytes()).hexdigest()) for k,v in INPUTS.items()},
        interpretation='Two initialization seeds, fixed OWT order; observed approximation with a loss gap, no statistical equivalence or final-horizon sparse-QR superiority claim.',
        numerical_error_role='diagnostic',new_gpu_experiments_remaining=0)
    write(OUT/'summary.json',summary)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','axes.grid':True,'grid.alpha':.18})
    colors={204:'#2563eb',205:'#c2410c'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for r in rows:
        curve=r['validation_curve'];x=[v['call'] for v in curve if v['call']>0]
        for key,style,label in [('stock_ce','-',f"Stock {r['seed']}"),('bf16_ce','--',f"BF16 {r['seed']}")]:
            axes[0].plot(x,[v[key] for v in curve if v['call']>0],style,color=colors[r['seed']],label=label,lw=1.6)
        axes[1].plot([v['call'] for v in curve],[v['delta'] for v in curve],marker='.',color=colors[r['seed']],label=str(r['seed']))
    axes[0].set(title='Full-horizon validation CE',xlabel='Training calls',ylabel='Cross-entropy')
    axes[1].set(title='BF16 minus stock CE',xlabel='Training calls',ylabel='CE difference')
    axes[1].axhline(0,color='#555',lw=.7);axes[0].legend(fontsize=8);axes[1].legend(title='Model seed',fontsize=8)
    for fmt in ('png','svg'):fig.savefig(OUT/f'full_horizon.{fmt}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    warmvals={v['call']:v['ce'] for v in cached['validation']}
    axes[0].axhline(0,color='#555',label='Cached warm10/QR50',lw=1.2)
    for r,color,label in zip(controls['results'],('#2563eb','#c2410c'),('QR20','QR50')):
        axes[0].plot([v['call'] for v in r['validation']],[v['ce']-warmvals[v['call']] for v in r['validation']],marker='.',color=color,label=label)
    axes[0].set(title='2k control CE minus cached warm',xlabel='Training calls',ylabel='CE difference')
    axes[0].legend(fontsize=8)
    batched=[r for r in profile if r['batch']==12];x=list(range(4))
    axes[1].bar([v-.18 for v in x],[r['original_cuda_ms'] for r in batched],width=.36,label='Original',color='#64748b')
    axes[1].bar([v+.18 for v in x],[r['reuse_cuda_ms'] for r in batched],width=.36,label='CQ reuse',color='#2563eb')
    axes[1].set(xticks=x,xticklabels=[r['dimension'] for r in batched],title='QR primitive, batch 12',xlabel='Factor dimension',ylabel='Median CUDA ms')
    axes[1].legend(fontsize=8)
    for fmt in ('png','svg'):fig.savefig(OUT/f'controls_and_profile.{fmt}',dpi=180)
    plt.close(fig)

    endpoints='\n'.join(f"| {r['seed']} | {r['stock_final_ce']:.9f} | {r['bf16_final_ce']:.9f} | +{r['ce_delta']:.9f} | +{r['perplexity_increase_percent']:.4f}% |" for r in rows)
    shorttable='\n'.join(f"| {r['case'].replace('cached_bf16_warm10_qr50','Warm10/QR50 (cached)').replace('bf16_','')} | {r['final_ce']:.9f} | {r['mean_step_ms']:.3f} | {r['mean_optimizer_ms']:.3f} |" for r in short)
    qrtable='\n'.join(f"| {r['batch']} | {r['dimension']} | {r['original_cuda_ms']:.3f} | {r['reuse_cuda_ms']:.3f} | {r['qr_time_saving_percent']:.2f}% | {r['numerical']['order_disagreements']}/{r['numerical']['total_columns']} |" for r in profile)
    report=f'''# TurboSOAP bounded demonstration: completed

The fixed compiled full-BF16 method approaches stock SOAP learning with a small
consistent final loss gap on two model seeds, while halving persistent matrix
state and reducing complete-step time against the original stock implementation.
All approved experiments and frozen measurement audits completed. The strongest
claim is a measured quality/time/memory tradeoff; these data do not establish
equivalence, general superiority, or that warm tracking alone explains speed.

## Recipe and full-horizon learning

One 109,396,992-parameter Transformer on cached OpenWebText, model seeds 204/205,
fixed data order, 10,000 optimizer calls (first matrix call initializes state),
stock matrix betas(.95,.95)/covariance.95. Same full10k LR horizon, matrix/auxiliary
peakLR.0015/.0006, decay.05, clip1, physical8 × accumulation4, 1,024-token rows,
full1,024-row validation. Full two-sided coverage:48 matrices/96 factors. BF16
M/V/C/Q and matrix arithmetic, FP32 master weights and accumulation/solver
islands; fullgraph shape batches, warm refresh10/QR repair50. See frozen protocols.

| Seed | Stock QR10 CE | Full BF16 CE | Paired CE delta | Perplexity increase |
| --- | ---: | ---: | ---: | ---: |
{endpoints}

Mean paired CE delta **+{summary['mean_paired_ce_delta']:.9f}**. The peak observed
gap is at500 calls: +{rows[0]['peak_observed_ce_gap']['delta']:.6f} / +{rows[1]['peak_observed_ce_gap']['delta']:.6f}
for204/205. Early deviation matters even though the gap shrinks late. Two model
initializations with a fixed data order are not a statistical equivalence test.
The research validation panel was used in earlier tuning; it is not untouched test data.

![Full-horizon curves and CE gaps](full_horizon.png)

## Efficiency and precision

Persistent matrix state: stock2,491,416,576bytes (2.320GiB), BF161,245,708,320bytes
(1.160GiB), approximately50% less. BF16 peak allocated GPU memory8.223GiB
versus stock9.397GiB (seed205); this is allocated tensor memory, not device reservation.

Seed204 uses a newly matched100-call window101..200: stock380.740ms versus
BF16314.638ms, **17.36% less complete-step time**. Its 201-call stock replay
exactly reproduces the cached full control at0/50/100/200. Seed205 has a newly
paired9,900-call window101..10000: stock380.548ms versus BF16309.533ms,
**18.66% less time**. Keep those window scopes separate; do not average them.
Timed steps include packing, casts, parameter writes and auxiliary AdamW.
Startup, evaluation and diagnostics are excluded from step means but included
in campaign elapsed and execution caps. Compilation/storage/batching/cadence
change together relative to original stock; no single-kernel causal attribution.

All sampled state is finite; candidate captures remain12 graphs, no breaks.
Seed204/205 max actual-M errors .9924%/1.0356%, normalized Gram~.002366/.002367.
Seed205 crosses the historical1% M screen; both cross the original absolute
Gram screen. Raw flags remain unchanged. Under current user steering these
finite measurements are diagnostics, not loss acceptance thresholds. Diagonal-V
transport and M preservation are approximate; small Gram/M errors do not
certify covariance alignment.

## Short cadence controls

Two fresh full-BF16 pure-QR arms, seed204,2,000 calls each, unchanged10k LR
horizon; compare with cached warm prefix. Same initialization/gradients/full
validation, precision/coverage and CQ-reuse QR. Only cadence changes.

| Policy | CE at2000 | Step ms | Optimizer ms |
| --- | ---: | ---: | ---: |
{shorttable}

Complete shared100-call cycles101..2000 (1,900 calls). QR20/QR50 are sequential
new timings on one device; warm timing is cached across launches. Warm is
.003557 CE behind QR20 and .004740 ahead of QR50 at the endpoint, better than
QR50 at all seven noninitial observations. QR20 is better at six of seven.
The modest warm-versus-hold quality gain has extra work; its tiny observed
whole-step difference from QR50 is not a precise newly paired timing result.
These are single-seed early-prefix observations, not final-horizon superiority.

QR20 totals cold1/ordinary1900/QR99, QR50 cold1/ordinary1960/QR39, no warm events.
Both maintain eight training graphs without breaks and final finite model,
matrix and auxiliary state. Max actual-M errors .5499%/.5105% are diagnostics.

![Control loss gaps and isolated QR profile](controls_and_profile.png)

## Same-state QR attribution

Fixed synthetic C/Q at existing dimensions, BF16-rounded then promoted to FP32;
both original and CQ reuse use identical FP32 high matmul precision and FP32 QR
outputs/solver. No compilation or training. Two warmups/ten counterbalanced
rounds, CUDA and synchronized wall samples. Original uses three dense products,
reuse one. Batch12 matches real shape groups; pair comparisons keep batch fixed.

| Batch | Dimension | Original CUDA ms | Reuse CUDA ms | Time reduction | Reordered columns |
| ---: | ---: | ---: | ---: | ---: | ---: |
{qrtable}

At batch12, reuse saves0.95–4.63% of this primitive, as QR solver time dominates.
Products/sort alone improve more; all component samples are in the scalar
evidence. Component medians are not an additive decomposition. Batching,
precision, compilation and full optimizer timing are distinct questions.

Near-tie score bands reach2.68e-5 of max score; normalized FP32-output Gram is
at most1.19e-6. Raw basis and coordinate-M differences are large under sign/
ordering changes (coordinate-M relative difference up to.877); sign-aligned
basis difference reaches.0932 and V-coordinate difference.0404. Those raw
coordinate comparisons are not physical update errors. Max physical-M relative
difference is.0009064 and next ordinary-update relative difference.0011161
(~.112%), cosine at least.9999994. This is a synthetic one-factor probe with
a fixed32D right identity using actual BF16 transport/ordinary kernels. It
does not establish bitwise equivalence, exact state transport or trained
trajectory agreement. Input hashes, finite state and nonnegative V checks pass.
Current training speed claims continue to name the original stock control;
there is no newly trained efficient-FP32-QR10 comparator.

## Scope, provenance and stopping point

No production optimizer change. Production source pins match the ledger; frozen
source/input/support/event/replay/evaluation/capture audits passed. Controls had
17 CPU checks; profile had4 against independent float64 and actual reference/
production primitives. Controls/profile supervised wall time (including any
queue/startup) totaled1,500seconds, an upper bound on added claimed GPU execution;
their hard execution caps reserve1,800+600seconds, below2,700 authorized seconds.
No retries, additional full runs, sweeps, new datasets or tensor checkpoints.

Broad warm reuse, partial-basis methods and BF16 storage overlap prior work.
See the completed [novelty review](../review_20261026/REVIEW_SYNTHESIS.md).
A paper can center the specific dense kernel/precision policy and this modest
approximation tradeoff. General optimizer superiority and novelty of warm
tracking or BF16 storage are not established. The bounded demonstration is
complete; drafting can use these results without adding a benchmark suite.

Reproduce this report on CPU with benchmarks/summarize_bf16_completion.py.
[Summary and input SHA256 provenance](summary.json), exportable PNG/SVG figures,
[replication review](../bf16_replicate_20261027_seed204/REVIEW.md),
[controls](../bf16_refresh_controls_20261027/REVIEW.md),
[QR profile](../bf16_qr_attribution_20261027/result.json). Historical evidence
and frozen protocols remain unchanged. The completion monitor is disabled
after saving the final ledger and backup; no GPU work remains in this plan.
'''
    (OUT/'REPORT.md').write_text(report)
    table=''.join(f"<tr><td>{r['seed']}</td><td>{r['stock_final_ce']:.6f}</td><td>{r['bf16_final_ce']:.6f}</td><td>+{r['ce_delta']:.6f}</td></tr>" for r in rows)
    html=f'''<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font:14px var(--font-sans,system-ui);color:var(--foreground,#172033)}}
.lead{{font-size:18px;font-weight:650;line-height:1.4;margin:0 0 12px}}p{{line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:8px 6px;border-bottom:1px solid var(--border,#dbe0e8);text-align:right}}th:first-child,td:first-child{{text-align:left}}
.plot{{background:white;border-radius:12px;padding:16px;margin:16px 0}}img{{width:100%;display:block}}.muted{{color:var(--muted-foreground,#606a7a);font-size:12px}}
</style></head><body><p class="lead">Two seeds: a small loss gap, half the matrix state</p>
<table><tr><th>Model seed</th><th>Stock CE</th><th>Full BF16 CE</th><th>Delta</th></tr>{table}</table>
<p>Mean paired gap <b>+0.010660 CE</b> · matrix state <b>2.320 → 1.160 GiB</b>.<br>
Measured step-time reductions: <b>17.36%</b> (seed204,100-call prefix) and <b>18.66%</b> (seed205,9,900 calls).</p>
<div class="plot"><img src="{OUT/'full_horizon.png'}" alt="Two seed loss curves and BF16 minus stock gaps"></div>
<p>At2k calls: QR20 <b>3.99038</b>, warm <b>3.99394</b>, QR50 <b>3.99868</b> CE. Short controls support a modest tradeoff, not final-horizon superiority.</p>
<div class="plot"><img src="{OUT/'controls_and_profile.png'}" alt="Short control loss gaps and isolated QR primitive timings"></div>
<p class="muted">All bounded experiments and audits completed. QR product reuse saves0.95–4.63% at batch12; the solver dominates. Same-state synthetic following-update difference ≤0.112%. Finite numerical screens remain diagnostics. Two fixed-order model seeds do not establish statistical equivalence. Warm control timing is cached across launches; training speed is relative to original stock SOAP.</p>
</body></html>'''
    (OUT/'review.html').write_text(html)
    assert len(rows)==2 and all(math.isfinite(r['ce_delta']) for r in rows)
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),mean_paired_ce_delta=summary['mean_paired_ce_delta'],files=len(list(OUT.iterdir())))))


if __name__=='__main__':main()
