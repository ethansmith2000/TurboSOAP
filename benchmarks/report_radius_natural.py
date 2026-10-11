"""Durable report and standalone figure from completed, qualified evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

def main():
    b=Path(sys.argv[1]); s=json.loads((b/'summary.json').read_text())
    rows=[x for x in s['rows'] if not x['repeat']]
    lines=['# Radius .6: actual natural-cycle qualification','',
        f"Completed 2026-10-06. Qualification: **{'PASS' if s['qualified'] else 'FAIL'}**. "
        f"18 cases, {s['updates']:,} updates, {s['wall_seconds']:.1f}s managed execution.",'',
        'Radius .6/cubic2 reaches maximum sampled history M error .2689% and '
        'per-refresh error .2661%, both below1%. Cubic3 lowers these to .1374% '
        'and .0824%. At width1536/covariance .999, complete optimizer costs are '
        'QR18.216 ms, .5/cubic2 14.176 ms, .6/cubic2 14.190 ms and .6/cubic3 '
        '14.980 ms. The radius-only change has essentially unchanged cost; the '
        'extra correction adds about5.6% whole optimizer cost in this stratum. '
        'Candidate learning benefit remains unmeasured.','',
        'Actual gradients, covariance accumulation, Adam state and model weights evolve for '
        '420 updates per case. QR resets at ages200/400; both factors through6144. '
        'Strict M uses highest FP32 matmul precision, basis work high, fixed V carry/reindex. '
        'The seeded BF16-autocast MockBlock workload is inherited unchanged.','',
        '| Width | Cov beta | Policy | Pass | Max Gram | Max history M (%) | Max refresh M (%) | Optimizer ms | Step wall ms |',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x['width']} | {x['beta']} | {x['policy']} | {x['qualified']} | {x['max_gram']:.6g} | {100*x['max_global_m']:.4f} | {100*x['max_transport_m']:.4f} | {x['optimizer_ms']:.3f} | {x['step_wall_ms']:.3f} |")
    lines+=['','![Momentum history and complete optimizer cost](radius_natural.png)','',
        '## Verification and scope','',
        f"Eight historical matched QR/.5 controls are exact: "
        f"{all(x['exact'] for x in s['historical_anchors'])}. "
        f"Both .6 repeats reproduce final model and full optimizer state hashes, every loss "
        f"and all numerical probes: {all(x['exact_model_optimizer_losses_probes'] for x in s['controls'])}."]
    for x in s['controls']:
        lines.append(f"Repeat optimizer range/mean for {x['policy']}: {100*x['optimizer_relative_range']:.3f}%.")
    lines+=['',f"{s['refresh_matrix_checks']:,} per-refresh matrix checks and "
        f"{s['history_matrix_checks']:,} actual-history matrix checks, including repeats. "
        'Whole-history M is compared with an independently accumulated physical-gradient EMA '
        'at init and every20 updates. This is sampled actual history; it does not bound every '
        'intermediate update or establish long-run convergence. Gates remain Gram<.05 and '
        'both M errors<1%. Earlier failures remain unchanged.','',
        'Complete optimizer event times include covariance accumulation, Adam updates, '
        'both bases, strict M transport, V handling, casts and observer dispatch. '
        'Model/backward and host dispatch also enter step wall time. Cold initialization '
        'is separate in summary.json; diagnostic work, physical EMA oracle, hashes and '
        'logging are excluded. Offline checks can still perturb subsequent execution. '
        'These are instrumented synthetic costs, not CIFAR throughput or time to quality. '
        'Own-trajectory covariance residuals do not establish causal candidate attribution.','',
        '## Decision','',
        ('Both .6 policies pass the actual-cycle numerical screen. Advance .6/cubic2 as the '
         'cheap candidate and keep .6/cubic3 as a numerical control in bounded paired CIFAR '
         'learning. Include matched QR/.5 controls, fixed-update anchors and newly declared '
         'wall endpoints on CPUs116/117 with interleaved repeats and unchanged3% wall/2% '
         'optimizer gates. Seed139 first; seeds271/811 and longer horizons follow a useful '
         'learning effect. Alternating3/2 stays a documented fallback. No default promotion.'
         if s['qualified'] else 'Qualification failed; preserve all failures and diagnose '
         'the failing criterion before candidate learning. No default promotion.'),'',
        '## Evidence and resources','',
        '200 CPU tests passed before launch; ten focused policy/adapter checks also passed. '
        'Execution source/config/protocol hashes are retained in manifest.json; reporting '
        'source is separately frozen in analysis_source. No weights, checkpoints or tensor '
        'snapshots saved. One GPU3 claim, private supervisor, no queued wait. Existing dirty '
        'source and all historical evidence preserved. See artifact_inventory.json and '
        'PROVENANCE.md for final integrity/resource state.']
    (b/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={'qr20':'#5c6570','fro_cubic2_cap05':'#3671ad','fro_cubic2_cap06':'#d07825','fro_cubic3_cap06':'#389263'}
    groups=[(768,.99),(768,.999),(1536,.99),(1536,.999)]
    import numpy as np
    xpos=np.arange(4)
    for i,(p,color) in enumerate(colors.items()):
        xs=[next(x for x in rows if (x['width'],x['beta'],x['policy'])==(w,beta,p)) for w,beta in groups]
        for ax,key,scale in [(axes[0,0],'max_global_m',100),(axes[0,1],'max_transport_m',100),(axes[1,0],'optimizer_ms',1),(axes[1,1],'step_wall_ms',1)]:
            ax.bar(xpos+(i-1.5)*.19,[x[key]*scale for x in xs],width=.18,label=p,color=color)
    for ax in axes.flat:
        ax.set_xticks(xpos,[f'{w} / {beta}' for w,beta in groups]);ax.set_xlabel('Width / covariance beta')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax,title in [(axes[0,0],'Maximum sampled actual-history M error'),(axes[0,1],'Maximum per-refresh M error')]:
        ax.set_title(title);ax.set_ylabel('Relative error (%)');ax.axhline(1,color='#a22',ls='--',lw=1,label='Declared 1% limit')
    axes[1,0].set_title('Complete optimizer event cost');axes[1,0].set_ylabel('ms / update')
    axes[1,1].set_title('Instrumented model + optimizer step wall');axes[1,1].set_ylabel('ms / update')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=3,fontsize=9)
    fig.savefig(b/'radius_natural.png',dpi=160);plt.close(fig)
    dest=b/'analysis_source';dest.mkdir(exist_ok=True)
    src=Path(__file__);shutil.copy2(src,dest/src.name)
    (b/'analysis_manifest.json').write_text(json.dumps({src.name:hashlib.sha256(src.read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__': main()
