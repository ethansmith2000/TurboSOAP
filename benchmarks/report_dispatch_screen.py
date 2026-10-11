"""Standalone report for the prospectively bounded dispatch diagnostic."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text());d=json.loads((b/'screen.json').read_text())
    assert s['numerical_exact'] and s['observer_complete'] and s['profile_replays_exact']
    effect=100*s['same_over_near_model_effect'];material=s['material_placement_effect']
    lines=['# Autograd on one core: bounded dispatch diagnostic','',
        f"Completed six QR arms / {s['primary_steps']:,} primary updates in {s['all_in_seconds']:.1f}s. "
        f"Numerical controls exact: **{s['numerical_exact']}**. Local repeat qualification: "
        f"**{s['local_repeat_qualified']}**. Material directional placement effect: **{material}**.",'',
        f"Main116 in both conditions; autograd116 (same core) versus117 (adjacent physical core, "
        f"same L3). Same-core model/backward event mean changes {effect:+.2f}%. "
        'Interpret this through the predeclared gates and complete observed spreads below. '
        'No optimizer/affinity default is promoted; this1000-update QR diagnostic does '
        'not qualify full3500-update QR/Fro timing or historical wall-quality endpoints.','',
        '| Condition | Mean training wall s | Wall range/mean | Mean model event ms | Mean optimizer event ms | Optimizer range/mean | Repeat gate |',
        '|---|---:|---:|---:|---:|---:|---|']
    for x in s['conditions']:
        lines.append(f"| {x['kind']} | {x['training_mean_seconds']:.4f} | {100*x['wall_range_over_mean']:.3f}% | {x['model_mean_ms']:.4f} | {x['optimizer_mean_ms']:.4f} | {100*x['optimizer_range_over_mean']:.3f}% | {x['repeat_gate']} |")
    lines+=['',
        'Use all three runs per condition, unchanged3% wall/2% optimizer limits, '
        'no trimming. First50 excluded only from event means; full wall includes them. '
        'A material model placement effect additionally requires >=5% mean difference, '
        'nonoverlapping model ranges and all three adjacent paired directions agreeing. '
        f"Adjacent same/near model effects: {[round(100*x,3) for x in s['adjacent_pair_effects']]}%.",'',
        '![Primary timing and thread accounting](dispatch_screen.png)','',
        '## What the new observations measure','',
        '| Arm | Main runtime s | Autograd runtime s | Other thread runtime s | Main runqueue wait s | Autograd runqueue wait s | Observer overhead s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in s['thread_rows']:
        lines.append(f"| {x['label']} | {x['main_runtime_seconds']:.4f} | {x['autograd_runtime_seconds']:.4f} | {x['other_runtime_seconds']:.4f} | {x['main_runqueue_wait_seconds']:.4f} | {x['autograd_runqueue_wait_seconds']:.4f} | {x['observation_seconds']:.4f} |")
    lines+=['',
        'Thread runtime and runqueue wait come from own-process /proc schedstat '
        'counters. Interval snapshots include offline guard/probe/hash work, so they '
        'are not exact model-phase accounting and must not be added to primary wall. '
        'User/system tick counters have kernel clock resolution; main user/system '
        'CPU by phase instead uses RUSAGE_THREAD. Four additional rusage calls per '
        'update are charged in primary timing identically in both conditions. '
        'Thread/GPU observations are outside primary timing; their measured overhead '
        'and potential perturbation remain explicit. No observer feeds back.','',
        '| Arm | Optimizer main user CPU s | Optimizer main system CPU s | Model main user CPU s | Model main system CPU s |',
        '|---|---:|---:|---:|---:|']
    for x in s['thread_rows']:
        o=x['phase_cpu']['optimizer'];m=x['phase_cpu']['model_backward']
        lines.append(f"| {x['label']} | {o['main_user_cpu_ms']:.4f} | {o['main_system_cpu_ms']:.4f} | {m['main_user_cpu_ms']:.4f} | {m['main_system_cpu_ms']:.4f} |")
    lines+=['','## Actual GPU work in separate profiler windows','',
        f"All six plain/profile replay controls exact, including cross-placement: {s['profile_replays_exact']}. "
        f"GPU operation-name/category/count signatures exact across four traces: {s['profile_gpu_operation_signatures_exact']}.",'',
        '| Placement | Replay | Model GPU sum ms/update | Optimizer GPU sum ms/update | GPU work events | Unattributed |',
        '|---|---|---:|---:|---:|---:|']
    for x in s['traces']:
        lines.append(f"| {x['kind']} | {x['mode']} | {x['phases']['timing/model_backward']['gpu_work']['sum_us']/4000:.4f} | {x['phases']['timing/optimizer']['gpu_work']['sum_us']/4000:.4f} | {x['gpu_work']['count']} | {x['unattributed_gpu_events']} |")
    lines+=['',
        'Only kernel/gpu_memcpy/gpu_memset categories count as GPU work, attributed '
        'by launch correlation to enclosing CPU phases. GPU annotation spans are '
        'excluded. Profiler windows contain four updates and substantial observer '
        'perturbation; actual kernel sums are diagnostics, not primary throughput. '
        'CUDA-event spans include dispatch gaps. Host/device intervals overlap. '
        'Sparse readings every250 updates do not establish causal clock effects.','',
        '## Numerical work and evidence','',
        f"All12 historical875/1000 observations, all six historical875 full optimizer "
        f"hashes, final optimizer hashes and tracking probes match. {s['refresh_guards']} "
        f"finite/geometry guards and {s['matrix_probes']} targeted M checks pass. "
        f"Full strict-reference QR failures retained separately: {s['full_strict_reference_failures']}.",'',
        'Seed139, covariance .99, Adam(.9,.999), both factors, strict highest M, high '
        'basis, V reindex, QR20, original LR .0005/200-epoch schedule. Every arm '
        'starts from identical initial model/RNG/input and a fresh optimizer. '
        'Six primary plus six replay placement records; original two masks restored. '
        'Other processes untouched. Existing isolated trainer byte-identical; '
        'execution adapter and observer are additive.','',
        '213 CPU tests and10 focused tests passed before launch. Protocol, executed '
        'source/config, audit hashes, metrics/logs/thread records and compressed '
        'traces retained. No saved weights/checkpoints/tensor files. Two diagnostic '
        'states remained in RAM only. Derived reporting source is frozen separately '
        'in analysis_source. [Decision](DECISION.md), [protocol](PROTOCOL.md), '
        '[provenance](PROVENANCE.md).']
    (b/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');runs=d['runs'];xx=np.arange(6)
    colors=['#586673' if x['kind']=='near' else '#2f8b9c' for x in runs];labels=[x['label'] for x in runs]
    for ax,key,title,ylabel in [(axes[0,0],'model_backward_cuda_ms','Model/backward event spans','ms/update'),(axes[0,1],'training_seconds','Complete1000-update training wall','seconds')]:
        ax.scatter(xx,[x[key] for x in runs],c=colors,s=55);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(xx,labels,rotation=25)
        for kind,col in [('near','#586673'),('same','#2f8b9c')]:ax.axhline(np.mean([x[key] for x in runs if x['kind']==kind]),color=col,ls='--',alpha=.6)
        ax.grid(axis='y',alpha=.2)
    ax=axes[1,0];bottom=np.zeros(6)
    for key,color,label in [('main_runtime_seconds','#586673','main'),('autograd_runtime_seconds','#2f8b9c','autograd'),('other_runtime_seconds','#ca732b','other')]:
        values=[x[key] for x in s['thread_rows']];ax.bar(xx,values,bottom=bottom,color=color,label=label);bottom+=values
    ax.set_title('Thread runtime, including offline diagnostics');ax.set_ylabel('CPU seconds');ax.legend(fontsize=9);ax.set_xticks(xx,labels,rotation=25)
    ax=axes[1,1];ax.bar(xx,[x['autograd_runqueue_wait_seconds'] for x in s['thread_rows']],color=colors)
    ax.set_title('Autograd runqueue wait, including offline intervals');ax.set_ylabel('seconds');ax.set_xticks(xx,labels,rotation=25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(b/'dispatch_screen.png',dpi=160);plt.close(fig)
    a=b/'analysis_source';a.mkdir(exist_ok=True);shutil.copy2(Path(__file__),a/Path(__file__).name)
    (b/'analysis_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()},indent=2)+'\n')

if __name__=='__main__':main()
