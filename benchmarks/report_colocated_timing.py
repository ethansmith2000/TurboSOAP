"""Outcome report for full-horizon same-core QR/Fro qualification."""
import json
from pathlib import Path
import sys
import shutil
import hashlib

def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text());assert s['status']=='completed'
    lines=['# Full-horizon same-core timing qualification','',
        f"Six3500-update QR/Fro arms completed {s['primary_updates']:,} updates in {s['all_in_seconds']:.1f}s. "
        f"Full qualification: **{s['qualified']}**; numerical exactness: **{s['numerical_exact']}**.",'',
        '| Method | Mean training wall s | Wall range/mean | Mean optimizer ms | Optimizer range/mean | Repeat gate |',
        '|---|---:|---:|---:|---:|---|']
    for x in s['conditions']:lines.append(f"| {x['kind']} | {x['mean_training_seconds']:.4f} | {100*x['wall_range_over_mean']:.3f}% | {x['mean_optimizer_ms']:.4f} | {100*x['optimizer_range_over_mean']:.3f}% | {x['repeat_gate']} |")
    lines+=['',
        'All three repeats per method, unchanged3% full wall/2% steady optimizer '
        'limits, no trimming. Main and autograd both116. First50 excluded only '
        'from event means, not full wall. New diagnostic records are outside primary '
        'timing and can perturb following execution. This is instrumented scope, '
        'not uninstrumented throughput, cross-study hardware speed or a wall-quality '
        'learning winner. Prior failed endpoints remain unchanged.','',
        '![Full-horizon repeats](colocated_timing.png)','',
        '| Arm | CE at3500 | Main runtime s | Autograd runtime s | Autograd queue wait s | Observer overhead s |',
        '|---|---:|---:|---:|---:|---:|']
    for x in s['rows']:lines.append(f"| {x['label']} | {x['ce']:.6f} | {x['main_runtime_seconds']:.3f} | {x['autograd_runtime_seconds']:.3f} | {x['autograd_runqueue_wait_seconds']:.3f} | {x['observer_seconds']:.3f} |")
    lines+=['',
        f"All18 historical fixed observations/full optimizer hashes exact: {s['numerical_exact']}. "
        f"{s['refresh_guards']} guards and {s['matrix_probes']} targeted M probes. "
        f"Separate full strict-reference failures preserved: {s['full_strict_reference_failures']}. "
        f"Observer complete: {s['observer_complete']}; compiler stable: {s['compiler_stable']}; "
        f"plain/profile replays exact: {s['profile_replays_exact']}; corrected profiles complete: {s['corrected_profiles_complete']}.",'',
        'Seed139, covariance .99, Adam(.9,.999), both factors, strict highest M/high '
        'basis, V carry/reindex, QR20 versus Fro .5/cubic2 warm20/QR200, LR .0005 '
        'and unchanged200-epoch schedule/data/evaluation. Same initial model/RNG/input '
        'and fresh optimizer in every arm. Twelve primary/replay placement records; '
        'two original masks restored. All old numerical work is retained.','',
        'Affinity vectors are deduplicated losslessly; full per-thread counters '
        'and definitions retained. Snapshot intervals include offline work and '
        'cannot be added to training wall. Runtime and runqueue wait are distinct. '
        'Main user/system phase CPU uses RUSAGE_THREAD. Fourteen own-GPU readings '
        'per arm are sparse diagnostics, not a clock controller. No other process '
        'or permanent affinity/default changes.','',
        'Four-update plain/profile1/profile2 triplets use two RAM-only final states. '
        'Corrected actual kernel/memcpy/memset accounting excludes GPU annotations '
        'and attributes work by launch correlation. Windows are perturbed and '
        'excluded from repeat gates. CUDA-event spans include dispatch gaps; host '
        'and device intervals overlap. No synthetic timing is attached to losses.','',
        '214 CPU tests passed before launch. No weights/checkpoints/tensor files. '
        'Executed source/config/protocol/metrics/logs/records/traces retained; '
        'reporting source frozen separately. See [decision](DECISION.md), '
        '[protocol](PROTOCOL.md), [provenance](PROVENANCE.md) and inventories.']
    (b/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title,ylabel in [(axes[0],'training_seconds','Complete3500-update training wall','seconds'),(axes[1],'optimizer_ms','Complete optimizer events','steady ms/update')]:
        for i,x in enumerate(s['rows']):ax.scatter(i,x[key],color='#586673' if x['kind']=='qr' else '#2f8b9c',s=55)
        ax.set_xticks(range(6),[x['label'] for x in s['rows']]);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
    fig.savefig(b/'colocated_timing.png',dpi=160);plt.close(fig)
    a=b/'analysis_source';a.mkdir(exist_ok=True)
    if Path(__file__).resolve()!=(a/Path(__file__).name).resolve():shutil.copy2(__file__,a/Path(__file__).name)
    (b/'analysis_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()},indent=2)+'\n')

if __name__=='__main__':main()
