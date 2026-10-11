"""Freeze actual width1536 learning past LR warmup with six timing controls."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_wide_pilot_20261007')
    previous=Path('/workspace/optimizer_replay_results/cifar_soap_wide2048_pilot_20261007')
    assert json.loads((prior/'summary.json').read_text())['feasibility_qualified']
    assert json.loads((previous/'summary.json').read_text())['owned_teardown_released']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
        matrix_lr=.0005,covariance_beta=.99,branch_steps=3500)
    c=json.loads((b/'config.json').read_text())
    c.update(width=1536,depth=1,heads=24,smoke_steps=25,max_wall_seconds=2400)
    arms=[]
    for i,kind in enumerate(['qr','fro','fro','qr','qr','fro']):
        a=dict(label=f'{kind}_{i}',kind=kind,policy='qr20' if kind=='qr' else 'warm20',
            matrix_lr=.0005,matrix_beta2=.999,covariance_beta=.99,transport_precision='highest',
            basis_ns_iterations=2,basis_rotation_cap=.1,tracking_probe=True,capture_replay=False,
            timing_affinity=dict(main=116,autograd=116))
        if kind=='fro':a['movement_policy']='fro_cubic2_cap05'
        arms.append(a)
    c['memory_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=3500,
        evaluation_updates=[100,200,400,420,875,1750,3500],wall_seconds=None,timing_chunk=25,
        maximum_gram_error=.05,guard_every_refresh=True,tracking_probe_kind='wide_selected',
        tracking_ages=[20,180,200,220,380,400,860,1740,3480],
        control_pairs=[['qr_0','qr_3'],['qr_0','qr_4'],['fro_1','fro_2'],['fro_1','fro_5']],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02,
        profile_steps=0,execution_warmup_updates=0,maximum_peak_allocated_bytes=24*2**30)
    names=['cifar_wide_learning.py','cifar_wide_learning_runner.py','owned_matrix_hooks.py',
        'wide_probe.py','wide_learning_checks.py','prepare_wide_learning.py','summarize_wide_learning.py',
        'cifar_affinity_screen.py','compact_dispatch_screen.py','dispatch_diagnostics.py',
        'timing_diagnostics.py','analyze_timing_traces.py','cifar_tracking_probe.py',
        'movement_retraction.py','warm_retraction.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Actual width1536 fixed3500 learning and local timing qualification

Prospective 2026-10-07. Six interleaved arms QR/Fro/Fro/QR/QR/Fro, three repeats
per method, seed139. Actual CIFAR100 ViT width1536/depth1/heads24, batch256 and
evaluation512. Original frozen split, augmentation, eager-bf16 evaluation and
LR .0005 on200epoch schedule with5epoch LR warmup.3500 updates is20 epochs;
early learning past LR warmup, not convergence at200 epochs.28,660,324 parameters,
four SOAP matrices/eight factors through6144. Both factors, covariance .99 every
gradient step, Adam(.9,.999), V carry/reindex, high basis/highest strict M.
QR20 versus Frocap.5/cubic2 warm20/resetQR200. No EMA/cap/damping/controller,
LR/covariance/seed/width sweep or optimizer default change.

One fixed25-update global QR10 compilation/parity smoke, discard all trained
state. NO per-arm disposable400-update warmup: fixed work and full initialization
cost are the target. Restore original weights and complete RNG before every
fresh empty optimizer. All3500 measured updates, including first50 and cold
state initialization/eigh, count full trainingwall. First50 omitted only from
steady event means. No uncounted trained prefix or adaptive burn-in. Restore
owned instance attributes/group keys after final hashes; delete optimizer/probe/
lease and require immediate weakref release without explicit GC/cache clearing.
Memory before updates, peak, after cleanup and cleanupduration retained. Every
setup/smoke/offline check/cleanup contributes to total all-in time.

Evaluate100/200/400/420 (preserve original1536 anchors),875/1750/3500. The existing
420 pilot is a prerequisite, not a GPU job rerun for context. This new horizon
retains early anchors and adds later learning. Timing flushes at every25 updates
AND each observation:141 chunks/arm, with20/5-update chunks around420. No missing
or double-counted work. Loss/RNG/input/augmentation/model/fulloptimizer hashes
at all seven observations; evaluation restores RNG/weights. Primary wall and
augmentation/model/completeoptimizer CUDA events include online covariance,
refresh, retraction, M transport, V handling, auxiliaryAdam, casts and dispatch.
Exclude offline guards/probes/evaluation/hashes/observerflushes/cleanup from
primary wall, include all in total all-in. CUDA events include host dispatch
gaps and are not exact kernel duration or uninstrumented deployment throughput.
No equal-wall endpoint or synthetic-timing/loss splice in this study.

Guards initialization/every scheduled refresh:175/arm,1050 total. Optimizer age
is trainingstep-1. Probe ages20/180/200/220/380/400/860/1740/3480:36 matrix samples/
72 factor rows perarm,216/432 total. Require selected-basis updateerror<1e-6,
physicalMdrift<1%, Gram<.05, finite parameters/moments. Abort failure and preserve
partial evidence; no fallback/retry/relaxed gates. Same selected-basis/physicalM
scope as both wide pilots. Deliberately omit second fullQR reference shadows;
no fullstrict-pass claim. Earlier fullstrict QR/reset discrepancies preserved.
Raw offline generator diagnostics use unchanged high math, twelve deterministic
float64 poweriterations with local seed9100+dimension, Frobound/hypothetical
.5 clipping, Rayleighdiagonal gaps, before/executed residuals/Gram/movement and
gradient/covariance magnitudes. Operator norm is a lower proxy, bound/proxy an
upper slack proxy, not exact norm/certification. No additionalQR/eigh/SVD,
feedback or globalRNG use; offline probe time retained separately.

Require all24 original1536 observations and all144 early matrix probes to match
raw prior pilot numerically, including model/fulloptimizer/RNG/input/augmentation
hashes. All three repeats per method must match all seven observations, complete
optimizer hash, chunk losses/gradientnorms, guards and probes exactly. Independent
seeds are not collected. Compiler counters stable; masks/observer data complete.
Use ALL three fixed3500 repeats per method: full trainingwall range/mean<=3%,
steady completeoptimizer-event range/mean<=2%, unchanged. BOTH methods must pass.
No trimming or retroactive qualification of previous failed endpoints. If timing
fails, preserve failure and retain only qualified fixed-work learning/numerical
scope. No extra repeats, warmup extension, tolerance change or automatic wall
campaign. If it passes, only this local instrumented scope is timing-qualified;
any future wall-budget comparison needs its own prospective declaration.

Main/autograd CPU116/116, own process only, original masks restored. Compact
host/thread/process CPU/GC/queue/affinity observer; own-GPU sample every10 flushes
across arms (84 samples total), cadence perturbed at420 by partialchunks. Snapshot
intervals include offline work and omit unmatched transient threads. Complete
mask catalog/counters retained; no online feedback or other-job/clocks/GC changes.
No profiler replays, tensor snapshots or trace files.

One exclusive nonqueued free RTX5090, prospectiveGPU3 UUID recorded beforelaunch,
private manifold supervisor. Internal2400/external2460s, peakallocated cap24GiB
checked at budgetboundaries. Pilot cost supports about850s primary training plus
bounded hashes/evaluation/diagnostics; no runtime promise. Failure/OOM/timeout
preserves partial records without automatic retry. About322GiB shared disk free;
expected compact evidence<25MiB, hard100MiB. No model weights/checkpoints/tensors/
traces saved. Retain compact config/seed/provenance/source/logs/metrics/diagnostics/
summaries/manifests until explicit cleanup. Preserve dirty trees and frozen evidence.
''')
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        historical_bundle=str(prior),historical_screen_sha256=sha(prior/'screen.json'),
        historical_summary_sha256=sha(prior/'summary.json'),
        prerequisite_bundle=str(previous),prerequisite_summary_sha256=sha(previous/'summary.json'),
        evidence_budget_bytes=100*2**20,
        original_owned_runner_sha256=sha(REPO/'benchmarks/cifar_wide2048_runner.py'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)

if __name__=='__main__':main()
