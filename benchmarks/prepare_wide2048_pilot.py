"""Freeze one actual width2048 pilot with owned adapter cleanup."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_wide_pilot_20261007')
    assert json.loads((prior/'summary.json').read_text())['feasibility_qualified']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
        matrix_lr=.0005,covariance_beta=.99,branch_steps=420)
    c=json.loads((b/'config.json').read_text())
    c.update(width=2048,depth=1,heads=32,smoke_steps=25,max_wall_seconds=2400)
    arms=[]
    for i,kind in enumerate(['qr','fro','qr']):
        a=dict(label=f'{kind}_{i}',kind=kind,policy='qr20' if kind=='qr' else 'warm20',
            matrix_lr=.0005,matrix_beta2=.999,covariance_beta=.99,transport_precision='highest',
            basis_ns_iterations=2,basis_rotation_cap=.1,tracking_probe=True,capture_replay=False,
            timing_affinity=dict(main=116,autograd=116))
        if kind=='fro':a['movement_policy']='fro_cubic2_cap05'
        arms.append(a)
    c['memory_screen']=dict(arms=arms,equal_updates=420,maximum_updates=420,
        evaluation_updates=[100,200,400,420],wall_seconds=None,timing_chunk=10,
        maximum_gram_error=.05,guard_every_refresh=True,tracking_probe_kind='wide_selected',
        tracking_ages=[20,180,200,220,380,400],control_pairs=[['qr_0','qr_2']],profile_steps=0,
        maximum_peak_allocated_bytes=24*2**30)
    names=['cifar_wide2048_pilot.py','cifar_wide2048_runner.py','owned_matrix_hooks.py','wide_probe.py',
        'prepare_wide2048_pilot.py','summarize_wide2048_pilot.py','cifar_affinity_screen.py',
        'compact_dispatch_screen.py','dispatch_diagnostics.py','timing_diagnostics.py',
        'analyze_timing_traces.py','cifar_tracking_probe.py','movement_retraction.py','warm_retraction.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Actual shallow width2048 pilot with explicit adapter teardown

Prospective 2026-10-07. One new width, no sweep. QR/Fro/QR, each 420 updates,
seed139. CIFAR100 ViT embedding2048, depth1, heads32, batch256/evaluation512.
Original frozen data/split, augmentation and eager-bf16 evaluation. LR .0005,
original 200-epoch schedule with 5-epoch LR warmup. 420 updates is 2.4 epochs,
still within warmup: early learning only, no convergence or cross-width ranking.
Both factors: actual matrix shapes6144x2048,2048x2048,8192x2048,2048x8192;
50,796,644 parameters, eight factors. GG+Q FP32 storage1472MiB and matrix M/V
384MiB are algebra only; other states/activations/casts/workspaces add memory.
Covariance .99 every gradient step, Adam(.9,.999), V carry/reindex, high basis
arithmetic/highest strict M. QR20 versus Fro cap.5/cubic2 warm20/resetQR200.
No EMA, cap, damping, controller, covariance, LR or production default changes.

Use the SAME numerical kernels and diagnostic probe as the1536 pilot. A new
isolated runner adds owned-hook restoration and memory observations only.
Capture the two instance attributes before adding our adapters. After recording
all arm metrics/hashes, close the probe and restore the previous instance
attributes and configured group keys; delete probe/optimizer/lease references.
Require weak references to both optimizer and matrix optimizer to be dead
immediately, without explicit GC collection or allocator cache clearing. Capture
allocated bytes before measured update1, per-arm process peak, after cleanup and
cleanup time separately; all-in includes cleanup. Check the cold-smoke optimizer
is also released. Retain original GC settings. Per-arm process peak still includes
shared model/data/compiler/diagnostic work and is not pure optimizer storage.
No identical-width GPU rerun to quantify memory changes against earlier evidence.
CPU trajectory/probe/RNG parity and object-release checks must pass before launch.

One global fixed25-update QR10 compilation/parity smoke, discard its trained
state. No400-update disposable policy warmup. Restore original weights and
complete RNG before each fresh empty optimizer. All420 measured updates count
full training wall, including initialization/eigh and first50. First50 omitted
only from steady event means. Capture augmentation/model/complete optimizer CUDA
events, full measured training wall and total all-in separately. Offline guards,
probes, evaluations, hashes, observer flushes and teardown are outside primary
wall and included all-in. CUDA events include host dispatch gaps, not exact kernel
cost or uninstrumented deployment throughput. No profiler replays/traces/states.

Evaluate100/200/400/420, timing chunks10. Optimizer age is trainingstep-1;
QR reset ages200/400 execute atsteps201/401. Guard init/every refresh:21 perarm.
Probe ages20/180/200/220/380/400:24 matrix samples/48 factor rows perarm.
Require selected-basis updateerror<1e-6, physical-M drift<1%, Gram<.05, finite
parameters/moments. Abort numerical failure, preserve partial evidence. No relaxed
gates, fallback, retries or horizon extension. Scope remains selected-basis /
physical-M only; no second fullQR reference shadow and no fullstrict-pass claim.
Older fullstrict QR/reset disagreements and the warmed90 wall failure remain.

Offline generator uses matched high math: raw Fro/sqrt2 bound, hypothetical
cap.5 scale, rawRMS, covariance trace/Fro, gradientRMS, before/executed tracking
residual/Gram, Rayleigh adjacent diagonal gaps (not eigenvalue gaps), unaligned
basis movement (QR sign/sort confound). Twelve deterministic float64 power
iterations with local seed9100+dimension give an operator norm lower proxy;
bound/proxy is an upper proxy on bound slack, not exact norm or a certificate.
No SVD, additionalQR/eigh, online feedback or global RNG use. Diagnostic time
retained separately, not a deployed controller cost.

Require exact QR repeats at all four observations, model/fulloptimizer/RNG/input/
augmentation/evaluation, guard/probe scalars and chunk losses; timings separate.
Fro one trajectory; NO three-repeat wall qualification, equal-wall endpoint,
learning winner, independent-seed statistics or synthetic-cost/loss splice.
Compiler counters must be stable. Main/autograd116/116 in this process only;
original masks restored, complete observer/mask records. No other job changes.
Cross-width diagnostics are descriptive, with differing model/gradient/covariance
histories and matrix roles. Do not infer dimension causality from two trajectories.

Prospective GPU3 UUID recorded before launch, exclusive nonqueued gpu-claim,
private manifold supervisor. Internal2400/external2460 seconds, allocated-memory
cap24GiB checked at budget boundaries. OOM/runtime/memory/release failure retains
partial records, no automatic retry. About322GiB shared disk free. Expected
compact evidence<25MiB, hard100MiB. No weights/checkpoints/tensors/traces saved.
Retain configs/seeds/source/version/logs/metrics/diagnostics/summaries/manifests
until explicit cleanup. Preserve dirty working trees and all completed evidence.
After feasibility, longer matched1536 learning past LR warmup and its own fresh
three-repeat timing qualification are the next research stage.
''')
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        previous_width_bundle=str(prior),previous_width_screen_sha256=sha(prior/'screen.json'),
        previous_width_summary_sha256=sha(prior/'summary.json'),evidence_budget_bytes=100*2**20,
        original_wide_runner_sha256=sha(REPO/'benchmarks/cifar_wide_runner.py'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)

if __name__=='__main__':main()
