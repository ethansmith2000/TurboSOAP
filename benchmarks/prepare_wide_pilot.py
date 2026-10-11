"""Freeze a bounded shallow actual-model width1536 pilot, not a wall campaign."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
        matrix_lr=.0005,covariance_beta=.99,branch_steps=420)
    c=json.loads((b/'config.json').read_text())
    c.update(width=1536,depth=1,heads=24,smoke_steps=25,max_wall_seconds=2400)
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
    names=['cifar_wide_pilot.py','cifar_wide_runner.py','wide_probe.py','prepare_wide_pilot.py',
        'summarize_wide_pilot.py','cifar_affinity_screen.py','compact_dispatch_screen.py',
        'dispatch_diagnostics.py','timing_diagnostics.py','analyze_timing_traces.py',
        'cifar_tracking_probe.py','movement_retraction.py','warm_retraction.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Shallow actual-model width1536 pilot

Prospective 2026-10-07. Three interleaved arms QR/Fro/QR, each420 updates,
seed139. Actual CIFAR100 ViT embedding1536, depth1, heads24, batch256 and
evalbatch512; original frozen data/split/augmentation/eager-bf16 evaluation,
LR .0005 original200epoch schedule and5epoch LR warmup. This is2.4 scheduled
epochs, still within LR warmup: early learning only, no convergence/width ranking.
Both factors through6144; covariance .99 accumulates everygradient step,
Adam(.9,.999), V carry/reindex, high basis arithmetic/highest strict M.
QR20 versus Frobeniuscap.5/cubic2 warm20/resetQR200; no EMA/controller change.

One fixed25-update global QR10 cold compile/parity smoke, discarded. No per-arm
400-update warmup: fixed work at this new shape is the feasibility target.
Restore original weights/complete RNG before each fresh empty optimizer.
All420 updates, including cold state initialization/eigh and first50, count full
trainingwall. First50 omitted only from steady eventmeans. Record coldsmoke,
full trainingwall, augmentation/model/completeoptimizer CUDA events and all-in
separately. Offline guards/probes/evaluation/hash/observerflushes excluded from
primary wall but included all-in; all probes stay RAM-only. GPU events include
host dispatch gaps; no exact kernel/untimed deployment throughput claim.

Evaluate100/200/400/420;10-update timingchunks. Optimizerage is trainingstep-1;
resetsage200/400 execute atsteps201/401. Guards initialization/everyrefresh:
21 guards/arm. Six probeages20/180/200/220/380/400, fourmatrices/eightfactors:
24 targetedmoment samples and48 factorrows perarm. Targetedselected-basis
updateerror<1e-6, physicalMdrift<1%, Gram<.05, finiteparameters/moments.
Abort numericalfailure, preserve partialevidence, no relaxedgates/fallback.

Prospective probe scope: selected-basis formula and physicalM only; no repeated
fullQRreference shadow at these largefactors. No fullstrict-pass claim. Preserve
all earlier fullstrict QR/reset discrepancies. Diagnosticrawgenerator uses
matched high arithmetic, recordsnormF/sqrt2, hypothetical .5 clippingscale,
rawRMS, covariance trace/Fro, before/executed residual/Gram, gradientRMS,
Rayleighdiagonal adjacent gaps (NOT eigengaps), unalignedbasis movement (QR
sorting/signs confoundcross-policy interpretation). Twelve fixed float64
poweriterations, fixedlocal seed9100+dimension, approximate ||A||2 frombelow;
bound/proxy is an upper proxy on slack, NOT exactnorm/certification. No SVD,
no additionalQR/eigh and no feedback/globalRNG consumption. Charge offline
probe time separately; these are not onlinecontroller costs.

QRrepeat requires exact allfour observations/model/fulloptimizer/RNG/input/
augmentation/evaluation/guards/probe scalars/chunklosses; compare timing separately.
Fro has one trajectory. This is feasibility/numerics/cost evidence only: NO
three-repeat wallqualification, equalwall endpoint, learningwinner, seedinference
or syntheticcost/loss splice. Report all arms; no trimming/reruns ifrepeat fails.
Measuredcompiler counters must remainstable. Sameprocess main/autograd116/116,
originalmasks restored; no changes to otherjobs/clocks/GC/defaults.

Oneexclusive nonqueued GPU3 RTX5090 UUID recordedbeforelaunch, private
manifold supervisor. Internal2400/external2460seconds, 24GiBallocatedpeak
checkedatbudgetboundaries; OOM/memory/budgetfailure preservespartialevidence.
28,660,324params;8factors1536/4608/6144. GG+Q828MiB algebraonly; measured
peak includes otherstates/activation/casts/workspaces. About322GiBshared disk
free. Expected<25MiBcompactevidence/hard100MiB. No weights/checkpoints/tensors/
traces saved. Retaincompact configs/provenance/metrics/logs/summaries/manifests
until explicitcleanup. Originalsource and completedfrozen evidence preserved.
''')
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        original_runner_sha256=sha(REPO/'benchmarks/cifar_radius_learning_runner.py'),
        evidence_budget_bytes=100*2**20)
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)

if __name__=='__main__':main()
