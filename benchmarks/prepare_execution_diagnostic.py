"""Freeze the declared 2x2x2 numerical execution diagnostic, no learning extension."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,covariance_beta=.99,branch_steps=3500)
    c=json.loads((b/'config.json').read_text())
    contexts=[]
    for repeat in range(2):
        for horizon,chunk in [(420,10),(3500,25),(420,25),(3500,10)]:
            contexts.append(dict(label=f'plan{horizon}_flush{chunk}_r{repeat}',
                                 allocation_horizon=horizon,timing_chunk=chunk,repeat=repeat))
    c.update(width=1536,depth=1,heads=24,max_wall_seconds=600,smoke_steps=25)
    c['execution_diagnostic']=dict(contexts=contexts,smoke_steps=25,measured_steps=25,
        fingerprint_steps=[1,2,20,21,25],per_process_seconds=150,
        maximum_peak_allocated_bytes=24*2**30,expected_evidence_bytes=25*2**20)
    names=['execution_fingerprints.py','cifar_execution_child.py','cifar_execution_diagnostic.py',
        'prepare_execution_diagnostic.py','summarize_execution_diagnostic.py','owned_matrix_hooks.py',
        'compact_dispatch_screen.py','dispatch_diagnostics.py','timing_diagnostics.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Fresh-process actual width1536 numerical execution factorial

Prospective 2026-10-07. Eight fresh independent processes: allocation horizon
420/3500 x timing flush10/25 x two repeats. Sequential order 420/10,3500/25,
420/25,3500/10, then repeat. Processes share the normal framework compile cache;
do not clear/edit it or force cache misses. Natural compiler cache use is recorded.
This tests fresh-process execution under current cache state, not a cold-cache
or independent-seed study. Order/cache/other-project load remain potential
confounds. No model hooks, kernel/backend/clock/GC/controller changes.

Per process fixed25 discarded QR10 compilation/parity smoke, restore original
weights and complete RNG, fresh QR20/.99 covariance/Adam(.9,.999)/strict highest
M/high basis/current V carry-reindex/both factors;25 measured updates. Exactly
400 optimizer updates total (200 smoke+200 measured). No400 disposable warmup,
3500 learning endpoint, old experiment replay, auto-extension or retry. ViT
1536/depth1/heads24, seed139, batch256, original CIFAR100 frozen split/augmentation,
LR .0005 on original200epoch/5epoch warmup schedule. Shared data/model math
unchanged. Main/autograd own116/116 AFTER global smoke, original masks restored.
Backend matmul high, cuDNN allow_tf32=True/benchmark=True as recorded baseline.
Do not enable determinism or select a hypothetical fix in this study.

Allocate complete epoch plans before compilation: ceil420/175=3 epochs=525
device batch-index tensors versus3500=20 epochs=3500; first25 batches identical.
Every process fingerprints measured1/2/20/21/25 and smoke1/25: actual logits/loss,
gradients before/after clip, model/full optimizer hashes, GG/Q tensor hashes and
finite/norm/max scalars, RNG/input/target/ids/counters. Copies are RAM-only and
discarded immediately. Hashing inside caller training scopes deliberately
synchronizes these diagnostic updates; this is NOT a timing/learning qualification
or an exact recreation of the previous host execution. Timing events and host
chunks document declared flush behavior; do not interpret their durations as
optimizer speed. Guards measured1/21 require finite weights/moments/Gram<.05.
Recorded finite fields must pass. This diagnostic adds no full strict QR shadow;
all previous full strict discrepancies and failed historical-bit gate remain.

Record backend determinism/TF32/reduced precision/SDP flags, allowed relevant
environment, Torch/CUDA/cuDNN versions and loaded library paths. Snapshot actual
Inductor PyCodeCache loaded module identifiers/source hashes and compiled Triton
configurations after smoke/measured/profile, rather than infer execution from
counter totals. These are loaded modules, not guaranteed per-operation backend
algorithm IDs. Runtime GPU kernel names/counts collected in one RAM-only
final-state model forward/backward window, with preceding plain call. No
optimizer step in either call. Require identical output/loss/gradient fingerprints
and unchanged model/fulloptimizer/complete RNG before/after. The profiler window
is at final25 weights and does not identify every kernel used at update1;
cuDNN internal algorithm IDs and library-internal binaries may remain unavailable.
No trace, model weights, checkpoints or tensor snapshot files. Backend library
paths/version metadata do not prove every internal implementation identical.

Compare initialization and chronological smoke/measured stages exactly. A mismatch
is a DIAGNOSTIC finding, not justification to loosen the failed historical gate.
Locate first sampled difference in input/RNG/output/loss/preclip/postclip/model/
covariance/basis/fulloptimizer against code/runtime provenance. Hash equality is
bit equality; hashes alone do not provide elementwise numerical error sizes.
No retained historical update1 fingerprints exist, so don't identify the first
historical divergence or attribute a cause from this factorial alone. If all25
agree report that bounded negative result only; no100-anchor claim or extension.

One exclusive nonqueuedGPU3 RTX5090 UUID GPU-55fc0669-93bd-9223-674d-212086d4258b
under private manifold supervisor. Internal600/external660seconds; child150s
within remaining global600, no retry. Per-child allocated cap24GiB. Capacity
checked before freeze, about322GiB free. Compact evidence expected<25MiB/hard
100MiB. Retain configs/source/provenance/logs/hashes/metrics/manifests until explicit
cleanup; no large artifacts. Preserve all pre-existing Python and23 prior
artifact inventories. Failure/timeout preserves partial evidence and releases
claim; no automatic learning launch.
''')
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
             config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
             evidence_budget_bytes=100*2**20,checkpoint_policy='No tensors/weights/checkpoints/traces; compact fingerprints only.')
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
