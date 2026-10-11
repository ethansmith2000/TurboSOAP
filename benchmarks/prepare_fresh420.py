"""Freeze a new pilot without changing any previous source or evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import sha,REPO
from fresh420_contract import validate_config,audit_primary_math

def verified(b):
    m=json.loads((b/'manifest.json').read_text())
    for n,h in m['source_sha256'].items():assert sha(b/'source'/n)==h,n
    for n,k in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(b/n)==m[k],n
    return m

def main():
    b=Path(sys.argv[1]).resolve();index=int(sys.argv[2]);uuid=sys.argv[3]
    assert index==7 and uuid=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
    assert not b.exists() and shutil.disk_usage(b.parent).free>5*2**30
    root=b.parent;baseline=root/'cifar_soap_saved_operand_20261008'
    wide=root/'cifar_soap_wide2048_pilot_20261007';original=root/'cifar_soap_wide_pilot_20261007'
    bm=verified(baseline);wm=verified(wide);om=verified(original)
    audit_primary_math((baseline/'source/benchmarks/cifar_saved_operand_child.py').read_text(),(REPO/'benchmarks/cifar_fresh420_child.py').read_text())
    b.mkdir();shutil.copytree(baseline/'source',b/'source');shutil.copytree(baseline/'upstream_reference',b/'upstream_reference')
    for n in ['split.json','data_prepared.json']:shutil.copy2(baseline/n,b/n)
    for n in ['wide_probe.py','cifar_tracking_probe.py','movement_retraction.py','warm_retraction.py']:
        shutil.copy2(wide/'source/benchmarks'/n,b/'source/benchmarks'/n)
    for n in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/owned_matrix_hooks.py']:
        assert bm['source_sha256'][n]==wm['source_sha256'][n],n
    for n in ['cifar_fresh420.py','cifar_fresh420_child.py','fresh420_contract.py','prepare_fresh420.py','summarize_fresh420.py']:
        shutil.copy2(REPO/'benchmarks'/n,b/'source/benchmarks'/n)
    old=json.loads((original/'config.json').read_text());c={**old,'max_wall_seconds':1200,
        'fresh420':dict(per_process_seconds=390,smoke_steps=25,measured_steps=420,fingerprint_steps=[1,2,20,21,25])}
    validate_config(c,old);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    task=root/'soap_saved_operand_round_20261008/NEXT_TASK.md'
    protocol=task.read_text()+'''\n## Frozen implementation declaration

Prepared new isolated runner: original saved-operand child primary recipe math
retained by ordered AST call audit; the 2048 pilot's unchanged owned wide probes,
movement kernel and base optimizers are frozen here. Each process has its own
25 QR10 smoke and restored initial model/complete RNG before its sole420 arm.
Allocation horizon420/525 batches, timing chunk10, CPUs116/116 after smoke.
Original1536 config is byte-semantically unchanged except1200 wall limit and
fresh-process metadata. Flags deterministicTrue/benchmarkFalse match latest
saved-operand backend; global algorithmsFalse. GPU7 same UUID, baseline deltas
declared ZERO. Preserve prior results; no old numerical anchors gate this study.

Exact QR comparison is an ordered emitted stream: initial model/RNG/batch plan,
full backend, smoke1/25 caller fingerprints, smoke train/eval parity, full loaded
code metadata, effective optimizer groups; measured1/2/20/21/25 fingerprints;
every finite/Gram guard, selected-basis probes, all42 chunk losses/maxnorms,
100/200/400/420 eval/model/fulloptimizer/RNG/input records; final state digests;
full posttrajectory and postobserver loaded metadata. Only guard/probe diagnostic
seconds, chunk event/host timing and observation trainingseconds are excluded.
Unknown chunk schema fails. QR2 checks each record before another update; mismatch
stops all work, preserves partial evidence, no retry. QR/Fro numerical trajectories
are expected to differ. Complete code metadata is compared ONLY new QR/QR; Fro's
loaded warm kernels remain separately recorded, never trimmed. Original saved
forward/backward ABI and replay parity are individually required for every arm.
Capture/kernel counts recorded independently; no equality gate for runtimecounts.

Model compile/first measured init/first50 and online finite/Gram checks count full
instrumented trainingwall. Caller hashes beforeclip/afterclip/afteroptimizer are
inside original host/event spans; synchronize before their timer so latent model/
optimizer completion stays charged, subtract measured hash-copy/CPUwork from host
trainingwall. Event spans still include those hashes: model event averages are
instrumented diagnostics. Only first50 excluded from steady optimizer event mean.
Synchronized offline wide probes/evals/statehashes/reportwrites/host-flush/profile/
cleanup are outside trainingwall, inside total all-in. This is not uninstrumented
deployment throughput and not a three-repeat wall qualification. Guards are newly
charged online; no comparison of old differently instrumented wall values.

After420 final eval/probes, only original initial-state and saved-operand model-only
replays run, unchanged helper math/ABI/lifecycle. No final-state extra replay or
optimizer snapshot. Replay retained RAM118,675,856bytes, cap128MiB (initial helper
stricter120MiB), discarded sequentially; transient probes/compile/profile within
24GiBpeak, including smoke peak. Offline probes use references to old selected
bases/moments only until flush, not file snapshots. Strict selected update<1e-6,
physicalM<1%,Gram<.05; no fullstrict reference claim. Owned immediate weakrefrelease
without GC/cache clears; restore own main/autograd masks even on failure.

Internal1200/external1260s/eachchild390 withinremaining1200. One nonqueued exclusive
claim/private supervisor/autostartfalse, no automatic next campaign. Expected
compact<25MiB/hard100MiB; no weights/checkpoints/tensor/features/trace files. Compact
configs/seed/source/provenance/metrics/logs/summaries/inventories and verified loaded
wrappers retained for audit until explicit cleanup. Shared disk321GiB free at
preflight.226 priorPythonfiles/33 inventories+members protected. No cache/library/
driver/env/clocks/GC/otherjob/commit/reset/push/delete changes.
'''
    (b/'PROTOCOL.md').write_text(protocol)
    m={**bm,'source_sha256':{str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        'config_sha256':sha(b/'config.json'),'protocol_sha256':sha(b/'PROTOCOL.md'),
        'baseline_bundle':str(baseline),'baseline_manifest_sha256':sha(baseline/'manifest.json'),
        'original_config_bundle':str(original),'original_config_sha256':sha(original/'config.json'),
        'owned_pilot_bundle':str(wide),'owned_pilot_manifest_sha256':sha(wide/'manifest.json'),
        'declared_baseline_backend_deltas':[],'device':dict(index=index,uuid=uuid),
        'next_task_sha256':sha(task),'base_repository_head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    for args,name in [(['diff'],'source_worktree.diff'),(['status','--short'],'source_worktree_status.txt')]:
        (b/name).write_text(subprocess.check_output(['git','-C',str(REPO),*args],text=True))
    print(b)

if __name__=='__main__':main()
