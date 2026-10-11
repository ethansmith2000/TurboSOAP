"""Freeze full-horizon QR/Frobenius timing qualification at recorded placement."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare, sha, REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_affinity_20261006')
    qualification=json.loads((prior/'summary.json').read_text())
    assert qualification['local_repeat_qualified'] and qualification['exact_across_placements']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=3504,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text());arms=[]
    for i,kind in enumerate(['qr','fro','fro','qr','qr','fro']):
        arm=dict(label=f'{kind}_{i}',kind=kind,policy='qr20' if kind=='qr' else 'warm20',
                 matrix_lr=.0005,matrix_beta2=.999,covariance_beta=.99,
                 transport_precision='highest',basis_ns_iterations=2,basis_rotation_cap=.1,
                 tracking_probe=True,capture_replay=i>=4,timing_affinity=dict(main=116,autograd=117))
        if kind=='fro':arm['movement_policy']='fro_cubic2_cap05'
        arms.append(arm)
    c['memory_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=3500,
        evaluation_updates=[875,1750,3500],wall_seconds=None,timing_chunk=25,
        maximum_gram_error=.05,timing_version=3,guard_every_refresh=True,
        tracking_probe_kind='strict_transport',tracking_ages=[20,180,200,220,860,1740,3480],
        control_pairs=[['qr_0','qr_3'],['qr_0','qr_4'],['fro_1','fro_2'],['fro_1','fro_5']],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02,profile_steps=4,
        expected_refresh_guards_per_arm=175,expected_targeted_matrices_per_arm=168)
    c['max_wall_seconds']=1200
    for name in ['cifar_timing_screen.py','cifar_affinity_screen.py','timing_diagnostics.py',
                 'analyze_timing_traces.py','cifar_tracking_probe.py','cifar_strict_transport.py',
                 'movement_retraction.py','warm_retraction.py','prepare_cifar_pinned_timing.py',
                 'summarize_cifar_timing.py','summarize_cifar_pinned_timing.py','plot_cifar_timing.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    topology=[]
    for cpu in [116,117]:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        assert cpu in os.sched_getaffinity(0)
        topology.append(dict(cpu=cpu,core=int((p/'topology/core_id').read_text()),
            socket=int((p/'topology/physical_package_id').read_text()),
            l3_shared=(p/'cache/index3/shared_cpu_list').read_text().strip()))
    assert topology[0]['core']!=topology[1]['core']
    assert topology[0]['socket']==topology[1]['socket'] and topology[0]['l3_shared']==topology[1]['l3_shared']
    (b/'cpu_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    protocol='''# Fixed-placement 3,500-update QR/Frobenius timing qualification

Prospective bounded experiment, 2026-10-06. Prior 1,000-update QR placement
qualification passed; unpinned 3,500-update QR/Fro wall gates failed. This stage
qualifies the longer, two-policy execution path. It does not erase old failures
or by itself collect an equal-wall learning comparison.

Six arms in order QR, Fro, Fro, QR, QR, Fro. Fresh identical initial model and
complete RNG/input streams; seed 139, covariance beta .99, Adam beta2 .999,
both factors, strict M transport, high basis arithmetic, both LRs .0005,
original 200-epoch schedule, warm every 20 optimizer ages and QR reset every 200.
Same corrected CIFAR split/augmentation/eager evaluation. No policy/math change.
Main thread CPU 116, autograd worker CPU 117 in EVERY primary arm and replay,
same shared L3 but distinct physical cores. Only these two threads of this new
process are pinned after smoke; assert masks at setup and restore original masks
in finally. No changes to other processes, GPU clocks, GC or other worker masks.

Evaluate at 875/1750/3500. Require all 18 historical observations exact, including
model/RNG/input hashes, metrics, Gram diagnostics and counters. Within each policy
require exact final full optimizer-state hash, all observations and targeted
probes. All 175 refresh guards and 168 targeted matrix probes per arm must be
present and pass existing Gram <.05 / targeted transport criteria. Preserve QR
full-reference disagreements separately. Fail rather than silently skip missing
records, profiles or placement evidence.

Use ALL three repeats per policy. Primary limits remain <=3% full 3,500-update
training-wall range/mean and <=2% steady optimizer-event range/mean. No trimming,
threshold changes, pooling methods or extending warmup after observing results.
The first 50 updates are excluded only from steady event means, as before; full
training wall includes them. Preserve all 25-update chunks, host wall/thread CPU/
process CPU, GC/context-switch observations and before/after compiler counters.
Require stable in-arm compiler counters. Host and device intervals overlap and
must not be added. CUDA-event intervals can contain CPU dispatch gaps.

After all six primary arms, retain only the last QR/Fro states in RAM for four
updates: plain, profiler1, profiler2 per policy, identical state/RNG at each replay.
Allocate maximum_updates+profile_steps input batches, including the exact epoch
boundary case that failed previously. Require exact model/optimizer/RNG/losses
within each replay triplet. Keep four compressed traces. Filter actual kernel,
gpu_memcpy and gpu_memset categories; retain CPU annotations separately. Require
positive actual device work, all three nonempty CPU phases, and complete launch
correlation attribution. GPU annotation spans are never counted as kernel work.
These windows are perturbed diagnostics, excluded from the primary timing gate.

If every gate passes, qualify only this local fixed-placement, instrumented
execution scope. New equal-wall learning endpoints require a separate declared
comparison. If any gate fails, preserve it and withhold wall-quality claims;
fixed-update algorithm research can continue independently.

One idle GPU3 via exclusive gpu-claim, no queued wait; private manifold supervisor.
Internal 1,200 s / external 1,260 s cap. At preparation about 324 GiB free.
Expected retained evidence <15 MiB, hard evidence budget 100 MiB. No weights,
checkpoints or tensor snapshots saved. Two final states stay in RAM only.
Retain frozen sources/config/protocol, source/version provenance, metrics/logs,
summary/plot, topology/masks and compressed traces for audit until explicit cleanup.
Only newly created temporary raw traces are removed after compression. Existing
working-tree changes, historical evidence and other projects are preserved.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text())
    historical=Path('/workspace/optimizer_replay_results/cifar_soap_movement_budget_20261006')
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),evidence_budget_bytes=100*2**20,
        historical_bundle=str(historical),historical_screen_sha256=sha(historical/'screen.json'),
        prerequisite=str(prior),prerequisite_summary_sha256=sha(prior/'summary.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
