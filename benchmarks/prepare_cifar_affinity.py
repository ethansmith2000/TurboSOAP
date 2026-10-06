"""Freeze a small, prospective CPU placement intervention after timing diagnosis."""
import json
from pathlib import Path
import shutil
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_timing_20261006')
    report=json.loads((prior/'screen.json').read_text())
    assert report['status']=='completed','Finish the existing bounded diagnostic first'
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=1004,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text());arms=[]
    for i,kind in enumerate(['near','far','far','near','near','far']):
        arms.append(dict(label=f'{kind}_{i}',kind=kind,policy='qr20',matrix_lr=.0005,
            matrix_beta2=.999,covariance_beta=.99,transport_precision='highest',
            basis_ns_iterations=2,basis_rotation_cap=.1,tracking_probe=True,capture_replay=i>=4,
            timing_affinity=dict(main=116,autograd=117 if kind=='near' else 124)))
    c['memory_screen']=dict(arms=arms,equal_updates=1000,maximum_updates=1000,
        evaluation_updates=[875,1000],wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,
        timing_version=3,guard_every_refresh=True,tracking_probe_kind='strict_transport',
        tracking_ages=[20,180,200,220,860],profile_steps=4,
        control_pairs=[['near_0',a['label']] for a in arms[1:]],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02)
    c['max_wall_seconds']=600
    for name in ['cifar_timing_screen.py','cifar_affinity_screen.py','timing_diagnostics.py',
                 'cifar_tracking_probe.py','cifar_strict_transport.py','movement_retraction.py',
                 'warm_retraction.py','prepare_cifar_affinity.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    topology=json.loads((prior/'cpu_topology.json').read_text());t={x['cpu']:x for x in topology}
    assert len({t[x]['core'] for x in [116,117,124]})==3
    assert len({t[x]['socket'] for x in [116,117,124]})==1
    assert t[116]['l3_shared']==t[117]['l3_shared']!=t[124]['l3_shared']
    (b/'cpu_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    protocol='''# CPU placement intervention: same numerical work, two placements

Prospective follow-up within the timing investigation, 2026-10-06. The preceding
six-arm unpinned diagnostic reproduced unstable timing, host/event model changes
and thread migration without in-arm compilation. This is a local causal
intervention on thread placement, not a retroactive explanation of undocumented
historical CPU placement, and not a comparison of GPU hardware or optimizers.

Six1000-update QR20 arms from identical initial weights/optimizer/RNG; order
near,far,far,near,near,far. CPU main thread116 in both. Autograd CUDA worker117
(same shared L3) or124 (different L3, same socket), all distinct physical cores.
These choices follow observed occupied regions of the current diagnostic and
verified sysfs topology; no core search. Pin ONLY these two threads of the NEW
experiment process after compile smoke. Do not modify other processes, GPU
clocks, other worker masks, GC, math, compilation or thread counts. Restore
original masks in finally. Actual tids/masks recorded at every arm/replay setup.

Same seed139, covariance .99, strict M, both factors, QR20, LR .0005 and original
200-epoch schedule.1000 updates is a dispatch diagnostic horizon, not learning
replication or full3500 wall qualification. Retain all guards and targeted probes
at20/180/200/220/860. Evaluate875 and1000; all six875 historical controls must be
exact. All twelve current observations, final full optimizer hashes, input/RNG
and probes must match across placement conditions. Profiles replay4 updates from
identical1000 states with plain/profile1/profile2 numerical checks, as before.

Within EACH placement retain <=3% training-wall and <=2% steady optimizer
range/mean, all three repeats, no trimming. For a material directional placement
effect require exact numerical controls, stable compile counters, >=5% relative
difference in mean model/backward event time, nonoverlapping three-run model
ranges, and the same direction in all3adjacent near/far pairs. Report raw host
wall/current-thread/process CPU, device events, profiler kernel sums and traces.
Profiler windows are short and perturbed; kernel totals are diagnostic. A failed
repeat gate remains failed even if a placement effect exists. A successful
1000-update placement intervention still requires longer QR/Fro qualification
before equal-wall learning claims. No improvement to optimizer math is implied.

One claimed GPU3, no wait; private supervisor, internal600s/external660s. Expected
<40MiB, bounded100MiB retained evidence, no persisted model/optimizer/tensor
snapshots. Source/config/protocol, logs, metrics, CPU topology/masks and compressed
traces retained for audit until explicit cleanup. Two end states in RAM only.
Check free space and claims afresh. Preserve prior experiment and failures.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        evidence_budget_bytes=100*2**20,prerequisite=str(prior),prerequisite_screen_sha256=sha(prior/'screen.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
