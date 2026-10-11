"""Freeze actual learning and prospective wall qualification after natural gates."""
import json
import os
from pathlib import Path
import shutil
import sys
from prepare_cifar_benchmark import prepare,sha,REPO

KINDS=['qr','fro05','fro06','fro06strong']
def arms():
    result=[]
    # Balanced reversal and a third interleaved round; no time-dependent ordering.
    for beta,order in [(.99,KINDS),(.999,KINDS),(.99,list(reversed(KINDS))),(.99,KINDS)]:
        for kind in order:
            previous=[a for a in result if a['kind']==kind and a['covariance_beta']==beta]
            arm=dict(label=f'{kind}_beta{beta}_r{len(previous)}',kind=kind,
                policy='qr20' if kind=='qr' else 'warm20',matrix_lr=.0005,matrix_beta2=.999,
                covariance_beta=beta,transport_precision='highest',basis_ns_iterations=2,
                basis_rotation_cap=.1,tracking_probe=True,capture_replay=False,
                timing_affinity=dict(main=116,autograd=117))
            if kind!='qr':arm['movement_policy']=dict(fro05='fro_cubic2_cap05',fro06='fro_cubic2_cap06',fro06strong='fro_cubic3_cap06')[kind]
            if previous:arm['repeat_of']=previous[0]['label']
            result.append(arm)
    return result

def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/soap_radius_natural_20261006')
    assert json.loads((prior/'summary.json').read_text())['qualified']
    timing=Path('/workspace/optimizer_replay_results/cifar_soap_pinned_timing_20261006')
    assert json.loads((timing/'summary.json').read_text())['qualified']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=5000,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text()); aa=arms()
    c['memory_screen']=dict(arms=aa,equal_updates=3500,maximum_updates=5000,
        evaluation_updates=[875,1750,3500],wall_seconds=90.,timing_chunk=25,
        maximum_gram_error=.05,timing_version=3,guard_every_refresh=True,
        tracking_probe_kind='strict_transport',tracking_ages=[20,180,200,220,860,1740,3480],
        control_pairs=[[a['repeat_of'],a['label']] for a in aa if 'repeat_of' in a],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02,profile_steps=0)
    c['max_wall_seconds']=2400
    for name in ['cifar_radius_learning.py','cifar_radius_learning_runner.py','cifar_affinity_screen.py',
        'timing_diagnostics.py','analyze_timing_traces.py','cifar_tracking_probe.py','cifar_strict_transport.py',
        'movement_retraction.py','warm_retraction.py','radius_natural.py','retraction_diagnostic.py',
        'benchmark_synthetic_scaling.py','prepare_radius_learning.py','summarize_radius_learning.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    topology=[]
    for cpu in [116,117]:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}');assert cpu in os.sched_getaffinity(0)
        topology.append(dict(cpu=cpu,core=int((p/'topology/core_id').read_text()),socket=int((p/'topology/physical_package_id').read_text()),l3_shared=(p/'cache/index3/shared_cpu_list').read_text().strip()))
    assert topology[0]['core']!=topology[1]['core'] and topology[0]['l3_shared']==topology[1]['l3_shared']
    (b/'cpu_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    protocol='''# Radius .6 paired actual CIFAR learning

Prospective 2026-10-06. Prerequisites: both .6 policies qualified in actual natural
cycles through factor6144, and recorded CPU placement qualified at3500 updates.
16 arms: QR20, Frobenius .5/cubic2, .6/cubic2, .6/cubic3 at covariance .99/.999;
three repeats per policy at .99 and one trajectory per policy at .999.
Order: four policies forward at .99, forward at .999, reversed at .99, forward
at .99. Every arm starts with identical model, augmentation/data/RNG and fresh
optimizer. Seed139, both factors, strict highest M, high basis arithmetic,
V carry/reindex fixed, Adam(.9,.999), both LRs .0005, original200-epoch schedule,
warm20/QR200. Same corrected cached CIFAR100 split, eager evaluation, recipe.
Only radius changes between .5/cubic2 and .6/cubic2; cubic3 is a fixed numerical
control. No LR/cadence/coefficients grid, controller, fallback or default change.

Primary anchor: CE at3500 updates. Evaluate875/1750/3500 and first25-update
boundary reaching NEW90 measured training seconds. Continue until both3500 and
90 seconds observed, capped5000. Retain actual crossing time and overshoot;
missing endpoint stays capped. Training-wall budget excludes initialization,
evaluation, offline guards/probes, hashing and logs; includes model, optimizer,
host launch/dispatch. It is not equal cumulative optimizer time. Keep original
update-indexed LR schedule, no per-arm horizon tuning. Never splice synthetic
timings into CIFAR losses. All historical failed90-second endpoints stay intact.

Main/autograd CPUs116/117 for every arm, same L3/distinct cores; record masks at
each build and restore originals finally. No other thread/process or clocks/GC
changes. No profiler reruns: prior profile path is qualified and unchanged.
Require stable in-arm compiler counters, all16 arm placement records and two
restored masks. Keep all25-update chunk host/event/GC/context-switch samples.

All three .99 repeats of EACH policy must reproduce every fixed-update model,
full optimizer-state hash, metrics/input/RNG/counters/geometry and targeted probes.
Use3500 observation's hash and timing, since final90-second step may differ.
Within-policy full3500 wall range/mean <=3%, steady optimizer-event range/mean
<=2%, no trimming or threshold changes. First50 excluded from steady event means
only; all3500 updates included in training wall. Wall interpretation requires
all four policy repeat gates, exact controls, complete endpoints, numerical
guards and stable compiler/masks. Single .999 arms are exploratory wall endpoints
with no beta-specific repeat qualification; report separately even if .99 passes.
Fixed-update .999 pairing remains valid. No timing or learning winner is assumed.

Every init/refresh guard: finite moments/bases, normalized Gram<.05; all24 matrix
probes at ages20/180/200/220/860/1740/3480: selected-basis update<1e-6,
physical M error<1%, Gram<.05. Abort actual failures and preserve partial evidence.
Full strict-refresh difference<1% remains separately reported; preserve prior
QR/reset disagreements. Warm/reference checks never redefine targeted gates.
Matched QR/.5 controls must reproduce all fixed875/1750/3500 historical anchors
at both betas (24 observations including repeats). Split/trainer/core source
hashes and initial model must match. Repeat diagnostics exclude timing fields.

No causal improvement claim from own-trajectory residual alone. Larger radius
needs useful actual learning and total cost before seeds271/811 and longer runs.
One-seed3500 result on original200-epoch schedule is early, not convergence or a
tuned-method ranking. Numerical correction control can identify drift-associated
effects but no mechanism is inferred from loss alone.

One exclusive GPU3 via gpu-claim, no queue; private manifold supervisor.
Internal2400s/external2460s. Initial323GiB free, expected<20MiB compact evidence,
hard100MiB. No weights/checkpoints/tensor snapshots; RAM state discarded.
Retain source/version/config/protocol, metrics/logs/masks, summaries/plots and
inventory for audit until explicit cleanup. Preserve dirty source and old evidence.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    history=Path('/workspace/optimizer_replay_results/cifar_soap_movement_budget_20261006')
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        evidence_budget_bytes=100*2**20,historical_bundle=str(history),historical_screen_sha256=sha(history/'screen.json'),
        prerequisite=str(prior),prerequisite_summary_sha256=sha(prior/'summary.json'),timing_prerequisite=str(timing),timing_prerequisite_sha256=sha(timing/'summary.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)

if __name__=='__main__':main()
