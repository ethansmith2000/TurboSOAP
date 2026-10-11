"""Freeze a narrow fixed-update variance-state learning ablation."""
import json
import os
from pathlib import Path
import shutil
import sys
from prepare_cifar_benchmark import prepare,sha,REPO

KINDS=['qr_carry','qr_overlap','fro_carry','fro_overlap']
def arms():
    aa=[]
    for beta,order in [(.99,KINDS),(.999,KINDS),(.99,list(reversed(KINDS)))]:
        for kind in order:
            repeated=any(x['kind']==kind and x['covariance_beta']==beta for x in aa)
            a=dict(label=f'{kind}_beta{beta}'+('_repeat' if repeated else ''),kind=kind,
                policy='qr20' if kind.startswith('qr') else 'warm20',matrix_lr=.0005,
                matrix_beta2=.999,covariance_beta=beta,transport_precision='highest',
                variance_policy='overlap' if kind.endswith('overlap') else 'permutation',
                basis_ns_iterations=2,basis_rotation_cap=.1,tracking_probe=True,capture_replay=False,
                timing_affinity=dict(main=116,autograd=117))
            if kind.startswith('fro'):a['movement_policy']='fro_cubic2_cap05'
            if repeated:a['repeat_of']=f'{kind}_beta{beta}'
            aa.append(a)
    return aa

def main():
    b=Path(sys.argv[1]).resolve();prior=Path('/workspace/optimizer_replay_results/cifar_soap_radius_learning_20261006')
    assert json.loads((prior/'summary.json').read_text())['fixed_update_qualified']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=3500,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text());aa=arms()
    c['memory_screen']=dict(arms=aa,equal_updates=3500,maximum_updates=3500,
        evaluation_updates=[875,1750,3500],wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,
        timing_version=3,guard_every_refresh=True,tracking_probe_kind='strict_transport',
        tracking_ages=[20,180,200,220,860,1740,3480],profile_steps=0,
        control_pairs=[[a['repeat_of'],a['label']] for a in aa if 'repeat_of' in a])
    c['max_wall_seconds']=1800
    for name in ['cifar_radius_learning_runner.py','cifar_variance_screen.py','variance_probe.py',
        'cifar_affinity_screen.py','timing_diagnostics.py','analyze_timing_traces.py','cifar_tracking_probe.py',
        'cifar_strict_transport.py','movement_retraction.py','warm_retraction.py',
        'prepare_variance_screen.py','summarize_variance_screen.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    topology=[]
    for cpu in [116,117]:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}');assert cpu in os.sched_getaffinity(0)
        topology.append(dict(cpu=cpu,core=int((p/'topology/core_id').read_text()),socket=int((p/'topology/physical_package_id').read_text()),l3_shared=(p/'cache/index3/shared_cpu_list').read_text().strip()))
    assert topology[0]['core']!=topology[1]['core'] and topology[0]['l3_shared']==topology[1]['l3_shared']
    (b/'cpu_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    protocol='''# Fixed-update variance-state ablation

Prospective 2026-10-06. Completed radius .6 learning did not close the warm/QR gap.
This study tests V handling as a hypothesis, not an established cause. Twelve
arms: QR20 and Frobenius .5/cubic2, each with current V carry/reindex or existing
squared-overlap transport, covariance .99/.999; repeat all four .99 methods.
Order four methods forward at .99, forward at .999, reversed .99 repeats.
Seed139, both factors, strict highest M, high basis arithmetic, Adam(.9,.999),
LR .0005, original200-epoch schedule, same cached corrected CIFAR recipe/split
and eager evaluation. Warm20/QR200, covariance every gradient-bearing step.
Every arm gets identical initial model/input/RNG and a fresh optimizer. Only
variance policy changes within each QR/Fro pair; no radius/cadence/LR tuning.

Current policy: carry V at warm events and QR column-order reindex at QR events.
Overlap policy: after the SAME basis and strict M step, U_L=Qold_L^T Qnew_L,
U_R=Qold_R^T Qnew_R, Vnew=(U_L^2)^T Vold U_R^2 at EVERY refresh, clamped >=0.
Reuse ResearchSOAP's existing implementation; do not replace the core. Its V
arithmetic inherits high FP32 matmul precision after strict M restores it. This
is a diagonal approximation, not exact full-second-moment history or a strict-
precision V ablation. The old controlled float64 moment oracle showed its path
dependence and missing cross moments; read/hash that evidence without rerunning.
It supports the question, not a prediction that overlap will improve training.

Primary CE at3500 updates; evaluate875/1750/3500. All12 arms stop at3500.
No wall endpoint or time-to-quality qualification: prior wall failures remain.
Retain full training wall and complete optimizer CUDA-event costs, including
covariance, Adam, basis, M/V handling, casts and host dispatch. First50 excluded
from steady event means only. Init/evaluation/offline probes/hashes/logging are
excluded from update timing. Report repeats and variability, not uninstrumented
throughput. No synthetic timing/learning splice. Fixed update-indexed LR remains.

Main/autograd CPUs116/117, assert/record every arm and restore original masks.
Require12 placement records, two restored masks, stable in-arm compiler counters,
complete175 guards and168 targeted matrix probes per arm. All four .99 repeats
must exactly reproduce fixed model/full optimizer-state/metric/RNG/input/geometry
records and all probes excluding diagnostic time. Carry QR/Fro controls reproduce
18 historical observations including repeats, plus historical fixed optimizer
hashes and old targeted fields. Split/trainer/core hashes and initial model match.
Common fixed-update input hashes must match across all12 arms.

Every init/refresh retains finite moments/parameters/bases, normalized Gram<.05,
nonnegative stored V. Seven sampled ages20/180/200/220/860/1740/3480 retain
selected-basis update error<1e-6, physical M error<1%, Gram<.05. Selected-basis
M oracle uses the ACTUAL V for both updates, so it does not test V correctness.
New independent two-sided V formula on actual selected bases with MATCHED high
arithmetic must equal actual V bitwise and be finite/nonnegative at all samples.
CPU tests establish orientation against a one-transform full diagonal covariance
Kronecker oracle, same-gradient Q/M independence, and pure permutation controls.
Abort actual failures and save partial evidence; no retrospective tolerance edits.
Preserve full strict-refresh <1% diagnostics separately, including QR/reset
disagreements. Additional high-versus-highest V/full-refresh differences are
recorded but do not turn this into a strict-precision or exact-history V claim.

Offline same-incoming-state shadows compare both V policies on the SAME actual
new Q and M. Record V mass ratio, relative V difference, physical update relative
difference/cosine and minimum V. No shadow feeds back. These quantify local
policy effects, not causal explanations of complete learning trajectories or
error versus a full historical second moment. No full large covariance saved.

Require every fixed-control/numerical/record completeness gate before interpreting
paired loss changes. One seed/early3500 updates cannot prove convergence or a
tuned-method ranking. Replicate useful survivors at seeds271/811 then longer
horizons; no automatic broad sweep or default promotion. V policy is an explicit
ablation; all earlier basis/radius/default evidence remains frozen.

One exclusively claimed GPU3/no queue, private manifold supervisor. Internal1800s
/external1860s. Initial324GiB free, expected<15MiB compact evidence, hard100MiB.
No weights/checkpoints/tensor snapshots; state RAM only. Retain source/config/
protocol/version, metrics/logs/masks, summaries/plots and hashes until explicit
research cleanup. Preserve dirty source, all old evidence and other projects.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    oracle=Path('/workspace/optimizer_replay_results/soap_budget_20261001/moment_oracle.json')
    m=json.loads((b/'manifest.json').read_text());m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),evidence_budget_bytes=100*2**20,
        historical_bundle=str(prior),historical_screen_sha256=sha(prior/'screen.json'),prerequisite_summary_sha256=sha(prior/'summary.json'),
        moment_oracle_path=str(oracle),moment_oracle_sha256=sha(oracle))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)

if __name__=='__main__':main()
