"""Freeze the new 90-second comparison with disposable policy warmup."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare, sha, REPO


def main():
    b = Path(sys.argv[1]).resolve()
    prior = Path('/workspace/optimizer_replay_results/cifar_soap_colocated_timing_20261007')
    assert json.loads((prior/'summary.json').read_text())['qualified']
    prepare(b, Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005, covariance_beta=.99, branch_steps=6000)
    c = json.loads((b/'config.json').read_text())
    arms = []
    for i, kind in enumerate(['qr', 'fro', 'fro', 'qr', 'qr', 'fro']):
        arm = dict(label=f'{kind}_{i}', kind=kind,
                   policy='qr20' if kind=='qr' else 'warm20', matrix_lr=.0005,
                   matrix_beta2=.999, covariance_beta=.99, transport_precision='highest',
                   basis_ns_iterations=2, basis_rotation_cap=.1, tracking_probe=True,
                   capture_replay=False, timing_affinity=dict(main=116, autograd=116))
        if kind=='fro': arm['movement_policy']='fro_cubic2_cap05'
        arms.append(arm)
    c['memory_screen'] = dict(arms=arms, equal_updates=3500, maximum_updates=6000,
        evaluation_updates=[875,1750,3500], wall_seconds=90., timing_chunk=25,
        maximum_gram_error=.05, guard_every_refresh=True,
        tracking_probe_kind='strict_transport', tracking_ages=[20,180,200,220,860,1740,3480],
        control_pairs=[['qr_0','qr_3'],['qr_0','qr_4'],['fro_1','fro_2'],['fro_1','fro_5']],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02,
        execution_warmup_updates=400, maximum_wall_overshoot_seconds=1., profile_steps=0)
    c['max_wall_seconds']=1800
    for name in ['cifar_radius_learning_runner.py','cifar_budget_runner.py','cifar_warmed_budget.py',
                 'budget_warmup.py','cifar_affinity_screen.py','compact_dispatch_screen.py',
                 'dispatch_diagnostics.py','timing_diagnostics.py','analyze_timing_traces.py',
                 'cifar_tracking_probe.py','cifar_strict_transport.py','movement_retraction.py',
                 'warm_retraction.py','prepare_warmed_budget.py','summarize_warmed_budget.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    protocol='''# New 90-second QR/Fro comparison after disposable warmup

Prospective 2026-10-07. Previous same-core full3500 qualification passes; this
new campaign must qualify its OWN controls. Preserve all previous failed wall
endpoints. Six arms QR/Fro/Fro/QR/QR/Fro, three repeats per method, seed139,
covariance .99, Adam(.9,.999), both factors, strict highest M/high basis,
V carry/reindex, QR20 versus Fro .5/cubic2 warm20/QR200. Original recipe,
data/split/augmentation/evaluation and LR .0005/200-epoch update schedule.
Main/autograd CPU116/116; twelve policy builds, two original masks restored.
No other job, clocks, GC, worker-count or optimizer default changes.

Global cold smoke128 compiles training and checks train/evaluation parity.
Before EACH arm, execute a FIXED400-update disposable warmup using that same
policy and placement; cover two QR200 cycles and the intermediate warm path.
Record duration, final model/optimizer hashes/counters, finite geometry guard
and compiler counters. Discard optimizer; restore original model AND complete
RNG (torch CPU/CUDA, Python, augmentation generators). Require restoration hashes
and a fresh empty measured optimizer. Reset observer rows before measurement.
Warmup repeats must match numerically within each method. No adaptive extension,
measured trajectory prefix or uncounted trained state. Cold setup and warmup are
outside the 90-second budget but individually retained and included in all-in time.
Allocation/state initialization during measured update1 remains charged.

Start measured training at update1 with the original LR/data stream. ALL measured
updates count toward 90 training seconds, including first50. Drop first50 only
from steady event means. Budget includes augmentation/model/optimizer and host
dispatch/synchronization; excludes initialization, warmup, evaluation, offline
guards/probes/hashes/observer flushes/logs. This is measured training time, not
total job time or exact cumulative optimizer time. Record all-in separately.
No synthetic timing/loss splice or uninstrumented deployment throughput claim.

Evaluate fixed875/1750/3500 and first25-update boundary at/above90 seconds.
Stop once both fixed3500 and the new wall endpoint exist; hard6000 updates.
Require the prior boundary below90, actual crossing>=90 and overshoot<=1s.
Missing/capped endpoint or larger overshoot fails wall qualification; preserve
records without extending limits or interpolating CE. Do not tune LR to endpoint.
Final endpoints may differ by steps; retain each CE, actual time, overshoot, steps,
model/full optimizer/RNG/input hashes and counters. Repeats share one seed and
are timing controls, not independent-seed statistical replications.

All18 historical fixed observations INCLUDING optimizer hashes, repeated fixed
trajectories and targeted probes must match previous qualified QR/Fro controls.
Require all175 guards through3500 per arm, all168 targeted M samples per arm at
ages20/180/200/220/860/1740/3480; every further init/refresh guard also passes.
Selected-basis update<1e-6, physical M<1%, Gram<.05, finite bases/moments. Abort
actual failures, preserve partial evidence. Full strict QR/reset disagreements
remain separate; no tolerance relaxation. Original trainer/core/base hashes
are preserved; new isolated runner differs only by disposable warmup insertion.

Use ALL three repeats per method. Fixed3500 full training-wall range/mean<=3%
and steady optimizer-event range/mean<=2% for BOTH methods, no trimming. Use
the3500 observation's times, not varying final endpoint lengths. Require stable
measured compiler counters, complete25-update chunk rows, recorded same-core
placement/restoration and own-GPU observations on the global250-update flush
cadence. Lossless affinity catalog and all observed thread counters retained.
Snapshot intervals include offline work and omit unmatched transient threads.
No profiler repetition: existing diagnostic is complete, this campaign saves no
model/optimizer snapshots or traces. No observer feeds back into policy or clocks.

Only if all numerical, warmup, record, repeat and endpoint gates pass interpret
this NEW campaign's90-second CE/steps as qualified at its local instrumented
scope. Report each repeat plus mean/range, not an independent-seed confidence
interval or tuned-method winner. No .999 wall inference, old endpoint promotion
or automatic seed/radius/V expansion. A useful learning effect needs replication.

One free RTX5090 recorded before first arm, exclusive nonqueued gpu-claim;
private manifold supervisor. Internal1800/external1860s. About322GiB shared disk
free at preparation. Expected<25MiB compact evidence, hard100MiB. No saved
weights/checkpoints/tensor snapshots; all disposable state stays in RAM and is
discarded. Retain source/config/protocol/provenance, metrics/logs, warmup records,
masks/observer records and summaries/inventories until explicit cleanup.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        evidence_budget_bytes=100*2**20,prerequisite=str(prior),
        prerequisite_summary_sha256=sha(prior/'summary.json'),historical_bundle=str(prior),
        historical_screen_sha256=sha(prior/'screen.json'),
        original_runner_sha256=sha(REPO/'benchmarks/cifar_radius_learning_runner.py'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    shutil.copy2(prior/'cpu_topology.json',b/'cpu_topology.json')
    print(b)


if __name__=='__main__':main()
