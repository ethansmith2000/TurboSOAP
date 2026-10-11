"""Freeze a single-core versus adjacent-core dispatch diagnostic."""
import json
import os
from pathlib import Path
import shutil
import sys
try:
    from .prepare_cifar_benchmark import prepare,sha,REPO
except ImportError:
    from prepare_cifar_benchmark import prepare,sha,REPO

def arms():
    return [dict(label=f'{kind}_{i}',kind=kind,policy='qr20',matrix_lr=.0005,matrix_beta2=.999,
        covariance_beta=.99,transport_precision='highest',basis_ns_iterations=2,basis_rotation_cap=.1,
        tracking_probe=True,capture_replay=i>=4,timing_affinity=dict(main=116,autograd=116 if kind=='same' else 117))
        for i,kind in enumerate(['near','same','same','near','near','same'])]

def main():
    b=Path(sys.argv[1]).resolve();assert not b.exists()
    audit=Path('/workspace/optimizer_replay_results/soap_dispatch_round_20261006/audit/audit.json')
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_affinity_20261006')
    old=json.loads((prior/'summary.json').read_text());assert old['local_repeat_qualified'] and old['exact_across_placements']
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=1004,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text());c['max_wall_seconds']=600
    c['memory_screen']=dict(arms=arms(),equal_updates=1000,maximum_updates=1000,evaluation_updates=[875,1000],
        wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=4,guard_every_refresh=True,
        tracking_probe_kind='strict_transport',tracking_ages=[20,180,200,220,860],profile_steps=4,
        control_pairs=[['near_0',x['label']] for x in arms()[1:]],
        timing_repeat_maximum_training_wall_range_over_mean=.03,timing_repeat_maximum_optimizer_range_over_mean=.02,
        expected_refresh_guards_per_arm=50,expected_targeted_matrices_per_arm=120,
        material_model_effect_minimum=.05)
    for name in ['cifar_radius_learning_runner.py','cifar_dispatch_screen.py','dispatch_diagnostics.py',
        'cifar_affinity_screen.py','timing_diagnostics.py','analyze_timing_traces.py','cifar_tracking_probe.py',
        'cifar_strict_transport.py','movement_retraction.py','warm_retraction.py',
        'prepare_dispatch_screen.py','summarize_dispatch_screen.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    topology=[]
    for cpu in [116,117]:
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}');assert cpu in os.sched_getaffinity(0)
        topology.append(dict(cpu=cpu,core=int((p/'topology/core_id').read_text()),
            socket=int((p/'topology/physical_package_id').read_text()),l3_shared=(p/'cache/index3/shared_cpu_list').read_text().strip()))
    assert topology[0]['core']!=topology[1]['core'] and topology[0]['l3_shared']==topology[1]['l3_shared']
    (b/'cpu_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    protocol='''# Autograd co-location: one execution factor, unchanged numerical work

Prospective2026-10-06. Retained timing audit identifies CPU-heavy optimizer
dispatch and variable non-main model/backward CPU; it cannot identify an exact
thread, in-arm GPU clock cause or scheduler wait. Earlier adjacent-versus-far L3
placement establishes a local CPU placement effect. Test autograd worker on main
core116 versus adjacent physical core117, same L3. Main116 in both. This changes
only the worker placement, not thread count, precision, compilation, GPU clocks,
GC or optimizer math. Co-location may help locality or hurt scheduling; unknown.
No other process masks are changed. Restore the two original masks finally.

Six QR20 arms,1000 updates each, near/same/same/near/near/same. Fresh identical
initial model/RNG/optimizer/input; seed139, covariance .99, Adam(.9,.999), both
factors, strict highest M/high basis, V reindex, LR .0005, original200-epoch
schedule and corrected cached CIFAR split/strong augmentation/eager evaluation.
Evaluate875/1000. No learning search or wall-budget endpoint. Original trainer
is unchanged; reuse isolated radius runner byte-identically, with opt-in observer
and existing affinity adapter. Repeat controls are necessary intervention arms,
not historical reruns to recover context.

Require every current fixed model/full optimizer-state/metrics/geometry/RNG/input
field exact across all six arms. All12 historical875/1000 observations and final
optimizer hashes match prior near_0. All six875 optimizer hashes match latest V
carry control. All probes excluding diagnostic time match the historical120 rows
per arm. Require50 finite/geometry guards per arm and selected-basis update<1e-6,
physical M error<1%, normalized Gram<.05. Preserve separate full strict-reference
QR disagreements. Require stable in-arm compiler counters, twelve placement records
(six primary plus six replay builds),
two restored masks, four own-GPU sparse samples per arm, complete40 chunk records.

Primary repeat gates unchanged: ALL three repeats in EACH condition require
<=3% complete1000-update training-wall range/mean and <=2% steady optimizer-event
range/mean. First50 excluded only from event means; full wall includes them. No
trimming, changed warmup or threshold relaxation. A material directional model
placement effect requires exact controls, both repeat gates, >=5% difference in
model/backward event means, nonoverlapping three-run model ranges and the same
direction in all three adjacent matched pairs. Report sign even if co-location
hurts. A numerical pass without those criteria is not a material-effect claim.

Retain original host wall/main CPU/process CPU/GC/context switches plus main
user/system CPU by phase. Four rusage calls per update add small observer work
identically in both conditions, charged in update timing. At each25-step chunk,
read only own-process thread affinity, last CPU, user/system ticks, schedstat
runtime/runqueue wait/timeslices. The reset-to-flush thread interval also contains
offline diagnostics, not exact phase timing; report this limitation. Ticks have
kernel clock resolution; runtime and runqueue wait are distinct. Sparse own-GPU
clocks/temp/power/utilization every250 updates and /proc snapshots are outside
primary update timing; their overhead is stored and can perturb following work.
No controller or online scheduling feedback. Do not add overlapping host/device
times or call CUDA-event spans kernel duration. No uninstrumented throughput claim.

After all primaries, last near/same states held in RAM for four-update
plain/profile1/profile2 replays with exact model/optimizer/RNG/loss checks.
Four compressed traces; corrected kernel/gpu_memcpy/gpu_memset accounting,
positive device work, all three CPU phases, complete launch attribution.
Compare actual GPU operation-name/category/count signatures between placements;
do not mistake GPU annotations for work. Profile times are perturbed diagnostic
windows, excluded from repeat gates. No weights/tensors saved or replay from old
experiments. Input allocation includes maximum_updates+profile_steps.

Successful local1000-update QR qualification does not restore historical failed
wall endpoints or qualify full3500-update QR/Fro. A useful intervention needs
separate full-horizon qualification before a new measured-wall learning campaign.
No optimizer/affinity default promotion from this diagnostic.

One available RTX5090 selected/recorded prospectively at launch, exclusive
gpu-claim without queued wait, private manifold supervisor; internal600/external
660 seconds. Check shared disk and live processes before launch. About323GiB free;
expected<20MiB compact evidence, hard100MiB. No weights/checkpoints/tensor files.
Two replay states in RAM only for specified diagnostic; discarded afterward.
Retain executed sources/config/protocol/provenance, metrics/logs/masks, thread
records and compressed traces until explicit cleanup. Preserve dirty source and
all frozen evidence. No historical rerun or other-project intervention.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text());variance=Path('/workspace/optimizer_replay_results/cifar_soap_variance_20261006')
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),evidence_budget_bytes=100*2**20,
        audit_path=str(audit),audit_sha256=sha(audit),historical_bundle=str(prior),historical_screen_sha256=sha(prior/'screen.json'),
        variance_bundle=str(variance),variance_screen_sha256=sha(variance/'screen.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)

if __name__=='__main__':main()
