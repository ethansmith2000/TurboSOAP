"""Freeze the prospective interleaved timing diagnostic, without training."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import prepare, sha, REPO


def main():
    b=Path(sys.argv[1]).resolve()
    base=Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2')
    prepare(b,base,matrix_lr=.0005,branch_steps=3504,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text())
    arms=[]
    for i,kind in enumerate(['qr','fro','fro','qr','qr','fro']):
        arm=dict(label=f'{kind}_{i}',kind=kind,policy='qr20' if kind=='qr' else 'warm20',
                 matrix_lr=.0005,matrix_beta2=.999,covariance_beta=.99,
                 transport_precision='highest',basis_ns_iterations=2,basis_rotation_cap=.1,
                 tracking_probe=True,capture_replay=i>=4)
        if kind=='fro':arm['movement_policy']='fro_cubic2_cap05'
        arms.append(arm)
    c['memory_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=3500,
        evaluation_updates=[875,1750,3500],wall_seconds=None,timing_chunk=25,
        maximum_gram_error=.05,timing_version=3,guard_every_refresh=True,
        tracking_probe_kind='strict_transport',tracking_ages=[20,180,200,220,860,1740,3480],
        control_pairs=[['qr_0','qr_3'],['qr_0','qr_4'],['fro_1','fro_2'],['fro_1','fro_5']],
        timing_repeat_maximum_training_wall_range_over_mean=.03,
        timing_repeat_maximum_optimizer_range_over_mean=.02,profile_steps=4)
    c['max_wall_seconds']=1200
    for name in ['cifar_timing_screen.py','timing_diagnostics.py','cifar_tracking_probe.py',
                 'cifar_strict_transport.py','movement_retraction.py','warm_retraction.py',
                 'prepare_cifar_timing.py','summarize_cifar_timing.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    protocol='''# Interleaved identical-work timing qualification

Prospective bounded diagnostic, 2026-10-06. No learning-policy search.
Six full3500-update arms: QR, Fro, Fro, QR, QR, Fro. Three repeats per policy,
with fresh identical initial weights, optimizer and complete RNG/data streams.
This balanced pair order reduces simple order confounding; it does not randomize
away all system drift. Use the same seed139, covariance .99, both factors,
strict M, warm20/QR200, LR .0005 and original200-epoch schedule as the prior
movement study. Reuse its875/1750/3500 historical hashes/metrics. Preserve
refresh guards and all seven targeted probes. Production/shared trainer unchanged.

Primary gate: all model/optimizer/RNG/input/metric controls and targeted probe
values exact within each policy, all18 historical observations exact, all
refresh guards and targeted diagnostics passing; range/mean of full3500 training
wall <=3% and steady optimizer event mean <=2% within EACH three-repeat policy.
No outlier removal, pooling policies, post-hoc warmup extension or threshold change.
The first50 updates are excluded only from steady phase means, matching history.
The full training-wall gate includes them. Retain all25-update timing chunks.
A passing gate qualifies this instrumented local session only; the old failure
is not erased and no equal-wall learning winner is inferred from this diagnostic.

Measure host wall, current-thread CPU and process CPU around augmentation,
model/backward (including LR setup, zero_grad and norm clipping) and optimizer.
These overlap device execution; never add host time to CUDA time. CUDA event
intervals can include dispatch starvation and are not kernel-only measurements.
Chunk process scheduling/GC counters include excluded guards/probes and are
correlational context; per-phase host spans cover train_step only. Log compiler
counters and read-only GPU clocks/power/temperature at arm boundaries. Do not
change clocks, CPU affinity, compile mode or GC settings in this stage.

After ALL primary arms, retain the last QR/Fro final states in RAM for4-step
replays crossing the next refresh. Run plain, profiler1, profiler2 from the exact
same model/optimizer/RNG state. Require exact final model/optimizer/RNG/losses.
Keep profiler phase CPU and kernel totals, device interval union/span and compressed
Chrome traces. CUDA union includes kernels/copies; overlaps handled explicitly.
Profiler perturbation and the short window preclude estimating natural throughput
or explaining the historical failure on their own. Record observed causes only
when supported; if unexplained, retain that limitation and propose one bounded
intervention rather than a broad sweep.

One currently idle claimed GPU3 with no queued wait. Private manifold supervisor
only. Internal1200s/external1260s wall bound. Shared disk initially324GiB free.
No weights/checkpoints/tensor snapshots saved. In-memory final states are discarded.
Compact configs, frozen sources, metrics, logs, source diff/provenance, and four
compressed4-step traces retained for audit; expected <40MiB, limit100MiB.
Temporary uncompressed traces are deleted only after compression. No deletion or
modification of historical evidence. Retention: keep this diagnostic bundle until
explicit research cleanup. No default changes or seed/algorithm expansion here.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text())
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
             config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
             evidence_budget_bytes=100*2**20,
             historical_bundle='/workspace/optimizer_replay_results/cifar_soap_movement_budget_20261006',
             checkpoint_policy='No persisted tensors; two final diagnostic states in RAM only.')
    history=Path(m['historical_bundle'])
    m['historical_screen_sha256']=sha(history/'screen.json')
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
