"""Freeze a narrow radius/correction attribution on unchanged CIFAR trajectories."""
import json
from pathlib import Path
import shutil
import sys
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    b=Path(sys.argv[1]).resolve()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_pinned_timing_20261006')
    result=json.loads((prior/'summary.json').read_text())
    assert result['status']=='completed' and all(p['exact_controls'] and p['guards_pass'] and p['targeted_pass'] for p in result['policies'])
    # Fixed-update attribution remains valid if the separate wall gate failed.
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=1000,covariance_beta=.99)
    c=json.loads((b/'config.json').read_text());arms=[]
    for i,beta in enumerate([.99,.999,.99]):
        arm=dict(label=f'fro_beta{beta}'+('_repeat' if i==2 else ''),kind='fro',policy='warm20',
            matrix_lr=.0005,matrix_beta2=.999,covariance_beta=beta,transport_precision='highest',
            basis_ns_iterations=2,basis_rotation_cap=.1,movement_policy='fro_cubic2_cap05',
            tracking_probe=True,capture_replay=False,timing_affinity=dict(main=116,autograd=117))
        if i==2:arm['repeat_of']=arms[0]['label']
        arms.append(arm)
    c['memory_screen']=dict(arms=arms,equal_updates=1000,maximum_updates=1000,
        evaluation_updates=[875,1000],wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,
        timing_version=3,guard_every_refresh=True,tracking_probe_kind='strict_transport',
        tracking_ages=[20,40,60,180,220,240,260,380,420,440,460,860],profile_steps=0,
        control_pairs=[[arms[0]['label'],arms[-1]['label']]])
    c['max_wall_seconds']=900
    for name in ['cifar_timing_screen.py','cifar_affinity_screen.py','cifar_radius_screen.py','radius_probe.py',
        'timing_diagnostics.py','analyze_timing_traces.py','cifar_tracking_probe.py','cifar_strict_transport.py',
        'movement_retraction.py','warm_retraction.py','prepare_cifar_radius.py','summarize_cifar_radius.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    protocol='''# Radius versus correction work on identical CIFAR states

Prospective bounded attribution, 2026-10-06. No new trained algorithm or default.
Three 1,000-update baseline Frobenius cap .5/cubic2 trajectories at covariance
.99, .999, then .99 repeat. Seed 139, both factors, strict M, high basis arithmetic,
Adam beta2 .999, LR .0005 and original 200-epoch schedule. Main/autograd CPUs
116/117; same cached split, augmentation and corrected evaluation. Real baseline
weights/metrics must match historical step 875 at both betas and the repeat.

At warm ages 20/40/60/180/220/240/260/380/420/440/460/860, apply these shadows to
identical incoming covariance, Q, M and V without feeding them back to training:
(1) baseline cap .5/cubic2; (2) radius-only cap .6/cubic2 diagnostic control;
(3) cap .6/cubic3 fixed; (4) cap .6 with cubic3 on odd warm indices and cubic2 on
even indices; (5) matched QR with its normal V reindexing as a cost reference.
All warm candidates use the SAME skew-Frobenius clipping rule and strict M.
The scheduled choice is predetermined from age//20, not an online error controller.
Warm20/QR200 gives five cubic3 and four cubic2 refreshes between resets, with a
stronger first refresh after each QR. This is four tightly related warm choices,
not a radius/coefficient grid. Cap .6/cubic2 isolates radius and is not presumed
qualified. Require the shadow baseline to equal the actual executed refresh bitwise.

Rationale in exact arithmetic for initially orthogonal Q: cubic polar maps Gram
error d to -.75*d*d+.25*d*d*d. Cap .5 gives d0=.25 and cubic2 |d|=.00140457.
Cap .6 gives d0=.36, cubic2 |d|=.00564376, cubic3 |d|=.000023934.
Incoming Gram error, finite precision and accumulated transport still need checks.
No guarantee follows from these scalars alone. No online spectral solve is added.

Record per-factor movement, Gram, covariance-residual change, and matrix world-M
preservation error. Shadow target: finite, Gram <.05 and world-M error <.01 at
every sampled state. Retain shadow threshold failures instead of relaxing them.
Also run a nine-refresh transport chain at starting ages 20 and 220 on the same
frozen covariance with NO new gradients/EMA/Adam updates; compare final/intermediate
physical M to the initial physical M. Require <1% maximum drift and Gram <.05 as
a prospective shadow screen. This is a frozen-covariance accumulation diagnostic,
not actual whole-history drift or a learning trajectory. Preserve that distinction.
All actual baseline guards and selected-basis transport tests remain unchanged.

At ages 20/40/220/240, measure the complete refresh wrapper for both factors,
covariance projection, clipping/retraction, strict M, matching V policy, casts and
host dispatch. Three rotated policy rounds, three calls per round, one warmup.
Timing excludes offline diagnostics and proxy construction; includes fresh state/
settings dictionaries. It excludes covariance accumulation and the Adam update,
so it is NOT whole-optimizer-step throughput. Report raw rounds and medians by
shape/age; no synthetic/CIFAR time-to-quality splice. Scheduled cost can be
summarized with 5/9 strong +4/9 cheap weights, explicitly as a sampled cost estimate.

Use exact repeat controls/probes with timing fields removed. Expect 150 actual
refresh guards, 864 targeted matrix probes, 3 exact historical step-875 anchors,
and both repeated observations. Exclude the repeated trajectory from pooled
candidate statistics. A shadow passing here still needs actual natural-cycle
state/transport qualification, then paired learning. Residual alone never selects
a learning winner. No new seeds, LR tuning, broad grid or default promotion.

Prerequisite: pinned timing experiment completed with exact numerical controls.
Its wall gate is separate; if it failed, this remains fixed-update attribution.
One exclusively claimed GPU3, no queued wait; private supervisor. Internal900 s /
external960 s limit. Expected <10 MiB compact evidence, hard100 MiB allowance;
about324 GiB initially free. No weights, checkpoints or tensor snapshots saved.
All states stay in RAM. Retain config/protocol, executed sources, numerical/cost
metrics, logs, summaries and manifest until explicit research cleanup.
'''
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    m=json.loads((b/'manifest.json').read_text());history=Path('/workspace/optimizer_replay_results/cifar_soap_movement_budget_20261006')
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),evidence_budget_bytes=100*2**20,
        prerequisite=str(prior),prerequisite_summary_sha256=sha(prior/'summary.json'),
        historical_bundle=str(history),historical_screen_sha256=sha(history/'screen.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
