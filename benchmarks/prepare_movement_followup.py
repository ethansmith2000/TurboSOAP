"""Freeze sequential, gated movement qualification and measured-wall learning."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare, sha, REPO


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['natural','learning'],required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prerequisite',type=Path,required=True)
    a=p.parse_args();b=a.bundle.resolve();prior=json.loads((a.prerequisite/'summary.json').read_text())
    assert prior['status']=='completed'
    if a.stage=='natural':
        selected=next(x for x in prior['rows'] if x['policy']=='fro_cubic2')
        assert selected['world_m_failures']==0 and selected['maximum_gram']<.05
        assert not b.exists();(b/'source/benchmarks').mkdir(parents=True)
        c=dict(widths=[768,1536],variants=['qr20','scaled_ns6','spectral_cubic2_cap05','fro_cubic2_cap05'],
            betas=[.99,.999],updates=420,wall_budget_seconds=900,attribution=False,
            repeat_variants=['spectral_cubic2_cap05','fro_cubic2_cap05'],seed=20261005,
            artifact_budget_bytes=20*2**20)
        source=['soap.py','soap_reference.py']+['benchmarks/'+n for n in [
            'warm_retraction.py','movement_retraction.py','movement_natural.py','retraction_diagnostic.py',
            'benchmark_synthetic_scaling.py','cifar_tracking_probe.py','prepare_movement_followup.py']]
        protocol='''# Skew-Frobenius cubic2 natural-cycle qualification

Selected after common-state attribution passed the unchanged sampled geometry/M
criteria. Only cubic2 cap .5 advances: it retains more movement than the cheaper
quintic cap .4 in this attribution. Quintic remains a measured shadow alternative.
Eigh/spectral measurements are never executed by this policy. 18 cases: QR20,
NS6/average cap .1, cubic2/row cap .5 and cubic2/skew-Frobenius cap .5 at widths
768/1536 and covariance .99/.999; repeat row/fro at width1536, beta .999.
Each runs 420 updates through QR resets 200/400. Existing seeded MockBlock,
64 activation rows, both factors, high basis and strict M. Offline diagnostics
every refresh; timings exclude diagnostics, cold init separately reported.
Keep seed/model/data identical to the prior synthetic experiment.

Qualification requires all cases complete, max normalized Gram <.05, sampled
per-refresh AND whole-history world-M errors <.01, exact model/loss/probe repeats.
Report own-trajectory tracking separately from causal common-state attribution.
No uninstrumented throughput, learning or cross-hardware claim follows.
One claimed GPU3, no queue; 900s internal / 960s external. Under20MiB evidence;
no weights or tensors. Initial free space 326GiB. Retain compact evidence for audit.
'''
        m={}
    else:
        assert all(x['qualified'] for x in prior['rows']) and prior['exact_repeats']
        prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
                matrix_lr=.0005,branch_steps=5000,covariance_beta=.99)
        c=json.loads((b/'config.json').read_text());arms=[]
        for beta in (.99,.999):
            for kind in ['qr20','ns6','row_cubic2','fro_cubic2']:
                arm=dict(label=f'{kind}_beta{beta}',policy='qr20' if kind=='qr20' else 'warm20',
                    matrix_lr=.0005,matrix_beta2=.999,covariance_beta=beta,transport_precision='highest',
                    basis_ns_iterations=6 if kind=='ns6' else 2,basis_rotation_cap=.1,tracking_probe=True)
                if kind=='row_cubic2':arm['retraction_variant']='spectral_cubic2_cap05'
                if kind=='fro_cubic2':arm['movement_policy']='fro_cubic2_cap05'
                arms.append(arm)
        for old in [arms[0],arms[3]]:
            arms.append({**old,'label':old['label']+'_repeat','repeat_of':old['label']})
        c['memory_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=5000,
            evaluation_updates=[875,1750,3500],wall_seconds=90.,timing_chunk=25,
            maximum_gram_error=.05,timing_version=2,guard_every_refresh=True,
            tracking_probe_kind='strict_transport',movement_attribution=False,
            tracking_ages=[20,180,200,220,860,1740,3480],
            control_pairs=[[arms[0]['label'],arms[-2]['label']],[arms[3]['label'],arms[-1]['label']]],
            timing_repeat_maximum_training_wall_range_over_mean=.03,
            timing_repeat_maximum_optimizer_range_over_mean=.02,
            checkpoint_policy='No weights or tensor snapshots. Common initial state in RAM only.')
        c['max_wall_seconds']=1600
        source=['benchmarks/'+n for n in ['cifar_lr_screen.py','cifar_tracking_probe.py',
            'cifar_strict_transport.py','warm_retraction.py','cifar_retraction_screen.py',
            'movement_retraction.py','cifar_movement_probe.py','cifar_movement_screen.py',
            'prepare_movement_followup.py']]
        protocol='''# Skew-Frobenius cubic2: paired fixed-update and measured-wall learning

Predeclared after common-state and 18-case natural-cycle qualification.
Ten arms: QR20, NS6 average cap .1, cubic2 row cap .5, cubic2 skew-Frobenius cap .5,
each at covariance .99/.999. Repeat QR and Frobenius at .99. Seed139, both factors,
high basis/strict M, Adam(.9,.999), LR .0005, original200-epoch schedule, warm20
and QR200. Same cached CIFAR100 split/augmentation and initial model/RNG.
Only clipping changes between row cubic2 and Frobenius cubic2. No online spectra,
controller, compiler or fallback added. Preserve the old cubic1 failures.

Observe fixed875/1750/3500 and first25-step boundary reaching 90 measured training
seconds, excluding init, evaluation and offline guards/probes. Continue until both
3500 and 90s are observed, with maximum5000 updates. A missing budget endpoint is
reported as capped, not imputed. Record actual crossing time and overshoot. The
budget is operational training wall time including launch/dispatch, not an
equal cumulative-optimizer-time budget. Retain original step-indexed LR schedule;
no per-arm LR/horizon tuning. Primary quality anchor: fixed3500 CE. Secondary:
measured90s CE. No synthetic timing is combined with learning observations.

For a repeat-qualified wall-time interpretation require exact fixed-update
weights/metrics/data/probes and each .99 repeated control's fixed3500 training
wall range/mean <=3% and optimizer range/mean <=2%. Otherwise show measured wall
endpoints as exploratory; do not promote a time-to-quality winner. Small speed
differences within repeat variability remain unresolved. Single seed and short
horizon cannot establish convergence or a tuned-method ranking.

Every actual refresh retains Gram<.05/finite guard. Seven selected ages retain
selected-basis update<1e-6 and world-M<.01, separately from unchanged full strict
refresh difference<1% diagnostic (QR/reset disagreements remain explicit).
12? Historical matching is derived from available shared anchors: three QR/NS6/
row controls at each beta, three observations each, expected18 matches. Abort on
an actual guard failure and retain its diagnostic; do not relax old thresholds.

One claimed GPU3, no queue; internal1600s/external1660s. Under25MiB compact evidence,
326GiB initially free. No weights/checkpoints/tensor snapshots. Retain source,
protocol, config, metrics and logs for audit. Production/defaults unchanged.
'''.replace('12? Historical','Historical')
        m=json.loads((b/'manifest.json').read_text())
    for rel in source:
        target=b/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n');(b/'PROTOCOL.md').write_text(protocol)
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        prerequisite=str(a.prerequisite.resolve()),prerequisite_summary_sha256=sha(a.prerequisite/'summary.json'),
        evidence_budget_bytes=25*2**20)
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
